"""TestNet-only NautilusTrader runtime and managed BTC short-straddle strategy."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from nautilus_trader.live.config import LiveExecEngineConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.events import (
    OrderAccepted,
    OrderCanceled,
    OrderCancelRejected,
    OrderExpired,
    OrderFilled,
    OrderRejected,
)
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.strategy import StrategyConfig

from config import Settings
from config.config import create_deribit_data_config, create_deribit_exec_config
from strategies.risk_management import compute_risk_thresholds, evaluate_buyback_risk


logger = logging.getLogger("btc_options_straddle.deribit_nautilus")
ENTRY_TIMEOUT_SECONDS = 12.0
QUOTE_STALE_SECONDS = 5.0
MAX_CLOSE_RETRIES = 3
MAX_CANCEL_RETRIES = 3
STARTUP_RECONCILIATION_GRACE_SECONDS = 35.0
_TOLERANCE = 1e-10


def _instrument_id(value: str) -> InstrumentId:
    """Convert the public Deribit instrument name into Nautilus symbology."""
    value = value.strip()
    if "." not in value:
        value = f"{value}.DERIBIT"
    result = InstrumentId.from_str(value)
    if result.venue.value != "DERIBIT":
        raise ValueError("The automated strategy accepts Deribit instruments only.")
    return result


class StrategySnapshot:
    """Thread-safe, serializable strategy status shared with Streamlit."""

    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self.path = path
        self._value: dict[str, Any] = {
            "node_state": "stopped",
            "strategy_state": "stopped",
            "message": "The TestNet strategy is stopped.",
            "events": [],
            "positions": {"call": 0.0, "put": 0.0},
            "open_orders": 0,
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            if self.path is not None:
                try:
                    self._value.update(json.loads(self.path.read_text(encoding="utf-8")))
                except (FileNotFoundError, OSError, json.JSONDecodeError):
                    pass
            self._value.update(deepcopy(values))
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                temporary.write_text(json.dumps(self._value, indent=2), encoding="utf-8")
                temporary.replace(self.path)

    def read(self) -> dict[str, Any]:
        with self._lock:
            if self.path is not None:
                try:
                    value = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        self._value = value
                except (FileNotFoundError, OSError, json.JSONDecodeError):
                    pass
            return deepcopy(self._value)


def _empty_state(call_name: str, put_name: str) -> dict[str, Any]:
    return {
        "version": 1,
        "call_instrument": call_name,
        "put_instrument": put_name,
        "state": "reconciling",
        "message": "Reconciling the account before enabling entry.",
        "open_qty": {"call": 0.0, "put": 0.0},
        "entry_qty": {"call": 0.0, "put": 0.0},
        "entry_value": {"call": 0.0, "put": 0.0},
        "roles": {},
        "venue_roles": {},
        "entry_started_at": None,
        "entry_failed": False,
        "exit_reason": None,
        "close_retries": {"call": 0, "put": 0},
        "retry_after": {"call": 0.0, "put": 0.0},
        "cancel_retries": {},
        "cancel_retry_after": {},
        "events": [],
    }


class DeribitBtcStraddleStrategy(Strategy):
    """Nautilus strategy for TestNet entry, fill tracking, risk exits and recovery."""

    def __init__(
        self,
        *,
        call_name: str,
        put_name: str,
        amount_min: float,
        state_path: Path,
        commands: queue.Queue[dict[str, Any]],
        snapshot: StrategySnapshot,
    ) -> None:
        super().__init__(
            StrategyConfig(
                strategy_id="BTCSTRADDLE-TESTNET",
                order_id_tag="01",
                use_uuid_client_order_ids=True,
            ),
        )
        self.call_id = _instrument_id(call_name)
        self.put_id = _instrument_id(put_name)
        self.call_name = call_name
        self.put_name = put_name
        self.amount_min = float(amount_min)
        self.state_path = state_path
        self.commands = commands
        self.snapshot = snapshot
        self._ids = {"call": self.call_id, "put": self.put_id}
        self._quote: dict[str, dict[str, float]] = {}
        self._state = self._load_state()
        self._started_monotonic = 0.0
        self._startup_reconciled = False
        self._resume_account_event_ns: int | None = None
        self._entry_started_monotonic: float | None = None
        self._cancel_requested: set[str] = set()
        self._cancel_requested_at: dict[str, float] = {}

    def _load_state(self) -> dict[str, Any]:
        state = _empty_state(self.call_name, self.put_name)
        try:
            restored = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return state
        if (
            restored.get("version") != 1
            or restored.get("call_instrument") != self.call_name
            or restored.get("put_instrument") != self.put_name
        ):
            state["state"] = "manual_review"
            state["message"] = "Saved strategy state does not match this option pair; review the TestNet account."
            return state
        for key in state:
            if key in restored:
                state[key] = restored[key]
        state["has_saved_state"] = True
        # Monotonic timestamps do not survive an OS reboot. Retry delays are
        # short-lived, so restart their cooldown rather than restoring them.
        state["retry_after"] = {"call": 0.0, "put": 0.0}
        state["cancel_retry_after"] = {}
        return state

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _event(self, message: str) -> None:
        events = self._state.setdefault("events", [])
        events.append(
            {
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "message": message,
            },
        )
        del events[:-50]
        self._state["message"] = message
        self._save_state()
        self._publish()

    def _publish(self) -> None:
        call_qty = float(self._state["open_qty"].get("call", 0.0))
        put_qty = float(self._state["open_qty"].get("put", 0.0))
        call_entry = self._average_entry("call")
        put_entry = self._average_entry("put")
        entry_premium = call_entry + put_entry if call_entry and put_entry else None
        ask_call = self._quote.get("call", {}).get("ask")
        ask_put = self._quote.get("put", {}).get("ask")
        buyback = ask_call + ask_put if ask_call is not None and ask_put is not None else None
        thresholds = compute_risk_thresholds(entry_premium) if entry_premium and entry_premium > 0 else None
        try:
            open_order_count = len(self._open_orders())
        except Exception:
            open_order_count = None
        self.snapshot.update(
            strategy_state=self._state.get("state", "unknown"),
            message=self._state.get("message", ""),
            positions={"call": call_qty, "put": put_qty},
            open_orders=open_order_count,
            entry_premium=entry_premium,
            buyback_premium=buyback,
            stop_loss=thresholds.stop_loss if thresholds else None,
            take_profit=thresholds.take_profit if thresholds else None,
            exit_reason=self._state.get("exit_reason"),
            events=list(self._state.get("events", [])),
            call_instrument=self.call_name,
            put_instrument=self.put_name,
        )

    def _average_entry(self, leg: str) -> float:
        quantity = float(self._state["entry_qty"].get(leg, 0.0))
        value = float(self._state["entry_value"].get(leg, 0.0))
        return value / quantity if quantity > 0 else 0.0

    def on_start(self) -> None:
        self._started_monotonic = time.monotonic()
        if self._state.get("state") == "stopped":
            self._state["state"] = "reconciling"
        for leg, instrument_id in self._ids.items():
            if self.cache.instrument(instrument_id) is None:
                self._state["state"] = "error"
                self._event(f"NautilusTrader did not load the {leg} option instrument.")
                self.stop()
                return
            self.subscribe_quote_ticks(instrument_id)
        self.clock.set_timer(
            name="btc-straddle-control",
            interval=timedelta(seconds=1),
            callback=self.on_control_timer,
        )
        if self._state.get("state") not in {"manual_review", "error"}:
            self._state["state"] = "reconciling"
        self._event(
            "Connected to Deribit TestNet; quote streams are subscribed. Waiting for exchange order and position reconciliation."
        )

    def on_quote_tick(self, tick: QuoteTick) -> None:
        leg = "call" if tick.instrument_id == self.call_id else "put" if tick.instrument_id == self.put_id else None
        if leg is None:
            return
        self._quote[leg] = {
            "bid": tick.bid_price.as_double(),
            "ask": tick.ask_price.as_double(),
            "time": time.monotonic(),
        }
        self._publish()

    def on_control_timer(self, _event: Any) -> None:
        self._drain_commands()
        if (
            not self._startup_reconciled
            and time.monotonic() - self._started_monotonic >= STARTUP_RECONCILIATION_GRACE_SECONDS
        ):
            self._reconcile_account_positions(startup=True)
        if self._state.get("state") in {"manual_review", "error", "stopped"}:
            return
        self._reconcile_account_positions(startup=False)
        self._advance_entry_recovery()
        self._advance_risk_exit()
        self._publish()

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            if command.get("action") == "enter":
                self._submit_entry(command)

    def _submit_entry(self, command: dict[str, Any]) -> None:
        if self._state.get("state") != "ready":
            self._event("Entry blocked: the strategy is not in a reconciled ready state.")
            return
        if any(float(qty) > _TOLERANCE for qty in self._state["open_qty"].values()):
            self._state["state"] = "manual_review"
            self._event("Entry blocked: an option position is already open for this pair.")
            return
        if self._open_orders():
            self._state["state"] = "manual_review"
            self._event("Entry blocked: existing open orders must be reconciled first.")
            return
        amount = float(command["amount_btc"])
        if amount < self.amount_min or amount <= 0:
            self._event(f"Entry blocked: amount must be at least {self.amount_min:g} BTC per leg.")
            return
        try:
            # Each entry is a new straddle cycle. Do not mix its premium basis
            # with fills from a previously closed or recovered position.
            self._state["entry_qty"] = {"call": 0.0, "put": 0.0}
            self._state["entry_value"] = {"call": 0.0, "put": 0.0}
            self._state["close_retries"] = {"call": 0, "put": 0}
            self._state["retry_after"] = {"call": 0.0, "put": 0.0}
            orders = {}
            for leg, price_key in (("call", "call_price_btc"), ("put", "put_price_btc")):
                instrument = self.cache.instrument(self._ids[leg])
                quantity = instrument.make_qty(Decimal(str(amount)))
                if abs(quantity.as_double() - amount) > max(_TOLERANCE, amount * 1e-8):
                    raise ValueError(f"Amount is not valid for the {leg} instrument size precision.")
                price = instrument.make_price(Decimal(str(command[price_key])))
                if price.as_double() <= 0:
                    raise ValueError("Both limit prices must be positive.")
                orders[leg] = self.order_factory.limit(
                    instrument_id=self._ids[leg],
                    order_side=OrderSide.SELL,
                    quantity=quantity,
                    price=price,
                    time_in_force=TimeInForce.GTC,
                    post_only=True,
                    tags=[f"btc-straddle:v1:entry:{leg}"],
                )
            self._state["state"] = "entry_working"
            self._state["entry_started_at"] = time.time()
            self._entry_started_monotonic = time.monotonic()
            self._state["entry_failed"] = False
            self._state["exit_reason"] = None
            for leg, order in orders.items():
                self._state["roles"][order.client_order_id.value] = f"entry:{leg}"
            self._event("Submitting both post-only TestNet entry orders through NautilusTrader.")
            self.submit_order(orders["call"])
            self.submit_order(orders["put"])
        except Exception as exc:
            self._state["entry_failed"] = True
            self._state["state"] = "recovering"
            self._event(f"Entry could not be submitted: {exc}")
            self._cancel_live_entry_orders()

    def _order_role(self, client_order_id: Any) -> str | None:
        key = getattr(client_order_id, "value", str(client_order_id))
        role = self._state["roles"].get(key)
        if role:
            return role
        order = self.cache.order(client_order_id)
        if order is not None:
            venue_order_id = getattr(order, "venue_order_id", None)
            venue_key = getattr(venue_order_id, "value", None)
            if venue_key:
                role = self._state["venue_roles"].get(venue_key)
                if role:
                    self._state["roles"][key] = role
                    return role
            for tag in order.tags:
                prefix = "btc-straddle:v1:"
                if str(tag).startswith(prefix):
                    role = str(tag)[len(prefix):]
                    self._state["roles"][key] = role
                    return role
        return None

    def on_order_accepted(self, event: OrderAccepted) -> None:
        role = self._order_role(event.client_order_id)
        if role:
            venue_order_id = getattr(event.venue_order_id, "value", None)
            if venue_order_id:
                self._state["venue_roles"][venue_order_id] = role
                self._save_state()
            self._event(f"Order accepted: {role} ({event.venue_order_id}).")

    def on_order_filled(self, event: OrderFilled) -> None:
        manual_review_latched = self._state.get("state") in {"manual_review", "error"}
        role = self._order_role(event.client_order_id)
        if role is None:
            self._state["state"] = "manual_review"
            self._event("An unrecognized TestNet fill arrived; automated actions are paused for review.")
            return
        leg = role.split(":", 1)[1]
        quantity = event.last_qty.as_double()
        price = event.last_px.as_double()
        if role.startswith("entry:"):
            self._state["entry_qty"][leg] += quantity
            self._state["entry_value"][leg] += quantity * price
            if self._entry_started_monotonic is None:
                self._entry_started_monotonic = time.monotonic()
            self._state["open_qty"][leg] += quantity
            if not manual_review_latched:
                self._state["state"] = "entry_working"
            self._event(f"{leg.title()} entry fill: {quantity:g} BTC at {price:g} BTC.")
        else:
            current = float(self._state["open_qty"].get(leg, 0.0))
            new_quantity = max(0.0, current - quantity)
            average_entry = self._average_entry(leg)
            self._state["open_qty"][leg] = new_quantity
            self._state["entry_qty"][leg] = new_quantity
            self._state["entry_value"][leg] = average_entry * new_quantity
            self._event(f"{leg.title()} close fill: {quantity:g} BTC at {price:g} BTC.")
        self._reconcile_account_positions(startup=False)
        self._finish_if_flat()

    def on_order_rejected(self, event: OrderRejected) -> None:
        role = self._order_role(event.client_order_id)
        if role and self._state.get("state") in {"manual_review", "error"}:
            self._event(f"{role} rejection received while manual review is latched; no automated action was resumed.")
            return
        if role and role.startswith("entry:"):
            self._state["entry_failed"] = True
            self._state["state"] = "recovering"
            self._event(f"{role} was rejected; cancelling any remaining entry and reducing unmatched exposure.")
            self._cancel_live_entry_orders()
        elif role and role.startswith(("close:", "recover:")):
            leg = role.split(":", 1)[1]
            retries = int(self._state["close_retries"].get(leg, 0)) + 1
            self._state["close_retries"][leg] = retries
            self._state["retry_after"][leg] = time.monotonic() + 2.0
            if retries >= MAX_CLOSE_RETRIES:
                self._state["state"] = "manual_review"
                self._event(f"{leg.title()} close was rejected repeatedly; check the TestNet position manually.")
            else:
                self._state["state"] = "exit_working"
                self._event(f"{leg.title()} close was rejected; retry {retries}/{MAX_CLOSE_RETRIES} is scheduled.")

    def on_order_canceled(self, event: OrderCanceled) -> None:
        role = self._order_role(event.client_order_id)
        order_id = event.client_order_id.value
        self._cancel_requested.discard(order_id)
        self._cancel_requested_at.pop(order_id, None)
        if role and role.startswith("entry:"):
            self._event(f"{role} remainder canceled; reconciling both legs.")
            self._advance_entry_recovery(force=True)
        elif role and role.startswith(("close:", "recover:")):
            self._state["state"] = "manual_review"
            self._event(f"{role} was canceled; automated close paused for manual review.")

    def on_order_expired(self, event: OrderExpired) -> None:
        self.on_order_canceled(event)  # An expired entry is terminal like a cancellation.

    def on_order_cancel_rejected(self, event: OrderCancelRejected) -> None:
        role = self._order_role(event.client_order_id)
        if role and self._state.get("state") in {"manual_review", "error"}:
            self._event(f"{role} cancel rejection received while manual review is latched; inspect the live order.")
            return
        if role and role.startswith("entry:"):
            order_id = event.client_order_id.value
            self._cancel_requested.discard(order_id)
            self._cancel_requested_at.pop(order_id, None)
            attempts = int(self._state["cancel_retries"].get(order_id, 0))
            if attempts >= MAX_CANCEL_RETRIES:
                self._state["state"] = "manual_review"
                self._event(f"{role} could not be canceled after {attempts} attempts; inspect the live order and position.")
            else:
                self._state["cancel_retry_after"][order_id] = time.monotonic() + 2.0
                self._state["state"] = "recovering"
                self._event(f"{role} cancel was rejected; retry {attempts + 1}/{MAX_CANCEL_RETRIES} is scheduled.")
        elif role and role.startswith(("close:", "recover:")):
            self._event(f"{role} cancel was rejected; the active reduce-only order remains under exchange monitoring.")

    def on_position_opened(self, _event: Any) -> None:
        self._reconcile_account_positions(startup=False)
        self._publish()

    def on_position_changed(self, _event: Any) -> None:
        self._reconcile_account_positions(startup=False)
        self._publish()

    def on_position_closed(self, _event: Any) -> None:
        self._reconcile_account_positions(startup=False)
        self._finish_if_flat()

    def on_degrade(self) -> None:
        self._state["message"] = "NautilusTrader is degraded; entries are paused while the connection recovers."
        self._state["state"] = "degraded"
        self._event(self._state["message"])

    def on_resume(self) -> None:
        if self._state.get("state") in {"manual_review", "error"}:
            return
        self._state["state"] = "reconciling"
        self._startup_reconciled = False
        self._started_monotonic = time.monotonic()
        try:
            account = self.portfolio.account(venue=self.call_id.venue)
            self._resume_account_event_ns = account.last_event.ts_event if account is not None else 0
        except Exception:
            self._resume_account_event_ns = 0
        self._event("NautilusTrader resumed; reconciling open orders and positions.")

    def _reconcile_account_positions(self, *, startup: bool) -> None:
        if startup and self._state.get("state") in {"manual_review", "error"}:
            self._startup_reconciled = True
            return
        if startup:
            try:
                account = self.portfolio.account(venue=self.call_id.venue)
                account_event_ns = account.last_event.ts_event if account is not None else 0
            except Exception:
                account = None
                account_event_ns = 0
            if account is None or (
                self._resume_account_event_ns is not None
                and account_event_ns <= self._resume_account_event_ns
            ):
                self._state["state"] = "reconciling"
                self._state["message"] = "Waiting for the authenticated Deribit account state before enabling orders."
                self._publish()
                return
            self._resume_account_event_ns = None

        positions: dict[str, list[Any]] = {"call": [], "put": []}
        try:
            for leg, instrument_id in self._ids.items():
                positions[leg] = self.cache.positions_open(instrument_id=instrument_id)
        except Exception as exc:
            if startup:
                self._state["state"] = "manual_review"
                self._state["message"] = f"Could not reconcile TestNet positions: {exc}"
                self._publish()
            return

        has_position = any(positions.values())
        saved_active = bool(self._state.get("has_saved_state")) and self._state.get("state") not in {"ready", "closed"}
        if startup and has_position and not saved_active:
            self._state["state"] = "manual_review"
            self._event("An existing TestNet position has no matching saved strategy cycle; no orders will be sent.")
            self._startup_reconciled = True
            return

        for leg, leg_positions in positions.items():
            signed = sum(float(position.signed_qty) for position in leg_positions)
            if signed > _TOLERANCE:
                self._state["state"] = "manual_review"
                self._state["message"] = f"Unexpected long exposure on the {leg}; automated actions are paused."
                self._publish()
                self._startup_reconciled = True
                return
            quantity = max(0.0, -signed)
            unowned = [position for position in leg_positions if position.strategy_id != self.id]
            if unowned:
                expected_qty = float(self._state["open_qty"].get(leg, 0.0))
                expected_entry = self._average_entry(leg)
                weighted_open = sum(
                    abs(float(position.signed_qty)) * float(position.avg_px_open)
                    for position in leg_positions
                    if float(position.signed_qty) < 0
                )
                actual_entry = weighted_open / quantity if quantity > 0 else 0.0
                qty_matches = abs(quantity - expected_qty) <= max(_TOLERANCE, expected_qty * 1e-8)
                price_matches = (
                    expected_qty <= _TOLERANCE
                    or expected_entry > 0
                    and abs(actual_entry - expected_entry) <= max(1e-8, expected_entry * 1e-6)
                )
                if not saved_active or not qty_matches or not price_matches:
                    self._state["state"] = "manual_review"
                    self._event(
                        f"The {leg} TestNet position cannot be matched to the saved strategy fills; no automatic order will be sent."
                    )
                    self._startup_reconciled = True
                    return
            if startup or quantity > _TOLERANCE or self._state["open_qty"].get(leg, 0.0) > _TOLERANCE:
                self._state["open_qty"][leg] = quantity
            if quantity > _TOLERANCE and startup and not self._state["entry_qty"].get(leg, 0.0):
                weighted = sum(
                    abs(float(position.signed_qty)) * float(position.avg_px_open)
                    for position in leg_positions
                    if float(position.signed_qty) < 0
                )
                self._state["entry_qty"][leg] = quantity
                self._state["entry_value"][leg] = weighted

        if startup:
            unknown = []
            active_roles: set[str] = set()
            for leg, instrument_id in self._ids.items():
                for order in self.cache.orders_open(instrument_id=instrument_id):
                    role = self._order_role(order.client_order_id)
                    if role is None:
                        unknown.append(order.client_order_id.value)
                    else:
                        active_roles.add(role.split(":", 1)[0])
            if unknown:
                self._state["state"] = "manual_review"
                self._event("Unrecognized open TestNet orders were found; cancel or reconcile them before continuing.")
                self._startup_reconciled = True
                return
            self._startup_reconciled = True
            state = self._state.get("state")
            if "close" in active_roles:
                self._state["state"] = "exit_working"
            elif "entry" in active_roles or "recover" in active_roles:
                self._state["state"] = "recovering"
                if "entry" in active_roles:
                    self._state["entry_failed"] = True
            elif state in {"reconciling", "ready", "closed", "stopped"}:
                if has_position:
                    self._state["state"] = "open" if min(self._state["open_qty"].values()) > _TOLERANCE else "recovering"
                else:
                    self._state["state"] = "ready"
        self._save_state()
        self._publish()

    def _open_orders(self, prefix: str | None = None) -> list[Any]:
        result = []
        for leg, instrument_id in self._ids.items():
            for order in self.cache.orders_open(instrument_id=instrument_id):
                role = self._order_role(order.client_order_id)
                if role is None:
                    continue
                if prefix is None or role.startswith(prefix):
                    result.append(order)
        return result

    def _cancel_live_entry_orders(self) -> None:
        now = time.monotonic()
        for order in self._open_orders("entry:"):
            order_id = order.client_order_id.value
            if order_id in self._cancel_requested:
                if now - self._cancel_requested_at.get(order_id, now) < 5.0:
                    continue
                self._cancel_requested.discard(order_id)
                self._cancel_requested_at.pop(order_id, None)
            if now < float(self._state["cancel_retry_after"].get(order_id, 0.0)):
                continue
            attempts = int(self._state["cancel_retries"].get(order_id, 0))
            if attempts >= MAX_CANCEL_RETRIES:
                self._state["state"] = "manual_review"
                self._event(f"{self._order_role(order.client_order_id)} could not be canceled after {attempts} attempts; inspect the live order and position.")
                return
            self._cancel_requested.add(order_id)
            self._cancel_requested_at[order_id] = now
            self._state["cancel_retries"][order_id] = attempts + 1
            try:
                self.cancel_order(order)
                self._save_state()
            except Exception as exc:
                self._cancel_requested.discard(order_id)
                self._cancel_requested_at.pop(order_id, None)
                self._state["cancel_retry_after"][order_id] = time.monotonic() + 2.0
                if attempts + 1 >= MAX_CANCEL_RETRIES:
                    self._state["state"] = "manual_review"
                    self._event(f"Entry cancel failed repeatedly: {exc}; inspect the live order and position.")
                else:
                    self._event(f"Entry cancel request failed; retry {attempts + 2}/{MAX_CANCEL_RETRIES} is scheduled: {exc}")

    def _advance_entry_recovery(self, *, force: bool = False) -> None:
        state = self._state.get("state")
        if state not in {"entry_working", "recovering", "reconciling"}:
            return
        started = self._entry_started_monotonic
        if started is None and self._state.get("entry_started_at"):
            started = time.monotonic() - max(0.0, time.time() - float(self._state["entry_started_at"]))
            self._entry_started_monotonic = started
        timed_out = started is not None and time.monotonic() - started >= ENTRY_TIMEOUT_SECONDS
        if self._state.get("entry_failed") or timed_out or force:
            self._state["state"] = "recovering"
            self._cancel_live_entry_orders()
        if self._open_orders("entry:"):
            return
        if self._open_orders("recover:"):
            return
        call_qty = float(self._state["open_qty"].get("call", 0.0))
        put_qty = float(self._state["open_qty"].get("put", 0.0))
        if abs(call_qty - put_qty) > max(_TOLERANCE, max(call_qty, put_qty) * 1e-8):
            leg = "call" if call_qty > put_qty else "put"
            excess = abs(call_qty - put_qty)
            self._state["state"] = "recovering"
            self._submit_close(leg, excess, role_prefix="recover")
            return
        if call_qty > _TOLERANCE and put_qty > _TOLERANCE:
            self._state["state"] = "open"
            self._state["entry_started_at"] = None
            self._entry_started_monotonic = None
            self._save_state()
            self._publish()
        elif call_qty <= _TOLERANCE and put_qty <= _TOLERANCE:
            self._state["state"] = "ready"
            self._state["entry_started_at"] = None
            self._entry_started_monotonic = None
            self._save_state()
            self._publish()

    def _fresh_quotes(self) -> bool:
        now = time.monotonic()
        return all(
            leg in self._quote
            and self._quote[leg].get("ask", 0.0) > 0
            and now - self._quote[leg].get("time", 0.0) <= QUOTE_STALE_SECONDS
            for leg in ("call", "put")
        )

    def _advance_risk_exit(self) -> None:
        if self._state.get("state") == "exit_working":
            self._retry_failed_closes()
            self._finish_if_flat()
            return
        if self._state.get("state") != "open" or not self._fresh_quotes():
            return
        call_qty = float(self._state["open_qty"].get("call", 0.0))
        put_qty = float(self._state["open_qty"].get("put", 0.0))
        if call_qty <= _TOLERANCE or put_qty <= _TOLERANCE:
            self._state["state"] = "recovering"
            self._advance_entry_recovery(force=True)
            return
        call_entry = self._average_entry("call")
        put_entry = self._average_entry("put")
        if call_entry <= 0 or put_entry <= 0:
            self._state["state"] = "manual_review"
            self._event("Entry basis is unavailable; automated risk exits are paused for review.")
            return
        buyback = self._quote["call"]["ask"] + self._quote["put"]["ask"]
        thresholds = compute_risk_thresholds(call_entry + put_entry)
        evaluation = evaluate_buyback_risk(buyback, thresholds)
        if evaluation.triggered:
            self._state["exit_reason"] = evaluation.reason
            self._state["state"] = "exit_working"
            self._state["close_retries"] = {"call": 0, "put": 0}
            self._event(
                f"{evaluation.reason.replace('_', ' ').title()} triggered at combined ask {buyback:.8f}; submitting reduce-only closes."
            )
            self._submit_close("call", call_qty, role_prefix="close")
            self._submit_close("put", put_qty, role_prefix="close")

    def _submit_close(self, leg: str, quantity_value: float, *, role_prefix: str) -> None:
        quantity_value = min(quantity_value, float(self._state["open_qty"].get(leg, 0.0)))
        if quantity_value <= _TOLERANCE:
            return
        role = f"{role_prefix}:{leg}"
        if any(
            self._order_role(order.client_order_id) in {f"close:{leg}", f"recover:{leg}"}
            for order in self._open_orders()
        ):
            return
        now = time.monotonic()
        if now < float(self._state["retry_after"].get(leg, 0.0)):
            return
        instrument = self.cache.instrument(self._ids[leg])
        quantity = instrument.make_qty(Decimal(str(quantity_value)))
        order = self.order_factory.market(
            instrument_id=self._ids[leg],
            order_side=OrderSide.BUY,
            quantity=quantity,
            reduce_only=True,
            tags=[f"btc-straddle:v1:{role}"],
        )
        self._state["roles"][order.client_order_id.value] = role
        self._state["state"] = "exit_working" if role_prefix == "close" else "recovering"
        self._event(f"Submitting {role} reduce-only market buy for {quantity.as_double():g} BTC.")
        self.submit_order(order)

    def _retry_failed_closes(self) -> None:
        if self._open_orders("close:") or self._open_orders("recover:"):
            return
        if self._state.get("state") == "manual_review":
            return
        reason = self._state.get("exit_reason")
        if reason:
            for leg in ("call", "put"):
                if float(self._state["open_qty"].get(leg, 0.0)) > _TOLERANCE:
                    self._submit_close(leg, float(self._state["open_qty"][leg]), role_prefix="close")
            return
        call_qty = float(self._state["open_qty"].get("call", 0.0))
        put_qty = float(self._state["open_qty"].get("put", 0.0))
        if abs(call_qty - put_qty) > max(_TOLERANCE, max(call_qty, put_qty) * 1e-8):
            leg = "call" if call_qty > put_qty else "put"
            self._submit_close(leg, abs(call_qty - put_qty), role_prefix="recover")
        elif call_qty > _TOLERANCE and put_qty > _TOLERANCE:
            self._state["state"] = "open"
            self._event("Unmatched entry exposure was reduced; the matched straddle is under risk monitoring.")
        else:
            self._state["state"] = "ready"
            self._event("Entry exposure was fully reduced; the strategy is ready for a new cycle.")

    def _finish_if_flat(self) -> None:
        if self._state.get("state") in {"manual_review", "error"}:
            return
        if all(float(qty) <= _TOLERANCE for qty in self._state["open_qty"].values()):
            if self._state.get("exit_reason"):
                self._state["state"] = "closed"
                reason = self._state["exit_reason"]
                self._state["exit_reason"] = None
                self._event(f"Strategy position closed after {reason.replace('_', ' ')}.")

    def on_stop(self) -> None:
        if self._state.get("state") in {"manual_review", "error"}:
            self._event("NautilusTrader stopped; the manual-review state remains latched.")
            return
        self._state["state"] = "stopped"
        self._event("NautilusTrader strategy stopped.")


class DeribitNautilusService:
    """Background, TestNet-only Nautilus node with a thread-safe command/status API."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        call_name: str,
        put_name: str,
        amount_min: float,
        state_path: Path,
        order_submission_enabled: bool,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self.call_name = call_name
        self.put_name = put_name
        self.amount_min = amount_min
        self.state_path = state_path
        self.order_submission_enabled = order_submission_enabled
        self.runtime_dir = state_path.parent
        self.command_path = self.runtime_dir / "commands.jsonl"
        self.snapshot_path = self.runtime_dir / "status.json"
        self.stop_path = self.runtime_dir / "stop.request"
        self.log_path = self.runtime_dir / "nautilus.log"
        account_key = hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:16]
        self.worker_lock_path = self.runtime_dir / f"{account_key}.worker.lock"
        self.snapshot = StrategySnapshot(self.snapshot_path)
        self._process: subprocess.Popen[Any] | None = None
        self._log_handle: Any | None = None

    def read(self) -> dict[str, Any]:
        current = self.snapshot.read()
        if self._process is not None:
            code = self._process.poll()
            if code is not None and current.get("node_state") not in {"stopped", "error"}:
                self.snapshot.update(
                    node_state="error",
                    strategy_state="error",
                    message=f"NautilusTrader worker exited unexpectedly (code {code}); inspect the local runtime log.",
                )
            if code is not None and self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
        return self.snapshot.read()

    def start(self) -> None:
        if not self.order_submission_enabled:
            raise RuntimeError("Enable DERIBIT_TESTNET_ORDER_SUBMISSION_ENABLED before starting the TestNet strategy.")
        if self._process is not None and self._process.poll() is None:
            return
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_worker_lock()
        try:
            self.stop_path.unlink()
        except FileNotFoundError:
            pass
        self.command_path.touch(exist_ok=True)
        self.snapshot.update(
            node_state="starting",
            strategy_state="starting",
            message="Starting the Deribit TestNet NautilusTrader worker.",
        )
        child_env = os.environ.copy()
        child_env["DERIBIT_TESTNET_CLIENT_ID"] = self._client_id
        child_env["DERIBIT_TESTNET_CLIENT_SECRET"] = self._client_secret
        child_env["DERIBIT_TESTNET"] = "true"
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        worker_args = [
            sys.executable,
            "-m",
            "execution.deribit_nautilus_worker",
            str(self.call_name),
            str(self.put_name),
            str(self.amount_min),
            str(self.state_path),
            str(self.command_path),
            str(self.snapshot_path),
            str(self.stop_path),
            str(self.worker_lock_path),
        ]
        try:
            self._process = subprocess.Popen(
                worker_args,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self._log_handle.close()
            self._log_handle = None
            self._release_worker_lock()
            self.snapshot.update(node_state="error", strategy_state="error", message="Could not launch the NautilusTrader worker process.")
            raise

    def _acquire_worker_lock(self) -> None:
        payload = {
            "owner_pid": os.getpid(),
            "worker_pid": None,
            "created_at": time.time(),
        }
        try:
            descriptor = os.open(str(self.worker_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing = json.loads(self.worker_lock_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            pid = existing.get("worker_pid")
            created_at = float(existing.get("created_at", 0.0) or 0.0)
            alive = False
            if pid:
                try:
                    os.kill(int(pid), 0)
                    alive = True
                except PermissionError:
                    alive = True
                except (OSError, ValueError):
                    alive = False
            try:
                lock_age = time.time() - self.worker_lock_path.stat().st_mtime
            except OSError:
                lock_age = 0.0
            starting = not existing.get("worker_pid") and min(time.time() - created_at, lock_age) < 60.0
            if alive or starting:
                raise RuntimeError(
                    "A NautilusTrader TestNet worker is already active or starting for this API account. "
                    "Use its existing Streamlit session to manage it."
                )
            self.worker_lock_path.unlink(missing_ok=True)
            return self._acquire_worker_lock()
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)

    def _release_worker_lock(self) -> None:
        try:
            existing = json.loads(self.worker_lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if existing.get("owner_pid") == os.getpid() or existing.get("worker_pid") == os.getpid():
            self.worker_lock_path.unlink(missing_ok=True)

    def submit_entry(self, amount_btc: float, call_price_btc: float, put_price_btc: float) -> None:
        current = self.read()
        if current.get("node_state") != "running" or current.get("strategy_state") != "ready":
            raise RuntimeError("Wait for the TestNet strategy to reconcile and become ready before submitting an entry.")
        command = {
            "action": "enter",
            "amount_btc": float(amount_btc),
            "call_price_btc": float(call_price_btc),
            "put_price_btc": float(put_price_btc),
        }
        with self.command_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(command, separators=(",", ":")) + "\n")
            stream.flush()

    def stop(self) -> None:
        current = self.read()
        if any(float(value) > _TOLERANCE for value in current.get("positions", {}).values()):
            raise RuntimeError("The monitor cannot be stopped while the TestNet strategy has open exposure.")
        if current.get("open_orders") != 0:
            raise RuntimeError("The monitor cannot be stopped until all selected-pair TestNet orders are confirmed closed.")
        if current.get("strategy_state") not in {"ready", "closed", "stopped", "manual_review", "error"}:
            raise RuntimeError("Wait for the current TestNet order or recovery action to finish before stopping the monitor.")
        if self._process is None or self._process.poll() is not None:
            return
        self.stop_path.write_text("stop\n", encoding="utf-8")


def run_nautilus_worker(
    *,
    call_name: str,
    put_name: str,
    amount_min: float,
    state_path: Path,
    command_path: Path,
    snapshot_path: Path,
    stop_path: Path,
    worker_lock_path: Path,
) -> None:
    """Run one node on its own process main thread, keeping Streamlit responsive."""
    from nautilus_trader.model.identifiers import TraderId

    from config.config import build_trading_node_config

    node: TradingNode | None = None
    strategy: DeribitBtcStraddleStrategy | None = None
    commands: queue.Queue[dict[str, Any]] = queue.Queue()
    snapshot = StrategySnapshot(snapshot_path)
    stop_event = threading.Event()
    watcher: threading.Thread | None = None
    client_secret = os.getenv("DERIBIT_TESTNET_CLIENT_SECRET", "")
    try:
        worker_lock_path.write_text(
            json.dumps({"owner_pid": os.getppid(), "worker_pid": os.getpid(), "created_at": time.time()}),
            encoding="utf-8",
        )
    except OSError:
        pass
    try:
        settings = Settings.from_env()
        if not settings.deribit_testnet:
            raise RuntimeError("The automated strategy is restricted to Deribit TestNet.")
        client_id = os.getenv("DERIBIT_TESTNET_CLIENT_ID", "")
        if not client_id or not client_secret:
            raise RuntimeError("Deribit TestNet credentials were unavailable in the worker environment.")
        call_id = _instrument_id(call_name)
        put_id = _instrument_id(put_name)
        instrument_ids = [call_id, put_id]
        data_config = create_deribit_data_config(
            client_id=client_id,
            client_secret=client_secret,
            instrument_ids=instrument_ids,
        )
        exec_config = create_deribit_exec_config(
            client_id=client_id,
            client_secret=client_secret,
            instrument_ids=instrument_ids,
        )
        exec_engine = LiveExecEngineConfig(
            reconciliation=True,
            reconciliation_instrument_ids=instrument_ids,
            open_check_interval_secs=10.0,
            position_check_interval_secs=30.0,
            reconciliation_startup_delay_secs=3.0,
        )
        node_config = build_trading_node_config(
            data_config,
            exec_config,
            trader_id=TraderId("BTCSTRADDLE-TESTNET"),
            exec_engine=exec_engine,
        )
        strategy = DeribitBtcStraddleStrategy(
            call_name=call_name,
            put_name=put_name,
            amount_min=amount_min,
            state_path=state_path,
            commands=commands,
            snapshot=snapshot,
        )
        node = TradingNode(config=node_config)
        node.trader.add_strategy(strategy)
        node.build()

        command_offset = command_path.stat().st_size if command_path.exists() else 0

        def watch_control_files() -> None:
            nonlocal command_offset
            while not stop_event.is_set():
                if stop_path.exists():
                    stop_event.set()
                    loop = node.kernel.loop
                    if loop.is_running():
                        loop.call_soon_threadsafe(node.stop)
                    return
                try:
                    with command_path.open("r", encoding="utf-8") as stream:
                        stream.seek(command_offset)
                        while True:
                            line = stream.readline()
                            if not line:
                                break
                            command_offset = stream.tell()
                            try:
                                command = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(command, dict):
                                commands.put(command)
                except FileNotFoundError:
                    pass
                stop_event.wait(0.2)

        watcher = threading.Thread(target=watch_control_files, name="deribit-nautilus-control", daemon=True)
        watcher.start()
        snapshot.update(node_state="running", message="NautilusTrader node built; connecting to Deribit TestNet.")
        node.run(raise_exception=True)
        strategy_state = strategy._state.get("state", "stopped")
        if strategy_state in {"manual_review", "error"}:
            message = strategy._state.get("message", "The strategy remains paused for manual review.")
        else:
            strategy_state = "stopped"
            message = "NautilusTrader TestNet node stopped."
        snapshot.update(node_state="stopped", strategy_state=strategy_state, message=message)
    except Exception as exc:
        safe_error = str(exc).replace(client_secret, "[redacted]") if client_secret else str(exc)
        logger.error("Deribit TestNet NautilusTrader worker failed: %s", safe_error)
        if strategy is not None:
            strategy._state["state"] = "manual_review"
            strategy._state["message"] = f"NautilusTrader worker failed: {safe_error}"
            strategy._save_state()
        snapshot.update(
            node_state="error",
            strategy_state="manual_review" if strategy is not None else "error",
            message=f"NautilusTrader startup failed: {safe_error}",
        )
    finally:
        stop_event.set()
        if watcher is not None:
            watcher.join(timeout=1.0)
        if node is not None:
            try:
                node.dispose()
            except Exception:
                logger.exception("Failed to dispose Deribit TestNet NautilusTrader node")
        try:
            lock_value = json.loads(worker_lock_path.read_text(encoding="utf-8"))
            if lock_value.get("worker_pid") == os.getpid():
                worker_lock_path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            pass
