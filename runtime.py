"""End-to-end BTC short-straddle runtime orchestration."""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import (
    Settings,
    build_deribit_client_configs,
    get_deribit_connection_status,
)
from config.config import create_deribit_data_config, create_deribit_exec_config, get_deribit_credentials
from nautilus_trader.model.identifiers import InstrumentId
from data.market_data import OptionQuote, filter_btc_option_instruments
from execution.deribit_nautilus import DeribitNautilusService
from execution.order_manager import (
    LegState,
    OrderEvent,
    OrderStatus,
    OrderRequest,
    StrategyState,
    StrategyStateMachine,
)
from strategies.atm_selection import OptionContract, select_atm_pair
from strategies.risk_management import compute_risk_thresholds
from strategies.short_straddle import StraddleLeg, create_short_straddle

logger = logging.getLogger("btc_options_straddle.runtime")


@dataclass
class RuntimePlan:
    """Prepared runtime plan for a single short-straddle cycle."""

    settings: Settings
    live_trading: bool
    connected: bool
    data_client_config: Any | None = None
    exec_client_config: Any | None = None
    reference_price: float | None = None
    call_contract: OptionContract | None = None
    put_contract: OptionContract | None = None
    short_straddle: Any | None = None
    orders: list[OrderRequest] = field(default_factory=list)
    fill_events: list[OrderEvent] = field(default_factory=list)
    machine_state: StrategyState | None = None
    entry_premium: float = 0.0
    thresholds: Any | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveNodeStartup:
    """Start the isolated NautilusTrader node for one selected BTC straddle pair."""

    settings: Settings
    data_client_config: Any | None = None
    exec_client_config: Any | None = None
    call_contract: OptionContract | None = None
    put_contract: OptionContract | None = None
    amount_min: float | None = None
    state_path: Path | None = None
    order_submission_enabled: bool = False
    service: DeribitNautilusService | None = None
    instrument_ids: list[str] = field(default_factory=list)

    def start(self, *, client_id: str | None = None, client_secret: str | None = None) -> DeribitNautilusService:
        """Start the TestNet data and execution clients with the selected-pair strategy."""
        if not self.settings.deribit_testnet:
            raise ValueError("The startup sequence is restricted to Deribit TestNet.")
        if not self.order_submission_enabled:
            raise RuntimeError("Enable TestNet order submission before starting the automated strategy.")
        if self.call_contract is None or self.put_contract is None:
            raise ValueError("A BTC Call and Put pair are required before the live strategy can start.")
        if self.call_contract.option_type.upper() != "CALL" or self.put_contract.option_type.upper() != "PUT":
            raise ValueError("The selected pair must contain one Call and one Put.")
        if self.call_contract.strike != self.put_contract.strike or self.call_contract.expiry != self.put_contract.expiry:
            raise ValueError("The selected Call and Put must share the same strike and expiry.")
        if client_id is None or client_secret is None:
            client_id, client_secret = get_deribit_credentials()
        if self.amount_min is None:
            self.amount_min = max(
                self.call_contract.min_trade_amount or 0.001,
                self.put_contract.min_trade_amount or 0.001,
            )
        instrument_ids = [
            InstrumentId.from_str(f"{name}.DERIBIT" if "." not in name else name)
            for name in (self.call_contract.instrument_id, self.put_contract.instrument_id)
        ]
        self.data_client_config = create_deribit_data_config(
            client_id=client_id,
            client_secret=client_secret,
            instrument_ids=instrument_ids,
        )
        self.exec_client_config = create_deribit_exec_config(
            client_id=client_id,
            client_secret=client_secret,
            instrument_ids=instrument_ids,
        )
        if self.state_path is None:
            account_key = hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:16]
            pair_key = hashlib.sha256(
                f"{self.call_contract.instrument_id}|{self.put_contract.instrument_id}".encode("utf-8")
            ).hexdigest()[:16]
            self.state_path = Path(__file__).resolve().parent / ".runtime" / "deribit" / f"{account_key}_{pair_key}.json"
        self.instrument_ids = [self.call_contract.instrument_id, self.put_contract.instrument_id]
        self.service = DeribitNautilusService(
            client_id=client_id,
            client_secret=client_secret,
            call_name=self.call_contract.instrument_id,
            put_name=self.put_contract.instrument_id,
            amount_min=self.amount_min,
            state_path=self.state_path,
            order_submission_enabled=self.order_submission_enabled,
        )
        self.service.start()
        return self.service

    def register_reconnect_handlers(self) -> dict[str, Any]:
        """Report the enabled NautilusTrader exchange reconciliation controls."""
        return {
            "startup_reconciliation": True,
            "open_order_check_seconds": 10,
            "position_check_seconds": 30,
            "strategy_state_persisted": True,
        }

    def register_market_data_subscription(self, instrument_ids: list[str]) -> list[str]:
        """Validate IDs against the selected pair subscribed by the strategy."""
        self.instrument_ids = list(dict.fromkeys(str(item) for item in instrument_ids if item))
        expected = {
            contract.instrument_id
            for contract in (self.call_contract, self.put_contract)
            if contract is not None
        }
        normalized = {item.removesuffix(".DERIBIT") for item in self.instrument_ids}
        if expected and not expected.issubset(normalized):
            raise ValueError("Subscriptions must include the selected Call and Put.")
        logger.info("NautilusTrader option subscriptions configured: %s", self.instrument_ids)
        return self.instrument_ids


class StrategyRuntime:
    """Coordinates the BTC ATM short-straddle flow across config, data, strategy, and execution modules."""

    def __init__(self, *, settings: Settings | None = None, live_trading: bool | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.live_trading = bool(live_trading if live_trading is not None else self.settings.live_trading_enabled)
        self.connection_status = get_deribit_connection_status()
        self.data_client_config = None
        self.exec_client_config = None
        self.connected = False
        self.plan: RuntimePlan | None = None
        self.live_node: LiveNodeStartup | None = None
        self._wire_configs()

    def _wire_configs(self) -> bool:
        """Attach Deribit TestNet configs when credentials are present."""
        if not self.settings.deribit_testnet:
            logger.warning("Deribit TestNet disabled in settings; runtime will remain in dry-run mode.")
            return False

        try:
            self.data_client_config, self.exec_client_config = build_deribit_client_configs()
        except ValueError:
            logger.warning("Deribit credentials missing; runtime will stay in dry-run mode.")
            self.data_client_config = None
            self.exec_client_config = None
            self.connected = False
            return False

        # Constructing client configs does not establish an exchange connection.
        self.connected = False
        self.live_node = LiveNodeStartup(self.settings, self.data_client_config, self.exec_client_config)
        logger.info(
            "Deribit TestNet client configs prepared (not connected): data_config=%s, exec_config=%s",
            type(self.data_client_config).__name__,
            type(self.exec_client_config).__name__,
        )
        return True

    def prepare_live_node(
        self,
        call_contract: OptionContract,
        put_contract: OptionContract,
        *,
        amount_min: float | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        state_path: Path | None = None,
        order_submission_enabled: bool = False,
    ) -> DeribitNautilusService:
        """Start the NautilusTrader TestNet node and selected-pair strategy in a worker."""
        if not self.settings.deribit_testnet:
            raise ValueError("Automated execution is restricted to Deribit TestNet.")
        self.live_node = LiveNodeStartup(
            settings=self.settings,
            data_client_config=self.data_client_config,
            exec_client_config=self.exec_client_config,
            call_contract=call_contract,
            put_contract=put_contract,
            amount_min=amount_min,
            state_path=state_path,
            order_submission_enabled=order_submission_enabled,
        )
        return self.live_node.start(client_id=client_id, client_secret=client_secret)

    def register_market_data_subscription(self, instrument_ids: list[str]) -> list[str]:
        if self.live_node is None:
            raise RuntimeError("Start the selected-pair strategy before registering subscriptions.")
        return self.live_node.register_market_data_subscription(instrument_ids)

    def register_reconnect_handlers(self) -> dict[str, Any]:
        if self.live_node is None:
            raise RuntimeError("Start the strategy before inspecting its reconciliation settings.")
        return self.live_node.register_reconnect_handlers()

    def _demo_catalog(self) -> tuple[float, list[dict[str, Any]]]:
        reference_price = 65000.0
        expiry = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        instruments: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for strike in [62000.0, 63000.0, 64000.0, 65000.0, 66000.0, 67000.0, 68000.0]:
            instruments.append(
                {
                    "instrument_id": f"BTC-{expiry[:-9].replace('-', '')}-{strike:g}-C",
                    "underlying": "BTC",
                    "option_type": "CALL",
                    "strike": strike,
                    "expiry": expiry,
                    "is_active": True,
                    "is_tradeable": True,
                    "multiplier": 1.0,
                    "tick_size": 0.5,
                    "min_quantity": 1.0,
                    "lot_size": 1.0,
                    "quote_currency": "USD",
                    "settlement_currency": "BTC",
                    "timestamp": now,
                }
            )
            instruments.append(
                {
                    "instrument_id": f"BTC-{expiry[:-9].replace('-', '')}-{strike:g}-P",
                    "underlying": "BTC",
                    "option_type": "PUT",
                    "strike": strike,
                    "expiry": expiry,
                    "is_active": True,
                    "is_tradeable": True,
                    "multiplier": 1.0,
                    "tick_size": 0.5,
                    "min_quantity": 1.0,
                    "lot_size": 1.0,
                    "quote_currency": "USD",
                    "settlement_currency": "BTC",
                    "timestamp": now,
                }
            )

        return reference_price, instruments

    def _mock_quotes(self, call_contract: OptionContract, put_contract: OptionContract) -> tuple[OptionQuote, OptionQuote]:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        call_quote = OptionQuote(
            instrument_id=call_contract.instrument_id,
            option_type=call_contract.option_type,
            strike=call_contract.strike,
            expiry=call_contract.expiry,
            bid=95.0,
            ask=100.0,
            bid_size=2.0,
            ask_size=2.0,
            timestamp=now,
            is_active=True,
        )
        put_quote = OptionQuote(
            instrument_id=put_contract.instrument_id,
            option_type=put_contract.option_type,
            strike=put_contract.strike,
            expiry=put_contract.expiry,
            bid=90.0,
            ask=95.0,
            bid_size=2.0,
            ask_size=2.0,
            timestamp=now,
            is_active=True,
        )
        return call_quote, put_quote

    def build_execution_plan(self) -> RuntimePlan:
        """Build a single-cycle plan across the project modules without requiring a live exchange session."""
        if self.live_trading:
            raise RuntimeError(
                "Live trading is not implemented. Set LIVE_TRADING_ENABLED=false to run the demo."
            )

        reference_price, instruments = self._demo_catalog()
        filtered = filter_btc_option_instruments(instruments)

        calls = [
            OptionContract(
                instrument_id=item.instrument_id,
                option_type=item.option_type,
                strike=item.strike,
                expiry=item.expiry,
                is_active=item.is_active,
            )
            for item in filtered
            if item.option_type == "CALL"
        ]
        puts = [
            OptionContract(
                instrument_id=item.instrument_id,
                option_type=item.option_type,
                strike=item.strike,
                expiry=item.expiry,
                is_active=item.is_active,
            )
            for item in filtered
            if item.option_type == "PUT"
        ]

        if not calls or not puts:
            raise ValueError("No valid BTC option instruments were discovered for the demo cycle.")

        call_contract, put_contract = select_atm_pair(reference_price, calls, puts)
        short_straddle = create_short_straddle(
            StraddleLeg(
                instrument_id=call_contract.instrument_id,
                option_type=call_contract.option_type,
                strike=call_contract.strike,
                expiry=call_contract.expiry,
                quantity=1.0,
            ),
            StraddleLeg(
                instrument_id=put_contract.instrument_id,
                option_type=put_contract.option_type,
                strike=put_contract.strike,
                expiry=put_contract.expiry,
                quantity=1.0,
            ),
            order_quantity=1.0,
        )

        orders = list(short_straddle.build_sell_orders())
        call_leg = LegState(call_contract.instrument_id, "SELL", requested_quantity=1.0)
        put_leg = LegState(put_contract.instrument_id, "SELL", requested_quantity=1.0)
        machine = StrategyStateMachine(call_leg, put_leg)
        machine.begin_entry()

        call_quote, put_quote = self._mock_quotes(call_contract, put_contract)
        if not call_quote.is_valid() or not put_quote.is_valid():
            raise ValueError("Quote validation failed for the demo straddle pair.")

        call_fill = OrderEvent(
            order_id="demo-call-fill",
            instrument_id=call_contract.instrument_id,
            side="SELL",
            status=OrderStatus.FILLED,
            quantity=1.0,
            filled_quantity=1.0,
            average_fill_price=call_quote.ask,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        put_fill = OrderEvent(
            order_id="demo-put-fill",
            instrument_id=put_contract.instrument_id,
            side="SELL",
            status=OrderStatus.FILLED,
            quantity=1.0,
            filled_quantity=1.0,
            average_fill_price=put_quote.ask,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        machine.handle_event(call_fill)
        machine.handle_event(put_fill)

        thresholds = compute_risk_thresholds(machine.entry_premium)
        plan = RuntimePlan(
            settings=self.settings,
            live_trading=self.live_trading,
            connected=self.connected,
            data_client_config=self.data_client_config,
            exec_client_config=self.exec_client_config,
            reference_price=reference_price,
            call_contract=call_contract,
            put_contract=put_contract,
            short_straddle=short_straddle,
            orders=orders,
            fill_events=[call_fill, put_fill],
            machine_state=machine.state,
            entry_premium=machine.entry_premium,
            thresholds=thresholds,
            summary={
                "mode": "demo",
                "status": "demo_ready",
                "entry_state": machine.state.value,
                "matched_quantity": machine.matched_quantity,
                "has_unmatched_exposure": machine.has_unmatched_exposure,
            },
        )
        self.plan = plan
        return plan

    def run(self) -> RuntimePlan:
        """Run the full strategy cycle and return the prepared execution plan."""
        return self.build_execution_plan()


def run_runtime(*, live_trading: bool | None = None) -> RuntimePlan:
    runtime = StrategyRuntime(live_trading=live_trading)
    return runtime.run()
