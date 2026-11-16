"""Explicit, TestNet-only Deribit authentication and option-order operations."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from strategies.atm_selection import OptionContract


TESTNET_API_ROOT = "https://test.deribit.com/api/v2"
REQUEST_TIMEOUT_SECONDS = 10
REQUIRED_SCOPES = "account:read trade:read_write"


class DeribitTestNetError(RuntimeError):
    """A safe-to-display TestNet API or validation error."""


class DeribitTestNetRequestUncertain(DeribitTestNetError):
    """A request may have reached Deribit but its response was not received."""


@dataclass
class DeribitTestNetClient:
    """A short-lived TestNet client whose endpoint cannot be changed to production."""

    client_id: str
    _client_secret: str = field(repr=False)
    order_submission_enabled: bool = False
    _access_token: str | None = field(default=None, repr=False)
    _refresh_token: str | None = field(default=None, repr=False)
    _token_expires_at: float = field(default=0.0, repr=False)
    _request_id: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.client_id.strip() or not self._client_secret.strip():
            raise DeribitTestNetError("Deribit TestNet client credentials are missing.")

    @property
    def authenticated(self) -> bool:
        # A stored refresh token keeps the session connected after the access
        # token expires; the next private request refreshes it server-side.
        return bool(self._access_token and self._refresh_token)

    def authenticate(self) -> dict[str, Any]:
        """Request an OAuth access token with only account-read and trade scopes."""
        result = self._public_rpc(
            "public/auth",
            {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self._client_secret,
                "scope": REQUIRED_SCOPES,
            },
        )
        self._store_tokens(result)
        granted = set(str(result.get("scope", "")).split())
        if granted and not {"account:read", "trade:read_write"}.issubset(granted):
            self._clear_tokens()
            raise DeribitTestNetError(
                "The TestNet API key did not grant both account:read and "
                "trade:read_write scopes. Check the key permissions in Deribit."
            )
        return {
            "scope": str(result.get("scope", "")),
            "expires_in": int(result.get("expires_in", 0) or 0),
        }

    def get_account_summary(self, currency: str = "BTC") -> dict[str, Any]:
        """Read the selected TestNet account summary after explicit authentication."""
        result = self._private_rpc(
            "private/get_account_summary", {"currency": currency.upper()}
        )
        if not isinstance(result, dict):
            raise DeribitTestNetError("Deribit returned an invalid account summary.")
        return result

    def get_open_orders(self, instrument_id: str) -> list[dict[str, Any]]:
        """List open TestNet orders for one instrument."""
        if not instrument_id.startswith("BTC-"):
            raise DeribitTestNetError("Only BTC option orders can be managed here.")
        result = self._private_rpc(
            "private/get_open_orders_by_instrument",
            {"instrument_name": instrument_id},
        )
        if not isinstance(result, list):
            raise DeribitTestNetError("Deribit returned an invalid open-orders response.")
        return [row for row in result if isinstance(row, dict)]

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel one open TestNet order after an explicit UI action."""
        if not order_id.strip():
            raise DeribitTestNetError("An order ID is required to cancel an order.")
        result = self._private_rpc("private/cancel", {"order_id": order_id})
        if not isinstance(result, dict):
            raise DeribitTestNetError("Deribit returned an invalid cancel response.")
        return result

    def submit_short_straddle(
        self,
        call: OptionContract,
        put: OptionContract,
        amount_btc: float,
        call_price_btc: float,
        put_price_btc: float,
    ) -> dict[str, Any]:
        """Submit two separate post-only limit sells and compensate if leg two fails.

        Deribit processes each leg independently. If the second submission fails,
        the method attempts to cancel the first leg's remaining open amount and
        reports the result; a completed or partial fill cannot be rolled back.
        """
        if not self.order_submission_enabled:
            raise DeribitTestNetError(
                "TestNet order submission is disabled. Set "
                "DERIBIT_TESTNET_ORDER_SUBMISSION_ENABLED=true to opt in."
            )
        if not self.authenticated:
            raise DeribitTestNetError("Connect to the Deribit TestNet account first.")
        self._validate_straddle(call, put)
        self._validate_amount(amount_btc, call)
        self._validate_amount(amount_btc, put)
        self._validate_price(call_price_btc, call)
        self._validate_price(put_price_btc, put)
        # Keep token renewal outside the uncertain order-submission branch.
        self._ensure_token()

        request_tag = uuid.uuid4().hex[:12]
        try:
            call_result = self._private_rpc(
                "private/sell",
                self._sell_params(
                    call,
                    amount_btc,
                    call_price_btc,
                    f"btcstraddle-{request_tag}-call",
                ),
            )
        except DeribitTestNetRequestUncertain as exc:
            return {
                "status": "call_response_uncertain",
                "call_order": None,
                "put_order": None,
                "error": str(exc),
                "compensation": None,
                "warning": (
                    "The Call request may have been accepted without returning an order ID. "
                    "Refresh open orders for both instruments and check fills before retrying."
                ),
            }
        call_order = call_result.get("order") if isinstance(call_result, dict) else None
        if not isinstance(call_order, dict) or not call_order.get("order_id"):
            raise DeribitTestNetError(
                "Deribit returned no Call order ID. Check TestNet open orders before retrying."
            )

        try:
            put_result = self._private_rpc(
                "private/sell",
                self._sell_params(
                    put,
                    amount_btc,
                    put_price_btc,
                    f"btcstraddle-{request_tag}-put",
                ),
            )
        except DeribitTestNetError as exc:
            compensation = self._cancel_first_leg_if_open(call_order)
            return {
                "status": "second_leg_failed",
                "call_order": call_order,
                "put_order": None,
                "error": str(exc),
                "compensation": compensation,
                "warning": (
                    "The legs are not atomic. A Call fill or partial fill may remain "
                    "as TestNet exposure even when its unfilled remainder is cancelled. "
                    "If the Put request timed out, it may also have been accepted; refresh "
                    "both instruments and check fills before retrying."
                ),
            }

        put_order = put_result.get("order") if isinstance(put_result, dict) else None
        if not isinstance(put_order, dict):
            return {
                "status": "put_response_uncertain",
                "call_order": call_order,
                "put_order": None,
                "error": "No Put order object was returned. Check TestNet open orders before retrying.",
                "compensation": self._cancel_first_leg_if_open(call_order),
                "warning": (
                    "The legs are not atomic. Check both instruments for orders and fills."
                ),
            }

        return {
            "status": "submitted",
            "call_order": call_order,
            "put_order": put_order,
            "warning": (
                "These are two separate GTC orders. Either leg can remain open or fill "
                "without the other. Review and manage each order independently."
            ),
        }

    @staticmethod
    def _validate_straddle(call: OptionContract, put: OptionContract) -> None:
        if call.option_type.upper() != "CALL" or put.option_type.upper() != "PUT":
            raise DeribitTestNetError("The selected contracts must be one Call and one Put.")
        if call.strike != put.strike or call.expiry != put.expiry:
            raise DeribitTestNetError("Call and Put must share the same strike and expiry.")
        if not call.is_active or not put.is_active:
            raise DeribitTestNetError("Both selected TestNet instruments must be active.")
        if not call.instrument_id.startswith("BTC-") or not call.instrument_id.endswith("-C"):
            raise DeribitTestNetError("The Call must be a BTC Call from the TestNet catalog.")
        if not put.instrument_id.startswith("BTC-") or not put.instrument_id.endswith("-P"):
            raise DeribitTestNetError("The Put must be a BTC Put from the TestNet catalog.")
        if call.min_trade_amount is None or put.min_trade_amount is None:
            raise DeribitTestNetError("Instrument amount rules are missing; refresh the TestNet chain.")
        if call.tick_size is None or put.tick_size is None:
            raise DeribitTestNetError("Instrument price rules are missing; refresh the TestNet chain.")

    @staticmethod
    def _decimal(value: float, description: str) -> Decimal:
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise DeribitTestNetError(f"{description} must be a valid number.") from None
        if not result.is_finite():
            raise DeribitTestNetError(f"{description} must be finite.")
        return result

    @classmethod
    def _validate_amount(cls, amount: float, contract: OptionContract) -> None:
        value = cls._decimal(amount, "Amount")
        increment = cls._decimal(contract.min_trade_amount or 0, "Minimum amount")
        if value <= 0 or increment <= 0 or value < increment:
            raise DeribitTestNetError(
                f"Amount must be at least {contract.min_trade_amount:g} BTC for {contract.instrument_id}."
            )
        units = value / increment
        if units != units.to_integral_value():
            raise DeribitTestNetError(
                f"Amount must be a multiple of {contract.min_trade_amount:g} BTC for {contract.instrument_id}."
            )

    @classmethod
    def _validate_price(cls, price: float, contract: OptionContract) -> None:
        value = cls._decimal(price, "Limit price")
        if value <= 0:
            raise DeribitTestNetError("Limit prices must be positive.")
        tick = cls._decimal(contract.tick_size or 0, "Price tick")
        if tick <= 0:
            raise DeribitTestNetError("The selected instrument has an invalid price tick.")
        for above_price, stepped_tick in sorted(contract.tick_size_steps):
            if value >= cls._decimal(above_price, "Tick step price"):
                tick = cls._decimal(stepped_tick, "Tick step")
            else:
                break
        units = value / tick
        if units != units.to_integral_value():
            raise DeribitTestNetError(
                f"Limit price must be a multiple of {tick} BTC for {contract.instrument_id}."
            )

    @staticmethod
    def _sell_params(
        contract: OptionContract,
        amount_btc: float,
        price_btc: float,
        label: str,
    ) -> dict[str, Any]:
        return {
            "instrument_name": contract.instrument_id,
            "amount": amount_btc,
            "type": "limit",
            "price": price_btc,
            "time_in_force": "good_til_cancelled",
            "post_only": True,
            "reject_post_only": True,
            "label": label,
        }

    def _cancel_first_leg_if_open(self, order: dict[str, Any]) -> dict[str, Any]:
        order_id = str(order.get("order_id") or "")
        state = str(order.get("order_state") or "unknown").lower()
        if not order_id:
            return {"status": "unknown", "message": "No order ID was available to cancel."}
        if state not in {"open", "untriggered"}:
            return {
                "status": "not_cancellable",
                "order_state": state,
                "message": "The first leg is not open; any fill or partial fill remains as exposure.",
            }
        try:
            result = self._private_rpc("private/cancel", {"order_id": order_id})
            return {
                "status": "cancel_requested",
                "order_state": result.get("order_state") if isinstance(result, dict) else None,
            }
        except DeribitTestNetError as exc:
            return {"status": "cancel_failed", "message": str(exc)}

    def _store_tokens(self, result: dict[str, Any]) -> None:
        access_token = result.get("access_token")
        refresh_token = result.get("refresh_token")
        if not access_token or not refresh_token:
            raise DeribitTestNetError("Deribit did not return authentication tokens.")
        expires_in = int(result.get("expires_in", 0) or 0)
        if expires_in <= 0:
            raise DeribitTestNetError("Deribit returned an invalid token lifetime.")
        self._access_token = str(access_token)
        self._refresh_token = str(refresh_token)
        self._token_expires_at = time.time() + expires_in

    def _clear_tokens(self) -> None:
        self._access_token = None
        self._refresh_token = None
        self._token_expires_at = 0.0

    def _ensure_token(self) -> None:
        if self._access_token and self._token_expires_at > time.time() + 30:
            return
        if self._refresh_token:
            try:
                result = self._public_rpc(
                    "public/auth",
                    {"grant_type": "refresh_token", "refresh_token": self._refresh_token},
                )
                self._store_tokens(result)
                return
            except DeribitTestNetError:
                self._clear_tokens()
        self.authenticate()

    def _private_rpc(self, method: str, params: dict[str, Any]) -> Any:
        self._ensure_token()
        return self._rpc(method, params, access_token=self._access_token)

    def _public_rpc(self, method: str, params: dict[str, Any]) -> Any:
        return self._rpc(method, params)

    def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        access_token: str | None = None,
    ) -> Any:
        self._request_id += 1
        body = json.dumps(
            {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "BTCOptionsStraddleTestNet/1.0",
        }
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        request = Request(
            f"{TESTNET_API_ROOT}/{method}",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code >= 500:
                raise DeribitTestNetRequestUncertain(
                    f"Deribit TestNet returned HTTP {exc.code} for {method}; order state may be uncertain."
                ) from None
            raise DeribitTestNetError(f"Deribit TestNet returned HTTP {exc.code} for {method}.") from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError):
            raise DeribitTestNetRequestUncertain(
                f"Could not complete the Deribit TestNet request for {method}. "
                "Check the network and refresh account/order status before retrying."
            ) from None

        error = payload.get("error") if isinstance(payload, dict) else None
        if error:
            code = error.get("code", "unknown")
            message = str(error.get("message", "Unknown Deribit API error"))
            raise DeribitTestNetError(f"Deribit {method} failed ({code}): {message}")
        if not isinstance(payload, dict) or "result" not in payload:
            raise DeribitTestNetError(f"Deribit {method} returned no result.")
        return payload["result"]
