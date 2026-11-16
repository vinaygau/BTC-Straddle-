
"""Project settings and Deribit TestNet configuration."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from nautilus_trader.adapters.deribit.config import (
    DeribitDataClientConfig,
    DeribitEnvironment,
    DeribitExecClientConfig,
    DeribitProductType,
)
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.live.config import LiveExecEngineConfig
from nautilus_trader.live.config import TradingNodeConfig
from nautilus_trader.model.identifiers import InstrumentId, TraderId


# Load .env from the project root, regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

__all__ = [
    "Settings",
    "configure_logging",
    "get_deribit_credentials",
    "get_deribit_connection_status",
    "create_deribit_data_config",
    "create_deribit_exec_config",
    "build_deribit_client_configs",
    "build_trading_node_config",
]


def _get_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Application settings."""

    app_env: str
    log_level: str
    deribit_testnet: bool
    live_trading_enabled: bool
    deribit_testnet_order_submission_enabled: bool

    @classmethod
    def from_env(cls):
        """Create application settings from environment variables."""
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            deribit_testnet=_get_bool("DERIBIT_TESTNET", True),
            live_trading_enabled=_get_bool(
                "LIVE_TRADING_ENABLED", False
            ),
            deribit_testnet_order_submission_enabled=_get_bool(
                "DERIBIT_TESTNET_ORDER_SUBMISSION_ENABLED", False
            ),
        )


def configure_logging(settings: Settings) -> None:
    """Configure application logging."""
    log_level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def get_deribit_credentials() -> tuple[str, str]:
    """Load Deribit API credentials securely."""

    client_id = os.getenv("DERIBIT_TESTNET_CLIENT_ID")
    client_secret = os.getenv("DERIBIT_TESTNET_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "Deribit TestNet credentials are missing. "
            "Check DERIBIT_TESTNET_CLIENT_ID and "
            "DERIBIT_TESTNET_CLIENT_SECRET in the project-root .env file."
        )

    return client_id, client_secret


def get_deribit_connection_status() -> dict[str, bool | str]:
    """Return a safe summary of the project’s Deribit TestNet readiness."""
    settings = Settings.from_env()
    ready = settings.deribit_testnet and bool(
        os.getenv("DERIBIT_TESTNET_CLIENT_ID")
        and os.getenv("DERIBIT_TESTNET_CLIENT_SECRET")
    )

    return {
        "deribit_testnet": bool(settings.deribit_testnet),
        "credentials_loaded": ready,
        "testnet_order_submission_enabled": bool(
            settings.deribit_testnet_order_submission_enabled
        ),
        "environment": "TESTNET" if settings.deribit_testnet else "UNSPECIFIED",
    }


def build_deribit_client_configs() -> tuple[DeribitDataClientConfig, DeribitExecClientConfig]:
    """Build both NautilusTrader Deribit configs for TestNet execution."""
    return create_deribit_data_config(), create_deribit_exec_config()


def build_trading_node_config(
    data_client_config: DeribitDataClientConfig | None = None,
    exec_client_config: DeribitExecClientConfig | None = None,
    *,
    trader_id: TraderId | str = "TRADER-001",
    exec_engine: LiveExecEngineConfig | None = None,
) -> TradingNodeConfig:
    """Construct the NautilusTrader live-node config for the Deribit TestNet runtime."""
    if data_client_config is None:
        data_client_config = create_deribit_data_config()
    if exec_client_config is None:
        exec_client_config = create_deribit_exec_config()

    return TradingNodeConfig(
        trader_id=trader_id,
        data_clients={"DERIBIT": data_client_config},
        exec_clients={"DERIBIT": exec_client_config},
        exec_engine=exec_engine or LiveExecEngineConfig(reconciliation=True),
    )


def _instrument_provider(instrument_ids: list[InstrumentId] | None) -> InstrumentProviderConfig | None:
    if not instrument_ids:
        return None
    return InstrumentProviderConfig(load_ids=frozenset(instrument_ids))


def create_deribit_data_config(
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    instrument_ids: list[InstrumentId] | None = None,
) -> DeribitDataClientConfig:
    """Create NautilusTrader Deribit market-data configuration."""

    settings = Settings.from_env()

    if not settings.deribit_testnet:
        raise ValueError("This project is configured for Deribit TestNet only.")

    if client_id is None or client_secret is None:
        client_id, client_secret = get_deribit_credentials()

    return DeribitDataClientConfig(
        api_key=client_id,
        api_secret=client_secret,
        environment=DeribitEnvironment.TESTNET,
        product_types=(DeribitProductType.OPTION,),
        instrument_provider=_instrument_provider(instrument_ids),
    )


def create_deribit_exec_config(
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    instrument_ids: list[InstrumentId] | None = None,
) -> DeribitExecClientConfig:
    """Create NautilusTrader Deribit execution configuration."""

    settings = Settings.from_env()

    if not settings.deribit_testnet:
        raise ValueError("This project is configured for Deribit TestNet only.")

    if client_id is None or client_secret is None:
        client_id, client_secret = get_deribit_credentials()

    return DeribitExecClientConfig(
        api_key=client_id,
        api_secret=client_secret,
        environment=DeribitEnvironment.TESTNET,
        product_types=(DeribitProductType.OPTION,),
        instrument_provider=_instrument_provider(instrument_ids),
    )
