
"""BTC options runtime entry point."""

import logging

from config import Settings, configure_logging
from runtime import StrategyRuntime


def main() -> int:
    settings = Settings.from_env()
    configure_logging(settings)

    logger = logging.getLogger("btc_options_straddle")
    logger.info(
        "Project environment ready: app_env=%s, deribit_testnet=%s, live_trading_enabled=%s",
        settings.app_env,
        settings.deribit_testnet,
        settings.live_trading_enabled,
    )

    runtime = StrategyRuntime(settings=settings, live_trading=settings.live_trading_enabled)
    plan = runtime.run()

    logger.info(
        "Runtime mode=%s, connected=%s, reference_price=%s, pair=(%s,%s)",
        plan.summary["mode"],
        plan.connected,
        plan.reference_price,
        plan.call_contract.instrument_id if plan.call_contract else None,
        plan.put_contract.instrument_id if plan.put_contract else None,
    )
    logger.info("Entry state=%s, matched_quantity=%s, entry_premium=%s", plan.machine_state, plan.summary["matched_quantity"], plan.entry_premium)
    logger.info("Risk thresholds: stop_loss=%s, take_profit=%s", plan.thresholds.stop_loss, plan.thresholds.take_profit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
