import pytest

from runtime import LiveNodeStartup, StrategyRuntime


def test_runtime_orchestrates_demo_cycle():
    runtime = StrategyRuntime(live_trading=False)
    plan = runtime.run()

    assert plan.connected in {True, False}
    assert plan.call_contract is not None
    assert plan.put_contract is not None
    assert plan.call_contract.instrument_id != plan.put_contract.instrument_id
    assert plan.call_contract.strike == plan.put_contract.strike
    assert plan.call_contract.expiry == plan.put_contract.expiry
    assert plan.summary["matched_quantity"] >= 1.0
    assert plan.entry_premium > 0.0
    assert plan.thresholds.stop_loss > plan.thresholds.take_profit


def test_live_node_startup_requires_explicit_testnet_pair_and_opt_in():
    runtime = StrategyRuntime(live_trading=False)
    with pytest.raises(RuntimeError, match="Enable TestNet order submission"):
        runtime.prepare_live_node(None, None)
    startup = LiveNodeStartup(runtime.settings, order_submission_enabled=True)
    with pytest.raises(ValueError, match="BTC Call and Put pair"):
        startup.start()
