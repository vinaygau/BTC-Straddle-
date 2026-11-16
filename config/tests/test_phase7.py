from strategies.risk_management import (
    RiskThresholds,
    compute_risk_thresholds,
    evaluate_buyback_risk,
    should_stop_loss,
    should_take_profit,
)


def test_risk_thresholds_follow_assignment_rules():
    thresholds = compute_risk_thresholds(100.0)
    assert thresholds.stop_loss == 101.0
    assert thresholds.take_profit == 98.0


def test_buyback_risk_triggers_on_stop_loss_and_take_profit():
    thresholds = RiskThresholds(100.0, 101.0, 98.0)

    stop_eval = evaluate_buyback_risk(101.0, thresholds)
    take_eval = evaluate_buyback_risk(98.0, thresholds)

    assert stop_eval.reason == "stop_loss"
    assert stop_eval.triggered is True
    assert take_eval.reason == "take_profit"
    assert take_eval.triggered is True


def test_buyback_risk_filters_values_between_thresholds():
    thresholds = RiskThresholds(100.0, 101.0, 98.0)
    eval_result = evaluate_buyback_risk(100.0, thresholds)
    assert eval_result.reason is None
    assert eval_result.triggered is False
    assert should_stop_loss(100.0, thresholds) is False
    assert should_take_profit(100.0, thresholds) is False
