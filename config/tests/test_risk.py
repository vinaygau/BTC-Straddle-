from strategies.risk_management import compute_risk_thresholds


def test_risk_thresholds_are_based_on_entry_premium():
    thresholds = compute_risk_thresholds(100.0)
    assert thresholds.stop_loss == 101.0
    assert thresholds.take_profit == 98.0
