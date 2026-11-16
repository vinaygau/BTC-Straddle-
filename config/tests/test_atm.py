from strategies.atm_selection import OptionContract, select_atm_pair


def test_select_atm_pair_prefers_matching_strike_and_expiry():
    call = OptionContract(
        instrument_id="BTC-25DEC24-50000-C",
        option_type="CALL",
        strike=50000.0,
        expiry="2024-12-25",
        is_active=True,
    )
    put = OptionContract(
        instrument_id="BTC-25DEC24-50000-P",
        option_type="PUT",
        strike=50000.0,
        expiry="2024-12-25",
        is_active=True,
    )

    selected_call, selected_put = select_atm_pair(49990.0, [call], [put])
    assert selected_call.strike == 50000.0
    assert selected_put.strike == 50000.0
    assert selected_call.expiry == selected_put.expiry
