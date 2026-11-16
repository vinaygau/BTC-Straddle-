from strategies.atm_selection import OptionContract, select_atm_pair


def test_select_atm_pair_uses_same_expiry_and_nearest_strike():
    calls = [
        OptionContract("BTC-25DEC27-49000-C", "CALL", 49000.0, "2027-12-25"),
        OptionContract("BTC-25DEC27-50000-C", "CALL", 50000.0, "2027-12-25"),
        OptionContract("BTC-25DEC27-51000-C", "CALL", 51000.0, "2027-12-25"),
    ]
    puts = [
        OptionContract("BTC-25DEC27-49000-P", "PUT", 49000.0, "2027-12-25"),
        OptionContract("BTC-25DEC27-50000-P", "PUT", 50000.0, "2027-12-25"),
        OptionContract("BTC-25DEC27-51000-P", "PUT", 51000.0, "2027-12-25"),
    ]

    call, put = select_atm_pair(50050.0, calls, puts)
    assert call.strike == 50000.0
    assert put.strike == 50000.0
    assert call.expiry == put.expiry


def test_select_atm_pair_rejects_mismatched_expiry():
    calls = [OptionContract("BTC-25DEC27-50000-C", "CALL", 50000.0, "2027-12-25")]
    puts = [OptionContract("BTC-25DEC28-50000-P", "PUT", 50000.0, "2028-12-25")]

    try:
        select_atm_pair(50000.0, calls, puts)
        assert False, "expected ValueError for mismatched expiry"
    except ValueError:
        pass
