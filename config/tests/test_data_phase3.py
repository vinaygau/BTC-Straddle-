from data import discover_btc_option_instruments


def test_discover_btc_option_instruments_filters_valid_btc_pairs():
    catalog = [
        {
            "instrument_id": "BTC-25DEC27-50000-C",
            "underlying": "BTC",
            "option_type": "CALL",
            "strike": 50000,
            "expiry": "2027-12-25T00:00:00Z",
            "is_active": True,
            "is_tradeable": True,
        },
        {
            "instrument_id": "BTC-25DEC27-50000-P",
            "underlying": "BTC",
            "option_type": "PUT",
            "strike": 50000,
            "expiry": "2027-12-25T00:00:00Z",
            "is_active": True,
            "is_tradeable": True,
        },
        {
            "instrument_id": "ETH-25DEC27-3000-C",
            "underlying": "ETH",
            "option_type": "CALL",
            "strike": 3000,
            "expiry": "2027-12-25T00:00:00Z",
            "is_active": True,
            "is_tradeable": True,
        },
        {
            "instrument_id": "BTC-25DEC24-50000-C",
            "underlying": "BTC",
            "option_type": "CALL",
            "strike": 50000,
            "expiry": "2024-12-25T00:00:00Z",
            "is_active": True,
            "is_tradeable": True,
        },
    ]

    instruments = discover_btc_option_instruments(catalog)
    ids = {item.instrument_id for item in instruments}
    assert "BTC-25DEC27-50000-C" in ids
    assert "BTC-25DEC27-50000-P" in ids
    assert "ETH-25DEC27-3000-C" not in ids
    assert "BTC-25DEC24-50000-C" not in ids
