from datetime import datetime, timezone

from data import binance_history


def test_historical_candle_download_maps_requested_utc_dates(monkeypatch, tmp_path):
    day1 = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    day2 = int(datetime(2025, 1, 2, tzinfo=timezone.utc).timestamp() * 1000)
    page = [
        [day1, "1", "2", ".5", "1.5", "10", day1 + 299_999, "15", 4, "6", "9", "0"],
        [day2, "1.5", "2", "1", "1.8", "12", day2 + 299_999, "20", 5, "7", "11", "0"],
    ]
    monkeypatch.setattr(binance_history, "_request_page", lambda *_args: page)

    candles = binance_history.download_btcusdt_history("5m", "2025-01-01", "2025-01-01")

    assert len(candles) == 1
    assert int(candles.iloc[0]["open_time"]) == day1
    assert list(candles.columns) == list(binance_history.KLINE_COLUMNS)
    output = binance_history.save_btcusdt_history(candles, "5m", "2025-01-01", "2025-01-01", tmp_path)
    assert output.is_file()
