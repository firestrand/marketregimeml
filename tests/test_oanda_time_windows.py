from datetime import datetime, timedelta
from marketregimeml.data.loaders.oanda import OANDADataLoader


def test_generate_time_windows_minute_granularity():
    start = datetime(2025, 1, 1)
    end = datetime(2025, 1, 15)
    wins = list(OANDADataLoader._generate_time_windows(start, end, "M5", 5000))
    assert len(wins) >= 1
    # Windows increase monotonically and cover the range
    assert wins[0][0].replace(tzinfo=None) >= start
    assert wins[-1][1].replace(tzinfo=None) <= end
    for i in range(1, len(wins)):
        assert wins[i][0].replace(tzinfo=None) > wins[i - 1][1].replace(tzinfo=None)


def test_generate_time_windows_hour_granularity():
    start = datetime(2025, 2, 1)
    end = datetime(2025, 3, 1)
    wins = list(OANDADataLoader._generate_time_windows(start, end, "H1", 5000))
    assert len(wins) >= 1
    assert wins[0][0].replace(tzinfo=None) >= start
    assert wins[-1][1].replace(tzinfo=None) <= end
