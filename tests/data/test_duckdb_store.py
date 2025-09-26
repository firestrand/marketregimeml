import os
import pandas as pd
from datetime import datetime, timedelta

from marketregimeml.data.storage.duckdb_store import DuckDBStore


def test_duckdb_store_write_read(tmp_path):
    db_path = tmp_path / "test.duckdb"
    store = DuckDBStore(str(db_path))

    # Build small OHLCV dataframe
    idx = pd.date_range(end=datetime(2025, 1, 1), periods=3, freq="D")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1],
            "close": [1.1, 1.2, 1.3],
            "volume": [100, 110, 120],
        },
        index=idx,
    )
    df.index.name = "timestamp"

    written = store.write_ohlcv(df, symbol="EUR_USD", timeframe="M5")
    assert written == 3

    # Read back
    out = store.read_ohlcv("EUR_USD", "M5")
    assert len(out) == 3
    assert out.index.min() == idx.min()
    assert store.get_last_timestamp("EUR_USD", "M5") == idx.max()

    # Write duplicates and ensure dedup
    store.write_ohlcv(df.iloc[-2:], symbol="EUR_USD", timeframe="M5")
    out2 = store.read_ohlcv("EUR_USD", "M5")
    assert len(out2) == 3  # no duplicates

