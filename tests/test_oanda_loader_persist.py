import pandas as pd
from datetime import datetime

from marketregimeml.data.loaders.oanda import OANDADataLoader
from marketregimeml.data.storage.duckdb_store import DuckDBStore


def test_oanda_loader_persists(monkeypatch, tmp_path):
    # Mock _fetch_ohlcv_impl to avoid network
    def fake_fetch(self, symbol, timeframe, start_date, end_date, limit):
        idx = pd.date_range(end=datetime(2025, 1, 10), periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [1.0] * 5,
                "high": [1.1] * 5,
                "low": [0.9] * 5,
                "close": [1.05] * 5,
                "volume": [100] * 5,
            },
            index=idx,
        )
        df.index.name = "timestamp"
        return df

    monkeypatch.setattr(OANDADataLoader, "_fetch_ohlcv_impl", fake_fetch, raising=True)

    db_path = tmp_path / "persist.duckdb"
    loader = OANDADataLoader(store_to_duckdb=True, duckdb_path=str(db_path))
    df = loader.fetch_ohlcv("EUR_USD", "M5", datetime(2025, 1, 1), datetime(2025, 1, 10))
    assert not df.empty

    store = DuckDBStore(str(db_path))
    out = store.read_ohlcv("EUR_USD", "M5")
    assert len(out) == 5

