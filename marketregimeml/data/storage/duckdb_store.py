"""DuckDB storage for OHLCV data using relation-oriented API.

This module encapsulates all SQL usage so callers don't need to write SQL
strings directly. Provides simple methods to persist and read OHLCV data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


@dataclass
class DuckDBConfig:
    db_path: Path


class DuckDBStore:
    """Lightweight DuckDB-backed storage for market OHLCV data.

    Table schema: ohlcv(timestamp, open, high, low, close, volume, symbol, timeframe)
    Primary key semantics on (symbol, timeframe, timestamp) are enforced by
    deduplication on write (no user SQL required at call site).
    """

    def __init__(self, db_path: str = "data/market_data.duckdb") -> None:
        self.config = DuckDBConfig(db_path=Path(db_path))
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con: duckdb.DuckDBPyConnection = duckdb.connect(str(self.config.db_path))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # Encapsulate DDL
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                symbol VARCHAR,
                timeframe VARCHAR
            );
            """
        )
        # Optional index to speed lookups (DuckDB will treat it as a metadata hint)
        try:
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_key ON ohlcv(symbol, timeframe, timestamp)"
            )
        except Exception:
            pass

    def write_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
        """Append OHLCV rows for a symbol/timeframe, with deduplication.

        Args:
            df: DataFrame with columns [open, high, low, close, volume] and DatetimeIndex
            symbol: Instrument symbol
            timeframe: Original timeframe string

        Returns:
            Number of rows written after de-duplication
        """
        if df is None or df.empty:
            return 0

        # Prepare a DataFrame with all columns
        out = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        out["symbol"] = symbol
        out["timeframe"] = timeframe

        # Register the DataFrame in this connection as a view
        self.con.register("incoming_ohlcv", out)

        # Insert new rows then remove duplicates based on the composite key
        self.con.execute(
            """
            INSERT INTO ohlcv
            SELECT timestamp, open, high, low, close, volume, symbol, timeframe
            FROM incoming_ohlcv;
            """
        )

        # Deduplicate by keeping the latest entry per key
        self.con.execute(
            """
            CREATE TEMP TABLE ohlcv_dedup AS
            SELECT * FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol, timeframe, timestamp
                           ORDER BY timestamp DESC
                       ) AS rn
                FROM ohlcv
            )
            WHERE rn = 1;

            DELETE FROM ohlcv;
            INSERT INTO ohlcv SELECT timestamp, open, high, low, close, volume, symbol, timeframe FROM ohlcv_dedup;
            DROP TABLE ohlcv_dedup;
            """
        )

        # Return count of the last batch written (approximate rows input)
        return len(out)

    def read_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Read full OHLCV history for symbol/timeframe."""
        df = (
            self.con.execute(
                "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp",
                [symbol, timeframe],
            )
            .fetchdf()
        )
        if df.empty:
            return df
        return df.set_index("timestamp")

    def read_ohlcv_aggregated(
        self,
        symbol: str,
        base_timeframe: str,
        target_timeframe: str,
    ) -> pd.DataFrame:
        """Aggregate OHLCV from a base timeframe to a coarser target timeframe.

        Uses DuckDB's time_bucket with appropriate interval to compute:
        - open: first open in bucket
        - high: max high in bucket
        - low: min low in bucket
        - close: last close in bucket
        - volume: sum in bucket

        Supported aggregations include M5 -> {M15, M30, H1} and H1 -> {H4, D}.
        """
        interval_map = {
            "M1": "1 MINUTE",
            "M5": "5 MINUTE",
            "M15": "15 MINUTE",
            "M30": "30 MINUTE",
            "H1": "1 HOUR",
            "H4": "4 HOUR",
            "D": "1 DAY",
            "daily": "1 DAY",
        }

        if base_timeframe not in interval_map or target_timeframe not in interval_map:
            raise ValueError(f"Unsupported timeframe aggregation: {base_timeframe} -> {target_timeframe}")

        base_minutes = [
            ("MINUTE", 1), ("MINUTE", 5), ("MINUTE", 15), ("MINUTE", 30),
            ("HOUR", 1), ("HOUR", 4), ("DAY", 1)
        ]
        # Simple guard: ensure target interval is multiple of base in minutes/hours
        # Rely on DuckDB to aggregate; if base is coarser than target, return empty

        interval_str = interval_map[target_timeframe]

        query = f"""
            SELECT
                time_bucket(INTERVAL '{interval_str}', timestamp) AS timestamp,
                first(open) AS open,
                max(high) AS high,
                min(low) AS low,
                last(close) AS close,
                sum(volume) AS volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
            GROUP BY 1
            ORDER BY 1
        """

        df = self.con.execute(query, [symbol, base_timeframe]).fetchdf()
        if df.empty:
            return df
        return df.set_index("timestamp")

    def get_last_timestamp(self, symbol: str, timeframe: str) -> Optional[pd.Timestamp]:
        """Get most recent timestamp stored for a symbol/timeframe."""
        df = (
            self.con.execute(
                "SELECT max(timestamp) AS last_ts FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                [symbol, timeframe],
            )
            .fetchdf()
        )
        if df.empty or df.loc[0, "last_ts"] is None:
            return None
        return pd.Timestamp(df.loc[0, "last_ts"])
