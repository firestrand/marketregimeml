#!/usr/bin/env python
"""Example: Bulk fetching OANDA data with DuckDB storage.

This example demonstrates how to:
1. Fetch large amounts of historical OANDA data in chunks
2. Store data efficiently in DuckDB for later use
3. Resume interrupted downloads from last stored timestamp
4. Handle rate limits with windowed fetching

Prerequisites:
    - Set OANDA_API_KEY and OANDA_ACCOUNT_ID in .env file
    - Install DuckDB: pip install duckdb

Usage:
    python examples/23_data_oanda_bulk_fetch.py EUR_USD M5 30

    Arguments:
        symbol: OANDA instrument (e.g., EUR_USD, GBP_USD)
        timeframe: OANDA granularity (e.g., M5, H1, D)
        days: Number of days of history to fetch
        resume: Whether to resume from last stored timestamp (default: True)

Example - Fetch 30 days of EUR/USD 5-minute data:
    python examples/23_data_oanda_bulk_fetch.py EUR_USD M5 30

Example - Fetch 7 days of Bitcoin hourly data, starting fresh:
    python examples/23_data_oanda_bulk_fetch.py BTC_USD H1 7 false
"""

import sys
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from marketregimeml.data.loaders.oanda import OANDADataLoader
from marketregimeml.data.storage.duckdb_store import DuckDBStore
from marketregimeml.utils.logging import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


def fetch_oanda_bulk(
    symbol: str,
    timeframe: str,
    days_back: int,
    resume: bool = True,
    window_days: int = 7
) -> pd.DataFrame:
    """Fetch historical OANDA data in chunks and store in DuckDB.

    This function fetches data in windows to avoid API limits and
    stores it progressively in DuckDB for efficient retrieval.

    Args:
        symbol: OANDA instrument symbol (e.g., 'EUR_USD')
        timeframe: OANDA granularity (e.g., 'M5', 'H1', 'D')
        days_back: Number of days of historical data to fetch
        resume: If True, resume from last stored timestamp
        window_days: Size of each fetch window in days (default: 7)

    Returns:
        DataFrame with all fetched data

    Example:
        >>> # Fetch 30 days of EUR/USD 5-minute data
        >>> df = fetch_oanda_bulk('EUR_USD', 'M5', 30)
        >>> print(f"Fetched {len(df)} candles")
        >>> print(df.tail())
    """
    # Initialize loader with DuckDB storage enabled
    print(f"Initializing OANDA loader with DuckDB storage...")
    loader = OANDADataLoader(store_to_duckdb=True)

    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)

    # Check for existing data if resuming
    if resume:
        try:
            store = DuckDBStore()
            last_timestamp = store.get_last_timestamp(symbol, timeframe)
            if last_timestamp is not None and last_timestamp > start_time:
                start_time = last_timestamp.to_pydatetime()
                print(f"✓ Resuming from last stored timestamp: {last_timestamp}")
                print(f"  Skipping already fetched data...")
        except Exception as e:
            logger.debug(f"Could not check for existing data: {e}")
            print("  Starting fresh download...")

    print(f"\nFetching {symbol} {timeframe} data:")
    print(f"  From: {start_time}")
    print(f"  To:   {end_time}")
    print(f"  Total days: {days_back}")
    print(f"  Window size: {window_days} days")
    print("")

    # Fetch data in windows to handle large requests
    all_data = []
    total_candles = 0
    window = timedelta(days=window_days)
    current_start = start_time
    window_num = 0

    while current_start < end_time:
        current_end = min(current_start + window, end_time)
        window_num += 1

        print(f"Window {window_num}: {current_start:%Y-%m-%d %H:%M} -> {current_end:%Y-%m-%d %H:%M}")

        try:
            # Fetch data for this window
            df = loader.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=current_start,
                end=current_end
            )

            if not df.empty:
                candles_fetched = len(df)
                total_candles += candles_fetched
                all_data.append(df)
                print(f"  ✓ Fetched {candles_fetched} candles")
            else:
                print(f"  - No data available for this window")

        except Exception as e:
            print(f"  ✗ Error fetching window: {e}")
            logger.error(f"Failed to fetch window {window_num}: {e}")

        # Move to next window
        current_start = current_end

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, axis=0)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)

        print(f"\n✅ Fetch complete!")
        print(f"  Total candles fetched: {total_candles}")
        print(f"  Data stored in: data/market_data.duckdb")
        print(f"  Date range: {combined_df.index[0]} to {combined_df.index[-1]}")

        return combined_df
    else:
        print("\n⚠️ No data was fetched")
        return pd.DataFrame()


def show_data_summary(symbol: str, timeframe: str):
    """Show summary of data stored in DuckDB for a symbol/timeframe.

    Args:
        symbol: OANDA instrument symbol
        timeframe: OANDA granularity

    Example:
        >>> show_data_summary('EUR_USD', 'M5')
        Data Summary for EUR_USD M5:
          Total candles: 8640
          Date range: 2024-01-01 to 2024-01-30
          Missing periods: 0
    """
    try:
        store = DuckDBStore()

        # Get data from DuckDB
        query = f"""
        SELECT COUNT(*) as count,
               MIN(timestamp) as start_date,
               MAX(timestamp) as end_date
        FROM market_data
        WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        """

        result = store.conn.execute(query).fetchone()

        if result and result[0] > 0:
            print(f"\nData Summary for {symbol} {timeframe}:")
            print(f"  Total candles: {result[0]:,}")
            print(f"  Date range: {result[1]} to {result[2]}")

            # Check for gaps
            df = store.get_data(symbol, timeframe)
            if not df.empty:
                expected_freq = pd.infer_freq(df.index)
                if expected_freq:
                    full_range = pd.date_range(
                        start=df.index[0],
                        end=df.index[-1],
                        freq=expected_freq
                    )
                    missing = len(full_range) - len(df)
                    print(f"  Missing periods: {missing}")

                    if missing > 0:
                        print(f"  Data completeness: {100 * len(df) / len(full_range):.1f}%")
        else:
            print(f"\n⚠️ No data found for {symbol} {timeframe}")

    except Exception as e:
        print(f"\n✗ Error reading from DuckDB: {e}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    # Parse arguments
    symbol = sys.argv[1]
    timeframe = sys.argv[2]
    days_back = int(sys.argv[3])

    # Optional resume parameter
    resume = True
    if len(sys.argv) >= 5:
        resume = sys.argv[4].lower() not in ("0", "false", "no")

    try:
        # Fetch the data
        df = fetch_oanda_bulk(
            symbol=symbol,
            timeframe=timeframe,
            days_back=days_back,
            resume=resume
        )

        # Show summary of stored data
        if not df.empty:
            show_data_summary(symbol, timeframe)

            # Show sample of latest data
            print(f"\nLatest 5 candles:")
            print(df.tail())

    except Exception as e:
        print(f"\n✗ Error: {e}")
        logger.error(f"Bulk fetch failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()