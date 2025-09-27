"""Test Binance data loader functionality."""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

sys.path.insert(0, "/Users/firestrand/Projects/marketregimeml")

from marketregimeml.data.loaders.binance import BinanceDataLoader


class TestBinanceDataLoader(unittest.TestCase):
    """Test cases for BinanceDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = BinanceDataLoader()

    def test_initialization(self):
        """Test loader initialization."""
        self.assertIsNotNone(self.loader)
        self.assertEqual(self.loader.base_url, "https://api.binance.com")

        # Test testnet initialization
        testnet_loader = BinanceDataLoader(testnet=True)
        self.assertEqual(testnet_loader.base_url, "https://testnet.binance.vision")

    def test_normalize_symbol(self):
        """Test symbol normalization."""
        test_cases = [
            ("BTC_USDT", "BTCUSDT"),
            ("BTC/USDT", "BTCUSDT"),
            ("btc-usdt", "BTCUSDT"),
            ("ETH_USD", "ETHUSDT"),  # USD -> USDT
            ("ETH/USD", "ETHUSDT"),
            ("ETHUSDT", "ETHUSDT"),  # Already normalized
        ]

        for input_symbol, expected in test_cases:
            result = self.loader._normalize_symbol(input_symbol)
            self.assertEqual(result, expected, f"Failed for {input_symbol}")

    def test_timeframe_mapping(self):
        """Test timeframe mapping."""
        valid_timeframes = [
            "M1",
            "M5",
            "M15",
            "H1",
            "H4",
            "D",
            "1m",
            "5m",
            "1h",
            "1d",
        ]

        for tf in valid_timeframes:
            self.assertIn(tf, self.loader.TIMEFRAME_MAP)

        # Check mappings
        self.assertEqual(self.loader.TIMEFRAME_MAP["M1"], "1m")
        self.assertEqual(self.loader.TIMEFRAME_MAP["H1"], "1h")
        self.assertEqual(self.loader.TIMEFRAME_MAP["D"], "1d")

    @patch("marketregimeml.data.loaders.binance.requests.Session")
    def test_fetch_ohlcv_mock(self, mock_session):
        """Test fetch_ohlcv with mocked API response."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [
                1640995200000,  # timestamp
                "46432.10",  # open
                "46500.00",  # high
                "46400.00",  # low
                "46450.50",  # close
                "123.456",  # volume
                1640998800000,  # close_time
                "5730000.00",  # quote_volume
                1000,  # trades
                "60.123",  # taker_buy_base
                "2790000.00",  # taker_buy_quote
                "0",  # ignore
            ],
            [
                1640998800000,
                "46450.50",
                "46550.00",
                "46420.00",
                "46500.00",
                "120.789",
                1641002400000,
                "5610000.00",
                950,
                "58.456",
                "2720000.00",
                "0",
            ],
        ]
        mock_response.raise_for_status = MagicMock()

        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance

        # Create loader and fetch data
        loader = BinanceDataLoader()
        loader.session = mock_session_instance

        df = loader.fetch_ohlcv("BTC_USDT", "H1", limit=2)

        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(
            list(df.columns), ["open", "high", "low", "close", "volume"]
        )

        # Verify data types
        for col in ["open", "high", "low", "close", "volume"]:
            self.assertEqual(df[col].dtype, np.float64)

        # Verify index is datetime
        self.assertIsInstance(df.index, pd.DatetimeIndex)

        # Verify values
        self.assertAlmostEqual(df.iloc[0]["open"], 46432.10, places=2)
        self.assertAlmostEqual(df.iloc[0]["close"], 46450.50, places=2)
        self.assertAlmostEqual(df.iloc[1]["close"], 46500.00, places=2)

    def test_fetch_ohlcv_invalid_timeframe(self):
        """Test fetch_ohlcv with invalid timeframe."""
        with self.assertRaises(ValueError) as context:
            self.loader.fetch_ohlcv("BTC_USDT", "INVALID")

        self.assertIn("Invalid timeframe", str(context.exception))

    @patch("marketregimeml.data.loaders.binance.requests.Session")
    def test_fetch_multiple(self, mock_session):
        """Test fetching multiple symbols."""
        # Mock empty response for simplicity
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance

        loader = BinanceDataLoader()
        loader.session = mock_session_instance

        symbols = ["BTC_USDT", "ETH_USDT"]
        results = loader.fetch_multiple(symbols, "H1", limit=10)

        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn("BTC_USDT", results)
        self.assertIn("ETH_USDT", results)

        for symbol in symbols:
            self.assertIsInstance(results[symbol], pd.DataFrame)

    @patch("marketregimeml.data.loaders.binance.requests.Session")
    def test_data_validation(self, mock_session):
        """Test that fetched data passes validation."""
        # Mock valid OHLCV data
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [
                1640995200000,
                "100.00",  # open
                "110.00",  # high
                "95.00",  # low
                "105.00",  # close
                "1000.00",  # volume
                1640998800000,
                "0",
                0,
                "0",
                "0",
                "0",
            ]
        ]
        mock_response.raise_for_status = MagicMock()

        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance

        loader = BinanceDataLoader()
        loader.session = mock_session_instance

        df = loader.fetch_ohlcv("TEST_USDT", "H1", limit=1)

        # Data should pass validation
        self.assertEqual(len(df), 1)

        # Verify OHLC relationships
        row = df.iloc[0]
        self.assertGreaterEqual(row["high"], row["low"])
        self.assertGreaterEqual(row["high"], row["open"])
        self.assertGreaterEqual(row["high"], row["close"])
        self.assertLessEqual(row["low"], row["open"])
        self.assertLessEqual(row["low"], row["close"])


def run_integration_test():
    """Run a real integration test with Binance API."""
    print("\n" + "=" * 60)
    print("BINANCE INTEGRATION TEST (REAL API)")
    print("=" * 60)

    try:
        loader = BinanceDataLoader()

        # Test 1: Fetch recent BTC data
        print("\n1. Fetching BTC/USDT hourly data...")
        df = loader.fetch_ohlcv("BTC_USDT", "H1", limit=24)

        if not df.empty:
            print(f"   ✅ Fetched {len(df)} candles")
            print(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")
            print(f"   24h High: ${df['high'].max():,.2f}")
            print(f"   24h Low: ${df['low'].min():,.2f}")
            print(f"   24h Volume: {df['volume'].sum():,.2f} BTC")
        else:
            print("   ❌ No data received")

        # Test 2: Fetch ETH data
        print("\n2. Fetching ETH/USDT data...")
        df_eth = loader.fetch_ohlcv(
            "ETH_USDT", "5m", limit=12
        )  # Last hour in 5m candles

        if not df_eth.empty:
            print(f"   ✅ Fetched {len(df_eth)} candles")
            print(f"   Latest ETH price: ${df_eth['close'].iloc[-1]:,.2f}")
        else:
            print("   ❌ No data received")

        # Test 3: Test symbol normalization
        print("\n3. Testing symbol formats...")
        test_symbols = ["BTC/USDT", "btc_usdt", "BTC-USDT"]
        for symbol in test_symbols:
            df_test = loader.fetch_ohlcv(symbol, "1h", limit=1)
            if not df_test.empty:
                print(f"   ✅ '{symbol}' normalized and fetched successfully")
            else:
                print(f"   ❌ Failed to fetch '{symbol}'")

        # Test 4: Multiple symbols
        print("\n4. Fetching multiple symbols...")
        symbols = ["BTC_USDT", "ETH_USDT", "BNB_USDT"]
        results = loader.fetch_multiple(symbols, "1h", limit=5)

        for symbol, data in results.items():
            if not data.empty:
                latest_price = data["close"].iloc[-1]
                print(f"   ✅ {symbol}: ${latest_price:,.2f}")
            else:
                print(f"   ❌ {symbol}: No data")

        print("\n✅ Binance integration test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Run integration test
    print("\n" + "=" * 60)
    response = input("\nRun integration test with real Binance API? (y/n): ")
    if response.lower() == "y":
        run_integration_test()
    else:
        print("Skipping integration test.")
