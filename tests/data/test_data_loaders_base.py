"""Tests for the base data loader class."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from marketregimeml.data.loaders.base import MarketDataLoader


class MockDataLoader(MarketDataLoader):
    """Mock implementation of MarketDataLoader for testing."""

    def _validate_config(self) -> None:
        """Mock validation."""
        pass

    def _fetch_ohlcv_impl(
        self, symbol, timeframe, start_date, end_date=None, limit=None
    ) -> pd.DataFrame:
        """Mock implementation of _fetch_ohlcv_impl."""
        # Generate mock data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        data = {
            "open": np.random.uniform(100, 110, 100),
            "high": np.random.uniform(110, 120, 100),
            "low": np.random.uniform(90, 100, 100),
            "close": np.random.uniform(95, 115, 100),
            "volume": np.random.uniform(1000, 10000, 100),
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = "timestamp"

        # Ensure OHLC relationships
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        return df

    def fetch_ohlcv(
        self, symbol, timeframe, start_date, end_date=None, limit=None
    ) -> pd.DataFrame:
        """Mock fetch_ohlcv - delegates to _fetch_ohlcv_impl."""
        return self._fetch_ohlcv_impl(symbol, timeframe, start_date, end_date, limit)

    def fetch_multiple(
        self, symbols, timeframe, start_date, end_date=None, limit=None
    ) -> dict:
        """Mock fetch_multiple."""
        return {
            symbol: self.fetch_ohlcv(
                symbol, timeframe, start_date, end_date, limit
            )
            for symbol in symbols
        }

    def get_available_symbols(self) -> list:
        """Mock get_available_symbols."""
        return ["EUR_USD", "GBP_USD", "SPY", "QQQ"]

    def get_available_timeframes(self) -> list:
        """Mock get_available_timeframes."""
        return ["1min", "5min", "1h", "1d"]


class TestMarketDataLoader:
    """Test suite for MarketDataLoader."""

    @pytest.fixture
    def loader(self):
        """Create a mock data loader instance."""
        return MockDataLoader()

    @pytest.fixture
    def valid_ohlcv_df(self):
        """Create a valid OHLCV DataFrame."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="1h")
        data = {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "volume": [
                1000,
                1100,
                1200,
                1300,
                1400,
                1500,
                1600,
                1700,
                1800,
                1900,
            ],
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = "timestamp"
        return df

    def test_validate_symbol(self, loader):
        """Test symbol validation."""
        assert loader.validate_symbol("EUR_USD") is True
        assert loader.validate_symbol("INVALID") is False

    def test_validate_timeframe(self, loader):
        """Test timeframe validation."""
        assert loader.validate_timeframe("1h") is True
        assert loader.validate_timeframe("invalid") is False

    def test_standardize_ohlcv(self, loader):
        """Test OHLCV standardization."""
        # Create DataFrame with non-standard column names
        dates = pd.date_range(start="2023-01-01", periods=5, freq="1h")
        data = {
            "Open": [100, 101, 102, 103, 104],
            "High": [102, 103, 104, 105, 106],
            "Low": [99, 100, 101, 102, 103],
            "Close": [101, 102, 103, 104, 105],
            "Vol": [1000, 1100, 1200, 1300, 1400],
        }
        df = pd.DataFrame(data, index=dates)

        standardized = loader.standardize_ohlcv(df)

        # Check column names
        assert list(standardized.columns) == [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        # Check index name
        assert standardized.index.name == "timestamp"

        # Check data types
        for col in standardized.columns:
            assert standardized[col].dtype == "float64"

    def test_standardize_ohlcv_missing_volume(self, loader):
        """Test standardization when volume is missing."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="1h")
        data = {
            "open": [100, 101, 102, 103, 104],
            "high": [102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
        }
        df = pd.DataFrame(data, index=dates)

        standardized = loader.standardize_ohlcv(df)

        # Check that volume was added with zeros
        assert "volume" in standardized.columns
        assert (standardized["volume"] == 0).all()

    def test_validate_ohlcv_valid_data(self, loader, valid_ohlcv_df):
        """Test validation with valid data."""
        assert loader.validate_ohlcv(valid_ohlcv_df) is True

    def test_validate_ohlcv_empty_dataframe(self, loader):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            loader.validate_ohlcv(df)

    def test_validate_ohlcv_missing_columns(self, loader):
        """Test validation with missing columns."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="1h")
        data = {
            "open": [100, 101, 102, 103, 104],
            "close": [101, 102, 103, 104, 105],
        }
        df = pd.DataFrame(data, index=dates)

        with pytest.raises(ValueError, match="Missing columns"):
            loader.validate_ohlcv(df)

    def test_validate_ohlcv_invalid_high_low(self, loader):
        """Test validation with high < low."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="1h")
        data = {
            "open": [100, 101, 102, 103, 104],
            "high": [98, 103, 104, 105, 106],  # First high is less than low
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
        df = pd.DataFrame(data, index=dates)

        with pytest.raises(ValueError, match="High < Low"):
            loader.validate_ohlcv(df)

    def test_validate_ohlcv_nan_values(self, loader):
        """Test validation with NaN values."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="1h")
        data = {
            "open": [100, np.nan, 102, 103, 104],
            "high": [102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
        df = pd.DataFrame(data, index=dates)

        with pytest.raises(ValueError, match="NaN values found"):
            loader.validate_ohlcv(df)

    def test_validate_ohlcv_negative_prices(self, loader):
        """Test validation with negative prices."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="1h")
        data = {
            "open": [100, -101, 102, 103, 104],
            "high": [102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
        df = pd.DataFrame(data, index=dates)

        with pytest.raises(ValueError, match="Negative prices found"):
            loader.validate_ohlcv(df)

    def test_resample_ohlcv(self, loader, valid_ohlcv_df):
        """Test OHLCV resampling."""
        # Resample from hourly to 4-hourly
        resampled = loader.resample_ohlcv(valid_ohlcv_df, "4h")

        # Check that we have fewer rows
        assert len(resampled) < len(valid_ohlcv_df)

        # Check that OHLC relationships are preserved
        assert (resampled["high"] >= resampled["low"]).all()
        assert (resampled["high"] >= resampled["open"]).all()
        assert (resampled["high"] >= resampled["close"]).all()
        assert (resampled["low"] <= resampled["open"]).all()
        assert (resampled["low"] <= resampled["close"]).all()

        # Check that volume is summed
        assert resampled["volume"].sum() == pytest.approx(
            valid_ohlcv_df["volume"].sum()
        )

    def test_fetch_ohlcv(self, loader):
        """Test fetch_ohlcv method."""
        df = loader.fetch_ohlcv("EUR_USD", "1h", "2023-01-01", "2023-01-31")

        # Check that DataFrame is not empty
        assert not df.empty

        # Check that it has the right columns
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

        # Check that it passes validation
        assert loader.validate_ohlcv(df) is True

    def test_fetch_multiple(self, loader):
        """Test fetch_multiple method."""
        symbols = ["EUR_USD", "GBP_USD"]
        data = loader.fetch_multiple(symbols, "1h", "2023-01-01", "2023-01-31")

        # Check that we got data for all symbols
        assert set(data.keys()) == set(symbols)

        # Check that each DataFrame is valid
        for symbol, df in data.items():
            assert loader.validate_ohlcv(df) is True
