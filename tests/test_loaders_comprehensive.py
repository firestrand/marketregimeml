"""Comprehensive tests for data loaders to improve coverage."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import responses

from marketregimeml.data.loaders.base import MarketDataLoader
from marketregimeml.data.loaders.oanda import OANDADataLoader
from marketregimeml.data.loaders.alphavantage import AlphaVantageLoader


class TestMarketDataLoaderBase:
    """Test base market data loader functionality."""

    class ConcreteLoader(MarketDataLoader):
        """Concrete implementation for testing."""

        def fetch_ohlcv(self, symbol, timeframe, start_date, end_date=None):
            """Mock implementation."""
            dates = pd.date_range(
                start=start_date, end=end_date or datetime.now(), freq="D"
            )
            n = len(dates)

            data = pd.DataFrame(
                {
                    "open": 100 + np.random.randn(n),
                    "high": 101 + np.random.randn(n),
                    "low": 99 + np.random.randn(n),
                    "close": 100 + np.random.randn(n),
                    "volume": np.random.randint(1000, 10000, n),
                },
                index=dates,
            )

            # Ensure OHLC relationships
            data["high"] = data[["open", "high", "close"]].max(axis=1)
            data["low"] = data[["open", "low", "close"]].min(axis=1)

            return data

    @pytest.fixture
    def loader(self):
        """Create concrete loader instance."""
        return self.ConcreteLoader()

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "open": 100 + np.random.randn(100) * 2,
                "high": 102 + np.random.randn(100) * 2,
                "low": 98 + np.random.randn(100) * 2,
                "close": 100 + np.random.randn(100) * 2,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        # Fix OHLC relationships
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)

        return data

    def test_validate_ohlcv_valid(self, loader, sample_ohlcv):
        """Test OHLCV validation with valid data."""
        # Should not raise any exception
        loader.validate_ohlcv(sample_ohlcv)

    def test_validate_ohlcv_invalid_relationships(self, loader):
        """Test validation with invalid OHLC relationships."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")

        # High < Low (invalid)
        invalid_data = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [95] * 10,  # Less than low
                "low": [98] * 10,
                "close": [100] * 10,
                "volume": [1000] * 10,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="high.*low"):
            loader.validate_ohlcv(invalid_data)

    def test_validate_ohlcv_negative_prices(self, loader):
        """Test validation with negative prices."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")

        invalid_data = pd.DataFrame(
            {
                "open": [100, -5, 100, 100, 100, 100, 100, 100, 100, 100],
                "high": [102] * 10,
                "low": [98] * 10,
                "close": [100] * 10,
                "volume": [1000] * 10,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="negative"):
            loader.validate_ohlcv(invalid_data)

    def test_validate_ohlcv_missing_columns(self, loader):
        """Test validation with missing columns."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")

        # Missing 'close' column
        invalid_data = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [102] * 10,
                "low": [98] * 10,
                "volume": [1000] * 10,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="columns"):
            loader.validate_ohlcv(invalid_data)

    def test_standardize_ohlcv(self, loader, sample_ohlcv):
        """Test OHLCV standardization."""
        # Add extra columns
        sample_ohlcv["extra"] = 1.0

        standardized = loader.standardize_ohlcv(sample_ohlcv)

        # Should have exactly OHLCV columns
        assert list(standardized.columns) == [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert isinstance(standardized.index, pd.DatetimeIndex)

    def test_resample_ohlcv(self, loader, sample_ohlcv):
        """Test OHLCV resampling."""
        # Resample to weekly
        weekly = loader.resample_ohlcv(sample_ohlcv, "W")

        assert len(weekly) < len(sample_ohlcv)
        # Check aggregation rules
        assert weekly["high"].max() <= sample_ohlcv["high"].max()
        assert weekly["low"].min() >= sample_ohlcv["low"].min()
        assert (
            weekly["volume"].sum() <= sample_ohlcv["volume"].sum() * 1.01
        )  # Allow small float error

    def test_fetch_multiple(self, loader):
        """Test fetching multiple symbols."""
        symbols = ["EUR_USD", "GBP_USD"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        data = loader.fetch_multiple(symbols, "1D", start_date, end_date)

        assert isinstance(data, dict)
        assert len(data) == 2
        assert "EUR_USD" in data
        assert "GBP_USD" in data

        for symbol, df in data.items():
            assert isinstance(df, pd.DataFrame)
            assert "close" in df.columns
