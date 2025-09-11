"""Comprehensive tests for AlphaVantage data loader."""

import pytest
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from marketregimeml.data.loaders.alphavantage import AlphaVantageLoader


class TestAlphaVantageComprehensive:
    """Comprehensive test suite for AlphaVantage data loader."""

    @pytest.fixture
    def mock_av_response_daily(self):
        """Mock Alpha Vantage daily response."""
        return {
            "Time Series (Daily)": {
                "2023-01-03": {
                    "1. open": "100.00",
                    "2. high": "105.00",
                    "3. low": "99.00",
                    "4. close": "103.00",
                    "5. volume": "1000000",
                },
                "2023-01-02": {
                    "1. open": "98.00",
                    "2. high": "102.00",
                    "3. low": "97.00",
                    "4. close": "100.50",
                    "5. volume": "950000",
                },
            },
            "Meta Data": {
                "1. Information": "Daily Prices",
                "2. Symbol": "AAPL",
                "3. Last Refreshed": "2023-01-03",
                "4. Output Size": "Compact",
                "5. Time Zone": "US/Eastern",
            },
        }

    @pytest.fixture
    def mock_av_response_intraday(self):
        """Mock Alpha Vantage intraday response."""
        return {
            "Time Series (5min)": {
                "2023-01-03 16:00:00": {
                    "1. open": "103.00",
                    "2. high": "104.00",
                    "3. low": "102.50",
                    "4. close": "103.50",
                    "5. volume": "50000",
                },
                "2023-01-03 15:55:00": {
                    "1. open": "102.00",
                    "2. high": "103.20",
                    "3. low": "101.80",
                    "4. close": "103.00",
                    "5. volume": "45000",
                },
            },
            "Meta Data": {
                "1. Information": "Intraday (5min) open, high, low, close prices and volume",
                "2. Symbol": "AAPL",
                "3. Last Refreshed": "2023-01-03 16:00:00",
                "4. Interval": "5min",
                "5. Output Size": "Compact",
                "6. Time Zone": "US/Eastern",
            },
        }

    @pytest.fixture
    def mock_loader(self):
        """Create mock AlphaVantage loader."""
        with patch.dict(
            os.environ, {"ALPHAVANTAGE_API_KEY": "test_api_key_12345"}
        ):
            loader = AlphaVantageLoader(cache_enabled=True)
            return loader

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch.dict(
            os.environ, {"ALPHAVANTAGE_API_KEY": "test_api_key_12345"}
        ):
            loader = AlphaVantageLoader()

            assert loader.api_key == "test_api_key_12345"
            assert loader.min_request_interval == 12.5
            assert loader.daily_call_limit == 500

    def test_init_without_api_key(self):
        """Test initialization raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match="ALPHAVANTAGE_API_KEY environment variable not set",
            ):
                AlphaVantageLoader()

    def test_init_cache_disabled(self):
        """Test initialization with cache disabled."""
        with patch.dict(
            os.environ, {"ALPHAVANTAGE_API_KEY": "test_api_key_12345"}
        ):
            loader = AlphaVantageLoader(cache_enabled=False)

            assert loader.cache is None

    def test_validate_config(self, mock_loader):
        """Test configuration validation."""
        # Should not raise error with valid API key
        mock_loader._validate_config()

        # Test with invalid API key format
        mock_loader.api_key = "short"
        with pytest.raises(ValueError, match="Invalid API key format"):
            mock_loader._validate_config()

    def test_validate_symbol(self, mock_loader):
        """Test symbol validation."""
        # Valid symbols
        assert mock_loader._validate_symbol("AAPL") is True
        assert mock_loader._validate_symbol("MSFT") is True

        # Invalid symbols
        assert mock_loader._validate_symbol("") is False
        assert mock_loader._validate_symbol("TOOLONGSYMBOL") is False
        assert mock_loader._validate_symbol("123") is False
        assert mock_loader._validate_symbol("AA-PL") is False

    def test_validate_timeframe(self, mock_loader):
        """Test timeframe validation."""
        # Valid timeframes
        valid_timeframes = [
            "1min",
            "5min",
            "15min",
            "30min",
            "60min",
            "daily",
            "weekly",
            "monthly",
        ]
        for tf in valid_timeframes:
            assert mock_loader._validate_timeframe(tf) is True

        # Mapped timeframes
        mapped_timeframes = [
            "1m",
            "5m",
            "15m",
            "30m",
            "1h",
            "60m",
            "1d",
            "1w",
            "1mo",
        ]
        for tf in mapped_timeframes:
            assert mock_loader._validate_timeframe(tf) is True

        # Invalid timeframes
        assert mock_loader._validate_timeframe("invalid") is False
        assert mock_loader._validate_timeframe("2min") is False

    def test_normalize_timeframe(self, mock_loader):
        """Test timeframe normalization."""
        # Test mapping
        assert mock_loader._normalize_timeframe("1m") == "1min"
        assert mock_loader._normalize_timeframe("1h") == "60min"
        assert mock_loader._normalize_timeframe("1d") == "daily"

        # Test direct timeframes
        assert mock_loader._normalize_timeframe("5min") == "5min"
        assert mock_loader._normalize_timeframe("weekly") == "weekly"

    def test_rate_limiting(self, mock_loader):
        """Test rate limiting functionality."""
        # Test initial state
        assert mock_loader._should_rate_limit() is False

        # Simulate recent request
        mock_loader.last_request_time = time.time()
        assert mock_loader._should_rate_limit() is True

        # Test daily limit
        mock_loader.daily_call_count = 500
        mock_loader.last_request_time = time.time() - 20  # 20 seconds ago
        assert mock_loader._should_rate_limit() is True

    def test_reset_daily_counter(self, mock_loader):
        """Test daily counter reset."""
        # Set up previous day data
        mock_loader.daily_call_count = 100
        mock_loader.last_reset_date = datetime.now().date() - timedelta(days=1)

        mock_loader._reset_daily_counter_if_needed()

        assert mock_loader.daily_call_count == 0
        assert mock_loader.last_reset_date == datetime.now().date()

    def test_get_cache_key(self, mock_loader):
        """Test cache key generation."""
        key = mock_loader._get_cache_key(
            "AAPL", "daily", "2023-01-01", "2023-12-31"
        )

        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hash length

        # Same parameters should produce same key
        key2 = mock_loader._get_cache_key(
            "AAPL", "daily", "2023-01-01", "2023-12-31"
        )
        assert key == key2

        # Different parameters should produce different key
        key3 = mock_loader._get_cache_key(
            "MSFT", "daily", "2023-01-01", "2023-12-31"
        )
        assert key != key3

    def test_get_from_cache(self, mock_loader):
        """Test cache retrieval."""
        # Empty cache
        result = mock_loader._get_from_cache("test_key")
        assert result is None

        # Add item to cache
        test_data = pd.DataFrame({"close": [100, 101, 102]})
        mock_loader.cache["test_key"] = {
            "data": test_data,
            "timestamp": time.time(),
        }

        # Retrieve from cache
        result = mock_loader._get_from_cache("test_key")
        pd.testing.assert_frame_equal(result, test_data)

        # Test expired cache
        mock_loader.cache["test_key"]["timestamp"] = (
            time.time() - 1000
        )  # Old timestamp
        result = mock_loader._get_from_cache("test_key")
        assert result is None

    def test_store_in_cache(self, mock_loader):
        """Test cache storage."""
        test_data = pd.DataFrame({"close": [100, 101, 102]})
        mock_loader._store_in_cache("test_key", test_data)

        assert "test_key" in mock_loader.cache
        cached_item = mock_loader.cache["test_key"]
        assert "data" in cached_item
        assert "timestamp" in cached_item
        pd.testing.assert_frame_equal(cached_item["data"], test_data)

    def test_parse_daily_response(self, mock_loader, mock_av_response_daily):
        """Test parsing of daily response."""
        df = mock_loader._parse_daily_response(mock_av_response_daily, "AAPL")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        ]
        assert df["symbol"].iloc[0] == "AAPL"
        assert df["close"].iloc[0] == 100.50  # Should be sorted by date
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_parse_intraday_response(
        self, mock_loader, mock_av_response_intraday
    ):
        """Test parsing of intraday response."""
        df = mock_loader._parse_intraday_response(
            mock_av_response_intraday, "AAPL"
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        ]
        assert df["symbol"].iloc[0] == "AAPL"
        assert isinstance(df.index, pd.DatetimeIndex)

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_make_api_request(self, mock_sleep, mock_loader):
        """Test API request making."""
        mock_loader.av.get_daily = Mock(return_value=("data", "metadata"))

        # Test successful request
        data, metadata = mock_loader._make_api_request(
            mock_loader.av.get_daily, symbol="AAPL", outputsize="compact"
        )

        assert data == "data"
        assert metadata == "metadata"
        assert mock_loader.daily_call_count == 1
        mock_loader.av.get_daily.assert_called_once_with(
            symbol="AAPL", outputsize="compact"
        )

    def test_make_api_request_with_rate_limit(self, mock_loader):
        """Test API request with rate limiting."""
        mock_loader.last_request_time = time.time()  # Simulate recent request
        mock_loader.av.get_daily = Mock(return_value=("data", "metadata"))

        with patch("time.sleep") as mock_sleep:
            data, metadata = mock_loader._make_api_request(
                mock_loader.av.get_daily, symbol="AAPL"
            )

        # Should have slept due to rate limit
        mock_sleep.assert_called()

    def test_make_api_request_daily_limit(self, mock_loader):
        """Test API request with daily limit reached."""
        mock_loader.daily_call_count = 500  # At limit

        with pytest.raises(RuntimeError, match="Daily API call limit reached"):
            mock_loader._make_api_request(
                mock_loader.av.get_daily, symbol="AAPL"
            )

    def test_fetch_daily_data(self, mock_loader, mock_av_response_daily):
        """Test fetching daily data."""
        mock_loader.av.get_daily = Mock(
            return_value=(mock_av_response_daily, {})
        )

        with patch("time.sleep"):  # Mock sleep
            df = mock_loader.fetch_daily_data("AAPL")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "close" in df.columns
        assert "symbol" in df.columns
        mock_loader.av.get_daily.assert_called_once()

    def test_fetch_intraday_data(self, mock_loader, mock_av_response_intraday):
        """Test fetching intraday data."""
        mock_loader.av.get_intraday = Mock(
            return_value=(mock_av_response_intraday, {})
        )

        with patch("time.sleep"):  # Mock sleep
            df = mock_loader.fetch_intraday_data("AAPL", "5min")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "close" in df.columns
        assert "symbol" in df.columns
        mock_loader.av.get_intraday.assert_called_once()

    def test_fetch_data_with_cache(self, mock_loader, mock_av_response_daily):
        """Test fetch data with caching."""
        mock_loader.av.get_daily = Mock(
            return_value=(mock_av_response_daily, {})
        )

        with patch("time.sleep"):
            # First call should hit API
            df1 = mock_loader.fetch_daily_data("AAPL")

            # Second call should hit cache
            df2 = mock_loader.fetch_daily_data("AAPL")

        # Should only call API once
        assert mock_loader.av.get_daily.call_count == 1
        pd.testing.assert_frame_equal(df1, df2)

    def test_get_available_symbols(self, mock_loader):
        """Test getting available symbols."""
        symbols = mock_loader.get_available_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_error_handling_invalid_symbol(self, mock_loader):
        """Test error handling for invalid symbol."""
        with pytest.raises(ValueError, match="Invalid symbol"):
            mock_loader.fetch_daily_data("INVALID123")

    def test_error_handling_invalid_timeframe(self, mock_loader):
        """Test error handling for invalid timeframe."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            mock_loader.fetch_intraday_data("AAPL", "invalid")

    def test_api_error_handling(self, mock_loader):
        """Test API error handling."""
        mock_loader.av.get_daily = Mock(side_effect=Exception("API Error"))

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Failed to fetch data"):
                mock_loader.fetch_daily_data("AAPL")

    def test_empty_response_handling(self, mock_loader):
        """Test handling of empty API response."""
        empty_response = {"Time Series (Daily)": {}}
        mock_loader.av.get_daily = Mock(return_value=(empty_response, {}))

        with patch("time.sleep"):
            df = mock_loader.fetch_daily_data("AAPL")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_malformed_response_handling(self, mock_loader):
        """Test handling of malformed API response."""
        malformed_response = {"invalid": "response"}
        mock_loader.av.get_daily = Mock(return_value=(malformed_response, {}))

        with patch("time.sleep"):
            with pytest.raises(ValueError, match="Unexpected response format"):
                mock_loader.fetch_daily_data("AAPL")

    def test_date_filtering(self, mock_loader, mock_av_response_daily):
        """Test date filtering functionality."""
        mock_loader.av.get_daily = Mock(
            return_value=(mock_av_response_daily, {})
        )

        with patch("time.sleep"):
            df = mock_loader.fetch_daily_data(
                "AAPL", start_date="2023-01-03", end_date="2023-01-03"
            )

        assert len(df) == 1
        assert df.index[0].date() == pd.Timestamp("2023-01-03").date()

    def test_multiple_symbols_fetch(self, mock_loader, mock_av_response_daily):
        """Test fetching multiple symbols."""
        mock_loader.av.get_daily = Mock(
            return_value=(mock_av_response_daily, {})
        )

        with patch("time.sleep"):
            results = mock_loader.fetch_multiple(["AAPL", "MSFT"], "daily")

        assert isinstance(results, dict)
        assert "AAPL" in results
        assert "MSFT" in results
        assert mock_loader.av.get_daily.call_count == 2

    def test_cache_disabled_flow(self):
        """Test data flow with cache disabled."""
        with patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "test_key"}):
            loader = AlphaVantageLoader(cache_enabled=False)

            # Cache operations should be no-ops
            assert loader._get_from_cache("test") is None
            loader._store_in_cache("test", pd.DataFrame())  # Should not error

    def test_cleanup_old_cache_entries(self, mock_loader):
        """Test cleanup of old cache entries."""
        # Add old and new entries
        mock_loader.cache["old"] = {
            "data": pd.DataFrame(),
            "timestamp": time.time() - 1000,
        }  # Old
        mock_loader.cache["new"] = {
            "data": pd.DataFrame(),
            "timestamp": time.time(),
        }  # Recent

        mock_loader._cleanup_cache()

        assert "old" not in mock_loader.cache
        assert "new" in mock_loader.cache
