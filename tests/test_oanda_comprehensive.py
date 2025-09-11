"""Comprehensive tests for OANDA data loader."""

import pytest
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from marketregimeml.data.loaders.oanda import OANDADataLoader


class TestOANDAComprehensive:
    """Comprehensive test suite for OANDA data loader."""

    @pytest.fixture
    def mock_oanda_response(self):
        """Mock OANDA candles response."""
        mock_candle = Mock()
        mock_candle.time = "2023-01-03T10:00:00.000000000Z"
        mock_candle.bid = Mock(o=1.0500, h=1.0550, l=1.0480, c=1.0520)
        mock_candle.ask = Mock(o=1.0502, h=1.0552, l=1.0482, c=1.0522)
        mock_candle.mid = Mock(o=1.0501, h=1.0551, l=1.0481, c=1.0521)
        mock_candle.volume = 1000
        mock_candle.complete = True

        mock_candle2 = Mock()
        mock_candle2.time = "2023-01-03T10:05:00.000000000Z"
        mock_candle2.bid = Mock(o=1.0520, h=1.0570, l=1.0510, c=1.0540)
        mock_candle2.ask = Mock(o=1.0522, h=1.0572, l=1.0512, c=1.0542)
        mock_candle2.mid = Mock(o=1.0521, h=1.0571, l=1.0511, c=1.0541)
        mock_candle2.volume = 1200
        mock_candle2.complete = True

        mock_response = Mock()
        mock_response.candles = [mock_candle, mock_candle2]

        return mock_response

    @pytest.fixture
    def mock_loader(self):
        """Create mock OANDA loader."""
        with patch.dict(
            os.environ,
            {
                "OANDA_API_KEY": "test_key",
                "OANDA_ACCOUNT_ID": "test_account",
                "OANDA_ENVIRONMENT": "practice",
            },
        ):
            with patch("marketregimeml.data.loaders.oanda.v20.Context"):
                loader = OANDADataLoader(cache_enabled=True)
                return loader

    def test_init_with_credentials(self):
        """Test initialization with credentials."""
        with patch.dict(
            os.environ,
            {
                "OANDA_API_KEY": "test_key",
                "OANDA_ACCOUNT_ID": "test_account",
                "OANDA_ENVIRONMENT": "practice",
            },
        ):
            with patch(
                "marketregimeml.data.loaders.oanda.v20.Context"
            ) as mock_context:
                loader = OANDADataLoader()

                assert loader.api_key == "test_key"
                assert loader.account_id == "test_account"
                assert loader.environment == "practice"
                assert loader.hostname == "api-fxpractice.oanda.com"
                mock_context.assert_called_once()

    def test_init_live_environment(self):
        """Test initialization with live environment."""
        with patch.dict(
            os.environ,
            {
                "OANDA_API_KEY": "test_key",
                "OANDA_ACCOUNT_ID": "test_account",
                "OANDA_ENVIRONMENT": "live",
            },
        ):
            with patch("marketregimeml.data.loaders.oanda.v20.Context"):
                loader = OANDADataLoader()

                assert loader.environment == "live"
                assert loader.hostname == "api-fxtrade.oanda.com"

    def test_init_missing_api_key(self):
        """Test initialization raises error without API key."""
        with patch.dict(
            os.environ, {"OANDA_ACCOUNT_ID": "test_account"}, clear=True
        ):
            with pytest.raises(
                ValueError, match="OANDA_API_KEY environment variable not set"
            ):
                OANDADataLoader()

    def test_init_missing_account_id(self):
        """Test initialization raises error without account ID."""
        with patch.dict(os.environ, {"OANDA_API_KEY": "test_key"}, clear=True):
            with pytest.raises(
                ValueError,
                match="OANDA_ACCOUNT_ID environment variable not set",
            ):
                OANDADataLoader()

    def test_init_default_environment(self):
        """Test initialization with default environment."""
        with patch.dict(
            os.environ,
            {"OANDA_API_KEY": "test_key", "OANDA_ACCOUNT_ID": "test_account"},
        ):
            with patch("marketregimeml.data.loaders.oanda.v20.Context"):
                loader = OANDADataLoader()

                assert loader.environment == "practice"

    def test_validate_config(self, mock_loader):
        """Test configuration validation."""
        # Mock successful account info response
        mock_response = Mock()
        mock_response.account = Mock(id="test_account")
        mock_loader.api.account.get = Mock(return_value=mock_response)

        # Should not raise error
        mock_loader._validate_config()

        mock_loader.api.account.get.assert_called_once_with(
            mock_loader.account_id
        )

    def test_validate_config_error(self, mock_loader):
        """Test configuration validation with error."""
        mock_loader.api.account.get = Mock(side_effect=Exception("API Error"))

        with pytest.raises(ValueError, match="Invalid OANDA configuration"):
            mock_loader._validate_config()

    def test_validate_symbol(self, mock_loader):
        """Test symbol validation."""
        # Valid symbols
        assert mock_loader._validate_symbol("EUR_USD") is True
        assert mock_loader._validate_symbol("GBP_JPY") is True
        assert mock_loader._validate_symbol("AUD_CAD") is True

        # Invalid symbols
        assert mock_loader._validate_symbol("") is False
        assert mock_loader._validate_symbol("EURUSD") is False  # No underscore
        assert (
            mock_loader._validate_symbol("EUR/USD") is False
        )  # Wrong separator
        assert (
            mock_loader._validate_symbol("EUR_USD_GBP") is False
        )  # Too many parts

    def test_validate_timeframe(self, mock_loader):
        """Test timeframe validation."""
        # Valid timeframes
        valid_timeframes = ["S5", "M1", "M5", "H1", "H4", "D", "W", "M"]
        for tf in valid_timeframes:
            assert mock_loader._validate_timeframe(tf) is True

        # Mapped timeframes
        mapped_timeframes = ["1min", "5min", "1h", "4h", "1d"]
        for tf in mapped_timeframes:
            assert mock_loader._validate_timeframe(tf) is True

        # Invalid timeframes
        assert mock_loader._validate_timeframe("invalid") is False
        assert mock_loader._validate_timeframe("M3") is False

    def test_normalize_timeframe(self, mock_loader):
        """Test timeframe normalization."""
        # Test mapping
        assert mock_loader._normalize_timeframe("1min") == "M1"
        assert mock_loader._normalize_timeframe("5min") == "M5"
        assert mock_loader._normalize_timeframe("1h") == "H1"
        assert mock_loader._normalize_timeframe("1d") == "D"

        # Test direct timeframes
        assert mock_loader._normalize_timeframe("M15") == "M15"
        assert mock_loader._normalize_timeframe("H4") == "H4"

    def test_rate_limiting(self, mock_loader):
        """Test rate limiting functionality."""
        # Test initial state
        assert mock_loader._should_rate_limit() is False

        # Simulate recent request
        mock_loader.last_request_time = time.time()
        assert mock_loader._should_rate_limit() is True

    def test_parse_candles_response(self, mock_loader, mock_oanda_response):
        """Test parsing of candles response."""
        df = mock_loader._parse_candles_response(
            mock_oanda_response, "EUR_USD", "mid"
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
        assert df["symbol"].iloc[0] == "EUR_USD"
        assert df["close"].iloc[0] == 1.0521  # First candle mid close
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_parse_candles_response_bid(
        self, mock_loader, mock_oanda_response
    ):
        """Test parsing with bid prices."""
        df = mock_loader._parse_candles_response(
            mock_oanda_response, "EUR_USD", "bid"
        )

        assert df["close"].iloc[0] == 1.0520  # First candle bid close

    def test_parse_candles_response_ask(
        self, mock_loader, mock_oanda_response
    ):
        """Test parsing with ask prices."""
        df = mock_loader._parse_candles_response(
            mock_oanda_response, "EUR_USD", "ask"
        )

        assert df["close"].iloc[0] == 1.0522  # First candle ask close

    def test_parse_candles_response_incomplete(
        self, mock_loader, mock_oanda_response
    ):
        """Test parsing with incomplete candles filtered out."""
        # Make first candle incomplete
        mock_oanda_response.candles[0].complete = False

        df = mock_loader._parse_candles_response(
            mock_oanda_response, "EUR_USD", "mid"
        )

        # Should only have 1 candle (the complete one)
        assert len(df) == 1
        assert df["close"].iloc[0] == 1.0541  # Second candle close

    def test_make_api_request(self, mock_loader):
        """Test making API request."""
        mock_response = Mock()
        mock_loader.api.instrument.candles = Mock(return_value=mock_response)

        response = mock_loader._make_api_request("EUR_USD", "M5", count=100)

        assert response == mock_response
        mock_loader.api.instrument.candles.assert_called_once()

    def test_make_api_request_with_rate_limit(self, mock_loader):
        """Test API request with rate limiting."""
        mock_loader.last_request_time = time.time()  # Simulate recent request
        mock_response = Mock()
        mock_loader.api.instrument.candles = Mock(return_value=mock_response)

        with patch("time.sleep") as mock_sleep:
            response = mock_loader._make_api_request(
                "EUR_USD", "M5", count=100
            )

        # Should have slept due to rate limit
        mock_sleep.assert_called()
        assert response == mock_response

    def test_fetch_data(self, mock_loader, mock_oanda_response):
        """Test fetching data."""
        mock_loader.api.instrument.candles = Mock(
            return_value=mock_oanda_response
        )

        with patch("time.sleep"):  # Mock sleep
            df = mock_loader.fetch_data("EUR_USD", "M5")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "close" in df.columns
        assert "symbol" in df.columns
        mock_loader.api.instrument.candles.assert_called_once()

    def test_fetch_data_with_date_range(
        self, mock_loader, mock_oanda_response
    ):
        """Test fetching data with date range."""
        mock_loader.api.instrument.candles = Mock(
            return_value=mock_oanda_response
        )

        start_date = "2023-01-01"
        end_date = "2023-01-31"

        with patch("time.sleep"):
            df = mock_loader.fetch_data(
                "EUR_USD", "M5", start_date=start_date, end_date=end_date
            )

        assert isinstance(df, pd.DataFrame)
        mock_loader.api.instrument.candles.assert_called_once()

    def test_fetch_data_with_count(self, mock_loader, mock_oanda_response):
        """Test fetching data with count parameter."""
        mock_loader.api.instrument.candles = Mock(
            return_value=mock_oanda_response
        )

        with patch("time.sleep"):
            df = mock_loader.fetch_data("EUR_USD", "M5", count=500)

        assert isinstance(df, pd.DataFrame)
        # Check that count was passed to API call
        call_args = mock_loader.api.instrument.candles.call_args[1]
        assert call_args["count"] == 500

    def test_get_available_symbols(self, mock_loader):
        """Test getting available symbols."""
        mock_response = Mock()
        mock_instrument = Mock()
        mock_instrument.name = "EUR_USD"
        mock_instrument.type = "CURRENCY"
        mock_response.instruments = [mock_instrument]

        mock_loader.api.account.instruments = Mock(return_value=mock_response)

        symbols = mock_loader.get_available_symbols()

        assert isinstance(symbols, list)
        assert "EUR_USD" in symbols
        mock_loader.api.account.instruments.assert_called_once_with(
            mock_loader.account_id
        )

    def test_get_available_symbols_error(self, mock_loader):
        """Test get available symbols with API error."""
        mock_loader.api.account.instruments = Mock(
            side_effect=Exception("API Error")
        )

        symbols = mock_loader.get_available_symbols()

        # Should return common symbols as fallback
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "EUR_USD" in symbols

    def test_error_handling_invalid_symbol(self, mock_loader):
        """Test error handling for invalid symbol."""
        with pytest.raises(ValueError, match="Invalid symbol"):
            mock_loader.fetch_data("INVALID", "M5")

    def test_error_handling_invalid_timeframe(self, mock_loader):
        """Test error handling for invalid timeframe."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            mock_loader.fetch_data("EUR_USD", "invalid")

    def test_api_error_handling(self, mock_loader):
        """Test API error handling."""
        mock_loader.api.instrument.candles = Mock(
            side_effect=Exception("API Error")
        )

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Failed to fetch data"):
                mock_loader.fetch_data("EUR_USD", "M5")

    def test_empty_response_handling(self, mock_loader):
        """Test handling of empty API response."""
        empty_response = Mock()
        empty_response.candles = []
        mock_loader.api.instrument.candles = Mock(return_value=empty_response)

        with patch("time.sleep"):
            df = mock_loader.fetch_data("EUR_USD", "M5")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_format_datetime_for_api(self, mock_loader):
        """Test datetime formatting for API."""
        # Test string date
        result = mock_loader._format_datetime_for_api("2023-01-01")
        assert result == "2023-01-01T00:00:00.000000000Z"

        # Test datetime object
        dt = datetime(2023, 1, 1, 12, 30, 45)
        result = mock_loader._format_datetime_for_api(dt)
        assert result == "2023-01-01T12:30:45.000000000Z"

        # Test pandas timestamp
        ts = pd.Timestamp("2023-01-01 15:30:00")
        result = mock_loader._format_datetime_for_api(ts)
        assert result == "2023-01-01T15:30:00.000000000Z"

    def test_format_datetime_for_api_none(self, mock_loader):
        """Test datetime formatting with None."""
        result = mock_loader._format_datetime_for_api(None)
        assert result is None

    def test_multiple_symbols_fetch(self, mock_loader, mock_oanda_response):
        """Test fetching multiple symbols."""
        mock_loader.api.instrument.candles = Mock(
            return_value=mock_oanda_response
        )

        with patch("time.sleep"):
            results = mock_loader.fetch_multiple(["EUR_USD", "GBP_JPY"], "M5")

        assert isinstance(results, dict)
        assert "EUR_USD" in results
        assert "GBP_JPY" in results
        assert mock_loader.api.instrument.candles.call_count == 2

    def test_pricing_type_parameter(self, mock_loader, mock_oanda_response):
        """Test different pricing types."""
        mock_loader.api.instrument.candles = Mock(
            return_value=mock_oanda_response
        )

        pricing_types = ["mid", "bid", "ask"]

        for pricing in pricing_types:
            with patch("time.sleep"):
                df = mock_loader.fetch_data("EUR_USD", "M5", pricing=pricing)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2

    def test_cache_functionality(self, mock_loader, mock_oanda_response):
        """Test caching functionality."""
        mock_loader.api.instrument.candles = Mock(
            return_value=mock_oanda_response
        )

        with patch("time.sleep"):
            # First call should hit API
            df1 = mock_loader.fetch_data("EUR_USD", "M5")

            # Second call should hit cache if implemented
            df2 = mock_loader.fetch_data("EUR_USD", "M5")

        # Basic functionality test
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)

    def test_large_date_range_chunking(self, mock_loader, mock_oanda_response):
        """Test handling of large date ranges with chunking."""
        mock_loader.api.instrument.candles = Mock(
            return_value=mock_oanda_response
        )

        # Request large date range
        start_date = "2022-01-01"
        end_date = "2023-12-31"

        with patch("time.sleep"):
            df = mock_loader.fetch_data(
                "EUR_USD", "M5", start_date=start_date, end_date=end_date
            )

        assert isinstance(df, pd.DataFrame)
        # Should make multiple API calls for large ranges if chunking implemented
        # At minimum should make one call
        assert mock_loader.api.instrument.candles.call_count >= 1

    def test_connection_error_retry(self, mock_loader):
        """Test retry logic for connection errors."""
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.candles = []

        mock_loader.api.instrument.candles = Mock(
            side_effect=[Exception("Connection error"), mock_response]
        )

        with patch("time.sleep"):
            with patch.object(
                mock_loader, "_retry_on_failure", return_value=True
            ):
                df = mock_loader.fetch_data("EUR_USD", "M5")

        assert isinstance(df, pd.DataFrame)
        # Should have retried once
        assert mock_loader.api.instrument.candles.call_count == 2

    def test_max_candles_limit(self, mock_loader, mock_oanda_response):
        """Test handling of OANDA's max candles limit."""
        mock_loader.api.instrument.candles = Mock(
            return_value=mock_oanda_response
        )

        # Request more than typical limit
        with patch("time.sleep"):
            df = mock_loader.fetch_data("EUR_USD", "M5", count=10000)

        # Should handle the limit appropriately
        assert isinstance(df, pd.DataFrame)

        # Check that count was adjusted or handled properly
        call_args = mock_loader.api.instrument.candles.call_args[1]
        assert "count" in call_args
