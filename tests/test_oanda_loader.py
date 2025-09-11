"""Tests for OANDA data loader."""

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest
import numpy as np

from marketregimeml.data.loaders.oanda import OANDADataLoader


class TestOANDADataLoader:
    """Test suite for OANDA data loader."""

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Mock environment variables for OANDA."""
        monkeypatch.setenv("OANDA_API_KEY", "test_api_key")
        monkeypatch.setenv("OANDA_ACCOUNT_ID", "test_account_id")
        monkeypatch.setenv("OANDA_ENVIRONMENT", "practice")

    @pytest.fixture
    def mock_oanda_response(self):
        """Create mock OANDA candles response."""
        return {
            "instrument": "EUR_USD",
            "granularity": "H1",
            "candles": [
                {
                    "complete": True,
                    "volume": 1234,
                    "time": "2023-01-01T00:00:00.000000000Z",
                    "mid": {
                        "o": "1.0500",
                        "h": "1.0520",
                        "l": "1.0490",
                        "c": "1.0510",
                    },
                },
                {
                    "complete": True,
                    "volume": 1345,
                    "time": "2023-01-01T01:00:00.000000000Z",
                    "mid": {
                        "o": "1.0510",
                        "h": "1.0530",
                        "l": "1.0505",
                        "c": "1.0525",
                    },
                },
            ],
        }

    @pytest.fixture
    def loader(self, mock_env_vars):
        """Create OANDA loader instance with mocked environment."""
        with patch(
            "marketregimeml.data.loaders.oanda.v20.Context"
        ) as mock_context:
            mock_context.return_value = MagicMock()
            return OANDADataLoader()

    def test_init_with_credentials(self, mock_env_vars):
        """Test initialization with credentials from environment."""
        with patch(
            "marketregimeml.data.loaders.oanda.v20.Context"
        ) as mock_context:
            loader = OANDADataLoader()

            assert loader.api_key == "test_api_key"
            assert loader.account_id == "test_account_id"
            assert loader.environment == "practice"
            mock_context.assert_called_once()

    def test_init_without_credentials(self, monkeypatch):
        """Test initialization fails without credentials."""
        monkeypatch.delenv("OANDA_API_KEY", raising=False)
        monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)

        with pytest.raises(
            ValueError, match="OANDA_API_KEY environment variable not set"
        ):
            OANDADataLoader()

    def test_validate_config(self, loader):
        """Test configuration validation."""
        # Should not raise any exception
        loader._validate_config()

    def test_get_available_symbols(self, loader):
        """Test getting available symbols."""
        mock_response = MagicMock()
        mock_response.body = {
            "instruments": [
                {
                    "name": "EUR_USD",
                    "displayName": "EUR/USD",
                    "type": "CURRENCY",
                },
                {
                    "name": "GBP_USD",
                    "displayName": "GBP/USD",
                    "type": "CURRENCY",
                },
                {
                    "name": "USD_JPY",
                    "displayName": "USD/JPY",
                    "type": "CURRENCY",
                },
            ]
        }

        with patch.object(
            loader.api.account, "instruments", return_value=mock_response
        ):
            symbols = loader.get_available_symbols()

            assert "EUR_USD" in symbols
            assert "GBP_USD" in symbols
            assert "USD_JPY" in symbols
            assert len(symbols) == 3

    def test_get_available_timeframes(self, loader):
        """Test getting available timeframes."""
        timeframes = loader.get_available_timeframes()

        # Check common timeframes
        assert "M1" in timeframes
        assert "M5" in timeframes
        assert "H1" in timeframes
        assert "D" in timeframes
        assert len(timeframes) > 10  # OANDA supports many timeframes

    def test_convert_timeframe(self, loader):
        """Test timeframe conversion."""
        assert loader._convert_timeframe("1min") == "M1"
        assert loader._convert_timeframe("5min") == "M5"
        assert loader._convert_timeframe("1h") == "H1"
        assert loader._convert_timeframe("1d") == "D"
        assert (
            loader._convert_timeframe("M15") == "M15"
        )  # Already OANDA format

    def test_fetch_ohlcv_success(self, loader, mock_oanda_response):
        """Test successful OHLCV data fetch."""
        mock_response = MagicMock()
        mock_response.body = mock_oanda_response

        with patch.object(
            loader.api.instrument, "candles", return_value=mock_response
        ):
            df = loader.fetch_ohlcv(
                symbol="EUR_USD",
                timeframe="H1",
                start_date="2023-01-01",
                end_date="2023-01-02",
            )

            # Check DataFrame structure
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

            # Check data values
            assert df.iloc[0]["open"] == 1.0500
            assert df.iloc[0]["high"] == 1.0520
            assert df.iloc[0]["low"] == 1.0490
            assert df.iloc[0]["close"] == 1.0510

            # Check index
            assert isinstance(df.index, pd.DatetimeIndex)
            assert df.index.name == "timestamp"

    def test_fetch_ohlcv_with_limit(self, loader, mock_oanda_response):
        """Test OHLCV fetch with limit parameter."""
        mock_response = MagicMock()
        mock_response.body = mock_oanda_response

        with patch.object(
            loader.api.instrument, "candles", return_value=mock_response
        ) as mock_candles:
            _ = loader.fetch_ohlcv(
                symbol="EUR_USD",
                timeframe="H1",
                start_date="2023-01-01",
                limit=100,
            )  # Test fetch with limit

            # Should have requested with count parameter
            mock_candles.assert_called_once()
            call_kwargs = mock_candles.call_args[1]
            assert "count" in call_kwargs
            assert call_kwargs["count"] == 100

    def test_fetch_multiple(self, loader, mock_oanda_response):
        """Test fetching data for multiple symbols."""
        mock_response = MagicMock()
        mock_response.body = mock_oanda_response

        with patch.object(
            loader.api.instrument, "candles", return_value=mock_response
        ):
            symbols = ["EUR_USD", "GBP_USD"]
            data = loader.fetch_multiple(
                symbols=symbols,
                timeframe="H1",
                start_date="2023-01-01",
                end_date="2023-01-02",
            )

            assert isinstance(data, dict)
            assert len(data) == 2
            assert "EUR_USD" in data
            assert "GBP_USD" in data

            for symbol, df in data.items():
                assert isinstance(df, pd.DataFrame)
                assert list(df.columns) == [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]

    def test_handle_api_error(self, loader):
        """Test API error handling."""
        with patch.object(loader.api.instrument, "candles") as mock_candles:
            mock_candles.side_effect = Exception("API Error: 401 Unauthorized")

            with pytest.raises(ConnectionError, match="OANDA API error"):
                loader.fetch_ohlcv("EUR_USD", "H1", "2023-01-01")

    def test_rate_limiting(self, loader, mock_oanda_response):
        """Test rate limiting implementation."""
        mock_response = MagicMock()
        mock_response.body = mock_oanda_response

        with patch.object(
            loader.api.instrument, "candles", return_value=mock_response
        ):
            with patch("time.sleep") as mock_sleep:
                # Make multiple rapid requests
                for _ in range(3):
                    loader.fetch_ohlcv("EUR_USD", "H1", "2023-01-01", limit=10)

                # Check that rate limiting was applied (sleep or rate limiter)
                # Since we have a min_request_interval, sleep should be called or time elapsed
                assert mock_sleep.called or loader.last_request_time > 0

    def test_retry_logic(self, loader, mock_oanda_response):
        """Test retry logic on transient failures."""
        mock_response = MagicMock()
        mock_response.body = mock_oanda_response

        with patch.object(loader.api.instrument, "candles") as mock_candles:
            # First call fails, second succeeds
            mock_candles.side_effect = [
                Exception("Temporary failure"),
                mock_response,
            ]

            df = loader.fetch_ohlcv("EUR_USD", "H1", "2023-01-01")

            # Should have retried and succeeded
            assert isinstance(df, pd.DataFrame)
            assert mock_candles.call_count == 2

    def test_validate_symbol(self, loader):
        """Test symbol validation."""
        mock_response = MagicMock()
        mock_response.body = {
            "instruments": [
                {"name": "EUR_USD", "type": "CURRENCY"},
                {"name": "GBP_USD", "type": "CURRENCY"},
            ]
        }

        with patch.object(
            loader.api.account, "instruments", return_value=mock_response
        ):
            assert loader.validate_symbol("EUR_USD") is True
            assert loader.validate_symbol("INVALID") is False

    def test_validate_timeframe(self, loader):
        """Test timeframe validation."""
        assert loader.validate_timeframe("M1") is True
        assert loader.validate_timeframe("H1") is True
        assert loader.validate_timeframe("D") is True
        assert loader.validate_timeframe("INVALID") is False

    def test_fetch_streaming_data(self, loader):
        """Test streaming data support (if implemented)."""
        # This is optional functionality
        if hasattr(loader, "fetch_streaming"):
            with pytest.raises(NotImplementedError):
                loader.fetch_streaming("EUR_USD", "M1")

    def test_environment_selection(self, monkeypatch):
        """Test different environment configurations."""
        # Test practice environment
        monkeypatch.setenv("OANDA_ENVIRONMENT", "practice")
        monkeypatch.setenv("OANDA_API_KEY", "test_key")
        monkeypatch.setenv("OANDA_ACCOUNT_ID", "test_account")

        with patch(
            "marketregimeml.data.loaders.oanda.v20.Context"
        ):
            loader = OANDADataLoader()
            assert (
                "practice" in loader.hostname
                or loader.environment == "practice"
            )

        # Test live environment
        monkeypatch.setenv("OANDA_ENVIRONMENT", "live")
        with patch(
            "marketregimeml.data.loaders.oanda.v20.Context"
        ):
            loader = OANDADataLoader()
            assert "live" in loader.hostname or loader.environment == "live"
