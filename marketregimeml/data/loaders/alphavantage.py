"""Alpha Vantage data loader implementation."""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import requests

from marketregimeml.data.loaders.base import MarketDataLoader
from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class AlphaVantageLoader(MarketDataLoader):
    """Data loader for Alpha Vantage stock market data.

    Uses the Alpha Vantage API to fetch historical stock data.
    Supports multiple timeframes from 1-minute to monthly.

    Note: Free tier is limited to 5 API calls per minute and 500 per day.
    """

    # Supported timeframes and their API function mappings
    TIMEFRAME_MAPPING = {
        "1min": ("TIME_SERIES_INTRADAY", "1min"),
        "5min": ("TIME_SERIES_INTRADAY", "5min"),
        "15min": ("TIME_SERIES_INTRADAY", "15min"),
        "30min": ("TIME_SERIES_INTRADAY", "30min"),
        "60min": ("TIME_SERIES_INTRADAY", "60min"),
        "1h": ("TIME_SERIES_INTRADAY", "60min"),
        "daily": ("TIME_SERIES_DAILY", None),
        "1d": ("TIME_SERIES_DAILY", None),
        "weekly": ("TIME_SERIES_WEEKLY", None),
        "1w": ("TIME_SERIES_WEEKLY", None),
        "monthly": ("TIME_SERIES_MONTHLY", None),
        "1m": ("TIME_SERIES_MONTHLY", None),
    }

    def __init__(self, cache_enabled: bool = True):
        """Initialize Alpha Vantage data loader.

        Args:
            cache_enabled: Whether to use caching for data requests

        Raises:
            ValueError: If ALPHAVANTAGE_API_KEY is not set
        """
        # Get API key from environment
        self.api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY environment variable not set")

        # API settings
        self.base_url = "https://www.alphavantage.co/query"

        # Rate limiting (5 calls per minute for free tier)
        self.rate_limit_calls = 5
        self.rate_limit_period = 60  # seconds
        self.call_times = []

        # Additional rate limit attributes for compatibility
        self.min_request_interval = 12.5  # 60 seconds / 5 calls = 12 seconds minimum
        self.daily_call_limit = 500  # Free tier limit

        # Initialize parent class
        super().__init__(cache_enabled)

        logger.info("Alpha Vantage loader initialized")

    def _validate_config(self) -> None:
        """Validate Alpha Vantage configuration."""
        # Simple validation - just check API key exists
        if not self.api_key or len(self.api_key) < 10:
            raise ValueError("Invalid API key format")
        logger.debug("Alpha Vantage configuration validated")

    def _apply_rate_limiting(self):
        """Apply rate limiting to API requests."""
        current_time = time.time()

        # Remove old call times
        self.call_times = [
            t for t in self.call_times if current_time - t < self.rate_limit_period
        ]

        # If we've made too many calls, wait
        if len(self.call_times) >= self.rate_limit_calls:
            wait_time = self.rate_limit_period - (current_time - self.call_times[0])
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                # Clear old times after waiting
                self.call_times = []

        # Record this call
        self.call_times.append(time.time())

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Alpha Vantage format.

        Args:
            timeframe: Timeframe string (e.g., '1d', '5min')

        Returns:
            Converted timeframe for internal use
        """
        # Return the timeframe as-is if it's in our mapping
        if timeframe.lower() in self.TIMEFRAME_MAPPING:
            return timeframe.lower()

        # Handle common alternatives
        alternatives = {
            "1day": "1d",
            "day": "1d",
            "d": "1d",
            "1week": "1w",
            "week": "1w",
            "w": "1w",
            "1month": "1m",
            "month": "1m",
            "m": "1m",
            "1hour": "1h",
            "hour": "1h",
            "h": "1h",
        }

        converted = alternatives.get(timeframe.lower(), timeframe.lower())

        if converted not in self.TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        return converted

    def _parse_response(self, data: dict, timeframe: str) -> pd.DataFrame:
        """Parse Alpha Vantage response into DataFrame.

        Args:
            data: JSON response from Alpha Vantage
            timeframe: Original timeframe requested

        Returns:
            DataFrame with OHLCV data
        """
        # Find the time series key in the response
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key or "series" in key.lower():
                time_series_key = key
                break

        if not time_series_key:
            raise ValueError("No time series data found in response")

        time_series = data[time_series_key]

        # Parse into DataFrame
        records = []
        for timestamp, values in time_series.items():
            record = {
                "timestamp": pd.to_datetime(timestamp),
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": float(values.get("5. volume", values.get("6. volume", 0))),
            }

            # Add adjusted close if available
            if "5. adjusted close" in values:
                record["adjusted_close"] = float(values["5. adjusted close"])
            elif "6. adjusted close" in values:
                record["adjusted_close"] = float(values["6. adjusted close"])

            records.append(record)

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        return df

    def _fetch_ohlcv_impl(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime],
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Alpha Vantage.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timeframe: Timeframe (e.g., '1d', '5min')
            start_date: Start date for data fetch
            end_date: End date for data fetch
            limit: Maximum number of data points

        Returns:
            DataFrame with OHLCV data
        """
        # Apply rate limiting
        self._apply_rate_limiting()

        # Convert timeframe
        tf_converted = self._convert_timeframe(timeframe)
        function, interval = self.TIMEFRAME_MAPPING[tf_converted]

        # Build request parameters
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full" if not limit else "compact",
        }

        # Add interval for intraday data
        if interval:
            params["interval"] = interval

        try:
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for errors
            if "Error Message" in data:
                raise ConnectionError(
                    f"Alpha Vantage API error: {data['Error Message']}"
                )
            elif "Note" in data:
                raise ConnectionError(
                    f"Alpha Vantage API limit reached: {data['Note']}"
                )

            # Parse response
            df = self._parse_response(data, timeframe)

            # Filter by date range
            if start_date:
                start_date = pd.to_datetime(start_date)
                # For intraday, we need to be more flexible with date comparison
                if interval:  # Intraday data
                    # Compare just the date part for intraday
                    df = df[df.index.date >= start_date.date()]
                else:
                    df = df[df.index >= start_date]

            if end_date:
                end_date = pd.to_datetime(end_date)
                if interval:  # Intraday data
                    # Compare just the date part for intraday
                    df = df[df.index.date <= end_date.date()]
                else:
                    df = df[df.index <= end_date]

            # Apply limit if specified
            if limit and len(df) > limit:
                df = df.tail(limit)

            # Standardize the data
            df = self.standardize_ohlcv(df)

            # Validate the data
            self.validate_ohlcv(df)

            logger.info(f"Fetched {len(df)} records for {symbol} {timeframe}")
            return df

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Alpha Vantage API error: {e}")
        except Exception as e:
            raise ConnectionError(f"Error fetching data: {e}")

    def fetch_multiple(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of stock symbols
            timeframe: Timeframe for all symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            limit: Maximum number of data points per symbol

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}

        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, timeframe, start_date, end_date, limit)
                result[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                result[symbol] = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )

        return result

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols.

        Note: Alpha Vantage doesn't provide a symbol listing API,
        so we return a list of common symbols.

        Returns:
            List of common stock symbols
        """
        # Return common US stock symbols
        common_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "FB",
            "TSLA",
            "NVDA",
            "JPM",
            "JNJ",
            "V",
            "PG",
            "UNH",
            "HD",
            "MA",
            "DIS",
            "PYPL",
            "BAC",
            "ADBE",
            "NFLX",
            "CMCSA",
            "PFE",
            "INTC",
            "VZ",
            "T",
            "KO",
        ]

        return common_symbols

    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes.

        Returns:
            List of supported timeframe strings
        """
        return list(self.TIMEFRAME_MAPPING.keys())

    def fetch_streaming(self, symbol: str, timeframe: str) -> None:
        """Fetch streaming data (not implemented).

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for streaming

        Raises:
            NotImplementedError: Alpha Vantage doesn't support streaming
        """
        raise NotImplementedError("Alpha Vantage does not support streaming data")
