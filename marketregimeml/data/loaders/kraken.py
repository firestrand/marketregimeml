"""Kraken cryptocurrency data loader implementation."""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

from marketregimeml.data.loaders.base import MarketDataLoader
from marketregimeml.utils.logging import get_logger
from marketregimeml.utils.validation import OHLCVValidator

logger = get_logger(__name__)


class KrakenDataLoader(MarketDataLoader):
    """
    Data loader for Kraken cryptocurrency exchange.

    Uses the public REST API (no authentication required for market data).

    API Documentation: https://docs.kraken.com/rest/
    """

    BASE_URL = "https://api.kraken.com"

    # Kraken trading pairs mapping
    PAIR_MAP = {
        "BTC/USD": "XBTUSD",
        "BTCUSD": "XBTUSD",
        "BTC_USD": "XBTUSD",
        "ETH/USD": "ETHUSD",
        "ETHUSD": "ETHUSD",
        "ETH_USD": "ETHUSD",
        "XRP/USD": "XRPUSD",
        "XRPUSD": "XRPUSD",
        "XRP_USD": "XRPUSD",
        "SOL/USD": "SOLUSD",
        "SOLUSD": "SOLUSD",
        "SOL_USD": "SOLUSD",
    }

    # Timeframe mapping (Kraken uses minutes)
    TIMEFRAME_MAP = {
        "M1": 1,  # 1 minute
        "M5": 5,  # 5 minutes
        "M15": 15,  # 15 minutes
        "M30": 30,  # 30 minutes
        "H1": 60,  # 1 hour
        "H4": 240,  # 4 hours
        "D": 1440,  # 1 day (24 hours)
        "W": 10080,  # 1 week
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
    }

    # Maximum data points per request
    MAX_LIMIT = 720

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize Kraken data loader.

        Args:
            cache_enabled: Whether to enable caching
        """
        self.base_url = self.BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MarketRegimeML/1.0"})

        super().__init__(cache_enabled=cache_enabled)
        logger.info("Initialized Kraken data loader")

    def _validate_config(self) -> None:
        """Validate Kraken configuration.

        No API key required for public market data.
        """
        # Kraken public API doesn't require authentication
        pass

    def _normalize_pair(self, symbol: str) -> str:
        """
        Normalize symbol to Kraken pair format.

        Args:
            symbol: Symbol in various formats

        Returns:
            Normalized Kraken pair
        """
        symbol_upper = symbol.upper()

        # Check if it's in our mapping
        if symbol_upper in self.PAIR_MAP:
            return self.PAIR_MAP[symbol_upper]

        # Try to handle other formats
        # Remove common separators
        cleaned = (
            symbol_upper.replace("/", "").replace("_", "").replace("-", "")
        )

        # Common crypto to Kraken format
        if cleaned.startswith("BTC"):
            return "XBTUSD"
        elif cleaned.startswith("ETH"):
            return "ETHUSD"
        elif cleaned.startswith("XRP"):
            return "XRPUSD"
        elif cleaned.startswith("SOL"):
            return "SOLUSD"
        else:
            # Return as-is and let Kraken API handle it
            return symbol_upper

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "H1",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol from Kraken.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USD', 'ETH/USD')
            timeframe: Candle timeframe
            start: Start datetime (UTC)
            end: End datetime (UTC)
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        # Normalize inputs
        pair = self._normalize_pair(symbol)

        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        interval = self.TIMEFRAME_MAP[timeframe]

        # Prepare parameters
        params = {"pair": pair, "interval": interval}

        # Convert datetime to Unix timestamp
        if start:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            params["since"] = int(start.timestamp())

        # Make API request
        endpoint = f"{self.base_url}/0/public/OHLC"

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()

            data = response.json()

            # Check for errors
            if data.get("error"):
                raise Exception(f"Kraken API error: {data['error']}")

            # Extract OHLC data
            result = data.get("result", {})

            # Find the actual pair key (Kraken may return a slightly different key)
            ohlc_data = None
            for key in result:
                if key != "last":
                    ohlc_data = result[key]
                    break

            if not ohlc_data:
                logger.warning(f"No data returned for {pair}")
                return pd.DataFrame()

            # Convert to DataFrame
            # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
            df = pd.DataFrame(
                ohlc_data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vwap",
                    "volume",
                    "count",
                ],
            )

            # Convert timestamp to datetime index
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="s", utc=True
            )
            df.set_index("timestamp", inplace=True)

            # Convert price/volume columns to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            # Keep only OHLCV columns
            df = df[["open", "high", "low", "close", "volume"]]

            # Apply limit if specified
            if limit and len(df) > limit:
                df = df.tail(limit)

            # Validate data using OHLCVValidator
            if not OHLCVValidator.validate(df):
                logger.warning(f"Data validation issues for {symbol}")

            logger.info(f"Fetched {len(df)} {timeframe} candles for {symbol}")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Kraken data: {e}")
            raise

    def fetch_multiple(
        self,
        symbols: List[str],
        timeframe: str = "H1",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading symbols
            timeframe: Candle timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum number of candles per symbol

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}

        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, timeframe, start, end, limit)
                results[symbol] = df

                # Respect rate limits
                # Kraken has a call rate limit based on a counter
                time.sleep(1)  # Conservative delay

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        return results

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading pairs from Kraken.

        Returns:
            List of available symbols
        """
        endpoint = f"{self.base_url}/0/public/AssetPairs"

        try:
            response = self.session.get(endpoint)
            response.raise_for_status()

            data = response.json()

            if data.get("error"):
                raise Exception(f"Kraken API error: {data['error']}")

            pairs = list(data.get("result", {}).keys())
            return sorted(pairs)

        except Exception as e:
            logger.error(f"Failed to fetch asset pairs: {e}")
            return list(self.PAIR_MAP.values())

    def get_server_time(self) -> datetime:
        """
        Get Kraken server time.

        Returns:
            Server time as datetime
        """
        endpoint = f"{self.base_url}/0/public/Time"

        try:
            response = self.session.get(endpoint)
            response.raise_for_status()

            data = response.json()

            if data.get("error"):
                raise Exception(f"Kraken API error: {data['error']}")

            unix_time = data["result"]["unixtime"]
            return datetime.fromtimestamp(unix_time, tz=timezone.utc)

        except Exception as e:
            logger.error(f"Failed to get server time: {e}")
            return datetime.now(timezone.utc)

    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes.

        Returns:
            List of available timeframe strings
        """
        return list(self.TIMEFRAME_MAP.keys())

    def _fetch_ohlcv_impl(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime],
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Implementation of OHLCV fetch for Kraken.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            limit: Max records

        Returns:
            DataFrame with OHLCV data
        """
        # Delegate to existing fetch_ohlcv method
        return self.fetch_ohlcv(symbol, timeframe, start_date, end_date, limit)

    def _validate_fetch_params(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime],
        limit: Optional[int],
    ) -> None:
        """Validate fetch parameters.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            limit: Max records

        Raises:
            ValueError: If parameters are invalid
        """
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. Valid options: {list(self.TIMEFRAME_MAP.keys())}"
            )

        if limit and limit > self.MAX_LIMIT:
            logger.warning(
                f"Limit {limit} exceeds max {self.MAX_LIMIT}, will be capped"
            )
