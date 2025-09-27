"""Binance cryptocurrency data loader implementation."""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

from marketregimeml.data.loaders.base import MarketDataLoader
from marketregimeml.utils.logging import get_logger
from marketregimeml.utils.validation import OHLCVValidator

logger = get_logger(__name__)


class BinanceDataLoader(MarketDataLoader):
    """
    Data loader for Binance cryptocurrency exchange.

    Uses the public REST API (no authentication required for market data).
    Supports spot market data only.

    API Documentation: https://binance-docs.github.io/apidocs/spot/en/
    """

    BASE_URL = "https://api.binance.com"

    # Timeframe mapping from standard to Binance format
    TIMEFRAME_MAP = {
        "M1": "1m",
        "M5": "5m",
        "M15": "15m",
        "M30": "30m",
        "H1": "1h",
        "H4": "4h",
        "D": "1d",
        "W": "1w",
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
    }

    # Maximum data points per request
    MAX_LIMIT = 1000

    def __init__(self, testnet: bool = False, cache_enabled: bool = True):
        """
        Initialize Binance data loader.

        Args:
            testnet: If True, use testnet API (for testing)
            cache_enabled: Whether to enable caching
        """
        if testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = self.BASE_URL

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MarketRegimeML/1.0"})

        super().__init__(cache_enabled=cache_enabled)
        logger.info(f"Initialized Binance data loader (testnet={testnet})")

    def _validate_config(self) -> None:
        """Validate Binance configuration.

        No API key required for public market data.
        """
        # Binance public API doesn't require authentication

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for Binance.

        Converts formats like 'BTC_USDT', 'BTC/USDT' to 'BTCUSDT'.

        Args:
            symbol: Symbol in various formats

        Returns:
            Normalized symbol for Binance API
        """
        # Remove common separators and convert to uppercase
        normalized = symbol.upper().replace("_", "").replace("/", "").replace("-", "")

        # Handle common variations
        if "USD" in normalized and not normalized.endswith("USDT"):
            # Convert USD to USDT if not already
            normalized = normalized.replace("USD", "USDT")

        return normalized

    def _get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = MAX_LIMIT,
    ) -> List:
        """
        Get kline/candlestick data from Binance.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of data points to return

        Returns:
            List of kline data
        """
        endpoint = f"{self.base_url}/api/v3/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, self.MAX_LIMIT),
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Binance data: {e}")
            raise

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "H1",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC_USDT', 'ETH_USDT')
            timeframe: Candle timeframe
            start: Start datetime (UTC)
            end: End datetime (UTC)
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        # Normalize inputs
        symbol = self._normalize_symbol(symbol)

        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        interval = self.TIMEFRAME_MAP[timeframe]

        # Convert datetime to milliseconds
        start_ms = None
        end_ms = None

        if start:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            start_ms = int(start.timestamp() * 1000)

        if end:
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            end_ms = int(end.timestamp() * 1000)

        # If no start/end specified, get last 'limit' candles
        if not start and not end:
            if not limit:
                limit = 500  # Default

            klines = self._get_klines(symbol, interval, limit=limit)

        else:
            # Fetch data in batches if needed
            all_klines = []
            current_start = start_ms

            while True:
                batch = self._get_klines(
                    symbol,
                    interval,
                    start_time=current_start,
                    end_time=end_ms,
                    limit=self.MAX_LIMIT,
                )

                if not batch:
                    break

                all_klines.extend(batch)

                # Check if we got all data
                if len(batch) < self.MAX_LIMIT:
                    break

                # Update start time for next batch
                last_time = batch[-1][0]
                current_start = last_time + 1

                # Respect rate limits
                time.sleep(0.1)

                # Apply limit if specified
                if limit and len(all_klines) >= limit:
                    all_klines = all_klines[:limit]
                    break

            klines = all_klines

        # Convert to DataFrame
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        # Convert types and create index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Convert price/volume columns to float
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]]

        # Validate data using OHLCVValidator
        if not OHLCVValidator.validate(df):
            logger.warning(f"Data validation issues for {symbol}")

        logger.info(f"Fetched {len(df)} {timeframe} candles for {symbol}")

        return df

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

                # Respect rate limits (weight=1 for klines endpoint)
                # Binance allows 1200 weight per minute
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        return results

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.

        Returns:
            List of available symbols
        """
        endpoint = f"{self.base_url}/api/v3/exchangeInfo"

        try:
            response = self.session.get(endpoint)
            response.raise_for_status()

            exchange_info = response.json()
            symbols = []

            for symbol_info in exchange_info["symbols"]:
                if symbol_info["status"] == "TRADING":
                    symbols.append(symbol_info["symbol"])

            return sorted(symbols)

        except Exception as e:
            logger.error(f"Failed to fetch exchange info: {e}")
            return []

    def get_server_time(self) -> datetime:
        """
        Get Binance server time.

        Returns:
            Server time as datetime
        """
        endpoint = f"{self.base_url}/api/v3/time"

        try:
            response = self.session.get(endpoint)
            response.raise_for_status()

            server_time_ms = response.json()["serverTime"]
            return datetime.fromtimestamp(server_time_ms / 1000, tz=timezone.utc)

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
        """Implementation of OHLCV fetch for Binance.

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
                f"Limit {limit} exceeds max {self.MAX_LIMIT}, will fetch in batches"
            )
