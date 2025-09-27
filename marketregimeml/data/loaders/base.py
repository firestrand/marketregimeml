"""Abstract base class for market data loaders."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd

from marketregimeml.utils.validation import OHLCVValidator
from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class MarketDataLoader(ABC):
    """Abstract base class for market data loaders.

    All data loaders should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, cache_enabled: bool = True):
        """Initialize the data loader.

        Args:
            cache_enabled: Whether to use caching for data requests
        """
        self.cache_enabled = cache_enabled
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate data source configuration.

        Should check for required API keys, credentials, etc.

        Raises:
            ValueError: If configuration is invalid
        """

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol.

        Args:
            symbol: Symbol to fetch (e.g., 'EUR_USD', 'SPY')
            timeframe: Timeframe/granularity (e.g., '1h', 'D', '5min')
            start_date: Start date for data fetch
            end_date: End date for data fetch (None for latest)
            limit: Maximum number of records to fetch

        Returns:
            DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
            Index should be DatetimeIndex named 'timestamp'

        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If API connection fails
        """
        # Template method pattern - common preprocessing
        start_date = self._parse_date(start_date)
        end_date = self._parse_date(end_date) if end_date else None

        # Validate parameters
        self._validate_fetch_params(symbol, timeframe, start_date, end_date, limit)

        # Call implementation-specific method
        df = self._fetch_ohlcv_impl(symbol, timeframe, start_date, end_date, limit)

        # Common postprocessing
        if not self.validate_ohlcv(df):
            logger.warning("Data validation issues detected")
        df = self.standardize_ohlcv(df)

        return df

    @abstractmethod
    def _fetch_ohlcv_impl(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime],
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Implementation-specific OHLCV fetch.

        Subclasses implement this with their specific API calls.
        """

    def _parse_date(self, date_input: Union[str, datetime]) -> datetime:
        """Parse date input to datetime."""
        if isinstance(date_input, str):
            return pd.to_datetime(date_input)
        return date_input

    def _validate_fetch_params(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime],
        limit: Optional[int],
    ) -> None:
        """Validate fetch parameters."""
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if not timeframe:
            raise ValueError("Timeframe cannot be empty")
        if end_date and start_date > end_date:
            raise ValueError("Start date must be before end date")
        if limit and limit <= 0:
            raise ValueError("Limit must be positive")

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
            symbols: List of symbols to fetch
            timeframe: Timeframe/granularity
            start_date: Start date for data fetch
            end_date: End date for data fetch
            limit: Maximum number of records per symbol

        Returns:
            Dictionary mapping symbol to DataFrame

        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If API connection fails
        """
        if not symbols:
            raise ValueError("Symbols list cannot be empty")

        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.fetch_ohlcv(
                    symbol, timeframe, start_date, end_date, limit
                )
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue

        return result

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from data source.

        Returns:
            List of available symbol strings
        """

    @abstractmethod
    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes from data source.

        Returns:
            List of available timeframe strings
        """

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is available.

        Args:
            symbol: Symbol to validate

        Returns:
            True if symbol is valid
        """
        available = self.get_available_symbols()
        return symbol in available

    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is available.

        Args:
            timeframe: Timeframe to validate

        Returns:
            True if timeframe is valid
        """
        available = self.get_available_timeframes()
        return timeframe in available

    def standardize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize OHLCV DataFrame format.

        Ensures consistent column names and index format across all data sources.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Standardized DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
            and DatetimeIndex named 'timestamp'
        """
        # Use shared standardization utility
        df = OHLCVValidator.standardize(df)

        # Apply base-specific naming convention
        df.index.name = "timestamp"

        return df

    def validate_ohlcv(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data quality.

        Args:
            df: OHLCV DataFrame to validate

        Returns:
            True if data passes validation

        Raises:
            ValueError: If validation fails with details
        """
        # Use shared validation utility
        return OHLCVValidator.validate(df)

    def resample_ohlcv(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to different timeframe.

        Args:
            df: OHLCV DataFrame
            target_timeframe: Target timeframe (pandas freq string)

        Returns:
            Resampled OHLCV DataFrame
        """
        resampled = df.resample(target_timeframe).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Remove NaN rows (from gaps in data)
        resampled = resampled.dropna()

        return resampled
