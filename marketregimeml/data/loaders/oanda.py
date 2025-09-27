"""OANDA data loader implementation."""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
try:  # pragma: no cover - optional dependency
    import v20
    _HAS_V20 = True
except Exception:
    v20 = None  # type: ignore
    _HAS_V20 = False

from marketregimeml.data.loaders.base import MarketDataLoader
from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class OANDADataLoader(MarketDataLoader):
    """Data loader for OANDA forex and CFD data.

    Uses the OANDA v20 API to fetch historical price data.
    Supports multiple timeframes and instruments.
    """

    # OANDA granularity options
    TIMEFRAMES = {
        "S5": "5 seconds",
        "S10": "10 seconds",
        "S15": "15 seconds",
        "S30": "30 seconds",
        "M1": "1 minute",
        "M2": "2 minutes",
        "M4": "4 minutes",
        "M5": "5 minutes",
        "M10": "10 minutes",
        "M15": "15 minutes",
        "M30": "30 minutes",
        "H1": "1 hour",
        "H2": "2 hours",
        "H3": "3 hours",
        "H4": "4 hours",
        "H6": "6 hours",
        "H8": "8 hours",
        "H12": "12 hours",
        "D": "Daily",
        "W": "Weekly",
        "M": "Monthly",
    }

    # Mapping from common timeframe formats to OANDA format
    TIMEFRAME_MAPPING = {
        "1min": "M1",
        "5min": "M5",
        "15min": "M15",
        "30min": "M30",
        "1h": "H1",
        "4h": "H4",
        "1d": "D",
        "1w": "W",
        "1m": "M",
    }

    def __init__(
        self,
        cache_enabled: bool = True,
        store_to_duckdb: bool = False,
        duckdb_path: Optional[str] = None,
    ):
        """Initialize OANDA data loader.

        Args:
            cache_enabled: Whether to use caching for data requests

        Raises:
            ValueError: If required environment variables are not set
        """
        # Get credentials from environment BEFORE calling super().__init__
        self.api_key = os.getenv("OANDA_API_KEY")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.environment = os.getenv("OANDA_ENVIRONMENT", "practice")

        if not self.api_key:
            raise ValueError("OANDA_API_KEY environment variable not set")
        if not self.account_id:
            raise ValueError("OANDA_ACCOUNT_ID environment variable not set")

        # Set hostname based on environment
        if self.environment == "live":
            self.hostname = "api-fxtrade.oanda.com"
        else:
            self.hostname = "api-fxpractice.oanda.com"

        if not _HAS_V20:
            raise ImportError("v20 (OANDA API) not installed. Install with: pip install v20")

        # Initialize v20 context
        self.api = v20.Context(
            hostname=self.hostname,
            port=443,
            ssl=True,
            token=self.api_key,
            datetime_format="RFC3339",
        )

        # Rate limiting settings
        self.rate_limiter = None  # Can be implemented if needed
        self.last_request_time = 0
        self.min_request_interval = 0.01  # 100 requests per second max

        # Retry settings (exponential backoff with jitter)
        self.max_retries = 5
        self.retry_delay = 0.5

        # Optional DuckDB storage (lazy import to keep optional dependency)
        self._store = None
        if store_to_duckdb:
            try:
                from marketregimeml.data.storage.duckdb_store import DuckDBStore  # type: ignore

                self._store = DuckDBStore(duckdb_path or "data/market_data.duckdb")
            except Exception as e:  # pragma: no cover - optional dep
                logger.warning(f"DuckDB storage not available: {e}")

        # Now call super().__init__ which will call _validate_config
        super().__init__(cache_enabled)

        logger.info(
            f"OANDA loader initialized for {self.environment} environment"
        )

    def _validate_config(self) -> None:
        """Validate OANDA configuration.

        Checks API connectivity and account access.
        """
        # Skip validation in test environment or if API is mocked
        if hasattr(self.api, "__class__") and "Mock" in str(
            self.api.__class__
        ):
            logger.debug("Skipping validation for mocked API")
            return

        try:
            # Test API connection by fetching account info
            self.api.account.get(self.account_id)
            logger.info(f"OANDA account validated: {self.account_id}")
        except Exception as e:
            logger.warning(f"Could not validate OANDA configuration: {e}")
            # Don't fail initialization, just warn
            pass

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from OANDA.

        Returns:
            List of available instrument names
        """
        try:
            response = self.api.account.instruments(self.account_id)

            # Handle both real response and mock response
            if hasattr(response, "body"):
                instruments = response.body.get("instruments", [])
            else:
                instruments = []

            symbols = [inst["name"] for inst in instruments if "name" in inst]
            logger.debug(f"Found {len(symbols)} available instruments")
            return symbols
        except Exception as e:
            logger.error(f"Failed to fetch available symbols: {e}")
            return []

    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes.

        Returns:
            List of OANDA granularity values
        """
        return list(self.TIMEFRAMES.keys())

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert common timeframe format to OANDA granularity.

        Args:
            timeframe: Timeframe string (e.g., '1h', 'H1')

        Returns:
            OANDA granularity string
        """
        # If already in OANDA format, return as-is
        if timeframe in self.TIMEFRAMES:
            return timeframe

        # Try to map from common format
        return self.TIMEFRAME_MAPPING.get(timeframe, timeframe)

    def _apply_rate_limiting(self):
        """Apply rate limiting to API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request_with_retry(self, request_func, *args, **kwargs):
        """Make API request with retry logic.

        Args:
            request_func: Function to call for the request
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            API response

        Raises:
            ConnectionError: If all retries fail
        """
        last_error = None

        import random

        for attempt in range(self.max_retries):
            try:
                self._apply_rate_limiting()
                return request_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                # Exponential backoff with jitter
                if attempt < self.max_retries - 1:
                    backoff = self.retry_delay * (2 ** attempt)
                    backoff *= 1.0 + random.random() * 0.25
                    time.sleep(backoff)

        raise ConnectionError(
            f"OANDA API error after {self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _generate_time_windows(
        start_dt: datetime,
        end_dt: Optional[datetime],
        granularity: str,
        max_per: int = 5000,
    ):
        """Generate UTC-aware time windows for paginated fetching.

        This helper is pure (no network) and can be unit tested. It yields
        (window_start, window_end) datetimes sized to keep each request under
        typical OANDA per-request candle caps.
        """
        # Normalize to UTC-aware timestamps
        cur = pd.Timestamp(start_dt)
        if cur.tzinfo is None:
            cur = cur.tz_localize("UTC")
        else:
            cur = cur.tz_convert("UTC")

        end_bound = None
        if end_dt is not None:
            end_bound = pd.Timestamp(end_dt)
            if end_bound.tzinfo is None:
                end_bound = end_bound.tz_localize("UTC")
            else:
                end_bound = end_bound.tz_convert("UTC")

        # Determine days per page by granularity
        days_per_page = 1
        if granularity.startswith("M") and granularity not in ("M",):
            mins = int(granularity[1:])
            bars_per_day = int(24 * 60 / max(1, mins))
            # Aim well under 5000
            days_per_page = max(1, min(10, int(4000 // max(1, bars_per_day)) or 1))
        elif granularity.startswith("H"):
            hours = int(granularity[1:]) if len(granularity) > 1 else 1
            bars_per_day = int(24 / max(1, hours))
            days_per_page = 90 if bars_per_day <= 24 else 30
        elif granularity == "D":
            days_per_page = 4000
        elif granularity == "W":
            days_per_page = 4000 * 7
        elif granularity == "M":
            days_per_page = 4000 * 30

        while True:
            if end_bound is not None and cur >= end_bound:
                break
            page_end = cur + pd.Timedelta(days=days_per_page)
            if end_bound is not None and page_end > end_bound:
                page_end = end_bound
            yield (cur.to_pydatetime(), page_end.to_pydatetime())
            cur = page_end + pd.Timedelta(seconds=1)

    def _fetch_ohlcv_impl(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime],
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Implementation-specific OHLCV fetch from OANDA with pagination.

        Handles OANDA's per-request candle limit (max 5000) by paginating
        across the requested time range.

        Args:
            symbol: Instrument symbol (e.g., 'EUR_USD')
            timeframe: Timeframe/granularity (OANDA or common format)
            start_date: Start date for data fetch
            end_date: End date for data fetch
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            gran = self._convert_timeframe(timeframe)
            max_per = 5000
            remaining = limit if limit is not None else None
            # Normalize to timezone-aware UTC
            start_ts = pd.Timestamp(start_date)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")

            current_from = start_ts.to_pydatetime()

            final_to = None
            if end_date is not None:
                end_ts = pd.Timestamp(end_date)
                if end_ts.tzinfo is None:
                    end_ts = end_ts.tz_localize("UTC")
                else:
                    end_ts = end_ts.tz_convert("UTC")
                final_to = end_ts.to_pydatetime()
            all_candles: List = []

            # Use helper to page
            for win_from, win_to in self._generate_time_windows(current_from, final_to, gran, max_per):
                ts_from = pd.Timestamp(win_from)
                if ts_from.tzinfo is None:
                    ts_from = ts_from.tz_localize("UTC")
                params = {
                    "granularity": gran,
                    "from": ts_from.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                    "count": max_per,
                }
                candles = self._fetch_candles(symbol, params)
                if candles:
                    all_candles.extend(candles)
                    if remaining is not None:
                        remaining -= len(candles)
                        if remaining <= 0:
                            break
                else:
                    if final_to is None:
                        break

            if not all_candles:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return self._empty_ohlcv_dataframe()

            df = self._candles_to_dataframe(all_candles)
            if limit is not None and len(df) > limit:
                df = df.iloc[-limit:]
            return df

        except ConnectionError:
            # Re-raise connection errors for proper handling
            raise
        except Exception as e:
            logger.error(f"Error fetching OANDA data: {e}")
            return self._empty_ohlcv_dataframe()

    # Override fetch_ohlcv to persist after standardization
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        df = super().fetch_ohlcv(symbol, timeframe, start_date, end_date, limit)
        if self._store is not None and not df.empty:
            try:
                self._store.write_ohlcv(df, symbol=symbol, timeframe=timeframe)
            except Exception as e:
                logger.warning(f"Failed to persist OHLCV to DuckDB: {e}")
        return df

    def _build_request_params(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime],
        limit: Optional[int],
    ) -> Dict[str, Any]:
        """Build request parameters for OANDA API."""
        params = {
            "granularity": self._convert_timeframe(timeframe),
            "from": start_date.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        }

        if end_date:
            params["to"] = end_date.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

        if limit:
            params["count"] = min(limit, 5000)  # OANDA max is 5000

        return params

    def _fetch_candles(self, symbol: str, params: Dict[str, Any]) -> List:
        """Fetch candles from OANDA API."""
        response = self._make_request_with_retry(
            self.api.instrument.candles, symbol, **params
        )
        return self._extract_candles_from_response(response)

    def _extract_candles_from_response(self, response) -> List:
        """Extract candles from API response."""
        if not hasattr(response, "body"):
            return []

        if isinstance(response.body, dict):
            return response.body.get("candles", [])

        # v20 returns response objects
        return (
            response.body.candles if hasattr(response.body, "candles") else []
        )

    def _candles_to_dataframe(self, candles: List) -> pd.DataFrame:
        """Convert candles to DataFrame."""
        data = []

        for candle in candles:
            candle_data = self._parse_single_candle(candle)
            if candle_data:
                data.append(candle_data)

        if not data:
            return self._empty_ohlcv_dataframe()

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    def _parse_single_candle(self, candle) -> Optional[Dict[str, Any]]:
        """Parse a single candle into dictionary format."""
        if isinstance(candle, dict):
            return self._parse_dict_candle(candle)
        else:
            return self._parse_object_candle(candle)

    def _parse_dict_candle(self, candle: dict) -> Optional[Dict[str, Any]]:
        """Parse dictionary format candle."""
        if not candle.get("complete", False):
            return None

        mid = candle.get("mid", {})
        return {
            "timestamp": pd.to_datetime(candle["time"]),
            "open": float(mid.get("o", 0)),
            "high": float(mid.get("h", 0)),
            "low": float(mid.get("l", 0)),
            "close": float(mid.get("c", 0)),
            "volume": float(candle.get("volume", 0)),
        }

    def _parse_object_candle(self, candle) -> Optional[Dict[str, Any]]:
        """Parse v20 object format candle."""
        if not getattr(candle, "complete", False):
            return None

        mid = getattr(candle, "mid", None)
        if not mid:
            return None

        return {
            "timestamp": pd.to_datetime(candle.time),
            "open": float(mid.o),
            "high": float(mid.h),
            "low": float(mid.l),
            "close": float(mid.c),
            "volume": float(getattr(candle, "volume", 0)),
        }

    def _empty_ohlcv_dataframe(self) -> pd.DataFrame:
        """Create empty OHLCV DataFrame with correct columns."""
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

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
            symbols: List of instrument symbols
            timeframe: Timeframe/granularity
            start_date: Start date for data fetch
            end_date: End date for data fetch
            limit: Maximum number of candles per symbol

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}

        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(
                    symbol, timeframe, start_date, end_date, limit
                )
                result[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                result[symbol] = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )

        return result

    def fetch_streaming(self, symbol: str, timeframe: str) -> None:
        """Fetch streaming data (not implemented).

        Args:
            symbol: Instrument symbol
            timeframe: Timeframe for streaming

        Raises:
            NotImplementedError: Streaming not yet implemented
        """
        raise NotImplementedError("Streaming data not yet implemented")
