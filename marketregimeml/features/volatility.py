"""Volatility feature calculators.

This module provides specialized volatility measures beyond simple
standard deviation. For basic volatility, use common_features.PriceFeatures.calculate_volatility.
"""

import numpy as np
import pandas as pd

from marketregimeml.features.base import BaseFeatureCalculator
from marketregimeml.features.common_features import PriceFeatures
from marketregimeml.utils.logging import get_logger

# Optional Numba acceleration
try:  # pragma: no cover - optional dependency
    from numba import jit
except Exception:  # Fallback: define no-op decorator

    def jit(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        return wrapper


logger = get_logger(__name__)


@jit(nopython=True, cache=True)
def _yang_zhang_core(open_prices, high_prices, low_prices, close_prices, window):
    """Numba-optimized Yang-Zhang volatility calculation.

    Yang-Zhang volatility is considered one of the most accurate volatility estimators
    as it accounts for overnight gaps and intraday price movements.
    """
    n = len(close_prices)
    result = np.empty(n)
    result[:] = np.nan

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    for i in range(window - 1, n):
        # Overnight component (close-to-open)
        overnight_sum = 0.0
        for j in range(i - window + 1, i + 1):
            if j > 0:
                overnight_ret = np.log(open_prices[j] / close_prices[j - 1])
                overnight_sum += overnight_ret**2
        overnight_var = overnight_sum / window

        # Open-to-close component
        oc_sum = 0.0
        for j in range(i - window + 1, i + 1):
            oc_ret = np.log(close_prices[j] / open_prices[j])
            oc_sum += oc_ret**2
        oc_var = oc_sum / window

        # Rogers-Satchell component
        rs_sum = 0.0
        for j in range(i - window + 1, i + 1):
            rs_val = np.log(high_prices[j] / close_prices[j]) * np.log(
                high_prices[j] / open_prices[j]
            ) + np.log(low_prices[j] / close_prices[j]) * np.log(
                low_prices[j] / open_prices[j]
            )
            rs_sum += rs_val
        rs_var = rs_sum / window

        # Combine components
        result[i] = np.sqrt(overnight_var + k * oc_var + (1 - k) * rs_var) * np.sqrt(
            252
        )

    return result


@jit(nopython=True, cache=True)
def _garman_klass_core(high_prices, low_prices, close_prices, window):
    """Numba-optimized Garman-Klass volatility calculation.

    Garman-Klass volatility uses high, low, and close prices to estimate volatility
    more efficiently than close-to-close volatility.
    """
    n = len(close_prices)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        sum_val = 0.0
        for j in range(i - window + 1, i + 1):
            hl_ratio = np.log(high_prices[j] / low_prices[j])
            cc_ratio = np.log(close_prices[j] / close_prices[j - 1]) if j > 0 else 0
            sum_val += 0.5 * hl_ratio**2 - (2 * np.log(2) - 1) * cc_ratio**2

        result[i] = np.sqrt(sum_val / window) * np.sqrt(252)

    return result


@jit(nopython=True, cache=True)
def _parkinson_core(high_prices, low_prices, window):
    """Numba-optimized Parkinson volatility calculation.

    Parkinson volatility uses only high and low prices, making it useful
    when close prices are unreliable or for intraday volatility estimation.
    """
    n = len(high_prices)
    result = np.empty(n)
    result[:] = np.nan

    factor = 1 / (4 * np.log(2))

    for i in range(window - 1, n):
        sum_val = 0.0
        for j in range(i - window + 1, i + 1):
            hl_ratio = np.log(high_prices[j] / low_prices[j])
            sum_val += hl_ratio**2

        result[i] = np.sqrt(factor * sum_val / window) * np.sqrt(252)

    return result


@jit(nopython=True, cache=True)
def _rogers_satchell_core(open_prices, high_prices, low_prices, close_prices, window):
    """Numba-optimized Rogers-Satchell volatility calculation.

    Rogers-Satchell volatility captures drift-independent volatility,
    making it suitable for trending markets.
    """
    n = len(close_prices)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        sum_val = 0.0
        for j in range(i - window + 1, i + 1):
            hc = np.log(high_prices[j] / close_prices[j])
            ho = np.log(high_prices[j] / open_prices[j])
            lc = np.log(low_prices[j] / close_prices[j])
            lo = np.log(low_prices[j] / open_prices[j])
            sum_val += hc * ho + lc * lo

        result[i] = np.sqrt(sum_val / window) * np.sqrt(252)

    return result


@jit(nopython=True, cache=True)
def _garch_variance_core(returns, omega, alpha, beta):
    """Numba-optimized GARCH(1,1) variance calculation."""
    n = len(returns)
    variance = np.empty(n)

    # Initialize with unconditional variance
    variance[0] = np.var(returns)

    for i in range(1, n):
        variance[i] = omega + alpha * returns[i - 1] ** 2 + beta * variance[i - 1]

    return np.sqrt(variance) * np.sqrt(252)


class VolatilityFeatures(BaseFeatureCalculator):
    """Advanced volatility feature calculators.

    This class provides specialized volatility measures:
    - Yang-Zhang: Most accurate, accounts for gaps
    - Garman-Klass: Uses high-low-close information
    - Parkinson: Simple high-low volatility
    - Rogers-Satchell: Drift-independent
    - GARCH: Conditional volatility

    For simple rolling standard deviation, use PriceFeatures.calculate_volatility()
    """

    def __init__(self):
        """Initialize volatility calculator."""
        self.min_window = 2
        self._price_features = PriceFeatures()
        logger.debug("VolatilityFeatures initialized")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volatility features.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data

        Returns
        -------
        pd.DataFrame
            Calculated volatility features
        """
        features = pd.DataFrame(index=data.index)

        # Check what data is available
        has_ohlc = all(col in data.columns for col in ["open", "high", "low", "close"])
        has_hlc = all(col in data.columns for col in ["high", "low", "close"])
        has_close = "close" in data.columns

        if has_ohlc:
            # Calculate all OHLC-based volatilities
            features["yang_zhang"] = self.yang_zhang(
                data["open"], data["high"], data["low"], data["close"]
            )
            features["rogers_satchell"] = self.rogers_satchell(
                data["open"], data["high"], data["low"], data["close"]
            )

        if has_hlc:
            features["garman_klass"] = self.garman_klass(
                data["high"], data["low"], data["close"]
            )
            features["parkinson"] = self.parkinson(data["high"], data["low"])

        if has_close:
            # Basic volatility using common_features
            returns = self._price_features.calculate_returns(data["close"])
            features["volatility"] = self._price_features.calculate_volatility(returns)
            features["vol_of_vol"] = self.volatility_of_volatility(data["close"])

        return features

    def yang_zhang(
        self,
        open_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Calculate Yang-Zhang volatility estimator.

        Most accurate volatility estimator that accounts for overnight gaps
        and has minimum estimation error.

        Args:
            open_prices: Open price series
            high_prices: High price series
            low_prices: Low price series
            close_prices: Close price series
            window: Rolling window size

        Returns:
            Yang-Zhang volatility series
        """
        self.validate_window(window, min_window=self.min_window)

        result = _yang_zhang_core(
            open_prices.values,
            high_prices.values,
            low_prices.values,
            close_prices.values,
            window,
        )

        return pd.Series(result, index=close_prices.index, name=f"yang_zhang_{window}")

    def garman_klass(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Calculate Garman-Klass volatility estimator.

        More efficient than close-to-close volatility, uses high, low, close.

        Args:
            high_prices: High price series
            low_prices: Low price series
            close_prices: Close price series
            window: Rolling window size

        Returns:
            Garman-Klass volatility series
        """
        self.validate_window(window, min_window=self.min_window)

        result = _garman_klass_core(
            high_prices.values, low_prices.values, close_prices.values, window
        )

        return pd.Series(
            result, index=close_prices.index, name=f"garman_klass_{window}"
        )

    def parkinson(
        self, high_prices: pd.Series, low_prices: pd.Series, window: int = 20
    ) -> pd.Series:
        """Calculate Parkinson volatility estimator.

        Uses only high and low prices, useful for intraday volatility.

        Args:
            high_prices: High price series
            low_prices: Low price series
            window: Rolling window size

        Returns:
            Parkinson volatility series
        """
        self.validate_window(window, min_window=self.min_window)

        result = _parkinson_core(high_prices.values, low_prices.values, window)

        return pd.Series(result, index=high_prices.index, name=f"parkinson_{window}")

    def rogers_satchell(
        self,
        open_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Calculate Rogers-Satchell volatility estimator.

        Drift-independent volatility measure, good for trending markets.

        Args:
            open_prices: Open price series
            high_prices: High price series
            low_prices: Low price series
            close_prices: Close price series
            window: Rolling window size

        Returns:
            Rogers-Satchell volatility series
        """
        self.validate_window(window, min_window=self.min_window)

        result = _rogers_satchell_core(
            open_prices.values,
            high_prices.values,
            low_prices.values,
            close_prices.values,
            window,
        )

        return pd.Series(
            result, index=close_prices.index, name=f"rogers_satchell_{window}"
        )

    def close_to_close(self, close_prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate standard close-to-close volatility.

        Delegates to PriceFeatures for consistency.

        Args:
            close_prices: Close price series
            window: Rolling window size

        Returns:
            Close-to-close volatility series
        """
        returns = self._price_features.calculate_returns(close_prices, method="log")
        return self._price_features.calculate_volatility(
            returns, window=window, annualize=True
        )

    def garch_volatility(
        self,
        returns: pd.Series,
        omega: float = 0.00001,
        alpha: float = 0.1,
        beta: float = 0.85,
    ) -> pd.Series:
        """Calculate GARCH(1,1) conditional volatility.

        Models time-varying volatility with volatility clustering.

        Args:
            returns: Return series
            omega: Constant term
            alpha: ARCH term coefficient
            beta: GARCH term coefficient

        Returns:
            GARCH volatility series
        """
        # Remove NaN values
        clean_returns = returns.dropna()

        if len(clean_returns) < 10:
            logger.warning("Insufficient data for GARCH calculation")
            return pd.Series(index=returns.index, name="garch_volatility")

        try:
            result = _garch_variance_core(clean_returns.values, omega, alpha, beta)

            # Align with original index
            vol_series = pd.Series(
                index=clean_returns.index, data=result, name="garch_volatility"
            )
            return vol_series.reindex(returns.index)

        except Exception as e:
            logger.error(f"GARCH calculation failed: {e}")
            return pd.Series(index=returns.index, name="garch_volatility")

    def realized_volatility(
        self,
        high_freq_returns: pd.Series,
        freq: str = "5min",
        daily_window: int = 1,
    ) -> pd.Series:
        """Calculate realized volatility from high-frequency data.

        Args:
            high_freq_returns: High-frequency return series
            freq: Frequency of returns ('1min', '5min', etc.)
            daily_window: Number of days for calculation

        Returns:
            Realized volatility series
        """
        # Square returns and sum within each day
        squared_returns = high_freq_returns**2

        # Resample to daily frequency
        realized_var = squared_returns.resample("D").sum()

        # Take square root and annualize
        realized_vol = np.sqrt(realized_var) * np.sqrt(252)
        realized_vol.name = "realized_volatility"

        return realized_vol

    def volatility_of_volatility(
        self, volatility: pd.Series, window: int = 20
    ) -> pd.Series:
        """Calculate volatility of volatility (vol-of-vol).

        Measures the stability of volatility itself.

        Args:
            volatility: Volatility series
            window: Rolling window size

        Returns:
            Vol-of-vol series
        """
        vol_returns = volatility.pct_change()
        vol_of_vol = vol_returns.rolling(window=window).std() * np.sqrt(252)
        vol_of_vol.name = f"vol_of_vol_{window}"
        return vol_of_vol
