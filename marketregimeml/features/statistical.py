"""Statistical feature calculators.

This module provides advanced statistical features. For basic rolling statistics,
use common_features.StatisticalFeatures or base.BaseFeatureCalculator.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from marketregimeml.features.base import BaseFeatureCalculator
from marketregimeml.features.common_features import StatisticalFeatures as CommonStats
from marketregimeml.utils.logging import get_logger

# Optional Numba acceleration
try:  # pragma: no cover - optional dependency
    from numba import jit
except Exception:
    def jit(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        return wrapper

# Try to import entropy libraries
try:
    import antropy as ant

    ANTROPY_AVAILABLE = True
except ImportError:
    ANTROPY_AVAILABLE = False
    warnings.warn(
        "antropy not installed. Some entropy features will use fallback implementations."
    )

logger = get_logger(__name__)


@jit(nopython=True, cache=True)
def _rolling_skewness_core(data, window):
    """Numba-optimized rolling skewness calculation."""
    n = len(data)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        segment = data[i - window + 1 : i + 1]

        mean = np.mean(segment)
        std = np.std(segment)

        if std > 0:
            skew = np.mean(((segment - mean) / std) ** 3)
            result[i] = skew

    return result


@jit(nopython=True, cache=True)
def _rolling_kurtosis_core(data, window):
    """Numba-optimized rolling kurtosis calculation."""
    n = len(data)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        segment = data[i - window + 1 : i + 1]

        mean = np.mean(segment)
        std = np.std(segment)

        if std > 0:
            kurt = np.mean(((segment - mean) / std) ** 4) - 3
            result[i] = kurt

    return result


@jit(nopython=True, cache=True)
def _autocorrelation_core(data, window, lag):
    """Numba-optimized rolling autocorrelation calculation."""
    n = len(data)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        if i >= window + lag - 1:
            segment = data[i - window + 1 : i + 1]
            segment_lag = data[i - window + 1 - lag : i + 1 - lag]

            # Calculate correlation
            mean1 = np.mean(segment)
            mean2 = np.mean(segment_lag)

            cov = np.mean((segment - mean1) * (segment_lag - mean2))
            std1 = np.std(segment)
            std2 = np.std(segment_lag)

            if std1 > 0 and std2 > 0:
                result[i] = cov / (std1 * std2)

    return result


class StatisticalFeatures(BaseFeatureCalculator):
    """Advanced statistical feature calculators.

    This class provides specialized statistical features.
    For basic statistics (mean, std, skew, kurt), use:
    - CommonStats for rolling statistics
    - BaseFeatureCalculator.rolling_operation() for custom operations
    """

    def __init__(self):
        """Initialize statistical calculator."""
        self._common_stats = CommonStats()
        logger.debug("StatisticalFeatures initialized")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all statistical features.

        Parameters
        ----------
        data : pd.DataFrame
            Input data (typically with 'close' or 'returns' column)

        Returns
        -------
        pd.DataFrame
            Calculated statistical features
        """
        features = pd.DataFrame(index=data.index)

        # Use returns if available, otherwise calculate from close
        if 'returns' in data.columns:
            returns = data['returns']
        elif 'close' in data.columns:
            returns = data['close'].pct_change()
        else:
            return features  # No suitable data

        # Basic statistics (delegating to common/base)
        features['skewness'] = self.rolling_skewness(returns)
        features['kurtosis'] = self.rolling_kurtosis(returns)
        features['autocorr'] = self.autocorrelation(returns)

        # Advanced statistics
        features['hurst'] = self.hurst_exponent(returns)
        features['jarque_bera'] = self.jarque_bera_stat(returns)

        # Entropy measures (if data is long enough)
        if len(returns) > 100:
            features['shannon_entropy'] = self.shannon_entropy(returns)
            features['approx_entropy'] = self.approximate_entropy(returns)

        return features

    def rolling_skewness(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling skewness - delegates to CommonStats."""
        return self._common_stats.calculate_rolling_skewness(data, window)

    def rolling_kurtosis(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling kurtosis - delegates to CommonStats."""
        return self._common_stats.calculate_rolling_kurtosis(data, window)

    def autocorrelation(
        self, data: pd.Series, window: int = 20, lag: int = 1
    ) -> pd.Series:
        """Calculate rolling autocorrelation.

        Measures serial correlation in returns.

        Args:
            data: Input time series
            window: Rolling window size
            lag: Lag for autocorrelation

        Returns:
            Rolling autocorrelation series
        """
        clean_data = data.fillna(0)
        result = _autocorrelation_core(clean_data.values, window, lag)
        return pd.Series(
            result, index=data.index, name=f"autocorr_{window}_lag{lag}"
        )

    def partial_autocorrelation(
        self, data: pd.Series, window: int = 50, lag: int = 1
    ) -> pd.Series:
        """Calculate rolling partial autocorrelation.

        Measures direct correlation at specific lag, removing intermediate effects.

        Args:
            data: Input time series
            window: Rolling window size
            lag: Lag for PACF

        Returns:
            Rolling partial autocorrelation series
        """
        from statsmodels.tsa.stattools import pacf

        result = np.empty(len(data))
        result[:] = np.nan

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna()

            if len(segment) >= window:
                try:
                    pacf_values = pacf(segment, nlags=lag, method="ywunbiased")
                    result[i] = pacf_values[lag]
                except Exception:
                    pass

        return pd.Series(
            result, index=data.index, name=f"pacf_{window}_lag{lag}"
        )

    def jarque_bera_stat(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling Jarque-Bera test statistic.

        Tests for normality of distribution.
        Higher values indicate departure from normality.

        Args:
            data: Input time series
            window: Rolling window size

        Returns:
            Rolling JB statistic series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna()

            if len(segment) >= 3:
                jb_stat, _ = stats.jarque_bera(segment)
                result[i] = jb_stat

        return pd.Series(
            result, index=data.index, name=f"jarque_bera_{window}"
        )

    def shapiro_wilk_stat(
        self, data: pd.Series, window: int = 20
    ) -> pd.Series:
        """Calculate rolling Shapiro-Wilk test statistic.

        Another test for normality, more powerful for small samples.

        Args:
            data: Input time series
            window: Rolling window size

        Returns:
            Rolling SW statistic series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna()

            if 3 <= len(segment) <= 5000:  # SW test limitations
                try:
                    sw_stat, _ = stats.shapiro(segment)
                    result[i] = sw_stat
                except Exception:
                    pass

        return pd.Series(
            result, index=data.index, name=f"shapiro_wilk_{window}"
        )

    def cross_correlation(
        self,
        data1: pd.Series,
        data2: pd.Series,
        window: int = 20,
        lag: int = 0,
    ) -> pd.Series:
        """Calculate rolling cross-correlation between two series.

        Useful for multi-asset analysis and lead-lag relationships.

        Args:
            data1: First time series
            data2: Second time series
            window: Rolling window size
            lag: Lag for cross-correlation

        Returns:
            Rolling cross-correlation series
        """
        result = np.empty(len(data1))
        result[:] = np.nan

        for i in range(window - 1, len(data1)):
            if i >= window + abs(lag) - 1:
                if lag >= 0:
                    seg1 = data1.iloc[i - window + 1 : i + 1]
                    seg2 = data2.iloc[i - window + 1 - lag : i + 1 - lag]
                else:
                    seg1 = data1.iloc[
                        i - window + 1 + abs(lag) : i + 1 + abs(lag)
                    ]
                    seg2 = data2.iloc[i - window + 1 : i + 1]

                if len(seg1) == len(seg2):
                    corr = seg1.corr(seg2)
                    result[i] = corr

        return pd.Series(
            result, index=data1.index, name=f"cross_corr_{window}_lag{lag}"
        )

    def regime_stability(
        self, regimes: pd.Series, window: int = 20
    ) -> pd.Series:
        """Calculate regime stability metric.

        Measures how stable/persistent regimes are over time.

        Args:
            regimes: Series of regime labels
            window: Rolling window size

        Returns:
            Regime stability series (0-1, higher is more stable)
        """
        result = np.empty(len(regimes))
        result[:] = np.nan

        for i in range(window - 1, len(regimes)):
            segment = regimes.iloc[i - window + 1 : i + 1]

            # Count regime changes
            changes = (segment != segment.shift(1)).sum() - 1

            # Stability = 1 - (changes / max_possible_changes)
            max_changes = window - 1
            stability = 1 - (changes / max_changes)
            result[i] = stability

        return pd.Series(result, index=regimes.index, name="regime_stability")

    def hurst_exponent(
        self,
        data: pd.Series,
        window: int = 100,
        min_lag: int = 2,
        max_lag: Optional[int] = None,
    ) -> pd.Series:
        """Calculate rolling Hurst exponent.

        Measures long-term memory and trend persistence.
        H > 0.5: Trending
        H = 0.5: Random walk
        H < 0.5: Mean-reverting

        Args:
            data: Input time series
            window: Rolling window size
            min_lag: Minimum lag for R/S analysis
            max_lag: Maximum lag (default: window//4)

        Returns:
            Rolling Hurst exponent series
        """
        if max_lag is None:
            max_lag = window // 4

        result = np.empty(len(data))
        result[:] = np.nan

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna().values

            if len(segment) >= window:
                lags = []
                rs_values = []

                for lag in range(min_lag, min(max_lag + 1, len(segment) // 2)):
                    # R/S analysis
                    n_chunks = len(segment) // lag

                    if n_chunks > 0:
                        rs_lag = []

                        for j in range(n_chunks):
                            chunk = segment[j * lag : (j + 1) * lag]

                            if len(chunk) > 1:
                                # Calculate cumulative deviations
                                mean = np.mean(chunk)
                                deviations = chunk - mean
                                cumsum = np.cumsum(deviations)

                                # Range and standard deviation
                                R = np.max(cumsum) - np.min(cumsum)
                                S = np.std(chunk)

                                if S > 0:
                                    rs_lag.append(R / S)

                        if rs_lag:
                            lags.append(np.log(lag))
                            rs_values.append(np.log(np.mean(rs_lag)))

                # Estimate Hurst exponent via linear regression
                if len(lags) > 1:
                    hurst = np.polyfit(lags, rs_values, 1)[0]
                    result[i] = hurst

        return pd.Series(result, index=data.index, name=f"hurst_{window}")

    def adf_test(
        self,
        data: pd.Series,
        window: int = 100,
        regression: str = "c",
        autolag: str = "AIC",
    ) -> pd.Series:
        """Calculate rolling Augmented Dickey-Fuller test statistic.

        Tests for unit root (non-stationarity) in time series.
        More negative values indicate stronger stationarity.

        Args:
            data: Input time series
            window: Rolling window size
            regression: Regression type ('c': constant, 'ct': constant+trend, 'ctt': constant+trend+trend^2)
            autolag: Method to determine lag ('AIC', 'BIC', 't-stat', None)

        Returns:
            Rolling ADF test statistic series
        """
        from statsmodels.tsa.stattools import adfuller

        result = np.empty(len(data))
        result[:] = np.nan

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna()

            if (
                len(segment) >= window * 0.8
            ):  # Need at least 80% non-NaN values
                try:
                    adf_result = adfuller(
                        segment, regression=regression, autolag=autolag
                    )
                    result[i] = adf_result[0]  # ADF test statistic
                except Exception:
                    pass

        return pd.Series(result, index=data.index, name=f"adf_stat_{window}")

    def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """Calculate rolling Information Ratio.

        Risk-adjusted return relative to benchmark.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            window: Rolling window size

        Returns:
            Rolling information ratio series
        """
        excess_returns = returns - benchmark_returns

        mean_excess = excess_returns.rolling(window=window).mean()
        std_excess = excess_returns.rolling(window=window).std()

        ir = mean_excess / std_excess * np.sqrt(252)
        ir.name = f"information_ratio_{window}"

        return ir

    def skewness(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Simple wrapper for rolling_skewness for test compatibility."""
        return self.rolling_skewness(data, window)

    def kurtosis(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Simple wrapper for rolling_kurtosis for test compatibility."""
        return self.rolling_kurtosis(data, window)

    # ============== NEW METHODS USING LIBRARIES ==============

    def rolling_mean(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling mean - delegates to base class."""
        return self.rolling_operation(data, window, 'mean')

    def rolling_std(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling standard deviation - delegates to base class."""
        return self.rolling_operation(data, window, 'std')

    def rolling_median(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling median - delegates to base class."""
        return self.rolling_operation(data, window, 'median')

    def rolling_min(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling minimum - delegates to base class."""
        return self.rolling_operation(data, window, 'min')

    def rolling_max(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling maximum - delegates to base class."""
        return self.rolling_operation(data, window, 'max')

    def rolling_quantile(
        self, data: pd.Series, window: int = 20, quantile: float = 0.5
    ) -> pd.Series:
        """Calculate rolling quantile.

        Args:
            data: Input time series
            window: Rolling window size
            quantile: Quantile to calculate (0-1)

        Returns:
            Rolling quantile series
        """
        result = data.rolling(window=window, min_periods=window).quantile(
            quantile
        )
        result.name = f"quantile_{window}_q{int(quantile*100)}"
        return result

    def rolling_correlation(
        self, series1: pd.Series, series2: pd.Series, window: int = 20
    ) -> pd.Series:
        """Calculate rolling correlation between two series.

        Args:
            series1: First time series
            series2: Second time series
            window: Rolling window size

        Returns:
            Rolling correlation series
        """
        result = series1.rolling(window=window, min_periods=window).corr(
            series2
        )
        result.name = f"correlation_{window}"
        return result

    def rolling_statistics(
        self, data: pd.Series, window: int = 20
    ) -> pd.DataFrame:
        """Calculate comprehensive rolling statistics.

        Args:
            data: Input time series
            window: Rolling window size

        Returns:
            DataFrame with multiple statistics
        """
        rolling = data.rolling(window=window, min_periods=window)

        stats_df = pd.DataFrame(
            {
                "mean": rolling.mean(),
                "std": rolling.std(),
                "min": rolling.min(),
                "max": rolling.max(),
                "median": rolling.median(),
                "skewness": self.rolling_skewness(data, window),
                "kurtosis": self.rolling_kurtosis(data, window),
            }
        )

        return stats_df

    def shannon_entropy(
        self, data: pd.Series, window: int = 50, bins: int = 10
    ) -> pd.Series:
        """Calculate rolling Shannon entropy.

        Measures information content/uncertainty in the distribution.

        Args:
            data: Input time series
            window: Rolling window size
            bins: Number of bins for histogram

        Returns:
            Rolling Shannon entropy series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna()

            if len(segment) >= window:
                # Create histogram
                counts, _ = np.histogram(segment, bins=bins)
                # Normalize to get probabilities
                probs = counts / counts.sum()
                # Remove zero probabilities
                probs = probs[probs > 0]
                # Calculate Shannon entropy
                entropy = -np.sum(probs * np.log(probs))
                result[i] = entropy

        return pd.Series(
            result, index=data.index, name=f"shannon_entropy_{window}"
        )

    def approximate_entropy(
        self, data: pd.Series, window: int = 50, m: int = 2, r: float = 0.2
    ) -> pd.Series:
        """Calculate rolling approximate entropy.

        Measures regularity and unpredictability of time series.

        Args:
            data: Input time series
            window: Rolling window size
            m: Pattern length
            r: Tolerance for matches (fraction of std)

        Returns:
            Rolling approximate entropy series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        if ANTROPY_AVAILABLE:
            for i in range(window - 1, len(data)):
                segment = data.iloc[i - window + 1 : i + 1].dropna().values

                if len(segment) >= window:
                    try:
                        # Use antropy library for efficient calculation
                        tolerance = r * np.std(segment)
                        apen = ant.app_entropy(
                            segment, order=m, tolerance=tolerance, metric="chebyshev"
                        )
                        result[i] = apen
                    except (ValueError, RuntimeError) as e:
                        logger.debug(
                            f"Approximate entropy calculation failed at index {i}: {e}"
                        )
                        result[i] = np.nan
        else:
            # Fallback implementation
            logger.warning(
                "Using fallback ApEn implementation. Install antropy for better performance."
            )
            for i in range(window - 1, len(data)):
                segment = data.iloc[i - window + 1 : i + 1].dropna().values

                if len(segment) >= window:
                    result[i] = self._approximate_entropy_fallback(
                        segment, m, r
                    )

        return pd.Series(
            result, index=data.index, name=f"approx_entropy_{window}_m{m}"
        )

    def sample_entropy(
        self, data: pd.Series, window: int = 50, m: int = 2, r: float = 0.2
    ) -> pd.Series:
        """Calculate rolling sample entropy.

        Improved version of approximate entropy with better consistency.

        Args:
            data: Input time series
            window: Rolling window size
            m: Pattern length
            r: Tolerance for matches (fraction of std)

        Returns:
            Rolling sample entropy series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        if ANTROPY_AVAILABLE:
            for i in range(window - 1, len(data)):
                segment = data.iloc[i - window + 1 : i + 1].dropna().values

                if len(segment) >= window:
                    try:
                        # Use antropy library
                        tolerance = r * np.std(segment)
                        sampen = ant.sample_entropy(
                            segment, order=m, tolerance=tolerance, metric="chebyshev"
                        )
                        result[i] = sampen
                    except (ValueError, RuntimeError) as e:
                        logger.debug(
                            f"Sample entropy calculation failed at index {i}: {e}"
                        )
                        result[i] = np.nan
        else:
            # Simple fallback
            logger.warning(
                "Using fallback SampEn implementation. Install antropy for better performance."
            )
            for i in range(window - 1, len(data)):
                segment = data.iloc[i - window + 1 : i + 1].dropna().values

                if len(segment) >= window:
                    result[i] = self._sample_entropy_fallback(segment, m, r)

        return pd.Series(
            result, index=data.index, name=f"sample_entropy_{window}_m{m}"
        )

    def permutation_entropy(
        self, data: pd.Series, window: int = 50, order: int = 3
    ) -> pd.Series:
        """Calculate rolling permutation entropy.

        Measures complexity based on ordinal patterns.

        Args:
            data: Input time series
            window: Rolling window size
            order: Permutation order (pattern length)

        Returns:
            Rolling permutation entropy series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        if ANTROPY_AVAILABLE:
            for i in range(window - 1, len(data)):
                segment = data.iloc[i - window + 1 : i + 1].dropna().values

                if len(segment) >= window:
                    try:
                        # Use antropy library
                        permen = ant.perm_entropy(
                            segment, order=order, normalize=False
                        )
                        result[i] = permen
                    except (ValueError, RuntimeError) as e:
                        logger.debug(
                            f"Permutation entropy calculation failed at index {i}: {e}"
                        )
                        result[i] = np.nan
        else:
            # Simple fallback using ordinal patterns
            logger.warning(
                "Using fallback PermEn implementation. Install antropy for better performance."
            )
            for i in range(window - 1, len(data)):
                segment = data.iloc[i - window + 1 : i + 1].dropna().values

                if len(segment) >= window:
                    result[i] = self._permutation_entropy_fallback(
                        segment, order
                    )

        return pd.Series(
            result, index=data.index, name=f"perm_entropy_{window}_o{order}"
        )

    # ============== FALLBACK IMPLEMENTATIONS ==============

    def _approximate_entropy_fallback(
        self, U: np.ndarray, m: int, r: float
    ) -> float:
        """Fallback implementation of approximate entropy."""
        N = len(U)
        r_scaled = r * np.std(U)

        def _maxdist(xi, xj):
            """Maximum distance between patterns."""
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            patterns = np.array([U[i : i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)

            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j]) <= r_scaled:
                        C[i] += 1

            phi = (N - m + 1) ** -1 * np.sum(np.log(C / (N - m + 1)))
            return phi

        try:
            return _phi(m) - _phi(m + 1)
        except Exception:
            return 0.0

    def _sample_entropy_fallback(
        self, U: np.ndarray, m: int, r: float
    ) -> float:
        """Fallback implementation of sample entropy."""
        N = len(U)
        r_scaled = r * np.std(U)

        def _maxdist(xi, xj):
            """Maximum distance between patterns."""
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            patterns = np.array([U[i : i + m] for i in range(N - m + 1)])
            A = 0

            for i in range(N - m):
                template = patterns[i]
                for j in range(i + 1, N - m + 1):
                    if _maxdist(template, patterns[j]) <= r_scaled:
                        A += 1

            return A

        try:
            A = _phi(m)
            B = _phi(m + 1)

            if B == 0:
                return float("inf")

            return -np.log(B / A)
        except Exception:
            return 0.0

    def _permutation_entropy_fallback(
        self, time_series: np.ndarray, order: int
    ) -> float:
        """Fallback implementation of permutation entropy."""
        n = len(time_series)
        if n < order:
            return np.nan

        # Get all possible permutations
        from itertools import permutations

        perms = list(permutations(range(order)))
        perm_count = {perm: 0 for perm in perms}

        # Count ordinal patterns
        for i in range(n - order + 1):
            segment = time_series[i : i + order]
            sorted_indices = tuple(np.argsort(segment))
            if sorted_indices in perm_count:
                perm_count[sorted_indices] += 1

        # Calculate probabilities
        total = sum(perm_count.values())
        if total == 0:
            return 0.0

        probs = np.array(
            [count / total for count in perm_count.values() if count > 0]
        )

        # Calculate entropy
        return -np.sum(probs * np.log(probs))
