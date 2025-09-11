"""Entropy and complexity measures for market data."""

import numpy as np
import pandas as pd
# Optional Numba acceleration
try:  # pragma: no cover - optional dependency
    from numba import jit
except Exception:
    def jit(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        return wrapper
from typing import Optional

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)

# Export the existing class
__all__ = ["EntropyFeatures"]


@jit(nopython=True, cache=True)
def _approximate_entropy_core(data, m, r):
    """Numba-optimized Approximate Entropy calculation.

    ApEn measures the regularity and unpredictability of time series data.
    Lower values indicate more regular/predictable patterns.
    """
    n = len(data)

    def _pattern_count(m_len):
        patterns = np.zeros(n - m_len + 1)

        for i in range(n - m_len + 1):
            template = data[i : i + m_len]
            count = 0

            for j in range(n - m_len + 1):
                if np.max(np.abs(data[j : j + m_len] - template)) <= r:
                    count += 1

            patterns[i] = count / (n - m_len + 1)

        return patterns

    phi_m = _pattern_count(m)
    phi_m1 = _pattern_count(m + 1)

    # Calculate ApEn
    apen = 0.0
    valid_count = 0

    for i in range(len(phi_m)):
        if phi_m[i] > 0:
            apen += np.log(phi_m[i])
            valid_count += 1

    if valid_count > 0:
        apen = apen / valid_count

    apen1 = 0.0
    valid_count = 0

    for i in range(len(phi_m1)):
        if phi_m1[i] > 0:
            apen1 += np.log(phi_m1[i])
            valid_count += 1

    if valid_count > 0:
        apen1 = apen1 / valid_count

    return apen - apen1


@jit(nopython=True, cache=True)
def _sample_entropy_core(data, m, r):
    """Numba-optimized Sample Entropy calculation.

    SampEn is similar to ApEn but doesn't count self-matches,
    making it more consistent and less biased.
    """
    n = len(data)

    def _pattern_count(m_len):
        count = 0

        for i in range(n - m_len):
            template = data[i : i + m_len]

            for j in range(i + 1, n - m_len):
                if np.max(np.abs(data[j : j + m_len] - template)) <= r:
                    count += 1

        return count

    B = _pattern_count(m)
    A = _pattern_count(m + 1)

    if B == 0:
        return np.nan

    return -np.log(A / B)


@jit(nopython=True, cache=True)
def _permutation_entropy_core(data, order):
    """Numba-optimized Permutation Entropy calculation.

    Measures the complexity of a time series by looking at
    the relative frequencies of ordinal patterns.
    """
    n = len(data)

    if n < order:
        return np.nan

    # Count ordinal patterns
    n_perms = 1
    for i in range(2, order + 1):
        n_perms *= i

    pattern_counts = np.zeros(n_perms)

    for i in range(n - order + 1):
        segment = data[i : i + order]

        # Get ranks (ordinal pattern)
        sorted_indices = np.argsort(segment)
        rank = 0
        multiplier = 1

        for j in range(order):
            rank += sorted_indices[j] * multiplier
            multiplier *= order

        # Simple hash for pattern
        pattern_idx = rank % n_perms
        pattern_counts[pattern_idx] += 1

    # Calculate entropy
    total = np.sum(pattern_counts)
    entropy = 0.0

    for count in pattern_counts:
        if count > 0:
            p = count / total
            entropy -= p * np.log(p)

    # Normalize
    return entropy / np.log(n_perms)


class EntropyFeatures:
    """Entropy and complexity feature calculators."""

    def __init__(self, min_window: int = 10):
        """Initialize entropy calculator.

        Args:
            min_window: Minimum window size for calculations
        """
        self.min_window = min_window
        logger.debug("EntropyFeatures initialized")

    def approximate_entropy(
        self, data: pd.Series, window: int = 100, m: int = 2, r: float = 0.2
    ) -> pd.Series:
        """Calculate rolling Approximate Entropy.

        ApEn quantifies the unpredictability of fluctuations in a time series.

        Args:
            data: Input time series
            window: Rolling window size
            m: Pattern length
            r: Tolerance for matches (as fraction of std)

        Returns:
            Approximate entropy series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        # Clean data
        clean_data = data.ffill().bfill()

        for i in range(window - 1, len(data)):
            segment = clean_data.iloc[i - window + 1 : i + 1].values

            # Normalize by standard deviation
            if np.std(segment) > 0:
                r_scaled = r * np.std(segment)
                apen = _approximate_entropy_core(segment, m, r_scaled)
                result[i] = apen

        return pd.Series(
            result, index=data.index, name=f"approximate_entropy_{window}"
        )

    def sample_entropy(
        self, data: pd.Series, window: int = 100, m: int = 2, r: float = 0.2
    ) -> pd.Series:
        """Calculate rolling Sample Entropy.

        SampEn is more consistent than ApEn and doesn't include self-matches.

        Args:
            data: Input time series
            window: Rolling window size
            m: Pattern length
            r: Tolerance for matches (as fraction of std)

        Returns:
            Sample entropy series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        # Clean data
        clean_data = data.ffill().bfill()

        for i in range(window - 1, len(data)):
            segment = clean_data.iloc[i - window + 1 : i + 1].values

            # Normalize by standard deviation
            if np.std(segment) > 0:
                r_scaled = r * np.std(segment)
                sampen = _sample_entropy_core(segment, m, r_scaled)
                result[i] = sampen

        return pd.Series(
            result, index=data.index, name=f"sample_entropy_{window}"
        )

    def permutation_entropy(
        self, data: pd.Series, window: int = 100, order: int = 3
    ) -> pd.Series:
        """Calculate rolling Permutation Entropy.

        Measures complexity by examining ordinal patterns in the time series.

        Args:
            data: Input time series
            window: Rolling window size
            order: Order of permutation patterns

        Returns:
            Permutation entropy series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        # Clean data
        clean_data = data.ffill().bfill()

        for i in range(window - 1, len(data)):
            segment = clean_data.iloc[i - window + 1 : i + 1].values
            permen = _permutation_entropy_core(segment, order)
            result[i] = permen

        return pd.Series(
            result, index=data.index, name=f"permutation_entropy_{window}"
        )

    def multiscale_entropy(
        self,
        data: pd.Series,
        window: int = 100,
        scales: list = [1, 2, 4, 8],
        m: int = 2,
        r: float = 0.2,
    ) -> pd.DataFrame:
        """Calculate Multi-scale Entropy.

        Analyzes complexity at multiple time scales.

        Args:
            data: Input time series
            window: Rolling window size
            scales: List of scales to analyze
            m: Pattern length
            r: Tolerance for matches

        Returns:
            DataFrame with entropy at each scale
        """
        results = {}

        for scale in scales:
            # Coarse-grain the time series
            if scale > 1:
                # Average over non-overlapping windows of size 'scale'
                coarse = data.rolling(window=scale, min_periods=scale).mean()[
                    ::scale
                ]
            else:
                coarse = data.copy()

            # Calculate sample entropy for this scale
            entropy = self.sample_entropy(
                coarse, window=window // scale, m=m, r=r
            )
            results[f"mse_scale_{scale}"] = entropy.reindex(
                data.index, method="ffill"
            )

        return pd.DataFrame(results, index=data.index)

    def rolling_entropy(
        self, data: pd.Series, window: int = 100, bins: int = 10
    ) -> pd.Series:
        """Calculate rolling Shannon entropy using histogram binning.

        Args:
            data: Input time series
            window: Rolling window size
            bins: Number of bins for histogram

        Returns:
            Rolling entropy series
        """
        return self.shannon_entropy(data, window, bins)

    def shannon_entropy(
        self, data: pd.Series, window: int = 100, bins: int = 10
    ) -> pd.Series:
        """Calculate rolling Shannon Entropy.

        Measures information content using probability distribution.

        Args:
            data: Input time series
            window: Rolling window size
            bins: Number of bins for histogram

        Returns:
            Shannon entropy series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna()

            if len(segment) > 1:
                # Create histogram
                counts, _ = np.histogram(segment, bins=bins)

                # Calculate probabilities
                probs = counts / np.sum(counts)

                # Calculate entropy
                entropy = 0
                for p in probs:
                    if p > 0:
                        entropy -= p * np.log2(p)

                result[i] = entropy

        return pd.Series(
            result, index=data.index, name=f"shannon_entropy_{window}"
        )

    def renyi_entropy(
        self,
        data: pd.Series,
        window: int = 100,
        alpha: float = 2.0,
        bins: int = 10,
    ) -> pd.Series:
        """Calculate rolling Rényi Entropy.

        Generalization of Shannon entropy with parameter alpha.

        Args:
            data: Input time series
            window: Rolling window size
            alpha: Rényi parameter (alpha=1 gives Shannon entropy)
            bins: Number of bins for histogram

        Returns:
            Rényi entropy series
        """
        if alpha == 1:
            return self.shannon_entropy(data, window, bins)

        result = np.empty(len(data))
        result[:] = np.nan

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna()

            if len(segment) > 1:
                # Create histogram
                counts, _ = np.histogram(segment, bins=bins)

                # Calculate probabilities
                probs = counts / np.sum(counts)

                # Calculate Rényi entropy
                sum_p_alpha = np.sum(probs**alpha)

                if sum_p_alpha > 0:
                    entropy = np.log2(sum_p_alpha) / (1 - alpha)
                    result[i] = entropy

        return pd.Series(
            result, index=data.index, name=f"renyi_entropy_{window}_a{alpha}"
        )

    def spectral_entropy(
        self, data: pd.Series, window: int = 100, nperseg: Optional[int] = None
    ) -> pd.Series:
        """Calculate rolling Spectral Entropy.

        Measures entropy in frequency domain.

        Args:
            data: Input time series
            window: Rolling window size
            nperseg: Length of each segment for FFT

        Returns:
            Spectral entropy series
        """
        from scipy import signal

        result = np.empty(len(data))
        result[:] = np.nan

        if nperseg is None:
            nperseg = min(window // 4, 256)

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna().values

            if len(segment) > nperseg:
                # Calculate power spectral density
                freqs, psd = signal.periodogram(segment, scaling="density")

                # Normalize PSD
                psd_norm = psd / np.sum(psd)

                # Calculate spectral entropy
                entropy = 0
                for p in psd_norm:
                    if p > 0:
                        entropy -= p * np.log2(p)

                result[i] = entropy

        return pd.Series(
            result, index=data.index, name=f"spectral_entropy_{window}"
        )

    def conditional_entropy(
        self, data: pd.Series, window: int = 100, bins: int = 10, lag: int = 1
    ) -> pd.Series:
        """Calculate rolling Conditional Entropy.

        Measures the amount of information needed to describe data given past values.

        Args:
            data: Input time series
            window: Rolling window size
            bins: Number of bins for discretization
            lag: Lag for conditional dependence

        Returns:
            Conditional entropy series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        for i_outer in range(window - 1, len(data)):
            segment = data.iloc[i_outer - window + 1 : i_outer + 1].dropna()

            if len(segment) > lag:
                # Create lagged series
                current = segment[lag:].values
                lagged = segment[:-lag].values

                # Discretize data
                hist2d, _, _ = np.histogram2d(current, lagged, bins=bins)

                # Calculate joint and marginal probabilities
                p_xy = hist2d / hist2d.sum()
                p_y = hist2d.sum(axis=0) / hist2d.sum()

                # Calculate conditional entropy
                h_cond = 0
                for i_bin in range(bins):
                    for j_bin in range(bins):
                        if p_xy[i_bin, j_bin] > 0 and p_y[j_bin] > 0:
                            h_cond -= p_xy[i_bin, j_bin] * np.log2(
                                p_xy[i_bin, j_bin] / p_y[j_bin]
                            )

                result[i_outer] = h_cond

        return pd.Series(
            result, index=data.index, name=f"conditional_entropy_{window}"
        )

    def transfer_entropy(
        self,
        source: pd.Series,
        target: pd.Series,
        window: int = 100,
        lag: int = 1,
        bins: int = 10,
    ) -> pd.Series:
        """Calculate rolling Transfer Entropy.

        Measures information flow from source to target series.

        Args:
            source: Source time series
            target: Target time series
            window: Rolling window size
            lag: Time lag for transfer
            bins: Number of bins for discretization

        Returns:
            Transfer entropy series
        """
        # Align series
        aligned_source = source.reindex(target.index)

        result = np.empty(len(target))
        result[:] = np.nan

        for i in range(window - 1, len(target)):
            source_seg = aligned_source.iloc[i - window + 1 : i + 1].dropna()
            target_seg = target.iloc[i - window + 1 : i + 1].dropna()

            if len(source_seg) > lag and len(target_seg) > lag:
                # Prepare lagged series
                target_future = target_seg[lag:].values
                target_past = target_seg[:-lag].values
                source_past = source_seg[:-lag].values

                # Discretize
                min_len = min(
                    len(target_future), len(target_past), len(source_past)
                )
                target_future = target_future[:min_len]
                target_past = target_past[:min_len]
                source_past = source_past[:min_len]

                # Calculate transfer entropy using conditional entropy
                # TE = H(target_future|target_past) - H(target_future|target_past,source_past)
                try:
                    # Simple approximation
                    hist_tt, _, _ = np.histogram2d(
                        target_future, target_past, bins=bins
                    )
                    hist_tts = np.histogramdd(
                        [target_future, target_past, source_past], bins=bins
                    )[0]

                    # Normalize
                    p_tt = hist_tt / hist_tt.sum()
                    p_tts = hist_tts / hist_tts.sum()

                    # Calculate entropies
                    h_t_given_t = -np.sum(
                        p_tt[p_tt > 0] * np.log2(p_tt[p_tt > 0])
                    )
                    h_t_given_ts = -np.sum(
                        p_tts[p_tts > 0] * np.log2(p_tts[p_tts > 0])
                    )

                    te = h_t_given_t - h_t_given_ts
                    result[i] = max(0, te)  # Transfer entropy is non-negative
                except Exception:
                    pass

        return pd.Series(
            result, index=target.index, name=f"transfer_entropy_{window}"
        )

    def lempel_ziv_complexity(
        self, data: pd.Series, window: int = 100, threshold: float = 0
    ) -> pd.Series:
        """Calculate rolling Lempel-Ziv complexity.

        Measures the complexity of a binary sequence.

        Args:
            data: Input time series
            window: Rolling window size
            threshold: Threshold for binarization (default: median)

        Returns:
            Lempel-Ziv complexity series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna()

            if len(segment) > 1:
                # Binarize the data
                if threshold == 0:
                    threshold = segment.median()
                binary_string = "".join(
                    ["1" if x > threshold else "0" for x in segment]
                )

                # Calculate LZ complexity
                n = len(binary_string)
                c = 1  # Complexity counter
                substr_len = 1  # Length of current substring
                i_idx = 0  # Current position
                k = 1  # Look-ahead position
                k_max = 1  # Maximum length found

                while k + k_max <= n:
                    # Check if substring exists in history
                    if (
                        binary_string[i_idx : i_idx + k_max]
                        == binary_string[k : k + k_max]
                    ):
                        k_max += 1
                    else:
                        if k_max > substr_len:
                            substr_len = k_max

                        i_idx = k
                        if k + 1 <= n:
                            c += 1
                            k = i_idx + 1
                            k_max = 1
                            substr_len = 1
                        else:
                            break

                # Normalize by theoretical maximum
                b = n / np.log2(n) if n > 1 else 1
                result[i] = c / b

        return pd.Series(result, index=data.index, name=f"lempel_ziv_{window}")

    def correlation_dimension(
        self,
        data: pd.Series,
        window: int = 100,
        max_dim: int = 10,
        r: Optional[float] = None,
    ) -> pd.Series:
        """Calculate rolling Correlation Dimension.

        Estimates the fractal dimension of the time series attractor.

        Args:
            data: Input time series
            window: Rolling window size
            max_dim: Maximum embedding dimension
            r: Radius for correlation sum (default: std/10)

        Returns:
            Correlation dimension series
        """
        result = np.empty(len(data))
        result[:] = np.nan

        for i in range(window - 1, len(data)):
            segment = data.iloc[i - window + 1 : i + 1].dropna().values

            if len(segment) > max_dim * 2:
                if r is None:
                    r = np.std(segment) / 10

                dimensions = []
                correlation_sums = []

                for m in range(2, min(max_dim + 1, len(segment) // 2)):
                    # Embed time series
                    embedded = np.array(
                        [
                            segment[j : j + m]
                            for j in range(len(segment) - m + 1)
                        ]
                    )
                    n_points = len(embedded)

                    # Calculate correlation sum
                    c_r = 0
                    for j in range(n_points):
                        for k in range(j + 1, n_points):
                            if np.linalg.norm(embedded[j] - embedded[k]) < r:
                                c_r += 1

                    c_r = (
                        2 * c_r / (n_points * (n_points - 1))
                        if n_points > 1
                        else 0
                    )

                    if c_r > 0:
                        dimensions.append(m)
                        correlation_sums.append(np.log(c_r))

                # Estimate dimension from scaling
                if len(dimensions) > 2:
                    # Linear fit in log-log space
                    slope, _ = np.polyfit(dimensions, correlation_sums, 1)
                    result[i] = abs(slope)

        return pd.Series(
            result, index=data.index, name=f"correlation_dim_{window}"
        )
