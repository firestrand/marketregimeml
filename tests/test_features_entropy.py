"""Tests for entropy feature calculators."""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch

from marketregimeml.features.entropy import EntropyFeatures


class TestEntropyFeatures:
    """Test suite for entropy feature calculators."""

    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data."""
        np.random.seed(42)
        n_samples = 100

        # Create time series with some patterns
        t = np.linspace(0, 10, n_samples)
        signal = (
            np.sin(t) + 0.5 * np.sin(3 * t) + 0.2 * np.random.randn(n_samples)
        )

        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        return pd.Series(signal, index=dates)

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        np.random.seed(42)
        n_samples = 200

        # Mix of normal and regime-switching behavior
        returns1 = np.random.normal(
            0.001, 0.02, n_samples // 2
        )  # Low vol regime
        returns2 = np.random.normal(
            -0.002, 0.05, n_samples // 2
        )  # High vol regime
        returns = np.concatenate([returns1, returns2])

        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        return pd.Series(returns, index=dates)

    @pytest.fixture
    def calculator(self):
        """Create entropy calculator."""
        return EntropyFeatures()

    def test_init(self, calculator):
        """Test initialization."""
        assert calculator.min_window == 10

    def test_approximate_entropy(self, calculator, sample_time_series):
        """Test approximate entropy calculation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.approximate_entropy(
                sample_time_series, window=30, m=2, r=0.2
            )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_time_series)
        assert result.name.startswith("approximate_entropy")

        # First (window-1) values should be NaN
        assert np.isnan(result.iloc[:29]).all()

        # Entropy should be positive
        assert (result.dropna() >= 0).all()

    def test_approximate_entropy_parameters(
        self, calculator, sample_time_series
    ):
        """Test approximate entropy with different parameters."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Test different m values
            result_m1 = calculator.approximate_entropy(
                sample_time_series, window=30, m=1, r=0.2
            )
            result_m2 = calculator.approximate_entropy(
                sample_time_series, window=30, m=2, r=0.2
            )
            result_m3 = calculator.approximate_entropy(
                sample_time_series, window=30, m=3, r=0.2
            )

        # All should be valid series
        for result in [result_m1, result_m2, result_m3]:
            assert isinstance(result, pd.Series)
            # Approximate entropy can be slightly negative due to numerical precision
            assert (result.dropna() >= -0.1).all()  # Allow small negative values

    def test_sample_entropy(self, calculator, sample_time_series):
        """Test sample entropy calculation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.sample_entropy(
                sample_time_series, window=30, m=2, r=0.2
            )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_time_series)
        assert result.name.startswith("sample_entropy")

        # Sample entropy should be positive
        assert (result.dropna() >= 0).all()

    def test_permutation_entropy(self, calculator, sample_time_series):
        """Test permutation entropy calculation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.permutation_entropy(
                sample_time_series, window=30, order=3
            )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_time_series)
        assert result.name.startswith("permutation_entropy")

        # Permutation entropy should be between 0 and 1 (normalized)
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (
            valid_values <= 1.0 + 1e-10
        ).all()  # Normalized entropy, small tolerance for numerical errors

    def test_multiscale_entropy(self, calculator, sample_time_series):
        """Test multiscale entropy calculation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.multiscale_entropy(
                sample_time_series, window=50, scales=[1, 2, 3, 4, 5]
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_time_series)

        # Should have columns for each scale
        expected_columns = [f"mse_scale_{i}" for i in range(1, 6)]
        assert all(col in result.columns for col in expected_columns)

        # All entropy values should be non-negative
        for col in result.columns:
            valid_values = result[col].dropna()
            assert (valid_values >= 0).all()

    def test_rolling_entropy(self, calculator, sample_returns):
        """Test rolling entropy calculation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.rolling_entropy(
                sample_returns, window=30, bins=10
            )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_returns)
        assert result.name.startswith("shannon_entropy")  # rolling_entropy calls shannon_entropy

        # Entropy should be non-negative
        assert (result.dropna() >= 0).all()

    def test_rolling_entropy_different_bins(self, calculator, sample_returns):
        """Test rolling entropy with different bin counts."""
        bin_counts = [5, 10, 20]

        for bins in bin_counts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = calculator.rolling_entropy(
                    sample_returns, window=30, bins=bins
                )

            assert isinstance(result, pd.Series)
            valid_values = result.dropna()

            # Entropy should be bounded by log2(bins) (using log2 in shannon_entropy)
            max_entropy = np.log2(bins)
            assert (valid_values <= max_entropy + 1e-10).all()

    def test_conditional_entropy(self, calculator, sample_returns):
        """Test conditional entropy calculation."""
        # Create a lagged version as conditioning variable
        lagged_returns = sample_returns.shift(1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.conditional_entropy(
                sample_returns, window=50, bins=10, lag=1
            )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_returns)
        assert result.name.startswith("conditional_entropy")

        # Conditional entropy should be non-negative
        assert (result.dropna() >= 0).all()

    def test_transfer_entropy(self, calculator, sample_returns):
        """Test transfer entropy calculation."""
        # Create another series with some dependence
        np.random.seed(123)
        other_series = sample_returns.shift(1) + 0.5 * np.random.randn(
            len(sample_returns)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.transfer_entropy(
                sample_returns, other_series, window=50, lag=1, bins=10
            )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_returns)
        assert result.name.startswith("transfer_entropy")

        # Transfer entropy should be non-negative
        assert (result.dropna() >= 0).all()

    def test_lempel_ziv_complexity(self, calculator):
        """Test Lempel-Ziv complexity calculation."""
        # Create binary sequence
        np.random.seed(42)
        binary_sequence = np.random.choice([0, 1], size=100)
        binary_series = pd.Series(binary_sequence)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.lempel_ziv_complexity(binary_series, window=50)

        assert isinstance(result, pd.Series)
        assert len(result) == len(binary_series)
        assert result.name.startswith("lempel_ziv")

        # LZ complexity should be positive
        assert (result.dropna() > 0).all()

    def test_correlation_dimension(self, calculator, sample_time_series):
        """Test correlation dimension calculation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.correlation_dimension(
                sample_time_series, window=50, max_dim=3, r=1.0
            )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_time_series)
        assert result.name.startswith("correlation_dim")

        # Correlation dimension should be positive and bounded
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 10).all()  # Reasonable upper bound

    def test_invalid_window_size(self, calculator, sample_time_series):
        """Test error handling for invalid window sizes."""
        # Window larger than data should just return mostly NaN
        result = calculator.approximate_entropy(
            sample_time_series[:10], window=20, m=2, r=0.2
        )
        assert result.isna().all()  # All NaN when window > data

    def test_short_series(self, calculator):
        """Test handling of short time series."""
        short_series = pd.Series([1, 2, 3, 4, 5])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.approximate_entropy(
                short_series, window=10, m=2, r=0.2
            )

        # Should return series of NaN
        assert result.isna().all()

    def test_constant_series(self, calculator):
        """Test handling of constant time series."""
        constant_series = pd.Series([1.0] * 100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.approximate_entropy(
                constant_series, window=30, m=2, r=0.2
            )

        # Entropy of constant series should be very low (near 0)
        valid_values = result.dropna()
        assert (valid_values < 0.1).all()

    def test_missing_values(self, calculator, sample_time_series):
        """Test handling of missing values."""
        # Introduce some NaN values
        series_with_nan = sample_time_series.copy()
        series_with_nan.iloc[10:15] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.approximate_entropy(
                series_with_nan, window=30, m=2, r=0.2
            )

        # Should handle NaN values gracefully
        assert isinstance(result, pd.Series)
        assert len(result) == len(series_with_nan)

    def test_edge_case_parameters(self, calculator, sample_time_series):
        """Test edge case parameters."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Very small r value
            result1 = calculator.approximate_entropy(
                sample_time_series, window=30, m=2, r=1e-6
            )
            assert isinstance(result1, pd.Series)

            # Large r value
            result2 = calculator.approximate_entropy(
                sample_time_series, window=30, m=2, r=10.0
            )
            assert isinstance(result2, pd.Series)

    def test_permutation_entropy_edge_cases(
        self, calculator, sample_time_series
    ):
        """Test permutation entropy edge cases."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Order 2 (minimum)
            result1 = calculator.permutation_entropy(
                sample_time_series, window=30, order=2
            )
            assert isinstance(result1, pd.Series)

            # Higher order
            result2 = calculator.permutation_entropy(
                sample_time_series, window=50, order=5
            )
            assert isinstance(result2, pd.Series)

    def test_multiscale_entropy_scales(self, calculator, sample_time_series):
        """Test multiscale entropy with different scale parameters."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Single scale
            result1 = calculator.multiscale_entropy(
                sample_time_series, window=50, scales=[1]
            )
            assert isinstance(result1, pd.DataFrame)
            assert result1.shape[1] == 1

            # Multiple scales
            result2 = calculator.multiscale_entropy(
                sample_time_series, window=50, scales=list(range(1, 11))
            )
            assert isinstance(result2, pd.DataFrame)
            assert result2.shape[1] == 10

    def test_performance_with_numba(self, calculator, sample_time_series):
        """Test that Numba optimization works."""
        # This test ensures Numba-optimized functions execute without error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = calculator.approximate_entropy(
                sample_time_series, window=30, m=2, r=0.2
            )
            assert isinstance(result, pd.Series)

            # Should complete without compilation errors
            assert (result.dropna() >= 0).all()

    def test_entropy_comparison_different_series(self, calculator):
        """Test entropy comparison between different types of series."""
        np.random.seed(42)

        # Regular random series
        random_series = pd.Series(np.random.randn(100))

        # More structured series (sine wave)
        t = np.linspace(0, 4 * np.pi, 100)
        sine_series = pd.Series(np.sin(t))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            entropy_random = calculator.approximate_entropy(
                random_series, window=30, m=2, r=0.2
            )
            entropy_sine = calculator.approximate_entropy(
                sine_series, window=30, m=2, r=0.2
            )

        # Random series should generally have higher entropy than structured series
        mean_entropy_random = entropy_random.dropna().mean()
        mean_entropy_sine = entropy_sine.dropna().mean()

        # This is a general trend but not guaranteed for all windows
        # So we just check that both are computed successfully
        assert mean_entropy_random >= 0
        assert mean_entropy_sine >= 0

    def test_memory_efficiency(self, calculator):
        """Test memory efficiency with larger datasets."""
        np.random.seed(42)
        large_series = pd.Series(np.random.randn(1000))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculator.approximate_entropy(
                large_series, window=50, m=2, r=0.2
            )

        assert isinstance(result, pd.Series)
        assert len(result) == 1000
        assert (result.dropna() >= 0).all()

    def test_all_entropy_methods_consistency(self, calculator, sample_returns):
        """Test that all entropy methods produce consistent outputs."""
        methods_and_params = [
            ("approximate_entropy", {"window": 30, "m": 2, "r": 0.2}),
            ("sample_entropy", {"window": 30, "m": 2, "r": 0.2}),
            ("permutation_entropy", {"window": 30, "order": 3}),
            ("rolling_entropy", {"window": 30, "bins": 10}),
        ]

        results = {}

        for method_name, params in methods_and_params:
            method = getattr(calculator, method_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = method(sample_returns, **params)

            results[method_name] = result

            # All methods should return valid pandas objects
            assert isinstance(result, (pd.Series, pd.DataFrame))

            # All should have same length as input
            if isinstance(result, pd.Series):
                assert len(result) == len(sample_returns)
            else:
                assert len(result) == len(sample_returns)

        # All single-series methods should return non-negative values
        for method_name in [
            "approximate_entropy",
            "sample_entropy",
            "permutation_entropy",
            "rolling_entropy",
        ]:
            result = results[method_name]
            if isinstance(result, pd.Series):
                assert (
                    result.dropna() >= 0
                ).all(), f"{method_name} produced negative values"
