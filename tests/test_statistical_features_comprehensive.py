"""Comprehensive tests for statistical features to improve coverage."""

import pytest
import numpy as np
import pandas as pd
from marketregimeml.features.statistical import StatisticalFeatures


class TestStatisticalFeaturesComprehensive:
    """Comprehensive tests for statistical features."""

    @pytest.fixture
    def stat_features(self):
        """Create StatisticalFeatures instance."""
        return StatisticalFeatures()

    @pytest.fixture
    def sample_series(self):
        """Create sample time series data."""
        np.random.seed(42)
        return pd.Series(np.random.randn(100))

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "returns": np.random.randn(100),
                "volume": np.random.exponential(1, 100),
                "volatility": np.abs(np.random.randn(100)),
            }
        )

    def test_rolling_mean(self, stat_features, sample_series):
        """Test rolling mean calculation."""
        result = stat_features.rolling_mean(sample_series, window=10)
        assert len(result) == len(sample_series)
        assert not result.isna().all()
        # First window-1 values should be NaN
        assert result.iloc[:9].isna().all()
        assert not result.iloc[9:].isna().all()

    def test_rolling_std(self, stat_features, sample_series):
        """Test rolling standard deviation."""
        result = stat_features.rolling_std(sample_series, window=10)
        assert len(result) == len(sample_series)
        # Std should be positive
        valid_values = result.dropna()
        assert (valid_values >= 0).all()

    def test_rolling_skewness(self, stat_features, sample_series):
        """Test rolling skewness calculation."""
        result = stat_features.rolling_skewness(sample_series, window=20)
        assert len(result) == len(sample_series)
        # Skewness can be positive or negative
        valid_values = result.dropna()
        assert len(valid_values) > 0

    def test_rolling_kurtosis(self, stat_features, sample_series):
        """Test rolling kurtosis calculation."""
        result = stat_features.rolling_kurtosis(sample_series, window=20)
        assert len(result) == len(sample_series)
        valid_values = result.dropna()
        assert len(valid_values) > 0

    def test_rolling_min_max(self, stat_features, sample_series):
        """Test rolling min and max."""
        min_result = stat_features.rolling_min(sample_series, window=10)
        max_result = stat_features.rolling_max(sample_series, window=10)

        assert len(min_result) == len(sample_series)
        assert len(max_result) == len(sample_series)

        # Min should be <= Max
        valid_idx = ~(min_result.isna() | max_result.isna())
        assert (min_result[valid_idx] <= max_result[valid_idx]).all()

    def test_rolling_quantile(self, stat_features, sample_series):
        """Test rolling quantile calculation."""
        # Test median (50th percentile)
        result = stat_features.rolling_quantile(
            sample_series, window=10, quantile=0.5
        )
        assert len(result) == len(sample_series)

        # Test 25th and 75th percentiles
        q25 = stat_features.rolling_quantile(
            sample_series, window=10, quantile=0.25
        )
        q75 = stat_features.rolling_quantile(
            sample_series, window=10, quantile=0.75
        )

        valid_idx = ~(q25.isna() | q75.isna())
        assert (q25[valid_idx] <= q75[valid_idx]).all()

    def test_autocorrelation_basic(self, stat_features, sample_series):
        """Test autocorrelation calculation."""
        result = stat_features.autocorrelation(sample_series, window=20, lag=1)
        assert len(result) == len(sample_series)

        # Autocorrelation should be between -1 and 1
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert ((valid_values >= -1) & (valid_values <= 1)).all()

    def test_rolling_correlation(self, stat_features, sample_dataframe):
        """Test rolling correlation."""
        result = stat_features.rolling_correlation(
            sample_dataframe["returns"], sample_dataframe["volume"], window=20
        )
        assert len(result) == len(sample_dataframe)

        # Correlation should be between -1 and 1
        valid_values = result.dropna()
        assert ((valid_values >= -1) & (valid_values <= 1)).all()

    def test_partial_autocorrelation(self, stat_features, sample_series):
        """Test partial autocorrelation calculation."""
        result = stat_features.partial_autocorrelation(
            sample_series, window=50, lag=1
        )
        assert len(result) == len(sample_series)

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert ((valid_values >= -1) & (valid_values <= 1)).all()

    def test_jarque_bera_stat(self, stat_features, sample_series):
        """Test Jarque-Bera test statistic."""
        try:
            result = stat_features.jarque_bera_stat(sample_series, window=20)
            assert len(result) == len(sample_series)

            valid_values = result.dropna()
            if len(valid_values) > 0:
                assert (
                    valid_values >= 0
                ).all()  # JB statistic is non-negative
        except (ValueError, TypeError):
            # Handle scipy version compatibility issues
            pytest.skip("Scipy version compatibility issue with jarque_bera")

    def test_shapiro_wilk_stat(self, stat_features, sample_series):
        """Test Shapiro-Wilk test statistic."""
        result = stat_features.shapiro_wilk_stat(sample_series, window=20)
        assert len(result) == len(sample_series)

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert ((valid_values >= 0) & (valid_values <= 1)).all()

    def test_cross_correlation(self, stat_features, sample_dataframe):
        """Test cross-correlation between series."""
        result = stat_features.cross_correlation(
            sample_dataframe["returns"],
            sample_dataframe["volume"],
            window=20,
            lag=0,
        )
        assert len(result) == len(sample_dataframe)

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert ((valid_values >= -1) & (valid_values <= 1)).all()

    def test_hurst_exponent(self, stat_features, sample_series):
        """Test Hurst exponent calculation."""
        result = stat_features.hurst_exponent(sample_series, window=80)
        assert len(result) == len(sample_series)

        valid_values = result.dropna()
        if len(valid_values) > 0:
            # Hurst should be between 0 and 1
            assert ((valid_values > 0) & (valid_values < 1)).all()

    def test_adf_test(self, stat_features, sample_series):
        """Test ADF test statistic."""
        result = stat_features.adf_test(sample_series, window=80)
        assert len(result) == len(sample_series)

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(isinstance(x, (int, float)) for x in valid_values)

    def test_information_ratio(self, stat_features, sample_dataframe):
        """Test information ratio calculation."""
        result = stat_features.information_ratio(
            sample_dataframe["returns"],
            sample_dataframe["volatility"],
            window=50,
        )
        assert len(result) == len(sample_dataframe)

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(isinstance(x, (int, float)) for x in valid_values)

    def test_rolling_statistics_comprehensive(
        self, stat_features, sample_series
    ):
        """Test comprehensive rolling statistics."""
        result = stat_features.rolling_statistics(sample_series, window=20)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_series)

        expected_columns = [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "skewness",
            "kurtosis",
        ]
        assert all(col in result.columns for col in expected_columns)

    def test_regime_stability(self, stat_features):
        """Test regime stability calculation."""
        regimes = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0] * 10)
        result = stat_features.regime_stability(regimes, window=20)
        assert len(result) == len(regimes)

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert ((valid_values >= 0) & (valid_values <= 1)).all()

    def test_shannon_entropy(self, stat_features, sample_series):
        """Test Shannon entropy calculation."""
        result = stat_features.shannon_entropy(
            sample_series, window=50, bins=10
        )
        assert len(result) == len(sample_series)

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert (valid_values >= 0).all()  # Entropy is non-negative

    def test_entropy_methods(self, stat_features, sample_series):
        """Test various entropy methods."""
        # Test approximate entropy
        approx_result = stat_features.approximate_entropy(
            sample_series, window=50, m=2, r=0.2
        )
        assert len(approx_result) == len(sample_series)

        # Test sample entropy
        sample_result = stat_features.sample_entropy(
            sample_series, window=50, m=2, r=0.2
        )
        assert len(sample_result) == len(sample_series)

        # Test permutation entropy
        perm_result = stat_features.permutation_entropy(
            sample_series, window=50, order=3
        )
        assert len(perm_result) == len(sample_series)

        # All entropy measures should be non-negative
        for result in [approx_result, sample_result, perm_result]:
            valid_values = result.dropna()
            if len(valid_values) > 0:
                # Allow for inf values in sample entropy
                finite_values = valid_values[np.isfinite(valid_values)]
                if len(finite_values) > 0:
                    assert (finite_values >= 0).all()

    def test_different_window_sizes(self, stat_features, sample_series):
        """Test with different window sizes."""
        for window in [5, 10, 20, 50]:
            result = stat_features.rolling_mean(sample_series, window=window)
            assert len(result) == len(sample_series)
            # Check NaN pattern
            assert result.iloc[: window - 1].isna().all()
            if window < len(sample_series):
                assert not result.iloc[window:].isna().all()

    def test_edge_case_small_data(self, stat_features):
        """Test with very small dataset."""
        small_series = pd.Series([1, 2, 3, 4, 5])
        result = stat_features.rolling_mean(small_series, window=3)
        assert len(result) == 5
        assert result.iloc[2] == 2.0  # Mean of [1,2,3]

    def test_edge_case_constant_values(self, stat_features):
        """Test with constant values."""
        const_series = pd.Series([5.0] * 50)

        # Mean should be constant
        mean = stat_features.rolling_mean(const_series, window=10)
        assert (mean.dropna() == 5.0).all()

        # Std should be 0
        std = stat_features.rolling_std(const_series, window=10)
        assert (std.dropna() == 0.0).all()

    def test_edge_case_missing_values(self, stat_features):
        """Test handling of missing values."""
        series_with_nan = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10])
        result = stat_features.rolling_mean(series_with_nan, window=3)
        assert len(result) == len(series_with_nan)
        # Should handle NaN appropriately
        assert not result.isna().all()

    def test_performance_large_data(self, stat_features):
        """Test performance with larger dataset."""
        large_series = pd.Series(np.random.randn(10000))
        result = stat_features.rolling_mean(large_series, window=100)
        assert len(result) == 10000
        assert not result.iloc[100:].isna().any()

    def test_correlation_with_self(self, stat_features, sample_series):
        """Test correlation of series with itself."""
        result = stat_features.rolling_correlation(
            sample_series, sample_series, window=20
        )
        # Correlation with self should be 1
        valid_values = result.dropna()
        assert np.allclose(valid_values, 1.0)

    def test_quantile_extremes(self, stat_features, sample_series):
        """Test quantile at extremes."""
        q0 = stat_features.rolling_quantile(
            sample_series, window=10, quantile=0.0
        )
        q100 = stat_features.rolling_quantile(
            sample_series, window=10, quantile=1.0
        )
        min_vals = stat_features.rolling_min(sample_series, window=10)
        max_vals = stat_features.rolling_max(sample_series, window=10)

        valid_idx = ~q0.isna()
        assert np.allclose(q0[valid_idx], min_vals[valid_idx])
        assert np.allclose(q100[valid_idx], max_vals[valid_idx])
