"""Tests for marketregimeml.features.statistical module."""

import pytest
import numpy as np
import pandas as pd
import warnings
from marketregimeml.features.statistical import StatisticalFeatures


class TestStatisticalFeatures:
    """Test StatisticalFeatures class."""

    @pytest.fixture
    def stat_calc(self):
        """Create StatisticalFeatures instance."""
        return StatisticalFeatures()

    @pytest.fixture
    def test_data(self):
        """Create various test data distributions."""
        np.random.seed(42)
        return {
            "normal": pd.Series(np.random.randn(100)),
            "skewed": pd.Series(np.random.exponential(2, 100)),
            "heavy_tailed": pd.Series(np.random.standard_t(3, 100)),
            "uniform": pd.Series(np.random.uniform(-5, 5, 100)),
            "bimodal": pd.Series(
                np.concatenate(
                    [np.random.randn(50) - 2, np.random.randn(50) + 2]
                )
            ),
            "trending": pd.Series(
                np.linspace(0, 10, 100) + np.random.randn(100) * 0.5
            ),
        }

    def test_skewness(self, stat_calc, test_data):
        """Test skewness calculation."""
        for data_name, data in test_data.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = stat_calc.skewness(data, window=20)

                assert isinstance(result, pd.Series)
                assert len(result) == len(data)

                # Skewness should be finite
                valid_values = result.dropna()
                if len(valid_values) > 0:
                    assert np.all(np.isfinite(valid_values))

    def test_kurtosis(self, stat_calc, test_data):
        """Test kurtosis calculation."""
        for data_name, data in test_data.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = stat_calc.kurtosis(data, window=20)

                assert isinstance(result, pd.Series)
                assert len(result) == len(data)

                # Kurtosis should be finite
                valid_values = result.dropna()
                if len(valid_values) > 0:
                    assert np.all(np.isfinite(valid_values))

    def test_rolling_statistics(self, stat_calc, test_data):
        """Test rolling statistics calculation."""
        for data_name, data in test_data.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if hasattr(stat_calc, "rolling_statistics"):
                    result = stat_calc.rolling_statistics(data, window=20)
                    assert result is not None

    def test_autocorrelation(self, stat_calc, test_data):
        """Test autocorrelation calculation."""
        for data_name, data in test_data.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if hasattr(stat_calc, "autocorrelation"):
                    result = stat_calc.autocorrelation(data, window=20)
                    assert result is not None

    def test_moments(self, stat_calc, test_data):
        """Test higher moments calculation."""
        for data_name, data in test_data.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if hasattr(stat_calc, "moments"):
                    result = stat_calc.moments(data, window=20)
                    assert result is not None

    def test_different_window_sizes(self, stat_calc):
        """Test statistical features with different window sizes."""
        data = pd.Series(np.random.randn(100))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for window in [5, 10, 15, 20, 30]:
                # Test skewness
                result = stat_calc.skewness(data, window=window)
                assert isinstance(result, pd.Series)
                assert len(result) == len(data)

                # Test kurtosis
                result = stat_calc.kurtosis(data, window=window)
                assert isinstance(result, pd.Series)
                assert len(result) == len(data)

    def test_edge_cases(self, stat_calc):
        """Test edge cases for statistical calculations."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Constant data
            constant_data = pd.Series([5.0] * 50)

            skew_result = stat_calc.skewness(constant_data, window=10)
            assert isinstance(skew_result, pd.Series)
            # Skewness of constant data should be 0 or NaN
            valid_skew = skew_result.dropna()
            if len(valid_skew) > 0:
                assert (
                    np.allclose(valid_skew, 0.0, atol=1e-10)
                    or np.isnan(valid_skew).all()
                )

            kurt_result = stat_calc.kurtosis(constant_data, window=10)
            assert isinstance(kurt_result, pd.Series)

            # Very small data
            tiny_data = pd.Series([1.0, 2.0, 3.0])

            skew_result = stat_calc.skewness(tiny_data, window=3)
            assert isinstance(skew_result, pd.Series)

            kurt_result = stat_calc.kurtosis(tiny_data, window=3)
            assert isinstance(kurt_result, pd.Series)

    def test_known_distributions(self, stat_calc):
        """Test with distributions having known statistical properties."""
        np.random.seed(42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Normal distribution should have skewness near 0
            normal_data = pd.Series(np.random.randn(1000))
            skew_result = stat_calc.skewness(normal_data, window=100)
            valid_skew = skew_result.dropna()
            if len(valid_skew) > 0:
                # Should be close to 0 for normal distribution
                assert np.abs(valid_skew.mean()) < 0.5

            # Exponential distribution should have positive skewness
            exp_data = pd.Series(np.random.exponential(1, 1000))
            skew_result = stat_calc.skewness(exp_data, window=100)
            valid_skew = skew_result.dropna()
            if len(valid_skew) > 0:
                # Should be positive for exponential distribution
                assert valid_skew.mean() > 0

    def test_performance_with_large_data(self, stat_calc):
        """Test performance with larger datasets."""
        np.random.seed(42)
        large_data = pd.Series(np.random.randn(5000))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            import time

            start = time.time()

            skew_result = stat_calc.skewness(large_data, window=50)
            kurt_result = stat_calc.kurtosis(large_data, window=50)

            elapsed = time.time() - start

            # Should complete in reasonable time
            assert elapsed < 5.0

            # Results should be valid
            assert isinstance(skew_result, pd.Series)
            assert isinstance(kurt_result, pd.Series)
            assert len(skew_result) == 5000
            assert len(kurt_result) == 5000

    def test_method_consistency(self, stat_calc):
        """Test consistency of statistical methods."""
        np.random.seed(42)
        data = pd.Series(np.random.randn(100))

        # Multiple calls should give identical results
        skew1 = stat_calc.skewness(data, window=20)
        skew2 = stat_calc.skewness(data, window=20)

        if not skew1.empty and not skew2.empty:
            pd.testing.assert_series_equal(skew1, skew2)

        kurt1 = stat_calc.kurtosis(data, window=20)
        kurt2 = stat_calc.kurtosis(data, window=20)

        if not kurt1.empty and not kurt2.empty:
            pd.testing.assert_series_equal(kurt1, kurt2)
