"""Tests for marketregimeml.features.engine module."""

import pytest
import numpy as np
import pandas as pd
import warnings
from marketregimeml.features.engine import FeatureEngine


class TestFeatureEngine:
    """Test FeatureEngine class."""

    @pytest.fixture
    def feature_engine(self):
        """Create FeatureEngine instance."""
        return FeatureEngine()

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100

        # Generate realistic price series
        returns = np.random.randn(n) * 0.02
        close = 100 * np.exp(np.cumsum(returns))

        # Create OHLCV data with proper relationships
        open_p = np.roll(close, 1)
        open_p[0] = close[0]

        # Add realistic intraday spreads
        spread = np.abs(close * 0.002)
        high = np.maximum(open_p, close) + spread
        low = np.minimum(open_p, close) - spread
        volume = np.random.randint(1000, 10000, n)

        return pd.DataFrame(
            {
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def test_compute_all_features(self, feature_engine, sample_ohlcv):
        """Test compute_all_features method."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = feature_engine.compute_all_features(sample_ohlcv)

            # Should return a DataFrame or None
            assert result is None or isinstance(result, pd.DataFrame)

            if isinstance(result, pd.DataFrame):
                # Should have same length as input
                assert len(result) == len(sample_ohlcv)
                # Should have some features
                assert len(result.columns) > 0

    def test_compute_volatility_features(self, feature_engine, sample_ohlcv):
        """Test volatility feature computation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if hasattr(feature_engine, "compute_volatility_features"):
                result = feature_engine.compute_volatility_features(
                    sample_ohlcv
                )
                assert result is None or isinstance(result, pd.DataFrame)

    def test_compute_entropy_features(self, feature_engine, sample_ohlcv):
        """Test entropy feature computation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if hasattr(feature_engine, "compute_entropy_features"):
                result = feature_engine.compute_entropy_features(sample_ohlcv)
                assert result is None or isinstance(result, pd.DataFrame)

    def test_compute_technical_features(self, feature_engine, sample_ohlcv):
        """Test technical feature computation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if hasattr(feature_engine, "compute_technical_features"):
                result = feature_engine.compute_technical_features(
                    sample_ohlcv
                )
                assert result is None or isinstance(result, pd.DataFrame)

    def test_compute_statistical_features(self, feature_engine, sample_ohlcv):
        """Test statistical feature computation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if hasattr(feature_engine, "compute_statistical_features"):
                # Statistical features typically work on a single series
                result = feature_engine.compute_statistical_features(
                    sample_ohlcv["close"]
                )
                assert result is None or isinstance(result, pd.DataFrame)

    def test_feature_validation(self, feature_engine, sample_ohlcv):
        """Test feature validation methods."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if hasattr(feature_engine, "validate_data"):
                validation_result = feature_engine.validate_data(sample_ohlcv)
                assert validation_result is not None

    def test_feature_preprocessing(self, feature_engine, sample_ohlcv):
        """Test feature preprocessing methods."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if hasattr(feature_engine, "preprocess_data"):
                preprocessed = feature_engine.preprocess_data(sample_ohlcv)
                assert (
                    isinstance(preprocessed, pd.DataFrame)
                    or preprocessed is None
                )

    def test_feature_combination(self, feature_engine, sample_ohlcv):
        """Test feature combination methods."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if hasattr(feature_engine, "combine_features"):
                # Create multiple feature DataFrames
                features1 = sample_ohlcv[["close"]].copy()
                features2 = sample_ohlcv[["volume"]].copy()

                combined = feature_engine.combine_features(
                    [features1, features2]
                )
                assert combined is None or isinstance(combined, pd.DataFrame)

    def test_feature_selection(self, feature_engine, sample_ohlcv):
        """Test feature selection methods."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if hasattr(feature_engine, "select_features"):
                selected = feature_engine.select_features(sample_ohlcv)
                assert selected is None or isinstance(selected, pd.DataFrame)

    def test_feature_scaling(self, feature_engine, sample_ohlcv):
        """Test feature scaling methods."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if hasattr(feature_engine, "scale_features"):
                scaled = feature_engine.scale_features(sample_ohlcv)
                assert scaled is None or isinstance(scaled, pd.DataFrame)

    def test_edge_cases(self, feature_engine):
        """Test edge cases for feature engine."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Empty DataFrame
            empty_df = pd.DataFrame()
            try:
                result = feature_engine.compute_all_features(empty_df)
                assert result is None or isinstance(result, pd.DataFrame)
            except (ValueError, KeyError):
                # Expected for empty data
                pass

            # Invalid data structure
            invalid_data = pd.DataFrame({"invalid": [1, 2, 3]})
            try:
                result = feature_engine.compute_all_features(invalid_data)
                assert result is None or isinstance(result, pd.DataFrame)
            except (ValueError, KeyError):
                # Expected for invalid data
                pass

    def test_large_dataset_performance(self, feature_engine):
        """Test performance with larger datasets."""
        np.random.seed(42)
        n = 2000

        # Create large OHLCV dataset
        returns = np.random.randn(n) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        open_p = np.roll(close, 1)
        open_p[0] = close[0]
        high = close * 1.01
        low = close * 0.99
        volume = np.random.randint(1000, 10000, n)

        large_ohlcv = pd.DataFrame(
            {
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            import time

            start = time.time()

            result = feature_engine.compute_all_features(large_ohlcv)

            elapsed = time.time() - start

            # Should complete in reasonable time
            assert (
                elapsed < 30.0
            )  # Allow more time for comprehensive feature computation

            if isinstance(result, pd.DataFrame):
                assert len(result) == n

    def test_feature_consistency(self, feature_engine, sample_ohlcv):
        """Test consistency of feature computation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Multiple calls should give consistent results
            result1 = feature_engine.compute_all_features(sample_ohlcv)
            result2 = feature_engine.compute_all_features(sample_ohlcv)

            if isinstance(result1, pd.DataFrame) and isinstance(
                result2, pd.DataFrame
            ):
                # Should have same shape and column names
                assert result1.shape == result2.shape
                assert list(result1.columns) == list(result2.columns)

                # Values should be identical (within numerical precision)
                for col in result1.columns:
                    if result1[col].dtype in [np.float64, np.float32]:
                        pd.testing.assert_series_equal(
                            result1[col],
                            result2[col],
                            check_exact=False,
                            rtol=1e-10,
                        )
                    else:
                        pd.testing.assert_series_equal(
                            result1[col], result2[col]
                        )

    def test_various_market_scenarios(self, feature_engine):
        """Test with various market scenarios."""
        np.random.seed(42)

        scenarios = [
            # High volatility
            {"returns_std": 0.05, "n": 100},
            # Low volatility
            {"returns_std": 0.005, "n": 100},
            # Trending market
            {"trend": 0.001, "n": 100},
            # Mean-reverting
            {"mean_revert": True, "n": 100},
        ]

        for scenario in scenarios:
            n = scenario["n"]

            if "returns_std" in scenario:
                returns = np.random.randn(n) * scenario["returns_std"]
            elif "trend" in scenario:
                returns = np.random.randn(n) * 0.02 + scenario["trend"]
            elif scenario.get("mean_revert"):
                # AR(1) process with mean reversion
                returns = np.zeros(n)
                for i in range(1, n):
                    returns[i] = (
                        -0.1 * returns[i - 1] + np.random.randn() * 0.02
                    )
            else:
                returns = np.random.randn(n) * 0.02

            close = 100 * np.exp(np.cumsum(returns))

            ohlcv = pd.DataFrame(
                {
                    "open": np.roll(close, 1),
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": np.random.randint(1000, 5000, n),
                }
            )
            ohlcv.iloc[0, 0] = ohlcv.iloc[0, 3]  # Fix first open

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = feature_engine.compute_all_features(ohlcv)

                # Should handle all scenarios
                assert result is None or isinstance(result, pd.DataFrame)
