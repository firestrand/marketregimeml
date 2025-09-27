"""Extended tests for feature engine to improve coverage."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from marketregimeml.features.engine import FeatureEngine
from marketregimeml.features.technical import TechnicalIndicators
from marketregimeml.features.volatility import VolatilityFeatures
from marketregimeml.features.statistical import StatisticalFeatures


class TestFeatureEngineExtended:
    """Extended test suite for FeatureEngine."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        close = 100 + np.random.randn(100).cumsum()
        data = pd.DataFrame(
            {
                "open": close + np.random.randn(100) * 0.5,
                "high": close + np.abs(np.random.randn(100)) * 1.5,
                "low": close - np.abs(np.random.randn(100)) * 1.5,
                "close": close,
                "volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )
        return data

    @pytest.fixture
    def feature_engine(self):
        """Create feature engine instance."""
        config = {
            "feature_sets": {
                "returns": {"enabled": True, "periods": [1, 5, 20]},
                "volatility": {"enabled": True, "windows": [10, 20]},
                "technical": {
                    "enabled": True,
                    "indicators": ["rsi", "macd", "bollinger_bands"],
                },
            }
        }
        return FeatureEngine(config)

    def test_compute_return_features(self, feature_engine, sample_ohlcv):
        """Test return feature computation."""
        features = feature_engine._compute_return_features(
            sample_ohlcv, periods=[1, 5, 20]
        )

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv)
        assert "returns_1" in features.columns
        assert "returns_5" in features.columns
        assert "returns_20" in features.columns

        # Check that returns are calculated correctly
        expected_1 = sample_ohlcv["close"].pct_change(1)
        pd.testing.assert_series_equal(
            features["returns_1"], expected_1, check_names=False
        )

    def test_compute_volatility_features(self, feature_engine, sample_ohlcv):
        """Test volatility feature computation."""
        features = feature_engine._compute_volatility_features(
            sample_ohlcv, windows=[10, 20]
        )

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv)

        # Should have volatility columns for each window
        expected_cols = ["volatility_10", "volatility_20"]
        for col in expected_cols:
            assert col in features.columns
            # Volatility should be positive where not NaN
            valid = features[col].dropna()
            assert (valid >= 0).all()

    def test_compute_technical_features_rsi(self, feature_engine, sample_ohlcv):
        """Test technical features computation with RSI."""
        features = feature_engine._compute_technical_features(
            sample_ohlcv, indicators=["rsi"], params={"rsi_window": 14}
        )

        assert isinstance(features, pd.DataFrame)
        assert "rsi_14" in features.columns

        # RSI should be between 0 and 100
        valid_rsi = features["rsi_14"].dropna()
        assert ((valid_rsi >= 0) & (valid_rsi <= 100)).all()

    def test_compute_technical_features_macd(self, feature_engine, sample_ohlcv):
        """Test technical features computation with MACD."""
        features = feature_engine._compute_technical_features(
            sample_ohlcv,
            indicators=["macd"],
            params={"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
        )

        assert isinstance(features, pd.DataFrame)
        assert "macd" in features.columns
        assert "macd_signal" in features.columns
        assert "macd_histogram" in features.columns

        # Histogram should be difference between MACD and signal
        valid_idx = ~features["macd"].isna() & ~features["macd_signal"].isna()
        if valid_idx.any():
            np.testing.assert_array_almost_equal(
                features.loc[valid_idx, "macd_histogram"].values,
                (
                    features.loc[valid_idx, "macd"]
                    - features.loc[valid_idx, "macd_signal"]
                ).values,
                decimal=10,
            )

    def test_compute_technical_features_bollinger(self, feature_engine, sample_ohlcv):
        """Test technical features computation with Bollinger Bands."""
        features = feature_engine._compute_technical_features(
            sample_ohlcv,
            indicators=["bollinger_bands"],
            params={"bb_window": 20, "bb_std": 2},
        )

        assert isinstance(features, pd.DataFrame)
        assert "bb_upper" in features.columns
        assert "bb_middle" in features.columns
        assert "bb_lower" in features.columns

        # Check band ordering
        valid_idx = ~features["bb_middle"].isna()
        if valid_idx.any():
            assert (
                features.loc[valid_idx, "bb_upper"]
                >= features.loc[valid_idx, "bb_middle"]
            ).all()
            assert (
                features.loc[valid_idx, "bb_middle"]
                >= features.loc[valid_idx, "bb_lower"]
            ).all()

    def test_compute_technical_features_sma(self, feature_engine, sample_ohlcv):
        """Test technical features computation with SMA."""
        features = feature_engine._compute_technical_features(
            sample_ohlcv, indicators=["sma"], params={"sma_window": 20}
        )

        assert isinstance(features, pd.DataFrame)
        assert "sma_20" in features.columns

        # SMA should be the mean of the window
        sma = features["sma_20"]
        expected = sample_ohlcv["close"].rolling(20).mean()
        pd.testing.assert_series_equal(sma, expected, check_names=False)

    def test_compute_technical_features_ema(self, feature_engine, sample_ohlcv):
        """Test technical features computation with EMA."""
        features = feature_engine._compute_technical_features(
            sample_ohlcv, indicators=["ema"], params={"ema_window": 20}
        )

        assert isinstance(features, pd.DataFrame)
        assert "ema_20" in features.columns

        # EMA should exist and be numeric
        ema = features["ema_20"]
        assert ema.dtype in [np.float64, np.float32]

    def test_handle_missing_values_forward_fill(self, feature_engine):
        """Test missing value handling with forward fill."""
        data = pd.DataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [np.nan, 2, np.nan, 4, 5]}
        )

        result = feature_engine._handle_missing_values(data, method="forward_fill")

        assert result.loc[1, "a"] == 1  # Forward filled
        assert result.loc[3, "a"] == 3  # Forward filled
        assert pd.isna(result.loc[0, "b"])  # Still NaN (no previous value)
        assert result.loc[2, "b"] == 2  # Forward filled

    def test_handle_missing_values_backward_fill(self, feature_engine):
        """Test missing value handling with backward fill."""
        data = pd.DataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [np.nan, 2, np.nan, 4, 5]}
        )

        result = feature_engine._handle_missing_values(data, method="backward_fill")

        assert result.loc[1, "a"] == 3  # Backward filled
        assert result.loc[3, "a"] == 5  # Backward filled
        assert result.loc[0, "b"] == 2  # Backward filled
        assert result.loc[2, "b"] == 4  # Backward filled

    def test_handle_missing_values_interpolate(self, feature_engine):
        """Test missing value handling with interpolation."""
        data = pd.DataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [1, np.nan, np.nan, np.nan, 5]}
        )

        result = feature_engine._handle_missing_values(data, method="interpolate")

        assert result.loc[1, "a"] == 2  # Interpolated
        assert result.loc[3, "a"] == 4  # Interpolated
        # Check linear interpolation for column b
        assert result.loc[1, "b"] == 2
        assert result.loc[2, "b"] == 3
        assert result.loc[3, "b"] == 4

    def test_handle_missing_values_drop(self, feature_engine):
        """Test missing value handling with drop."""
        data = pd.DataFrame({"a": [1, np.nan, 3, np.nan, 5], "b": [1, 2, 3, 4, 5]})

        result = feature_engine._handle_missing_values(data, method="drop")

        assert len(result) == 3  # Dropped 2 rows with NaN
        assert list(result.index) == [0, 2, 4]

    def test_handle_missing_values_mean(self, feature_engine):
        """Test missing value handling with mean."""
        data = pd.DataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [10, np.nan, 30, 40, 50]}
        )

        result = feature_engine._handle_missing_values(data, method="mean")

        # Mean of [1, 3, 5] = 3
        assert result.loc[1, "a"] == 3
        assert result.loc[3, "a"] == 3
        # Mean of [10, 30, 40, 50] = 32.5
        assert result.loc[1, "b"] == 32.5

    def test_handle_missing_values_median(self, feature_engine):
        """Test missing value handling with median."""
        data = pd.DataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [10, np.nan, 30, 40, 50]}
        )

        result = feature_engine._handle_missing_values(data, method="median")

        # Median of [1, 3, 5] = 3
        assert result.loc[1, "a"] == 3
        assert result.loc[3, "a"] == 3
        # Median of [10, 30, 40, 50] = 35
        assert result.loc[1, "b"] == 35

    def test_handle_missing_values_unknown(self, feature_engine):
        """Test missing value handling with unknown method."""
        data = pd.DataFrame(
            {
                "a": [1, np.nan, 3],
            }
        )

        # Should fall back to forward fill
        with patch("marketregimeml.features.engine.logger") as mock_logger:
            result = feature_engine._handle_missing_values(
                data, method="unknown_method"
            )
            mock_logger.warning.assert_called_once()
            assert result.loc[1, "a"] == 1  # Forward filled

    def test_normalize_features_standard(self, feature_engine):
        """Test feature normalization with standard scaling."""
        features = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "close": [100, 101, 102, 103, 104],  # Should be excluded
            }
        )

        normalized = feature_engine.normalize_features(features, method="standard")

        # Check that close is preserved
        pd.testing.assert_series_equal(normalized["close"], features["close"])

        # Check that features are normalized (mean ~0, std ~1)
        assert abs(normalized["feature1"].mean()) < 1e-10
        assert abs(normalized["feature1"].std() - 1) < 1e-10

    def test_normalize_features_minmax(self, feature_engine):
        """Test feature normalization with min-max scaling."""
        features = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "close": [100, 101, 102, 103, 104],
            }
        )

        normalized = feature_engine.normalize_features(features, method="minmax")

        # Check that features are in [0, 1] range
        assert normalized["feature1"].min() == 0
        assert normalized["feature1"].max() == 1
        assert normalized["feature2"].min() == 0
        assert normalized["feature2"].max() == 1

    def test_normalize_features_robust(self, feature_engine):
        """Test feature normalization with robust scaling."""
        features = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 100],  # Has outlier
                "feature2": [10, 20, 30, 40, 50],
                "close": [100, 101, 102, 103, 104],
            }
        )

        normalized = feature_engine.normalize_features(features, method="robust")

        # Robust scaler should handle outliers better
        # Median should be close to 0
        assert abs(normalized["feature2"].median()) < 0.5

    def test_normalize_features_exclude_cols(self, feature_engine):
        """Test feature normalization with custom excluded columns."""
        features = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "keep_me": [100, 200, 300, 400, 500],
            }
        )

        normalized = feature_engine.normalize_features(
            features, method="standard", exclude_cols=["keep_me"]
        )

        # Check that excluded column is preserved
        pd.testing.assert_series_equal(normalized["keep_me"], features["keep_me"])

        # Check that other features are normalized
        assert abs(normalized["feature1"].mean()) < 1e-10

    def test_select_features(self, feature_engine):
        """Test feature selection."""
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "feature4": np.ones(100),  # Constant feature
                "feature5": np.random.randn(100),
            }
        )

        # Mock target variable
        target = pd.Series(np.random.randint(0, 2, 100))

        selected = feature_engine.select_features(
            features, target=target, method="variance", k=3
        )

        # Should drop constant feature
        assert "feature4" not in selected.columns
        # Should keep k features
        assert len(selected.columns) <= 3

    def test_select_features_correlation(self, feature_engine):
        """Test feature selection with correlation method."""
        # Create correlated features
        base = np.random.randn(100)
        features = pd.DataFrame(
            {
                "feature1": base + np.random.randn(100) * 0.1,
                "feature2": base
                + np.random.randn(100) * 0.1,  # Highly correlated with feature1
                "feature3": np.random.randn(100),  # Independent
                "feature4": np.random.randn(100),  # Independent
            }
        )

        selected = feature_engine.select_features(
            features, method="correlation", threshold=0.9
        )

        # Should drop one of the highly correlated features
        assert len(selected.columns) < 4

    def test_compute_features_integration(self, feature_engine, sample_ohlcv):
        """Test full feature computation pipeline."""
        config = {
            "returns": {"periods": [1, 5]},
            "volatility": {"windows": [10, 20]},
            "technical": {"indicators": ["rsi"], "params": {"rsi_window": 14}},
        }

        features = feature_engine.compute_features(sample_ohlcv, config)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv)

        # Check that all feature types are included
        assert "returns_1" in features.columns
        assert "volatility_10" in features.columns
        assert "rsi_14" in features.columns

    def test_compute_features_with_caching(self, feature_engine, sample_ohlcv):
        """Test feature computation with caching."""
        feature_engine.cache_enabled = True

        # First call - should compute
        features1 = feature_engine.compute_features(sample_ohlcv)

        # Second call with same data - should use cache
        features2 = feature_engine.compute_features(sample_ohlcv)

        pd.testing.assert_frame_equal(features1, features2)

    def test_compute_features_empty_data(self, feature_engine):
        """Test feature computation with empty data."""
        empty_data = pd.DataFrame()
        features = feature_engine.compute_features(empty_data)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == 0

    def test_compute_features_missing_columns(self, feature_engine):
        """Test feature computation with missing required columns."""
        incomplete_data = pd.DataFrame({"close": [100, 101, 102]})

        # Should handle gracefully and compute what it can
        features = feature_engine.compute_features(incomplete_data)
        assert isinstance(features, pd.DataFrame)
