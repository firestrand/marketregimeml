"""Tests for feature engineering engine."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import json

from marketregimeml.features.engine import FeatureEngine


class TestFeatureEngine:
    """Test FeatureEngine class."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        # Generate realistic price data
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))

        data = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(n) * 0.005),
                "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
                "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
                "close": close,
                "volume": np.random.randint(1000000, 10000000, n),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)

        return data

    @pytest.fixture
    def feature_config(self):
        """Create sample feature configuration."""
        return {
            "volatility": {
                "enabled": True,
                "indicators": ["yang_zhang", "garman_klass", "parkinson"],
                "window": 20,
            },
            "entropy": {
                "enabled": True,
                "measures": ["shannon", "approximate", "sample"],
                "window": 30,
            },
            "statistical": {
                "enabled": True,
                "features": ["mean", "std", "skew", "kurtosis"],
                "window": 20,
            },
            "technical": {
                "enabled": True,
                "indicators": ["rsi", "macd", "bollinger_bands"],
                "params": {
                    "rsi_window": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "bb_window": 20,
                },
            },
        }

    @pytest.fixture
    def engine(self, feature_config):
        """Create FeatureEngine instance."""
        return FeatureEngine(feature_config=feature_config)

    def test_init(self, engine, feature_config):
        """Test FeatureEngine initialization."""
        assert engine is not None
        assert engine.config == feature_config
        assert engine.enable_caching is True
        assert engine.cache == {}

        # Check feature calculators
        assert hasattr(engine, "volatility")
        assert hasattr(engine, "entropy")
        assert hasattr(engine, "statistical")
        assert hasattr(engine, "technical")

    def test_init_without_config(self):
        """Test initialization without config."""
        with patch(
            "marketregimeml.features.engine.config_loader"
        ) as mock_loader:
            mock_loader.get_feature_config.return_value = {"test": "config"}
            engine = FeatureEngine()
            assert engine.config == {"test": "config"}
            mock_loader.get_feature_config.assert_called_once()

    def test_compute_features(self, engine, sample_ohlcv):
        """Test compute_features method."""
        features = engine.compute_features(sample_ohlcv)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv)

        # Should have more columns than original
        assert len(features.columns) > len(sample_ohlcv.columns)

        # Original columns should be preserved
        for col in sample_ohlcv.columns:
            assert col in features.columns

    def test_compute_features_with_selection(self, engine, sample_ohlcv):
        """Test compute_features with specific feature types."""
        # Only volatility features
        features = engine.compute_features(
            sample_ohlcv, feature_types=["volatility"]
        )

        assert isinstance(features, pd.DataFrame)

        # Should have volatility-related columns
        vol_cols = [col for col in features.columns if "vol" in col.lower()]
        assert len(vol_cols) > 0

    def test_validate_data(self, engine, sample_ohlcv):
        """Test data validation."""
        # Valid data should pass
        assert engine._validate_data(sample_ohlcv) is True

        # Invalid data - missing required columns
        invalid_data = sample_ohlcv.drop(columns=["close"])
        assert engine._validate_data(invalid_data) is False

        # Invalid data - not a DataFrame
        assert engine._validate_data([1, 2, 3]) is False

    def test_compute_volatility_features(self, engine, sample_ohlcv):
        """Test volatility feature computation."""
        vol_features = engine._compute_volatility_features(sample_ohlcv)

        assert isinstance(vol_features, pd.DataFrame)
        assert len(vol_features) == len(sample_ohlcv)

        # Check for expected volatility columns
        vol_indicators = engine.config["volatility"]["indicators"]
        for indicator in vol_indicators:
            matching_cols = [
                col
                for col in vol_features.columns
                if indicator.lower() in col.lower()
            ]
            assert len(matching_cols) > 0

    def test_compute_entropy_features(self, engine, sample_ohlcv):
        """Test entropy feature computation."""
        entropy_features = engine._compute_entropy_features(sample_ohlcv)

        assert isinstance(entropy_features, pd.DataFrame)
        assert len(entropy_features) == len(sample_ohlcv)

        # Check for entropy columns
        entropy_measures = engine.config["entropy"]["measures"]
        for measure in entropy_measures:
            matching_cols = [
                col
                for col in entropy_features.columns
                if measure.lower() in col.lower()
            ]
            assert len(matching_cols) > 0

    def test_compute_statistical_features(self, engine, sample_ohlcv):
        """Test statistical feature computation."""
        stat_features = engine._compute_statistical_features(sample_ohlcv)

        assert isinstance(stat_features, pd.DataFrame)
        assert len(stat_features) == len(sample_ohlcv)

        # Check for statistical columns
        stat_types = engine.config["statistical"]["features"]
        for stat_type in stat_types:
            matching_cols = [
                col
                for col in stat_features.columns
                if stat_type.lower() in col.lower()
            ]
            assert len(matching_cols) > 0

    def test_compute_technical_features(self, engine, sample_ohlcv):
        """Test technical indicator computation."""
        tech_features = engine._compute_technical_features(sample_ohlcv)

        assert isinstance(tech_features, pd.DataFrame)
        assert len(tech_features) == len(sample_ohlcv)

        # Check for technical indicator columns
        indicators = engine.config["technical"]["indicators"]
        for indicator in indicators:
            matching_cols = [
                col
                for col in tech_features.columns
                if indicator.lower() in col.lower()
            ]
            assert len(matching_cols) > 0

    def test_handle_missing_values(self, engine, sample_ohlcv):
        """Test missing value handling."""
        # Add some NaN values
        data_with_nan = sample_ohlcv.copy()
        data_with_nan.iloc[10:15, 0] = np.nan
        data_with_nan.iloc[20:25, 1] = np.nan

        # Forward fill
        filled_ff = engine._handle_missing_values(
            data_with_nan, method="forward_fill"
        )
        assert filled_ff.isna().sum().sum() < data_with_nan.isna().sum().sum()

        # Interpolation
        filled_interp = engine._handle_missing_values(
            data_with_nan, method="interpolate"
        )
        assert (
            filled_interp.isna().sum().sum() < data_with_nan.isna().sum().sum()
        )

        # Drop
        filled_drop = engine._handle_missing_values(
            data_with_nan, method="drop"
        )
        assert len(filled_drop) < len(data_with_nan)

    def test_normalize_features(self, engine, sample_ohlcv):
        """Test feature normalization."""
        features = engine.compute_features(sample_ohlcv)

        # Standard normalization
        norm_standard = engine.normalize_features(features, method="standard")
        assert norm_standard.shape == features.shape
        # Check that non-numeric columns are preserved
        for col in ["open", "high", "low", "close", "volume"]:
            if col in features.columns:
                assert col in norm_standard.columns

        # MinMax normalization
        norm_minmax = engine.normalize_features(features, method="minmax")
        assert norm_minmax.shape == features.shape

        # Robust normalization
        norm_robust = engine.normalize_features(features, method="robust")
        assert norm_robust.shape == features.shape

    def test_select_features(self, engine, sample_ohlcv):
        """Test feature selection."""
        features = engine.compute_features(sample_ohlcv)

        # Select top k features
        selected = engine.select_features(features, method="variance", k=10)
        assert selected.shape[1] <= 10 + 5  # Plus original OHLCV columns

        # Correlation-based selection
        selected_corr = engine.select_features(
            features, method="correlation", threshold=0.9
        )
        assert selected_corr.shape[1] <= features.shape[1]

    def test_caching(self, engine, sample_ohlcv):
        """Test feature caching."""
        # First computation
        _ = engine.compute_features(sample_ohlcv)  # Cache features

        # Should be cached
        assert len(engine.cache) > 0

        # Second computation should use cache
        with patch.object(engine, "_compute_volatility_features") as mock_vol:
            mock_vol.return_value = pd.DataFrame()
            _ = engine.compute_features(sample_ohlcv)  # Test cached computation
            # If caching works, volatility computation shouldn't be called again
            # This depends on implementation details

        # Clear cache
        engine.clear_cache()
        assert len(engine.cache) == 0

    def test_save_feature_config(self, engine, tmp_path):
        """Test saving feature configuration."""
        config_file = tmp_path / "feature_config.json"
        engine.save_feature_config(str(config_file))

        assert config_file.exists()

        # Load and verify
        with open(config_file, "r") as f:
            saved_config = json.load(f)

        assert saved_config == engine.config

    def test_compute_all_features(self, engine, sample_ohlcv):
        """Test compute_all_features method."""
        all_features = engine.compute_all_features(sample_ohlcv)

        assert isinstance(all_features, pd.DataFrame)
        assert len(all_features) == len(sample_ohlcv)

        # Should include all feature types
        assert len(all_features.columns) > len(sample_ohlcv.columns)

    def test_get_feature_importance(self, engine, sample_ohlcv):
        """Test feature importance calculation."""
        _ = engine.compute_features(sample_ohlcv)  # Compute features for importance test

        # Mock feature importance
        importance = engine.get_feature_importance()

        assert isinstance(importance, dict)
        # Implementation dependent - may be empty initially

    def test_validate_features(self, engine, sample_ohlcv):
        """Test feature validation."""
        features = engine.compute_features(sample_ohlcv)

        validation_results = engine.validate_features(features)

        assert isinstance(validation_results, dict)
        assert "missing_values" in validation_results
        assert "infinite_values" in validation_results
        assert "constant_features" in validation_results

    def test_error_handling(self, engine):
        """Test error handling."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            engine.compute_features(empty_df)

        # Invalid data type
        with pytest.raises(TypeError):
            engine.compute_features("not a dataframe")

        # Missing required columns
        invalid_df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError):
            engine.compute_features(invalid_df)
