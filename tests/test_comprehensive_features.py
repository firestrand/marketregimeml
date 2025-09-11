"""Unit tests for comprehensive feature engineering module."""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from marketregimeml.features.comprehensive import ComprehensiveFeatures


class TestComprehensiveFeatures(unittest.TestCase):
    """Test comprehensive feature engineering."""

    def setUp(self):
        """Set up test fixtures."""
        self.feature_engineer = ComprehensiveFeatures()
        self.ohlcv = self._create_sample_ohlcv()

    def _create_sample_ohlcv(self, n_samples: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq="H")

        # Generate realistic OHLCV data
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

        df = pd.DataFrame(index=dates)
        df["close"] = close_prices
        df["open"] = (
            df["close"].shift(1).fillna(close_prices[0])
            + np.random.randn(n_samples) * 0.1
        )
        df["high"] = np.maximum(df["open"], df["close"]) + np.abs(
            np.random.randn(n_samples) * 0.2
        )
        df["low"] = np.minimum(df["open"], df["close"]) - np.abs(
            np.random.randn(n_samples) * 0.2
        )
        df["volume"] = np.random.uniform(1000, 5000, n_samples)

        return df

    def test_initialization(self):
        """Test feature engineer initialization."""
        self.assertIsNotNone(self.feature_engineer)
        self.assertIsNotNone(self.feature_engineer.technical)
        self.assertIsNotNone(self.feature_engineer.volatility)
        self.assertIsNotNone(self.feature_engineer.statistical)
        self.assertIsNotNone(self.feature_engineer.entropy)

    def test_create_features_basic(self):
        """Test basic feature creation."""
        features = self.feature_engineer.create_features(
            self.ohlcv, feature_sets=["returns"], windows=[5, 10]
        )

        # Check that features were created
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.ohlcv))

        # Check expected columns exist
        self.assertIn("returns", features.columns)
        self.assertIn("log_returns", features.columns)
        self.assertIn("abs_returns", features.columns)
        self.assertIn("squared_returns", features.columns)

        # Check rolling features
        self.assertIn("returns_mean_5", features.columns)
        self.assertIn("returns_std_10", features.columns)

    def test_create_features_price(self):
        """Test price feature creation."""
        features = self.feature_engineer.create_features(
            self.ohlcv, feature_sets=["price"], windows=[5, 20]
        )

        self.assertIn("price_momentum_5", features.columns)
        self.assertIn("price_momentum_20", features.columns)
        self.assertIn("price_position", features.columns)
        self.assertIn("price_to_ma_5", features.columns)
        self.assertIn("price_to_ma_20", features.columns)

    def test_create_features_volatility(self):
        """Test volatility feature creation."""
        features = self.feature_engineer.create_features(
            self.ohlcv,
            feature_sets=[
                "volatility",
                "returns",
            ],  # Need returns for vol ratios
            windows=[5, 10, 20],
        )

        # Check volatility measures
        self.assertIn("parkinson_10", features.columns)
        self.assertIn("garman_klass_20", features.columns)
        self.assertIn("yang_zhang_10", features.columns)

        # Check volatility ratios
        self.assertIn("vol_ratio_5_20", features.columns)

    def test_create_features_technical(self):
        """Test technical indicator features."""
        features = self.feature_engineer.create_features(
            self.ohlcv, feature_sets=["technical"]
        )

        # Check RSI
        self.assertIn("rsi_7", features.columns)
        self.assertIn("rsi_14", features.columns)
        self.assertIn("rsi_21", features.columns)

        # Check MACD
        self.assertIn("macd", features.columns)
        self.assertIn("macd_signal", features.columns)
        self.assertIn("macd_histogram", features.columns)

        # Check Bollinger Bands
        self.assertIn("bb_position_10", features.columns)
        self.assertIn("bb_width_20", features.columns)

        # Check other indicators
        self.assertIn("stoch_k", features.columns)
        self.assertIn("stoch_d", features.columns)
        self.assertIn("atr_14", features.columns)

    def test_create_features_microstructure(self):
        """Test market microstructure features."""
        features = self.feature_engineer.create_features(
            self.ohlcv,
            feature_sets=[
                "microstructure",
                "returns",
            ],  # Need returns for Amihud
        )

        self.assertIn("volume_ratio", features.columns)
        self.assertIn("volume_trend", features.columns)
        self.assertIn("price_efficiency", features.columns)
        self.assertIn("hl_spread", features.columns)
        self.assertIn("co_spread", features.columns)
        self.assertIn("amihud_illiquidity", features.columns)

    def test_create_features_no_infinities(self):
        """Test that features don't contain infinities."""
        features = self.feature_engineer.create_features(
            self.ohlcv, feature_sets=["returns", "volatility", "technical"]
        )

        # Check no infinities
        self.assertFalse(np.isinf(features.values).any())

        # Check NaN handling (should be filled)
        self.assertFalse(
            features.isnull().all().any()
        )  # No column should be all NaN

    def test_create_ml_optimized_features(self):
        """Test ML-optimized feature creation."""
        features = self.feature_engineer.create_ml_optimized_features(
            self.ohlcv, max_features=30
        )

        self.assertIsInstance(features, pd.DataFrame)
        self.assertLessEqual(len(features.columns), 30)
        self.assertGreater(len(features.columns), 0)

        # Should have no infinities
        self.assertFalse(np.isinf(features.values).any())

    def test_create_regime_optimized_features_3_regimes(self):
        """Test regime-optimized features for 3 regimes."""
        features = self.feature_engineer.create_regime_optimized_features(
            self.ohlcv, n_regimes=3
        )

        # For 3 regimes, expect simpler features
        self.assertIn("returns", features.columns)
        self.assertIn("ma_trend", features.columns)
        self.assertIn("momentum", features.columns)
        self.assertIn("volatility", features.columns)
        self.assertIn("vol_change", features.columns)
        self.assertIn("rsi", features.columns)

        # Should have fewer features than 5+ regimes
        self.assertLess(len(features.columns), 15)

    def test_create_regime_optimized_features_5_regimes(self):
        """Test regime-optimized features for 5+ regimes."""
        features = self.feature_engineer.create_regime_optimized_features(
            self.ohlcv, n_regimes=5
        )

        # For 5+ regimes, expect more granular features
        self.assertIn("returns_5", features.columns)
        self.assertIn("returns_10", features.columns)
        self.assertIn("volatility_5", features.columns)
        self.assertIn("volatility_20", features.columns)
        self.assertIn("rsi_7", features.columns)
        self.assertIn("rsi_14", features.columns)

        # Should have more features than 3 regimes
        self.assertGreater(len(features.columns), 10)

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()

        features = self.feature_engineer.create_features(
            empty_df, feature_sets=["returns"]
        )

        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 0)

    def test_single_row_handling(self):
        """Test handling of single row DataFrame."""
        single_row = self.ohlcv.iloc[:1]

        features = self.feature_engineer.create_features(
            single_row, feature_sets=["returns"]
        )

        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), 1)
        # Should handle gracefully without errors

    def test_custom_windows(self):
        """Test custom window sizes."""
        custom_windows = [3, 7, 15]

        features = self.feature_engineer.create_features(
            self.ohlcv, feature_sets=["returns"], windows=custom_windows
        )

        # Check custom windows were used
        self.assertIn("returns_mean_3", features.columns)
        self.assertIn("returns_std_7", features.columns)
        self.assertIn("returns_skew_15", features.columns)

        # Check default windows were not used
        self.assertNotIn("returns_mean_50", features.columns)

    def test_feature_consistency(self):
        """Test that features are consistent across calls."""
        features1 = self.feature_engineer.create_features(
            self.ohlcv, feature_sets=["returns"], windows=[10]
        )

        features2 = self.feature_engineer.create_features(
            self.ohlcv, feature_sets=["returns"], windows=[10]
        )

        # Should produce identical results
        pd.testing.assert_frame_equal(features1, features2)

    def test_all_feature_sets(self):
        """Test creating all feature sets at once."""
        features = self.feature_engineer.create_features(
            self.ohlcv, feature_sets=None  # Default = all
        )

        # Should have many features
        self.assertGreater(len(features.columns), 30)

        # Check samples from each feature set
        self.assertIn("returns", features.columns)  # returns
        self.assertIn("price_momentum_5", features.columns)  # price
        self.assertIn("parkinson_10", features.columns)  # volatility
        self.assertIn("rsi_14", features.columns)  # technical
        self.assertIn("sample_entropy", features.columns)  # entropy
        self.assertIn("volume_ratio", features.columns)  # microstructure


if __name__ == "__main__":
    unittest.main()
