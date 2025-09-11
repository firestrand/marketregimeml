"""Test suite for enhanced HMM features following TDD approach."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from marketregimeml.models.hmm import HMMRegimeDetector


class TestHMMOptimization:
    """Test BIC/AIC optimization for optimal regime selection."""

    @pytest.fixture
    def sample_features(self):
        """Generate sample feature data for testing."""
        np.random.seed(42)

        # Create synthetic regime-switching data
        returns = np.concatenate(
            [
                np.random.normal(0.001, 0.01, 150),  # Bull regime
                np.random.normal(-0.001, 0.02, 200),  # Bear regime
                np.random.normal(0.0005, 0.015, 150),  # Neutral regime
            ]
        )

        volatility = np.concatenate(
            [
                np.random.normal(0.01, 0.002, 150),  # Low vol
                np.random.normal(0.02, 0.005, 200),  # High vol
                np.random.normal(0.015, 0.003, 150),  # Medium vol
            ]
        )

        return pd.DataFrame({"returns": returns, "volatility": volatility})

    def test_optimize_n_regimes_bic(self, sample_features):
        """Test BIC-based optimization finds optimal number of regimes."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)

        results = detector.optimize_n_regimes(
            sample_features,
            min_regimes=2,
            max_regimes=4,
            criterion="bic",
            n_init=3,
        )

        assert "optimal_n_regimes" in results
        assert "bic" in results
        assert "aic" in results
        assert "log_likelihood" in results
        assert "n_parameters" in results
        assert results["criterion"] == "bic"
        assert 2 <= results["optimal_n_regimes"] <= 4
        assert len(results["bic"]) == 3  # Tested 2, 3, 4 regimes

        # BIC values should be computed
        assert all(isinstance(b, float) for b in results["bic"])

        # Model should be refitted with optimal regimes
        assert detector.n_regimes == results["optimal_n_regimes"]
        assert detector.is_fitted

    def test_optimize_n_regimes_aic(self, sample_features):
        """Test AIC-based optimization."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)

        results = detector.optimize_n_regimes(
            sample_features,
            min_regimes=2,
            max_regimes=3,
            criterion="aic",
            n_init=2,
        )

        assert results["criterion"] == "aic"
        assert "optimal_n_regimes" in results
        assert "aic" in results
        assert len(results["aic"]) == 2  # Tested 2, 3 regimes

    def test_fit_with_bic_selection(self, sample_features):
        """Test model fitting with BIC-based model selection."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features, n_init=3, select_best="bic")

        assert detector.is_fitted
        assert hasattr(detector, "best_model_idx")
        assert detector.diagnostics["bic"] is not None
        assert detector.diagnostics["aic"] is not None

    def test_fit_with_aic_selection(self, sample_features):
        """Test model fitting with AIC-based model selection."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features, n_init=3, select_best="aic")

        assert detector.is_fitted
        assert hasattr(detector, "best_model_idx")


class TestRegimeLabeling:
    """Test interpretable regime labeling functionality."""

    @pytest.fixture
    def fitted_detector(self):
        """Create a fitted HMM detector for testing."""
        np.random.seed(42)

        # Create clear regime data
        bull_returns = np.random.normal(0.002, 0.01, 100)
        bear_returns = np.random.normal(-0.002, 0.02, 100)
        sideways_returns = np.random.normal(0, 0.015, 100)

        returns = np.concatenate(
            [bull_returns, bear_returns, sideways_returns]
        )
        volatility = np.abs(returns) * 2 + np.random.normal(0, 0.001, 300)

        features = pd.DataFrame({"returns": returns, "volatility": volatility})

        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(features, n_init=5)

        return detector, features

    def test_label_regimes_volatility_return(self, fitted_detector):
        """Test regime labeling based on volatility and returns."""
        detector, features = fitted_detector

        labels = detector.label_regimes(features, method="volatility_return")

        assert isinstance(labels, dict)
        assert len(labels) == 3

        # Check label format
        valid_labels = [
            "Bull_Normal",
            "Bull_HighVol",
            "Bear_Normal",
            "Bear_Crisis",
            "Sideways_Normal",
            "Sideways_HighVol",
        ]

        for regime_id, label in labels.items():
            assert isinstance(regime_id, int)
            assert isinstance(label, str)
            assert label in valid_labels or label.startswith("Regime_")

        # Labels should be stored in detector
        assert detector.regime_names == labels

    def test_label_regimes_return_quantile(self, fitted_detector):
        """Test regime labeling based on return quantiles."""
        detector, features = fitted_detector

        labels = detector.label_regimes(features, method="return_quantile")

        assert isinstance(labels, dict)
        assert len(labels) == 3

        valid_labels = ["Bear_Market", "Bull_Market", "Neutral_Market"]

        for regime_id, label in labels.items():
            assert label in valid_labels or label.startswith("Regime_")

    def test_label_regimes_custom(self, fitted_detector):
        """Test custom/default regime labeling."""
        detector, features = fitted_detector

        labels = detector.label_regimes(features, method="custom")

        assert isinstance(labels, dict)
        assert len(labels) == 3

        # For 3 regimes, should have Bear, Neutral, Bull
        label_values = list(labels.values())
        assert "Bear" in label_values
        assert "Neutral" in label_values
        assert "Bull" in label_values

    def test_label_regimes_different_n_regimes(self):
        """Test labeling works for different numbers of regimes."""
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "returns": np.random.normal(0, 0.01, 200),
                "volatility": np.random.uniform(0.005, 0.02, 200),
            }
        )

        # Test with 2 regimes
        detector2 = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector2.fit(features, n_init=2)
        labels2 = detector2.label_regimes(features, method="custom")
        assert set(labels2.values()) == {"Bear", "Bull"}

        # Test with 4 regimes
        detector4 = HMMRegimeDetector(n_regimes=4, random_state=42)
        detector4.fit(features, n_init=2)
        labels4 = detector4.label_regimes(features, method="custom")
        expected_labels = {
            "Strong_Bear",
            "Weak_Bear",
            "Weak_Bull",
            "Strong_Bull",
        }
        assert set(labels4.values()) == expected_labels


class TestRegimeStatistics:
    """Test regime statistics calculation."""

    @pytest.fixture
    def detector_with_data(self):
        """Create detector with fitted model and data."""
        np.random.seed(42)

        # Create regime-switching data
        n_samples = 300
        regime_sequence = np.array([0] * 100 + [1] * 100 + [2] * 100)

        # Generate features based on regime
        returns = np.zeros(n_samples)
        volatility = np.zeros(n_samples)

        for i in range(n_samples):
            if regime_sequence[i] == 0:  # Bull
                returns[i] = np.random.normal(0.002, 0.01)
                volatility[i] = np.random.normal(0.01, 0.002)
            elif regime_sequence[i] == 1:  # Bear
                returns[i] = np.random.normal(-0.002, 0.02)
                volatility[i] = np.random.normal(0.02, 0.005)
            else:  # Sideways
                returns[i] = np.random.normal(0, 0.015)
                volatility[i] = np.random.normal(0.015, 0.003)

        features = pd.DataFrame({"returns": returns, "volatility": volatility})

        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(features, n_init=3)
        detector.label_regimes(features)

        return detector, features

    def test_get_regime_statistics(self, detector_with_data):
        """Test calculation of regime statistics."""
        detector, features = detector_with_data

        stats = detector.get_regime_statistics(features)

        assert isinstance(stats, pd.DataFrame)
        assert len(stats) <= 3  # Up to 3 regimes

        # Check required columns
        required_cols = [
            "regime_id",
            "regime_name",
            "count",
            "proportion",
            "returns_mean",
            "returns_std",
            "returns_min",
            "returns_max",
            "returns_skew",
            "returns_kurtosis",
            "volatility_mean",
            "volatility_std",
            "avg_duration",
            "max_duration",
            "min_duration",
        ]

        for col in required_cols:
            assert col in stats.columns

        # Validate proportions sum to 1
        assert np.isclose(stats["proportion"].sum(), 1.0)

        # Check data types
        assert stats["count"].dtype == np.int64
        assert stats["proportion"].dtype == np.float64

    def test_regime_statistics_consistency(self, detector_with_data):
        """Test that regime statistics are internally consistent."""
        detector, features = detector_with_data

        stats = detector.get_regime_statistics(features)
        regimes = detector.predict(features)

        # Verify counts match
        for _, row in stats.iterrows():
            regime_id = row["regime_id"]
            actual_count = (regimes == regime_id).sum()
            assert row["count"] == actual_count

        # Verify total counts
        assert stats["count"].sum() == len(features)

    def test_calculate_regime_durations(self, detector_with_data):
        """Test regime duration calculation."""
        detector, _ = detector_with_data

        # Test with known regime sequence
        regimes = np.array([0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2, 1])

        durations_0 = detector._calculate_regime_durations(regimes, 0)
        assert durations_0 == [3, 2]

        durations_1 = detector._calculate_regime_durations(regimes, 1)
        assert durations_1 == [2, 1]

        durations_2 = detector._calculate_regime_durations(regimes, 2)
        assert durations_2 == [4]


class TestParameterCounting:
    """Test parameter counting for model selection."""

    def test_count_parameters_full_covariance(self):
        """Test parameter counting with full covariance."""
        detector = HMMRegimeDetector(
            n_regimes=3, covariance_type="full", random_state=42
        )

        # Create dummy data and fit
        features = pd.DataFrame(
            {
                "returns": np.random.normal(0, 0.01, 100),
                "volatility": np.random.normal(0.01, 0.002, 100),
            }
        )
        detector.fit(features, n_init=2)

        n_params = detector._count_parameters()

        # Expected parameters:
        # - Initial probs: 3-1 = 2
        # - Transition matrix: 3*(3-1) = 6
        # - Means: 3*2 = 6
        # - Full covariances: 3*2*(2+1)/2 = 9
        expected = 2 + 6 + 6 + 9
        assert n_params == expected

    def test_count_parameters_diag_covariance(self):
        """Test parameter counting with diagonal covariance."""
        detector = HMMRegimeDetector(
            n_regimes=2, covariance_type="diag", random_state=42
        )

        features = pd.DataFrame(
            {
                "returns": np.random.normal(0, 0.01, 100),
                "volatility": np.random.normal(0.01, 0.002, 100),
            }
        )
        detector.fit(features, n_init=2)

        n_params = detector._count_parameters()

        # Expected parameters:
        # - Initial probs: 2-1 = 1
        # - Transition matrix: 2*(2-1) = 2
        # - Means: 2*2 = 4
        # - Diag covariances: 2*2 = 4
        expected = 1 + 2 + 4 + 4
        assert n_params == expected


class TestModelPersistence:
    """Test model saving and loading with new features."""

    def test_save_load_with_labels(self, tmp_path):
        """Test saving and loading model with regime labels."""
        # Create and fit model
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)

        features = pd.DataFrame(
            {
                "returns": np.random.normal(0, 0.01, 200),
                "volatility": np.random.uniform(0.005, 0.02, 200),
            }
        )

        detector.fit(features, n_init=3)
        detector.label_regimes(features, method="volatility_return")

        # Save model
        filepath = tmp_path / "hmm_with_labels.pkl"
        detector.save(str(filepath))

        # Load model
        loaded = HMMRegimeDetector.load(str(filepath))

        # Verify labels are preserved
        assert loaded.regime_names == detector.regime_names
        assert loaded.is_fitted
        assert loaded.n_regimes == detector.n_regimes

        # Verify predictions are the same
        orig_pred = detector.predict(features)
        loaded_pred = loaded.predict(features)
        np.testing.assert_array_equal(orig_pred, loaded_pred)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_optimize_with_invalid_range(self):
        """Test optimization with invalid regime range."""
        detector = HMMRegimeDetector(n_regimes=2)
        features = pd.DataFrame({"returns": np.random.normal(0, 0.01, 50)})

        with pytest.raises(
            ValueError, match="min_regimes.*must be <= max_regimes"
        ):
            # Min > Max should fail
            detector.optimize_n_regimes(features, min_regimes=5, max_regimes=2)

    def test_label_regimes_before_fit(self):
        """Test labeling regimes before model is fitted."""
        detector = HMMRegimeDetector(n_regimes=3)
        features = pd.DataFrame({"returns": np.random.normal(0, 0.01, 100)})

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.label_regimes(features)

    def test_empty_regime_handling(self):
        """Test handling of regimes with no data points."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)

        # Create data that might result in empty regimes
        features = pd.DataFrame(
            {
                "returns": np.random.normal(0, 0.001, 50),  # Very low variance
                "volatility": np.ones(50) * 0.01,
            }
        )

        detector.fit(features, n_init=2)
        stats = detector.get_regime_statistics(features)

        # Should handle empty regimes gracefully
        assert len(stats) <= 3
        assert stats["count"].sum() == len(features)


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self):
        """Test complete workflow from optimization to statistics."""
        np.random.seed(42)

        # Generate realistic market data
        n_samples = 500
        features = pd.DataFrame(
            {
                "returns": np.random.normal(0, 0.015, n_samples),
                "volatility": np.abs(np.random.normal(0.01, 0.005, n_samples)),
                "volume": np.random.uniform(1000, 5000, n_samples),
            }
        )

        # 1. Initialize detector
        detector = HMMRegimeDetector(random_state=42)

        # 2. Find optimal number of regimes
        opt_results = detector.optimize_n_regimes(
            features, min_regimes=2, max_regimes=4, criterion="bic", n_init=3
        )

        # 3. Label regimes
        labels = detector.label_regimes(features, method="volatility_return")

        # 4. Get statistics
        stats = detector.get_regime_statistics(features)

        # 5. Make predictions
        regimes = detector.predict(features)
        probabilities = detector.predict_proba(features)

        # Validate complete workflow
        assert detector.is_fitted
        assert 2 <= opt_results["optimal_n_regimes"] <= 4
        assert len(labels) == detector.n_regimes
        assert len(stats) <= detector.n_regimes
        assert len(regimes) == len(features)
        assert probabilities.shape == (len(features), detector.n_regimes)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
