"""Core unit tests for market regime detection functionality."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector


class TestCoreRegimeDetection:
    """Test core regime detection functionality."""

    @pytest.fixture
    def simple_regime_data(self):
        """Create simple data with clear regimes."""
        np.random.seed(42)

        # Create 3 distinct regimes
        regime_data = []
        regime_labels = []

        # Regime 0: Low mean, low volatility
        data_0 = np.random.normal(0, 0.5, (100, 3))
        regime_data.append(data_0)
        regime_labels.extend([0] * 100)

        # Regime 1: High mean, medium volatility
        data_1 = np.random.normal(2, 1.0, (100, 3))
        regime_data.append(data_1)
        regime_labels.extend([1] * 100)

        # Regime 2: Negative mean, high volatility
        data_2 = np.random.normal(-1, 2.0, (100, 3))
        regime_data.append(data_2)
        regime_labels.extend([2] * 100)

        # Combine data
        X = np.vstack(regime_data)
        df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])

        return df, np.array(regime_labels)

    @pytest.fixture
    def time_series_data(self):
        """Create time series data with regime switches."""
        np.random.seed(42)

        n_samples = 500
        data = []
        regimes = []

        # Generate time series with regime switches
        current_regime = 0
        regime_means = [0, 2, -1]
        regime_stds = [0.5, 1.0, 2.0]

        for i in range(n_samples):
            # Switch regime occasionally
            if np.random.random() < 0.05:
                current_regime = np.random.choice([0, 1, 2])

            # Generate data for current regime
            value = np.random.normal(
                regime_means[current_regime], regime_stds[current_regime]
            )
            data.append(
                [value, value * 0.5 + np.random.normal(0, 0.1), np.abs(value)]
            )
            regimes.append(current_regime)

        df = pd.DataFrame(data, columns=["returns", "momentum", "volatility"])
        return df, np.array(regimes)

    def test_hmm_basic_functionality(self, simple_regime_data):
        """Test HMM basic functionality."""
        data, true_regimes = simple_regime_data

        # Create and fit model
        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(data)

        # Basic assertions
        assert model.is_fitted
        assert model.n_features == 3

        # Predict regimes
        predictions = model.predict(data)
        assert len(predictions) == len(data)
        assert predictions.min() >= 0
        assert predictions.max() < 3
        assert len(np.unique(predictions)) <= 3

        # Get probabilities
        proba = model.predict_proba(data)
        assert proba.shape == (len(data), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_gmm_basic_functionality(self, simple_regime_data):
        """Test GMM basic functionality."""
        data, true_regimes = simple_regime_data

        # Create and fit model
        model = GMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(data)

        # Basic assertions
        assert model.is_fitted
        assert model.n_features == 3

        # Predict regimes
        predictions = model.predict(data)
        assert len(predictions) == len(data)
        assert predictions.min() >= 0
        assert predictions.max() < 3
        assert len(np.unique(predictions)) <= 3

        # Get probabilities
        proba = model.predict_proba(data)
        assert proba.shape == (len(data), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_hmm_time_series(self, time_series_data):
        """Test HMM with time series data."""
        data, true_regimes = time_series_data

        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(data)

        predictions = model.predict(data)

        # Check for regime transitions
        transitions = np.diff(predictions)
        n_transitions = np.sum(transitions != 0)

        # Should have some transitions but not too many
        assert n_transitions > 5  # At least some transitions
        assert n_transitions < len(data) * 0.5  # Not switching every step

    def test_gmm_clustering(self):
        """Test GMM correctly clusters data."""
        # Create clearly separated clusters
        X, y = make_blobs(
            n_samples=300,
            n_features=2,
            centers=3,
            cluster_std=0.5,
            random_state=42,
        )

        df = pd.DataFrame(X, columns=["feature1", "feature2"])

        model = GMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(df)

        predictions = model.predict(df)

        # Check that most points in same true cluster get same prediction
        # (labels might be permuted)
        from sklearn.metrics import adjusted_rand_score

        ari = adjusted_rand_score(y, predictions)
        assert ari > 0.5  # Reasonable clustering performance

    def test_regime_persistence(self, time_series_data):
        """Test that regimes show persistence (not random switching)."""
        data, _ = time_series_data

        # Test both models
        for ModelClass in [HMMRegimeDetector, GMMRegimeDetector]:
            model = ModelClass(n_regimes=3, random_state=42)
            model.fit(data)
            predictions = model.predict(data)

            # Calculate regime durations
            durations = []
            current_regime = predictions[0]
            current_duration = 1

            for regime in predictions[1:]:
                if regime == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_regime = regime
                    current_duration = 1
            durations.append(current_duration)

            # Average duration should be reasonable
            avg_duration = np.mean(durations)
            assert (
                avg_duration > 2
            )  # Regimes should persist for at least 2 periods
            assert (
                avg_duration < len(data) / 2
            )  # Should have multiple regime switches

    def test_model_scoring(self, simple_regime_data):
        """Test model scoring methods."""
        data, _ = simple_regime_data

        for ModelClass in [HMMRegimeDetector, GMMRegimeDetector]:
            model = ModelClass(n_regimes=3, random_state=42)
            model.fit(data)

            # Test log likelihood
            ll = model.score(data, metric="log_likelihood")
            assert isinstance(ll, float)
            assert ll < 0  # Log likelihood should be negative

            # Test AIC
            aic = model.score(data, metric="aic")
            assert isinstance(aic, float)
            assert aic > 0

            # Test BIC
            bic = model.score(data, metric="bic")
            assert isinstance(bic, float)
            assert bic > 0
            assert bic > aic  # BIC penalizes complexity more

    def test_different_n_regimes(self, simple_regime_data):
        """Test models with different numbers of regimes."""
        data, _ = simple_regime_data

        for n in [2, 3, 4]:
            # Test HMM
            hmm = HMMRegimeDetector(n_regimes=n, random_state=42)
            hmm.fit(data)
            predictions = hmm.predict(data)
            assert len(np.unique(predictions)) <= n

            # Test GMM
            gmm = GMMRegimeDetector(n_regimes=n, random_state=42)
            gmm.fit(data)
            predictions = gmm.predict(data)
            assert len(np.unique(predictions)) <= n

    def test_reproducibility(self, simple_regime_data):
        """Test that results are reproducible with same random state."""
        data, _ = simple_regime_data

        for ModelClass in [HMMRegimeDetector, GMMRegimeDetector]:
            # Train two models with same random state
            model1 = ModelClass(n_regimes=3, random_state=42)
            model2 = ModelClass(n_regimes=3, random_state=42)

            model1.fit(data)
            model2.fit(data)

            pred1 = model1.predict(data)
            pred2 = model2.predict(data)

            # Predictions should be identical
            assert np.array_equal(pred1, pred2)

    def test_empty_data_handling(self):
        """Test handling of edge cases."""
        # Empty dataframe
        empty_df = pd.DataFrame()

        model = HMMRegimeDetector(n_regimes=3)
        with pytest.raises((ValueError, IndexError)):
            model.fit(empty_df)

        # Single sample
        single_df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            model.fit(single_df)

    def test_predict_unfitted(self, simple_regime_data):
        """Test prediction on unfitted model raises error."""
        data, _ = simple_regime_data

        for ModelClass in [HMMRegimeDetector, GMMRegimeDetector]:
            model = ModelClass(n_regimes=3)
            with pytest.raises(ValueError, match="fitted"):
                model.predict(data)

    def test_feature_consistency(self, simple_regime_data):
        """Test that models require consistent features."""
        data, _ = simple_regime_data

        # Train on 3 features
        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(data)

        # Try to predict with different number of features
        wrong_data = data[["feature1", "feature2"]]  # Only 2 features

        with pytest.raises((ValueError, IndexError)):
            model.predict(wrong_data)

    def test_probability_consistency(self, simple_regime_data):
        """Test that probabilities and predictions are consistent."""
        data, _ = simple_regime_data

        for ModelClass in [HMMRegimeDetector, GMMRegimeDetector]:
            model = ModelClass(n_regimes=3, random_state=42)
            model.fit(data)

            predictions = model.predict(data)
            probabilities = model.predict_proba(data)

            # Predictions should match argmax of probabilities
            prob_predictions = np.argmax(probabilities, axis=1)
            assert np.array_equal(predictions, prob_predictions)

    def test_regime_statistics(self, time_series_data):
        """Test regime statistics calculation."""
        data, _ = time_series_data

        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(data)

        stats = model.get_regime_statistics(data)

        assert isinstance(stats, dict)
        assert len(stats) <= 3

        total_count = sum(s["count"] for s in stats.values())
        assert total_count == len(data)

        for regime_id, regime_stats in stats.items():
            assert regime_id >= 0 and regime_id < 3
            assert "count" in regime_stats
            assert "percentage" in regime_stats
            assert regime_stats["count"] > 0
            assert 0 < regime_stats["percentage"] <= 100
