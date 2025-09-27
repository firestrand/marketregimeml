"""Tests for base regime detector."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from marketregimeml.models.base import BaseRegimeDetector


class SimpleTestModel:
    """Simple test model that can be pickled."""

    def __init__(self, n_regimes):
        self.n_regimes = n_regimes
        self.fitted = False

    def fit(self, X):
        self.fitted = True
        return self

    def predict(self, X):
        np.random.seed(42)
        return np.random.randint(0, self.n_regimes, len(X))

    def predict_proba(self, X):
        n_samples = len(X)
        proba = np.random.rand(n_samples, self.n_regimes)
        return proba / proba.sum(axis=1, keepdims=True)


class ConcreteRegimeDetector(BaseRegimeDetector):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, n_regimes=3, random_state=None, **kwargs):
        super().__init__(n_regimes=n_regimes, random_state=random_state, **kwargs)
        # Use a simple test model instead of Mock for pickling
        self.model = SimpleTestModel(n_regimes)

    def fit(self, features, **kwargs):
        """Mock fit implementation."""
        self.is_fitted = True
        self.n_features = features.shape[1]
        self.feature_names = list(features.columns)
        from datetime import datetime

        self.fit_date = datetime.now()
        return self

    def predict(self, features):
        """Mock predict implementation."""
        np.random.seed(42)
        return np.random.randint(0, self.n_regimes, len(features))

    def predict_proba(self, features):
        """Mock predict_proba implementation."""
        np.random.seed(42)
        return np.random.dirichlet([1] * self.n_regimes, len(features))

    def _compute_log_likelihood(self, features):
        """Mock log-likelihood computation."""
        return -100.0

    def _count_parameters(self):
        """Mock parameter counting."""
        return 10


class TestBaseRegimeDetector:
    """Test suite for BaseRegimeDetector."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature data."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def detector(self):
        """Create concrete detector instance."""
        return ConcreteRegimeDetector(n_regimes=3, random_state=42)

    @pytest.fixture
    def fitted_detector(self, detector, sample_features):
        """Create fitted detector instance."""
        detector.fit(sample_features)
        return detector

    def test_init_default(self):
        """Test initialization with default parameters."""
        detector = ConcreteRegimeDetector()

        assert detector.n_regimes == 3
        assert detector.random_state is None
        assert not detector.is_fitted
        assert detector.fit_date is None
        assert detector.n_features is None
        assert detector.feature_names is None
        assert detector.train_log_likelihood is None
        assert len(detector.regime_names) == 3
        assert detector.diagnostics == {}

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        detector = ConcreteRegimeDetector(n_regimes=4, random_state=42)

        assert detector.n_regimes == 4
        assert detector.random_state == 42
        assert len(detector.regime_names) == 4

    def test_default_regime_names_2_regimes(self):
        """Test default regime names for 2 regimes."""
        detector = ConcreteRegimeDetector(n_regimes=2)

        expected = {0: "Bear", 1: "Bull"}
        assert detector.regime_names == expected

    def test_default_regime_names_3_regimes(self):
        """Test default regime names for 3 regimes."""
        detector = ConcreteRegimeDetector(n_regimes=3)

        expected = {0: "Bear", 1: "Neutral", 2: "Bull"}
        assert detector.regime_names == expected

    def test_default_regime_names_4_regimes(self):
        """Test default regime names for 4 regimes."""
        detector = ConcreteRegimeDetector(n_regimes=4)

        expected = {0: "Crisis", 1: "Bear", 2: "Bull", 3: "Rally"}
        assert detector.regime_names == expected

    def test_default_regime_names_generic(self):
        """Test default regime names for 5 regimes."""
        detector = ConcreteRegimeDetector(n_regimes=5)

        expected = {
            0: "Deep_Bear",
            1: "Bear",
            2: "Neutral",
            3: "Bull",
            4: "Strong_Bull",
        }
        assert detector.regime_names == expected

    def test_fit_predict(self, detector, sample_features):
        """Test fit_predict method."""
        predictions = detector.fit_predict(sample_features)

        assert detector.is_fitted
        assert len(predictions) == len(sample_features)
        assert all(0 <= pred < 3 for pred in predictions)

    def test_score_not_fitted(self, detector, sample_features):
        """Test score raises error when not fitted."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.score(sample_features)

    def test_score_log_likelihood(self, fitted_detector, sample_features):
        """Test score with log-likelihood metric."""
        score = fitted_detector.score(sample_features, metric="log_likelihood")

        assert isinstance(score, float)
        assert score == -100.0  # Mock value

    def test_score_aic(self, fitted_detector, sample_features):
        """Test score with AIC metric."""
        score = fitted_detector.score(sample_features, metric="aic")

        assert isinstance(score, float)
        # AIC = 2*k - 2*log_likelihood = 2*10 - 2*(-100) = 220
        assert score == 220

    def test_score_bic(self, fitted_detector, sample_features):
        """Test score with BIC metric."""
        score = fitted_detector.score(sample_features, metric="bic")

        assert isinstance(score, float)
        # BIC = log(n)*k - 2*log_likelihood = log(100)*10 - 2*(-100)
        expected = np.log(100) * 10 - 2 * (-100)
        assert abs(score - expected) < 1e-10

    def test_score_unknown_metric(self, fitted_detector, sample_features):
        """Test score with unknown metric raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            fitted_detector.score(sample_features, metric="unknown")

    def test_compute_aic(self, fitted_detector, sample_features):
        """Test AIC computation."""
        aic = fitted_detector._compute_aic(sample_features)

        # AIC = 2*k - 2*log_likelihood
        expected = 2 * 10 - 2 * (-100)
        assert aic == expected

    def test_compute_bic(self, fitted_detector, sample_features):
        """Test BIC computation."""
        bic = fitted_detector._compute_bic(sample_features)

        # BIC = log(n)*k - 2*log_likelihood
        expected = np.log(100) * 10 - 2 * (-100)
        assert abs(bic - expected) < 1e-10

    def test_get_regime_statistics_with_predictions(
        self, fitted_detector, sample_features
    ):
        """Test get_regime_statistics with provided regimes."""
        # Create deterministic regime sequence
        regimes = np.array([0, 0, 1, 1, 2, 2] * 16 + [0, 0, 1, 2])  # 100 samples

        stats = fitted_detector.get_regime_statistics(sample_features, regimes=regimes)

        assert len(stats) == 3
        for regime in range(3):
            assert "count" in stats[regime]
            assert "percentage" in stats[regime]
            assert "mean_duration" in stats[regime]

            # Check feature statistics
            for col in sample_features.columns:
                assert f"{col}_mean" in stats[regime]
                assert f"{col}_std" in stats[regime]

    def test_get_regime_statistics_without_predictions_fitted(
        self, fitted_detector, sample_features
    ):
        """Test get_regime_statistics without regimes on fitted model."""
        stats = fitted_detector.get_regime_statistics(sample_features)

        assert isinstance(stats, dict)
        # Should have statistics for regimes that were predicted

    def test_get_regime_statistics_without_predictions_not_fitted(
        self, detector, sample_features
    ):
        """Test get_regime_statistics raises error when not fitted."""
        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.get_regime_statistics(sample_features)

    def test_calculate_mean_duration(self, detector):
        """Test mean duration calculation."""
        # Test sequence: [0, 0, 1, 1, 1, 0, 2, 2]
        regimes = np.array([0, 0, 1, 1, 1, 0, 2, 2])

        # Regime 0 appears as [0,0] and [0] -> durations [2, 1] -> mean = 1.5
        duration_0 = detector._calculate_mean_duration(regimes, 0)
        assert duration_0 == 1.5

        # Regime 1 appears as [1,1,1] -> duration [3] -> mean = 3.0
        duration_1 = detector._calculate_mean_duration(regimes, 1)
        assert duration_1 == 3.0

        # Regime 2 appears as [2,2] -> duration [2] -> mean = 2.0
        duration_2 = detector._calculate_mean_duration(regimes, 2)
        assert duration_2 == 2.0

    def test_calculate_mean_duration_no_occurrences(self, detector):
        """Test mean duration with regime that doesn't occur."""
        regimes = np.array([0, 0, 1, 1])

        # Regime 2 never appears
        duration = detector._calculate_mean_duration(regimes, 2)
        assert duration == 0

    def test_get_transition_matrix_with_regimes(self, detector):
        """Test transition matrix calculation with provided regimes."""
        # Simple sequence: 0->1->2->0
        regimes = np.array([0, 1, 2, 0, 1, 2])

        trans_matrix = detector.get_transition_matrix(regimes=regimes)

        assert trans_matrix.shape == (3, 3)
        # Each row should sum to 1 (probabilities)
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)

    def test_get_transition_matrix_with_features(
        self, fitted_detector, sample_features
    ):
        """Test transition matrix calculation using predicted regimes."""
        trans_matrix = fitted_detector.get_transition_matrix(features=sample_features)

        assert trans_matrix.shape == (3, 3)
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)

    def test_get_transition_matrix_no_input(self, detector):
        """Test transition matrix raises error without input."""
        with pytest.raises(
            ValueError, match="Either regimes or features must be provided"
        ):
            detector.get_transition_matrix()

    def test_get_regime_confidence(self, fitted_detector, sample_features):
        """Test regime confidence calculation."""
        confidence_df = fitted_detector.get_regime_confidence(sample_features)

        assert len(confidence_df) == len(sample_features)

        # Check columns
        expected_cols = [
            "regime",
            "regime_name",
            "confidence",
            "uncertainty",
        ] + [f"prob_regime_{i}" for i in range(3)]
        assert all(col in confidence_df.columns for col in expected_cols)

        # Check value ranges
        assert all(0 <= conf <= 1 for conf in confidence_df["confidence"])
        assert all(0 <= unc <= 1 for unc in confidence_df["uncertainty"])
        assert all(0 <= reg < 3 for reg in confidence_df["regime"])

        # Check regime names
        assert all(
            name in ["Bear", "Neutral", "Bull"] for name in confidence_df["regime_name"]
        )

    def test_cross_validate(self, detector, sample_features):
        """Test cross-validation."""
        cv_results = detector.cross_validate(
            sample_features, n_splits=3, metric="log_likelihood"
        )

        assert "train_scores" in cv_results
        assert "test_scores" in cv_results
        assert "train_mean" in cv_results
        assert "train_std" in cv_results
        assert "test_mean" in cv_results
        assert "test_std" in cv_results

        assert len(cv_results["train_scores"]) == 3
        assert len(cv_results["test_scores"]) == 3

    def test_save_pickle(self, fitted_detector):
        """Test saving model in pickle format."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            fitted_detector.save(filepath)

            # Check file was created
            assert Path(filepath).exists()

            # Check content
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            assert "model" in model_data
            assert "n_regimes" in model_data
            assert "is_fitted" in model_data
            assert model_data["n_regimes"] == 3
            assert model_data["is_fitted"] == True

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_json(self, fitted_detector):
        """Test saving model in JSON format (metadata only)."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            fitted_detector.save(filepath)

            # Check file was created
            assert Path(filepath).exists()

            # Check content
            with open(filepath, "r") as f:
                model_data = json.load(f)

            assert "model" not in model_data  # Should be excluded
            assert "n_regimes" in model_data
            assert "is_fitted" in model_data
            assert model_data["n_regimes"] == 3

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_creates_directories(self, fitted_detector):
        """Test save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "model.pkl"

            fitted_detector.save(str(filepath))

            assert filepath.exists()
            assert filepath.parent.exists()

    def test_load_pickle(self, fitted_detector):
        """Test loading model from pickle format."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            # Save model
            fitted_detector.save(filepath)

            # Load model
            loaded_detector = ConcreteRegimeDetector.load(filepath)

            assert loaded_detector.n_regimes == fitted_detector.n_regimes
            assert loaded_detector.is_fitted == fitted_detector.is_fitted
            assert loaded_detector.n_features == fitted_detector.n_features
            assert loaded_detector.feature_names == fitted_detector.feature_names

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            ConcreteRegimeDetector.load("nonexistent.pkl")

    def test_load_json_raises_error(self):
        """Test loading from JSON raises error."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
            f.write(b'{"test": "data"}')

        try:
            with pytest.raises(ValueError, match="Cannot load full model from JSON"):
                ConcreteRegimeDetector.load(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_get_diagnostics(self, fitted_detector):
        """Test get_diagnostics method."""
        # Add some diagnostics
        fitted_detector.diagnostics = {
            "test_metric": 42,
            "another_metric": "value",
        }

        diagnostics = fitted_detector.get_diagnostics()

        assert diagnostics == fitted_detector.diagnostics
        assert diagnostics["test_metric"] == 42

    def test_set_regime_names_valid(self, detector):
        """Test setting valid regime names."""
        new_names = {0: "Low", 1: "Medium", 2: "High"}

        detector.set_regime_names(new_names)

        assert detector.regime_names == new_names

    def test_set_regime_names_invalid_count(self, detector):
        """Test setting regime names with wrong count."""
        wrong_names = {0: "Low", 1: "High"}  # Only 2 names for 3 regimes

        with pytest.raises(ValueError, match="Must provide names for all 3 regimes"):
            detector.set_regime_names(wrong_names)

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        # This tests that the abstract methods are actually abstract
        with pytest.raises(TypeError):
            BaseRegimeDetector()

    def test_regime_statistics_empty_regime(self, fitted_detector, sample_features):
        """Test regime statistics handling when a regime has no samples."""
        # Create regimes where regime 2 never appears
        regimes = np.array([0, 1] * 50)  # Only regimes 0 and 1

        stats = fitted_detector.get_regime_statistics(sample_features, regimes=regimes)

        # Should only have stats for regimes that appeared
        assert len(stats) == 2
        assert 0 in stats
        assert 1 in stats
        assert 2 not in stats

    def test_transition_matrix_edge_cases(self, detector):
        """Test transition matrix with edge cases."""
        # Single regime
        regimes = np.array([0, 0, 0])
        trans_matrix = detector.get_transition_matrix(regimes=regimes)

        # Should handle division by zero gracefully
        assert trans_matrix.shape == (3, 3)
        # Row 0 should sum to 1, others should be 0
        assert abs(trans_matrix[0].sum() - 1.0) < 1e-10

    def test_cross_validate_different_metrics(self, detector, sample_features):
        """Test cross-validation with different metrics."""
        # Test with AIC
        cv_results_aic = detector.cross_validate(
            sample_features, n_splits=2, metric="aic"
        )

        assert "train_scores" in cv_results_aic
        assert len(cv_results_aic["train_scores"]) == 2

        # Test with BIC
        cv_results_bic = detector.cross_validate(
            sample_features, n_splits=2, metric="bic"
        )

        assert "train_scores" in cv_results_bic
        assert len(cv_results_bic["train_scores"]) == 2

        # Results should be different
        assert cv_results_aic["train_scores"] != cv_results_bic["train_scores"]

    def test_regime_confidence_entropy_calculation(
        self, fitted_detector, sample_features
    ):
        """Test entropy calculation in regime confidence."""
        # Mock predict_proba to return deterministic probabilities
        with patch.object(fitted_detector, "predict_proba") as mock_predict_proba:
            # High confidence case (low entropy)
            high_conf_proba = np.array([[0.95, 0.025, 0.025], [0.85, 0.075, 0.075]])
            mock_predict_proba.return_value = high_conf_proba

            confidence_df = fitted_detector.get_regime_confidence(
                sample_features.iloc[:2]
            )

            # Check that uncertainty is low for high confidence predictions
            # Using 0.6 threshold since entropy of [0.85, 0.075, 0.075] is ~0.47
            assert all(unc < 0.6 for unc in confidence_df["uncertainty"])

    def test_fit_updates_metadata(self, detector, sample_features):
        """Test that fit properly updates model metadata."""
        assert detector.fit_date is None
        assert detector.n_features is None
        assert detector.feature_names is None

        detector.fit(sample_features)

        assert detector.fit_date is not None
        assert detector.n_features == sample_features.shape[1]
        assert detector.feature_names == list(sample_features.columns)
        assert detector.is_fitted is True
