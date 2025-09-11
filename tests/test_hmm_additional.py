"""Additional tests for HMM to achieve 90% coverage."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from marketregimeml.models.hmm import HMMRegimeDetector


class TestHMMAdditionalCoverage:
    """Additional test cases for HMM to reach 90% coverage."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return pd.DataFrame(
            np.random.randn(100, 3),
            columns=["feature1", "feature2", "feature3"],
        )

    def test_save_and_load(self, sample_data):
        """Test model persistence."""
        # Train model
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(sample_data)
        original_predictions = detector.predict(sample_data)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "hmm_model.pkl"
            detector.save(str(filepath))

            # Load model
            loaded_detector = HMMRegimeDetector.load(str(filepath))

            # Check loaded model works
            assert loaded_detector.is_fitted
            assert loaded_detector.n_regimes == 2
            loaded_predictions = loaded_detector.predict(sample_data)

            # Predictions should be the same
            np.testing.assert_array_equal(
                original_predictions, loaded_predictions
            )

    def test_get_regime_statistics(self, sample_data):
        """Test regime statistics calculation."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_data)

        # Get regime statistics
        stats = detector.get_regime_statistics(sample_data)

        assert len(stats) == 3
        for regime_id, regime_stats in stats.items():
            assert "count" in regime_stats
            assert regime_stats["count"] >= 0
            # Check for feature-specific stats
            for col in sample_data.columns:
                assert f"{col}_mean" in regime_stats
                assert f"{col}_std" in regime_stats

    def test_get_transition_matrix(self, sample_data):
        """Test transition matrix retrieval."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(sample_data)

        # Get transition matrix
        trans_matrix = detector.get_transition_matrix()

        assert trans_matrix.shape == (2, 2)
        # Rows should sum to 1 (transition probabilities)
        assert np.allclose(trans_matrix.sum(axis=1), 1.0)
        # All probabilities should be between 0 and 1
        assert np.all((trans_matrix >= 0) & (trans_matrix <= 1))

    def test_get_stationary_distribution(self, sample_data):
        """Test stationary distribution calculation."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_data)

        # Get stationary distribution
        stationary = detector.get_stationary_distribution()

        assert len(stationary) == 3
        # Should sum to 1
        assert np.isclose(stationary.sum(), 1.0)
        # All probabilities should be between 0 and 1
        assert np.all((stationary >= 0) & (stationary <= 1))

    def test_stability_metrics(self, sample_data):
        """Test stability metric calculation."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(sample_data)

        # Get regime predictions
        regimes = detector.predict(sample_data)

        # Calculate mean duration manually
        durations = []
        current_regime = regimes[0]
        current_duration = 1

        for regime in regimes[1:]:
            if regime == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = regime
                current_duration = 1
        durations.append(current_duration)

        # Check that we have regime changes
        assert len(durations) > 1

    def test_cross_validate(self, sample_data):
        """Test cross-validation functionality."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)

        # Run cross-validation
        cv_results = detector.cross_validate(sample_data, n_splits=3)

        assert "test_scores" in cv_results
        assert "train_scores" in cv_results
        assert len(cv_results["test_scores"]) == 3
        assert len(cv_results["train_scores"]) == 3

    def test_score_method(self, sample_data):
        """Test model scoring."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(sample_data)

        # Test log likelihood scoring
        score = detector.score(sample_data)
        assert isinstance(score, float)

        # Test with different metric
        score_aic = detector.score(sample_data, metric="aic")
        assert isinstance(score_aic, float)

        score_bic = detector.score(sample_data, metric="bic")
        assert isinstance(score_bic, float)

    def test_get_diagnostics(self, sample_data):
        """Test diagnostic information retrieval."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(sample_data)

        # Get diagnostics
        diagnostics = detector.get_diagnostics()

        assert "model_type" in diagnostics
        assert diagnostics["model_type"] == "HMMRegimeDetector"
        assert "n_regimes" in diagnostics
        assert "n_features" in diagnostics
        assert "log_likelihood" in diagnostics
        assert "aic" in diagnostics
        assert "bic" in diagnostics

    def test_regime_confidence(self, sample_data):
        """Test regime confidence calculation."""
        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(sample_data)

        # Get regime confidence
        confidence = detector.get_regime_confidence(sample_data)

        assert len(confidence) == len(sample_data)
        # Confidence should be between 0 and 1
        assert np.all((confidence >= 0) & (confidence <= 1))

    def test_predict_with_uncertainty(self, sample_data):
        """Test prediction with uncertainty estimation."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_data)

        # Get predictions with probabilities
        proba = detector.predict_proba(sample_data)
        predictions = detector.predict(sample_data)

        # Max probability should correspond to prediction
        predicted_from_proba = np.argmax(proba, axis=1)
        np.testing.assert_array_equal(predictions, predicted_from_proba)

    def test_fit_with_validation(self, sample_data):
        """Test fitting with validation data."""
        # Split data
        train_data = sample_data.iloc[:80]
        val_data = sample_data.iloc[80:]

        detector = HMMRegimeDetector(n_regimes=2, random_state=42)

        # Fit on training data
        detector.fit(train_data)

        # Validate on validation data
        val_score = detector.score(val_data)
        assert isinstance(val_score, float)

    def test_regime_ordering(self, sample_data):
        """Test that regimes are ordered consistently."""
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_data)

        # Get regime statistics to check ordering
        stats = detector.get_regime_statistics(sample_data)

        # Check that regimes are numbered 0, 1, 2
        assert set(stats.keys()) == {0, 1, 2}

    def test_n_iter_parameter(self, sample_data):
        """Test different numbers of iterations."""
        # Test with very few iterations
        detector_few = HMMRegimeDetector(
            n_regimes=2, n_iter=5, random_state=42
        )
        detector_few.fit(sample_data)

        # Test with many iterations
        detector_many = HMMRegimeDetector(
            n_regimes=2, n_iter=200, random_state=42
        )
        detector_many.fit(sample_data)

        # Both should work
        assert detector_few.is_fitted
        assert detector_many.is_fitted

    def test_different_covariance_types_extended(self, sample_data):
        """Test all covariance types more thoroughly."""
        for cov_type in ["spherical", "tied", "diag", "full"]:
            detector = HMMRegimeDetector(
                n_regimes=2, covariance_type=cov_type, random_state=42
            )
            detector.fit(sample_data)
            predictions = detector.predict(sample_data)
            assert len(predictions) == len(sample_data)

    def test_edge_case_single_regime(self, sample_data):
        """Test with single regime."""
        detector = HMMRegimeDetector(n_regimes=1, random_state=42)
        detector.fit(sample_data)

        predictions = detector.predict(sample_data)
        # All predictions should be 0 (single regime)
        assert np.all(predictions == 0)

    def test_edge_case_many_regimes(self, sample_data):
        """Test with many regimes relative to data size."""
        # Use 10 regimes for 100 data points
        detector = HMMRegimeDetector(n_regimes=10, random_state=42)
        detector.fit(sample_data)

        predictions = detector.predict(sample_data)
        assert len(predictions) == len(sample_data)
        # Should have at most 10 unique regimes
        assert len(np.unique(predictions)) <= 10
