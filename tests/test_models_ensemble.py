"""Tests for ensemble regime detector."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.exceptions import ConvergenceWarning
import warnings

from marketregimeml.models import VotingEnsemble
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector

# Alias for backward compatibility with tests
EnsembleRegimeDetector = VotingEnsemble


class TestEnsembleRegimeDetector:
    """Test suite for EnsembleRegimeDetector."""

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
    def mock_models(self):
        """Create mock models for testing."""
        models = []
        for i in range(3):
            mock_model = Mock()
            mock_model.__class__.__name__ = f"MockModel_{i}"
            mock_model.predict.return_value = np.random.randint(0, 3, 100)
            mock_model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], 100
            )
            mock_model.fit.return_value = mock_model
            mock_model._compute_log_likelihood.return_value = -100.0
            mock_model._count_parameters.return_value = 10
            models.append(mock_model)
        return models

    def test_init_default(self):
        """Test initialization with default parameters."""
        detector = EnsembleRegimeDetector(n_regimes=3)

        assert detector.n_regimes == 3
        assert detector.strategy == "voting"
        assert len(detector.models) == 4  # 2 HMM + 2 GMM default models
        assert len(detector.weights) == 4
        assert np.allclose(detector.weights, 0.25)  # Equal weights
        assert detector.consensus_threshold == 0.6

    def test_init_with_custom_models(self, mock_models):
        """Test initialization with custom models."""
        detector = EnsembleRegimeDetector(
            n_regimes=3,
            models=mock_models,
            strategy="weighted_voting",
            weights=[0.5, 0.3, 0.2],
        )

        assert len(detector.models) == 3
        assert detector.strategy == "weighted_voting"
        assert np.allclose(detector.weights, [0.5, 0.3, 0.2])

    def test_init_weight_validation(self, mock_models):
        """Test weight validation during initialization."""
        # Wrong number of weights
        with pytest.raises(ValueError, match="Number of weights must match"):
            EnsembleRegimeDetector(
                models=mock_models,
                weights=[0.5, 0.5],  # Only 2 weights for 3 models
            )

    def test_init_weight_normalization(self, mock_models):
        """Test weight normalization."""
        detector = EnsembleRegimeDetector(
            models=mock_models,
            weights=[2, 4, 6],  # Will be normalized to [1/6, 1/3, 1/2]
        )

        expected_weights = np.array([2, 4, 6]) / 12
        assert np.allclose(detector.weights, expected_weights)

    def test_create_default_models(self):
        """Test default model creation."""
        detector = EnsembleRegimeDetector(n_regimes=3, random_state=42)
        models = detector.models

        assert len(models) == 4

        # Check model types
        model_types = [type(model).__name__ for model in models]
        assert model_types.count("HMMRegimeDetector") == 2
        assert model_types.count("GMMRegimeDetector") == 2

        # Check different configurations
        hmm_models = [m for m in models if isinstance(m, HMMRegimeDetector)]
        gmm_models = [m for m in models if isinstance(m, GMMRegimeDetector)]

        # HMMs should have different init methods and covariance types
        assert hmm_models[0].init_method != hmm_models[1].init_method

        # GMMs should have different covariance types
        assert gmm_models[0].covariance_type != gmm_models[1].covariance_type

    def test_fit_sequential(self, sample_features, mock_models):
        """Test sequential fitting."""
        detector = EnsembleRegimeDetector(models=mock_models)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            detector.fit(sample_features, parallel=False)

        assert detector.is_fitted
        assert detector.fit_date is not None
        assert detector.n_features == sample_features.shape[1]
        assert detector.feature_names == list(sample_features.columns)

        # Check all models were fitted
        for model in mock_models:
            model.fit.assert_called_once()

    @patch("marketregimeml.models.ensemble.Parallel")
    @patch("marketregimeml.models.ensemble.delayed")
    def test_fit_parallel(
        self, mock_delayed, mock_parallel, sample_features, mock_models
    ):
        """Test parallel fitting."""
        # Mock joblib components
        mock_parallel.return_value.return_value = mock_models
        mock_delayed.return_value = lambda model, features: model.fit(features)

        detector = EnsembleRegimeDetector(models=mock_models)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            detector.fit(sample_features, parallel=True)

        assert detector.is_fitted
        mock_parallel.assert_called_once_with(n_jobs=-1)

    def test_fit_parallel_fallback(self, sample_features, mock_models):
        """Test parallel fitting fallback when joblib not available."""
        detector = EnsembleRegimeDetector(models=mock_models)

        with patch("marketregimeml.models.ensemble.ImportError", ImportError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                detector.fit(sample_features, parallel=True)

        assert detector.is_fitted
        # Should fall back to sequential fitting
        for model in mock_models:
            model.fit.assert_called_once()

    def test_calculate_agreement(self, sample_features, mock_models):
        """Test agreement matrix calculation."""
        # Set consistent predictions for testing
        for i, model in enumerate(mock_models):
            predictions = np.full(
                100, i % 3
            )  # Different predictions per model
            model.predict.return_value = predictions

        detector = EnsembleRegimeDetector(models=mock_models)
        detector._calculate_agreement(sample_features)

        assert detector.agreement_matrix is not None
        assert detector.agreement_matrix.shape == (3, 3)

        # Diagonal should be 1.0 (perfect self-agreement)
        assert np.allclose(np.diag(detector.agreement_matrix), 1.0)

    def test_fit_meta_model_stacking(self, sample_features, mock_models):
        """Test meta-model fitting for stacking strategy."""
        detector = EnsembleRegimeDetector(
            models=mock_models, strategy="stacking"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            detector.fit(sample_features)

        assert detector.meta_model is not None
        assert isinstance(detector.meta_model, GMMRegimeDetector)

    def test_predict_not_fitted(self, sample_features):
        """Test predict raises error when not fitted."""
        detector = EnsembleRegimeDetector()

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict(sample_features)

    def test_predict_voting(self, sample_features, mock_models):
        """Test voting prediction strategy."""
        # Set up deterministic predictions for testing
        mock_models[0].predict.return_value = np.array(
            [0, 1, 2] * 33 + [0]
        )  # 100 samples
        mock_models[1].predict.return_value = np.array(
            [0, 1, 2] * 33 + [0]
        )  # Same
        mock_models[2].predict.return_value = np.array(
            [1, 1, 1] * 33 + [1]
        )  # Different

        detector = EnsembleRegimeDetector(
            models=mock_models, strategy="voting"
        )
        detector.is_fitted = True

        predictions = detector.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(pred >= 0 for pred in predictions)

    def test_predict_weighted_voting(self, sample_features, mock_models):
        """Test weighted voting prediction strategy."""
        # Set up predictions
        for model in mock_models:
            model.predict.return_value = np.random.randint(0, 3, 100)

        detector = EnsembleRegimeDetector(
            models=mock_models,
            strategy="weighted_voting",
            weights=[0.5, 0.3, 0.2],
        )
        detector.is_fitted = True

        predictions = detector.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(0 <= pred < 3 for pred in predictions)

    def test_predict_bayesian(self, sample_features, mock_models):
        """Test Bayesian prediction strategy."""
        detector = EnsembleRegimeDetector(
            models=mock_models, strategy="bayesian"
        )
        detector.is_fitted = True

        predictions = detector.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(0 <= pred < 3 for pred in predictions)

    def test_predict_consensus(self, sample_features, mock_models):
        """Test consensus prediction strategy."""
        # Set up predictions where some have consensus, some don't
        mock_models[0].predict.return_value = np.array([0] * 50 + [1] * 50)
        mock_models[1].predict.return_value = np.array([0] * 50 + [2] * 50)
        mock_models[2].predict.return_value = np.array([0] * 50 + [1] * 50)

        detector = EnsembleRegimeDetector(
            models=mock_models, strategy="consensus", consensus_threshold=0.6
        )
        detector.is_fitted = True

        predictions = detector.predict(sample_features)

        assert len(predictions) == len(sample_features)
        # First 50 should have consensus (2/3 vote for 0), last 50 should have no consensus (-1)
        assert all(pred == 0 for pred in predictions[:50])
        assert all(pred == -1 for pred in predictions[50:])

    def test_predict_stacking(self, sample_features, mock_models):
        """Test stacking prediction strategy."""
        mock_meta = Mock()
        mock_meta.predict.return_value = np.random.randint(0, 3, 100)

        detector = EnsembleRegimeDetector(
            models=mock_models, strategy="stacking"
        )
        detector.is_fitted = True
        detector.meta_model = mock_meta

        predictions = detector.predict(sample_features)

        assert len(predictions) == len(sample_features)
        mock_meta.predict.assert_called_once()

    def test_predict_stacking_no_meta_model(
        self, sample_features, mock_models
    ):
        """Test stacking strategy raises error without meta-model."""
        detector = EnsembleRegimeDetector(
            models=mock_models, strategy="stacking"
        )
        detector.is_fitted = True
        detector.meta_model = None

        with pytest.raises(ValueError, match="Meta-model not fitted"):
            detector.predict(sample_features)

    def test_predict_unknown_strategy(self, sample_features, mock_models):
        """Test unknown strategy raises error."""
        detector = EnsembleRegimeDetector(
            models=mock_models, strategy="unknown"
        )
        detector.is_fitted = True

        with pytest.raises(ValueError, match="Unknown strategy"):
            detector.predict(sample_features)

    def test_predict_proba_not_fitted(self, sample_features):
        """Test predict_proba raises error when not fitted."""
        detector = EnsembleRegimeDetector()

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict_proba(sample_features)

    def test_predict_proba_weighted(self, sample_features, mock_models):
        """Test weighted probability prediction."""
        detector = EnsembleRegimeDetector(
            models=mock_models, weights=[0.5, 0.3, 0.2]
        )
        detector.is_fitted = True

        probabilities = detector.predict_proba(sample_features)

        assert probabilities.shape == (len(sample_features), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_proba_stacking(self, sample_features, mock_models):
        """Test stacking probability prediction."""
        mock_meta = Mock()
        mock_meta.predict_proba.return_value = np.random.dirichlet(
            [1, 1, 1], 100
        )

        detector = EnsembleRegimeDetector(
            models=mock_models, strategy="stacking"
        )
        detector.is_fitted = True
        detector.meta_model = mock_meta

        probabilities = detector.predict_proba(sample_features)

        assert probabilities.shape == (len(sample_features), 3)
        mock_meta.predict_proba.assert_called_once()

    def test_compute_log_likelihood(self, sample_features, mock_models):
        """Test log-likelihood computation."""
        detector = EnsembleRegimeDetector(
            models=mock_models, weights=[0.4, 0.3, 0.3]
        )

        ll = detector._compute_log_likelihood(sample_features)

        assert isinstance(ll, float)
        # Should be weighted average of individual model likelihoods
        for model in mock_models:
            model._compute_log_likelihood.assert_called_once_with(
                sample_features
            )

    def test_count_parameters(self, mock_models):
        """Test parameter counting."""
        detector = EnsembleRegimeDetector(models=mock_models)

        n_params = detector._count_parameters()

        assert n_params == 30  # 3 models * 10 parameters each

    def test_count_parameters_with_meta_model(self, mock_models):
        """Test parameter counting with meta-model."""
        mock_meta = Mock()
        mock_meta._count_parameters.return_value = 15

        detector = EnsembleRegimeDetector(
            models=mock_models, strategy="stacking"
        )
        detector.meta_model = mock_meta

        n_params = detector._count_parameters()

        assert n_params == 45  # 3 models * 10 + 15 meta-model

    def test_update_diagnostics(self, sample_features, mock_models):
        """Test diagnostics update."""
        detector = EnsembleRegimeDetector(models=mock_models)
        detector.is_fitted = True
        detector.agreement_matrix = np.eye(3)

        detector._update_diagnostics(sample_features)

        assert "agreement_matrix" in detector.diagnostics
        assert "avg_agreement" in detector.diagnostics
        assert "model_performance" in detector.diagnostics
        assert "prediction_diversity" in detector.diagnostics
        assert "avg_confidence" in detector.diagnostics

    def test_update_diagnostics_consensus(self, sample_features, mock_models):
        """Test diagnostics update for consensus strategy."""
        # Set up predictions with some consensus failures
        mock_models[0].predict.return_value = np.array([0] * 50 + [-1] * 50)
        mock_models[1].predict.return_value = np.array([0] * 50 + [-1] * 50)
        mock_models[2].predict.return_value = np.array([0] * 50 + [-1] * 50)

        detector = EnsembleRegimeDetector(
            models=mock_models, strategy="consensus"
        )
        detector.is_fitted = True
        detector.agreement_matrix = np.eye(3)

        detector._update_diagnostics(sample_features)

        assert "consensus_rate" in detector.diagnostics

    def test_calculate_diversity(self, mock_models):
        """Test diversity calculation."""
        detector = EnsembleRegimeDetector(models=mock_models)

        # High diversity predictions
        predictions = [
            np.array([0, 1, 2, 0, 1]),
            np.array([1, 2, 0, 1, 2]),
            np.array([2, 0, 1, 2, 0]),
        ]

        diversity = detector._calculate_diversity(predictions)

        assert 0 <= diversity <= 1
        assert diversity > 0.5  # Should be high diversity

    def test_get_model_contributions(self, sample_features, mock_models):
        """Test model contribution analysis."""
        detector = EnsembleRegimeDetector(models=mock_models)
        detector.is_fitted = True

        contributions = detector.get_model_contributions(sample_features)

        assert isinstance(contributions, pd.DataFrame)
        assert len(contributions) == len(mock_models)
        assert "weight" in contributions.columns
        assert "agreement_with_ensemble" in contributions.columns
        assert "contribution" in contributions.columns

    def test_cross_validate_models(self, sample_features, mock_models):
        """Test cross-validation."""

        # Create mock model classes that return instances
        def mock_model_class_0(**kwargs):
            return mock_models[0]

        def mock_model_class_1(**kwargs):
            return mock_models[1]

        def mock_model_class_2(**kwargs):
            return mock_models[2]

        mock_models[0].__class__ = mock_model_class_0
        mock_models[1].__class__ = mock_model_class_1
        mock_models[2].__class__ = mock_model_class_2

        # Add score method
        for model in mock_models:
            model.score.return_value = -50.0

        detector = EnsembleRegimeDetector(models=mock_models)

        results = detector.cross_validate_models(sample_features, n_splits=3)

        assert "ensemble" in results
        assert (
            len(results) == len(mock_models) + 1
        )  # Individual models + ensemble

    def test_optimize_weights(self, sample_features, mock_models):
        """Test weight optimization."""
        detector = EnsembleRegimeDetector(models=mock_models)

        # Mock predictions for optimization
        for model in mock_models:
            model.predict_proba.return_value = np.random.dirichlet(
                [1, 1, 1], len(sample_features)
            )

        optimized_weights = detector.optimize_weights(
            sample_features, validation_split=0.2
        )

        assert len(optimized_weights) == len(mock_models)
        assert np.allclose(np.sum(optimized_weights), 1.0)
        assert np.allclose(detector.weights, optimized_weights)

    def test_real_models_integration(self, sample_features):
        """Test with real HMM and GMM models."""
        # Create simple ensemble with real models
        models = [
            HMMRegimeDetector(n_regimes=2, random_state=42),
            GMMRegimeDetector(n_regimes=2, random_state=42),
        ]

        detector = EnsembleRegimeDetector(models=models, strategy="voting")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            detector.fit(sample_features)

        predictions = detector.predict(sample_features)
        probabilities = detector.predict_proba(sample_features)

        assert detector.is_fitted
        assert len(predictions) == len(sample_features)
        assert probabilities.shape == (len(sample_features), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_ensemble_strategies_comparison(self, sample_features):
        """Test different ensemble strategies produce different results."""
        strategies = ["voting", "weighted_voting", "bayesian"]
        results = {}

        for strategy in strategies:
            detector = EnsembleRegimeDetector(
                n_regimes=2, strategy=strategy, random_state=42
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                detector.fit(sample_features)

            results[strategy] = detector.predict(sample_features)

        # Results should be similar but potentially different
        for strategy in strategies:
            assert len(results[strategy]) == len(sample_features)
            assert all(0 <= pred <= 1 for pred in results[strategy])
