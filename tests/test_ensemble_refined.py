"""Refined tests for ensemble models following TDD principles."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from marketregimeml.models.ensemble import (
    VotingEnsemble,
    StackingEnsemble,
    BaggingEnsemble,
    BoostingEnsemble,
    WeightedEnsemble,
    HierarchicalEnsemble,
    wrap_model_for_sklearn,
)
from marketregimeml.models.sklearn_wrappers import (
    ModelWrapper,
    UnsupervisedWrapper,
)


class TestSklearnWrappers:
    """Test sklearn wrapper functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.n_regimes = 3
        model.fit = Mock(return_value=model)
        model.predict = Mock(return_value=np.array([0, 1, 2, 0, 1]))
        model.predict_proba = Mock(return_value=np.random.rand(5, 3))
        return model

    def test_model_wrapper_init(self, mock_model):
        """Test ModelWrapper initialization."""
        wrapper = ModelWrapper(mock_model, n_regimes=3)

        assert wrapper.model == mock_model
        assert wrapper.n_regimes == 3
        assert len(wrapper.classes_) == 3
        assert wrapper._estimator_type == "classifier"

    def test_model_wrapper_fit(self, mock_model):
        """Test ModelWrapper fit method."""
        wrapper = ModelWrapper(mock_model)
        X = np.random.randn(10, 5)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        result = wrapper.fit(X, y)

        assert result == wrapper
        assert np.array_equal(wrapper.classes_, np.array([0, 1, 2]))
        mock_model.fit.assert_called_once()

    def test_model_wrapper_predict(self, mock_model):
        """Test ModelWrapper predict method."""
        wrapper = ModelWrapper(mock_model)
        X = np.random.randn(5, 5)

        predictions = wrapper.predict(X)

        assert len(predictions) == 5
        mock_model.predict.assert_called_once_with(X)

    def test_unsupervised_wrapper(self, mock_model):
        """Test UnsupervisedWrapper for HMM/GMM models."""
        # Remove predict_proba to simulate unsupervised model
        del mock_model.predict_proba

        wrapper = UnsupervisedWrapper(mock_model, n_regimes=3)
        X = np.random.randn(10, 5)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        # Fit should ignore y
        wrapper.fit(X, y)
        mock_model.fit.assert_called_once_with(X)

        # Predict should work
        predictions = wrapper.predict(X)
        assert len(predictions) == 5

    def test_wrap_model_for_sklearn_helper(self, mock_model):
        """Test the wrap_model_for_sklearn helper function."""
        # Test with model that has classifier attribute (MLRegimeClassifier)
        ml_model = Mock()
        ml_model.classifier = Mock()
        wrapped = wrap_model_for_sklearn(ml_model)
        assert wrapped == ml_model.classifier

        # Test with regular model (should wrap with ModelWrapper)
        regular_model = Mock(spec=["fit", "predict"])
        wrapped = wrap_model_for_sklearn(regular_model, n_regimes=3)
        assert isinstance(wrapped, ModelWrapper)


class TestVotingEnsembleRefined:
    """Refined tests for VotingEnsemble."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 50
        n_features = 3

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1, 2], n_samples)

        return X, y

    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        models = []
        for i in range(3):
            model = Mock()
            model.n_regimes = 3
            model.fit = Mock(return_value=model)
            model.predict = Mock(return_value=np.array([0, 1, 2] * 17)[:50])
            model.is_fitted = True
            models.append((f"model_{i}", model))
        return models

    def test_voting_ensemble_init(self):
        """Test VotingEnsemble initialization."""
        ensemble = VotingEnsemble(n_regimes=4, voting="soft")

        assert ensemble.n_regimes == 4
        assert ensemble.voting == "soft"
        assert ensemble.models == []
        assert ensemble.weights is None

    def test_voting_ensemble_add_model(self):
        """Test adding models to ensemble."""
        ensemble = VotingEnsemble()

        model = Mock()
        model.n_regimes = 3

        ensemble.add_model("test_model", model)
        assert len(ensemble.models) == 1
        assert ensemble.models[0] == ("test_model", model)

        # Add with weight
        ensemble.add_model("weighted_model", model, weight=0.7)
        assert len(ensemble.models) == 2
        assert ensemble.weights is not None
        assert len(ensemble.weights) == 2

    def test_voting_ensemble_fit_with_mocks(self, sample_data, mock_models):
        """Test VotingEnsemble fit with mock models."""
        X, y = sample_data
        ensemble = VotingEnsemble(models=mock_models)

        # Mock the sklearn VotingClassifier
        with patch(
            "marketregimeml.models.ensemble.VotingClassifier"
        ) as MockVoting:
            mock_voting = Mock()
            mock_voting.fit = Mock(return_value=mock_voting)
            MockVoting.return_value = mock_voting

            ensemble.fit(X, y)

            assert ensemble.is_fitted
            assert ensemble.ensemble_ == mock_voting
            MockVoting.assert_called_once()
            mock_voting.fit.assert_called_once()

    def test_voting_ensemble_predict(self, sample_data, mock_models):
        """Test VotingEnsemble prediction."""
        X, y = sample_data
        ensemble = VotingEnsemble(models=mock_models)

        # Setup mock voting classifier
        with patch(
            "marketregimeml.models.ensemble.VotingClassifier"
        ) as MockVoting:
            mock_voting = Mock()
            mock_voting.fit = Mock(return_value=mock_voting)
            mock_voting.predict = Mock(return_value=y)
            MockVoting.return_value = mock_voting

            ensemble.fit(X, y)
            predictions = ensemble.predict(X)

            assert len(predictions) == len(X)
            mock_voting.predict.assert_called_once_with(X)

    def test_voting_ensemble_default_models(self, sample_data):
        """Test VotingEnsemble with default models."""
        X, y = sample_data
        ensemble = VotingEnsemble()

        with patch(
            "marketregimeml.models.ensemble.VotingClassifier"
        ) as MockVoting:
            mock_voting = Mock()
            mock_voting.fit = Mock(return_value=mock_voting)
            MockVoting.return_value = mock_voting

            # Should create default models
            ensemble.fit(X, y)

            assert len(ensemble.models) == 3  # HMM, GMM, ML
            assert ensemble.is_fitted


class TestStackingEnsembleRefined:
    """Refined tests for StackingEnsemble."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], 50)
        return X, y

    def test_stacking_ensemble_init(self):
        """Test StackingEnsemble initialization."""
        ensemble = StackingEnsemble(cv_folds=3)

        assert ensemble.cv_folds == 3
        assert ensemble.use_probabilities is True
        assert ensemble.models == []
        assert ensemble.meta_learner is None

    def test_stacking_ensemble_fit(self, sample_data):
        """Test StackingEnsemble fit."""
        X, y = sample_data
        ensemble = StackingEnsemble()

        with patch(
            "marketregimeml.models.ensemble.StackingClassifier"
        ) as MockStacking:
            mock_stacking = Mock()
            mock_stacking.fit = Mock(return_value=mock_stacking)
            MockStacking.return_value = mock_stacking

            ensemble.fit(X, y)

            assert ensemble.is_fitted
            assert ensemble.ensemble_ == mock_stacking
            MockStacking.assert_called_once()


class TestBaggingEnsembleRefined:
    """Refined tests for BaggingEnsemble."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], 50)
        return X, y

    def test_bagging_ensemble_init(self):
        """Test BaggingEnsemble initialization."""
        base_model = Mock()
        ensemble = BaggingEnsemble(
            base_model=base_model, n_estimators=5, max_samples=0.8
        )

        assert ensemble.base_model == base_model
        assert ensemble.n_estimators == 5
        assert ensemble.max_samples == 0.8
        assert ensemble.bootstrap is True

    def test_bagging_ensemble_fit(self, sample_data):
        """Test BaggingEnsemble fit."""
        X, y = sample_data
        ensemble = BaggingEnsemble(n_estimators=3)

        with patch(
            "marketregimeml.models.ensemble.BaggingClassifier"
        ) as MockBagging:
            mock_bagging = Mock()
            mock_bagging.fit = Mock(return_value=mock_bagging)
            MockBagging.return_value = mock_bagging

            ensemble.fit(X, y)

            assert ensemble.is_fitted
            assert ensemble.ensemble_ == mock_bagging
            MockBagging.assert_called_once()


class TestBoostingEnsembleRefined:
    """Refined tests for BoostingEnsemble."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], 50)
        return X, y

    def test_boosting_ensemble_init(self):
        """Test BoostingEnsemble initialization."""
        ensemble = BoostingEnsemble(
            algorithm="gradient", n_estimators=10, learning_rate=0.1
        )

        assert ensemble.algorithm == "gradient"
        assert ensemble.n_estimators == 10
        assert ensemble.learning_rate == 0.1

    def test_boosting_adaboost(self, sample_data):
        """Test AdaBoost ensemble."""
        X, y = sample_data
        ensemble = BoostingEnsemble(algorithm="adaboost", n_estimators=5)

        with patch(
            "marketregimeml.models.ensemble.AdaBoostClassifier"
        ) as MockAda:
            mock_ada = Mock()
            mock_ada.fit = Mock(return_value=mock_ada)
            mock_ada.predict = Mock(return_value=y)
            MockAda.return_value = mock_ada

            ensemble.fit(X, y)
            predictions = ensemble.predict(X)

            assert ensemble.is_fitted
            assert len(predictions) == len(X)
            MockAda.assert_called_once()

    def test_boosting_gradient(self, sample_data):
        """Test Gradient Boosting ensemble."""
        X, y = sample_data
        ensemble = BoostingEnsemble(algorithm="gradient", n_estimators=5)

        with patch(
            "marketregimeml.models.ensemble.GradientBoostingClassifier"
        ) as MockGB:
            mock_gb = Mock()
            mock_gb.fit = Mock(return_value=mock_gb)
            mock_gb.predict = Mock(return_value=y)
            mock_gb.feature_importances_ = np.random.rand(3)
            MockGB.return_value = mock_gb

            ensemble.fit(X, y)

            assert ensemble.is_fitted

            # Test feature importances
            importances = ensemble.get_feature_importances()
            assert len(importances) == 3


class TestWeightedEnsembleRefined:
    """Refined tests for WeightedEnsemble."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], 50)
        return X, y

    @pytest.fixture
    def mock_models(self):
        """Create mock models."""
        models = []
        for i in range(2):
            model = Mock()
            model.fit = Mock(return_value=model)
            model.predict = Mock(return_value=np.array([0, 1, 2] * 17)[:50])
            model.predict_proba = Mock(return_value=np.random.rand(50, 3))
            models.append(model)
        return models

    def test_weighted_ensemble_init(self):
        """Test WeightedEnsemble initialization."""
        models = [Mock(), Mock()]
        weights = [0.6, 0.4]

        ensemble = WeightedEnsemble(
            models=models, weights=weights, weight_method="manual"
        )

        assert ensemble.models == models
        assert ensemble.weights == weights
        assert ensemble.weight_method == "manual"

    def test_weighted_ensemble_manual_weights(self, sample_data, mock_models):
        """Test manual weight assignment."""
        X, y = sample_data
        weights = [0.7, 0.3]

        ensemble = WeightedEnsemble(
            models=mock_models, weights=weights, weight_method="manual"
        )

        ensemble.fit(X, y)

        assert ensemble.is_fitted
        assert ensemble.weights == weights
        assert len(ensemble.fitted_models_) == 2

        # Test prediction
        predictions = ensemble.predict(X)
        assert len(predictions) == len(X)

    def test_weighted_ensemble_performance_weights(
        self, sample_data, mock_models
    ):
        """Test performance-based weight calculation."""
        X, y = sample_data

        ensemble = WeightedEnsemble(
            models=mock_models, weight_method="performance"
        )

        with patch(
            "marketregimeml.models.ensemble.cross_val_score"
        ) as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.7, 0.9])

            ensemble.fit(X, y)

            assert ensemble.is_fitted
            assert ensemble.weights is not None
            assert len(ensemble.weights) == 2
            assert np.allclose(sum(ensemble.weights), 1.0)


class TestHierarchicalEnsembleRefined:
    """Refined tests for HierarchicalEnsemble."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], 50)
        return X, y

    @pytest.fixture
    def mock_models(self):
        """Create mock models for hierarchy."""
        level1 = []
        for i in range(2):
            model = Mock()
            model.fit = Mock(return_value=model)
            model.predict = Mock(return_value=np.array([0, 1, 2] * 17)[:50])
            level1.append(model)

        level2 = []
        model = Mock()
        model.fit = Mock(return_value=model)
        model.predict = Mock(return_value=np.array([1, 0, 2] * 17)[:50])
        level2.append(model)

        return [level1, level2]

    def test_hierarchical_ensemble_init(self):
        """Test HierarchicalEnsemble initialization."""
        ensemble = HierarchicalEnsemble(aggregation="voting")

        assert ensemble.aggregation == "voting"
        assert ensemble.levels == []
        assert ensemble.fitted_levels_ == []

    def test_hierarchical_ensemble_add_level(self):
        """Test adding levels to hierarchy."""
        ensemble = HierarchicalEnsemble()

        level1 = [Mock(), Mock()]
        ensemble.add_level(level1)

        assert len(ensemble.levels) == 1
        assert ensemble.levels[0] == level1

        level2 = [Mock()]
        ensemble.add_level(level2)

        assert len(ensemble.levels) == 2

    def test_hierarchical_ensemble_fit(self, sample_data, mock_models):
        """Test HierarchicalEnsemble fit."""
        X, y = sample_data

        ensemble = HierarchicalEnsemble(levels=mock_models)
        ensemble.fit(X, y)

        assert ensemble.is_fitted
        assert len(ensemble.fitted_levels_) == 2

        # Check that all models were fitted
        for level in mock_models:
            for model in level:
                model.fit.assert_called()

    def test_hierarchical_ensemble_predict(self, sample_data, mock_models):
        """Test HierarchicalEnsemble prediction."""
        X, y = sample_data

        ensemble = HierarchicalEnsemble(
            levels=mock_models, aggregation="voting"
        )
        ensemble.fit(X, y)

        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_hierarchical_ensemble_empty_levels(self, sample_data):
        """Test error handling for empty levels."""
        X, y = sample_data
        ensemble = HierarchicalEnsemble()

        with pytest.raises(
            ValueError, match="At least one level must be added"
        ):
            ensemble.fit(X, y)
