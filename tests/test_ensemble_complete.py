"""Complete tests for ensemble models."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from marketregimeml.models.ensemble import (
    VotingEnsemble,
    StackingEnsemble,
    BaggingEnsemble,
    BoostingEnsemble,
    WeightedEnsemble,
    HierarchicalEnsemble,
    MLRegimeClassifier,  # Import from ensemble where it's aliased
)
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector


class TestVotingEnsemble:
    """Test VotingEnsemble class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1, 2], n_samples)

        return X, y

    @pytest.fixture
    def base_models(self):
        """Create base models."""
        return [
            ("hmm", HMMRegimeDetector(n_regimes=3)),
            ("gmm", GMMRegimeDetector(n_regimes=3)),
            ("svm", MLRegimeClassifier(classifier_type="svm")),
        ]

    def test_init(self):
        """Test VotingEnsemble initialization."""
        ensemble = VotingEnsemble()

        assert ensemble.models == []
        assert ensemble.voting == "hard"
        assert ensemble.weights is None
        assert ensemble.n_regimes == 3

    def test_init_with_params(self, base_models):
        """Test VotingEnsemble with parameters."""
        weights = [0.5, 0.3, 0.2]
        ensemble = VotingEnsemble(
            models=base_models, voting="soft", weights=weights, n_regimes=4
        )

        assert ensemble.models == base_models
        assert ensemble.voting == "soft"
        assert ensemble.weights == weights
        assert ensemble.n_regimes == 4

    def test_add_model(self):
        """Test adding models."""
        ensemble = VotingEnsemble()

        model1 = HMMRegimeDetector(n_regimes=3)
        ensemble.add_model("hmm", model1)

        assert len(ensemble.models) == 1
        assert ensemble.models[0] == ("hmm", model1)

        model2 = GMMRegimeDetector(n_regimes=3)
        ensemble.add_model("gmm", model2, weight=0.7)

        assert len(ensemble.models) == 2
        assert ensemble.weights is not None
        assert len(ensemble.weights) == 2

    def test_fit(self, sample_data, base_models):
        """Test fitting ensemble."""
        X, y = sample_data
        ensemble = VotingEnsemble(models=base_models)

        ensemble.fit(X, y)

        assert ensemble.is_fitted
        assert hasattr(ensemble, "ensemble_")
        assert isinstance(ensemble.ensemble_, VotingClassifier)

    def test_predict(self, sample_data, base_models):
        """Test prediction."""
        X, y = sample_data
        ensemble = VotingEnsemble(models=base_models)

        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_predict_proba(self, sample_data, base_models):
        """Test probability prediction."""
        X, y = sample_data
        ensemble = VotingEnsemble(models=base_models, voting="soft")

        ensemble.fit(X, y)
        probabilities = ensemble.predict_proba(X)

        assert probabilities.shape == (len(X), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_weighted_voting(self, sample_data):
        """Test weighted voting."""
        X, y = sample_data

        models = [
            ("model1", MLRegimeClassifier(classifier_type="logistic")),
            ("model2", MLRegimeClassifier(classifier_type="svm")),
        ]
        weights = [0.7, 0.3]

        ensemble = VotingEnsemble(
            models=models, voting="soft", weights=weights
        )

        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)


class TestStackingEnsemble:
    """Test StackingEnsemble class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1, 2], n_samples)

        return X, y

    @pytest.fixture
    def base_models(self):
        """Create base models."""
        return [
            ("rf", MLRegimeClassifier(classifier_type="random_forest")),
            ("svm", MLRegimeClassifier(classifier_type="svm")),
        ]

    def test_init(self):
        """Test StackingEnsemble initialization."""
        ensemble = StackingEnsemble()

        assert ensemble.models == []
        assert ensemble.meta_learner is None
        assert ensemble.use_probabilities is True
        assert ensemble.cv_folds == 5

    def test_init_with_params(self, base_models):
        """Test StackingEnsemble with parameters."""
        meta_learner = LogisticRegression()

        ensemble = StackingEnsemble(
            models=base_models,
            meta_learner=meta_learner,
            use_probabilities=False,
            cv_folds=3,
        )

        assert ensemble.models == base_models
        assert ensemble.meta_learner == meta_learner
        assert ensemble.use_probabilities is False
        assert ensemble.cv_folds == 3

    def test_fit(self, sample_data, base_models):
        """Test fitting stacking ensemble."""
        X, y = sample_data
        ensemble = StackingEnsemble(models=base_models)

        ensemble.fit(X, y)

        assert ensemble.is_fitted
        assert hasattr(ensemble, "ensemble_")
        assert isinstance(ensemble.ensemble_, StackingClassifier)

    def test_predict(self, sample_data, base_models):
        """Test stacking prediction."""
        X, y = sample_data
        ensemble = StackingEnsemble(models=base_models)

        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_custom_meta_learner(self, sample_data, base_models):
        """Test with custom meta-learner."""
        X, y = sample_data

        from sklearn.ensemble import RandomForestClassifier

        meta_learner = RandomForestClassifier(n_estimators=10, random_state=42)

        ensemble = StackingEnsemble(
            models=base_models, meta_learner=meta_learner
        )

        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)


class TestBaggingEnsemble:
    """Test BaggingEnsemble class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1, 2], n_samples)

        return X, y

    def test_init(self):
        """Test BaggingEnsemble initialization."""
        base_model = MLRegimeClassifier(classifier_type="decision_tree")
        ensemble = BaggingEnsemble(base_model=base_model)

        assert ensemble.base_model == base_model
        assert ensemble.n_estimators == 10
        assert ensemble.max_samples == 1.0
        assert ensemble.max_features == 1.0
        assert ensemble.bootstrap is True

    def test_fit_predict(self, sample_data):
        """Test bagging fit and predict."""
        X, y = sample_data

        base_model = MLRegimeClassifier(classifier_type="decision_tree")
        ensemble = BaggingEnsemble(
            base_model=base_model, n_estimators=5, max_samples=0.8
        )

        ensemble.fit(X, y)

        assert ensemble.is_fitted
        assert hasattr(ensemble, "ensemble_")

        predictions = ensemble.predict(X)
        assert len(predictions) == len(X)

    def test_out_of_bag_score(self, sample_data):
        """Test out-of-bag scoring."""
        X, y = sample_data

        base_model = MLRegimeClassifier(classifier_type="decision_tree")
        ensemble = BaggingEnsemble(
            base_model=base_model, n_estimators=5, oob_score=True
        )

        ensemble.fit(X, y)

        assert hasattr(ensemble, "oob_score_")
        assert 0 <= ensemble.oob_score_ <= 1


class TestBoostingEnsemble:
    """Test BoostingEnsemble class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1, 2], n_samples)

        return X, y

    def test_init(self):
        """Test BoostingEnsemble initialization."""
        ensemble = BoostingEnsemble()

        assert ensemble.algorithm == "adaboost"
        assert ensemble.n_estimators == 50
        assert ensemble.learning_rate == 1.0

    def test_adaboost(self, sample_data):
        """Test AdaBoost ensemble."""
        X, y = sample_data

        ensemble = BoostingEnsemble(
            algorithm="adaboost", n_estimators=10, learning_rate=0.5
        )

        ensemble.fit(X, y)

        assert ensemble.is_fitted
        predictions = ensemble.predict(X)
        assert len(predictions) == len(X)

    def test_gradient_boosting(self, sample_data):
        """Test Gradient Boosting ensemble."""
        X, y = sample_data

        ensemble = BoostingEnsemble(
            algorithm="gradient", n_estimators=10, learning_rate=0.1
        )

        ensemble.fit(X, y)

        assert ensemble.is_fitted
        predictions = ensemble.predict(X)
        assert len(predictions) == len(X)

    def test_feature_importances(self, sample_data):
        """Test getting feature importances."""
        X, y = sample_data

        ensemble = BoostingEnsemble(algorithm="gradient", n_estimators=10)

        ensemble.fit(X, y)

        importances = ensemble.get_feature_importances()
        assert len(importances) == X.shape[1]
        assert np.allclose(importances.sum(), 1.0)


class TestWeightedEnsemble:
    """Test WeightedEnsemble class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1, 2], n_samples)

        return X, y

    def test_init(self):
        """Test WeightedEnsemble initialization."""
        ensemble = WeightedEnsemble()

        assert ensemble.models == []
        assert ensemble.weights == []
        assert ensemble.weight_method == "manual"

    def test_manual_weights(self, sample_data):
        """Test manual weight assignment."""
        X, y = sample_data

        models = [
            MLRegimeClassifier(classifier_type="logistic"),
            MLRegimeClassifier(classifier_type="svm"),
        ]
        weights = [0.6, 0.4]

        ensemble = WeightedEnsemble(
            models=models, weights=weights, weight_method="manual"
        )

        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)

    def test_performance_weights(self, sample_data):
        """Test performance-based weights."""
        X, y = sample_data

        models = [
            MLRegimeClassifier(classifier_type="logistic"),
            MLRegimeClassifier(classifier_type="decision_tree"),
        ]

        ensemble = WeightedEnsemble(models=models, weight_method="performance")

        ensemble.fit(X, y)

        # Weights should be calculated based on performance
        assert ensemble.weights is not None
        assert len(ensemble.weights) == len(models)
        assert np.allclose(sum(ensemble.weights), 1.0)

    def test_optimize_weights(self, sample_data):
        """Test weight optimization."""
        X, y = sample_data

        models = [
            MLRegimeClassifier(classifier_type="logistic"),
            MLRegimeClassifier(classifier_type="svm"),
        ]

        ensemble = WeightedEnsemble(models=models, weight_method="optimize")

        ensemble.fit(X, y)

        # Weights should be optimized
        assert ensemble.weights is not None
        assert len(ensemble.weights) == len(models)
        assert all(0 <= w <= 1 for w in ensemble.weights)


class TestHierarchicalEnsemble:
    """Test HierarchicalEnsemble class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1, 2], n_samples)

        return X, y

    def test_init(self):
        """Test HierarchicalEnsemble initialization."""
        ensemble = HierarchicalEnsemble()

        assert ensemble.levels == []
        assert ensemble.aggregation == "voting"

    def test_add_level(self):
        """Test adding levels."""
        ensemble = HierarchicalEnsemble()

        level1 = [
            MLRegimeClassifier(classifier_type="logistic"),
            MLRegimeClassifier(classifier_type="svm"),
        ]

        ensemble.add_level(level1)
        assert len(ensemble.levels) == 1

        level2 = [MLRegimeClassifier(classifier_type="random_forest")]

        ensemble.add_level(level2)
        assert len(ensemble.levels) == 2

    def test_fit_predict(self, sample_data):
        """Test hierarchical fit and predict."""
        X, y = sample_data

        ensemble = HierarchicalEnsemble()

        # Add first level
        level1 = [
            MLRegimeClassifier(classifier_type="logistic"),
            MLRegimeClassifier(classifier_type="decision_tree"),
        ]
        ensemble.add_level(level1)

        # Add second level
        level2 = [MLRegimeClassifier(classifier_type="random_forest")]
        ensemble.add_level(level2)

        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == len(X)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_level_outputs(self, sample_data):
        """Test getting outputs from each level."""
        X, y = sample_data

        ensemble = HierarchicalEnsemble()

        level1 = [
            MLRegimeClassifier(classifier_type="logistic"),
            MLRegimeClassifier(classifier_type="svm"),
        ]
        ensemble.add_level(level1)

        ensemble.fit(X, y)

        level_outputs = ensemble.get_level_outputs(X)
        assert len(level_outputs) == 1
        assert level_outputs[0].shape[0] == len(X)
