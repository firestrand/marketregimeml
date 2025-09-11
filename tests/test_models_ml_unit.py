"""Unit tests for ML-based regime detection models.

Following DRY, KISS, and SOLID principles with bottom-up testing.
Minimal mocking, using real calculations where possible.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

from marketregimeml.models.ml import (
    RandomForestRegimeClassifier,
    SVMRegimeClassifier,
    XGBoostRegimeClassifier,
    HAS_XGBOOST,
)


class TestDataHelper:
    """DRY: Centralized test data generation for ML model tests."""
    
    @staticmethod
    def create_regime_features(n_samples=200, n_features=5, n_regimes=3):
        """Create features with clear regime patterns."""
        np.random.seed(42)
        
        samples_per_regime = n_samples // n_regimes
        features = []
        labels = []
        
        for regime in range(n_regimes):
            # Each regime has different statistical properties
            mean = np.array([regime * 2] * n_features)
            std = 0.5 + regime * 0.3
            
            regime_data = np.random.normal(mean, std, (samples_per_regime, n_features))
            features.append(regime_data)
            labels.extend([regime] * samples_per_regime)
        
        # Add remaining samples to last regime
        remaining = n_samples - len(labels)
        if remaining > 0:
            mean = np.array([(n_regimes-1) * 2] * n_features)
            std = 0.5 + (n_regimes-1) * 0.3
            extra_data = np.random.normal(mean, std, (remaining, n_features))
            features.append(extra_data)
            labels.extend([n_regimes-1] * remaining)
        
        return np.vstack(features), np.array(labels)
    
    @staticmethod
    def create_time_series_features(n_samples=100):
        """Create time series features for testing."""
        np.random.seed(42)
        
        # Create features with temporal patterns
        t = np.linspace(0, 10, n_samples)
        
        # Feature 1: Trending
        feature1 = t + np.random.normal(0, 0.5, n_samples)
        
        # Feature 2: Cyclical
        feature2 = np.sin(t) + np.random.normal(0, 0.2, n_samples)
        
        # Feature 3: Mean-reverting
        feature3 = np.cumsum(np.random.normal(0, 1, n_samples))
        feature3 = feature3 - pd.Series(feature3).rolling(20, center=True).mean().fillna(0)
        
        # Feature 4: Volatility clustering
        volatility = np.where(t < 5, 0.5, 1.5)
        feature4 = np.random.normal(0, volatility)
        
        # Feature 5: Random noise
        feature5 = np.random.normal(0, 1, n_samples)
        
        return np.column_stack([feature1, feature2, feature3, feature4, feature5])


class TestRandomForestRegimeClassifier:
    """Test Random Forest regime classifier."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return RandomForestRegimeClassifier(
            n_regimes=3,
            n_estimators=10,  # Small for fast tests
            random_state=42
        )
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        return TestDataHelper.create_regime_features(n_samples=150)
    
    def test_init(self, model):
        """Test model initialization."""
        assert model.n_regimes == 3
        assert model.n_estimators == 10
        assert model.random_state == 42
        assert model.model_ is None
        assert not model.is_fitted
    
    def test_fit_supervised(self, model, data):
        """Test supervised fitting with labels."""
        features, labels = data
        
        model.fit(features, y=labels)
        
        assert model.is_fitted
        assert model.model_ is not None
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == features.shape[1]
        assert model.cluster_labels_ is None  # No clustering in supervised mode
    
    def test_fit_unsupervised(self, model):
        """Test unsupervised fitting without labels."""
        features = TestDataHelper.create_time_series_features(100)
        
        model.fit(features)
        
        assert model.is_fitted
        assert model.model_ is not None
        assert model.cluster_labels_ is not None
        assert len(model.cluster_labels_) == len(features)
        assert len(np.unique(model.cluster_labels_)) <= model.n_regimes
    
    def test_predict(self, model, data):
        """Test prediction after fitting."""
        features, labels = data
        
        model.fit(features, y=labels)
        predictions = model.predict(features)
        
        assert len(predictions) == len(features)
        assert predictions.min() >= 0
        assert predictions.max() < model.n_regimes
        
        # Should have reasonable accuracy on training data
        accuracy = np.mean(predictions == labels)
        assert accuracy > 0.7
    
    def test_predict_proba(self, model, data):
        """Test probability prediction."""
        features, labels = data
        
        model.fit(features, y=labels)
        probabilities = model.predict_proba(features)
        
        assert probabilities.shape == (len(features), model.n_regimes)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1),
            np.ones(len(features))
        )
        # All probabilities should be between 0 and 1
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()
    
    def test_feature_importances(self, model, data):
        """Test feature importance calculation."""
        features, labels = data
        
        model.fit(features, y=labels)
        
        importances = model.feature_importances_
        assert len(importances) == features.shape[1]
        assert (importances >= 0).all()
        assert importances.sum() > 0
    
    def test_oob_score(self):
        """Test out-of-bag score calculation."""
        model = RandomForestRegimeClassifier(
            n_regimes=3,
            n_estimators=10,
            oob_score=True,
            random_state=42
        )
        
        features, labels = TestDataHelper.create_regime_features()
        model.fit(features, y=labels)
        
        assert hasattr(model, 'oob_score_')
        assert 0 <= model.oob_score_ <= 1
    
    def test_score(self, model, data):
        """Test scoring method."""
        features, labels = data
        
        model.fit(features, y=labels)
        score = model.score(features, labels)
        
        assert 0 <= score <= 1
        # Should have good score on training data
        assert score > 0.7
    
    def test_cross_validate(self, model):
        """Test cross-validation."""
        features, labels = TestDataHelper.create_regime_features(n_samples=100)
        
        scores = model.cross_validate(features, labels, cv=3)
        
        assert 'test_score' in scores
        assert 'train_score' in scores
        assert len(scores['test_score']) == 3
        assert all(0 <= s <= 1 for s in scores['test_score'])
    
    def test_tune_hyperparameters(self, model):
        """Test hyperparameter tuning."""
        features, labels = TestDataHelper.create_regime_features(n_samples=100)
        
        # The model has tune_hyperparameters method
        if hasattr(model, 'tune_hyperparameters'):
            param_grid = {
                'n_estimators': [5, 10],
                'max_depth': [3, 5]
            }
            
            best_params = model.tune_hyperparameters(features, labels, param_grid)
            
            assert isinstance(best_params, dict)
    
    def test_persistence(self, model, tmp_path, data):
        """Test model save and load."""
        features, labels = data
        model.fit(features, y=labels)
        
        # Save
        filepath = tmp_path / "rf_model.pkl"
        model.save(filepath)
        assert filepath.exists()
        
        # Load
        loaded_model = RandomForestRegimeClassifier.load(filepath)
        assert loaded_model.is_fitted
        
        # Should give same predictions
        original_pred = model.predict(features)
        loaded_pred = loaded_model.predict(features)
        np.testing.assert_array_equal(original_pred, loaded_pred)


class TestSVMRegimeClassifier:
    """Test SVM regime classifier."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return SVMRegimeClassifier(
            n_regimes=3,
            kernel='rbf',
            C=1.0,
            random_state=42
        )
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        return TestDataHelper.create_regime_features(n_samples=100)
    
    def test_init(self, model):
        """Test model initialization."""
        assert model.n_regimes == 3
        assert model.kernel == 'rbf'
        assert model.C == 1.0
        assert model.model_ is None
    
    def test_fit_predict(self, model, data):
        """Test fit and predict."""
        features, labels = data
        
        model.fit(features, y=labels)
        predictions = model.predict(features)
        
        assert model.is_fitted
        assert len(predictions) == len(features)
        assert predictions.min() >= 0
        assert predictions.max() < model.n_regimes
    
    def test_predict_proba_with_probability(self):
        """Test probability prediction with probability=True."""
        model = SVMRegimeClassifier(
            n_regimes=3,
            kernel='rbf',
            probability=True,  # Enable probability
            random_state=42
        )
        
        features, labels = TestDataHelper.create_regime_features(n_samples=100)
        model.fit(features, y=labels)
        
        probabilities = model.predict_proba(features)
        
        assert probabilities.shape == (len(features), model.n_regimes)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1),
            np.ones(len(features)),
            decimal=5
        )
    
    def test_predict_proba_without_probability(self, model, data):
        """Test probability prediction without probability=True."""
        features, labels = data
        model.fit(features, y=labels)
        
        # Should use decision function fallback
        probabilities = model.predict_proba(features)
        
        assert probabilities.shape == (len(features), model.n_regimes)
        # Should still sum to 1 (softmax of decision function)
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1),
            np.ones(len(features))
        )
    
    def test_linear_kernel(self):
        """Test with linear kernel."""
        model = SVMRegimeClassifier(
            n_regimes=2,
            kernel='linear',
            random_state=42
        )
        
        # Create linearly separable data
        features = np.vstack([
            np.random.randn(50, 2) - 2,
            np.random.randn(50, 2) + 2
        ])
        labels = np.array([0] * 50 + [1] * 50)
        
        model.fit(features, y=labels)
        predictions = model.predict(features)
        
        # Should achieve perfect separation
        accuracy = np.mean(predictions == labels)
        assert accuracy > 0.95
    
    def test_unsupervised_mode(self, model):
        """Test unsupervised fitting."""
        features = TestDataHelper.create_time_series_features(100)
        
        model.fit(features)
        
        assert model.is_fitted
        assert model.cluster_labels_ is not None
        predictions = model.predict(features)
        assert len(predictions) == len(features)


@pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not installed")
class TestXGBoostRegimeClassifier:
    """Test XGBoost regime classifier."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return XGBoostRegimeClassifier(
            n_regimes=3,
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        return TestDataHelper.create_regime_features(n_samples=150)
    
    def test_init(self, model):
        """Test model initialization."""
        assert model.n_regimes == 3
        assert model.n_estimators == 10
        assert model.max_depth == 3
        assert model.learning_rate == 0.1
    
    def test_fit_predict(self, model, data):
        """Test fit and predict."""
        features, labels = data
        
        model.fit(features, y=labels)
        predictions = model.predict(features)
        
        assert model.is_fitted
        assert len(predictions) == len(features)
        assert predictions.min() >= 0
        assert predictions.max() < model.n_regimes
    
    def test_predict_proba(self, model, data):
        """Test probability prediction."""
        features, labels = data
        
        model.fit(features, y=labels)
        probabilities = model.predict_proba(features)
        
        assert probabilities.shape == (len(features), model.n_regimes)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1),
            np.ones(len(features)),
            decimal=5
        )
    
    def test_feature_importances(self, model, data):
        """Test feature importance from XGBoost."""
        features, labels = data
        
        model.fit(features, y=labels)
        
        assert hasattr(model, 'feature_importances_')
        importances = model.feature_importances_
        assert len(importances) == features.shape[1]
        assert (importances >= 0).all()
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        model = XGBoostRegimeClassifier(
            n_regimes=3,
            n_estimators=100,
            early_stopping_rounds=10,
            random_state=42
        )
        
        features, labels = TestDataHelper.create_regime_features(n_samples=200)
        
        # Split data for validation
        n_train = 150
        X_train, X_val = features[:n_train], features[n_train:]
        y_train, y_val = labels[:n_train], labels[n_train:]
        
        model.fit(X_train, y=y_train, eval_set=[(X_val, y_val)])
        
        assert model.is_fitted
        # Should have stopped early
        assert hasattr(model.model_, 'best_iteration')


class TestMLModelComparison:
    """Test comparison between different ML models."""
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        return TestDataHelper.create_regime_features(n_samples=200)
    
    def test_model_consistency(self, data):
        """Test that different models give similar results on same data."""
        features, labels = data
        
        models = [
            RandomForestRegimeClassifier(n_regimes=3, random_state=42),
            SVMRegimeClassifier(n_regimes=3, random_state=42),
        ]
        
        if HAS_XGBOOST:
            models.append(XGBoostRegimeClassifier(n_regimes=3, random_state=42))
        
        predictions = []
        for model in models:
            model.fit(features, y=labels)
            pred = model.predict(features)
            predictions.append(pred)
        
        # Models should agree on most predictions
        for i in range(len(predictions) - 1):
            agreement = np.mean(predictions[i] == predictions[i+1])
            assert agreement > 0.6  # At least 60% agreement
    
    def test_unsupervised_consistency(self):
        """Test unsupervised mode consistency."""
        features = TestDataHelper.create_time_series_features(100)
        
        models = [
            RandomForestRegimeClassifier(n_regimes=3, random_state=42),
            SVMRegimeClassifier(n_regimes=3, random_state=42),
        ]
        
        for model in models:
            model.fit(features)  # Unsupervised
            
            assert model.is_fitted
            assert model.cluster_labels_ is not None
            
            predictions = model.predict(features)
            assert len(predictions) == len(features)
            assert len(np.unique(predictions)) <= model.n_regimes


class TestMLModelEdgeCases:
    """Test edge cases for ML models."""
    
    def test_single_feature(self):
        """Test with single feature."""
        model = RandomForestRegimeClassifier(n_regimes=2, random_state=42)
        
        features = np.random.randn(100, 1)  # Single feature
        labels = np.random.randint(0, 2, 100)
        
        model.fit(features, y=labels)
        predictions = model.predict(features)
        
        assert len(predictions) == len(features)
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        model = RandomForestRegimeClassifier(n_regimes=2, n_estimators=5)
        
        features = np.random.randn(10, 3)
        labels = np.array([0, 1] * 5)
        
        model.fit(features, y=labels)
        predictions = model.predict(features)
        
        assert len(predictions) == len(features)
    
    def test_imbalanced_classes(self):
        """Test with imbalanced classes."""
        model = RandomForestRegimeClassifier(n_regimes=3, random_state=42)
        
        # Create imbalanced dataset
        features = np.vstack([
            np.random.randn(80, 3),  # Class 0: 80 samples
            np.random.randn(15, 3) + 2,  # Class 1: 15 samples
            np.random.randn(5, 3) + 4,  # Class 2: 5 samples
        ])
        labels = np.array([0]*80 + [1]*15 + [2]*5)
        
        model.fit(features, y=labels)
        predictions = model.predict(features)
        
        # Should predict all classes despite imbalance
        unique_predictions = np.unique(predictions)
        assert len(unique_predictions) >= 2  # At least 2 classes predicted
    
    def test_constant_features(self):
        """Test with constant features."""
        model = RandomForestRegimeClassifier(n_regimes=2)
        
        # Some features are constant
        features = np.column_stack([
            np.ones(100),  # Constant feature
            np.random.randn(100),  # Variable feature
            np.ones(100) * 5,  # Another constant
        ])
        labels = np.random.randint(0, 2, 100)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(features, y=labels)
            predictions = model.predict(features)
        
        assert len(predictions) == len(features)