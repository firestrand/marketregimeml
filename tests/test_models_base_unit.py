"""Unit tests for base model class.

Following DRY, KISS, and SOLID principles with bottom-up testing.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import json
from pathlib import Path

from marketregimeml.models.base import BaseRegimeDetector


class ConcreteRegimeDetector(BaseRegimeDetector):
    """Concrete implementation for testing abstract base class."""
    
    def __init__(self, n_regimes=3, **kwargs):
        super().__init__(n_regimes=n_regimes, **kwargs)
        self.fitted = False
        self.training_data = None
    
    def fit(self, X, y=None):
        """Simple fit implementation."""
        self.fitted = True
        self.training_data = X
        self.n_features = X.shape[1] if len(X.shape) > 1 else 1
        return self
    
    def predict(self, X):
        """Simple predict implementation."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        n_samples = len(X)
        # Return random predictions for testing
        np.random.seed(42)
        return np.random.randint(0, self.n_regimes, n_samples)
    
    def predict_proba(self, X):
        """Simple predict_proba implementation."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        n_samples = len(X)
        # Return random probabilities
        np.random.seed(42)
        probs = np.random.rand(n_samples, self.n_regimes)
        return probs / probs.sum(axis=1, keepdims=True)
    
    def _compute_log_likelihood(self, X):
        """Compute log likelihood for AIC/BIC calculation."""
        # Simple mock implementation
        n_samples = len(X)
        return -n_samples * np.log(self.n_regimes)
    
    def _count_parameters(self):
        """Count number of model parameters."""
        # Simple mock: n_regimes * n_features
        if hasattr(self, 'n_features'):
            return self.n_regimes * self.n_features
        return self.n_regimes * 3  # default
    
    @property
    def is_fitted(self):
        return self.fitted


class TestBaseRegimeDetector:
    """Test BaseRegimeDetector functionality."""
    
    @pytest.fixture
    def model(self):
        """Create concrete model instance."""
        return ConcreteRegimeDetector(n_regimes=3, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_regimes == 3
        assert model.random_state == 42
        assert not model.is_fitted
        assert model.metadata is not None
    
    def test_fit_predict(self, model, data):
        """Test fit and predict workflow."""
        X, y = data
        
        # Fit model
        model.fit(X, y)
        assert model.is_fitted
        
        # Predict
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert predictions.min() >= 0
        assert predictions.max() < model.n_regimes
    
    def test_fit_predict_chain(self, model, data):
        """Test fit_predict method."""
        X, y = data
        
        predictions = model.fit_predict(X, y)
        
        assert model.is_fitted
        assert len(predictions) == len(X)
    
    def test_predict_proba(self, model, data):
        """Test probability prediction."""
        X, y = data
        
        model.fit(X, y)
        probs = model.predict_proba(X)
        
        assert probs.shape == (len(X), model.n_regimes)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            probs.sum(axis=1),
            np.ones(len(X))
        )
    
    def test_score(self, model, data):
        """Test scoring method."""
        X, y = data
        
        model.fit(X, y)
        score = model.score(X, y)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_get_params(self, model):
        """Test get_params method."""
        params = model.get_params()
        
        assert isinstance(params, dict)
        assert 'n_regimes' in params
        assert params['n_regimes'] == 3
    
    def test_set_params(self, model):
        """Test set_params method."""
        model.set_params(n_regimes=5)
        
        assert model.n_regimes == 5
    
    def test_regime_names(self, model):
        """Test regime name generation."""
        # Default names
        names = model.regime_names
        assert len(names) == model.n_regimes
        assert all(isinstance(name, str) for name in names)
        
        # Custom names
        custom_names = ['Low', 'Medium', 'High']
        model.set_regime_names(custom_names)
        assert model.regime_names == custom_names
    
    def test_get_regime_statistics(self, model, data):
        """Test regime statistics calculation."""
        X, y = data
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        stats = model.get_regime_statistics(X, predictions)
        
        assert isinstance(stats, dict)
        assert len(stats) == model.n_regimes
        
        for regime_id, regime_stats in stats.items():
            assert 'count' in regime_stats
            assert 'proportion' in regime_stats
            assert regime_stats['count'] >= 0
            assert 0 <= regime_stats['proportion'] <= 1
    
    def test_get_transition_matrix(self, model):
        """Test transition matrix calculation."""
        # Create sequence with known transitions
        sequence = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        
        trans_matrix = model.get_transition_matrix(sequence)
        
        assert trans_matrix.shape == (model.n_regimes, model.n_regimes)
        # Rows should sum to 1 (probabilities)
        row_sums = trans_matrix.sum(axis=1)
        valid_rows = row_sums > 0
        np.testing.assert_array_almost_equal(
            row_sums[valid_rows],
            np.ones(valid_rows.sum())
        )
    
    def test_analyze_regime_transitions(self, model):
        """Test regime transition analysis."""
        sequence = np.array([0, 0, 0, 1, 1, 2, 2, 2, 0])
        
        analysis = model.analyze_regime_transitions(sequence)
        
        assert isinstance(analysis, dict)
        assert 'transition_matrix' in analysis
        assert 'stationary_distribution' in analysis
        assert 'mean_duration' in analysis
    
    def test_save_load(self, model, tmp_path, data):
        """Test model persistence."""
        X, y = data
        model.fit(X, y)
        
        # Save
        filepath = tmp_path / "test_model.pkl"
        model.save(filepath)
        assert filepath.exists()
        
        # Load
        loaded_model = ConcreteRegimeDetector.load(filepath)
        assert loaded_model.is_fitted
        assert loaded_model.n_regimes == model.n_regimes
        
        # Check predictions match
        orig_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_equal(orig_pred, loaded_pred)
    
    def test_save_load_json(self, model, tmp_path, data):
        """Test JSON persistence."""
        X, y = data
        model.fit(X, y)
        
        # Save as JSON
        filepath = tmp_path / "test_model.json"
        model.save_model(str(filepath), format='json')
        assert filepath.exists()
        
        # Check JSON structure
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        assert 'metadata' in model_data
        assert 'n_regimes' in model_data
        assert 'regime_names' in model_data
    
    def test_get_diagnostics(self, model, data):
        """Test diagnostics generation."""
        X, y = data
        model.fit(X, y)
        
        diagnostics = model.get_diagnostics(X)
        
        assert isinstance(diagnostics, dict)
        # Should have some diagnostic information
        assert len(diagnostics) > 0
    
    def test_cross_validate(self, model, data):
        """Test cross-validation."""
        X, y = data
        
        scores = model.cross_validate(X, y, cv=3)
        
        assert isinstance(scores, dict)
        assert 'test_score' in scores
        assert len(scores['test_score']) == 3
    
    def test_auto_optimize_regimes(self, model, data):
        """Test automatic regime optimization."""
        X, y = data
        
        if hasattr(model, 'auto_optimize_regimes'):
            optimal_n = model.auto_optimize_regimes(X, min_regimes=2, max_regimes=5)
            
            assert isinstance(optimal_n, int)
            assert 2 <= optimal_n <= 5


class TestBaseRegimeDetectorEdgeCases:
    """Test edge cases for base model."""
    
    def test_empty_data(self):
        """Test with empty data."""
        model = ConcreteRegimeDetector()
        X = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            model.fit(X)
    
    def test_single_sample(self):
        """Test with single sample."""
        model = ConcreteRegimeDetector()
        X = np.array([[1, 2, 3]])
        
        model.fit(X)
        pred = model.predict(X)
        
        assert len(pred) == 1
    
    def test_invalid_n_regimes(self):
        """Test with invalid n_regimes."""
        with pytest.raises(ValueError):
            ConcreteRegimeDetector(n_regimes=0)
        
        with pytest.raises(ValueError):
            ConcreteRegimeDetector(n_regimes=-1)
    
    def test_predict_before_fit(self):
        """Test prediction before fitting."""
        model = ConcreteRegimeDetector()
        X = np.random.randn(10, 3)
        
        with pytest.raises(ValueError):
            model.predict(X)
    
    def test_mismatched_features(self):
        """Test with mismatched feature dimensions."""
        model = ConcreteRegimeDetector()
        X_train = np.random.randn(50, 3)
        X_test = np.random.randn(10, 5)  # Different number of features
        
        model.fit(X_train)
        # Should handle gracefully or raise clear error
        try:
            pred = model.predict(X_test)
            assert len(pred) == len(X_test)
        except ValueError:
            pass  # Expected for dimension mismatch


class TestRegimeNameHandling:
    """Test regime name functionality."""
    
    def test_default_regime_names(self):
        """Test default regime name generation."""
        model = ConcreteRegimeDetector(n_regimes=3)
        
        names = model.regime_names
        assert len(names) == 3
        assert all(isinstance(n, str) for n in names)
    
    def test_custom_regime_names(self):
        """Test custom regime names."""
        model = ConcreteRegimeDetector(n_regimes=3)
        custom = ['Bear', 'Neutral', 'Bull']
        
        model.set_regime_names(custom)
        assert model.regime_names == custom
    
    def test_invalid_regime_names(self):
        """Test invalid regime names."""
        model = ConcreteRegimeDetector(n_regimes=3)
        
        # Wrong number of names
        with pytest.raises(ValueError):
            model.set_regime_names(['One', 'Two'])
        
        # Non-string names
        with pytest.raises((ValueError, TypeError)):
            model.set_regime_names([1, 2, 3])


class TestMetadataHandling:
    """Test metadata functionality."""
    
    def test_metadata_initialization(self):
        """Test metadata is initialized."""
        model = ConcreteRegimeDetector()
        
        assert model.metadata is not None
        assert isinstance(model.metadata, dict)
    
    def test_metadata_update_on_fit(self):
        """Test metadata updates on fit."""
        model = ConcreteRegimeDetector()
        X = np.random.randn(100, 5)
        
        model.fit(X)
        
        # Should update fit date
        if 'fit_date' in model.metadata:
            assert model.metadata['fit_date'] is not None
    
    def test_metadata_persistence(self, tmp_path):
        """Test metadata is saved and loaded."""
        model = ConcreteRegimeDetector()
        model.metadata['custom_field'] = 'test_value'
        
        filepath = tmp_path / "model.pkl"
        model.save(filepath)
        
        loaded = ConcreteRegimeDetector.load(filepath)
        assert loaded.metadata.get('custom_field') == 'test_value'