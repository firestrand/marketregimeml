"""
Simple integration tests for deep learning models.

This version avoids complex import logic that might mask issues.
"""

import pytest
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Direct imports - let them fail if there's a real problem
from marketregimeml.models.deep_learning import (
    LSTMRegimeDetector,
    TransformerRegimeDetector,
    CNNLSTMRegimeDetector,
)


class TestDeepLearningSimple:
    """Simple integration tests for deep learning models."""

    @pytest.fixture
    def sample_data(self):
        """Create small sample dataset for fast testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        # Create synthetic data with regime patterns
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # Add some structure to simulate regimes
        data.iloc[:33] *= 0.5  # Low volatility regime
        data.iloc[33:66] *= 1.5  # High volatility regime
        data.iloc[66:] *= 1.0  # Normal regime

        return data

    def test_lstm_model(self, sample_data):
        """Test LSTM model basic workflow."""
        model = LSTMRegimeDetector(
            n_regimes=3,
            hidden_size=16,
            num_layers=1,
            dropout=0.1,
            learning_rate=0.01,
            batch_size=16,
            epochs=2,
            sequence_length=10,
            random_state=42,
        )

        # Test fitting
        model.fit(sample_data)
        assert model.is_fitted

        # Test prediction
        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert all(0 <= p < 3 for p in predictions)

        # Test probability prediction
        probabilities = model.predict_proba(sample_data)
        assert probabilities.shape == (len(sample_data), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)

    def test_transformer_model(self, sample_data):
        """Test Transformer model basic workflow."""
        model = TransformerRegimeDetector(
            n_regimes=3,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            dropout=0.1,
            learning_rate=0.01,
            batch_size=16,
            epochs=2,
            sequence_length=10,
            random_state=42,
        )

        # Test fitting
        model.fit(sample_data)
        assert model.is_fitted

        # Test predictions
        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

        probabilities = model.predict_proba(sample_data)
        assert probabilities.shape == (len(sample_data), 3)

    def test_cnn_lstm_model(self, sample_data):
        """Test CNN-LSTM hybrid model."""
        model = CNNLSTMRegimeDetector(
            n_regimes=3,
            num_filters=8,
            filter_size=3,
            lstm_hidden_size=16,
            lstm_num_layers=1,
            dropout=0.1,
            learning_rate=0.01,
            batch_size=16,
            epochs=2,
            sequence_length=10,
            random_state=42,
        )

        # Test workflow
        model.fit(sample_data)
        assert model.is_fitted

        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

        probabilities = model.predict_proba(sample_data)
        assert probabilities.shape == (len(sample_data), 3)
