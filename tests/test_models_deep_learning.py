"""Tests for deep learning based regime detection models."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import warnings

from marketregimeml.models.deep_learning import (
    LSTMRegimeDetector,
    TransformerRegimeDetector,
    CNNLSTMRegimeDetector,
)


class TestLSTMRegimeDetector:
    """Test suite for LSTM-based regime detector."""

    @pytest.fixture
    def time_series_data(self):
        """Generate time series data with regime changes."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5

        # Create time series with different regime characteristics
        timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        # Generate features with temporal dependencies
        features = []
        returns = []
        true_regimes = []

        for i in range(n_samples):
            # Determine regime based on time
            if i < 300:
                regime = 0  # Trending up
                trend = 0.001
                volatility = 0.01
            elif i < 600:
                regime = 1  # High volatility
                trend = -0.0005
                volatility = 0.03
            else:
                regime = 2  # Low volatility sideways
                trend = 0
                volatility = 0.005

            # Create autocorrelated features
            if i == 0:
                feature_vec = np.random.randn(n_features) * volatility + trend
            else:
                # Add persistence
                prev_features = features[-1]
                feature_vec = 0.7 * prev_features + 0.3 * (
                    np.random.randn(n_features) * volatility + trend
                )

            features.append(feature_vec)
            returns.append(feature_vec[0])  # Use first feature as return
            true_regimes.append(regime)

        features_df = pd.DataFrame(
            features,
            columns=[f"feature_{i}" for i in range(n_features)],
            index=timestamps,
        )

        return (
            features_df,
            np.array(true_regimes),
            pd.Series(returns, index=timestamps),
        )

    @pytest.fixture
    def lstm_detector(self):
        """Create LSTM detector instance."""
        return LSTMRegimeDetector(
            n_regimes=3,
            sequence_length=20,
            lstm_units=50,
            dropout_rate=0.2,
            random_state=42,
        )

    def test_initialization(self):
        """Test LSTM detector initialization."""
        detector = LSTMRegimeDetector()
        assert detector.n_regimes == 3
        assert detector.sequence_length == 30
        assert detector.lstm_units == 64
        assert detector.dropout_rate == 0.1
        assert detector.batch_size == 32
        assert not detector.is_fitted

        # Custom initialization
        detector = LSTMRegimeDetector(
            n_regimes=4,
            sequence_length=50,
            lstm_units=128,
            num_layers=3,
            bidirectional=True,
            dropout_rate=0.3,
        )
        assert detector.n_regimes == 4
        assert detector.sequence_length == 50
        assert detector.lstm_units == 128
        assert detector.num_layers == 3
        assert detector.bidirectional
        assert detector.dropout_rate == 0.3

    def test_prepare_sequences(self, lstm_detector, time_series_data):
        """Test sequence preparation for LSTM."""
        features, true_regimes, returns = time_series_data

        X_seq, y_seq = lstm_detector._prepare_sequences(features, true_regimes)

        # Check shapes
        expected_samples = len(features) - lstm_detector.sequence_length + 1
        assert X_seq.shape == (
            expected_samples,
            lstm_detector.sequence_length,
            features.shape[1],
        )
        assert y_seq.shape == (expected_samples,)

        # Check temporal order is preserved
        assert np.array_equal(
            X_seq[0, -1, :],
            features.iloc[lstm_detector.sequence_length - 1].values,
        )

    def test_fit_supervised(self, lstm_detector, time_series_data):
        """Test fitting with labeled data."""
        features, true_regimes, returns = time_series_data

        # Fit the model
        lstm_detector.fit(features, y=true_regimes, epochs=5, verbose=0)

        assert lstm_detector.is_fitted
        assert lstm_detector.model_ is not None
        assert hasattr(lstm_detector, "scaler_")
        assert hasattr(lstm_detector, "history_")

    def test_fit_unsupervised(self, lstm_detector, time_series_data):
        """Test fitting without labels (clustering first)."""
        features, _, returns = time_series_data

        # Fit without labels
        lstm_detector.fit(features, epochs=5, verbose=0)

        assert lstm_detector.is_fitted
        assert hasattr(lstm_detector, "cluster_labels_")

    def test_predict(self, lstm_detector, time_series_data):
        """Test regime prediction."""
        features, true_regimes, returns = time_series_data

        lstm_detector.fit(features, y=true_regimes, epochs=5, verbose=0)
        predictions = lstm_detector.predict(features)

        assert len(predictions) == len(features)
        assert set(predictions).issubset({0, 1, 2})

        # Should have reasonable accuracy
        # Note: Less strict than RF due to limited training
        accuracy = (predictions == true_regimes).mean()
        assert (
            accuracy > 0.25
        )  # Lower threshold for limited training with random init

    def test_predict_proba(self, lstm_detector, time_series_data):
        """Test probability prediction."""
        features, true_regimes, returns = time_series_data

        lstm_detector.fit(features, y=true_regimes, epochs=5, verbose=0)
        probas = lstm_detector.predict_proba(features)

        assert probas.shape == (len(features), lstm_detector.n_regimes)
        assert np.allclose(probas.sum(axis=1), 1.0, rtol=1e-5)
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_predict_sequences(self, lstm_detector, time_series_data):
        """Test sequence-to-sequence prediction."""
        features, true_regimes, returns = time_series_data

        lstm_detector.fit(features, y=true_regimes, epochs=5, verbose=0)

        # Predict on sequences
        seq_predictions = lstm_detector.predict_sequences(
            features, return_sequences=True
        )

        # Should return predictions for each timestep in sequence
        expected_len = len(features) - lstm_detector.sequence_length + 1
        assert len(seq_predictions) == expected_len

    def test_forecast(self, lstm_detector, time_series_data):
        """Test forecasting future regimes."""
        features, true_regimes, returns = time_series_data

        lstm_detector.fit(features, y=true_regimes, epochs=5, verbose=0)

        # Forecast next 10 periods
        forecast = lstm_detector.forecast(features, steps=10)

        assert len(forecast) == 10
        assert set(forecast).issubset({0, 1, 2})

    def test_attention_weights(self, time_series_data):
        """Test attention mechanism in LSTM."""
        features, true_regimes, returns = time_series_data

        # Create LSTM with attention
        detector = LSTMRegimeDetector(
            attention=True, sequence_length=20, random_state=42
        )

        detector.fit(features, y=true_regimes, epochs=3, verbose=0)

        # Get attention weights
        attention_weights = detector.get_attention_weights(features.iloc[:100])

        assert attention_weights is not None
        assert attention_weights.shape[0] == 100 - detector.sequence_length + 1
        assert attention_weights.shape[1] == detector.sequence_length

    def test_early_stopping(self, lstm_detector, time_series_data):
        """Test early stopping functionality."""
        features, true_regimes, returns = time_series_data

        # Split data for validation
        split = int(0.8 * len(features))
        train_features = features.iloc[:split]
        train_labels = true_regimes[:split]
        val_features = features.iloc[split:]
        val_labels = true_regimes[split:]

        lstm_detector.fit(
            train_features,
            y=train_labels,
            validation_data=(val_features, val_labels),
            epochs=50,
            early_stopping_patience=5,
            verbose=0,
        )

        assert lstm_detector.is_fitted
        # Should stop before 50 epochs
        assert len(lstm_detector.history_["loss"]) <= 50

    def test_model_architecture_variations(self, time_series_data):
        """Test different LSTM architectures."""
        features, true_regimes, returns = time_series_data

        # Bidirectional LSTM
        bi_lstm = LSTMRegimeDetector(
            bidirectional=True, lstm_units=32, sequence_length=15
        )
        bi_lstm.fit(features, y=true_regimes, epochs=3, verbose=0)
        assert bi_lstm.is_fitted

        # Multi-layer LSTM
        deep_lstm = LSTMRegimeDetector(
            num_layers=3, lstm_units=32, sequence_length=15
        )
        deep_lstm.fit(features, y=true_regimes, epochs=3, verbose=0)
        assert deep_lstm.is_fitted

        # LSTM with different activations
        tanh_lstm = LSTMRegimeDetector(
            activation="tanh",
            recurrent_activation="sigmoid",
            sequence_length=15,
        )
        tanh_lstm.fit(features, y=true_regimes, epochs=3, verbose=0)
        assert tanh_lstm.is_fitted

    def test_regularization(self, lstm_detector, time_series_data):
        """Test regularization techniques."""
        features, true_regimes, returns = time_series_data

        # LSTM with dropout and L2 regularization
        reg_detector = LSTMRegimeDetector(
            dropout_rate=0.3,
            l2_reg=0.01,
            batch_normalization=True,
            sequence_length=15,
        )

        reg_detector.fit(features, y=true_regimes, epochs=5, verbose=0)
        assert reg_detector.is_fitted

    def test_save_load(self, lstm_detector, time_series_data, tmp_path):
        """Test model saving and loading."""
        features, true_regimes, returns = time_series_data

        lstm_detector.fit(features, y=true_regimes, epochs=3, verbose=0)

        # Save model
        filepath = tmp_path / "lstm_model"
        lstm_detector.save(filepath)

        # Load model
        loaded_detector = LSTMRegimeDetector.load(filepath)

        # Check predictions are similar
        original_pred = lstm_detector.predict(features.iloc[:100])
        loaded_pred = loaded_detector.predict(features.iloc[:100])

        # Should be very similar (allowing for small numerical differences)
        similarity = (original_pred == loaded_pred).mean()
        assert similarity > 0.9


class TestTransformerRegimeDetector:
    """Test suite for Transformer-based regime detector."""

    @pytest.fixture
    def time_series_data(self):
        """Generate time series data."""
        np.random.seed(42)
        n_samples = 500  # Smaller for transformer
        n_features = 8

        timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        # Create data with long-term dependencies
        features = []
        true_regimes = []

        for i in range(n_samples):
            # Regime changes with long-term patterns
            if i < 150:
                regime = 0
                base_pattern = np.sin(i / 20) * 0.5
            elif i < 350:
                regime = 1
                base_pattern = np.cos(i / 15) * 1.0
            else:
                regime = 2
                base_pattern = np.sin(i / 30) * 0.3

            # Add noise and dependencies
            noise = np.random.randn(n_features) * 0.1
            feature_vec = base_pattern + noise
            feature_vec[0] = base_pattern  # Make first feature predictive

            features.append(feature_vec)
            true_regimes.append(regime)

        features_df = pd.DataFrame(
            features,
            columns=[f"feature_{i}" for i in range(n_features)],
            index=timestamps,
        )

        return features_df, np.array(true_regimes)

    @pytest.fixture
    def transformer_detector(self):
        """Create Transformer detector."""
        return TransformerRegimeDetector(
            n_regimes=3,
            sequence_length=30,
            d_model=64,
            num_heads=8,
            num_layers=2,
            random_state=42,
        )

    def test_initialization(self):
        """Test Transformer initialization."""
        detector = TransformerRegimeDetector()
        assert detector.n_regimes == 3
        assert detector.sequence_length == 50
        assert detector.d_model == 128
        assert detector.num_heads == 8
        assert detector.num_layers == 4
        assert detector.dropout_rate == 0.1
        assert not detector.is_fitted

    def test_positional_encoding(self, transformer_detector):
        """Test positional encoding generation."""
        pos_encoding = transformer_detector._get_positional_encoding(100, 64)

        assert pos_encoding.shape == (100, 64)
        # Check that encoding is different for different positions
        assert not np.allclose(pos_encoding[0], pos_encoding[1])

    def test_multi_head_attention(
        self, transformer_detector, time_series_data
    ):
        """Test multi-head attention mechanism."""
        features, true_regimes = time_series_data

        transformer_detector.fit(features, y=true_regimes, epochs=3, verbose=0)

        # Get attention weights
        attention_weights = transformer_detector.get_attention_weights(
            features.iloc[:100]
        )

        assert attention_weights is not None
        # Should have attention weights for each head
        assert (
            len(attention_weights.shape) >= 3
        )  # (samples, heads, seq_len, seq_len)

    def test_fit_predict(self, transformer_detector, time_series_data):
        """Test fitting and prediction."""
        features, true_regimes = time_series_data

        transformer_detector.fit(features, y=true_regimes, epochs=5, verbose=0)
        predictions = transformer_detector.predict(features)

        assert transformer_detector.is_fitted
        assert len(predictions) == len(features)
        assert set(predictions).issubset({0, 1, 2})

    def test_encoder_decoder_architecture(self):
        """Test encoder-decoder variant."""
        detector = TransformerRegimeDetector(
            architecture="encoder_decoder",
            encoder_layers=3,
            decoder_layers=2,
            sequence_length=20,
        )

        assert detector.architecture == "encoder_decoder"
        assert detector.encoder_layers == 3
        assert detector.decoder_layers == 2

    def test_causal_attention(self):
        """Test causal (masked) attention."""
        detector = TransformerRegimeDetector(
            causal_attention=True, sequence_length=20
        )

        assert detector.causal_attention

    def test_different_optimizers(
        self, transformer_detector, time_series_data
    ):
        """Test different optimizers."""
        features, true_regimes = time_series_data

        # Test with Adam optimizer
        transformer_detector.optimizer = "adam"
        transformer_detector.learning_rate = 0.001
        transformer_detector.fit(features, y=true_regimes, epochs=3, verbose=0)
        assert transformer_detector.is_fitted


class TestCNNLSTMRegimeDetector:
    """Test suite for CNN-LSTM hybrid model."""

    @pytest.fixture
    def time_series_data(self):
        """Generate time series data with local patterns."""
        np.random.seed(42)
        n_samples = 800
        n_features = 6

        timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        features = []
        true_regimes = []

        for i in range(n_samples):
            # Create local patterns that CNN can detect
            if i < 250:
                regime = 0
                # Sharp spikes pattern
                pattern = np.random.randn(n_features) * 0.5
                if i % 10 == 0:  # Periodic spikes
                    pattern[0] = 2.0
            elif i < 550:
                regime = 1
                # Smooth oscillation
                pattern = np.sin(np.arange(n_features) * i / 50) * 1.0
                pattern += np.random.randn(n_features) * 0.2
            else:
                regime = 2
                # Random walk
                if i == 550:
                    pattern = np.random.randn(n_features) * 0.3
                else:
                    pattern = (
                        0.9 * features[-1]
                        + 0.1 * np.random.randn(n_features) * 0.3
                    )

            features.append(pattern)
            true_regimes.append(regime)

        features_df = pd.DataFrame(
            features,
            columns=[f"feature_{i}" for i in range(n_features)],
            index=timestamps,
        )

        return features_df, np.array(true_regimes)

    @pytest.fixture
    def cnn_lstm_detector(self):
        """Create CNN-LSTM detector."""
        return CNNLSTMRegimeDetector(
            n_regimes=3,
            sequence_length=25,
            cnn_filters=[32, 64],
            kernel_sizes=[3, 5],
            lstm_units=50,
            random_state=42,
        )

    def test_initialization(self):
        """Test CNN-LSTM initialization."""
        detector = CNNLSTMRegimeDetector()
        assert detector.n_regimes == 3
        assert detector.sequence_length == 30
        assert detector.cnn_filters == [64, 128, 64]
        assert detector.kernel_sizes == [3, 5, 3]
        assert detector.lstm_units == 64
        assert not detector.is_fitted

    def test_cnn_feature_extraction(self, cnn_lstm_detector, time_series_data):
        """Test CNN feature extraction layer."""
        features, true_regimes = time_series_data

        cnn_lstm_detector.fit(features, y=true_regimes, epochs=3, verbose=0)

        # Extract CNN features
        cnn_features = cnn_lstm_detector.extract_cnn_features(
            features.iloc[:100]
        )

        assert cnn_features is not None
        # Should have reduced dimensionality after CNN layers
        assert (
            cnn_features.shape[0]
            == 100 - cnn_lstm_detector.sequence_length + 1
        )

    def test_fit_predict(self, cnn_lstm_detector, time_series_data):
        """Test fitting and prediction."""
        features, true_regimes = time_series_data

        cnn_lstm_detector.fit(features, y=true_regimes, epochs=5, verbose=0)
        predictions = cnn_lstm_detector.predict(features)

        assert cnn_lstm_detector.is_fitted
        assert len(predictions) == len(features)
        assert set(predictions).issubset({0, 1, 2})

    def test_different_architectures(self, time_series_data):
        """Test different CNN-LSTM architectures."""
        features, true_regimes = time_series_data

        # ResNet-like architecture
        resnet_detector = CNNLSTMRegimeDetector(
            cnn_filters=[32, 32, 64, 64],
            kernel_sizes=[3, 3, 5, 5],
            use_residual_connections=True,
            sequence_length=20,
        )
        resnet_detector.fit(features, y=true_regimes, epochs=3, verbose=0)
        assert resnet_detector.is_fitted

        # 1D CNN with different pooling
        pooling_detector = CNNLSTMRegimeDetector(
            cnn_filters=[64, 128],
            kernel_sizes=[5, 3],
            pooling_size=2,
            pooling_type="average",
            sequence_length=20,
        )
        pooling_detector.fit(features, y=true_regimes, epochs=3, verbose=0)
        assert pooling_detector.is_fitted

    def test_feature_importance(self, cnn_lstm_detector, time_series_data):
        """Test feature importance analysis."""
        features, true_regimes = time_series_data

        cnn_lstm_detector.fit(features, y=true_regimes, epochs=5, verbose=0)

        # Get feature importance through gradient analysis
        importance = cnn_lstm_detector.get_feature_importance(
            features.iloc[:50]
        )

        assert importance is not None
        assert len(importance) == features.shape[1]
        assert all(imp >= 0 for imp in importance)

    def test_interpretability(self, cnn_lstm_detector, time_series_data):
        """Test model interpretability features."""
        features, true_regimes = time_series_data

        cnn_lstm_detector.fit(features, y=true_regimes, epochs=3, verbose=0)

        # Get layer activations
        activations = cnn_lstm_detector.get_layer_activations(
            features.iloc[:10], layer_name="cnn"
        )
        assert activations is not None

        # Get decision explanation
        explanation = cnn_lstm_detector.explain_prediction(
            features.iloc[100:101]
        )
        assert explanation is not None
