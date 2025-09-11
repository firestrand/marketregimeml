"""Isolated tests for models to ensure proper coverage measurement."""

import pytest
import numpy as np
import pandas as pd

# Import all models
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector
from marketregimeml.models import EnsembleRegimeDetector
from marketregimeml.models.garch import (
    GARCHRegimeDetector,
    MSGARCHRegimeDetector,
)
from marketregimeml.models.ml import (
    RandomForestRegimeClassifier,
    XGBoostRegimeClassifier,
    SVMRegimeClassifier,
)


class TestModelsIsolated:
    """Test all models with proper isolation."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature data."""
        np.random.seed(42)
        n_samples = 500

        # Generate synthetic data without sklearn
        returns = np.random.randn(n_samples) * 0.02
        volatility = np.abs(np.random.randn(n_samples)) * 0.1 + 0.1
        volume = np.abs(np.random.randn(n_samples)) * 1000000 + 1000000

        # Convert to DataFrame with proper index
        return pd.DataFrame(
            {"returns": returns, "volatility": volatility, "volume": volume},
            index=pd.date_range("2023-01-01", periods=n_samples, freq="D"),
        )

    def test_hmm_full_workflow(self, sample_features):
        """Test HMM model complete workflow."""
        # Initialize
        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        assert not model.is_fitted

        # Fit
        model.fit(sample_features)
        assert model.is_fitted
        assert model.model is not None

        # Predict
        regimes = model.predict(sample_features)
        assert len(regimes) == len(sample_features)
        assert regimes.min() >= 0
        assert regimes.max() < 3

        # Predict proba
        proba = model.predict_proba(sample_features)
        assert proba.shape == (len(sample_features), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

        # Get diagnostics
        diag = model.get_diagnostics()
        assert "regime_counts" in diag
        assert "transition_matrix" in diag
        assert "log_likelihood" in diag

        # Score
        score = model.score(sample_features)
        assert isinstance(score, float)

    def test_gmm_full_workflow(self, sample_features):
        """Test GMM model complete workflow."""
        # Initialize
        model = GMMRegimeDetector(n_regimes=3, random_state=42)
        assert not model.is_fitted

        # Fit
        model.fit(sample_features)
        assert model.is_fitted
        assert model.model is not None

        # Predict
        regimes = model.predict(sample_features)
        assert len(regimes) == len(sample_features)
        assert regimes.min() >= 0
        assert regimes.max() < 3

        # Predict proba
        proba = model.predict_proba(sample_features)
        assert proba.shape == (len(sample_features), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        # Get diagnostics
        diag = model.get_diagnostics()
        assert "regime_counts" in diag
        assert "bic" in diag
        assert "aic" in diag

        # Score
        score = model.score(sample_features)
        assert isinstance(score, float)

    def test_ensemble_with_mock_models(self, sample_features):
        """Test Ensemble model with mock base models."""
        from unittest.mock import Mock

        # Create mock models
        mock_models = []
        for i in range(3):
            mock_model = Mock()
            mock_model.fit = Mock(return_value=mock_model)
            mock_model.predict = Mock(
                return_value=np.random.choice(
                    [0, 1, 2], size=len(sample_features)
                )
            )
            mock_model.predict_proba = Mock(
                return_value=np.random.rand(len(sample_features), 3)
            )
            mock_model.is_fitted = True
            mock_model.__class__.__name__ = f"MockModel{i}"
            mock_models.append(mock_model)

        # Initialize ensemble
        ensemble = EnsembleRegimeDetector(
            models=mock_models, strategy="voting", n_regimes=3
        )

        # Fit
        ensemble.fit(sample_features)
        assert ensemble.is_fitted

        # Predict
        regimes = ensemble.predict(sample_features)
        assert len(regimes) == len(sample_features)

        # Get diagnostics
        diag = ensemble.get_diagnostics()
        assert "model_performance" in diag
        assert "agreement_matrix" in diag

    def test_garch_basic(self, sample_features):
        """Test GARCH model basic functionality."""
        model = GARCHRegimeDetector(n_regimes=3)

        # Test initialization
        assert model.n_regimes == 3
        assert not model.is_fitted

        # Test fit with DataFrame containing returns
        returns_df = pd.DataFrame({"returns": sample_features["returns"]})
        model.fit(returns_df)
        assert model.is_fitted

        # Test predict
        regimes = model.predict(returns_df)
        assert len(regimes) == len(returns_df)
        assert regimes.min() >= 0
        assert regimes.max() < 3

    def test_msgarch_basic(self, sample_features):
        """Test MS-GARCH model basic functionality."""
        model = MSGARCHRegimeDetector(n_regimes=2)

        # Test initialization
        assert model.n_regimes == 2
        assert not model.is_fitted

        # Test fit with DataFrame containing returns
        returns_df = pd.DataFrame({"returns": sample_features["returns"]})
        model.fit(returns_df)
        assert model.is_fitted

        # Test predict
        regimes = model.predict(returns_df)
        assert len(regimes) == len(returns_df)
        assert regimes.min() >= 0
        assert regimes.max() < 2

    def test_random_forest_classifier(self, sample_features):
        """Test Random Forest regime classifier."""
        model = RandomForestRegimeClassifier(
            n_estimators=10, max_depth=3, random_state=42  # Small for speed
        )

        # Create synthetic labels
        y = np.random.choice([0, 1, 2], size=len(sample_features))

        # Fit
        model.fit(sample_features, y)
        assert model.is_fitted

        # Predict
        predictions = model.predict(sample_features)
        assert len(predictions) == len(sample_features)
        assert predictions.min() >= 0
        assert predictions.max() <= 2

        # Predict proba
        proba = model.predict_proba(sample_features)
        assert proba.shape == (len(sample_features), 3)

        # Feature importance
        importance = model.get_feature_importance()
        assert len(importance) == sample_features.shape[1]

    def test_xgboost_classifier(self, sample_features):
        """Test XGBoost regime classifier."""
        model = XGBoostRegimeClassifier(
            n_estimators=10, max_depth=3, random_state=42  # Small for speed
        )

        # Create synthetic labels
        y = np.random.choice([0, 1, 2], size=len(sample_features))

        # Fit
        model.fit(sample_features, y)
        assert model.is_fitted

        # Predict
        predictions = model.predict(sample_features)
        assert len(predictions) == len(sample_features)

        # Predict proba
        proba = model.predict_proba(sample_features)
        assert proba.shape == (len(sample_features), 3)

    def test_svm_classifier(self, sample_features):
        """Test SVM regime classifier."""
        model = SVMRegimeClassifier(kernel="rbf", C=1.0, random_state=42)

        # Use smaller dataset for SVM (slower)
        small_features = sample_features.iloc[:100]
        y = np.random.choice([0, 1, 2], size=len(small_features))

        # Fit
        model.fit(small_features, y)
        assert model.is_fitted

        # Predict
        predictions = model.predict(small_features)
        assert len(predictions) == len(small_features)

        # Decision function
        decision = model.decision_function(small_features)
        assert decision.shape[0] == len(small_features)

    def test_model_persistence(self, sample_features, tmp_path):
        """Test model save/load functionality."""
        # Train a model
        model = HMMRegimeDetector(n_regimes=2, random_state=42)
        model.fit(sample_features)
        original_predictions = model.predict(sample_features)

        # Save model
        save_path = tmp_path / "test_model.pkl"
        model.save(str(save_path))
        assert save_path.exists()

        # Load model
        loaded_model = HMMRegimeDetector.load(str(save_path))
        assert loaded_model.is_fitted

        # Check predictions are the same
        loaded_predictions = loaded_model.predict(sample_features)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_cross_model_compatibility(self, sample_features):
        """Test that different models can work with same data."""
        models = [
            HMMRegimeDetector(n_regimes=3, random_state=42),
            GMMRegimeDetector(n_regimes=3, random_state=42),
            GARCHRegimeDetector(n_regimes=3),
        ]

        predictions = []
        for model in models:
            if isinstance(model, GARCHRegimeDetector):
                # GARCH needs DataFrame with returns
                returns_df = pd.DataFrame(
                    {"returns": sample_features["returns"]}
                )
                model.fit(returns_df)
                pred = model.predict(returns_df)
            else:
                model.fit(sample_features)
                pred = model.predict(sample_features)

            predictions.append(pred)

            # Check basic properties
            assert len(pred) == len(sample_features)
            assert pred.min() >= 0
            assert pred.max() < 3

        # Check that models give different predictions (they should)
        assert not np.array_equal(predictions[0], predictions[1])

    def test_model_with_missing_data(self, sample_features):
        """Test model behavior with missing data."""
        # Add some NaN values
        features_with_nan = sample_features.copy()
        features_with_nan.iloc[10:15, 0] = np.nan

        model = GMMRegimeDetector(n_regimes=3)

        # Should handle NaN appropriately (either error or interpolate)
        try:
            model.fit(features_with_nan)
            # If it doesn't error, check it still works
            predictions = model.predict(features_with_nan)
            assert len(predictions) == len(features_with_nan)
        except (ValueError, TypeError):
            # Expected behavior - model rejects NaN values
            pass

    def test_arbitrary_regime_counts(self, sample_features):
        """Test models with different regime counts."""
        for n_regimes in [2, 3, 5, 7]:
            model = GMMRegimeDetector(n_regimes=n_regimes, random_state=42)
            model.fit(sample_features)

            predictions = model.predict(sample_features)
            unique_regimes = np.unique(predictions)

            # Should have at most n_regimes unique values
            assert len(unique_regimes) <= n_regimes
            assert predictions.min() >= 0
            assert predictions.max() < n_regimes

    def test_fuzzy_predictions(self, sample_features):
        """Test fuzzy regime predictions."""
        model = HMMRegimeDetector(n_regimes=3, random_state=42)
        model.fit(sample_features)

        # Get fuzzy predictions
        fuzzy = model.predict_fuzzy(sample_features)

        assert "crisp" in fuzzy
        assert "confidence" in fuzzy
        assert len(fuzzy["crisp"]) == len(sample_features)
        assert len(fuzzy["confidence"]) == len(sample_features)

        # Confidence should be between 0 and 1
        assert fuzzy["confidence"].min() >= 0
        assert fuzzy["confidence"].max() <= 1
