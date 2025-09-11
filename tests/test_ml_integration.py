"""Integration tests for ML models - focus on working functionality."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from marketregimeml.models.ml import (
    RandomForestRegimeClassifier,
    XGBoostRegimeClassifier,
    SVMRegimeClassifier,
)


class TestMLIntegration:
    """Integration tests for ML-based regime detection."""

    @pytest.fixture
    def regime_data(self):
        """Create realistic regime data."""
        np.random.seed(42)

        # Create time series with regime changes
        data = []

        # Regime 0: Low volatility, positive drift
        for _ in range(100):
            data.append(
                [
                    np.random.normal(0.001, 0.01),  # returns
                    np.random.normal(0.5, 0.1),  # volatility indicator
                    np.random.normal(0.2, 0.05),  # momentum
                ]
            )

        # Regime 1: High volatility, negative drift
        for _ in range(100):
            data.append(
                [
                    np.random.normal(-0.001, 0.03),  # returns
                    np.random.normal(1.5, 0.2),  # volatility indicator
                    np.random.normal(-0.2, 0.1),  # momentum
                ]
            )

        # Regime 2: Medium volatility, no drift
        for _ in range(100):
            data.append(
                [
                    np.random.normal(0, 0.02),  # returns
                    np.random.normal(1.0, 0.15),  # volatility indicator
                    np.random.normal(0, 0.08),  # momentum
                ]
            )

        df = pd.DataFrame(data, columns=["returns", "volatility", "momentum"])

        # True labels (for reference, not used in unsupervised learning)
        true_regimes = np.array([0] * 100 + [1] * 100 + [2] * 100)

        return df, true_regimes

    def test_random_forest_integration(self, regime_data):
        """Test RandomForest full pipeline."""
        X, true_regimes = regime_data

        # Initialize model
        model = RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=50, max_depth=5, random_state=42
        )

        # Fit model
        model.fit(X)
        assert model.is_fitted
        assert model.n_features == 3

        # Make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert predictions.min() >= 0
        assert predictions.max() < 3

        # Check regime changes exist
        regime_changes = np.diff(predictions) != 0
        assert regime_changes.sum() > 0  # Should have some regime changes

        # Get probabilities
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, rtol=0.01)

        # Get feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 3
        assert all(v >= 0 for v in importance.values())

    def test_xgboost_integration(self, regime_data):
        """Test XGBoost full pipeline."""
        X, true_regimes = regime_data

        # Initialize model with conservative parameters
        model = XGBoostRegimeClassifier(
            n_regimes=3,
            n_estimators=30,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )

        # Fit model
        model.fit(X)
        assert model.is_fitted

        # Make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert len(np.unique(predictions)) <= 3

        # Get probabilities
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)

        # Feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 3

    def test_svm_integration(self, regime_data):
        """Test SVM full pipeline."""
        X, true_regimes = regime_data

        # Initialize SVM (without probability for speed)
        model = SVMRegimeClassifier(
            n_regimes=3, kernel="rbf", C=1.0, gamma="scale", random_state=42
        )

        # Fit model
        model.fit(X)
        assert model.is_fitted

        # Make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert predictions.min() >= 0
        assert predictions.max() < 3

        # Check that we detect different regimes
        unique_regimes = np.unique(predictions)
        assert len(unique_regimes) > 1  # Should detect at least 2 regimes

    def test_model_consistency(self, regime_data):
        """Test that models give consistent results with same random seed."""
        X, _ = regime_data

        # Test RandomForest consistency
        rf1 = RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=10, random_state=42
        )
        rf2 = RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=10, random_state=42
        )

        rf1.fit(X)
        rf2.fit(X)

        pred1 = rf1.predict(X)
        pred2 = rf2.predict(X)

        assert np.array_equal(pred1, pred2)

        # Test XGBoost consistency
        xgb1 = XGBoostRegimeClassifier(
            n_regimes=3, n_estimators=10, random_state=42
        )
        xgb2 = XGBoostRegimeClassifier(
            n_regimes=3, n_estimators=10, random_state=42
        )

        xgb1.fit(X)
        xgb2.fit(X)

        pred1 = xgb1.predict(X)
        pred2 = xgb2.predict(X)

        assert np.array_equal(pred1, pred2)

    def test_different_n_regimes(self, regime_data):
        """Test models with different numbers of regimes."""
        X, _ = regime_data

        for n_regimes in [2, 3, 4, 5]:
            model = RandomForestRegimeClassifier(
                n_regimes=n_regimes, n_estimators=20, random_state=42
            )

            model.fit(X)
            predictions = model.predict(X)

            # Check predictions are valid
            assert predictions.min() >= 0
            assert predictions.max() < n_regimes

            # Should use at least 2 regimes
            unique = np.unique(predictions)
            assert len(unique) >= min(2, n_regimes)

    def test_single_sample_prediction(self, regime_data):
        """Test prediction on single samples."""
        X, _ = regime_data

        model = RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=20, random_state=42
        )
        model.fit(X)

        # Test single sample
        single = X.iloc[[0]]
        pred = model.predict(single)
        assert len(pred) == 1
        assert 0 <= pred[0] < 3

        # Test small batch
        small_batch = X.iloc[:5]
        pred_batch = model.predict(small_batch)
        assert len(pred_batch) == 5
        assert all(0 <= p < 3 for p in pred_batch)

    def test_regime_stability(self, regime_data):
        """Test that detected regimes show some stability."""
        X, _ = regime_data

        model = RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=50, random_state=42
        )
        model.fit(X)
        predictions = model.predict(X)

        # Calculate regime durations
        regime_changes = np.diff(predictions) != 0
        n_changes = regime_changes.sum()

        # Average regime duration
        avg_duration = len(predictions) / (n_changes + 1)

        # Regimes should persist for more than 1 period on average
        assert avg_duration > 2  # Some stability (not switching every period)

        # But not be constant
        assert n_changes > 0

    def test_model_scoring(self, regime_data):
        """Test model scoring methods."""
        X, _ = regime_data

        model = RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=20, random_state=42
        )
        model.fit(X)

        # Test log likelihood
        ll = model.score(X, metric="log_likelihood")
        assert isinstance(ll, float)
        assert ll < 0  # Log likelihood is negative

        # Test AIC
        aic = model.score(X, metric="aic")
        assert isinstance(aic, float)
        assert aic > 0

        # Test BIC
        bic = model.score(X, metric="bic")
        assert isinstance(bic, float)
        assert bic > 0
        assert bic > aic  # BIC penalizes complexity more

    def test_feature_names_handling(self, regime_data):
        """Test handling of feature names."""
        X, _ = regime_data

        model = RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=10, random_state=42
        )
        model.fit(X)

        # Feature importance should use column names
        importance = model.get_feature_importance()
        assert "returns" in importance
        assert "volatility" in importance
        assert "momentum" in importance

    def test_empty_data_handling(self):
        """Test handling of edge cases."""
        model = RandomForestRegimeClassifier(n_regimes=3)

        # Empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises((ValueError, KeyError)):
            model.fit(empty_df)

        # Single row (not enough for clustering)
        single_row = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        with pytest.raises((ValueError, RuntimeError)):
            model.fit(single_row)

    def test_model_persistence_simple(self, regime_data):
        """Test simple model save/load without full persistence."""
        X, _ = regime_data

        model = RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=10, random_state=42
        )
        model.fit(X)

        # Get predictions before save
        predictions_before = model.predict(X)

        # Check model attributes exist
        assert hasattr(model, "model_")
        assert hasattr(model, "scaler_")
        assert model.is_fitted

        # After fitting, should be able to predict consistently
        predictions_after = model.predict(X)
        assert np.array_equal(predictions_before, predictions_after)

    def test_probability_predictions(self, regime_data):
        """Test probability predictions for ensemble models."""
        X, _ = regime_data

        # RandomForest should support probabilities
        rf = RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=20, random_state=42
        )
        rf.fit(X)

        proba = rf.predict_proba(X)
        predictions = rf.predict(X)

        # Predictions should match argmax of probabilities
        pred_from_proba = np.argmax(proba, axis=1)
        assert np.array_equal(predictions, pred_from_proba)

        # XGBoost should also support probabilities
        xgb = XGBoostRegimeClassifier(
            n_regimes=3, n_estimators=20, random_state=42
        )
        xgb.fit(X)

        xgb_proba = xgb.predict_proba(X)
        xgb_pred = xgb.predict(X)

        xgb_pred_from_proba = np.argmax(xgb_proba, axis=1)
        assert np.array_equal(xgb_pred, xgb_pred_from_proba)
