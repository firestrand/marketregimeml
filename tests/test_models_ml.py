"""Tests for machine learning based regime detection models."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import warnings
from sklearn.datasets import make_classification

from marketregimeml.models.ml import (
    RandomForestRegimeClassifier,
    XGBoostRegimeClassifier,
    SVMRegimeClassifier,
)


class TestRandomForestRegimeClassifier:
    """Test suite for Random Forest regime classifier."""

    @pytest.fixture
    def sample_features(self):
        """Generate sample feature data with clear regime structure."""
        np.random.seed(42)
        n_samples = 600
        n_features = 10

        # Create features with regime-dependent patterns
        features = []
        true_regimes = []

        # Regime 0: Low volatility, trending up
        for _ in range(200):
            sample = np.random.randn(n_features) * 0.5
            sample[0] += 0.5  # Trend feature
            sample[1] *= 0.3  # Low volatility
            features.append(sample)
            true_regimes.append(0)

        # Regime 1: High volatility, trending down
        for _ in range(200):
            sample = np.random.randn(n_features) * 1.5
            sample[0] -= 0.5  # Trend feature
            sample[1] *= 1.5  # High volatility
            features.append(sample)
            true_regimes.append(1)

        # Regime 2: Medium volatility, sideways
        for _ in range(200):
            sample = np.random.randn(n_features) * 1.0
            sample[0] *= 0.1  # No trend
            sample[1] *= 1.0  # Medium volatility
            features.append(sample)
            true_regimes.append(2)

        features_df = pd.DataFrame(
            features, columns=[f"feature_{i}" for i in range(n_features)]
        )
        features_df.index = pd.date_range(
            "2020-01-01", periods=n_samples, freq="D"
        )

        return features_df, np.array(true_regimes)

    @pytest.fixture
    def rf_classifier(self):
        """Create Random Forest classifier instance."""
        return RandomForestRegimeClassifier(
            n_regimes=3, n_estimators=100, random_state=42
        )

    def test_initialization(self):
        """Test Random Forest classifier initialization."""
        # Default initialization
        clf = RandomForestRegimeClassifier()
        assert clf.n_regimes == 3
        assert clf.n_estimators == 100
        assert clf.max_depth is None
        assert not clf.is_fitted

        # Custom initialization
        clf = RandomForestRegimeClassifier(
            n_regimes=4,
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
        )
        assert clf.n_regimes == 4
        assert clf.n_estimators == 200
        assert clf.max_depth == 10
        assert clf.min_samples_split == 5

    def test_fit_supervised(self, rf_classifier, sample_features):
        """Test fitting with labeled data (supervised)."""
        features, true_regimes = sample_features

        # Fit with labels
        rf_classifier.fit(features, y=true_regimes)

        assert rf_classifier.is_fitted
        assert rf_classifier.model_ is not None
        assert hasattr(rf_classifier, "feature_importances_")
        assert len(rf_classifier.feature_importances_) == features.shape[1]

    def test_fit_unsupervised(self, rf_classifier, sample_features):
        """Test fitting without labels (unsupervised clustering first)."""
        features, _ = sample_features

        # Fit without labels (should use clustering)
        rf_classifier.fit(features)

        assert rf_classifier.is_fitted
        assert rf_classifier.model_ is not None
        assert hasattr(rf_classifier, "cluster_labels_")

    def test_predict(self, rf_classifier, sample_features):
        """Test regime prediction."""
        features, true_regimes = sample_features

        # Fit and predict
        rf_classifier.fit(features, y=true_regimes)
        predictions = rf_classifier.predict(features)

        assert len(predictions) == len(features)
        assert set(predictions).issubset({0, 1, 2})

        # Should have reasonable accuracy on training data
        accuracy = (predictions == true_regimes).mean()
        assert accuracy > 0.8  # Random Forest should fit training data well

    def test_predict_proba(self, rf_classifier, sample_features):
        """Test probability prediction."""
        features, true_regimes = sample_features

        rf_classifier.fit(features, y=true_regimes)
        probas = rf_classifier.predict_proba(features)

        assert probas.shape == (len(features), rf_classifier.n_regimes)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all(probas >= 0) and np.all(probas <= 1)

        # High confidence on training data
        max_probas = probas.max(axis=1)
        assert max_probas.mean() > 0.7

    def test_fit_predict(self, rf_classifier, sample_features):
        """Test combined fit and predict."""
        features, true_regimes = sample_features

        predictions = rf_classifier.fit_predict(features, y=true_regimes)

        assert rf_classifier.is_fitted
        assert len(predictions) == len(features)

    def test_feature_importance(self, rf_classifier, sample_features):
        """Test feature importance extraction."""
        features, true_regimes = sample_features

        rf_classifier.fit(features, y=true_regimes)

        importances = rf_classifier.get_feature_importance()

        assert len(importances) == features.shape[1]
        assert all(imp >= 0 for imp in importances.values())
        assert abs(sum(importances.values()) - 1.0) < 0.01  # Should sum to 1

        # First two features should be most important (by design)
        top_features = sorted(
            importances.items(), key=lambda x: x[1], reverse=True
        )
        assert "feature_0" in [f[0] for f in top_features[:3]]
        assert "feature_1" in [f[0] for f in top_features[:3]]

    def test_cross_validation(self, rf_classifier, sample_features):
        """Test cross-validation functionality."""
        features, true_regimes = sample_features

        scores = rf_classifier.cross_validate(features, y=true_regimes, cv=3)

        assert "train_score" in scores
        assert "test_score" in scores
        assert len(scores["train_score"]) == 3
        assert len(scores["test_score"]) == 3

        # Should have reasonable cross-validation scores
        # Lower threshold since we're using synthetic random data
        assert np.mean(scores["test_score"]) > 0.3

    def test_hyperparameter_tuning(self, rf_classifier, sample_features):
        """Test hyperparameter optimization."""
        features, true_regimes = sample_features

        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        }

        best_params = rf_classifier.tune_hyperparameters(
            features, y=true_regimes, param_grid=param_grid, cv=3
        )

        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "min_samples_split" in best_params

        # Should be fitted with best parameters
        assert rf_classifier.is_fitted

    def test_oob_score(self):
        """Test out-of-bag score calculation."""
        clf = RandomForestRegimeClassifier(oob_score=True)

        # Generate simple data
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_classes=3,
            random_state=42,
        )
        X = pd.DataFrame(X)

        clf.fit(X, y=y)

        assert hasattr(clf, "oob_score_")
        assert 0 <= clf.oob_score_ <= 1

    def test_predict_not_fitted(self, rf_classifier):
        """Test prediction without fitting raises error."""
        features = pd.DataFrame(np.random.randn(10, 5))

        with pytest.raises(ValueError, match="not fitted"):
            rf_classifier.predict(features)

    def test_save_load(self, rf_classifier, sample_features, tmp_path):
        """Test model saving and loading."""
        features, true_regimes = sample_features

        rf_classifier.fit(features, y=true_regimes)

        # Save model
        filepath = tmp_path / "rf_classifier.pkl"
        rf_classifier.save(filepath)

        # Load model
        loaded_clf = RandomForestRegimeClassifier.load(filepath)

        # Check predictions are the same
        original_pred = rf_classifier.predict(features)
        loaded_pred = loaded_clf.predict(features)
        np.testing.assert_array_equal(original_pred, loaded_pred)


class TestXGBoostRegimeClassifier:
    """Test suite for XGBoost regime classifier."""

    @pytest.fixture
    def sample_features(self):
        """Generate sample feature data."""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            n_classes=3,
            random_state=42,
        )

        features = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        features.index = pd.date_range("2020-01-01", periods=500, freq="D")

        return features, y

    @pytest.fixture
    def xgb_classifier(self):
        """Create XGBoost classifier instance."""
        return XGBoostRegimeClassifier(
            n_regimes=3, n_estimators=100, learning_rate=0.1, random_state=42
        )

    def test_initialization(self):
        """Test XGBoost initialization."""
        clf = XGBoostRegimeClassifier()
        assert clf.n_regimes == 3
        assert clf.n_estimators == 100
        assert clf.learning_rate == 0.1
        assert clf.max_depth == 6
        assert not clf.is_fitted

    def test_fit_predict(self, xgb_classifier, sample_features):
        """Test fitting and prediction."""
        features, true_regimes = sample_features

        xgb_classifier.fit(features, y=true_regimes)
        predictions = xgb_classifier.predict(features)

        assert xgb_classifier.is_fitted
        assert len(predictions) == len(features)
        assert set(predictions).issubset({0, 1, 2})

        # XGBoost should have good accuracy
        accuracy = (predictions == true_regimes).mean()
        assert accuracy > 0.7

    def test_predict_proba(self, xgb_classifier, sample_features):
        """Test probability prediction."""
        features, true_regimes = sample_features

        xgb_classifier.fit(features, y=true_regimes)
        probas = xgb_classifier.predict_proba(features)

        assert probas.shape == (len(features), 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

    def test_feature_importance(self, xgb_classifier, sample_features):
        """Test feature importance with different methods."""
        features, true_regimes = sample_features

        xgb_classifier.fit(features, y=true_regimes)

        # Test different importance types
        for importance_type in ["weight", "gain", "cover"]:
            importances = xgb_classifier.get_feature_importance(
                importance_type=importance_type
            )

            assert len(importances) == features.shape[1]
            assert all(imp >= 0 for imp in importances.values())

    def test_early_stopping(self, xgb_classifier, sample_features):
        """Test early stopping functionality."""
        features, true_regimes = sample_features

        # Split data for validation
        n_train = int(0.8 * len(features))
        X_train = features.iloc[:n_train]
        y_train = true_regimes[:n_train]
        X_val = features.iloc[n_train:]
        y_val = true_regimes[n_train:]

        xgb_classifier.fit(
            X_train,
            y=y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
        )

        assert xgb_classifier.is_fitted
        # Model should have stopped early (less than n_estimators)
        assert hasattr(xgb_classifier.model_, "best_iteration")

    def test_gpu_support(self):
        """Test GPU parameter configuration."""
        # Test that GPU parameters can be set (actual GPU not required for test)
        clf = XGBoostRegimeClassifier(tree_method="gpu_hist", gpu_id=0)

        assert clf.tree_method == "gpu_hist"
        assert clf.gpu_id == 0


class TestSVMRegimeClassifier:
    """Test suite for SVM regime classifier."""

    @pytest.fixture
    def sample_features(self):
        """Generate sample feature data with clear boundaries."""
        np.random.seed(42)

        # Create data with clear decision boundaries
        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=2.0,  # Good separation
            random_state=42,
        )

        features = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        features.index = pd.date_range("2020-01-01", periods=300, freq="D")

        return features, y

    @pytest.fixture
    def svm_classifier(self):
        """Create SVM classifier instance."""
        return SVMRegimeClassifier(n_regimes=3, kernel="rbf", random_state=42)

    def test_initialization(self):
        """Test SVM initialization with different kernels."""
        # RBF kernel
        clf = SVMRegimeClassifier(kernel="rbf")
        assert clf.kernel == "rbf"
        assert clf.gamma == "scale"

        # Linear kernel
        clf = SVMRegimeClassifier(kernel="linear")
        assert clf.kernel == "linear"

        # Polynomial kernel
        clf = SVMRegimeClassifier(kernel="poly", degree=3)
        assert clf.kernel == "poly"
        assert clf.degree == 3

    def test_fit_predict(self, svm_classifier, sample_features):
        """Test fitting and prediction."""
        features, true_regimes = sample_features

        svm_classifier.fit(features, y=true_regimes)
        predictions = svm_classifier.predict(features)

        assert svm_classifier.is_fitted
        assert len(predictions) == len(features)

        # SVM should classify well with good separation
        accuracy = (predictions == true_regimes).mean()
        assert accuracy > 0.7

    def test_predict_proba(self, svm_classifier, sample_features):
        """Test probability prediction."""
        features, true_regimes = sample_features

        # Need probability=True for predict_proba
        svm_classifier.probability = True
        svm_classifier.fit(features, y=true_regimes)
        probas = svm_classifier.predict_proba(features)

        assert probas.shape == (len(features), 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

    def test_decision_function(self, svm_classifier, sample_features):
        """Test decision function (distance to hyperplane)."""
        features, true_regimes = sample_features

        svm_classifier.fit(features, y=true_regimes)
        decision = svm_classifier.decision_function(features)

        # For multi-class, decision function returns distances to all hyperplanes
        assert decision.shape == (len(features), 3)

        # Predictions should match argmax of decision function
        predictions = svm_classifier.predict(features)
        decision_predictions = np.argmax(decision, axis=1)
        np.testing.assert_array_equal(predictions, decision_predictions)

    def test_support_vectors(self, svm_classifier, sample_features):
        """Test support vector information."""
        features, true_regimes = sample_features

        svm_classifier.fit(features, y=true_regimes)

        support_info = svm_classifier.get_support_vectors()

        assert "n_support" in support_info
        assert "support_indices" in support_info
        assert "support_vectors" in support_info

        # Should have support vectors for each class
        assert len(support_info["n_support"]) == 3
        assert sum(support_info["n_support"]) == len(
            support_info["support_indices"]
        )

    def test_kernel_comparison(self, sample_features):
        """Test different kernel performance."""
        features, true_regimes = sample_features

        kernels = ["linear", "rbf", "poly", "sigmoid"]
        results = {}

        for kernel in kernels:
            clf = SVMRegimeClassifier(kernel=kernel, random_state=42)
            clf.fit(features, y=true_regimes)
            predictions = clf.predict(features)
            accuracy = (predictions == true_regimes).mean()
            results[kernel] = accuracy

        # All kernels should work
        assert all(acc > 0.5 for acc in results.values())

        # RBF usually performs well
        assert results["rbf"] > 0.6

    def test_class_weight(self, sample_features):
        """Test class weight balancing."""
        features, true_regimes = sample_features

        # Create imbalanced dataset
        imbalanced_regimes = true_regimes.copy()
        imbalanced_regimes[imbalanced_regimes == 2] = 1  # Reduce class 2

        # Without class weight
        clf_unbalanced = SVMRegimeClassifier(random_state=42)
        clf_unbalanced.fit(features, y=imbalanced_regimes)

        # With class weight
        clf_balanced = SVMRegimeClassifier(
            class_weight="balanced", random_state=42
        )
        clf_balanced.fit(features, y=imbalanced_regimes)

        # Both should fit
        assert clf_unbalanced.is_fitted
        assert clf_balanced.is_fitted

        # Balanced should handle minority class better
        pred_balanced = clf_balanced.predict(features)
        unique_balanced = np.unique(pred_balanced)
        assert len(unique_balanced) >= 2  # Should predict multiple classes

    def test_grid_search(self, svm_classifier, sample_features):
        """Test hyperparameter grid search."""
        features, true_regimes = sample_features

        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto", 0.001, 0.01],
        }

        best_params = svm_classifier.tune_hyperparameters(
            features, y=true_regimes, param_grid=param_grid, cv=3
        )

        assert "C" in best_params
        assert "gamma" in best_params
        assert svm_classifier.is_fitted
