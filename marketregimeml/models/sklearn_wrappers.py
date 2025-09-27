"""Simple sklearn wrappers for regime detection models."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ModelWrapper(ClassifierMixin, BaseEstimator):
    """Simple wrapper to make any regime detector sklearn-compatible."""

    def __init__(self, model, n_regimes: int = 3):
        self.model = model
        self.n_regimes = n_regimes
        self.classes_ = np.arange(n_regimes)

    def fit(self, X, y=None):
        """Fit the model."""
        self.model.fit(X)
        return self

    def predict(self, X):
        """Predict regimes."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict regime probabilities."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # Return one-hot encoded predictions if no probabilities available
            predictions = self.predict(X)
            n_samples = len(predictions)
            proba = np.zeros((n_samples, self.n_regimes))
            proba[np.arange(n_samples), predictions] = 1.0
            return proba


class UnsupervisedWrapper(ModelWrapper):
    """Wrapper for unsupervised models like HMM and GMM."""

    def fit(self, X, y=None):
        """Fit unsupervised model."""
        # Unsupervised models don't use y
        self.model.fit(X)
        return self
