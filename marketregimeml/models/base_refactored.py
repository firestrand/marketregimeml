"""Base class for regime detection models - Refactored version.

Following SOLID principles with mixins and interfaces.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from marketregimeml.models.interfaces import (
    IRegimeDetector,
    IScorable,
    IStatistical,
)
from marketregimeml.models.mixins import (
    PersistenceMixin,
    ValidationMixin,
    StatisticsMixin,
    CrossValidationMixin,
    OptimizationMixin,
)
from marketregimeml.models.configs import ModelMetadata
from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class BaseRegimeDetector(
    BaseEstimator,
    IRegimeDetector,
    IScorable,
    IStatistical,
    PersistenceMixin,
    ValidationMixin,
    StatisticsMixin,
    CrossValidationMixin,
    OptimizationMixin,
    ABC,
):
    """Base class for regime detection models.

    Refactored to use mixins and interfaces for better separation of concerns.
    Each mixin handles a specific aspect of functionality.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """Initialize base regime detector.

        Parameters
        ----------
        n_regimes : int
            Number of regimes to detect
        random_state : int, optional
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.verbose = verbose

        # State tracking
        self.is_fitted = False
        self._metadata = ModelMetadata()
        self._last_regimes = None
        self._last_probabilities = None

        # Validate parameters
        self._validate_regime_count(n_regimes)

    @property
    def n_features_(self) -> Optional[int]:
        """Number of features seen during fit."""
        return self._metadata.n_features

    @property
    def feature_names_(self) -> Optional[List[str]]:
        """Feature names seen during fit."""
        return self._metadata.feature_names

    def fit_predict(self, features: pd.DataFrame, **kwargs) -> np.ndarray:
        """Fit model and return predictions.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        **kwargs
            Additional fitting parameters

        Returns
        -------
        np.ndarray
            Regime predictions
        """
        self.fit(features, **kwargs)
        return self.predict(features)

    def score(
        self,
        features: pd.DataFrame,
        true_regimes: Optional[np.ndarray] = None,
        scoring: str = "log_likelihood",
    ) -> float:
        """Score the model.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        true_regimes : np.ndarray, optional
            True regime labels (for supervised metrics)
        scoring : str
            Scoring metric

        Returns
        -------
        float
            Score value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        if scoring == "log_likelihood":
            return self._compute_log_likelihood(features)
        elif scoring == "aic":
            return -self._compute_aic(features)  # Negative for higher=better
        elif scoring == "bic":
            return -self._compute_bic(features)
        elif scoring == "silhouette" and true_regimes is not None:
            from sklearn.metrics import silhouette_score

            predictions = self.predict(features)
            return silhouette_score(features, predictions)
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")

    def _compute_aic(self, features: pd.DataFrame) -> float:
        """Compute Akaike Information Criterion.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        float
            AIC value
        """
        log_likelihood = self._compute_log_likelihood(features)
        n_params = self._count_parameters()
        return -2 * log_likelihood + 2 * n_params

    def _compute_bic(self, features: pd.DataFrame) -> float:
        """Compute Bayesian Information Criterion.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        float
            BIC value
        """
        log_likelihood = self._compute_log_likelihood(features)
        n_params = self._count_parameters()
        n_samples = len(features)
        return -2 * log_likelihood + n_params * np.log(n_samples)

    @abstractmethod
    def _compute_log_likelihood(self, features: pd.DataFrame) -> float:
        """Compute log-likelihood of the data.

        Must be implemented by subclasses.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        float
            Log-likelihood value
        """

    @abstractmethod
    def _count_parameters(self) -> int:
        """Count number of model parameters.

        Must be implemented by subclasses.

        Returns
        -------
        int
            Number of parameters
        """

    def get_regime_confidence(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get confidence scores for regime assignments.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        pd.DataFrame
            DataFrame with regime predictions and confidence scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting confidence")

        probabilities = self.predict_proba(features)
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

        # Calculate entropy as uncertainty measure
        entropy = -np.sum(
            probabilities * np.log(probabilities + 1e-10), axis=1
        ) / np.log(self.n_regimes)

        return pd.DataFrame(
            {
                "regime": predictions,
                "confidence": confidence,
                "entropy": entropy,
                "uncertainty": 1 - confidence,
            },
            index=features.index,
        )

    def _update_metadata(self, features: pd.DataFrame) -> None:
        """Update model metadata after fitting.

        Parameters
        ----------
        features : pd.DataFrame
            Training features
        """
        self._metadata.fit_date = datetime.now()
        self._metadata.n_samples_train = len(features)
        self._metadata.n_features = features.shape[1]
        self._metadata.feature_names = list(features.columns)

        # Compute training log-likelihood if possible
        try:
            self._metadata.train_log_likelihood = self._compute_log_likelihood(features)
        except Exception as e:
            logger.debug(f"Could not compute training log-likelihood: {e}")

    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        params = []

        # Add main parameters
        params.append(f"n_regimes={self.n_regimes}")
        if self.random_state is not None:
            params.append(f"random_state={self.random_state}")

        # Add fitted status
        if self.is_fitted:
            params.append("fitted=True")
            if self._metadata.n_samples_train:
                params.append(f"n_samples={self._metadata.n_samples_train}")

        return f"{class_name}({', '.join(params)})"

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool
            If True, return parameters for sub-estimators

        Returns
        -------
        Dict[str, Any]
            Parameter names mapped to values
        """
        return {
            "n_regimes": self.n_regimes,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "BaseRegimeDetector":
        """Set parameters for this estimator.

        Parameters
        ----------
        **params
            Estimator parameters

        Returns
        -------
        self
            Estimator instance
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Parameter {key} not found")
        return self
