"""Interfaces for regime detection models.

Defines clean abstractions following Interface Segregation Principle.
Models can implement only the interfaces they need.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd


class IRegimeDetector(ABC):
    """Core interface for regime detection models.

    Minimal required methods that all regime detectors must implement.
    """

    @abstractmethod
    def fit(self, features: pd.DataFrame, **kwargs) -> "IRegimeDetector":
        """Fit the regime detection model.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix for training
        **kwargs
            Additional fitting parameters

        Returns
        -------
        self
            Fitted model instance
        """
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime labels.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix for prediction

        Returns
        -------
        np.ndarray
            Regime labels
        """
        pass

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix for prediction

        Returns
        -------
        np.ndarray
            Regime probabilities
        """
        pass


class IOptimizable(ABC):
    """Interface for models that support regime count optimization."""

    @abstractmethod
    def optimize_n_regimes(
        self,
        features: pd.DataFrame,
        min_regimes: int = 2,
        max_regimes: int = 10,
        criterion: str = "bic",
    ) -> int:
        """Find optimal number of regimes.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        min_regimes : int
            Minimum number of regimes
        max_regimes : int
            Maximum number of regimes
        criterion : str
            Optimization criterion

        Returns
        -------
        int
            Optimal number of regimes
        """
        pass

    @abstractmethod
    def get_optimization_scores(self) -> Dict[int, float]:
        """Get optimization scores for each regime count tested.

        Returns
        -------
        Dict[int, float]
            Mapping of regime count to optimization score
        """
        pass


class IPersistable(ABC):
    """Interface for models that support persistence."""

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save model to file.

        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "IPersistable":
        """Load model from file.

        Parameters
        ----------
        filepath : str
            Path to the saved model

        Returns
        -------
        IPersistable
            Loaded model instance
        """
        pass


class IStatistical(ABC):
    """Interface for models that provide statistical information."""

    @abstractmethod
    def get_regime_statistics(
        self, features: pd.DataFrame, regimes: Optional[np.ndarray] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each regime.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        regimes : np.ndarray, optional
            Regime assignments

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Statistics per regime
        """
        pass

    @abstractmethod
    def get_transition_matrix(
        self, regimes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get regime transition matrix.

        Parameters
        ----------
        regimes : np.ndarray, optional
            Regime sequence

        Returns
        -------
        np.ndarray
            Transition matrix
        """
        pass


class IScorable(ABC):
    """Interface for models that support scoring."""

    @abstractmethod
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
            True regime labels
        scoring : str
            Scoring metric

        Returns
        -------
        float
            Score value
        """
        pass

    @abstractmethod
    def _compute_log_likelihood(self, features: pd.DataFrame) -> float:
        """Compute log-likelihood.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        float
            Log-likelihood value
        """
        pass

    @abstractmethod
    def _compute_aic(self, features: pd.DataFrame) -> float:
        """Compute AIC.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        float
            AIC value
        """
        pass

    @abstractmethod
    def _compute_bic(self, features: pd.DataFrame) -> float:
        """Compute BIC.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        float
            BIC value
        """
        pass


class ICrossValidatable(ABC):
    """Interface for models that support cross-validation."""

    @abstractmethod
    def cross_validate(
        self,
        features: pd.DataFrame,
        cv: Optional[Union[int, Any]] = None,
        scoring: str = "log_likelihood",
    ) -> Dict[str, np.ndarray]:
        """Perform cross-validation.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        cv : int or cross-validation splitter
            Cross-validation strategy
        scoring : str
            Scoring metric

        Returns
        -------
        Dict[str, np.ndarray]
            Cross-validation results
        """
        pass


class IConfidenceBased(ABC):
    """Interface for models that provide confidence scores."""

    @abstractmethod
    def get_regime_confidence(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get confidence scores for regime assignments.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        pd.DataFrame
            Confidence scores
        """
        pass

    @abstractmethod
    def get_prediction_uncertainty(self, features: pd.DataFrame) -> np.ndarray:
        """Get uncertainty in predictions.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        np.ndarray
            Uncertainty scores
        """
        pass


class IEnsemble(ABC):
    """Interface for ensemble models."""

    @abstractmethod
    def add_model(self, model: IRegimeDetector, weight: float = 1.0) -> None:
        """Add a model to the ensemble.

        Parameters
        ----------
        model : IRegimeDetector
            Model to add
        weight : float
            Model weight
        """
        pass

    @abstractmethod
    def get_model_weights(self) -> np.ndarray:
        """Get weights of models in ensemble.

        Returns
        -------
        np.ndarray
            Model weights
        """
        pass

    @abstractmethod
    def get_model_predictions(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get individual model predictions.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        Dict[str, np.ndarray]
            Predictions from each model
        """
        pass