"""Mixins for regime detection models.

Provides reusable functionality that can be mixed into model classes.
Following Single Responsibility Principle - each mixin handles one aspect.
"""

from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class PersistenceMixin:
    """Mixin for model persistence (save/load) functionality."""

    def save(self, filepath: str) -> None:
        """Save model to file.

        Parameters
        ----------
        filepath : str
            Path to save the model

        Raises
        ------
        ValueError
            If model is not fitted
        """
        if not hasattr(self, "is_fitted") or not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        from marketregimeml.models.base import PersistenceHandler

        PersistenceHandler.save(
            {
                "model": self,
                "metadata": getattr(self, "_metadata", {}),
                "class_name": self.__class__.__name__,
                "module_name": self.__class__.__module__,
            },
            Path(filepath),
        )
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str, format: Optional[str] = None):
        """Load model from file.

        Parameters
        ----------
        filepath : str
            Path to the saved model
        format : str, optional
            File format ('pickle', 'joblib', 'json')

        Returns
        -------
        Model instance
        """
        from marketregimeml.models.base import PersistenceHandler

        try:
            model_data = PersistenceHandler.load(Path(filepath), format)

            # Handle different data structures
            if isinstance(model_data, dict) and "model" in model_data:
                model = model_data["model"]
                if hasattr(model, "_metadata"):
                    model._metadata = model_data.get("metadata", {})
                return model
            elif hasattr(model_data, "is_fitted"):  # Direct model object
                return model_data
            else:
                # Try to reconstruct from dict
                model = cls(**model_data.get("params", {}))
                for key, value in model_data.items():
                    if key != "params" and not key.startswith("_"):
                        setattr(model, key, value)
                return model

        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            raise ValueError(f"Failed to load model: {e}")


class ValidationMixin:
    """Mixin for parameter and data validation."""

    def _validate_regime_count(
        self,
        n_regimes: int,
        min_samples_per_regime: int = 10,
        n_samples: Optional[int] = None,
    ) -> None:
        """Validate regime count parameter.

        Parameters
        ----------
        n_regimes : int
            Number of regimes
        min_samples_per_regime : int
            Minimum samples needed per regime
        n_samples : int, optional
            Total number of samples

        Raises
        ------
        ValueError
            If regime count is invalid
        """
        if n_regimes < 2:
            raise ValueError(f"n_regimes must be at least 2, got {n_regimes}")

        if n_regimes > 10:
            warnings.warn(
                f"n_regimes={n_regimes} is very high. "
                "Consider using fewer regimes for better interpretability.",
                UserWarning,
            )

        if n_samples is not None:
            min_required = n_regimes * min_samples_per_regime
            if n_samples < min_required:
                raise ValueError(
                    f"Insufficient samples ({n_samples}) for {n_regimes} regimes. "
                    f"Need at least {min_required} samples."
                )

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare feature DataFrame.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        pd.DataFrame
            Validated features

        Raises
        ------
        ValueError
            If features are invalid
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty")

        if features.isna().any().any():
            n_missing = features.isna().sum().sum()
            logger.warning(f"Features contain {n_missing} missing values")
            features = features.fillna(method="ffill").fillna(method="bfill")

        if features.shape[0] < 20:
            raise ValueError(
                f"Insufficient samples ({features.shape[0]}). Need at least 20."
            )

        return features


class OptimizationMixin:
    """Mixin for regime count optimization."""

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
            Minimum number of regimes to test
        max_regimes : int
            Maximum number of regimes to test
        criterion : str
            Optimization criterion ('aic', 'bic')

        Returns
        -------
        int
            Optimal number of regimes
        """
        best_score = float("inf")
        best_n = min_regimes

        for n in range(min_regimes, max_regimes + 1):
            try:
                # Create a new instance with n regimes
                model = self.__class__(n_regimes=n, random_state=self.random_state)
                model.fit(features)

                if criterion == "bic":
                    score = model._compute_bic(features)
                else:
                    score = model._compute_aic(features)

                if score < best_score:
                    best_score = score
                    best_n = n

                logger.debug(f"n_regimes={n}, {criterion}={score:.2f}")

            except Exception as e:
                logger.warning(f"Failed to fit with n_regimes={n}: {e}")
                continue

        logger.info(f"Optimal n_regimes={best_n} with {criterion}={best_score:.2f}")
        return best_n


class StatisticsMixin:
    """Mixin for computing regime statistics."""

    def get_regime_statistics(
        self, features: pd.DataFrame, regimes: Optional[np.ndarray] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Compute statistics for each regime.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        regimes : np.ndarray, optional
            Regime assignments. If None, will predict.

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Statistics for each regime
        """
        if regimes is None:
            if not self.is_fitted:
                raise ValueError("Model must be fitted to compute statistics")
            regimes = self.predict(features)

        statistics = {}
        unique_regimes = np.unique(regimes)

        for regime in unique_regimes:
            mask = regimes == regime
            regime_features = features[mask]

            if len(regime_features) > 0:
                statistics[regime] = {
                    "count": len(regime_features),
                    "frequency": mask.mean(),
                    "mean": regime_features.mean().to_dict(),
                    "std": regime_features.std().to_dict(),
                    "median": regime_features.median().to_dict(),
                    "duration": self._calculate_mean_duration(regimes, regime),
                }
            else:
                statistics[regime] = {
                    "count": 0,
                    "frequency": 0.0,
                    "mean": {},
                    "std": {},
                    "median": {},
                    "duration": 0.0,
                }

        return statistics

    def _calculate_mean_duration(self, regimes: np.ndarray, regime: int) -> float:
        """Calculate mean duration in a regime.

        Parameters
        ----------
        regimes : np.ndarray
            Sequence of regime assignments
        regime : int
            Regime to calculate duration for

        Returns
        -------
        float
            Mean duration in the regime
        """
        if len(regimes) == 0:
            return 0.0

        in_regime = regimes == regime
        changes = np.diff(np.concatenate([[False], in_regime, [False]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        if len(starts) == 0 or len(ends) == 0:
            return 0.0

        durations = ends - starts
        return float(np.mean(durations))

    def get_transition_matrix(
        self, regimes: Optional[np.ndarray] = None, normalize: bool = True
    ) -> np.ndarray:
        """Compute regime transition matrix.

        Parameters
        ----------
        regimes : np.ndarray, optional
            Regime sequence. If None, uses last prediction.
        normalize : bool
            Whether to normalize rows to probabilities

        Returns
        -------
        np.ndarray
            Transition matrix
        """
        if regimes is None:
            if not hasattr(self, "_last_regimes"):
                raise ValueError("No regime sequence available")
            regimes = self._last_regimes

        n_regimes = len(np.unique(regimes))
        trans_matrix = np.zeros((n_regimes, n_regimes))

        for i in range(len(regimes) - 1):
            trans_matrix[regimes[i], regimes[i + 1]] += 1

        if normalize and trans_matrix.sum() > 0:
            row_sums = trans_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            trans_matrix = trans_matrix / row_sums

        return trans_matrix


class CrossValidationMixin:
    """Mixin for cross-validation functionality."""

    def cross_validate(
        self,
        features: pd.DataFrame,
        cv: Optional[Union[int, TimeSeriesSplit]] = None,
        scoring: str = "log_likelihood",
    ) -> Dict[str, np.ndarray]:
        """Perform cross-validation.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        cv : int or TimeSeriesSplit, optional
            Cross-validation splitter
        scoring : str
            Scoring metric

        Returns
        -------
        Dict[str, np.ndarray]
            Cross-validation scores
        """
        if cv is None:
            cv = TimeSeriesSplit(n_splits=5)
        elif isinstance(cv, int):
            cv = TimeSeriesSplit(n_splits=cv)

        scores = []
        train_scores = []

        for train_idx, test_idx in cv.split(features):
            train_data = features.iloc[train_idx]
            test_data = features.iloc[test_idx]

            # Create new model instance
            model = self.__class__(
                n_regimes=self.n_regimes, random_state=self.random_state
            )
            model.fit(train_data)

            if scoring == "log_likelihood":
                train_score = model._compute_log_likelihood(train_data)
                test_score = model._compute_log_likelihood(test_data)
            elif scoring == "aic":
                train_score = -model._compute_aic(
                    train_data
                )  # Negative for higher=better
                test_score = -model._compute_aic(test_data)
            elif scoring == "bic":
                train_score = -model._compute_bic(train_data)
                test_score = -model._compute_bic(test_data)
            else:
                raise ValueError(f"Unknown scoring metric: {scoring}")

            train_scores.append(train_score)
            scores.append(test_score)

        return {
            "test_scores": np.array(scores),
            "train_scores": np.array(train_scores),
            "mean_test_score": np.mean(scores),
            "std_test_score": np.std(scores),
            "mean_train_score": np.mean(train_scores),
            "std_train_score": np.std(train_scores),
        }


class FuzzyLogicMixin:
    """Mixin for fuzzy regime assignment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fuzzy_matching = False
        self._fuzzy_threshold = 0.7

    @property
    def fuzzy_matching(self) -> bool:
        """Whether to use fuzzy regime matching."""
        return self._fuzzy_matching

    @fuzzy_matching.setter
    def fuzzy_matching(self, value: bool) -> None:
        """Set fuzzy matching mode."""
        self._fuzzy_matching = bool(value)

    @property
    def fuzzy_threshold(self) -> float:
        """Threshold for fuzzy regime assignment."""
        return self._fuzzy_threshold

    @fuzzy_threshold.setter
    def fuzzy_threshold(self, value: float) -> None:
        """Set fuzzy threshold."""
        if not 0.5 <= value <= 1.0:
            raise ValueError("Fuzzy threshold must be between 0.5 and 1.0")
        self._fuzzy_threshold = value

    def apply_fuzzy_logic(
        self, probabilities: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply fuzzy logic to regime assignments.

        Parameters
        ----------
        probabilities : np.ndarray
            Regime probabilities
        threshold : float, optional
            Confidence threshold

        Returns
        -------
        regimes : np.ndarray
            Regime assignments (-1 for uncertain)
        confidence : np.ndarray
            Confidence scores
        """
        if threshold is None:
            threshold = self._fuzzy_threshold

        max_probs = probabilities.max(axis=1)
        regimes = probabilities.argmax(axis=1)

        # Mark uncertain predictions
        uncertain = max_probs < threshold
        regimes[uncertain] = -1

        return regimes, max_probs
