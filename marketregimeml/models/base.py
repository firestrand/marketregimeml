"""Base regime detector interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import pickle
from pathlib import Path
import warnings
import gzip
import joblib
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for regime optimization.

    Groups optimization-related parameters to improve cohesion.
    """

    auto_optimize: bool = False
    min_regimes: int = 2
    max_regimes: int = 9
    optimal_regimes: Optional[int] = None
    regime_scores: Dict[int, float] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate optimization configuration."""
        if self.min_regimes < 2:
            raise ValueError("min_regimes must be at least 2")
        if self.max_regimes > 20:
            raise ValueError("max_regimes should not exceed 20")
        if self.min_regimes > self.max_regimes:
            raise ValueError("min_regimes must be <= max_regimes")


@dataclass
class FuzzyConfig:
    """Configuration for fuzzy regime matching.

    Groups fuzzy matching parameters to improve cohesion.
    """

    enabled: bool = False
    threshold: float = 0.7

    def __post_init__(self):
        """Validate and adjust threshold."""
        self.threshold = max(0.5, min(1.0, self.threshold))

    def validate(self) -> None:
        """Validate fuzzy configuration."""
        if self.threshold < 0.5 or self.threshold > 1.0:
            raise ValueError("fuzzy_threshold must be between 0.5 and 1.0")


@dataclass
class ModelMetadata:
    """Metadata about the fitted model.

    Groups model metadata to improve cohesion.
    """

    fit_date: Optional[datetime] = None
    n_features: Optional[int] = None
    feature_names: Optional[List[str]] = None
    train_log_likelihood: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class PersistenceHandler:
    """Handles model persistence operations (save/load).

    This class follows Single Responsibility Principle by handling
    only persistence-related operations.
    """

    @staticmethod
    def save(
        model_data: Dict,
        filepath: Path,
        format: str = "pickle",
        compress: bool = False,
    ) -> None:
        """Save model data to file.

        Args:
            model_data: Dictionary containing model state
            filepath: Path to save the model
            format: Serialization format ('pickle', 'joblib', 'json')
            compress: Whether to compress the saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "pickle":
            if compress:
                if not str(filepath).endswith(".gz"):
                    filepath = Path(str(filepath) + ".gz")
                with gzip.open(filepath, "wb") as f:
                    pickle.dump(
                        model_data, f, protocol=pickle.HIGHEST_PROTOCOL
                    )
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(
                        model_data, f, protocol=pickle.HIGHEST_PROTOCOL
                    )

        elif format == "joblib":
            joblib.dump(model_data, filepath, compress=3 if compress else 0)

        elif format == "json":
            content = json.dumps(model_data, indent=2)
            if compress:
                with gzip.open(
                    str(filepath) + ".gz", "wt", encoding="utf-8"
                ) as f:
                    f.write(content)
            else:
                with open(filepath, "w") as f:
                    f.write(content)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def _detect_format(filepath: Path) -> str:
        """Auto-detect file format from extension."""
        filepath_str = str(filepath)
        if ".json" in filepath_str:
            return "json"
        elif ".joblib" in filepath_str:
            return "joblib"
        else:
            return "pickle"

    @staticmethod
    def _find_file(filepath: Path) -> Path:
        """Find file, checking for compressed version if needed."""
        if filepath.exists():
            return filepath
        compressed = Path(str(filepath) + ".gz")
        if compressed.exists():
            return compressed
        raise FileNotFoundError(f"Model file not found: {filepath}")

    @staticmethod
    def load(filepath: Path, format: Optional[str] = None) -> Dict:
        """Load model data from file.

        Args:
            filepath: Path to load the model from
            format: Serialization format (auto-detect if None)

        Returns:
            Dictionary containing model state
        """
        filepath = PersistenceHandler._find_file(Path(filepath))

        if format is None:
            format = PersistenceHandler._detect_format(filepath)

        # Load based on format
        is_compressed = str(filepath).endswith(".gz")

        if format == "pickle":
            open_func = gzip.open if is_compressed else open
            with open_func(filepath, "rb") as f:
                return pickle.load(f)

        elif format == "joblib":
            return joblib.load(filepath)

        elif format == "json":
            if is_compressed:
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(filepath, "r") as f:
                    return json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")


class BaseRegimeDetector(ABC):
    """Abstract base class for regime detection models.

    Defines the interface that all regime detectors must implement.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        random_state: Optional[int] = None,
        fuzzy_matching: bool = False,
        fuzzy_threshold: float = 0.7,
        auto_optimize_regimes: bool = False,
        min_regimes: int = 2,
        max_regimes: int = 9,
        **kwargs,
    ):
        """Initialize base regime detector.

        Args:
            n_regimes: Number of regimes to detect (recommended odd numbers 3,5,7,9)
            random_state: Random seed for reproducibility
            fuzzy_matching: Enable fuzzy regime assignment
            fuzzy_threshold: Threshold for crisp vs fuzzy assignment (0.5-1.0)
            auto_optimize_regimes: Automatically find optimal number of regimes
            min_regimes: Minimum regimes for auto-optimization
            max_regimes: Maximum regimes for auto-optimization
            **kwargs: Additional model-specific parameters
        """
        # Store init parameters for get_params
        self._init_params = {
            "n_regimes": n_regimes,
            "random_state": random_state,
            "fuzzy_matching": fuzzy_matching,
            "fuzzy_threshold": fuzzy_threshold,
            "auto_optimize_regimes": auto_optimize_regimes,
            "min_regimes": min_regimes,
            "max_regimes": max_regimes,
        }
        self._init_params.update(kwargs)

        # Core attributes (only 6 instance attributes now!)
        self.n_regimes = self._validate_regime_count(
            n_regimes, min_regimes, max_regimes
        )
        self.random_state = random_state
        self.model = None
        self.is_fitted = False

        # Use composition for configuration objects
        self.fuzzy_config = FuzzyConfig(
            enabled=fuzzy_matching, threshold=fuzzy_threshold
        )
        self.fuzzy_config.validate()

        self.optimization_config = OptimizationConfig(
            auto_optimize=auto_optimize_regimes,
            min_regimes=min_regimes,
            max_regimes=max_regimes,
        )
        self.optimization_config.validate()

        self.metadata = ModelMetadata()

        # Regime labels and names
        self.regime_names = self._default_regime_names()

        logger.info(
            f"Initialized {self.__class__.__name__} with {self.n_regimes} regimes"
            f"{' (fuzzy)' if self.fuzzy_config.enabled else ''}"
            f"{' (auto-optimize)' if self.optimization_config.auto_optimize else ''}"
        )

    # Backward compatibility properties
    @property
    def fuzzy_matching(self) -> bool:
        """Get fuzzy matching enabled state."""
        return (
            self.fuzzy_config.enabled
            if hasattr(self, "fuzzy_config")
            else False
        )

    @fuzzy_matching.setter
    def fuzzy_matching(self, value: bool) -> None:
        """Set fuzzy matching enabled state."""
        if hasattr(self, "fuzzy_config"):
            self.fuzzy_config.enabled = value

    @property
    def fuzzy_threshold(self) -> float:
        """Get fuzzy threshold."""
        return (
            self.fuzzy_config.threshold
            if hasattr(self, "fuzzy_config")
            else 0.7
        )

    @fuzzy_threshold.setter
    def fuzzy_threshold(self, value: float) -> None:
        """Set fuzzy threshold."""
        if hasattr(self, "fuzzy_config"):
            self.fuzzy_config.threshold = max(0.5, min(1.0, value))

    @property
    def auto_optimize_regimes(self) -> bool:
        """Get auto-optimize enabled state."""
        return (
            self.optimization_config.auto_optimize
            if hasattr(self, "optimization_config")
            else False
        )

    @auto_optimize_regimes.setter
    def auto_optimize_regimes(self, value: bool) -> None:
        """Set auto-optimize enabled state."""
        if hasattr(self, "optimization_config"):
            self.optimization_config.auto_optimize = value

    @property
    def min_regimes(self) -> int:
        """Get minimum regimes."""
        return (
            self.optimization_config.min_regimes
            if hasattr(self, "optimization_config")
            else 2
        )

    @min_regimes.setter
    def min_regimes(self, value: int) -> None:
        """Set minimum regimes."""
        if hasattr(self, "optimization_config"):
            self.optimization_config.min_regimes = value

    @property
    def max_regimes(self) -> int:
        """Get maximum regimes."""
        return (
            self.optimization_config.max_regimes
            if hasattr(self, "optimization_config")
            else 9
        )

    @max_regimes.setter
    def max_regimes(self, value: int) -> None:
        """Set maximum regimes."""
        if hasattr(self, "optimization_config"):
            self.optimization_config.max_regimes = value

    @property
    def optimal_regimes(self) -> Optional[int]:
        """Get optimal regimes."""
        return (
            self.optimization_config.optimal_regimes
            if hasattr(self, "optimization_config")
            else None
        )

    @optimal_regimes.setter
    def optimal_regimes(self, value: Optional[int]) -> None:
        """Set optimal regimes."""
        if hasattr(self, "optimization_config"):
            self.optimization_config.optimal_regimes = value

    @property
    def regime_scores(self) -> Dict[int, float]:
        """Get regime scores."""
        return (
            self.optimization_config.regime_scores
            if hasattr(self, "optimization_config")
            else {}
        )

    @regime_scores.setter
    def regime_scores(self, value: Dict[int, float]) -> None:
        """Set regime scores."""
        if hasattr(self, "optimization_config"):
            self.optimization_config.regime_scores = value

    @property
    def fit_date(self) -> Optional[datetime]:
        """Get fit date."""
        return self.metadata.fit_date if hasattr(self, "metadata") else None

    @fit_date.setter
    def fit_date(self, value: Optional[datetime]) -> None:
        """Set fit date."""
        if hasattr(self, "metadata"):
            self.metadata.fit_date = value

    @property
    def n_features(self) -> Optional[int]:
        """Get number of features."""
        return self.metadata.n_features if hasattr(self, "metadata") else None

    @n_features.setter
    def n_features(self, value: Optional[int]) -> None:
        """Set number of features."""
        if hasattr(self, "metadata"):
            self.metadata.n_features = value

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Get feature names."""
        return (
            self.metadata.feature_names if hasattr(self, "metadata") else None
        )

    @feature_names.setter
    def feature_names(self, value: Optional[List[str]]) -> None:
        """Set feature names."""
        if hasattr(self, "metadata"):
            self.metadata.feature_names = value

    @property
    def train_log_likelihood(self) -> Optional[float]:
        """Get training log likelihood."""
        return (
            self.metadata.train_log_likelihood
            if hasattr(self, "metadata")
            else None
        )

    @train_log_likelihood.setter
    def train_log_likelihood(self, value: Optional[float]) -> None:
        """Set training log likelihood."""
        if hasattr(self, "metadata"):
            self.metadata.train_log_likelihood = value

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics."""
        return self.metadata.diagnostics if hasattr(self, "metadata") else {}

    @diagnostics.setter
    def diagnostics(self, value: Dict[str, Any]) -> None:
        """Set diagnostics."""
        if hasattr(self, "metadata"):
            self.metadata.diagnostics = value

    def _validate_regime_count(
        self, n_regimes: int, min_regimes: int, max_regimes: int
    ) -> int:
        """Validate and optimize regime count.

        Args:
            n_regimes: Requested number of regimes
            min_regimes: Minimum allowed regimes
            max_regimes: Maximum allowed regimes

        Returns:
            Validated regime count
        """
        if n_regimes < 1:
            warnings.warn("Minimum 1 regime required. Setting to 1.")
            return 1

        if n_regimes > max_regimes:
            warnings.warn(
                f"Maximum {max_regimes} regimes recommended. Setting to {max_regimes}."
            )
            return max_regimes

        # Recommend odd numbers for better regime separation
        if n_regimes > 2 and n_regimes % 2 == 0:
            recommended = (
                n_regimes + 1 if n_regimes < max_regimes else n_regimes - 1
            )
            logger.info(
                f"Odd numbers recommended for regimes. Consider {recommended} instead of {n_regimes}"
            )

        return n_regimes

    # Predefined regime names for common configurations
    REGIME_NAME_TEMPLATES = {
        2: {0: "Bear", 1: "Bull"},
        3: {0: "Bear", 1: "Neutral", 2: "Bull"},
        4: {0: "Crisis", 1: "Bear", 2: "Bull", 3: "Rally"},
        5: {
            0: "Deep_Bear",
            1: "Bear",
            2: "Neutral",
            3: "Bull",
            4: "Strong_Bull",
        },
        7: {
            0: "Crisis",
            1: "Deep_Bear",
            2: "Bear",
            3: "Neutral",
            4: "Bull",
            5: "Strong_Bull",
            6: "Euphoria",
        },
        9: {
            0: "Crash",
            1: "Crisis",
            2: "Deep_Bear",
            3: "Bear",
            4: "Neutral",
            5: "Bull",
            6: "Strong_Bull",
            7: "Rally",
            8: "Euphoria",
        },
    }

    def _default_regime_names(self) -> Dict[int, str]:
        """Generate default regime names based on financial market conventions.

        Returns:
            Dictionary mapping regime number to meaningful name
        """
        # Use predefined names if available
        if self.n_regimes in self.REGIME_NAME_TEMPLATES:
            return self.REGIME_NAME_TEMPLATES[self.n_regimes].copy()

        # Generate names for custom regime counts
        return self._generate_custom_regime_names()

    def _generate_custom_regime_names(self) -> Dict[int, str]:
        """Generate regime names for non-standard regime counts.

        Returns:
            Dictionary mapping regime number to name
        """
        if self.n_regimes % 2 == 1:  # Odd number - use center as neutral
            center = self.n_regimes // 2
            names = {}
            for i in range(self.n_regimes):
                if i < center:
                    names[i] = f"Bear_L{center-i}"
                elif i == center:
                    names[i] = "Neutral"
                else:
                    names[i] = f"Bull_L{i-center}"
            return names
        else:  # Even number - no neutral
            return {i: f"Regime_{i}" for i in range(self.n_regimes)}

    @abstractmethod
    def fit(self, features: pd.DataFrame, **kwargs) -> "BaseRegimeDetector":
        """Fit the regime detection model.

        Args:
            features: Feature matrix (n_samples, n_features)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime labels.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Array of regime labels
        """
        pass

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Array of regime probabilities (n_samples, n_regimes)
        """
        pass

    def fit_predict(self, features: pd.DataFrame, **kwargs) -> np.ndarray:
        """Fit model and return predictions.

        Args:
            features: Feature matrix
            **kwargs: Additional fitting parameters

        Returns:
            Array of regime labels
        """
        self.fit(features, **kwargs)
        return self.predict(features)

    def score(
        self, features: pd.DataFrame, metric: str = "log_likelihood"
    ) -> float:
        """Score the model on given features.

        Args:
            features: Feature matrix
            metric: Scoring metric to use

        Returns:
            Score value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        if metric == "log_likelihood":
            return self._compute_log_likelihood(features)
        elif metric == "aic":
            return self._compute_aic(features)
        elif metric == "bic":
            return self._compute_bic(features)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @abstractmethod
    def _compute_log_likelihood(self, features: pd.DataFrame) -> float:
        """Compute log-likelihood of the data.

        Args:
            features: Feature matrix

        Returns:
            Log-likelihood value
        """
        pass

    def _compute_aic(self, features: pd.DataFrame) -> float:
        """Compute Akaike Information Criterion.

        Args:
            features: Feature matrix

        Returns:
            AIC value
        """
        log_likelihood = self._compute_log_likelihood(features)
        n_params = self._count_parameters()
        return 2 * n_params - 2 * log_likelihood

    def _compute_bic(self, features: pd.DataFrame) -> float:
        """Compute Bayesian Information Criterion.

        Args:
            features: Feature matrix

        Returns:
            BIC value
        """
        log_likelihood = self._compute_log_likelihood(features)
        n_params = self._count_parameters()
        n_samples = len(features)
        return np.log(n_samples) * n_params - 2 * log_likelihood

    @abstractmethod
    def _count_parameters(self) -> int:
        """Count the number of model parameters.

        Returns:
            Number of parameters
        """
        pass

    def get_regime_statistics(
        self, features: pd.DataFrame, regimes: Optional[np.ndarray] = None
    ) -> Dict[int, Dict[str, float]]:
        """Calculate statistics for each regime.

        Args:
            features: Feature matrix
            regimes: Optional regime labels (will predict if not provided)

        Returns:
            Dictionary of statistics per regime
        """
        if regimes is None:
            if not self.is_fitted:
                raise ValueError("Model must be fitted to predict regimes")
            regimes = self.predict(features)

        stats = {}

        for regime in range(self.n_regimes):
            mask = regimes == regime
            if mask.sum() > 0:
                regime_features = features[mask]

                stats[regime] = {
                    "count": mask.sum(),
                    "percentage": mask.sum() / len(regimes) * 100,
                    "mean_duration": self._calculate_mean_duration(
                        regimes, regime
                    ),
                }

                # Add feature statistics
                for col in features.columns:
                    stats[regime][f"{col}_mean"] = regime_features[col].mean()
                    stats[regime][f"{col}_std"] = regime_features[col].std()

        return stats

    def _calculate_mean_duration(
        self, regimes: np.ndarray, regime: int
    ) -> float:
        """Calculate mean duration of a regime.

        Args:
            regimes: Array of regime labels
            regime: Regime to calculate duration for

        Returns:
            Mean duration in periods
        """
        durations = []
        current_duration = 0

        for r in regimes:
            if r == regime:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0

    def get_transition_matrix(
        self,
        regimes: Optional[np.ndarray] = None,
        features: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Calculate regime transition probability matrix.

        Args:
            regimes: Regime sequence
            features: Feature matrix (to predict regimes if not provided)

        Returns:
            Transition probability matrix (n_regimes, n_regimes)
        """
        if regimes is None:
            if features is None:
                raise ValueError("Either regimes or features must be provided")
            regimes = self.predict(features)

        trans_matrix = np.zeros((self.n_regimes, self.n_regimes))

        for i in range(len(regimes) - 1):
            trans_matrix[regimes[i], regimes[i + 1]] += 1

        # Normalize rows to get probabilities
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        trans_matrix = trans_matrix / row_sums

        return trans_matrix

    def get_regime_confidence(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence scores for regime predictions.

        Args:
            features: Feature matrix

        Returns:
            DataFrame with regime predictions and confidence scores
        """
        proba = self.predict_proba(features)
        regimes = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)

        # Calculate entropy as uncertainty measure
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        max_entropy = np.log(self.n_regimes)
        uncertainty = entropy / max_entropy  # Normalized 0-1

        result = pd.DataFrame(
            {
                "regime": regimes,
                "regime_name": [self.regime_names[r] for r in regimes],
                "confidence": confidence,
                "uncertainty": uncertainty,
            },
            index=features.index,
        )

        # Add probability columns
        for i in range(self.n_regimes):
            result[f"prob_regime_{i}"] = proba[:, i]

        return result

    def cross_validate(
        self,
        features: pd.DataFrame,
        n_splits: int = 5,
        metric: str = "log_likelihood",
    ) -> Dict[str, List[float]]:
        """Perform time series cross-validation.

        Args:
            features: Feature matrix
            n_splits: Number of CV splits
            metric: Metric to evaluate

        Returns:
            Dictionary with train and test scores
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        train_scores = []
        test_scores = []

        for train_idx, test_idx in tscv.split(features):
            train_features = features.iloc[train_idx]
            test_features = features.iloc[test_idx]

            # Fit on train
            self.fit(train_features)

            # Score on both sets
            train_score = self.score(train_features, metric=metric)
            test_score = self.score(test_features, metric=metric)

            train_scores.append(train_score)
            test_scores.append(test_score)

        return {
            "train_scores": train_scores,
            "test_scores": test_scores,
            "train_mean": np.mean(train_scores),
            "train_std": np.std(train_scores),
            "test_mean": np.mean(test_scores),
            "test_std": np.std(test_scores),
        }

    def save(self, filepath: str) -> None:
        """Save model to file.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "n_regimes": self.n_regimes,
            "is_fitted": self.is_fitted,
            "fit_date": self.metadata.fit_date,
            "n_features": self.metadata.n_features,
            "feature_names": self.metadata.feature_names,
            "regime_names": self.regime_names,
            "diagnostics": self.metadata.diagnostics,
            "class_name": self.__class__.__name__,
        }

        if filepath.suffix == ".json":
            # JSON format (limited, mainly for metadata)
            json_data = {
                k: v for k, v in model_data.items() if k not in ["model"]
            }
            with open(filepath, "w") as f:
                json.dump(json_data, f, indent=2, default=str)
        else:
            # Pickle format (full model)
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "BaseRegimeDetector":
        """Load model from file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        if filepath.suffix == ".json":
            raise ValueError(
                "Cannot load full model from JSON. Use pickle format."
            )

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Create instance
        instance = cls(n_regimes=model_data["n_regimes"])

        # Restore state
        instance.model = model_data["model"]
        instance.is_fitted = model_data["is_fitted"]
        instance.fit_date = model_data["fit_date"]
        instance.n_features = model_data["n_features"]
        instance.feature_names = model_data["feature_names"]
        instance.regime_names = model_data["regime_names"]
        instance.diagnostics = model_data["diagnostics"]

        logger.info(f"Model loaded from {filepath}")
        return instance

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics.

        Returns:
            Dictionary of diagnostic metrics
        """
        return self.metadata.diagnostics

    def set_regime_names(self, names: Dict[int, str]) -> None:
        """Set custom regime names.

        Args:
            names: Dictionary mapping regime number to name
        """
        if len(names) != self.n_regimes:
            raise ValueError(
                f"Must provide names for all {self.n_regimes} regimes"
            )
        self.regime_names = names
        logger.info(f"Updated regime names: {names}")

    def predict_fuzzy(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict fuzzy regime assignments.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Dictionary containing:
                - 'crisp': Hard regime assignments
                - 'fuzzy': Fuzzy membership degrees
                - 'confidence': Confidence scores
                - 'dominant': Most likely regime per sample
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get probabilities from the model
        probabilities = self.predict_proba(features)

        # Calculate fuzzy assignments
        fuzzy_result = self._compute_fuzzy_assignment(probabilities)

        return fuzzy_result

    def _compute_fuzzy_assignment(
        self, probabilities: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute fuzzy regime assignments from probabilities.

        Args:
            probabilities: Regime probabilities (n_samples, n_regimes)

        Returns:
            Dictionary with fuzzy assignment results
        """
        n_samples, n_regimes = probabilities.shape

        # Crisp assignments (winner-take-all)
        crisp = np.argmax(probabilities, axis=1)

        # Confidence scores (max probability)
        confidence = np.max(probabilities, axis=1)

        # Fuzzy memberships based on threshold
        fuzzy = np.zeros((n_samples, n_regimes))

        for i in range(n_samples):
            probs = probabilities[i]
            max_prob = np.max(probs)

            if max_prob >= self.fuzzy_config.threshold:
                # High confidence: use crisp assignment
                fuzzy[i, crisp[i]] = 1.0
            else:
                # Low confidence: use fuzzy assignment
                # Normalize probabilities above a minimum threshold
                min_threshold = (
                    1.0 / n_regimes * 0.5
                )  # Half of uniform probability
                relevant_probs = np.where(probs >= min_threshold, probs, 0)

                if np.sum(relevant_probs) > 0:
                    fuzzy[i] = relevant_probs / np.sum(relevant_probs)
                else:
                    # Fallback to uniform distribution
                    fuzzy[i] = np.ones(n_regimes) / n_regimes

        # Dominant regime (can be different from crisp if fuzzy)
        dominant = np.argmax(fuzzy, axis=1)

        return {
            "crisp": crisp,
            "fuzzy": fuzzy,
            "confidence": confidence,
            "dominant": dominant,
            "probabilities": probabilities,
        }

    def analyze_regime_transitions(
        self, regimes: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze regime transition patterns.

        Args:
            regimes: Array of regime assignments

        Returns:
            Dictionary with transition analysis
        """
        # Transition matrix
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))

        for i in range(1, len(regimes)):
            prev_regime = regimes[i - 1]
            curr_regime = regimes[i]
            transition_matrix[prev_regime, curr_regime] += 1

        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_probs = np.divide(
            transition_matrix,
            row_sums[:, np.newaxis],
            out=np.zeros_like(transition_matrix),
            where=row_sums[:, np.newaxis] != 0,
        )

        # Regime statistics
        regime_counts = np.bincount(regimes, minlength=self.n_regimes)
        regime_durations = self._calculate_regime_durations(regimes)

        # Stability metrics
        persistence = np.diag(
            transition_probs
        )  # Probability of staying in same regime
        volatility = 1 - persistence  # Regime change probability

        return {
            "transition_matrix": transition_matrix,
            "transition_probabilities": transition_probs,
            "regime_counts": regime_counts,
            "regime_frequencies": regime_counts / len(regimes),
            "average_durations": regime_durations,
            "persistence": persistence,
            "volatility": volatility,
            "n_transitions": len(regimes) - 1,
        }

    def _calculate_regime_durations(
        self, regimes: np.ndarray
    ) -> Dict[int, float]:
        """Calculate average duration for each regime.

        Args:
            regimes: Array of regime assignments

        Returns:
            Dictionary mapping regime to average duration
        """
        durations = {i: [] for i in range(self.n_regimes)}

        if len(regimes) == 0:
            return {i: 0.0 for i in range(self.n_regimes)}

        current_regime = regimes[0]
        current_duration = 1

        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = regimes[i]
                current_duration = 1

        # Add the last regime duration
        durations[current_regime].append(current_duration)

        # Calculate averages
        avg_durations = {}
        for regime, dur_list in durations.items():
            avg_durations[regime] = np.mean(dur_list) if dur_list else 0.0

        return avg_durations

    def _calculate_optimization_score(
        self,
        model: "BaseRegimeDetector",
        features: pd.DataFrame,
        criteria: str,
    ) -> float:
        """Calculate optimization score for a model.

        Args:
            model: Fitted model to score
            features: Feature matrix
            criteria: Scoring criteria

        Returns:
            Score value (lower is better for optimization)
        """
        if criteria == "aic":
            log_likelihood = model._compute_log_likelihood(features.values)
            n_params = model._count_parameters()
            return 2 * n_params - 2 * log_likelihood

        elif criteria == "bic":
            log_likelihood = model._compute_log_likelihood(features.values)
            n_params = model._count_parameters()
            n_samples = len(features)
            return np.log(n_samples) * n_params - 2 * log_likelihood

        elif criteria == "silhouette":
            from sklearn.metrics import silhouette_score

            predictions = model.predict(features)
            if len(np.unique(predictions)) > 1:
                return -silhouette_score(features.values, predictions)
            return np.inf

        elif criteria == "calinski_harabasz":
            from sklearn.metrics import calinski_harabasz_score

            predictions = model.predict(features)
            if len(np.unique(predictions)) > 1:
                return -calinski_harabasz_score(features.values, predictions)
            return np.inf

        else:
            raise ValueError(f"Unknown criteria: {criteria}")

    def _create_model_copy(self, n_regimes: int) -> "BaseRegimeDetector":
        """Create a copy of the model with different regime count.

        Args:
            n_regimes: Number of regimes for the copy

        Returns:
            New model instance with specified regime count
        """
        init_params = {
            "n_regimes": n_regimes,
            "random_state": self.random_state,
            "fuzzy_matching": self.fuzzy_config.enabled,
            "fuzzy_threshold": self.fuzzy_config.threshold,
            "auto_optimize_regimes": False,  # Prevent infinite recursion
            "min_regimes": self.optimization_config.min_regimes,
            "max_regimes": self.optimization_config.max_regimes,
        }

        # Add model-specific parameters
        for attr in [
            "covariance_type",
            "n_iter",
            "tol",
            "init_method",
            "verbose",
        ]:
            if hasattr(self, attr):
                init_params[attr] = getattr(self, attr)

        return self.__class__(**init_params)

    def optimize_regime_count(
        self, features: pd.DataFrame, criteria: str = "aic", *,
        min_silhouette: float = 0.0,
        min_cluster_prop: float = 0.0,
    ) -> Dict[str, Any]:
        """Find optimal number of regimes using model selection criteria.

        Args:
            features: Feature matrix for optimization
            criteria: Selection criteria ('aic', 'bic', 'silhouette', 'calinski_harabasz')

        Returns:
            Dictionary with optimization results
        """
        if not self.optimization_config.auto_optimize:
            logger.warning(
                "Auto-optimization not enabled. Use auto_optimize_regimes=True"
            )
            return {"optimal_regimes": self.n_regimes, "scores": {}}

        scores = {}
        models = {}

        logger.info(
            f"Optimizing regime count from {self.optimization_config.min_regimes} to {self.optimization_config.max_regimes}"
        )

        # Try each regime count
        for n in range(
            self.optimization_config.min_regimes,
            self.optimization_config.max_regimes + 1,
        ):
            try:
                model_copy = self._create_model_copy(n)
                model_copy.fit(features)

                score = self._calculate_optimization_score(
                    model_copy, features, criteria
                )
                scores[n] = score
                models[n] = model_copy

                logger.info(f"  {n} regimes: {criteria.upper()} = {score:.4f}")

            except Exception as e:
                logger.warning(f"Failed to fit model with {n} regimes: {e}")
                scores[n] = np.inf

        # Find optimal number by criteria
        optimal_n = min(scores.keys(), key=lambda k: scores[k])
        chosen_model = models.get(optimal_n)

        # Guardrails: if using likelihood criteria and guardrails are set, verify separation
        def _min_prop(preds: np.ndarray) -> float:
            counts = np.bincount(preds)
            return float(counts.min() / len(preds)) if len(counts) else 0.0

        if criteria in ("aic", "bic") and (min_silhouette > 0.0 or min_cluster_prop > 0.0):
            try:
                # Evaluate silhouette for chosen and all tried models; fallback to best passing guardrails
                sil_scores: Dict[int, float] = {}
                passing: Dict[int, float] = {}
                X = features.values if hasattr(features, "values") else np.asarray(features)
                for n, m in models.items():
                    preds = m.predict(features)
                    if len(np.unique(preds)) < 2:
                        sil = 0.0
                    else:
                        from sklearn.metrics import silhouette_score

                        sil = float(silhouette_score(X, preds))
                    sil_scores[n] = sil
                    if sil >= min_silhouette and _min_prop(preds) >= min_cluster_prop:
                        passing[n] = sil

                if optimal_n not in passing:
                    # Fallback to best silhouette among passing
                    if passing:
                        optimal_n = max(passing.keys(), key=lambda k: passing[k])
                        chosen_model = models.get(optimal_n)
                        logger.info(
                            f"Guardrails adjusted optimal n_regimes to {optimal_n} based on silhouette"
                        )
            except Exception as e:
                logger.warning(f"Guardrails evaluation failed: {e}")

        self.optimization_config.optimal_regimes = optimal_n
        self.optimization_config.regime_scores = scores

        logger.info(f"Optimal regime count: {optimal_n} (criterion: {criteria})")

        return {
            "optimal_regimes": optimal_n,
            "scores": scores,
            "criterion": criteria,
            "best_model": chosen_model,
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Args:
            deep: If True, will return the parameters for this estimator
                and contained subobjects that are estimators.

        Returns:
            Dictionary of parameter names to values
        """
        if hasattr(self, "_init_params"):
            return self._init_params.copy()

        # Fallback - extract from current attributes
        params = {
            "n_regimes": self.n_regimes,
            "random_state": self.random_state,
            "fuzzy_matching": (
                self.fuzzy_config.enabled
                if hasattr(self, "fuzzy_config")
                else False
            ),
            "fuzzy_threshold": (
                self.fuzzy_config.threshold
                if hasattr(self, "fuzzy_config")
                else 0.7
            ),
            "auto_optimize_regimes": (
                self.optimization_config.auto_optimize
                if hasattr(self, "optimization_config")
                else False
            ),
            "min_regimes": (
                self.optimization_config.min_regimes
                if hasattr(self, "optimization_config")
                else 2
            ),
            "max_regimes": (
                self.optimization_config.max_regimes
                if hasattr(self, "optimization_config")
                else 9
            ),
        }

        # Add model-specific parameters if they exist
        for attr in ["covariance_type", "n_iter", "tol", "verbose"]:
            if hasattr(self, attr):
                params[attr] = getattr(self, attr)

        return params

    def set_params(self, **params) -> "BaseRegimeDetector":
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters

        Returns:
            Self
        """
        # Handle configuration object parameters
        config_mappings = {
            "fuzzy_matching": ("fuzzy_config", "enabled"),
            "fuzzy_threshold": ("fuzzy_config", "threshold"),
            "auto_optimize_regimes": ("optimization_config", "auto_optimize"),
            "min_regimes": ("optimization_config", "min_regimes"),
            "max_regimes": ("optimization_config", "max_regimes"),
        }

        for key, value in params.items():
            if key in config_mappings:
                # Set in configuration object
                config_obj, attr = config_mappings[key]
                if hasattr(self, config_obj):
                    setattr(getattr(self, config_obj), attr, value)
            elif hasattr(self, key):
                setattr(self, key, value)

            # Always store in init_params for future reference
            if not hasattr(self, "_init_params"):
                self._init_params = {}
            self._init_params[key] = value

        return self

    def save_model(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "pickle",
        compress: bool = False,
    ) -> None:
        """Save model to file.

        Args:
            filepath: Path to save the model
            metadata: Optional metadata to save with model
            format: Serialization format ('pickle', 'joblib', 'json')
            compress: Whether to compress the saved file

        Raises:
            OSError: If unable to save to the specified path
        """
        # Prepare model data
        model_data = {
            "model_class": self.__class__.__name__,
            "model_module": self.__class__.__module__,
            "model_state": self.__dict__.copy(),
            "parameters": self.get_params(),
            "is_fitted": self.is_fitted,
            "save_timestamp": datetime.now().isoformat(),
            "library_version": "1.0.0",
        }

        if metadata:
            model_data["metadata"] = metadata

        if hasattr(self, "model") and self.model is not None:
            model_data["wrapped_model"] = self.model

        # Convert for JSON if needed
        if format == "json":
            model_data = self._prepare_json_data(model_data)

        # Delegate to PersistenceHandler
        try:
            PersistenceHandler.save(
                model_data, Path(filepath), format, compress
            )
        except Exception as e:
            raise OSError(f"Failed to save model: {e}")

    @classmethod
    def _restore_model_from_data(
        cls, model_data: Dict
    ) -> "BaseRegimeDetector":
        """Restore model instance from loaded data.

        Args:
            model_data: Dictionary containing model state

        Returns:
            Restored model instance
        """
        # Handle backward compatibility
        if "model_class" not in model_data:
            logger.warning(
                "Loading model in old format, attempting migration..."
            )
            model_data = cls._migrate_old_format(model_data)

        # Verify class matches
        model_class_name = model_data.get("model_class")
        if model_class_name != cls.__name__:
            logger.warning(
                f"Model class mismatch: expected {cls.__name__}, got {model_class_name}"
            )

        # Create new instance
        params = model_data.get("parameters", {})
        model = cls(**params)

        # Restore state
        model_state = model_data.get("model_state", {})
        for key, value in model_state.items():
            if hasattr(model, key):
                setattr(model, key, value)

        # Restore wrapped model
        if "wrapped_model" in model_data:
            model.model = model_data["wrapped_model"]

        model.is_fitted = model_data.get("is_fitted", False)
        return model

    @classmethod
    def load_model(
        cls,
        filepath: Union[str, Path],
        format: Optional[str] = None,
        return_metadata: bool = False,
    ) -> Union["BaseRegimeDetector", Tuple["BaseRegimeDetector", Dict]]:
        """Load model from file.

        Args:
            filepath: Path to load the model from
            format: Serialization format (auto-detect if None)
            return_metadata: Whether to return metadata along with model

        Returns:
            Loaded model instance, optionally with metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or corrupted
        """
        try:
            # Load data using PersistenceHandler
            model_data = PersistenceHandler.load(Path(filepath), format)

            # Convert JSON data if needed
            if format == "json" or (
                format is None and (".json" in str(filepath))
            ):
                model_data = cls._restore_json_data(model_data)

            # Restore model from data
            model = cls._restore_model_from_data(model_data)

            logger.info(f"Model loaded from {filepath}")

            if return_metadata:
                metadata = model_data.get("metadata", {})
                metadata["save_timestamp"] = model_data.get("save_timestamp")
                metadata["library_version"] = model_data.get("library_version")
                return model, metadata
            else:
                return model

        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def _prepare_json_data(self, data: Dict) -> Dict:
        """Prepare data for JSON serialization.

        Converts numpy arrays and other non-JSON types.
        """
        json_data = {}

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = {
                    "_type": "ndarray",
                    "data": value.tolist(),
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                }
            elif isinstance(value, pd.DataFrame):
                json_data[key] = {
                    "_type": "dataframe",
                    "data": value.to_dict("records"),
                    "index": value.index.tolist(),
                    "columns": value.columns.tolist(),
                }
            elif isinstance(value, dict):
                json_data[key] = self._prepare_json_data(value)
            elif hasattr(value, "__dict__"):
                # Skip complex objects that can't be JSON serialized
                logger.warning(f"Skipping non-serializable object: {key}")
            else:
                json_data[key] = value

        return json_data

    @classmethod
    def _restore_json_data(cls, data: Dict) -> Dict:
        """Restore data from JSON format.

        Converts back to numpy arrays and other types.
        """
        restored_data = {}

        for key, value in data.items():
            if isinstance(value, dict) and "_type" in value:
                if value["_type"] == "ndarray":
                    restored_data[key] = np.array(
                        value["data"], dtype=value["dtype"]
                    ).reshape(value["shape"])
                elif value["_type"] == "dataframe":
                    restored_data[key] = pd.DataFrame(
                        value["data"],
                        index=value["index"],
                        columns=value["columns"],
                    )
            elif isinstance(value, dict):
                restored_data[key] = cls._restore_json_data(value)
            else:
                restored_data[key] = value

        return restored_data

    @classmethod
    def _migrate_old_format(cls, data: Dict) -> Dict:
        """Migrate old model format to new format.

        For backward compatibility.
        """
        # Try to infer model class and parameters
        migrated = {
            "model_class": cls.__name__,
            "model_module": cls.__module__,
            "model_state": data,
            "parameters": {},
            "is_fitted": True,  # Assume old models were fitted
            "save_timestamp": "unknown",
            "library_version": "0.1.0",
        }

        # Extract known parameters
        for param in ["n_regimes", "random_state", "covariance_type"]:
            if param in data:
                migrated["parameters"][param] = data[param]

        return migrated
