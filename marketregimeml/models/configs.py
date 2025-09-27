"""Configuration classes for regime detection models.

Separated from base.py following Single Responsibility Principle.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class OptimizationConfig:
    """Configuration for regime count optimization."""

    enabled: bool = False
    min_regimes: int = 2
    max_regimes: int = 10
    criterion: str = "bic"  # 'aic' or 'bic'
    n_jobs: int = 1
    verbose: bool = False

    def validate(self) -> None:
        """Validate configuration."""
        if self.min_regimes < 2:
            raise ValueError("min_regimes must be at least 2")
        if self.max_regimes > 20:
            raise ValueError("max_regimes should not exceed 20")
        if self.min_regimes > self.max_regimes:
            raise ValueError("min_regimes must be <= max_regimes")
        if self.criterion not in ["aic", "bic"]:
            raise ValueError("criterion must be 'aic' or 'bic'")


@dataclass
class FuzzyConfig:
    """Configuration for fuzzy regime matching."""

    enabled: bool = False
    threshold: float = 0.7
    transition_penalty: float = 0.1
    min_confidence: float = 0.5

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate configuration."""
        if not 0.5 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.5 and 1.0")
        if not 0.0 <= self.transition_penalty <= 1.0:
            raise ValueError("transition_penalty must be between 0.0 and 1.0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")


@dataclass
class ModelMetadata:
    """Metadata for a fitted model."""

    fit_date: Optional[datetime] = None
    n_samples_train: Optional[int] = None
    n_features: Optional[int] = None
    feature_names: Optional[list] = None
    train_log_likelihood: Optional[float] = None
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    optimization_scores: Dict[int, float] = field(default_factory=dict)
    version: str = "1.0.0"