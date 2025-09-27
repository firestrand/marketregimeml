"""Validation strategies following Strategy pattern and SOLID principles."""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple, List
import numpy as np
from sklearn.model_selection import KFold

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationStrategy(ABC):
    """Abstract base class for validation strategies.

    Follows Strategy pattern - each concrete strategy implements
    a different validation approach.
    """

    @abstractmethod
    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits.

        Args:
            X: Feature matrix
            y: Optional target labels
            **kwargs: Additional parameters

        Yields:
            Tuples of (train_indices, test_indices)
        """

    @abstractmethod
    def get_n_splits(self) -> int:
        """Get number of splits.

        Returns:
            Number of splits
        """


class TimeSeriesSplitStrategy(ValidationStrategy):
    """Time series split validation strategy.

    Respects temporal order and prevents look-ahead bias.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        gap: int = 0,
        expanding: bool = True,
        train_size: Optional[int] = None,
    ):
        """Initialize time series split strategy.

        Args:
            n_splits: Number of splits
            test_size: Size of test set as fraction
            gap: Gap between train and test sets
            expanding: Whether to use expanding window
            train_size: Fixed training size (for non-expanding)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding
        self.train_size = train_size
        logger.debug(f"TimeSeriesSplitStrategy initialized with {n_splits} splits")

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series splits.

        Args:
            X: Feature matrix
            y: Optional target labels

        Yields:
            Train and test indices
        """
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)

        # Calculate split points to ensure we get exactly n_splits
        # We need space for n_splits test sets plus initial training data
        min_train_size = 10  # Minimum training samples

        # Calculate step size to distribute splits evenly
        total_test_space = self.n_splits * test_size
        available_space = n_samples - min_train_size - self.gap

        if available_space < total_test_space:
            # Not enough data, use smaller steps
            step_size = (
                n_samples - test_size - min_train_size - self.gap
            ) // self.n_splits
        else:
            step_size = (n_samples - test_size) // self.n_splits

        splits_generated = 0
        for i in range(self.n_splits):
            # Calculate test start position
            test_start = min_train_size + self.gap + (i * step_size)
            test_end = min(test_start + test_size, n_samples)

            # Ensure we have room for the test set
            if test_end > n_samples:
                break

            # Calculate train indices
            train_end = test_start - self.gap - 1

            if self.expanding:
                # Expanding window - use all data up to gap before test
                train_start = 0
            else:
                # Fixed window
                if self.train_size:
                    # Fixed size window - calculate start based on desired size
                    train_start = max(0, train_end - self.train_size + 1)
                    # If we can't get the full train_size, skip this split
                    if (
                        train_end - train_start + 1 < self.train_size
                        and train_start == 0
                    ):
                        # Not enough data for fixed window, adjust test_start or skip
                        if train_end + 1 < self.train_size:
                            continue  # Skip this split
                else:
                    # Use all available data before test
                    train_start = 0

            # Ensure we have enough training data
            if train_end < train_start or train_end - train_start < min_train_size - 1:
                continue

            train_idx = np.arange(train_start, train_end + 1)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx
            splits_generated += 1

            if splits_generated >= self.n_splits:
                break

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


class WalkForwardStrategy(ValidationStrategy):
    """Walk-forward validation strategy.

    Fixed-size rolling windows for both training and testing.
    """

    def __init__(
        self,
        train_periods: int = 252,
        test_periods: int = 63,
        step: int = 21,
        n_splits: Optional[int] = None,
        anchored: bool = False,
    ):
        """Initialize walk-forward strategy.

        Args:
            train_periods: Number of periods for training
            test_periods: Number of periods for testing
            step: Step size between windows
            n_splits: Maximum number of splits
            anchored: If True, training always starts from beginning
        """
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.step = step
        self.n_splits = n_splits
        self.anchored = anchored
        logger.debug(
            f"WalkForwardStrategy initialized with {train_periods}/{test_periods} train/test"
        )

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits.

        Args:
            X: Feature matrix
            y: Optional target labels

        Yields:
            Train and test indices
        """
        n_samples = len(X)
        splits_generated = 0

        # Start position for first window
        start_pos = 0

        while True:
            if self.anchored:
                # Anchored mode - training always starts from 0
                train_start = 0
                train_end = start_pos + self.train_periods - 1
            else:
                # Rolling mode
                train_start = start_pos
                train_end = start_pos + self.train_periods - 1

            test_start = train_end + 1
            test_end = test_start + self.test_periods - 1

            # Check if we have enough data
            if test_end >= n_samples:
                break

            train_idx = np.arange(train_start, train_end + 1)
            test_idx = np.arange(test_start, test_end + 1)

            yield train_idx, test_idx

            splits_generated += 1

            # Check if we've generated enough splits
            if self.n_splits and splits_generated >= self.n_splits:
                break

            # Move to next window
            start_pos += self.step

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits if self.n_splits else 5  # Default


class PurgedCVStrategy(ValidationStrategy):
    """Purged cross-validation strategy.

    Prevents information leakage in time series by purging samples
    near test set from training.
    """

    def __init__(
        self, n_splits: int = 5, purge_gap: int = 10, embargo_pct: float = 0.0
    ):
        """Initialize purged CV strategy.

        Args:
            n_splits: Number of splits
            purge_gap: Number of samples to purge around test set
            embargo_pct: Percentage of test set to embargo at end
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.base_cv = KFold(n_splits=n_splits, shuffle=False)
        logger.debug(f"PurgedCVStrategy initialized with gap={purge_gap}")

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged CV splits.

        Args:
            X: Feature matrix
            y: Optional target labels

        Yields:
            Train and test indices
        """
        for train_idx, test_idx in self.base_cv.split(X):
            # Apply purging - remove training samples near test set
            purged_train_idx = []

            for train_i in train_idx:
                # Check if this training sample is too close to any test sample
                min_distance = np.min(np.abs(test_idx - train_i))
                if min_distance > self.purge_gap:
                    purged_train_idx.append(train_i)

            # Apply embargo - remove end portion of test set
            if self.embargo_pct > 0:
                embargo_size = int(len(test_idx) * self.embargo_pct)
                if embargo_size > 0:
                    test_idx = test_idx[:-embargo_size]

            yield np.array(purged_train_idx), test_idx

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


class RegimeAwareStrategy(ValidationStrategy):
    """Regime-aware validation strategy.

    Ensures each split contains samples from all regimes and
    respects regime transitions.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_regime_samples: int = 10,
        stratify: bool = True,
    ):
        """Initialize regime-aware strategy.

        Args:
            n_splits: Number of splits
            min_regime_samples: Minimum samples per regime in training
            stratify: Whether to stratify by regime
        """
        self.n_splits = n_splits
        self.min_regime_samples = min_regime_samples
        self.stratify = stratify
        logger.debug(f"RegimeAwareStrategy initialized with stratify={stratify}")

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        regimes: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate regime-aware splits.

        Args:
            X: Feature matrix
            y: Optional target labels
            regimes: Regime labels for each sample

        Yields:
            Train and test indices
        """
        if regimes is None:
            yield from self._fallback_split(X)
            return

        # Find regime transition points
        transitions = self._find_regime_transitions(regimes, len(X))
        n_blocks = len(transitions) - 1

        # Check if we have enough blocks
        if n_blocks < self.n_splits:
            yield from self._insufficient_blocks_split(X, n_blocks)
            return

        # Generate regime-aware splits
        yield from self._generate_regime_splits(X, regimes, transitions, n_blocks)

    def _fallback_split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Fall back to regular k-fold when no regimes provided."""
        logger.warning("No regimes provided, using regular k-fold")
        base_cv = KFold(n_splits=self.n_splits, shuffle=False)
        yield from base_cv.split(X)

    def _find_regime_transitions(
        self, regimes: np.ndarray, n_samples: int
    ) -> List[int]:
        """Find indices where regime transitions occur."""
        transitions = [0]
        for i in range(1, n_samples):
            if regimes[i] != regimes[i - 1]:
                transitions.append(i)
        transitions.append(n_samples)
        return transitions

    def _insufficient_blocks_split(
        self, X: np.ndarray, n_blocks: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Handle case with insufficient regime blocks."""
        logger.warning(
            f"Not enough regime blocks ({n_blocks}) for {self.n_splits} splits"
        )
        base_cv = KFold(n_splits=min(self.n_splits, n_blocks), shuffle=False)
        yield from base_cv.split(X)

    def _generate_regime_splits(
        self,
        X: np.ndarray,
        regimes: np.ndarray,
        transitions: List[int],
        n_blocks: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate splits based on regime blocks."""
        blocks_per_fold = max(1, n_blocks // self.n_splits)

        for i in range(self.n_splits):
            train_blocks, test_blocks = self._partition_blocks(
                transitions, i, blocks_per_fold, n_blocks
            )

            if test_blocks and train_blocks:
                test_idx = np.concatenate(test_blocks)
                train_idx = np.concatenate(train_blocks)

                # Apply stratification check if needed
                if self._should_yield_split(train_idx, regimes):
                    yield train_idx, test_idx

    def _partition_blocks(
        self,
        transitions: List[int],
        fold_idx: int,
        blocks_per_fold: int,
        n_blocks: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Partition blocks into train and test sets."""
        test_start_block = fold_idx * blocks_per_fold % n_blocks
        test_end_block = min(test_start_block + blocks_per_fold, n_blocks)

        train_blocks = []
        test_blocks = []

        for j in range(len(transitions) - 1):
            block_indices = np.arange(transitions[j], transitions[j + 1])
            if test_start_block <= j < test_end_block:
                test_blocks.append(block_indices)
            else:
                train_blocks.append(block_indices)

        return train_blocks, test_blocks

    def _should_yield_split(self, train_idx: np.ndarray, regimes: np.ndarray) -> bool:
        """Check if split should be yielded based on stratification requirements."""
        if not self.stratify:
            return True

        train_regimes = regimes[train_idx]
        unique_regimes = np.unique(train_regimes)

        regime_counts = {r: np.sum(train_regimes == r) for r in unique_regimes}

        # Check if any regime has too few samples
        for count in regime_counts.values():
            if count > 0 and count < self.min_regime_samples:
                logger.debug(
                    f"Split has insufficient samples for regime (min={self.min_regime_samples})"
                )
                return False

        return True

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


class ValidationContext:
    """Context class for validation strategies.

    Follows Strategy pattern - allows switching validation strategies.
    """

    def __init__(self, strategy: ValidationStrategy):
        """Initialize validation context.

        Args:
            strategy: Initial validation strategy
        """
        self.strategy = strategy
        logger.debug(f"ValidationContext initialized with {type(strategy).__name__}")

    def set_strategy(self, strategy: ValidationStrategy):
        """Set new validation strategy.

        Args:
            strategy: New validation strategy
        """
        self.strategy = strategy
        logger.debug(f"Strategy changed to {type(strategy).__name__}")

    def execute_validation(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Execute validation using current strategy.

        Args:
            X: Feature matrix
            y: Optional target labels
            **kwargs: Additional parameters for strategy

        Yields:
            Train and test indices
        """
        return self.strategy.split(X, y, **kwargs)

    def get_n_splits(self) -> int:
        """Get number of splits from current strategy.

        Returns:
            Number of splits
        """
        return self.strategy.get_n_splits()
