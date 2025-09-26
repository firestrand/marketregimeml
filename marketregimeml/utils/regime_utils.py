"""Shared utilities for regime detection following DRY principle."""

from typing import Dict
import numpy as np


class RegimeReorderingUtils:
    """Shared utilities for regime reordering and labeling.

    Eliminates code duplication between HMM and GMM models.
    Follows DRY principle by centralizing reordering logic.
    """

    @staticmethod
    def reorder_regimes_by_feature_mean(
        regimes: np.ndarray,
        features: np.ndarray,
        n_regimes: int,
        feature_index: int = 0,
    ) -> np.ndarray:
        """Reorder regimes by mean of specified feature.

        Args:
            regimes: Original regime labels
            features: Feature matrix
            n_regimes: Number of regimes
            feature_index: Index of feature to use for ordering (default: 0)

        Returns:
            Reordered regime labels
        """
        if features.shape[1] <= feature_index:
            raise ValueError(
                f"Feature index {feature_index} out of bounds for {features.shape[1]} features"
            )

        # Calculate mean of specified feature for each regime
        regime_means = []
        for i in range(n_regimes):
            mask = regimes == i
            if mask.sum() > 0:
                regime_means.append(features[mask, feature_index].mean())
            else:
                regime_means.append(0)

        # Get sorting order (ascending)
        order = np.argsort(regime_means)

        # Create mapping from old to new labels
        mapping = {old: new for new, old in enumerate(order)}

        # Apply mapping to regime labels
        reordered_regimes = np.array([mapping[regime] for regime in regimes])

        return reordered_regimes

    @staticmethod
    def reorder_probabilities(
        probabilities: np.ndarray,
        regimes: np.ndarray,
        features: np.ndarray,
        n_regimes: int,
        feature_index: int = 0,
    ) -> np.ndarray:
        """Reorder probability matrix to match regime reordering.

        Args:
            probabilities: Original probability matrix (n_samples, n_regimes)
            regimes: Regime labels for reference
            features: Feature matrix
            n_regimes: Number of regimes
            feature_index: Index of feature to use for ordering

        Returns:
            Reordered probability matrix
        """
        if probabilities.shape[1] != n_regimes:
            raise ValueError(
                f"Probability matrix has {probabilities.shape[1]} columns, expected {n_regimes}"
            )

        # Calculate regime ordering based on feature means
        regime_means = []
        for i in range(n_regimes):
            mask = regimes == i
            if mask.sum() > 0:
                regime_means.append(features[mask, feature_index].mean())
            else:
                regime_means.append(0)

        # Get sorting order
        order = np.argsort(regime_means)

        # Reorder probability columns
        reordered_probabilities = probabilities[:, order]

        return reordered_probabilities

    @staticmethod
    def get_regime_mapping(
        regimes: np.ndarray,
        features: np.ndarray,
        n_regimes: int,
        feature_index: int = 0,
    ) -> Dict[int, int]:
        """Get mapping from original to reordered regime labels.

        Args:
            regimes: Original regime labels
            features: Feature matrix
            n_regimes: Number of regimes
            feature_index: Index of feature to use for ordering

        Returns:
            Dictionary mapping old labels to new labels
        """
        # Calculate regime means
        regime_means = []
        for i in range(n_regimes):
            mask = regimes == i
            if mask.sum() > 0:
                regime_means.append(features[mask, feature_index].mean())
            else:
                regime_means.append(0)

        # Get sorting order
        order = np.argsort(regime_means)

        # Create mapping
        mapping = {int(old): int(new) for new, old in enumerate(order)}

        return mapping

    @staticmethod
    def validate_regime_labels(regimes: np.ndarray, n_regimes: int) -> bool:
        """Validate regime labels are in expected range.

        Args:
            regimes: Regime labels to validate
            n_regimes: Expected number of regimes

        Returns:
            True if valid, False otherwise
        """
        if len(regimes) == 0:
            return True

        min_regime = regimes.min()
        max_regime = regimes.max()

        return bool(min_regime >= 0 and max_regime < n_regimes)


def smooth_regimes_majority(labels: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a simple majority filter to regime labels.

    Reduces rapid switching by replacing each label with the majority
    label within a centered window. Uses odd window size; if even, it
    is incremented by 1.

    Args:
        labels: Sequence of integer regime labels (1D array)
        window: Window size for majority vote (default 5)

    Returns:
        Smoothed regime labels (np.ndarray)
    """
    if window < 1:
        return labels
    if window % 2 == 0:
        window += 1
    n = len(labels)
    if n == 0 or window == 1:
        return labels
    half = window // 2
    out = labels.copy()
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        segment = labels[start:end]
        # Majority label; tie-break by center label to avoid bias
        vals, counts = np.unique(segment, return_counts=True)
        maj = vals[np.argmax(counts)]
        out[i] = maj
    return out
