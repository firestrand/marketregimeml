"""Standalone metric functions for regime detection evaluation."""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score as sklearn_ari,
    normalized_mutual_info_score as sklearn_nmi,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def regime_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Calculate regime prediction accuracy.

    Args:
        true_labels: True regime labels
        pred_labels: Predicted regime labels

    Returns:
        Accuracy score between 0 and 1
    """
    if len(true_labels) == 0 or len(pred_labels) == 0:
        raise ValueError("Input arrays cannot be empty")
    if len(true_labels) != len(pred_labels):
        raise ValueError("Input arrays must have the same length")
    return accuracy_score(true_labels, pred_labels)


def regime_transition_matrix(regimes: np.ndarray) -> np.ndarray:
    """Calculate regime transition probability matrix.

    Args:
        regimes: Sequence of regime labels

    Returns:
        Transition probability matrix
    """
    n_regimes = len(np.unique(regimes))
    trans_matrix = np.zeros((n_regimes, n_regimes))

    for i in range(len(regimes) - 1):
        from_regime = regimes[i]
        to_regime = regimes[i + 1]
        trans_matrix[from_regime, to_regime] += 1

    # Normalize rows to get probabilities
    row_sums = trans_matrix.sum(axis=1)
    for i in range(n_regimes):
        if row_sums[i] > 0:
            trans_matrix[i] = trans_matrix[i] / row_sums[i]

    return trans_matrix


def regime_duration_stats(regimes: np.ndarray) -> Dict[int, Dict[str, float]]:
    """Calculate duration statistics for each regime.

    Args:
        regimes: Sequence of regime labels

    Returns:
        Dictionary with duration statistics per regime
    """
    duration_stats = {}

    # Track durations for each regime
    regime_durations = {}
    current_regime = regimes[0]
    current_duration = 1

    for i in range(1, len(regimes)):
        if regimes[i] == current_regime:
            current_duration += 1
        else:
            # Record duration
            if current_regime not in regime_durations:
                regime_durations[current_regime] = []
            regime_durations[current_regime].append(current_duration)

            # Start new regime
            current_regime = regimes[i]
            current_duration = 1

    # Don't forget the last regime
    if current_regime not in regime_durations:
        regime_durations[current_regime] = []
    regime_durations[current_regime].append(current_duration)

    # Calculate statistics
    for regime, durations in regime_durations.items():
        durations = np.array(durations)
        duration_stats[regime] = {
            "mean_duration": float(np.mean(durations)),
            "std_duration": (
                float(np.std(durations)) if len(durations) > 1 else 0.0
            ),
            "min_duration": float(np.min(durations)),
            "max_duration": float(np.max(durations)),
            "count": len(durations),
        }

    return duration_stats


def regime_stability(
    regimes: np.ndarray, window: Optional[int] = None
) -> float:
    """Calculate regime stability score.

    Stability is measured as 1 - (number of transitions / possible transitions).
    Higher values indicate more stable regimes.

    Args:
        regimes: Sequence of regime labels
        window: Optional window for local stability

    Returns:
        Stability score between 0 and 1
    """
    if len(regimes) <= 1:
        return 1.0

    transitions = np.sum(regimes[1:] != regimes[:-1])
    possible_transitions = len(regimes) - 1

    stability = 1.0 - (transitions / possible_transitions)

    return stability


def adjusted_rand_index(
    true_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """Calculate Adjusted Rand Index.

    Args:
        true_labels: True regime labels
        pred_labels: Predicted regime labels

    Returns:
        ARI score between -1 and 1
    """
    return sklearn_ari(true_labels, pred_labels)


def normalized_mutual_info(
    true_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """Calculate Normalized Mutual Information.

    Args:
        true_labels: True regime labels
        pred_labels: Predicted regime labels

    Returns:
        NMI score between 0 and 1
    """
    return sklearn_nmi(true_labels, pred_labels)


def silhouette_score_regimes(
    features: np.ndarray, labels: np.ndarray
) -> float:
    """Calculate silhouette score for regime clustering.

    Args:
        features: Feature matrix
        labels: Regime labels

    Returns:
        Silhouette score between -1 and 1
    """
    if len(np.unique(labels)) < 2:
        return 0.0

    return silhouette_score(features, labels)


def davies_bouldin_score_regimes(
    features: np.ndarray, labels: np.ndarray
) -> float:
    """Calculate Davies-Bouldin score for regime clustering.

    Lower values indicate better clustering.

    Args:
        features: Feature matrix
        labels: Regime labels

    Returns:
        Davies-Bouldin score (lower is better)
    """
    if len(np.unique(labels)) < 2:
        return float("inf")

    return davies_bouldin_score(features, labels)


def calinski_harabasz_score_regimes(
    features: np.ndarray, labels: np.ndarray
) -> float:
    """Calculate Calinski-Harabasz score for regime clustering.

    Higher values indicate better clustering.

    Args:
        features: Feature matrix
        labels: Regime labels

    Returns:
        Calinski-Harabasz score (higher is better)
    """
    if len(np.unique(labels)) < 2:
        return 0.0

    return calinski_harabasz_score(features, labels)


def regime_separation_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Calculate regime separation score.

    Measures how well separated the regimes are in feature space.

    Args:
        features: Feature matrix
        labels: Regime labels

    Returns:
        Separation score (higher is better)
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Calculate centroid for each regime
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroid = features[mask].mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Calculate inter-cluster distances
    inter_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_distances.append(dist)

    # Calculate intra-cluster distances
    intra_distances = []
    for label in unique_labels:
        mask = labels == label
        cluster_features = features[mask]
        if len(cluster_features) > 1:
            centroid = centroids[label]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            intra_distances.append(distances.mean())

    if not intra_distances:
        return 0.0

    # Separation score is ratio of average inter-cluster to intra-cluster distance
    avg_inter = np.mean(inter_distances) if inter_distances else 0.0
    avg_intra = np.mean(intra_distances)

    if avg_intra == 0:
        return float("inf") if avg_inter > 0 else 0.0

    return avg_inter / avg_intra


def regime_purity_score(
    true_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """Calculate purity score for regime predictions.

    Args:
        true_labels: True regime labels
        pred_labels: Predicted regime labels

    Returns:
        Purity score between 0 and 1
    """
    if len(true_labels) == 0:
        return 0.0

    # Create confusion matrix
    unique_pred = np.unique(pred_labels)

    total_correct = 0
    for pred_label in unique_pred:
        mask = pred_labels == pred_label
        if mask.sum() > 0:
            true_in_cluster = true_labels[mask]
            # Count most common true label in this predicted cluster
            most_common_count = np.bincount(true_in_cluster).max()
            total_correct += most_common_count

    return total_correct / len(true_labels)


def regime_consistency_score(
    regimes: np.ndarray, min_duration: int = 5
) -> float:
    """Calculate regime consistency score.

    Measures how consistent regime assignments are (avoiding rapid switching).

    Args:
        regimes: Sequence of regime labels
        min_duration: Minimum duration for a consistent regime

    Returns:
        Consistency score between 0 and 1
    """
    if len(regimes) == 0:
        return 0.0

    # Get duration statistics
    duration_stats = regime_duration_stats(regimes)

    # Calculate percentage of regimes that last at least min_duration
    total_periods = 0
    consistent_periods = 0

    for regime_stats in duration_stats.values():
        mean_duration = regime_stats["mean_duration"]
        count = regime_stats["count"]

        regime_periods = mean_duration * count
        total_periods += regime_periods

        if mean_duration >= min_duration:
            consistent_periods += regime_periods

    if total_periods == 0:
        return 0.0

    return consistent_periods / total_periods


def regime_prediction_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    features: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate comprehensive regime prediction metrics.

    Args:
        true_labels: True regime labels
        pred_labels: Predicted regime labels
        features: Optional feature matrix for clustering metrics

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": regime_accuracy(true_labels, pred_labels),
        "ari": adjusted_rand_index(true_labels, pred_labels),
        "nmi": normalized_mutual_info(true_labels, pred_labels),
        "purity": regime_purity_score(true_labels, pred_labels),
        "stability": regime_stability(pred_labels),
        "consistency": regime_consistency_score(pred_labels),
    }

    # Add clustering metrics if features provided
    if features is not None and len(np.unique(pred_labels)) > 1:
        metrics.update(
            {
                "silhouette": silhouette_score_regimes(features, pred_labels),
                "davies_bouldin": davies_bouldin_score_regimes(
                    features, pred_labels
                ),
                "calinski_harabasz": calinski_harabasz_score_regimes(
                    features, pred_labels
                ),
                "separation": regime_separation_score(features, pred_labels),
            }
        )

    return metrics
