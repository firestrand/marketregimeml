"""Performance metrics for regime detection models."""

from typing import Dict, Optional, Union

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class RegimeMetrics:
    """Clustering and regime detection specific metrics."""

    @staticmethod
    def adjusted_rand_index(
        true_labels: np.ndarray, pred_labels: np.ndarray
    ) -> float:
        """Calculate Adjusted Rand Index.

        Measures agreement between two clusterings, adjusted for chance.
        Range: [-1, 1], where 1 is perfect agreement.

        Args:
            true_labels: True regime labels
            pred_labels: Predicted regime labels

        Returns:
            ARI score
        """
        return adjusted_rand_score(true_labels, pred_labels)

    @staticmethod
    def normalized_mutual_info(
        true_labels: np.ndarray, pred_labels: np.ndarray
    ) -> float:
        """Calculate Normalized Mutual Information.

        Measures mutual information between two clusterings, normalized.
        Range: [0, 1], where 1 is perfect agreement.

        Args:
            true_labels: True regime labels
            pred_labels: Predicted regime labels

        Returns:
            NMI score
        """
        return normalized_mutual_info_score(true_labels, pred_labels)

    @staticmethod
    def adjusted_mutual_info(
        true_labels: np.ndarray, pred_labels: np.ndarray
    ) -> float:
        """Calculate Adjusted Mutual Information.

        Measures mutual information adjusted for chance.

        Args:
            true_labels: True regime labels
            pred_labels: Predicted regime labels

        Returns:
            AMI score
        """
        return adjusted_mutual_info_score(true_labels, pred_labels)

    @staticmethod
    def silhouette_coefficient(
        features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate Silhouette Coefficient.

        Measures how similar samples are to their own cluster vs other clusters.
        Range: [-1, 1], where 1 is perfectly separated clusters.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Silhouette score
        """
        if len(np.unique(labels)) > 1:
            return silhouette_score(features, labels)
        return 0.0

    @staticmethod
    def davies_bouldin_index(
        features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate Davies-Bouldin Index.

        Measures average similarity between clusters. Lower is better.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            DB index
        """
        if len(np.unique(labels)) > 1:
            return davies_bouldin_score(features, labels)
        return np.inf

    @staticmethod
    def calinski_harabasz_index(
        features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate Calinski-Harabasz Index.

        Measures ratio of between-cluster to within-cluster variance.
        Higher is better.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            CH index
        """
        if len(np.unique(labels)) > 1:
            return calinski_harabasz_score(features, labels)
        return 0.0

    @staticmethod
    def regime_stability(labels: np.ndarray) -> Dict[str, float]:
        """Calculate regime stability metrics.

        Args:
            labels: Sequence of regime labels

        Returns:
            Dictionary with stability metrics
        """
        # Transition rate
        transitions = np.sum(np.diff(labels) != 0)
        transition_rate = (
            transitions / (len(labels) - 1) if len(labels) > 1 else 0
        )

        # Average regime duration
        durations = []
        current_regime = labels[0]
        current_duration = 1

        for i in range(1, len(labels)):
            if labels[i] == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = labels[i]
                current_duration = 1
        durations.append(current_duration)

        avg_duration = np.mean(durations)
        duration_std = np.std(durations)

        # Persistence (probability of staying in same regime)
        n_regimes = len(np.unique(labels))
        persistence = 1 - transition_rate

        return {
            "transition_rate": transition_rate,
            "avg_duration": avg_duration,
            "duration_std": duration_std,
            "persistence": persistence,
            "n_transitions": transitions,
            "n_regimes_observed": n_regimes,
        }

    @staticmethod
    def regime_distribution(
        labels: np.ndarray,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Calculate regime distribution statistics.

        Args:
            labels: Regime labels

        Returns:
            Dictionary with distribution metrics
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        proportions = counts / len(labels)

        # Entropy of regime distribution
        entropy = -np.sum(proportions * np.log(proportions + 1e-10))
        max_entropy = np.log(len(unique_labels))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Gini coefficient (inequality of regime distribution)
        sorted_props = np.sort(proportions)
        n = len(sorted_props)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_props)) / (
            n * np.sum(sorted_props)
        ) - (n + 1) / n

        return {
            "regime_counts": counts,
            "regime_proportions": proportions,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "gini_coefficient": gini,
            "most_frequent_regime": unique_labels[np.argmax(counts)],
            "most_frequent_proportion": np.max(proportions),
        }

    @staticmethod
    def confidence_metrics(probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate confidence and uncertainty metrics.

        Args:
            probabilities: Regime probability matrix (n_samples, n_regimes)

        Returns:
            Dictionary with confidence metrics
        """
        # Average confidence (max probability)
        max_probs = np.max(probabilities, axis=1)
        avg_confidence = np.mean(max_probs)
        confidence_std = np.std(max_probs)

        # Entropy-based uncertainty
        entropy = -np.sum(
            probabilities * np.log(probabilities + 1e-10), axis=1
        )
        avg_entropy = np.mean(entropy)

        # Normalized entropy
        n_regimes = probabilities.shape[1]
        max_entropy = np.log(n_regimes)
        normalized_entropy = (
            avg_entropy / max_entropy if max_entropy > 0 else 0
        )

        # Margin (difference between top two probabilities)
        sorted_probs = np.sort(probabilities, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        avg_margin = np.mean(margins)

        return {
            "avg_confidence": avg_confidence,
            "confidence_std": confidence_std,
            "avg_entropy": avg_entropy,
            "normalized_entropy": normalized_entropy,
            "avg_margin": avg_margin,
            "min_confidence": np.min(max_probs),
            "max_confidence": np.max(max_probs),
        }

    @staticmethod
    def temporal_consistency_metrics(
        predictions: np.ndarray, window_size: int = 10
    ) -> Dict[str, float]:
        """Calculate temporal consistency metrics for regime predictions.

        Args:
            predictions: Time series of regime predictions
            window_size: Window size for local consistency calculation

        Returns:
            Dictionary with temporal consistency metrics
        """
        n_samples = len(predictions)

        # Local temporal consistency
        local_consistencies = []
        for i in range(window_size, n_samples - window_size):
            window = predictions[
                i - window_size // 2 : i + window_size // 2 + 1
            ]
            # Consistency = fraction of predictions that match the center prediction
            center_pred = predictions[i]
            consistency = np.mean(window == center_pred)
            local_consistencies.append(consistency)

        avg_local_consistency = (
            np.mean(local_consistencies) if local_consistencies else 0
        )

        # Regime run length statistics
        run_lengths = []
        current_regime = predictions[0]
        current_length = 1

        for i in range(1, len(predictions)):
            if predictions[i] == current_regime:
                current_length += 1
            else:
                run_lengths.append(current_length)
                current_regime = predictions[i]
                current_length = 1
        run_lengths.append(current_length)

        # Temporal autocorrelation
        if len(predictions) > 1:
            # Convert to numeric for autocorrelation
            numeric_preds = predictions.astype(float)
            autocorr_lag1 = np.corrcoef(numeric_preds[:-1], numeric_preds[1:])[
                0, 1
            ]
            autocorr_lag1 = 0 if np.isnan(autocorr_lag1) else autocorr_lag1
        else:
            autocorr_lag1 = 0

        return {
            "avg_local_consistency": avg_local_consistency,
            "local_consistency_std": (
                np.std(local_consistencies) if local_consistencies else 0
            ),
            "avg_run_length": np.mean(run_lengths),
            "run_length_std": np.std(run_lengths),
            "min_run_length": np.min(run_lengths),
            "max_run_length": np.max(run_lengths),
            "autocorr_lag1": autocorr_lag1,
        }

    @staticmethod
    def regime_quality_index(
        features: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate composite Regime Quality Index (RQI).

        Combines multiple metrics into a single quality score.

        Args:
            features: Feature matrix
            predictions: Regime predictions
            probabilities: Regime probabilities
            returns: Optional returns series for financial relevance

        Returns:
            Regime Quality Index (0-100, higher is better)
        """
        # Clustering quality (weight: 30%)
        silhouette = RegimeMetrics.silhouette_coefficient(
            features, predictions
        )
        silhouette_score = max(0, (silhouette + 1) / 2)  # Normalize to 0-1

        # Prediction confidence (weight: 20%)
        confidence_metrics = RegimeMetrics.confidence_metrics(probabilities)
        confidence_score = confidence_metrics["avg_confidence"]

        # Temporal consistency (weight: 25%)
        temporal_metrics = RegimeMetrics.temporal_consistency_metrics(
            predictions
        )
        temporal_score = temporal_metrics["avg_local_consistency"]

        # Regime balance (weight: 15%)
        distribution = RegimeMetrics.regime_distribution(predictions)
        balance_score = distribution["normalized_entropy"]

        # Financial relevance (weight: 10% if available)
        if returns is not None:
            # Calculate regime separation in returns
            regime_return_means = []
            for regime in np.unique(predictions):
                mask = predictions == regime
                if mask.sum() > 0:
                    regime_return_means.append(np.mean(returns[mask]))

            if len(regime_return_means) > 1:
                return_separation = np.std(regime_return_means)
                # Normalize by overall return volatility
                overall_vol = np.std(returns)
                financial_score = min(
                    1.0, return_separation / (overall_vol + 1e-8)
                )
            else:
                financial_score = 0

            # Weighted combination with financial relevance
            rqi = (
                0.30 * silhouette_score
                + 0.20 * confidence_score
                + 0.25 * temporal_score
                + 0.15 * balance_score
                + 0.10 * financial_score
            )
        else:
            # Weighted combination without financial relevance
            rqi = (
                0.35 * silhouette_score
                + 0.25 * confidence_score
                + 0.25 * temporal_score
                + 0.15 * balance_score
            )

        return rqi * 100  # Scale to 0-100
