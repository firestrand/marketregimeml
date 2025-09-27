"""Enhanced evaluation metrics for regime detection quality assessment."""

from typing import Dict, Union, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class EnhancedRegimeMetrics:
    """
    Enhanced metrics for comprehensive regime detection evaluation.

    Provides advanced metrics for assessing regime quality including:
    - Flip rate and stability metrics
    - Persistence and duration analysis
    - Cross-regime correlation analysis
    - Composite quality indices
    """

    def __init__(self):
        """Initialize EnhancedRegimeMetrics."""
        self.metrics_cache = {}

    # ============ Flip Rate Metrics ============

    def calculate_flip_rate(self, regimes: np.ndarray) -> float:
        """
        Calculate overall regime flip rate.

        Args:
            regimes: Array of regime labels

        Returns:
            Flip rate (proportion of transitions)
        """
        if len(regimes) <= 1:
            return 0.0

        transitions = np.sum(np.diff(regimes) != 0)
        flip_rate = transitions / (len(regimes) - 1)

        return float(flip_rate)

    def calculate_flip_rate_by_regime(self, regimes: np.ndarray) -> Dict[int, float]:
        """
        Calculate flip rate for each regime.

        Args:
            regimes: Array of regime labels

        Returns:
            Dictionary mapping regime ID to flip rate
        """
        unique_regimes = np.unique(regimes)
        regime_flip_rates = {}

        for regime_id in unique_regimes:
            # Find positions where this regime occurs
            regime_mask = regimes == regime_id
            regime_positions = np.where(regime_mask)[0]

            if len(regime_positions) <= 1:
                regime_flip_rates[int(regime_id)] = 0.0
                continue

            # Count transitions out of this regime
            transitions = 0
            for i in range(len(regime_positions) - 1):
                pos = regime_positions[i]
                if pos < len(regimes) - 1 and regimes[pos + 1] != regime_id:
                    transitions += 1

            # Calculate flip rate for this regime
            regime_flip_rates[int(regime_id)] = transitions / len(regime_positions)

        return regime_flip_rates

    def calculate_windowed_flip_rate(
        self, regimes: np.ndarray, window_size: int = 20
    ) -> np.ndarray:
        """
        Calculate flip rate in rolling windows.

        Args:
            regimes: Array of regime labels
            window_size: Size of rolling window

        Returns:
            Array of windowed flip rates
        """
        n = len(regimes)
        flip_rates = np.full(n, np.nan)

        for i in range(window_size, n):
            window = regimes[i - window_size : i]
            flip_rates[i] = self.calculate_flip_rate(window)

        return flip_rates

    # ============ Persistence Metrics ============

    def calculate_persistence(self, regimes: np.ndarray) -> Dict[str, float]:
        """
        Calculate regime persistence metrics.

        Args:
            regimes: Array of regime labels

        Returns:
            Dictionary with persistence statistics
        """
        # Calculate regime durations
        durations = []
        current_regime = regimes[0]
        current_duration = 1

        for regime in regimes[1:]:
            if regime == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = regime
                current_duration = 1
        durations.append(current_duration)

        # Calculate statistics
        durations = np.array(durations)

        return {
            "mean_duration": float(np.mean(durations)),
            "median_duration": float(np.median(durations)),
            "max_duration": int(np.max(durations)),
            "min_duration": int(np.min(durations)),
            "std_duration": float(np.std(durations)),
            "persistence_score": float(np.mean(durations) / len(regimes)),
        }

    def calculate_persistence_by_regime(
        self, regimes: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate persistence metrics for each regime.

        Args:
            regimes: Array of regime labels

        Returns:
            Dictionary mapping regime ID to persistence metrics
        """
        unique_regimes = np.unique(regimes)
        regime_persistence = {}

        for regime_id in unique_regimes:
            # Find durations for this regime
            durations = []
            in_regime = False
            current_duration = 0

            for regime in regimes:
                if regime == regime_id:
                    if not in_regime:
                        in_regime = True
                        current_duration = 1
                    else:
                        current_duration += 1
                else:
                    if in_regime:
                        durations.append(current_duration)
                        in_regime = False
                        current_duration = 0

            if in_regime:
                durations.append(current_duration)

            if durations:
                regime_persistence[int(regime_id)] = {
                    "mean_duration": float(np.mean(durations)),
                    "max_duration": int(np.max(durations)),
                    "min_duration": int(np.min(durations)),
                    "occurrences": len(durations),
                }
            else:
                regime_persistence[int(regime_id)] = {
                    "mean_duration": 0.0,
                    "max_duration": 0,
                    "min_duration": 0,
                    "occurrences": 0,
                }

        return regime_persistence

    def calculate_persistence_from_transition(
        self, transition_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate expected persistence from transition matrix.

        Args:
            transition_matrix: Regime transition probability matrix

        Returns:
            Array of expected durations for each regime
        """
        n_regimes = len(transition_matrix)
        expected_durations = np.zeros(n_regimes)

        for i in range(n_regimes):
            # Expected duration = 1 / (1 - self-transition probability)
            self_prob = transition_matrix[i, i]
            if self_prob < 1.0:
                expected_durations[i] = 1.0 / (1.0 - self_prob)
            else:
                expected_durations[i] = np.inf

        return expected_durations

    # ============ Correlation Analysis ============

    def calculate_cross_regime_correlation(
        self, features: Union[pd.DataFrame, np.ndarray], regimes: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Calculate feature correlations within each regime.

        Args:
            features: Feature matrix
            regimes: Array of regime labels

        Returns:
            Dictionary mapping regime ID to correlation matrix
        """
        if isinstance(features, pd.DataFrame):
            features = features.values

        unique_regimes = np.unique(regimes)
        correlations = {}

        for regime_id in unique_regimes:
            regime_mask = regimes == regime_id
            regime_features = features[regime_mask]

            if len(regime_features) > 1:
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(regime_features.T)
                correlations[int(regime_id)] = corr_matrix
            else:
                # Return identity matrix if not enough data
                n_features = features.shape[1]
                correlations[int(regime_id)] = np.eye(n_features)

        return correlations

    def calculate_correlation_stability(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        regimes: np.ndarray,
        window_size: int = 30,
    ) -> Dict[str, Any]:
        """
        Calculate correlation stability within regimes.

        Args:
            features: Feature matrix
            regimes: Array of regime labels
            window_size: Window size for stability calculation

        Returns:
            Dictionary with stability metrics
        """
        if isinstance(features, pd.DataFrame):
            features = features.values

        unique_regimes = np.unique(regimes)
        regime_stabilities = {}

        for regime_id in unique_regimes:
            regime_mask = regimes == regime_id
            regime_features = features[regime_mask]

            if len(regime_features) < 2 * window_size:
                regime_stabilities[int(regime_id)] = 1.0
                continue

            # Calculate correlations in windows
            correlations = []
            for i in range(window_size, len(regime_features)):
                window = regime_features[i - window_size : i]
                if len(window) > 1 and window.shape[1] > 1:
                    corr = np.corrcoef(window.T)[0, 1]  # Use first two features
                    if not np.isnan(corr):
                        correlations.append(corr)

            if correlations:
                # Stability as inverse of correlation variance
                stability = 1.0 / (1.0 + np.var(correlations))
                regime_stabilities[int(regime_id)] = float(stability)
            else:
                regime_stabilities[int(regime_id)] = 0.0

        mean_stability = np.mean(list(regime_stabilities.values()))

        return {
            "mean_stability": float(mean_stability),
            "regime_stability": regime_stabilities,
        }

    # ============ Quality Index ============

    def regime_quality_index(
        self,
        features: np.ndarray,
        regimes: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate composite regime quality index.

        Args:
            features: Feature matrix
            regimes: Array of regime labels
            probabilities: Regime probability matrix

        Returns:
            Dictionary with quality score and components
        """
        components = {}

        # 1. Separation score (clustering quality)
        try:
            if len(np.unique(regimes)) > 1:
                silhouette = silhouette_score(features, regimes)
                components["separation_score"] = (
                    silhouette + 1
                ) / 2  # Normalize to [0, 1]
            else:
                components["separation_score"] = 0.0
        except Exception:
            components["separation_score"] = 0.5

        # 2. Stability score (low flip rate)
        flip_rate = self.calculate_flip_rate(regimes)
        components["stability_score"] = 1.0 - flip_rate

        # 3. Confidence score (probability purity)
        components["confidence_score"] = self.calculate_regime_purity(probabilities)

        # 4. Persistence score
        persistence = self.calculate_persistence(regimes)
        max_possible_duration = len(regimes)
        components["persistence_score"] = min(
            persistence["mean_duration"] / max_possible_duration * 10, 1.0
        )

        # 5. Consistency score
        components["consistency_score"] = self.calculate_temporal_consistency(regimes)

        # Calculate weighted overall score
        weights = {
            "separation_score": 0.25,
            "stability_score": 0.20,
            "confidence_score": 0.20,
            "persistence_score": 0.20,
            "consistency_score": 0.15,
        }

        overall_score = sum(
            components[key] * weights[key] for key in weights if key in components
        )

        return {
            "overall_score": float(overall_score * 100),  # Convert to 0-100 scale
            "components": components,
        }

    # ============ Additional Metrics ============

    def calculate_temporal_consistency(
        self, regimes: np.ndarray, window_size: int = 5
    ) -> float:
        """
        Calculate temporal consistency of regime assignments.

        Args:
            regimes: Array of regime labels
            window_size: Size of consistency window

        Returns:
            Consistency score between 0 and 1
        """
        if len(regimes) <= window_size:
            return 1.0

        consistencies = []
        for i in range(window_size, len(regimes)):
            window = regimes[i - window_size : i]
            # Consistency as proportion of same regime in window
            mode_regime = stats.mode(window, keepdims=True)[0][0]
            consistency = np.sum(window == mode_regime) / window_size
            consistencies.append(consistency)

        return float(np.mean(consistencies))

    def calculate_regime_entropy(self, regimes: np.ndarray) -> float:
        """
        Calculate entropy of regime distribution.

        Args:
            regimes: Array of regime labels

        Returns:
            Entropy value
        """
        unique, counts = np.unique(regimes, return_counts=True)
        probabilities = counts / len(regimes)

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

        return float(entropy)

    def calculate_regime_purity(self, probabilities: np.ndarray) -> float:
        """
        Calculate purity of regime probability assignments.

        Args:
            probabilities: Regime probability matrix

        Returns:
            Purity score between 0 and 1
        """
        # Purity as mean of maximum probabilities
        max_probas = np.max(probabilities, axis=1)
        purity = np.mean(max_probas)

        return float(purity)

    def calculate_regime_separation(
        self, features: np.ndarray, regimes: np.ndarray
    ) -> float:
        """
        Calculate separation between regimes in feature space.

        Args:
            features: Feature matrix
            regimes: Array of regime labels

        Returns:
            Separation score between 0 and 1
        """
        unique_regimes = np.unique(regimes)

        if len(unique_regimes) <= 1:
            return 0.0

        # Calculate mean distance between regime centroids
        centroids = []
        for regime_id in unique_regimes:
            regime_features = features[regimes == regime_id]
            if len(regime_features) > 0:
                centroids.append(np.mean(regime_features, axis=0))

        if len(centroids) < 2:
            return 0.0

        # Calculate pairwise distances
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                distances.append(dist)

        # Normalize by feature scale
        feature_scale = np.std(features, axis=0).mean()
        if feature_scale > 0:
            mean_distance = np.mean(distances) / feature_scale
            # Sigmoid transformation to [0, 1]
            separation = 2 / (1 + np.exp(-mean_distance / 2)) - 1
        else:
            separation = 0.0

        return float(separation)

    # ============ Reporting ============

    def generate_report(
        self,
        features: np.ndarray,
        regimes: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            features: Feature matrix
            regimes: Array of regime labels
            probabilities: Regime probability matrix

        Returns:
            Dictionary with complete evaluation report
        """
        report = {}

        # Summary statistics
        flip_rate = self.calculate_flip_rate(regimes)
        persistence = self.calculate_persistence(regimes)
        quality = self.regime_quality_index(features, regimes, probabilities)

        report["summary"] = {
            "n_samples": len(regimes),
            "n_regimes": len(np.unique(regimes)),
            "overall_quality": quality["overall_score"],
            "flip_rate": flip_rate,
            "mean_persistence": persistence["mean_duration"],
        }

        # Stability metrics
        report["stability_metrics"] = {
            "flip_rate": flip_rate,
            "flip_rate_by_regime": self.calculate_flip_rate_by_regime(regimes),
            "persistence": persistence,
            "persistence_by_regime": self.calculate_persistence_by_regime(regimes),
            "temporal_consistency": self.calculate_temporal_consistency(regimes),
        }

        # Quality metrics
        report["quality_metrics"] = quality

        # Regime statistics
        unique, counts = np.unique(regimes, return_counts=True)
        report["regime_statistics"] = {
            "regime_proportions": dict(
                zip(unique.tolist(), (counts / len(regimes)).tolist())
            ),
            "regime_entropy": self.calculate_regime_entropy(regimes),
            "regime_purity": self.calculate_regime_purity(probabilities),
            "regime_separation": self.calculate_regime_separation(features, regimes),
        }

        # Recommendations
        recommendations = []
        if flip_rate > 0.5:
            recommendations.append(
                "High flip rate detected - consider smoothing or different model"
            )
        if persistence["mean_duration"] < 10:
            recommendations.append("Low persistence - regimes may be too sensitive")
        if quality["overall_score"] < 50:
            recommendations.append("Low quality score - consider parameter tuning")

        report["recommendations"] = recommendations

        return report


# Create alias for backward compatibility
RegimeMetrics = EnhancedRegimeMetrics
