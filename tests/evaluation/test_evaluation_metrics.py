"""Tests for evaluation metrics."""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from marketregimeml.evaluation import (
    regime_accuracy,
    regime_transition_matrix,
    regime_duration_stats,
    regime_stability,
    adjusted_rand_index,
    normalized_mutual_info,
    silhouette_score_regimes,
    davies_bouldin_score_regimes,
    calinski_harabasz_score_regimes,
    regime_separation_score,
)


class TestEvaluationMetrics:
    """Test evaluation metrics for regime detection."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions and true labels."""
        # True regimes with clear transitions
        true_regimes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0])

        # Predictions with some errors
        predictions = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0])

        return predictions, true_regimes

    @pytest.fixture
    def sample_features(self):
        """Create sample features for clustering metrics."""
        np.random.seed(42)

        # Create features with 3 clear clusters
        features = []
        labels = []

        # Cluster 0
        features.extend(np.random.normal(0, 0.5, (20, 3)))
        labels.extend([0] * 20)

        # Cluster 1
        features.extend(np.random.normal(3, 0.5, (20, 3)))
        labels.extend([1] * 20)

        # Cluster 2
        features.extend(np.random.normal(-3, 0.5, (20, 3)))
        labels.extend([2] * 20)

        return np.vstack(features), np.array(labels)

    def test_regime_accuracy(self, sample_predictions):
        """Test regime accuracy calculation."""
        predictions, true_regimes = sample_predictions

        acc = regime_accuracy(true_regimes, predictions)

        assert isinstance(acc, float)
        assert 0 <= acc <= 1

        # Perfect predictions
        perfect_acc = regime_accuracy(true_regimes, true_regimes)
        assert perfect_acc == 1.0

        # Random predictions
        random_pred = np.random.randint(0, 3, len(true_regimes))
        random_acc = regime_accuracy(true_regimes, random_pred)
        assert 0 <= random_acc <= 1

    def test_regime_transition_matrix(self, sample_predictions):
        """Test regime transition matrix calculation."""
        predictions, _ = sample_predictions

        trans_matrix = regime_transition_matrix(predictions)

        assert isinstance(trans_matrix, np.ndarray)
        assert trans_matrix.shape == (3, 3)

        # Rows should sum to 1 (probabilities)
        row_sums = trans_matrix.sum(axis=1)
        # Only check non-zero rows
        for i, row_sum in enumerate(row_sums):
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 0.01

        # Diagonal should have high values for stable regimes
        assert trans_matrix[0, 0] > 0  # Some stability in regime 0

    def test_regime_duration_stats(self, sample_predictions):
        """Test regime duration statistics."""
        predictions, _ = sample_predictions

        duration_stats = regime_duration_stats(predictions)

        assert isinstance(duration_stats, dict)

        # Check structure for each regime
        for regime_id, stats in duration_stats.items():
            assert "mean_duration" in stats
            assert "std_duration" in stats
            assert "min_duration" in stats
            assert "max_duration" in stats
            assert "count" in stats

            # All values should be positive
            assert stats["mean_duration"] > 0
            assert stats["min_duration"] > 0
            assert stats["max_duration"] >= stats["min_duration"]
            assert stats["count"] > 0

    def test_regime_stability(self, sample_predictions):
        """Test regime stability score."""
        predictions, _ = sample_predictions

        stability = regime_stability(predictions)

        assert isinstance(stability, float)
        assert 0 <= stability <= 1

        # Constant regime should have stability = 1
        constant = np.ones(100, dtype=int)
        assert regime_stability(constant) == 1.0

        # Alternating regime should have low stability
        alternating = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        assert regime_stability(alternating) < 0.5

    def test_adjusted_rand_index(self, sample_predictions):
        """Test adjusted Rand index."""
        predictions, true_regimes = sample_predictions

        ari = adjusted_rand_index(true_regimes, predictions)

        assert isinstance(ari, float)
        assert -1 <= ari <= 1

        # Perfect match
        perfect_ari = adjusted_rand_index(true_regimes, true_regimes)
        assert perfect_ari == 1.0

        # Random should be close to 0
        random_pred = np.random.randint(0, 3, len(true_regimes))
        random_ari = adjusted_rand_index(true_regimes, random_pred)
        assert -0.5 < random_ari < 0.5

    def test_normalized_mutual_info(self, sample_predictions):
        """Test normalized mutual information."""
        predictions, true_regimes = sample_predictions

        nmi = normalized_mutual_info(true_regimes, predictions)

        assert isinstance(nmi, float)
        assert 0 <= nmi <= 1

        # Perfect match
        perfect_nmi = normalized_mutual_info(true_regimes, true_regimes)
        assert perfect_nmi == 1.0

        # Random should be low
        random_pred = np.random.randint(0, 3, len(true_regimes))
        random_nmi = normalized_mutual_info(true_regimes, random_pred)
        assert random_nmi < 0.5

    def test_silhouette_score(self, sample_features):
        """Test silhouette score for regimes."""
        features, labels = sample_features

        score = silhouette_score_regimes(features, labels)

        assert isinstance(score, float)
        assert -1 <= score <= 1

        # Good clustering should have positive score
        assert score > 0

        # Perfect separation would have score close to 1
        assert score > 0.3  # Reasonable threshold for good separation

    def test_davies_bouldin_score(self, sample_features):
        """Test Davies-Bouldin score."""
        features, labels = sample_features

        score = davies_bouldin_score_regimes(features, labels)

        assert isinstance(score, float)
        assert score >= 0

        # Lower is better for Davies-Bouldin
        # Good clustering should have low score
        assert score < 2.0

    def test_calinski_harabasz_score(self, sample_features):
        """Test Calinski-Harabasz score."""
        features, labels = sample_features

        score = calinski_harabasz_score_regimes(features, labels)

        assert isinstance(score, float)
        assert score > 0

        # Higher is better for Calinski-Harabasz
        # Good clustering should have high score
        assert score > 50

    def test_regime_separation_score(self, sample_features):
        """Test regime separation score."""
        features, labels = sample_features

        score = regime_separation_score(features, labels)

        assert isinstance(score, float)
        assert score >= 0

        # Good separation should have high score
        assert score > 1.0

    def test_edge_cases(self):
        """Test edge cases for metrics."""
        # Single regime
        single_regime = np.zeros(100, dtype=int)

        # Should handle single regime gracefully
        stability = regime_stability(single_regime)
        assert stability == 1.0

        trans_matrix = regime_transition_matrix(single_regime)
        assert trans_matrix.shape[0] >= 1

        # Empty arrays
        empty = np.array([])
        with pytest.raises((ValueError, IndexError)):
            regime_accuracy(empty, empty)

        # Mismatched lengths
        pred1 = np.array([0, 1, 2])
        pred2 = np.array([0, 1])
        with pytest.raises(ValueError):
            regime_accuracy(pred1, pred2)

    def test_duration_stats_detailed(self):
        """Test detailed duration statistics."""
        # Create regimes with known durations
        regimes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0])

        stats = regime_duration_stats(regimes)

        # Regime 0: appears twice (3 steps, 2 steps)
        assert stats[0]["count"] == 2
        assert stats[0]["mean_duration"] == 2.5
        assert stats[0]["min_duration"] == 2
        assert stats[0]["max_duration"] == 3

        # Regime 1: appears once (2 steps)
        assert stats[1]["count"] == 1
        assert stats[1]["mean_duration"] == 2

        # Regime 2: appears once (4 steps)
        assert stats[2]["count"] == 1
        assert stats[2]["mean_duration"] == 4

    def test_transition_matrix_detailed(self):
        """Test transition matrix calculation details."""
        # Known transition pattern
        regimes = np.array([0, 0, 1, 1, 2, 2, 0])

        trans_matrix = regime_transition_matrix(regimes)

        # From regime 0: one self-transition (0->0), one to 1 (0->1), one from 2 (2->0)
        assert trans_matrix[0, 0] == 0.5  # P(0->0) = 1/2
        assert trans_matrix[0, 1] == 0.5  # P(0->1) = 1/2

        # From regime 1: one self-transition (1->1), one to 2 (1->2)
        assert trans_matrix[1, 1] == 0.5  # P(1->1) = 1/2
        assert trans_matrix[1, 2] == 0.5  # P(1->2) = 1/2

        # From regime 2: one self-transition (2->2), one to 0 (2->0)
        assert trans_matrix[2, 2] == 0.5  # P(2->2) = 1/2
        assert trans_matrix[2, 0] == 0.5  # P(2->0) = 1/2
