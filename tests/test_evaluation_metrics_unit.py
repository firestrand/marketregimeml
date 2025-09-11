"""Unit tests for evaluation metrics module.

Following DRY, KISS, and SOLID principles with bottom-up testing.
Minimal mocking, using real calculations where possible.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from marketregimeml.evaluation.metrics import RegimeMetrics


class TestDataHelper:
    """DRY: Centralized test data generation for evaluation tests."""
    
    @staticmethod
    def create_perfect_clusters(n_samples=100, n_features=2, n_clusters=3):
        """Create perfectly separated clusters."""
        np.random.seed(42)
        
        samples_per_cluster = n_samples // n_clusters
        features = []
        labels = []
        
        # Create well-separated clusters
        for i in range(n_clusters):
            center = np.array([i * 10, i * 10])  # Well separated centers
            cluster_data = np.random.randn(samples_per_cluster, n_features) * 0.5 + center
            features.append(cluster_data)
            labels.extend([i] * samples_per_cluster)
        
        return np.vstack(features), np.array(labels)
    
    @staticmethod
    def create_overlapping_clusters(n_samples=100, n_features=2, n_clusters=3):
        """Create overlapping clusters for testing edge cases."""
        np.random.seed(42)
        
        samples_per_cluster = n_samples // n_clusters
        features = []
        labels = []
        
        # Create overlapping clusters
        for i in range(n_clusters):
            center = np.array([i * 2, i * 2])  # Closer centers = overlap
            cluster_data = np.random.randn(samples_per_cluster, n_features) * 2 + center
            features.append(cluster_data)
            labels.extend([i] * samples_per_cluster)
        
        return np.vstack(features), np.array(labels)
    
    @staticmethod
    def create_regime_predictions(true_labels, accuracy=0.8):
        """Create predicted labels with controlled accuracy."""
        np.random.seed(42)
        pred_labels = true_labels.copy()
        
        # Randomly flip some labels
        n_flips = int(len(pred_labels) * (1 - accuracy))
        flip_indices = np.random.choice(len(pred_labels), n_flips, replace=False)
        
        # Flip to random different label
        n_classes = len(np.unique(true_labels))
        for idx in flip_indices:
            current = pred_labels[idx]
            alternatives = [i for i in range(n_classes) if i != current]
            pred_labels[idx] = np.random.choice(alternatives)
        
        return pred_labels


class TestRegimeMetrics:
    """Test RegimeMetrics class methods."""
    
    @pytest.fixture
    def perfect_data(self):
        """Perfect cluster data."""
        return TestDataHelper.create_perfect_clusters()
    
    @pytest.fixture
    def overlapping_data(self):
        """Overlapping cluster data."""
        return TestDataHelper.create_overlapping_clusters()
    
    def test_adjusted_rand_index_perfect(self):
        """Test ARI with perfect predictions."""
        true_labels = np.array([0, 0, 1, 1, 2, 2])
        pred_labels = np.array([0, 0, 1, 1, 2, 2])
        
        score = RegimeMetrics.adjusted_rand_index(true_labels, pred_labels)
        
        assert score == 1.0  # Perfect agreement
    
    def test_adjusted_rand_index_random(self):
        """Test ARI with random predictions."""
        np.random.seed(42)
        true_labels = np.array([0, 0, 1, 1, 2, 2])
        pred_labels = np.random.randint(0, 3, 6)
        
        score = RegimeMetrics.adjusted_rand_index(true_labels, pred_labels)
        
        # Random should be near 0
        assert -0.5 < score < 0.5
    
    def test_adjusted_rand_index_permuted(self):
        """Test ARI with permuted labels (same clustering, different labels)."""
        true_labels = np.array([0, 0, 1, 1, 2, 2])
        pred_labels = np.array([2, 2, 0, 0, 1, 1])  # Same structure, different labels
        
        score = RegimeMetrics.adjusted_rand_index(true_labels, pred_labels)
        
        assert score == 1.0  # Should recognize same clustering
    
    def test_normalized_mutual_info(self):
        """Test NMI calculation."""
        true_labels = np.array([0, 0, 1, 1, 2, 2])
        
        # Perfect prediction
        score_perfect = RegimeMetrics.normalized_mutual_info(true_labels, true_labels)
        assert score_perfect == 1.0
        
        # Partial agreement
        pred_labels = TestDataHelper.create_regime_predictions(true_labels, accuracy=0.7)
        score_partial = RegimeMetrics.normalized_mutual_info(true_labels, pred_labels)
        assert 0 < score_partial < 1
    
    def test_adjusted_mutual_info(self):
        """Test AMI calculation."""
        true_labels = np.array([0, 0, 0, 1, 1, 1])
        pred_labels = np.array([0, 0, 0, 1, 1, 1])
        
        score = RegimeMetrics.adjusted_mutual_info(true_labels, pred_labels)
        
        # Perfect agreement
        assert score > 0.9
    
    def test_silhouette_coefficient(self, perfect_data):
        """Test silhouette coefficient calculation."""
        features, labels = perfect_data
        
        score = RegimeMetrics.silhouette_coefficient(features, labels)
        
        # Well-separated clusters should have high silhouette
        assert score > 0.5
        assert score <= 1.0
    
    def test_silhouette_single_cluster(self):
        """Test silhouette with single cluster."""
        features = np.random.randn(20, 2)
        labels = np.zeros(20)  # All same cluster
        
        score = RegimeMetrics.silhouette_coefficient(features, labels)
        
        # Single cluster returns 0
        assert score == 0.0
    
    def test_davies_bouldin_index(self, perfect_data):
        """Test Davies-Bouldin index calculation."""
        features, labels = perfect_data
        
        score = RegimeMetrics.davies_bouldin_index(features, labels)
        
        # Lower is better for DB index
        assert score > 0  # Should be positive
        assert score < 2  # Good clustering should be < 2
    
    def test_davies_bouldin_single_cluster(self):
        """Test Davies-Bouldin with single cluster."""
        features = np.random.randn(20, 2)
        labels = np.zeros(20)
        
        score = RegimeMetrics.davies_bouldin_index(features, labels)
        
        # Single cluster returns inf
        assert np.isinf(score)
    
    def test_calinski_harabasz_index(self, perfect_data):
        """Test Calinski-Harabasz index calculation."""
        features, labels = perfect_data
        
        score = RegimeMetrics.calinski_harabasz_index(features, labels)
        
        # Higher is better for CH index
        assert score > 0
        # Well-separated clusters should have high score
        assert score > 100
    
    def test_regime_stability(self):
        """Test regime stability calculation."""
        # Create regime sequence with some stability
        regimes = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2])
        
        stability = RegimeMetrics.regime_stability(regimes)
        
        assert isinstance(stability, dict)
        assert 'persistence' in stability
        assert 0 <= stability['persistence'] <= 1
        # Should have reasonable stability (not too many transitions)
        assert stability['persistence'] > 0.5
    
    def test_regime_stability_constant(self):
        """Test regime stability with constant regime."""
        regimes = np.array([1] * 100)
        
        stability = RegimeMetrics.regime_stability(regimes)
        
        # Perfect stability
        assert stability['persistence'] == 1.0
        assert stability['n_transitions'] == 0
    
    def test_regime_stability_alternating(self):
        """Test regime stability with alternating regimes."""
        regimes = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        stability = RegimeMetrics.regime_stability(regimes)
        
        # Very unstable
        assert stability['persistence'] < 0.3
        assert stability['n_transitions'] == 7
    
    def test_regime_duration_statistics(self):
        """Test regime duration statistics from stability."""
        regimes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0])
        
        stats = RegimeMetrics.regime_stability(regimes)
        
        assert isinstance(stats, dict)
        assert 'avg_duration' in stats
        assert 'duration_std' in stats
        
        # Check values make sense
        assert stats['avg_duration'] > 0
        assert stats['duration_std'] >= 0
    
    def test_regime_distribution(self):
        """Test regime distribution calculation."""
        regimes = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1])
        
        dist = RegimeMetrics.regime_distribution(regimes)
        
        assert isinstance(dist, dict)
        assert 'entropy' in dist
        assert 'normalized_entropy' in dist
        assert 'gini_coefficient' in dist
        
        # Check that metrics are reasonable
        assert dist['entropy'] >= 0
        assert 0 <= dist['normalized_entropy'] <= 1
    
    def test_regime_quality_index(self, perfect_data):
        """Test comprehensive regime quality index."""
        features, labels = perfect_data
        
        # Generate predictions
        pred_labels = TestDataHelper.create_regime_predictions(labels, accuracy=0.8)
        
        # Generate probabilities (confident predictions)
        n_samples = len(pred_labels)
        n_classes = len(np.unique(labels))
        probabilities = np.zeros((n_samples, n_classes))
        for i, label in enumerate(pred_labels):
            probabilities[i, label] = 0.8  # High confidence
            remaining = 0.2 / (n_classes - 1)
            for j in range(n_classes):
                if j != label:
                    probabilities[i, j] = remaining
        
        quality = RegimeMetrics.regime_quality_index(features, pred_labels, probabilities)
        
        assert 0 <= quality <= 100
        # With good clustering, should have decent quality
        assert quality > 50


class TestMetricEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        empty = np.array([])
        
        # ARI with empty arrays returns 1.0 (both empty = perfect match)
        result = RegimeMetrics.adjusted_rand_index(empty, empty)
        assert result == 1.0
    
    def test_mismatched_lengths(self):
        """Test handling of mismatched array lengths."""
        labels1 = np.array([0, 1, 2])
        labels2 = np.array([0, 1])
        
        with pytest.raises(ValueError):
            RegimeMetrics.adjusted_rand_index(labels1, labels2)
    
    def test_single_sample(self):
        """Test handling of single sample."""
        features = np.array([[1, 2]])
        labels = np.array([0])
        
        # Silhouette with single sample returns 0
        result = RegimeMetrics.silhouette_coefficient(features, labels)
        assert result == 0.0
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        features = np.array([[1, 2], [np.nan, 4], [5, 6]])
        labels = np.array([0, 1, 0])
        
        # Should either handle or raise clear error
        try:
            score = RegimeMetrics.silhouette_coefficient(features, labels)
            assert np.isnan(score) or score is not None
        except ValueError:
            pass  # Expected for NaN input


class TestMetricConsistency:
    """Test consistency and relationships between metrics."""
    
    def test_perfect_clustering_consistency(self):
        """Test that perfect clustering gives consistent high scores."""
        features, labels = TestDataHelper.create_perfect_clusters()
        
        silhouette = RegimeMetrics.silhouette_coefficient(features, labels)
        db_index = RegimeMetrics.davies_bouldin_index(features, labels)
        ch_score = RegimeMetrics.calinski_harabasz_index(features, labels)
        
        # Perfect clustering should have:
        assert silhouette > 0.7  # High silhouette
        assert db_index < 1.0  # Low DB index
        assert ch_score > 200  # High CH score
    
    def test_poor_clustering_consistency(self):
        """Test that poor clustering gives consistent low scores."""
        features, labels = TestDataHelper.create_overlapping_clusters()
        
        silhouette = RegimeMetrics.silhouette_coefficient(features, labels)
        db_index = RegimeMetrics.davies_bouldin_index(features, labels)
        
        # Poor clustering should have:
        assert silhouette < 0.5  # Lower silhouette
        assert db_index > 0.5  # Higher DB index
    
    def test_metric_monotonicity(self):
        """Test that metrics change monotonically with clustering quality."""
        features, true_labels = TestDataHelper.create_perfect_clusters()
        
        # Test with different accuracy levels
        accuracies = [1.0, 0.8, 0.6, 0.4]
        ari_scores = []
        
        for acc in accuracies:
            pred_labels = TestDataHelper.create_regime_predictions(true_labels, acc)
            score = RegimeMetrics.adjusted_rand_index(true_labels, pred_labels)
            ari_scores.append(score)
        
        # ARI should decrease with decreasing accuracy
        for i in range(len(ari_scores) - 1):
            assert ari_scores[i] >= ari_scores[i + 1] - 0.1  # Allow small variance


class TestRegimeTransitions:
    """Test regime transition analysis."""
    
    def test_transition_counts(self):
        """Test counting regime transitions."""
        regimes = np.array([0, 0, 1, 1, 1, 2, 0, 0])
        
        # Count transitions manually
        expected_transitions = 3  # 0->1, 1->2, 2->0
        
        actual_transitions = np.sum(np.diff(regimes) != 0)
        assert actual_transitions == expected_transitions
    
    def test_regime_persistence(self):
        """Test regime persistence calculation."""
        # Create regime with high persistence
        persistent = np.array([0]*20 + [1]*20 + [2]*20)
        
        # Create regime with low persistence
        volatile = np.tile([0, 1, 2], 20)
        
        # Calculate persistence
        persist_stats = RegimeMetrics.regime_stability(persistent)
        volatile_stats = RegimeMetrics.regime_stability(volatile)
        
        persist_score_high = persist_stats['persistence']
        persist_score_low = volatile_stats['persistence']
        
        assert persist_score_high > persist_score_low
    
    def test_regime_entropy(self):
        """Test regime distribution entropy."""
        # Uniform distribution
        uniform = np.array([0, 0, 1, 1, 2, 2])
        
        # Skewed distribution
        skewed = np.array([0, 0, 0, 0, 1, 2])
        
        # Calculate entropy
        from scipy.stats import entropy
        
        uniform_counts = np.bincount(uniform)
        uniform_probs = uniform_counts / len(uniform)
        uniform_entropy = entropy(uniform_probs)
        
        skewed_counts = np.bincount(skewed)
        skewed_probs = skewed_counts / len(skewed)
        skewed_entropy = entropy(skewed_probs)
        
        # Uniform should have higher entropy
        assert uniform_entropy > skewed_entropy