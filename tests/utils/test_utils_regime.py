"""Unit tests for regime utilities.

Tests follow bottom-up approach with minimal mocking, using real data where possible.
"""

import pytest
import numpy as np
from marketregimeml.utils.regime_utils import RegimeReorderingUtils


class TestRegimeReorderingUtils:
    """Test regime reordering utilities."""

    def test_reorder_regimes_by_feature_mean_basic(self):
        """Test basic regime reordering by feature mean."""
        # Create test data with clear ordering
        regimes = np.array([1, 1, 0, 0, 2, 2])  # Original labels
        features = np.array(
            [
                [5.0, 1.0],  # regime 1 - high mean
                [6.0, 1.0],  # regime 1 - high mean
                [1.0, 2.0],  # regime 0 - low mean
                [2.0, 2.0],  # regime 0 - low mean
                [3.0, 3.0],  # regime 2 - medium mean
                [4.0, 3.0],  # regime 2 - medium mean
            ]
        )

        reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes=3, feature_index=0
        )

        # Regime 0 has mean 1.5, regime 1 has mean 5.5, regime 2 has mean 3.5
        # So order should be: 0->0, 2->1, 1->2
        expected = np.array([2, 2, 0, 0, 1, 1])
        np.testing.assert_array_equal(reordered, expected)

    def test_reorder_with_different_feature_index(self):
        """Test reordering using different feature column."""
        regimes = np.array([0, 0, 1, 1])
        features = np.array(
            [
                [1.0, 10.0],  # regime 0
                [2.0, 11.0],  # regime 0
                [3.0, 5.0],  # regime 1
                [4.0, 6.0],  # regime 1
            ]
        )

        # Using feature index 1
        reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes=2, feature_index=1
        )

        # Regime 0 has mean 10.5 (feature 1), regime 1 has mean 5.5 (feature 1)
        # So order should be: 1->0, 0->1
        expected = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(reordered, expected)

    def test_reorder_with_empty_regime(self):
        """Test handling of empty regimes."""
        # Regime 2 has no samples
        regimes = np.array([0, 0, 1, 1])
        features = np.array(
            [
                [1.0],
                [2.0],
                [3.0],
                [4.0],
            ]
        )

        reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes=3, feature_index=0
        )

        # Empty regime gets mean 0, so order is: empty(2), regime 0, regime 1
        # Mapping: 2->0, 0->1, 1->2
        expected = np.array([1, 1, 2, 2])
        np.testing.assert_array_equal(reordered, expected)

    def test_reorder_with_invalid_feature_index(self):
        """Test error handling for invalid feature index."""
        regimes = np.array([0, 1])
        features = np.array([[1.0], [2.0]])

        with pytest.raises(ValueError, match="Feature index 1 out of bounds"):
            RegimeReorderingUtils.reorder_regimes_by_feature_mean(
                regimes, features, n_regimes=2, feature_index=1
            )

    def test_reorder_probabilities_basic(self):
        """Test probability matrix reordering."""
        regimes = np.array([0, 0, 1, 1, 2, 2])
        features = np.array(
            [
                [1.0],  # regime 0 - low
                [2.0],  # regime 0 - low
                [5.0],  # regime 1 - high
                [6.0],  # regime 1 - high
                [3.0],  # regime 2 - medium
                [4.0],  # regime 2 - medium
            ]
        )

        # Original probabilities
        probabilities = np.array(
            [
                [0.8, 0.1, 0.1],  # Mostly regime 0
                [0.7, 0.2, 0.1],  # Mostly regime 0
                [0.1, 0.8, 0.1],  # Mostly regime 1
                [0.1, 0.7, 0.2],  # Mostly regime 1
                [0.1, 0.1, 0.8],  # Mostly regime 2
                [0.2, 0.1, 0.7],  # Mostly regime 2
            ]
        )

        reordered_probs = RegimeReorderingUtils.reorder_probabilities(
            probabilities, regimes, features, n_regimes=3, feature_index=0
        )

        # Order by mean: 0(1.5) -> 0, 2(3.5) -> 1, 1(5.5) -> 2
        # So columns should be reordered as [0, 2, 1]
        expected = np.array(
            [
                [0.8, 0.1, 0.1],  # [P(0), P(2), P(1)]
                [0.7, 0.1, 0.2],
                [0.1, 0.1, 0.8],
                [0.1, 0.2, 0.7],
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
            ]
        )

        np.testing.assert_array_almost_equal(reordered_probs, expected)

    def test_reorder_probabilities_wrong_dimensions(self):
        """Test error handling for mismatched probability dimensions."""
        regimes = np.array([0, 1])
        features = np.array([[1.0], [2.0]])
        probabilities = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])  # 3 columns

        with pytest.raises(
            ValueError, match="Probability matrix has 3 columns, expected 2"
        ):
            RegimeReorderingUtils.reorder_probabilities(
                probabilities, regimes, features, n_regimes=2
            )

    def test_get_regime_mapping(self):
        """Test getting regime mapping dictionary."""
        regimes = np.array([0, 0, 1, 1, 2, 2])
        features = np.array(
            [
                [3.0],  # regime 0 - medium
                [4.0],  # regime 0 - medium
                [1.0],  # regime 1 - low
                [2.0],  # regime 1 - low
                [5.0],  # regime 2 - high
                [6.0],  # regime 2 - high
            ]
        )

        mapping = RegimeReorderingUtils.get_regime_mapping(
            regimes, features, n_regimes=3, feature_index=0
        )

        # Order by mean: 1(1.5) -> 0, 0(3.5) -> 1, 2(5.5) -> 2
        expected_mapping = {1: 0, 0: 1, 2: 2}
        assert mapping == expected_mapping

    def test_get_regime_mapping_with_empty(self):
        """Test mapping with empty regime."""
        regimes = np.array([0, 0, 2, 2])  # No regime 1
        features = np.array([[3.0], [4.0], [1.0], [2.0]])

        mapping = RegimeReorderingUtils.get_regime_mapping(
            regimes, features, n_regimes=3
        )

        # Regime 1 is empty (mean=0), regime 2 has mean 1.5, regime 0 has mean 3.5
        # Order: 1(0) -> 0, 2(1.5) -> 1, 0(3.5) -> 2
        expected_mapping = {1: 0, 2: 1, 0: 2}
        assert mapping == expected_mapping

    def test_validate_regime_labels_valid(self):
        """Test validation of valid regime labels."""
        regimes = np.array([0, 1, 2, 0, 1, 2])

        assert RegimeReorderingUtils.validate_regime_labels(regimes, n_regimes=3)
        assert RegimeReorderingUtils.validate_regime_labels(regimes, n_regimes=4)

    def test_validate_regime_labels_invalid(self):
        """Test validation of invalid regime labels."""
        # Negative regime
        regimes = np.array([-1, 0, 1])
        assert not RegimeReorderingUtils.validate_regime_labels(regimes, n_regimes=3)

        # Regime exceeds n_regimes
        regimes = np.array([0, 1, 2, 3])
        assert not RegimeReorderingUtils.validate_regime_labels(regimes, n_regimes=3)

    def test_validate_regime_labels_edge_cases(self):
        """Test validation edge cases."""
        # Empty array
        assert RegimeReorderingUtils.validate_regime_labels(np.array([]), n_regimes=3)

        # Single regime
        assert RegimeReorderingUtils.validate_regime_labels(
            np.array([0, 0, 0]), n_regimes=1
        )

        # All same regime
        assert RegimeReorderingUtils.validate_regime_labels(
            np.array([1, 1, 1]), n_regimes=3
        )

    def test_consistency_between_methods(self):
        """Test consistency between reordering methods."""
        np.random.seed(42)
        n_samples = 100
        n_regimes = 3

        # Generate random data
        regimes = np.random.choice(n_regimes, n_samples)
        features = np.random.randn(n_samples, 2)

        # Get reordered regimes
        reordered_regimes = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes=n_regimes
        )

        # Get mapping
        mapping = RegimeReorderingUtils.get_regime_mapping(
            regimes, features, n_regimes=n_regimes
        )

        # Apply mapping manually
        manually_reordered = np.array([mapping[r] for r in regimes])

        # Should be identical
        np.testing.assert_array_equal(reordered_regimes, manually_reordered)

    def test_reordering_preserves_sample_count(self):
        """Test that reordering preserves number of samples per regime."""
        regimes = np.array([0, 0, 0, 1, 1, 2])
        features = np.random.randn(6, 1)

        # Count original
        original_counts = np.bincount(regimes)

        # Reorder
        reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes=3
        )

        # Count reordered
        reordered_counts = np.bincount(reordered)

        # Should have same counts, just potentially in different order
        assert sorted(original_counts) == sorted(reordered_counts)

    def test_deterministic_reordering(self):
        """Test that reordering is deterministic."""
        regimes = np.array([0, 1, 2, 0, 1, 2])
        features = np.array([[1.0], [3.0], [2.0], [1.5], [3.5], [2.5]])

        # Run multiple times
        results = []
        for _ in range(5):
            reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
                regimes, features, n_regimes=3
            )
            results.append(reordered)

        # All results should be identical
        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)
