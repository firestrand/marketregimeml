"""Comprehensive tests for regime reordering utilities following TDD approach."""

import pytest
import numpy as np
from marketregimeml.utils.regime_utils import RegimeReorderingUtils


class TestRegimeReorderingUtils:
    """Test suite for RegimeReorderingUtils following AAA pattern."""

    def test_reorder_regimes_by_feature_mean_basic(self):
        """Test basic regime reordering by feature mean."""
        # Arrange
        regimes = np.array([0, 1, 0, 1, 2, 2])
        features = np.array(
            [
                [1.0, 0.5],  # Regime 0: mean = 1.5
                [3.0, 2.0],  # Regime 1: mean = 3.0
                [2.0, 1.0],  # Regime 0: mean = 1.5
                [4.0, 3.0],  # Regime 1: mean = 3.0
                [0.5, 0.2],  # Regime 2: mean = 0.5
                [0.8, 0.3],  # Regime 2: mean = 0.5
            ]
        )
        n_regimes = 3
        # Expected order by feature 0 mean: Regime 2 (0.65), Regime 0 (1.5), Regime 1 (3.5)
        # So mapping: {2: 0, 0: 1, 1: 2}

        # Act
        reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes, feature_index=0
        )

        # Assert
        assert isinstance(reordered, np.ndarray)
        assert len(reordered) == len(regimes)
        assert set(reordered) == {0, 1, 2}

        # Check that regime with lowest mean gets label 0
        regime_means = []
        for i in range(n_regimes):
            mask = reordered == i
            if mask.sum() > 0:
                regime_means.append(features[mask, 0].mean())

        # Means should be in ascending order
        assert regime_means == sorted(regime_means)

    def test_reorder_regimes_by_feature_mean_different_feature(self):
        """Test regime reordering using different feature index."""
        # Arrange
        regimes = np.array([0, 1, 0, 1])
        features = np.array(
            [
                [1.0, 10.0],  # Regime 0
                [2.0, 5.0],  # Regime 1
                [1.5, 8.0],  # Regime 0
                [2.5, 7.0],  # Regime 1
            ]
        )
        n_regimes = 2

        # Act - Use feature index 1
        reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes, feature_index=1
        )

        # Assert - Check ordering by feature 1
        regime_means = []
        for i in range(n_regimes):
            mask = reordered == i
            if mask.sum() > 0:
                regime_means.append(features[mask, 1].mean())

        assert regime_means == sorted(regime_means)

    def test_reorder_regimes_empty_regime(self):
        """Test handling of empty regimes."""
        # Arrange
        regimes = np.array([0, 0, 2, 2])  # Regime 1 is empty
        features = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
        n_regimes = 3

        # Act
        reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes
        )

        # Assert
        assert isinstance(reordered, np.ndarray)
        assert len(reordered) == len(regimes)
        # Empty regime should get mean of 0 and be ordered first
        unique_regimes = set(reordered)
        assert len(unique_regimes) <= n_regimes

    def test_reorder_regimes_single_regime(self):
        """Test reordering with single regime."""
        # Arrange
        regimes = np.array([0, 0, 0, 0])
        features = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
        n_regimes = 1

        # Act
        reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes
        )

        # Assert
        assert np.array_equal(reordered, np.array([0, 0, 0, 0]))

    def test_reorder_regimes_invalid_feature_index(self):
        """Test error handling for invalid feature index."""
        # Arrange
        regimes = np.array([0, 1, 0, 1])
        features = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
        n_regimes = 2

        # Act & Assert
        with pytest.raises(ValueError, match="Feature index 5 out of bounds"):
            RegimeReorderingUtils.reorder_regimes_by_feature_mean(
                regimes, features, n_regimes, feature_index=5
            )

    def test_reorder_probabilities_basic(self):
        """Test basic probability matrix reordering."""
        # Arrange
        probabilities = np.array(
            [
                [0.8, 0.1, 0.1],  # Strong regime 0
                [0.2, 0.7, 0.1],  # Strong regime 1
                [0.1, 0.2, 0.7],  # Strong regime 2
                [0.6, 0.3, 0.1],  # Moderate regime 0
            ]
        )
        regimes = np.array([0, 1, 2, 0])
        features = np.array(
            [
                [2.0, 1.0],  # Regime 0
                [3.0, 2.0],  # Regime 1
                [1.0, 0.5],  # Regime 2
                [2.5, 1.5],  # Regime 0
            ]
        )
        n_regimes = 3

        # Act
        reordered_probs = RegimeReorderingUtils.reorder_probabilities(
            probabilities, regimes, features, n_regimes
        )

        # Assert
        assert reordered_probs.shape == probabilities.shape
        assert np.allclose(
            reordered_probs.sum(axis=1), 1.0
        )  # Probabilities should sum to 1

    def test_reorder_probabilities_wrong_shape(self):
        """Test error handling for incorrect probability matrix shape."""
        # Arrange
        probabilities = np.array([[0.5, 0.5], [0.6, 0.4]])  # Only 2 columns
        regimes = np.array([0, 1])
        features = np.array([[1.0], [2.0]])
        n_regimes = 3

        # Act & Assert
        with pytest.raises(
            ValueError, match="Probability matrix has 2 columns, expected 3"
        ):
            RegimeReorderingUtils.reorder_probabilities(
                probabilities, regimes, features, n_regimes
            )

    def test_get_regime_mapping_basic(self):
        """Test getting regime mapping dictionary."""
        # Arrange
        regimes = np.array([0, 1, 2, 0, 1, 2])
        features = np.array(
            [
                [3.0, 1.0],  # Regime 0: mean = 3.5
                [1.0, 2.0],  # Regime 1: mean = 1.5
                [2.0, 3.0],  # Regime 2: mean = 2.0
                [4.0, 1.5],  # Regime 0
                [2.0, 2.5],  # Regime 1
                [2.0, 3.5],  # Regime 2
            ]
        )
        n_regimes = 3
        # Expected order by feature 0: Regime 1 (1.5), Regime 2 (2.0), Regime 0 (3.5)

        # Act
        mapping = RegimeReorderingUtils.get_regime_mapping(
            regimes, features, n_regimes
        )

        # Assert
        assert isinstance(mapping, dict)
        assert len(mapping) == n_regimes
        assert all(
            isinstance(k, int) and isinstance(v, int)
            for k, v in mapping.items()
        )
        assert set(mapping.keys()) == {0, 1, 2}
        assert set(mapping.values()) == {0, 1, 2}

    def test_validate_regime_labels_valid(self):
        """Test validation of valid regime labels."""
        # Arrange
        regimes = np.array([0, 1, 2, 0, 1, 2])
        n_regimes = 3

        # Act
        is_valid = RegimeReorderingUtils.validate_regime_labels(
            regimes, n_regimes
        )

        # Assert
        assert is_valid is True

    def test_validate_regime_labels_invalid_negative(self):
        """Test validation with negative regime labels."""
        # Arrange
        regimes = np.array([0, 1, -1, 0])
        n_regimes = 2

        # Act
        is_valid = RegimeReorderingUtils.validate_regime_labels(
            regimes, n_regimes
        )

        # Assert
        assert is_valid is False

    def test_validate_regime_labels_invalid_too_high(self):
        """Test validation with regime labels exceeding expected range."""
        # Arrange
        regimes = np.array([0, 1, 2, 3])
        n_regimes = 3

        # Act
        is_valid = RegimeReorderingUtils.validate_regime_labels(
            regimes, n_regimes
        )

        # Assert
        assert is_valid is False

    def test_validate_regime_labels_empty(self):
        """Test validation of empty regime array."""
        # Arrange
        regimes = np.array([])
        n_regimes = 2

        # Act
        is_valid = RegimeReorderingUtils.validate_regime_labels(
            regimes, n_regimes
        )

        # Assert
        assert is_valid is True

    def test_reorder_regimes_consistency(self):
        """Test that reordering produces consistent results."""
        # Arrange
        np.random.seed(42)
        regimes = np.random.choice([0, 1, 2], size=100)
        features = np.random.randn(100, 2)
        n_regimes = 3

        # Act - Run multiple times
        result1 = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes
        )
        result2 = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes
        )

        # Assert
        assert np.array_equal(result1, result2)

    def test_reorder_preserves_regime_structure(self):
        """Test that reordering preserves the regime structure."""
        # Arrange
        regimes = np.array([0, 0, 1, 1, 2, 2])
        features = np.array(
            [
                [1.0, 0.5],
                [1.2, 0.7],  # Regime 0
                [3.0, 2.0],
                [3.5, 2.5],  # Regime 1
                [0.5, 0.2],
                [0.8, 0.3],  # Regime 2
            ]
        )
        n_regimes = 3

        # Act
        reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes
        )

        # Assert - Check that samples from same original regime stay together
        # Find where original regime 0 samples went
        original_regime_0_indices = np.where(regimes == 0)[0]
        new_labels_for_regime_0 = reordered[original_regime_0_indices]

        # All samples from original regime 0 should have same new label
        assert len(set(new_labels_for_regime_0)) == 1

        # Same for regime 1
        original_regime_1_indices = np.where(regimes == 1)[0]
        new_labels_for_regime_1 = reordered[original_regime_1_indices]
        assert len(set(new_labels_for_regime_1)) == 1

        # Same for regime 2
        original_regime_2_indices = np.where(regimes == 2)[0]
        new_labels_for_regime_2 = reordered[original_regime_2_indices]
        assert len(set(new_labels_for_regime_2)) == 1

    def test_regime_means_ordering_after_reordering(self):
        """Test that regime means are properly ordered after reordering."""
        # Arrange
        np.random.seed(123)
        regimes = np.array([0] * 20 + [1] * 20 + [2] * 20)

        # Create features where regime means are: 2 (regime 0), 1 (regime 1), 3 (regime 2)
        features = np.concatenate(
            [
                np.random.normal(2, 0.1, (20, 1)),  # Regime 0
                np.random.normal(1, 0.1, (20, 1)),  # Regime 1
                np.random.normal(3, 0.1, (20, 1)),  # Regime 2
            ]
        )
        n_regimes = 3

        # Act
        reordered = RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, n_regimes
        )

        # Assert - Calculate means for reordered regimes
        reordered_means = []
        for i in range(n_regimes):
            mask = reordered == i
            reordered_means.append(features[mask, 0].mean())

        # Means should be in ascending order
        assert reordered_means == sorted(reordered_means)

        # Verify the specific ordering: regime 1 -> 0, regime 0 -> 1, regime 2 -> 2
        # (since means are 1 < 2 < 3)
        original_regime_1_mask = regimes == 1
        assert np.all(
            reordered[original_regime_1_mask] == 0
        )  # Lowest mean -> label 0

        original_regime_0_mask = regimes == 0
        assert np.all(
            reordered[original_regime_0_mask] == 1
        )  # Middle mean -> label 1

        original_regime_2_mask = regimes == 2
        assert np.all(
            reordered[original_regime_2_mask] == 2
        )  # Highest mean -> label 2
