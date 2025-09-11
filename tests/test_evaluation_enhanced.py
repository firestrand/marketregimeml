"""Test suite for enhanced evaluation metrics following TDD approach."""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any


class TestRegimeFlipRate:
    """Test regime flip rate calculations."""

    @pytest.fixture
    def sample_regimes(self):
        """Generate sample regime sequences."""
        # Stable sequence (low flip rate)
        stable = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        # Unstable sequence (high flip rate)
        unstable = np.array([0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        # Mixed sequence
        mixed = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 1, 2])

        return {"stable": stable, "unstable": unstable, "mixed": mixed}

    def test_calculate_flip_rate(self, sample_regimes):
        """Test basic flip rate calculation."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        # Stable sequence should have low flip rate
        stable_rate = metrics.calculate_flip_rate(sample_regimes["stable"])
        assert 0 <= stable_rate <= 1
        assert stable_rate < 0.3  # Low flip rate

        # Unstable sequence should have high flip rate
        unstable_rate = metrics.calculate_flip_rate(sample_regimes["unstable"])
        assert unstable_rate > 0.7  # High flip rate

        # Mixed sequence should be in between
        mixed_rate = metrics.calculate_flip_rate(sample_regimes["mixed"])
        assert stable_rate < mixed_rate < unstable_rate

    def test_flip_rate_by_regime(self, sample_regimes):
        """Test flip rate calculation per regime."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        # Get per-regime flip rates
        regime_rates = metrics.calculate_flip_rate_by_regime(
            sample_regimes["mixed"]
        )

        assert isinstance(regime_rates, dict)
        assert all(0 <= rate <= 1 for rate in regime_rates.values())
        assert len(regime_rates) == 3  # Three regimes (0, 1, 2)

    def test_flip_rate_window(self):
        """Test windowed flip rate calculation."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        # Long sequence with varying stability
        regimes = np.array([0] * 50 + [1] * 50 + [0, 1, 2] * 20 + [2] * 40)

        # Calculate windowed flip rates
        window_rates = metrics.calculate_windowed_flip_rate(
            regimes, window_size=20
        )

        assert len(window_rates) == len(regimes)
        assert all(
            0 <= rate <= 1 for rate in window_rates if not np.isnan(rate)
        )

        # First window should be stable (all 0s)
        assert window_rates[20] < 0.1

        # Transition area should have higher flip rate
        assert max(window_rates[100:120]) > 0.5


class TestRegimePersistence:
    """Test regime persistence metrics."""

    @pytest.fixture
    def regime_sequence(self):
        """Create regime sequence with known persistence."""
        return np.array(
            [
                0,
                0,
                0,
                0,
                0,  # 5 periods in regime 0
                1,
                1,
                1,  # 3 periods in regime 1
                2,
                2,
                2,
                2,
                2,
                2,
                2,  # 7 periods in regime 2
                0,
                0,  # 2 periods in regime 0
                1,
                1,
                1,
                1,  # 4 periods in regime 1
            ]
        )

    def test_calculate_persistence(self, regime_sequence):
        """Test basic persistence calculation."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        persistence = metrics.calculate_persistence(regime_sequence)

        assert isinstance(persistence, dict)
        assert "mean_duration" in persistence
        assert "median_duration" in persistence
        assert "max_duration" in persistence
        assert "min_duration" in persistence
        assert "persistence_score" in persistence

        # Check values
        assert persistence["mean_duration"] == 4.2  # (5+3+7+2+4)/5
        assert persistence["max_duration"] == 7
        assert persistence["min_duration"] == 2
        assert 0 <= persistence["persistence_score"] <= 1

    def test_persistence_by_regime(self, regime_sequence):
        """Test persistence calculation per regime."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        regime_persistence = metrics.calculate_persistence_by_regime(
            regime_sequence
        )

        assert isinstance(regime_persistence, dict)
        assert len(regime_persistence) == 3  # Three unique regimes

        # Regime 0: durations [5, 2], mean = 3.5
        assert regime_persistence[0]["mean_duration"] == 3.5

        # Regime 1: durations [3, 4], mean = 3.5
        assert regime_persistence[1]["mean_duration"] == 3.5

        # Regime 2: duration [7], mean = 7
        assert regime_persistence[2]["mean_duration"] == 7.0

    def test_transition_matrix_persistence(self):
        """Test persistence from transition matrix."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        # Create transition matrix with known persistence
        transition_matrix = np.array(
            [
                [0.9, 0.05, 0.05],  # High persistence for regime 0
                [0.1, 0.8, 0.1],  # Medium persistence for regime 1
                [0.2, 0.2, 0.6],  # Low persistence for regime 2
            ]
        )

        persistence = metrics.calculate_persistence_from_transition(
            transition_matrix
        )

        assert len(persistence) == 3

        # Expected duration = 1 / (1 - self-transition probability)
        assert np.isclose(persistence[0], 1 / (1 - 0.9), rtol=0.01)  # ~10
        assert np.isclose(persistence[1], 1 / (1 - 0.8), rtol=0.01)  # ~5
        assert np.isclose(persistence[2], 1 / (1 - 0.6), rtol=0.01)  # ~2.5


class TestCrossRegimeCorrelation:
    """Test cross-regime correlation analysis."""

    @pytest.fixture
    def regime_data(self):
        """Generate data with regime-dependent correlations."""
        np.random.seed(42)
        n_samples = 300

        # Create regimes
        regimes = np.array([0] * 100 + [1] * 100 + [2] * 100)

        # Create features with different correlations per regime
        features = np.zeros((n_samples, 2))

        # Regime 0: High positive correlation
        cov0 = [[1, 0.8], [0.8, 1]]
        features[:100] = np.random.multivariate_normal([0, 0], cov0, 100)

        # Regime 1: Negative correlation
        cov1 = [[1, -0.6], [-0.6, 1]]
        features[100:200] = np.random.multivariate_normal([1, -1], cov1, 100)

        # Regime 2: No correlation
        cov2 = [[1, 0], [0, 1]]
        features[200:] = np.random.multivariate_normal([0, 0], cov2, 100)

        return (
            pd.DataFrame(features, columns=["feature1", "feature2"]),
            regimes,
        )

    def test_cross_regime_correlation(self, regime_data):
        """Test correlation analysis across regimes."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        features, regimes = regime_data
        metrics = EnhancedRegimeMetrics()

        correlations = metrics.calculate_cross_regime_correlation(
            features, regimes
        )

        assert isinstance(correlations, dict)
        assert len(correlations) == 3  # Three regimes

        # Check correlation matrices
        for regime_id, corr_matrix in correlations.items():
            assert corr_matrix.shape == (2, 2)
            assert np.allclose(
                np.diag(corr_matrix), 1.0
            )  # Diagonal should be 1
            assert np.allclose(
                corr_matrix, corr_matrix.T
            )  # Should be symmetric

        # Verify expected correlations
        assert correlations[0][0, 1] > 0.5  # Positive correlation in regime 0
        assert correlations[1][0, 1] < -0.3  # Negative correlation in regime 1
        assert abs(correlations[2][0, 1]) < 0.3  # Low correlation in regime 2

    def test_correlation_stability(self, regime_data):
        """Test correlation stability within regimes."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        features, regimes = regime_data
        metrics = EnhancedRegimeMetrics()

        stability = metrics.calculate_correlation_stability(
            features, regimes, window_size=30
        )

        assert isinstance(stability, dict)
        assert "mean_stability" in stability
        assert "regime_stability" in stability

        # Overall stability should be reasonable
        assert 0 <= stability["mean_stability"] <= 1

        # Per-regime stability
        assert len(stability["regime_stability"]) == 3
        for regime_id, stab in stability["regime_stability"].items():
            assert 0 <= stab <= 1


class TestRegimeQualityIndex:
    """Test composite Regime Quality Index."""

    @pytest.fixture
    def quality_test_data(self):
        """Generate comprehensive test data for quality metrics."""
        np.random.seed(42)

        # Good quality regimes (stable, well-separated)
        good_regimes = np.array([0] * 100 + [1] * 100 + [2] * 100)
        good_features = np.vstack(
            [
                np.random.normal(0, 0.5, (100, 2)),  # Regime 0
                np.random.normal(3, 0.5, (100, 2)),  # Regime 1
                np.random.normal(-2, 0.5, (100, 2)),  # Regime 2
            ]
        )
        good_probas = np.zeros((300, 3))
        good_probas[:100, 0] = 0.9
        good_probas[100:200, 1] = 0.9
        good_probas[200:, 2] = 0.9
        good_probas += 0.05  # Add small probability to other regimes
        good_probas = good_probas / good_probas.sum(axis=1, keepdims=True)

        # Poor quality regimes (unstable, overlapping)
        poor_regimes = np.random.randint(0, 3, 300)
        poor_features = np.random.normal(0, 1, (300, 2))
        poor_probas = np.ones((300, 3)) / 3  # Uniform probabilities

        return {
            "good": (good_features, good_regimes, good_probas),
            "poor": (poor_features, poor_regimes, poor_probas),
        }

    def test_regime_quality_index(self, quality_test_data):
        """Test composite quality index calculation."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        # Test good quality regimes
        good_features, good_regimes, good_probas = quality_test_data["good"]
        good_quality = metrics.regime_quality_index(
            good_features, good_regimes, good_probas
        )

        assert isinstance(good_quality, dict)
        assert "overall_score" in good_quality
        assert "components" in good_quality

        # Good quality should have high score
        assert 0 <= good_quality["overall_score"] <= 100
        assert good_quality["overall_score"] > 70  # Should be high quality

        # Test poor quality regimes
        poor_features, poor_regimes, poor_probas = quality_test_data["poor"]
        poor_quality = metrics.regime_quality_index(
            poor_features, poor_regimes, poor_probas
        )

        # Poor quality should have low score
        assert poor_quality["overall_score"] < 40  # Should be low quality

        # Good should score better than poor
        assert good_quality["overall_score"] > poor_quality["overall_score"]

    def test_quality_components(self, quality_test_data):
        """Test individual quality components."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        good_features, good_regimes, good_probas = quality_test_data["good"]
        quality = metrics.regime_quality_index(
            good_features, good_regimes, good_probas
        )

        components = quality["components"]

        # Check all components are present
        expected_components = [
            "separation_score",
            "stability_score",
            "confidence_score",
            "persistence_score",
            "consistency_score",
        ]

        for component in expected_components:
            assert component in components
            assert 0 <= components[component] <= 1

    def test_temporal_consistency(self):
        """Test temporal consistency metric."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        # Consistent sequence (regimes persist)
        consistent = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        consistency_high = metrics.calculate_temporal_consistency(
            consistent, window_size=3
        )

        # Inconsistent sequence (rapid changes)
        inconsistent = np.array([0, 1, 0, 2, 1, 0, 2, 1, 0])
        consistency_low = metrics.calculate_temporal_consistency(
            inconsistent, window_size=3
        )

        assert 0 <= consistency_high <= 1
        assert 0 <= consistency_low <= 1
        assert consistency_high > consistency_low


class TestAdvancedMetrics:
    """Test additional advanced metrics."""

    def test_regime_entropy(self):
        """Test regime entropy calculation."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        # Low entropy (one dominant regime)
        low_entropy_regimes = np.array([0] * 90 + [1] * 5 + [2] * 5)
        low_entropy = metrics.calculate_regime_entropy(low_entropy_regimes)

        # High entropy (balanced regimes)
        high_entropy_regimes = np.array([0] * 33 + [1] * 34 + [2] * 33)
        high_entropy = metrics.calculate_regime_entropy(high_entropy_regimes)

        assert 0 <= low_entropy <= np.log(3)
        assert 0 <= high_entropy <= np.log(3)
        assert low_entropy < high_entropy

    def test_regime_purity(self):
        """Test regime purity metric."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        # High purity probabilities
        high_purity_probas = np.array(
            [[0.95, 0.03, 0.02], [0.02, 0.96, 0.02], [0.01, 0.01, 0.98]]
        )

        # Low purity probabilities
        low_purity_probas = np.array(
            [[0.4, 0.3, 0.3], [0.35, 0.35, 0.3], [0.33, 0.33, 0.34]]
        )

        high_purity = metrics.calculate_regime_purity(high_purity_probas)
        low_purity = metrics.calculate_regime_purity(low_purity_probas)

        assert 0 <= high_purity <= 1
        assert 0 <= low_purity <= 1
        assert high_purity > low_purity
        assert high_purity > 0.9  # Should be very high
        assert low_purity < 0.5  # Should be low

    def test_regime_separation(self):
        """Test regime separation in feature space."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        metrics = EnhancedRegimeMetrics()

        # Well-separated features
        well_separated = np.vstack(
            [
                np.random.normal(0, 0.1, (50, 2)),  # Regime 0
                np.random.normal(5, 0.1, (50, 2)),  # Regime 1
                np.random.normal(-5, 0.1, (50, 2)),  # Regime 2
            ]
        )
        well_separated_regimes = np.array([0] * 50 + [1] * 50 + [2] * 50)

        # Overlapping features
        overlapping = np.random.normal(0, 1, (150, 2))
        overlapping_regimes = np.array([0] * 50 + [1] * 50 + [2] * 50)

        sep_good = metrics.calculate_regime_separation(
            well_separated, well_separated_regimes
        )
        sep_poor = metrics.calculate_regime_separation(
            overlapping, overlapping_regimes
        )

        assert sep_good > sep_poor
        assert sep_good > 0.5  # Should indicate good separation
        assert sep_poor < 0.3  # Should indicate poor separation


class TestMetricsIntegration:
    """Integration tests for all metrics."""

    def test_complete_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )
        from marketregimeml.models.hmm import HMMRegimeDetector

        # Generate synthetic data
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "returns": np.random.normal(0, 0.01, 500),
                "volatility": np.abs(np.random.normal(0.01, 0.002, 500)),
            }
        )

        # Fit model
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(features, n_init=3)

        regimes = detector.predict(features)
        probas = detector.predict_proba(features)

        # Calculate all metrics
        metrics = EnhancedRegimeMetrics()

        # Individual metrics
        flip_rate = metrics.calculate_flip_rate(regimes)
        persistence = metrics.calculate_persistence(regimes)
        quality = metrics.regime_quality_index(
            features.values, regimes, probas
        )

        # All metrics should be valid
        assert 0 <= flip_rate <= 1
        assert persistence["mean_duration"] > 0
        assert 0 <= quality["overall_score"] <= 100

    def test_metrics_report(self):
        """Test comprehensive metrics report generation."""
        from marketregimeml.evaluation.enhanced_metrics import (
            EnhancedRegimeMetrics,
        )

        # Create sample data
        np.random.seed(42)
        regimes = np.array([0] * 100 + [1] * 100 + [2] * 100)
        features = np.random.normal(0, 1, (300, 3))
        probas = np.eye(3)[regimes]  # Perfect predictions

        metrics = EnhancedRegimeMetrics()

        # Generate comprehensive report
        report = metrics.generate_report(features, regimes, probas)

        assert isinstance(report, dict)

        # Check main sections
        expected_sections = [
            "summary",
            "stability_metrics",
            "quality_metrics",
            "regime_statistics",
            "recommendations",
        ]

        for section in expected_sections:
            assert section in report

        # Summary should have key metrics
        assert "overall_quality" in report["summary"]
        assert "flip_rate" in report["summary"]
        assert "mean_persistence" in report["summary"]
