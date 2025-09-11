"""Comprehensive tests for evaluation modules following SOLID, DRY, KISS principles."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.evaluation.evaluator import ModelEvaluator
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector


class TestRegimeMetricsComprehensive:
    """Comprehensive tests for RegimeMetrics - Single Responsibility."""

    @pytest.fixture
    def sample_predictions(self):
        """Sample regime predictions - DRY principle."""
        np.random.seed(42)
        # Create realistic regime sequence with persistence
        regimes = []
        current = 0
        for _ in range(100):
            if np.random.random() < 0.1:  # 10% chance to switch
                current = np.random.choice([0, 1, 2])
            regimes.append(current)
        return np.array(regimes)

    @pytest.fixture
    def sample_probabilities(self):
        """Sample probability matrix - DRY principle."""
        np.random.seed(42)
        probs = np.random.rand(100, 3)
        # Normalize to sum to 1
        return probs / probs.sum(axis=1, keepdims=True)

    @pytest.fixture
    def sample_features(self):
        """Sample feature matrix."""
        np.random.seed(42)
        return np.random.randn(100, 5)

    def test_adjusted_rand_index(self, sample_predictions):
        """Test ARI calculation."""
        metrics = RegimeMetrics()
        true_labels = sample_predictions.copy()
        pred_labels = sample_predictions.copy()

        # Perfect match
        ari = metrics.adjusted_rand_index(true_labels, pred_labels)
        assert ari == 1.0

        # Random labels
        random_labels = np.random.randint(0, 3, size=len(true_labels))
        ari_random = metrics.adjusted_rand_index(true_labels, random_labels)
        assert -1 <= ari_random <= 1

    def test_normalized_mutual_info(self, sample_predictions):
        """Test NMI calculation."""
        metrics = RegimeMetrics()
        true_labels = sample_predictions.copy()

        # Perfect match
        nmi = metrics.normalized_mutual_info(true_labels, true_labels)
        assert nmi == 1.0

        # Different labels
        shifted_labels = np.roll(true_labels, 1)
        nmi_shifted = metrics.normalized_mutual_info(
            true_labels, shifted_labels
        )
        assert 0 <= nmi_shifted <= 1

    def test_silhouette_coefficient(self, sample_features, sample_predictions):
        """Test silhouette score calculation."""
        metrics = RegimeMetrics()

        score = metrics.silhouette_coefficient(
            sample_features, sample_predictions
        )
        assert -1 <= score <= 1

        # Single cluster edge case
        single_cluster = np.zeros(100, dtype=int)
        score_single = metrics.silhouette_coefficient(
            sample_features, single_cluster
        )
        assert score_single == 0

    def test_davies_bouldin_index(self, sample_features, sample_predictions):
        """Test Davies-Bouldin index."""
        metrics = RegimeMetrics()

        db_index = metrics.davies_bouldin_index(
            sample_features, sample_predictions
        )
        assert db_index >= 0

        # Single cluster edge case
        single_cluster = np.zeros(100, dtype=int)
        db_single = metrics.davies_bouldin_index(
            sample_features, single_cluster
        )
        assert db_single == np.inf

    def test_regime_stability(self, sample_predictions):
        """Test regime stability metrics."""
        metrics = RegimeMetrics()

        stability = metrics.regime_stability(sample_predictions)

        assert "transition_rate" in stability
        assert "avg_duration" in stability
        assert "persistence" in stability
        assert "n_transitions" in stability

        assert 0 <= stability["transition_rate"] <= 1
        assert stability["avg_duration"] > 0
        assert 0 <= stability["persistence"] <= 1

    def test_regime_distribution(self, sample_predictions):
        """Test regime distribution analysis."""
        metrics = RegimeMetrics()

        distribution = metrics.regime_distribution(sample_predictions)

        assert "regime_counts" in distribution
        assert "regime_proportions" in distribution
        assert "entropy" in distribution
        assert "gini_coefficient" in distribution

        assert distribution["entropy"] >= 0
        assert 0 <= distribution["gini_coefficient"] <= 1
        assert np.allclose(distribution["regime_proportions"].sum(), 1.0)

    def test_confidence_metrics(self, sample_probabilities):
        """Test confidence metrics from probabilities."""
        metrics = RegimeMetrics()

        confidence = metrics.confidence_metrics(sample_probabilities)

        assert "avg_confidence" in confidence
        assert "confidence_std" in confidence
        assert "avg_entropy" in confidence
        assert "avg_margin" in confidence

        assert 0 <= confidence["avg_confidence"] <= 1
        assert confidence["avg_entropy"] >= 0

    def test_temporal_consistency_metrics(self, sample_predictions):
        """Test temporal consistency metrics."""
        metrics = RegimeMetrics()

        temporal = metrics.temporal_consistency_metrics(
            sample_predictions, window_size=10
        )

        assert "avg_local_consistency" in temporal
        assert "avg_run_length" in temporal
        assert "autocorr_lag1" in temporal

        assert 0 <= temporal["avg_local_consistency"] <= 1
        assert temporal["avg_run_length"] > 0

    def test_regime_quality_index(
        self, sample_features, sample_predictions, sample_probabilities
    ):
        """Test composite quality index."""
        metrics = RegimeMetrics()

        rqi = metrics.regime_quality_index(
            sample_features, sample_predictions, sample_probabilities
        )

        assert 0 <= rqi <= 100

        # Test with returns included
        returns = np.random.randn(100) * 0.01
        rqi_with_returns = metrics.regime_quality_index(
            sample_features, sample_predictions, sample_probabilities, returns
        )

        assert 0 <= rqi_with_returns <= 100


class TestModelEvaluatorComprehensive:
    """Comprehensive tests for ModelEvaluator - Single Responsibility."""

    @pytest.fixture
    def sample_model(self):
        """Create a fitted model - DRY principle."""
        np.random.seed(42)
        model = HMMRegimeDetector(n_regimes=3)
        features = pd.DataFrame(np.random.randn(100, 5))
        model.fit(features)
        return model

    @pytest.fixture
    def sample_features(self):
        """Sample feature DataFrame."""
        np.random.seed(42)
        return pd.DataFrame(
            np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)]
        )

    def test_init(self, sample_model):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(sample_model)

        assert evaluator.model == sample_model
        assert isinstance(evaluator.regime_metrics, RegimeMetrics)

    def test_evaluate_model(self, sample_model, sample_features):
        """Test model evaluation."""
        evaluator = ModelEvaluator(sample_model)

        results = evaluator.evaluate_model(sample_features)

        assert "model_info" in results
        assert "model_diagnostics" in results
        assert "clustering_metrics" in results
        assert "regime_analysis" in results
        assert "temporal_metrics" in results
        assert "regime_quality_index" in results

    def test_evaluate_with_true_labels(self, sample_model, sample_features):
        """Test evaluation with true labels."""
        evaluator = ModelEvaluator(sample_model)

        true_regimes = np.random.randint(0, 3, size=100)
        results = evaluator.evaluate_model(
            sample_features, true_regimes=true_regimes
        )

        assert "supervised_metrics" in results
        assert "adjusted_rand_index" in results["supervised_metrics"]
        assert "normalized_mutual_info" in results["supervised_metrics"]

    def test_cross_validate(self, sample_model, sample_features):
        """Test cross-validation."""
        evaluator = ModelEvaluator(sample_model)

        cv_results = evaluator.cross_validate(sample_features, n_splits=3)

        assert "fold_results" in cv_results
        assert "aggregated_results" in cv_results
        assert len(cv_results["fold_results"]) == 3

    def test_compare_models(self, sample_features):
        """Test model comparison."""
        np.random.seed(42)

        # Create multiple models
        hmm = HMMRegimeDetector(n_regimes=3)
        hmm.fit(sample_features)

        gmm = GMMRegimeDetector(n_regimes=3)
        gmm.fit(sample_features)

        evaluator = ModelEvaluator(hmm)
        comparison = evaluator.compare_models([hmm, gmm], sample_features)

        assert len(comparison) == 2
        assert "model_type" in comparison.columns
        assert "regime_quality_index" in comparison.columns

    def test_generate_report(self, sample_model, sample_features):
        """Test report generation."""
        evaluator = ModelEvaluator(sample_model)

        report = evaluator.generate_report(sample_features)

        assert isinstance(report, str)
        assert "REGIME DETECTION MODEL EVALUATION REPORT" in report
        assert "Regime Quality Index" in report
