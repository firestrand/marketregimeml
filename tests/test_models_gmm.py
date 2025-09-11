"""Tests for Gaussian Mixture Model regime detector."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from marketregimeml.models.gmm import GMMRegimeDetector


class TestGMMRegimeDetector:
    """Test cases for GMM regime detector."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature data."""
        np.random.seed(42)
        n_samples = 500

        # Generate mixture of Gaussians
        n_components = 3
        weights = [0.3, 0.4, 0.3]

        features = []
        true_labels = []

        for i in range(n_components):
            n_comp = int(n_samples * weights[i])
            # Different means and covariances for each component
            mean = [i - 1, i * 2, i]
            cov = np.eye(3) * (1 + i * 0.5)
            comp_features = np.random.multivariate_normal(mean, cov, n_comp)
            features.append(comp_features)
            true_labels.extend([i] * n_comp)

        features = np.vstack(features)

        # Shuffle
        indices = np.random.permutation(len(features))
        features = features[indices]

        return pd.DataFrame(
            features,
            columns=["feature1", "feature2", "feature3"],
            index=pd.date_range("2023-01-01", periods=len(features), freq="D"),
        )

    def test_initialization(self):
        """Test GMM initialization."""
        detector = GMMRegimeDetector(
            n_regimes=3, covariance_type="full", max_iter=100, random_state=42
        )

        assert detector.n_regimes == 3
        assert detector.covariance_type == "full"
        assert detector.max_iter == 100
        assert detector.random_state == 42
        assert not detector.is_fitted
        assert detector.model is None

    def test_fit_basic(self, sample_features):
        """Test basic model fitting."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)

        # Fit model
        result = detector.fit(sample_features)

        assert result is detector  # Method chaining
        assert detector.is_fitted
        assert detector.model is not None
        assert detector.n_features == 3
        assert detector.feature_names == ["feature1", "feature2", "feature3"]
        assert detector.train_log_likelihood is not None

    def test_fit_with_optimal_selection(self, sample_features):
        """Test automatic optimal component selection."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)

        # Fit with automatic selection
        detector.fit(sample_features, select_optimal_n=True, max_components=5)

        assert detector.is_fitted
        assert detector.optimal_n_analysis is not None
        assert "n_regimes_range" in detector.optimal_n_analysis
        assert "aic_scores" in detector.optimal_n_analysis
        assert "bic_scores" in detector.optimal_n_analysis
        assert "optimal_n" in detector.optimal_n_analysis
        assert "selection_criterion" in detector.optimal_n_analysis

        # Optimal n should be within tested range
        assert 2 <= detector.n_regimes <= 5

    def test_predict(self, sample_features):
        """Test regime prediction."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features)

        # Predict regimes
        regimes = detector.predict(sample_features)

        assert isinstance(regimes, np.ndarray)
        assert len(regimes) == len(sample_features)
        assert regimes.min() >= 0
        assert regimes.max() < 3
        assert len(np.unique(regimes)) <= 3

    def test_predict_not_fitted(self, sample_features):
        """Test prediction error when not fitted."""
        detector = GMMRegimeDetector(n_regimes=3)

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict(sample_features)

    def test_predict_proba(self, sample_features):
        """Test probability prediction."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features)

        # Predict probabilities
        proba = detector.predict_proba(sample_features)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (len(sample_features), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_fit_predict(self, sample_features):
        """Test combined fit and predict."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)

        regimes = detector.fit_predict(sample_features)

        assert detector.is_fitted
        assert isinstance(regimes, np.ndarray)
        assert len(regimes) == len(sample_features)

    def test_sample(self, sample_features):
        """Test sample generation."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features)

        # Generate samples
        generated_features, generated_regimes = detector.sample(
            n_samples=100, random_state=42
        )

        assert isinstance(generated_features, pd.DataFrame)
        assert len(generated_features) == 100
        assert list(generated_features.columns) == [
            "feature1",
            "feature2",
            "feature3",
        ]
        assert isinstance(generated_regimes, np.ndarray)
        assert len(generated_regimes) == 100

    def test_covariance_types(self, sample_features):
        """Test different covariance types."""
        cov_types = ["full", "diag", "tied", "spherical"]

        for cov_type in cov_types:
            detector = GMMRegimeDetector(
                n_regimes=3, covariance_type=cov_type, random_state=42
            )
            detector.fit(sample_features)

            assert detector.is_fitted
            assert detector.model is not None

            # Check predictions work
            regimes = detector.predict(sample_features)
            assert len(regimes) == len(sample_features)

    def test_mahalanobis_distance(self, sample_features):
        """Test Mahalanobis distance calculation."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features)

        # Calculate distances
        distances = detector.get_mahalanobis_distance(sample_features)

        assert isinstance(distances, pd.DataFrame)
        assert distances.shape == (len(sample_features), 3)
        assert list(distances.columns) == [
            "distance_regime_0",
            "distance_regime_1",
            "distance_regime_2",
        ]
        assert (distances >= 0).all().all()  # All distances non-negative

    def test_outlier_scores(self, sample_features):
        """Test outlier score calculation."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features)

        # Calculate outlier scores
        scores = detector.get_outlier_scores(
            sample_features, threshold_percentile=95
        )

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_features)
        assert scores.name == "outlier_score"
        assert "threshold" in scores.attrs
        assert "threshold_percentile" in scores.attrs
        assert scores.attrs["threshold_percentile"] == 95

        # Check some outliers exist above threshold
        outliers = scores > scores.attrs["threshold"]
        assert outliers.sum() > 0
        assert outliers.sum() < len(scores) * 0.1  # Less than 10% outliers

    def test_regime_reordering(self, sample_features):
        """Test regime reordering by mean of first feature."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features)

        # Get predictions
        regimes = detector.predict(sample_features)

        # Calculate mean of first feature for each regime
        mean_features = []
        for i in range(3):
            mask = regimes == i
            if mask.sum() > 0:
                mean_features.append(
                    sample_features.loc[mask, "feature1"].mean()
                )
            else:
                mean_features.append(0)

        # Check that regimes are ordered by first feature mean
        assert mean_features[0] <= mean_features[1] <= mean_features[2]

    def test_diagnostics(self, sample_features):
        """Test model diagnostics."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features)

        diagnostics = detector.get_diagnostics()

        assert "regime_counts" in diagnostics
        assert "regime_proportions" in diagnostics
        assert "weights" in diagnostics
        assert "means" in diagnostics
        assert "covariances" in diagnostics
        assert "converged" in diagnostics
        assert "n_iter" in diagnostics
        assert "aic" in diagnostics
        assert "bic" in diagnostics
        assert "log_likelihood" in diagnostics
        assert "silhouette_score" in diagnostics
        assert "davies_bouldin_score" in diagnostics
        assert "calinski_harabasz_score" in diagnostics
        assert "avg_confidence" in diagnostics
        assert "avg_entropy" in diagnostics

        # Check values
        assert len(diagnostics["regime_counts"]) == 3
        assert np.allclose(diagnostics["regime_proportions"].sum(), 1.0)
        assert np.allclose(diagnostics["weights"].sum(), 1.0)
        assert diagnostics["means"].shape == (3, 3)
        assert isinstance(diagnostics["converged"], bool)

    def test_score_methods(self, sample_features):
        """Test scoring methods."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features)

        # Test different metrics
        ll_score = detector.score(sample_features, metric="log_likelihood")
        aic_score = detector.score(sample_features, metric="aic")
        bic_score = detector.score(sample_features, metric="bic")

        assert isinstance(ll_score, float)
        assert isinstance(aic_score, float)
        assert isinstance(bic_score, float)

        # Check relationships
        assert ll_score < 0  # Log-likelihood typically negative
        assert aic_score > 0
        assert bic_score > 0
        assert bic_score > aic_score  # BIC has larger penalty

    def test_parameter_counting(self, sample_features):
        """Test parameter counting for different configurations."""
        detector = GMMRegimeDetector(
            n_regimes=3, covariance_type="full", random_state=42
        )
        detector.fit(sample_features)

        n_params = detector._count_parameters()

        # For 3 regimes, 3 features, full covariance:
        # Mixing weights: 2 (3-1)
        # Means: 9 (3*3)
        # Full covariances: 18 (3 * 3*(3+1)/2)
        # Total: 2 + 9 + 18 = 29
        assert n_params == 29

    def test_different_n_regimes(self, sample_features):
        """Test with different numbers of regimes."""
        for n_regimes in [2, 3, 4, 5]:
            detector = GMMRegimeDetector(n_regimes=n_regimes, random_state=42)
            detector.fit(sample_features)

            regimes = detector.predict(sample_features)

            assert regimes.min() >= 0
            assert regimes.max() < n_regimes
            assert len(np.unique(regimes)) <= n_regimes

    def test_convergence_warning(self, sample_features):
        """Test handling of convergence warnings."""
        # Use very few iterations to likely not converge
        detector = GMMRegimeDetector(
            n_regimes=3, max_iter=1, random_state=42  # Very few iterations
        )

        # Should not raise error even if doesn't converge
        detector.fit(sample_features)
        assert detector.is_fitted

        # Check convergence status in diagnostics
        diagnostics = detector.get_diagnostics()
        # May or may not converge with 1 iteration
        assert isinstance(diagnostics["converged"], bool)

    def test_single_regime_handling(self, sample_features):
        """Test handling of single regime edge case."""
        detector = GMMRegimeDetector(n_regimes=1, random_state=42)
        detector.fit(sample_features)

        regimes = detector.predict(sample_features)

        assert (regimes == 0).all()

        # Silhouette score should be 0 for single cluster
        diagnostics = detector.get_diagnostics()
        assert diagnostics["silhouette_score"] == 0

    def test_init_params_variations(self):
        """Test different initialization parameters."""
        features = pd.DataFrame(np.random.randn(100, 2), columns=["f1", "f2"])

        init_params_list = [
            "kmeans",
            "random",
            "k-means++",
            "random_from_data",
        ]

        for init_params in init_params_list:
            detector = GMMRegimeDetector(
                n_regimes=2, init_params=init_params, random_state=42
            )
            detector.fit(features)

            assert detector.is_fitted

    def test_warm_start(self, sample_features):
        """Test warm start functionality."""
        detector = GMMRegimeDetector(
            n_regimes=3, warm_start=True, random_state=42
        )

        # First fit
        detector.fit(sample_features[:250])
        first_means = detector.model.means_.copy()

        # Second fit with warm start
        detector.fit(sample_features)

        assert detector.is_fitted
        # Means should have changed but started from previous values
        assert not np.allclose(first_means, detector.model.means_)

    def test_regularization(self, sample_features):
        """Test covariance regularization."""
        # Very small regularization
        detector1 = GMMRegimeDetector(
            n_regimes=3, reg_covar=1e-10, random_state=42
        )

        # Larger regularization
        detector2 = GMMRegimeDetector(
            n_regimes=3, reg_covar=1e-3, random_state=42
        )

        detector1.fit(sample_features)
        detector2.fit(sample_features)

        assert detector1.is_fitted
        assert detector2.is_fitted

        # Both should work but might have different covariances
        cov1 = detector1.model.covariances_
        cov2 = detector2.model.covariances_

        # With higher regularization, diagonal elements should be larger
        if detector1.covariance_type == "full":
            assert np.mean(np.diagonal(cov2[0])) >= np.mean(
                np.diagonal(cov1[0])
            )

    def test_multiple_initializations(self, sample_features):
        """Test multiple n_init parameter."""
        detector = GMMRegimeDetector(
            n_regimes=3, n_init=5, random_state=42  # Multiple initializations
        )

        detector.fit(sample_features)

        assert detector.is_fitted
        # Should have selected best model from multiple runs
        assert detector.model is not None

    def test_optimal_n_edge_cases(self, sample_features):
        """Test optimal n selection edge cases."""
        # Test with very small data
        small_features = sample_features.iloc[:20]

        detector = GMMRegimeDetector(n_regimes=3, random_state=42)

        # Should handle small data gracefully
        detector.fit(
            small_features,
            select_optimal_n=True,
            max_components=15,  # Will be limited by data size
        )

        assert detector.is_fitted
        # Should not test more components than data/10
        max_tested = min(15, len(small_features) // 10)
        if max_tested > 2:
            assert detector.optimal_n_analysis is not None

    def test_empty_regime_probabilities(self, sample_features):
        """Test handling of edge case with extreme probabilities."""
        detector = GMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(sample_features)

        # Create extreme features that might produce near-zero probabilities
        extreme_features = pd.DataFrame(
            np.ones((10, 3)) * 1000,
            columns=sample_features.columns,  # Very far from any component
        )

        proba = detector.predict_proba(extreme_features)

        # Should still sum to 1 even for extreme cases
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_separation_metrics_edge_cases(self, sample_features):
        """Test separation metrics with edge cases."""
        # Test with perfect separation (synthetic data)
        perfect_data = []
        for i in range(3):
            cluster = np.random.randn(50, 3) * 0.1 + i * 10  # Well separated
            perfect_data.append(cluster)

        perfect_features = pd.DataFrame(
            np.vstack(perfect_data), columns=["f1", "f2", "f3"]
        )

        detector = GMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(perfect_features)

        diagnostics = detector.get_diagnostics()

        # Should have good separation metrics
        assert diagnostics["silhouette_score"] > 0.5  # Good separation
        assert diagnostics["davies_bouldin_score"] < 1.0  # Low is better
        assert diagnostics["calinski_harabasz_score"] > 100  # High is better
