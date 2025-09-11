"""Test suite for validation framework following TDD approach."""

import pytest
import numpy as np
import pandas as pd
from typing import List, Tuple
from unittest.mock import Mock, patch


class TestCPCV:
    """Test Combinatorial Purged Cross-Validation."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        n_samples = 500

        # Create time series with autocorrelation
        returns = np.random.normal(0, 0.01, n_samples)
        for i in range(1, n_samples):
            returns[i] += 0.3 * returns[i - 1]  # Add autocorrelation

        features = pd.DataFrame(
            {
                "returns": returns,
                "volatility": np.abs(returns)
                + np.random.normal(0, 0.002, n_samples),
            }
        )
        features.index = pd.date_range(
            "2020-01-01", periods=n_samples, freq="D"
        )

        # Create synthetic labels (regimes)
        regimes = np.array([0] * 150 + [1] * 200 + [2] * 150)

        return features, regimes

    def test_cpcv_initialization(self):
        """Test CPCV initialization with various parameters."""
        from marketregimeml.validation import CombinatorialPurgedCV

        # Test default initialization
        cv = CombinatorialPurgedCV()
        assert cv.n_splits == 5
        assert cv.n_test_splits == 2
        assert cv.embargo_td is not None

        # Test custom initialization
        cv = CombinatorialPurgedCV(
            n_splits=10, n_test_splits=3, embargo_td=pd.Timedelta(days=5)
        )
        assert cv.n_splits == 10
        assert cv.n_test_splits == 3
        assert cv.embargo_td == pd.Timedelta(days=5)

    def test_cpcv_split_generation(self, sample_data):
        """Test that CPCV generates correct number of splits."""
        from marketregimeml.validation import CombinatorialPurgedCV

        features, regimes = sample_data
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)

        splits = list(cv.split(features))

        # Should generate C(n_splits, n_test_splits) combinations
        from math import comb

        expected_splits = comb(5, 2)
        assert len(splits) == expected_splits

        # Each split should have train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(np.intersect1d(train_idx, test_idx)) == 0  # No overlap

    def test_purging(self, sample_data):
        """Test that purging removes overlapping samples."""
        from marketregimeml.validation import CombinatorialPurgedCV

        features, regimes = sample_data
        cv = CombinatorialPurgedCV(
            n_splits=5, n_test_splits=2, embargo_td=pd.Timedelta(days=10)
        )

        for train_idx, test_idx in cv.split(features):
            # Check embargo: no training samples within embargo_td of test samples
            train_times = features.index[train_idx]
            test_times = features.index[test_idx]

            for test_time in test_times:
                # Find training samples too close to test
                min_distance = min(abs((train_times - test_time).days))
                assert min_distance >= 10  # Embargo of 10 days

    def test_path_dependence_handling(self):
        """Test handling of path-dependent samples."""
        from marketregimeml.validation import CombinatorialPurgedCV

        # Create data with explicit path dependence
        n_samples = 300
        features = pd.DataFrame({"feature": np.random.randn(n_samples)})
        features.index = pd.date_range(
            "2020-01-01", periods=n_samples, freq="D"
        )

        # Define path-dependent groups (e.g., overlapping windows)
        groups = np.zeros(n_samples)
        for i in range(0, n_samples, 10):
            groups[i : i + 20] = i // 10  # Overlapping groups

        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)

        for train_idx, test_idx in cv.split(features, groups=groups):
            # Check that path-dependent samples are handled correctly
            train_groups = groups[train_idx]
            test_groups = groups[test_idx]

            # No group should appear in both train and test
            common_groups = np.intersect1d(train_groups, test_groups)
            assert len(common_groups) == 0


class TestRegimeAwareSplitting:
    """Test regime-aware train/test splitting."""

    @pytest.fixture
    def regime_data(self):
        """Generate data with clear regime structure."""
        np.random.seed(42)

        # Create three distinct regimes
        regime1_features = np.random.normal(0, 0.5, (100, 3))
        regime2_features = np.random.normal(2, 0.7, (150, 3))
        regime3_features = np.random.normal(-1, 0.3, (100, 3))

        features = np.vstack(
            [regime1_features, regime2_features, regime3_features]
        )
        regimes = np.array([0] * 100 + [1] * 150 + [2] * 100)

        return features, regimes

    def test_regime_aware_split(self, regime_data):
        """Test basic regime-aware splitting."""
        from marketregimeml.validation import RegimeAwareSplitter

        features, regimes = regime_data
        splitter = RegimeAwareSplitter(test_size=0.3, random_state=42)

        train_idx, test_idx = splitter.split(features, regimes)

        # Check split sizes
        assert len(train_idx) + len(test_idx) == len(features)
        assert len(test_idx) / len(features) == pytest.approx(0.3, rel=0.1)

        # Check regime balance in both sets
        train_regimes = regimes[train_idx]
        test_regimes = regimes[test_idx]

        for regime_id in np.unique(regimes):
            train_prop = (train_regimes == regime_id).mean()
            test_prop = (test_regimes == regime_id).mean()
            overall_prop = (regimes == regime_id).mean()

            # Proportions should be similar
            assert abs(train_prop - overall_prop) < 0.1
            assert abs(test_prop - overall_prop) < 0.1

    def test_stratified_regime_split(self, regime_data):
        """Test stratified splitting by regime."""
        from marketregimeml.validation import RegimeAwareSplitter

        features, regimes = regime_data
        splitter = RegimeAwareSplitter(
            test_size=0.2, stratify=True, random_state=42
        )

        train_idx, test_idx = splitter.split(features, regimes)

        # Check that each regime is represented proportionally
        train_regimes = regimes[train_idx]
        test_regimes = regimes[test_idx]

        for regime_id in np.unique(regimes):
            original_ratio = (regimes == regime_id).mean()
            train_ratio = (train_regimes == regime_id).mean()
            test_ratio = (test_regimes == regime_id).mean()

            # Ratios should be very close (within 5%)
            assert abs(train_ratio - original_ratio) < 0.05
            assert abs(test_ratio - original_ratio) < 0.05

    def test_temporal_regime_split(self):
        """Test temporal splitting that respects regime boundaries."""
        from marketregimeml.validation import RegimeAwareSplitter

        # Create data with temporal regime structure
        n_samples = 400
        features = pd.DataFrame({"feature": np.random.randn(n_samples)})
        features.index = pd.date_range(
            "2020-01-01", periods=n_samples, freq="D"
        )

        # Regimes change over time
        regimes = np.array([0] * 100 + [1] * 100 + [2] * 100 + [0] * 100)

        splitter = RegimeAwareSplitter(
            test_size=0.25, temporal=True, min_regime_samples=20
        )

        train_idx, test_idx = splitter.split(features, regimes)

        # Test should be at the end (temporal)
        assert max(train_idx) < min(test_idx)

        # Each regime should have minimum samples in train
        train_regimes = regimes[train_idx]
        for regime_id in np.unique(regimes):
            regime_count = (train_regimes == regime_id).sum()
            if regime_count > 0:
                assert regime_count >= 20


class TestDeflatedSharpe:
    """Test Deflated Sharpe Ratio calculation."""

    def test_deflated_sharpe_calculation(self):
        """Test basic Deflated Sharpe Ratio calculation."""
        from marketregimeml.validation import calculate_deflated_sharpe

        # Generate returns with known Sharpe ratio
        np.random.seed(42)
        n_samples = 252  # One year of daily returns
        returns = np.random.normal(0.0005, 0.01, n_samples)  # ~0.8 Sharpe

        # Calculate deflated Sharpe
        dsr = calculate_deflated_sharpe(
            returns, n_trials=100, expected_sharpe=0.0
        )

        assert isinstance(dsr, dict)
        assert "deflated_sharpe" in dsr
        assert "original_sharpe" in dsr
        assert "p_value" in dsr

        # Original Sharpe should be positive
        assert dsr["original_sharpe"] > 0

        # P-value should be between 0 and 1
        assert 0 <= dsr["p_value"] <= 1

    def test_multiple_testing_correction(self):
        """Test correction for multiple testing."""
        from marketregimeml.validation import calculate_deflated_sharpe

        np.random.seed(42)

        # Generate multiple strategy returns
        n_strategies = 10
        n_samples = 252

        all_returns = []
        for _ in range(n_strategies):
            returns = np.random.normal(0.0001, 0.01, n_samples)
            all_returns.append(returns)

        # Best strategy (cherry-picked)
        sharpe_ratios = [
            r.mean() / r.std() * np.sqrt(252) for r in all_returns
        ]
        best_idx = np.argmax(sharpe_ratios)
        best_returns = all_returns[best_idx]

        # Calculate deflated Sharpe with multiple testing correction
        dsr = calculate_deflated_sharpe(
            best_returns, n_trials=n_strategies, expected_sharpe=0.0
        )

        # Deflated Sharpe should be lower than original due to selection bias
        assert dsr["deflated_sharpe"] < dsr["original_sharpe"]

    def test_regime_adjusted_sharpe(self):
        """Test Sharpe ratio adjusted for regime changes."""
        from marketregimeml.validation import calculate_regime_adjusted_sharpe

        np.random.seed(42)

        # Create returns with regime-dependent characteristics
        bull_returns = np.random.normal(0.002, 0.008, 100)
        bear_returns = np.random.normal(-0.001, 0.02, 100)

        returns = np.concatenate([bull_returns, bear_returns])
        regimes = np.array([0] * 100 + [1] * 100)

        # Calculate regime-adjusted Sharpe
        adjusted_sharpe = calculate_regime_adjusted_sharpe(returns, regimes)

        assert isinstance(adjusted_sharpe, dict)
        assert "overall_sharpe" in adjusted_sharpe
        assert "regime_sharpes" in adjusted_sharpe
        assert "weighted_sharpe" in adjusted_sharpe

        # Should have Sharpe for each regime
        assert len(adjusted_sharpe["regime_sharpes"]) == 2


class TestEmbargoLogic:
    """Test embargo functionality for preventing data leakage."""

    def test_embargo_periods(self):
        """Test that embargo periods are correctly applied."""
        from marketregimeml.validation import apply_embargo

        # Create time series data
        n_samples = 100
        index = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        # Define test indices
        test_indices = [20, 21, 22, 50, 51, 52]

        # Apply embargo of 5 days
        embargo_td = pd.Timedelta(days=5)
        train_indices = list(range(n_samples))

        purged_train = apply_embargo(
            train_indices, test_indices, index, embargo_td
        )

        # Check that embargo is applied
        for test_idx in test_indices:
            test_time = index[test_idx]

            for train_idx in purged_train:
                train_time = index[train_idx]
                time_diff = abs((train_time - test_time).days)

                # Should be at least embargo_td away
                assert time_diff >= 5

    def test_embargo_with_events(self):
        """Test embargo with event-based data."""
        from marketregimeml.validation import apply_embargo

        # Create event data with start and end times
        events = pd.DataFrame(
            {
                "start": pd.date_range("2020-01-01", periods=50, freq="2D"),
                "end": pd.date_range("2020-01-03", periods=50, freq="2D"),
            }
        )

        # Test events
        test_events = [10, 20, 30]
        train_events = list(range(50))

        # Apply embargo considering event duration
        purged_train = apply_embargo(
            train_events, test_events, events, embargo_td=pd.Timedelta(days=3)
        )

        # Verify embargo
        for test_idx in test_events:
            test_start = events.iloc[test_idx]["start"]
            test_end = events.iloc[test_idx]["end"]

            for train_idx in purged_train:
                train_start = events.iloc[train_idx]["start"]
                train_end = events.iloc[train_idx]["end"]

                # Check for overlap + embargo
                assert train_end < test_start - pd.Timedelta(
                    days=3
                ) or train_start > test_end + pd.Timedelta(days=3)


class TestWalkForwardValidation:
    """Test walk-forward validation for time series."""

    def test_walk_forward_basic(self):
        """Test basic walk-forward validation."""
        from marketregimeml.validation import WalkForwardCV

        # Create time series
        n_samples = 365
        data = pd.DataFrame({"feature": np.random.randn(n_samples)})
        data.index = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        # Walk-forward with expanding window
        wf = WalkForwardCV(
            n_splits=5, train_period=60, test_period=30, expanding=True
        )

        splits = list(wf.split(data))

        assert len(splits) == 5

        # Check expanding window
        prev_train_size = 0
        for i, (train_idx, test_idx) in enumerate(splits):
            if i > 0:
                # Train set should expand
                assert len(train_idx) > prev_train_size
            prev_train_size = len(train_idx)

            # Test set should be fixed size
            assert len(test_idx) == 30

    def test_walk_forward_rolling(self):
        """Test rolling walk-forward validation."""
        from marketregimeml.validation import WalkForwardCV

        n_samples = 365
        data = pd.DataFrame({"feature": np.random.randn(n_samples)})
        data.index = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        # Walk-forward with rolling window
        wf = WalkForwardCV(
            n_splits=5, train_period=60, test_period=30, expanding=False
        )

        splits = list(wf.split(data))

        # Check rolling window
        for train_idx, test_idx in splits:
            # Train set should be fixed size
            assert len(train_idx) == 60
            # Test set should be fixed size
            assert len(test_idx) == 30
            # Test should come after train
            assert min(test_idx) > max(train_idx)

    def test_walk_forward_with_gap(self):
        """Test walk-forward with gap between train and test."""
        from marketregimeml.validation import WalkForwardCV

        n_samples = 365
        data = pd.DataFrame({"feature": np.random.randn(n_samples)})
        data.index = pd.date_range("2020-01-01", periods=n_samples, freq="D")

        # Walk-forward with gap
        wf = WalkForwardCV(
            n_splits=3,
            train_period=60,
            test_period=30,
            gap=10,
            expanding=False,
        )

        splits = list(wf.split(data))

        for train_idx, test_idx in splits:
            # Check gap between train and test
            gap_size = min(test_idx) - max(train_idx) - 1
            assert gap_size == 10


class TestValidationMetrics:
    """Test validation-specific metrics."""

    def test_stability_across_folds(self):
        """Test metric stability across CV folds."""
        from marketregimeml.validation import calculate_cv_stability

        # Simulate metrics across folds
        fold_metrics = {
            "accuracy": [0.85, 0.83, 0.86, 0.84, 0.87],
            "sharpe": [1.2, 1.1, 1.3, 1.15, 1.25],
            "max_dd": [-0.15, -0.18, -0.14, -0.16, -0.17],
        }

        stability = calculate_cv_stability(fold_metrics)

        assert isinstance(stability, dict)

        for metric_name in fold_metrics:
            assert metric_name in stability

            # Should have mean, std, cv
            assert "mean" in stability[metric_name]
            assert "std" in stability[metric_name]
            assert "cv" in stability[metric_name]  # Coefficient of variation

            # CV should be std/mean
            expected_cv = stability[metric_name]["std"] / abs(
                stability[metric_name]["mean"]
            )
            assert abs(stability[metric_name]["cv"] - expected_cv) < 0.001

    def test_regime_validation_metrics(self):
        """Test regime-specific validation metrics."""
        from marketregimeml.validation import (
            calculate_regime_validation_metrics,
        )

        # Create predictions and true values by regime
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 2, 2])
        regimes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        metrics = calculate_regime_validation_metrics(y_true, y_pred, regimes)

        assert isinstance(metrics, dict)

        # Should have metrics for each regime
        for regime_id in np.unique(regimes):
            assert regime_id in metrics
            assert "accuracy" in metrics[regime_id]
            assert "precision" in metrics[regime_id]
            assert "recall" in metrics[regime_id]

        # Should have overall metrics
        assert "overall" in metrics
