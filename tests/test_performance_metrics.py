"""Tests for additional performance metrics."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from marketregimeml.evaluation.performance import (
    PerformanceEvaluator,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_information_ratio,
    calculate_regime_based_returns,
    calculate_regime_transition_returns,
    backtest_regime_strategy,
)


class TestPerformanceMetrics:
    """Test performance metrics for trading strategies."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        np.random.seed(42)
        # Generate returns with different characteristics
        bull_returns = np.random.normal(0.001, 0.01, 100)  # Positive bias
        bear_returns = np.random.normal(-0.0005, 0.015, 100)  # Negative bias
        neutral_returns = np.random.normal(0, 0.008, 100)  # No bias

        returns = np.concatenate([bull_returns, bear_returns, neutral_returns])
        dates = pd.date_range("2020-01-01", periods=300, freq="D")

        return pd.Series(returns, index=dates, name="returns")

    @pytest.fixture
    def sample_regimes(self):
        """Create sample regime predictions."""
        # Corresponding regimes: 0=bull, 1=bear, 2=neutral
        regimes = np.array([0] * 100 + [1] * 100 + [2] * 100)
        return regimes

    @pytest.fixture
    def sample_prices(self, sample_returns):
        """Create price series from returns."""
        prices = (1 + sample_returns).cumprod() * 100
        return prices

    def test_calculate_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = calculate_sharpe_ratio(sample_returns)

        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range

        # Test with different frequencies
        sharpe_monthly = calculate_sharpe_ratio(sample_returns, periods=21)
        assert sharpe_monthly != sharpe

        # Test with custom risk-free rate
        sharpe_rf = calculate_sharpe_ratio(
            sample_returns, risk_free_rate=0.02 / 252
        )
        assert (
            sharpe_rf < sharpe
        )  # Should be lower with positive risk-free rate

    def test_calculate_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = calculate_sortino_ratio(sample_returns)

        assert isinstance(sortino, float)
        assert -5 < sortino < 5

        # Sortino should generally be higher than Sharpe for same returns
        sharpe = calculate_sharpe_ratio(sample_returns)
        # This isn't always true but often is
        if sample_returns.mean() > 0:
            assert sortino >= sharpe

    def test_calculate_calmar_ratio(self, sample_returns):
        """Test Calmar ratio calculation."""
        calmar = calculate_calmar_ratio(sample_returns)

        assert isinstance(calmar, float)
        assert -10 < calmar < 10

        # Test with price series
        prices = (1 + sample_returns).cumprod() * 100
        calmar_prices = calculate_calmar_ratio(prices, is_prices=True)
        assert isinstance(calmar_prices, float)

    def test_calculate_max_drawdown(self, sample_returns):
        """Test maximum drawdown calculation."""
        max_dd, dd_duration = calculate_max_drawdown(sample_returns)

        assert isinstance(max_dd, float)
        assert isinstance(dd_duration, int)
        assert 0 <= max_dd <= 1  # Drawdown is between 0 and 100%
        assert dd_duration >= 0

        # Test with price series
        prices = (1 + sample_returns).cumprod() * 100
        max_dd_prices, _ = calculate_max_drawdown(prices, is_prices=True)
        assert 0 <= max_dd_prices <= 1

    def test_calculate_win_rate(self, sample_returns):
        """Test win rate calculation."""
        win_rate = calculate_win_rate(sample_returns)

        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1

        # Test with threshold
        win_rate_threshold = calculate_win_rate(
            sample_returns, threshold=0.001
        )
        assert (
            win_rate_threshold <= win_rate
        )  # Should be lower with higher threshold

    def test_calculate_profit_factor(self, sample_returns):
        """Test profit factor calculation."""
        profit_factor = calculate_profit_factor(sample_returns)

        assert isinstance(profit_factor, float)
        assert profit_factor >= 0

        # Test edge cases
        all_positive = pd.Series([0.01, 0.02, 0.03])
        pf_positive = calculate_profit_factor(all_positive)
        assert pf_positive == float("inf")

        all_negative = pd.Series([-0.01, -0.02, -0.03])
        pf_negative = calculate_profit_factor(all_negative)
        assert pf_negative == 0

    def test_calculate_information_ratio(self, sample_returns):
        """Test information ratio calculation."""
        # Create benchmark returns
        benchmark = sample_returns * 0.8 + np.random.normal(
            0, 0.002, len(sample_returns)
        )

        ir = calculate_information_ratio(sample_returns, benchmark)

        assert isinstance(ir, float)
        assert -5 < ir < 5

    def test_calculate_regime_based_returns(
        self, sample_returns, sample_regimes
    ):
        """Test regime-based return calculation."""
        regime_returns = calculate_regime_based_returns(
            sample_returns, sample_regimes
        )

        assert isinstance(regime_returns, dict)
        assert len(regime_returns) == 3  # Three regimes

        for regime, metrics in regime_returns.items():
            assert "mean_return" in metrics
            assert "total_return" in metrics
            assert "volatility" in metrics
            assert "sharpe_ratio" in metrics
            assert "win_rate" in metrics
            assert "max_drawdown" in metrics

            # Check value ranges
            assert -1 < metrics["mean_return"] < 1
            assert metrics["volatility"] >= 0
            assert 0 <= metrics["win_rate"] <= 1

    def test_calculate_regime_transition_returns(
        self, sample_returns, sample_regimes
    ):
        """Test regime transition return analysis."""
        transition_returns = calculate_regime_transition_returns(
            sample_returns, sample_regimes, periods_forward=5
        )

        assert isinstance(transition_returns, pd.DataFrame)

        # Should have entry for each unique transition
        n_regimes = len(np.unique(sample_regimes))
        assert len(transition_returns) <= n_regimes * n_regimes

        # Check columns
        expected_cols = [
            "from_regime",
            "to_regime",
            "count",
            "avg_return",
            "total_return",
            "volatility",
        ]
        for col in expected_cols:
            assert col in transition_returns.columns

    def test_backtest_regime_strategy(self, sample_prices, sample_regimes):
        """Test regime-based trading strategy backtest."""
        # Define simple strategy: long in regime 0, short in regime 1, flat in regime 2
        position_map = {0: 1.0, 1: -1.0, 2: 0.0}

        backtest_results = backtest_regime_strategy(
            sample_prices,
            sample_regimes,
            position_map=position_map,
            transaction_cost=0.001,
        )

        assert isinstance(backtest_results, dict)

        # Check required keys
        required_keys = [
            "returns",
            "cumulative_returns",
            "positions",
            "trades",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "num_trades",
        ]
        for key in required_keys:
            assert key in backtest_results

        # Check data types and ranges
        assert isinstance(backtest_results["returns"], pd.Series)
        assert isinstance(backtest_results["cumulative_returns"], pd.Series)
        assert isinstance(backtest_results["positions"], np.ndarray)
        assert backtest_results["num_trades"] >= 0
        assert 0 <= backtest_results["win_rate"] <= 1


class TestPerformanceEvaluator:
    """Test PerformanceEvaluator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        np.random.seed(42)
        n_periods = 200

        # Create price data
        returns = np.random.normal(0.0005, 0.01, n_periods)
        prices = pd.Series(
            (1 + returns).cumprod() * 100,
            index=pd.date_range("2020-01-01", periods=n_periods, freq="D"),
        )

        # Create regime predictions
        regimes = np.random.choice([0, 1, 2], n_periods, p=[0.5, 0.3, 0.2])

        # Create regime probabilities
        probabilities = np.random.dirichlet([2, 2, 2], n_periods)

        return {
            "prices": prices,
            "returns": prices.pct_change().dropna(),
            "regimes": regimes,
            "probabilities": probabilities,
        }

    @pytest.fixture
    def evaluator(self):
        """Create PerformanceEvaluator instance."""
        return PerformanceEvaluator()

    def test_init(self, evaluator):
        """Test PerformanceEvaluator initialization."""
        assert evaluator is not None
        assert hasattr(evaluator, "metrics_history")
        assert evaluator.metrics_history == []

    def test_evaluate_strategy(self, evaluator, sample_data):
        """Test strategy evaluation."""
        results = evaluator.evaluate_strategy(
            sample_data["prices"],
            sample_data["regimes"],
            position_map={0: 1.0, 1: -0.5, 2: 0.0},
        )

        assert isinstance(results, dict)
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "regime_returns" in results

        # Check that metrics were stored
        assert len(evaluator.metrics_history) == 1

    def test_compare_strategies(self, evaluator, sample_data):
        """Test strategy comparison."""
        strategies = {
            "long_only": {0: 1.0, 1: 1.0, 2: 1.0},
            "regime_based": {0: 1.0, 1: -1.0, 2: 0.0},
            "conservative": {0: 0.5, 1: 0.0, 2: 0.0},
        }

        comparison = evaluator.compare_strategies(
            sample_data["prices"], sample_data["regimes"], strategies
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == len(strategies)
        assert "total_return" in comparison.columns
        assert "sharpe_ratio" in comparison.columns

        # Check that all strategies were evaluated
        for strategy_name in strategies:
            assert strategy_name in comparison.index

    def test_calculate_confidence_weighted_returns(
        self, evaluator, sample_data
    ):
        """Test confidence-weighted return calculation."""
        weighted_returns = evaluator.calculate_confidence_weighted_returns(
            sample_data["returns"],
            sample_data["probabilities"],
            sample_data["regimes"],
        )

        assert isinstance(weighted_returns, pd.Series)
        assert len(weighted_returns) == len(sample_data["returns"])

        # Weighted returns should be different from original
        assert not np.allclose(
            weighted_returns.values, sample_data["returns"].values
        )

    def test_regime_performance_summary(self, evaluator, sample_data):
        """Test regime performance summary."""
        summary = evaluator.regime_performance_summary(
            sample_data["returns"], sample_data["regimes"]
        )

        assert isinstance(summary, pd.DataFrame)

        # Should have one row per regime
        unique_regimes = np.unique(sample_data["regimes"])
        assert len(summary) == len(unique_regimes)

        # Check columns
        expected_cols = [
            "count",
            "mean_return",
            "volatility",
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
        ]
        for col in expected_cols:
            assert col in summary.columns

    def test_calculate_regime_timing_value(self, evaluator, sample_data):
        """Test regime timing value calculation."""
        # Perfect prediction
        perfect_value = evaluator.calculate_regime_timing_value(
            sample_data["returns"],
            sample_data["regimes"],
            sample_data["regimes"],  # Perfect prediction
        )

        assert isinstance(perfect_value, float)
        assert perfect_value >= 0  # Should be positive for perfect prediction

        # Random prediction
        random_regimes = np.random.choice(
            [0, 1, 2], len(sample_data["regimes"])
        )
        random_value = evaluator.calculate_regime_timing_value(
            sample_data["returns"], sample_data["regimes"], random_regimes
        )

        assert perfect_value > random_value  # Perfect should beat random

    def test_plot_performance(self, evaluator, sample_data):
        """Test performance plotting."""
        with patch("matplotlib.pyplot.show"):
            fig = evaluator.plot_performance(
                sample_data["prices"], sample_data["regimes"], show=False
            )

            assert fig is not None
            assert len(fig.axes) >= 2  # Should have multiple subplots
