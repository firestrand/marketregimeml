"""Tests for visualization module."""

import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock

from marketregimeml.visualization.plots import (
    RegimePlotter,
    plot_regime_transitions,
    plot_regime_probabilities,
    plot_price_with_regimes,
    plot_regime_duration_distribution,
    plot_transition_matrix_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_regime_returns,
)


class TestRegimePlotter:
    """Test RegimePlotter class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price and regime data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create price data
        returns = np.random.randn(100) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))
        price_data = pd.Series(prices, index=dates, name="price")

        # Create regime data
        regimes = np.random.choice([0, 1, 2], size=100, p=[0.5, 0.3, 0.2])

        # Create regime probabilities
        probabilities = np.random.dirichlet([2, 2, 2], size=100)

        return {
            "prices": price_data,
            "regimes": regimes,
            "probabilities": probabilities,
            "dates": dates,
        }

    @pytest.fixture
    def plotter(self):
        """Create RegimePlotter instance."""
        return RegimePlotter()

    def test_init(self, plotter):
        """Test RegimePlotter initialization."""
        assert plotter is not None
        assert hasattr(plotter, "style")
        assert hasattr(plotter, "figsize")
        assert hasattr(plotter, "regime_colors")
        assert hasattr(plotter, "regime_labels")

    def test_custom_init(self):
        """Test RegimePlotter with custom parameters."""
        custom_colors = ["red", "green", "blue"]
        custom_labels = ["Bear", "Neutral", "Bull"]
        plotter = RegimePlotter(
            style="ggplot",  # Use a valid matplotlib style
            figsize=(12, 8),
            regime_colors=custom_colors,
            regime_labels=custom_labels,
        )

        assert plotter.style == "ggplot"
        assert plotter.figsize == (12, 8)
        assert plotter.regime_colors == custom_colors
        assert plotter.regime_labels == custom_labels

    @patch("matplotlib.pyplot.show")
    def test_plot_regimes(self, mock_show, plotter, sample_data):
        """Test plot_regimes method."""
        fig = plotter.plot_regimes(
            sample_data["prices"], sample_data["regimes"]
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1

        # Check that data was plotted
        ax = fig.axes[0]
        assert len(ax.lines) > 0 or len(ax.collections) > 0

    @patch("matplotlib.pyplot.show")
    def test_plot_probabilities(self, mock_show, plotter, sample_data):
        """Test plot_probabilities method."""
        fig = plotter.plot_probabilities(
            sample_data["probabilities"], sample_data["dates"]
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1

    @patch("matplotlib.pyplot.show")
    def test_plot_transitions(self, mock_show, plotter, sample_data):
        """Test plot_transitions method."""
        # Create transition matrix
        trans_matrix = np.array(
            [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
        )

        fig = plotter.plot_transitions(trans_matrix)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1

    @patch("matplotlib.pyplot.show")
    def test_plot_duration_distribution(self, mock_show, plotter, sample_data):
        """Test plot_duration_distribution method."""
        durations = {0: [5, 10, 8, 12, 6], 1: [3, 4, 6, 5], 2: [2, 3, 4, 3, 5]}

        fig = plotter.plot_duration_distribution(durations)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1

    @patch("matplotlib.pyplot.show")
    def test_plot_feature_importance(self, mock_show, plotter):
        """Test plot_feature_importance method."""
        importances = pd.Series(
            {
                "volatility": 0.3,
                "returns": 0.25,
                "volume": 0.2,
                "rsi": 0.15,
                "macd": 0.1,
            }
        )

        fig = plotter.plot_feature_importance(importances)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1

    @patch("matplotlib.pyplot.show")
    def test_plot_model_comparison(self, mock_show, plotter):
        """Test plot_model_comparison method."""
        metrics = pd.DataFrame(
            {
                "HMM": [0.85, 0.82, 0.78],
                "GMM": [0.83, 0.80, 0.76],
                "SVM": [0.88, 0.85, 0.82],
            },
            index=["Accuracy", "Precision", "Recall"],
        )

        fig = plotter.plot_model_comparison(metrics)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1

    @patch("matplotlib.pyplot.show")
    def test_plot_returns_by_regime(self, mock_show, plotter, sample_data):
        """Test plot_returns_by_regime method."""
        returns = pd.Series(
            np.random.randn(100) * 0.01, index=sample_data["dates"]
        )

        fig = plotter.plot_returns_by_regime(returns, sample_data["regimes"])

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1

    def test_save_figure(self, plotter, sample_data, tmp_path):
        """Test saving figures."""
        fig = plotter.plot_regimes(
            sample_data["prices"], sample_data["regimes"], show=False
        )

        # Save to file
        save_path = tmp_path / "test_plot.png"
        plotter.save_figure(fig, save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestStandalonePlotFunctions:
    """Test standalone plotting functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(100) * 0.01)), index=dates
        )
        regimes = np.random.choice([0, 1, 2], size=100)
        probabilities = np.random.dirichlet([2, 2, 2], size=100)

        return prices, regimes, probabilities, dates

    @patch("matplotlib.pyplot.show")
    def test_plot_regime_transitions(self, mock_show, sample_data):
        """Test plot_regime_transitions function."""
        _, regimes, _, dates = sample_data

        fig = plot_regime_transitions(regimes, dates)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    @patch("matplotlib.pyplot.show")
    def test_plot_regime_probabilities(self, mock_show, sample_data):
        """Test plot_regime_probabilities function."""
        _, _, probabilities, dates = sample_data

        fig = plot_regime_probabilities(probabilities, dates)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    @patch("matplotlib.pyplot.show")
    def test_plot_price_with_regimes(self, mock_show, sample_data):
        """Test plot_price_with_regimes function."""
        prices, regimes, _, _ = sample_data

        fig = plot_price_with_regimes(prices, regimes)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # Should have price and regime subplots

    @patch("matplotlib.pyplot.show")
    def test_plot_regime_duration_distribution(self, mock_show, sample_data):
        """Test plot_regime_duration_distribution function."""
        _, regimes, _, _ = sample_data

        fig = plot_regime_duration_distribution(regimes)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    @patch("matplotlib.pyplot.show")
    def test_plot_transition_matrix_heatmap(self, mock_show):
        """Test plot_transition_matrix_heatmap function."""
        trans_matrix = np.array(
            [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
        )

        fig = plot_transition_matrix_heatmap(trans_matrix)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    @patch("matplotlib.pyplot.show")
    def test_plot_feature_importance(self, mock_show):
        """Test plot_feature_importance function."""
        importances = {
            "volatility": 0.3,
            "returns": 0.25,
            "volume": 0.2,
            "rsi": 0.15,
            "macd": 0.1,
        }

        fig = plot_feature_importance(importances)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    @patch("matplotlib.pyplot.show")
    def test_plot_model_comparison(self, mock_show):
        """Test plot_model_comparison function."""
        metrics = {
            "HMM": {"accuracy": 0.85, "precision": 0.82},
            "GMM": {"accuracy": 0.83, "precision": 0.80},
            "SVM": {"accuracy": 0.88, "precision": 0.85},
        }

        fig = plot_model_comparison(metrics)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    @patch("matplotlib.pyplot.show")
    def test_plot_regime_returns(self, mock_show, sample_data):
        """Test plot_regime_returns function."""
        prices, regimes, _, _ = sample_data
        returns = prices.pct_change().dropna()

        fig = plot_regime_returns(returns, regimes[1:])

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_plot_with_custom_params(self, sample_data):
        """Test plots with custom parameters."""
        prices, regimes, _, _ = sample_data

        # Test with custom colors and labels
        fig = plot_price_with_regimes(
            prices,
            regimes,
            regime_colors=["red", "yellow", "green"],
            regime_labels=["Bear", "Neutral", "Bull"],
            figsize=(15, 10),
            show=False,
        )

        assert fig is not None
        assert fig.get_size_inches()[0] == 15
        assert fig.get_size_inches()[1] == 10

    def test_error_handling(self):
        """Test error handling in plot functions."""
        # Test with invalid data
        with pytest.raises(ValueError):
            plot_price_with_regimes([], [])

        # Test with mismatched data lengths
        prices = pd.Series([100, 101, 102])
        regimes = np.array([0, 1])

        with pytest.raises(ValueError):
            plot_price_with_regimes(prices, regimes)
