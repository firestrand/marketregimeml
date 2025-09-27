"""Visualization plots for market regime analysis."""

from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class RegimePlotter:
    """Class for creating regime visualization plots."""

    def __init__(
        self,
        style: str = "default",
        figsize: Tuple[int, int] = (12, 6),
        regime_colors: Optional[List[str]] = None,
        regime_labels: Optional[List[str]] = None,
    ):
        """Initialize RegimePlotter.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
            regime_colors: Colors for each regime
            regime_labels: Labels for each regime
        """
        self.style = style
        self.figsize = figsize
        self.regime_colors = regime_colors or [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
        ]
        self.regime_labels = regime_labels or [f"Regime {i}" for i in range(5)]

        # Set style
        if style != "default":
            plt.style.use(style)

    def plot_regimes(
        self,
        prices: pd.Series,
        regimes: np.ndarray,
        title: str = "Price with Regime Overlay",
        show: bool = True,
    ) -> Figure:
        """Plot prices with regime overlay.

        Args:
            prices: Price series
            regimes: Regime labels
            title: Plot title
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=self.figsize, height_ratios=[3, 1], sharex=True
        )

        # Plot price
        ax1.plot(prices.index, prices.values, color="black", linewidth=1)
        ax1.set_ylabel("Price")
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)

        # Add regime coloring
        for i in range(len(prices)):
            if i > 0:
                regime = regimes[i - 1] if i - 1 < len(regimes) else regimes[-1]
                ax1.axvspan(
                    prices.index[i - 1],
                    prices.index[i],
                    alpha=0.2,
                    color=self.regime_colors[regime],
                )

        # Plot regime indicator
        ax2.plot(
            prices.index[: len(regimes)],
            regimes,
            drawstyle="steps-post",
            linewidth=2,
        )
        ax2.set_ylabel("Regime")
        ax2.set_xlabel("Date")
        ax2.set_ylim(-0.5, max(regimes) + 0.5)
        ax2.grid(True, alpha=0.3)

        # Add legend
        patches = [
            mpatches.Patch(
                color=self.regime_colors[i],
                label=self.regime_labels[i],
                alpha=0.5,
            )
            for i in range(len(set(regimes)))
        ]
        ax1.legend(handles=patches, loc="upper left")

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_probabilities(
        self,
        probabilities: np.ndarray,
        dates: pd.DatetimeIndex,
        title: str = "Regime Probabilities",
        show: bool = True,
    ) -> Figure:
        """Plot regime probabilities over time.

        Args:
            probabilities: Regime probabilities (n_samples, n_regimes)
            dates: Date index
            title: Plot title
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        n_regimes = probabilities.shape[1]

        # Create stacked area plot
        ax.stackplot(
            dates[: len(probabilities)],
            *probabilities.T,
            labels=[self.regime_labels[i] for i in range(n_regimes)],
            colors=self.regime_colors[:n_regimes],
            alpha=0.7,
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_transitions(
        self,
        transition_matrix: np.ndarray,
        title: str = "Regime Transition Matrix",
        show: bool = True,
    ) -> Figure:
        """Plot transition matrix as heatmap.

        Args:
            transition_matrix: Transition probability matrix
            title: Plot title
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        n_regimes = transition_matrix.shape[0]

        # Create heatmap
        im = ax.imshow(transition_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Probability")

        # Set ticks and labels
        ax.set_xticks(range(n_regimes))
        ax.set_yticks(range(n_regimes))
        ax.set_xticklabels([self.regime_labels[i] for i in range(n_regimes)])
        ax.set_yticklabels([self.regime_labels[i] for i in range(n_regimes)])

        # Add text annotations
        for i in range(n_regimes):
            for j in range(n_regimes):
                _ = ax.text(
                    j,
                    i,
                    f"{transition_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        ax.set_xlabel("To Regime")
        ax.set_ylabel("From Regime")
        ax.set_title(title)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_duration_distribution(
        self,
        durations: Dict[int, List[int]],
        title: str = "Regime Duration Distribution",
        show: bool = True,
    ) -> Figure:
        """Plot distribution of regime durations.

        Args:
            durations: Dictionary mapping regime to list of durations
            title: Plot title
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, len(durations), figsize=(5 * len(durations), 4))

        if len(durations) == 1:
            axes = [axes]

        for idx, (regime, dur_list) in enumerate(durations.items()):
            ax = axes[idx]
            ax.hist(
                dur_list,
                bins=15,
                color=self.regime_colors[regime],
                alpha=0.7,
                edgecolor="black",
            )
            ax.set_xlabel("Duration (periods)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{self.regime_labels[regime]}")
            ax.grid(True, alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_feature_importance(
        self,
        importances: Union[pd.Series, Dict],
        title: str = "Feature Importance",
        show: bool = True,
    ) -> Figure:
        """Plot feature importances.

        Args:
            importances: Feature importances
            title: Plot title
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        if isinstance(importances, dict):
            importances = pd.Series(importances)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Sort importances
        importances = importances.sort_values(ascending=True)

        # Create horizontal bar plot
        ax.barh(range(len(importances)), importances.values, color="steelblue")
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(importances.index)
        ax.set_xlabel("Importance")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_model_comparison(
        self,
        metrics: Union[pd.DataFrame, Dict],
        title: str = "Model Comparison",
        show: bool = True,
    ) -> Figure:
        """Plot model comparison metrics.

        Args:
            metrics: Model metrics (models as columns, metrics as rows)
            title: Plot title
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        if isinstance(metrics, dict):
            # Convert nested dict to DataFrame
            metrics = pd.DataFrame(metrics)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create grouped bar plot
        x = np.arange(len(metrics.index))
        width = 0.8 / len(metrics.columns)

        for i, col in enumerate(metrics.columns):
            offset = (i - len(metrics.columns) / 2 + 0.5) * width
            ax.bar(x + offset, metrics[col], width, label=col)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.index)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_returns_by_regime(
        self,
        returns: pd.Series,
        regimes: np.ndarray,
        title: str = "Returns Distribution by Regime",
        show: bool = True,
    ) -> Figure:
        """Plot return distributions by regime.

        Args:
            returns: Return series
            regimes: Regime labels
            title: Plot title
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Group returns by regime
        regime_returns = {}
        for regime in np.unique(regimes):
            mask = regimes == regime
            regime_returns[regime] = returns.iloc[: len(regimes)][mask]

        # Create box plot
        data = [regime_returns[r] for r in sorted(regime_returns.keys())]
        labels = [self.regime_labels[r] for r in sorted(regime_returns.keys())]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Color boxes
        for patch, regime in zip(bp["boxes"], sorted(regime_returns.keys())):
            patch.set_facecolor(self.regime_colors[regime])
            patch.set_alpha(0.7)

        ax.set_xlabel("Regime")
        ax.set_ylabel("Returns")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        # Add horizontal line at zero
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def save_figure(
        self, fig: Figure, filepath: Union[str, Path], dpi: int = 100
    ) -> None:
        """Save figure to file.

        Args:
            fig: Figure to save
            filepath: Output filepath
            dpi: Resolution
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")


# Standalone functions for convenience
def plot_regime_transitions(
    regimes: np.ndarray, dates: Optional[pd.DatetimeIndex] = None, **kwargs
) -> Figure:
    """Plot regime transitions over time.

    Args:
        regimes: Regime labels
        dates: Date index (optional)
        **kwargs: Additional arguments for RegimePlotter

    Returns:
        Matplotlib figure
    """
    if dates is None:
        dates = pd.date_range("2020-01-01", periods=len(regimes), freq="D")

    # Create dummy price data
    prices = pd.Series(100 * np.ones(len(dates)), index=dates)

    plotter = RegimePlotter(**kwargs)
    return plotter.plot_regimes(prices, regimes, title="Regime Transitions")


def plot_regime_probabilities(
    probabilities: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    **kwargs,
) -> Figure:
    """Plot regime probabilities.

    Args:
        probabilities: Regime probabilities
        dates: Date index (optional)
        **kwargs: Additional arguments for RegimePlotter

    Returns:
        Matplotlib figure
    """
    if dates is None:
        dates = pd.date_range("2020-01-01", periods=len(probabilities), freq="D")

    plotter = RegimePlotter(**kwargs)
    return plotter.plot_probabilities(probabilities, dates)


def plot_price_with_regimes(
    prices: pd.Series,
    regimes: np.ndarray,
    regime_colors: Optional[List[str]] = None,
    regime_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    show: bool = True,
) -> Figure:
    """Plot prices with regime overlay.

    Args:
        prices: Price series
        regimes: Regime labels
        regime_colors: Colors for regimes
        regime_labels: Labels for regimes
        figsize: Figure size
        show: Whether to show the plot

    Returns:
        Matplotlib figure
    """
    if len(prices) == 0 or len(regimes) == 0:
        raise ValueError("Prices and regimes must not be empty")

    if len(prices) != len(regimes):
        raise ValueError("Prices and regimes must have the same length")

    plotter = RegimePlotter(
        figsize=figsize,
        regime_colors=regime_colors,
        regime_labels=regime_labels,
    )
    return plotter.plot_regimes(prices, regimes, show=show)


def plot_regime_duration_distribution(regimes: np.ndarray, **kwargs) -> Figure:
    """Plot regime duration distribution.

    Args:
        regimes: Regime labels
        **kwargs: Additional arguments for RegimePlotter

    Returns:
        Matplotlib figure
    """
    # Calculate durations
    durations = {}
    current_regime = regimes[0]
    current_duration = 1

    for regime in regimes[1:]:
        if regime == current_regime:
            current_duration += 1
        else:
            if current_regime not in durations:
                durations[current_regime] = []
            durations[current_regime].append(current_duration)
            current_regime = regime
            current_duration = 1

    # Add last duration
    if current_regime not in durations:
        durations[current_regime] = []
    durations[current_regime].append(current_duration)

    plotter = RegimePlotter(**kwargs)
    return plotter.plot_duration_distribution(durations)


def plot_transition_matrix_heatmap(transition_matrix: np.ndarray, **kwargs) -> Figure:
    """Plot transition matrix heatmap.

    Args:
        transition_matrix: Transition probability matrix
        **kwargs: Additional arguments for RegimePlotter

    Returns:
        Matplotlib figure
    """
    plotter = RegimePlotter(**kwargs)
    return plotter.plot_transitions(transition_matrix)


def plot_feature_importance(importances: Union[pd.Series, Dict], **kwargs) -> Figure:
    """Plot feature importances.

    Args:
        importances: Feature importances
        **kwargs: Additional arguments for RegimePlotter

    Returns:
        Matplotlib figure
    """
    plotter = RegimePlotter(**kwargs)
    return plotter.plot_feature_importance(importances)


def plot_model_comparison(metrics: Union[pd.DataFrame, Dict], **kwargs) -> Figure:
    """Plot model comparison.

    Args:
        metrics: Model metrics
        **kwargs: Additional arguments for RegimePlotter

    Returns:
        Matplotlib figure
    """
    plotter = RegimePlotter(**kwargs)
    return plotter.plot_model_comparison(metrics)


def plot_regime_returns(returns: pd.Series, regimes: np.ndarray, **kwargs) -> Figure:
    """Plot returns by regime.

    Args:
        returns: Return series
        regimes: Regime labels
        **kwargs: Additional arguments for RegimePlotter

    Returns:
        Matplotlib figure
    """
    plotter = RegimePlotter(**kwargs)
    return plotter.plot_returns_by_regime(returns, regimes)
