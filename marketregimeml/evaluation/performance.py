"""Performance metrics for trading strategies based on regime detection."""

from typing import Dict, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    """Calculate Sharpe ratio.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate per period
        periods: Number of periods per year (252 for daily)

    Returns:
        Sharpe ratio
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    excess_returns = returns - risk_free_rate

    if excess_returns.std() == 0:
        return 0.0

    return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    target_return: float = 0.0,
    periods: int = 252,
) -> float:
    """Calculate Sortino ratio.

    Args:
        returns: Return series
        target_return: Target return per period
        periods: Number of periods per year

    Returns:
        Sortino ratio
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float("inf")

    downside_std = np.sqrt(np.mean(downside_returns**2))

    if downside_std == 0:
        return float("inf")

    return np.sqrt(periods) * excess_returns.mean() / downside_std


def calculate_calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods: int = 252,
    is_prices: bool = False,
) -> float:
    """Calculate Calmar ratio.

    Args:
        returns: Return series or price series
        periods: Number of periods per year
        is_prices: Whether input is prices (True) or returns (False)

    Returns:
        Calmar ratio
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    if is_prices:
        prices = returns
        returns = prices.pct_change().dropna()

    annual_return = returns.mean() * periods
    max_dd, _ = calculate_max_drawdown(returns)

    if max_dd == 0:
        return float("inf")

    return annual_return / max_dd


def calculate_max_drawdown(
    data: Union[pd.Series, np.ndarray], is_prices: bool = False
) -> Tuple[float, int]:
    """Calculate maximum drawdown and duration.

    Args:
        data: Return or price series
        is_prices: Whether input is prices (True) or returns (False)

    Returns:
        Tuple of (max_drawdown, duration_in_periods)
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    if is_prices:
        prices = data
    else:
        prices = (1 + data).cumprod()

    # Calculate running maximum
    running_max = prices.expanding().max()

    # Calculate drawdown
    drawdown = (prices - running_max) / running_max

    # Find maximum drawdown
    max_drawdown = abs(drawdown.min())

    # Calculate duration
    if max_drawdown > 0:
        # Find when max drawdown occurred
        max_dd_idx = drawdown.idxmin()

        # Find start of drawdown (last peak before max drawdown)
        peaks = prices[:max_dd_idx][
            prices[:max_dd_idx] == running_max[:max_dd_idx]
        ]
        if len(peaks) > 0:
            start_idx = peaks.index[-1]

            # Find recovery point (if any)
            recovery_prices = prices[max_dd_idx:]
            recovery_point = recovery_prices[
                recovery_prices >= prices[start_idx]
            ]

            if len(recovery_point) > 0:
                end_idx = recovery_point.index[0]
                duration = len(prices[start_idx:end_idx])
            else:
                # Still in drawdown
                duration = len(prices[start_idx:])
        else:
            duration = 0
    else:
        duration = 0

    return max_drawdown, duration


def calculate_win_rate(
    returns: Union[pd.Series, np.ndarray], threshold: float = 0.0
) -> float:
    """Calculate win rate (percentage of positive returns).

    Args:
        returns: Return series
        threshold: Minimum return to count as a win

    Returns:
        Win rate between 0 and 1
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    if len(returns) == 0:
        return 0.0

    wins = (returns > threshold).sum()
    return wins / len(returns)


def calculate_profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate profit factor (gross profits / gross losses).

    Args:
        returns: Return series

    Returns:
        Profit factor
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float("inf") if gains > 0 else 0.0

    return gains / losses


def calculate_information_ratio(
    returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    periods: int = 252,
) -> float:
    """Calculate information ratio.

    Args:
        returns: Strategy return series
        benchmark_returns: Benchmark return series
        periods: Number of periods per year

    Returns:
        Information ratio
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = pd.Series(benchmark_returns)

    active_returns = returns - benchmark_returns

    if active_returns.std() == 0:
        return 0.0

    return np.sqrt(periods) * active_returns.mean() / active_returns.std()


def calculate_regime_based_returns(
    returns: pd.Series, regimes: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """Calculate return metrics for each regime.

    Args:
        returns: Return series
        regimes: Regime labels

    Returns:
        Dictionary of metrics per regime
    """
    regime_metrics = {}

    for regime in np.unique(regimes):
        regime_mask = regimes[: len(returns)] == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) > 0:
            regime_metrics[regime] = {
                "mean_return": regime_returns.mean(),
                "total_return": (1 + regime_returns).prod() - 1,
                "volatility": regime_returns.std(),
                "sharpe_ratio": calculate_sharpe_ratio(regime_returns),
                "win_rate": calculate_win_rate(regime_returns),
                "max_drawdown": calculate_max_drawdown(regime_returns)[0],
                "num_periods": len(regime_returns),
            }
        else:
            regime_metrics[regime] = {
                "mean_return": 0.0,
                "total_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "num_periods": 0,
            }

    return regime_metrics


def calculate_regime_transition_returns(
    returns: pd.Series, regimes: np.ndarray, periods_forward: int = 1
) -> pd.DataFrame:
    """Calculate returns following regime transitions.

    Args:
        returns: Return series
        regimes: Regime labels
        periods_forward: Number of periods to look forward

    Returns:
        DataFrame with transition statistics
    """
    transitions = []

    for i in range(len(regimes) - periods_forward):
        if i > 0 and regimes[i] != regimes[i - 1]:
            # Transition occurred
            from_regime = regimes[i - 1]
            to_regime = regimes[i]

            # Calculate forward returns
            forward_returns = returns.iloc[i : i + periods_forward]

            transitions.append(
                {
                    "from_regime": from_regime,
                    "to_regime": to_regime,
                    "forward_return": forward_returns.sum(),
                    "volatility": forward_returns.std(),
                }
            )

    if not transitions:
        return pd.DataFrame()

    df = pd.DataFrame(transitions)

    # Aggregate by transition type
    summary = (
        df.groupby(["from_regime", "to_regime"])
        .agg(
            {"forward_return": ["mean", "sum", "count"], "volatility": "mean"}
        )
        .reset_index()
    )

    summary.columns = [
        "from_regime",
        "to_regime",
        "avg_return",
        "total_return",
        "count",
        "volatility",
    ]

    return summary


def backtest_regime_strategy(
    prices: pd.Series,
    regimes: np.ndarray,
    position_map: Dict[int, float],
    transaction_cost: float = 0.001,
    initial_capital: float = 10000,
) -> Dict[str, Any]:
    """Backtest a regime-based trading strategy.

    Args:
        prices: Price series
        regimes: Regime predictions
        position_map: Dictionary mapping regime to position size
        transaction_cost: Transaction cost as fraction
        initial_capital: Starting capital

    Returns:
        Dictionary with backtest results
    """
    returns = prices.pct_change().fillna(0)

    # Generate positions from regimes
    positions = np.array(
        [position_map.get(r, 0.0) for r in regimes[: len(returns)]]
    )

    # Calculate trades (position changes)
    trades = np.diff(positions, prepend=positions[0])
    trade_costs = np.abs(trades) * transaction_cost

    # Calculate strategy returns
    strategy_returns = positions * returns.values - trade_costs

    # Calculate cumulative returns
    cumulative_returns = pd.Series(
        (1 + strategy_returns).cumprod() * initial_capital, index=returns.index
    )

    # Calculate metrics
    total_return = (cumulative_returns.iloc[-1] / initial_capital) - 1
    sharpe = calculate_sharpe_ratio(strategy_returns)
    max_dd, dd_duration = calculate_max_drawdown(strategy_returns)
    win_rate = calculate_win_rate(strategy_returns)
    profit_factor = calculate_profit_factor(strategy_returns)

    return {
        "returns": pd.Series(strategy_returns, index=returns.index),
        "cumulative_returns": cumulative_returns,
        "positions": positions,
        "trades": trades,
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "drawdown_duration": dd_duration,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": np.sum(trades != 0),
        "transaction_costs": trade_costs.sum(),
    }


class PerformanceEvaluator:
    """Comprehensive performance evaluation for regime-based strategies."""

    def __init__(self):
        """Initialize PerformanceEvaluator."""
        self.metrics_history = []

    def evaluate_strategy(
        self,
        prices: pd.Series,
        regimes: np.ndarray,
        position_map: Dict[int, float],
        benchmark_returns: Optional[pd.Series] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate a regime-based trading strategy.

        Args:
            prices: Price series
            regimes: Regime predictions
            position_map: Position sizing per regime
            benchmark_returns: Optional benchmark returns
            **kwargs: Additional arguments for backtest

        Returns:
            Dictionary with evaluation metrics
        """
        # Run backtest
        backtest_results = backtest_regime_strategy(
            prices, regimes, position_map, **kwargs
        )

        # Calculate regime-specific metrics
        returns = prices.pct_change().dropna()
        regime_metrics = calculate_regime_based_returns(
            returns, regimes[: len(returns)]
        )

        # Calculate transition analysis
        transition_metrics = calculate_regime_transition_returns(
            returns, regimes[: len(returns)]
        )

        # Information ratio if benchmark provided
        if benchmark_returns is not None:
            ir = calculate_information_ratio(
                backtest_results["returns"], benchmark_returns
            )
        else:
            ir = None

        results = {
            **backtest_results,
            "regime_returns": regime_metrics,
            "transition_analysis": transition_metrics,
            "information_ratio": ir,
        }

        # Store for comparison
        self.metrics_history.append(results)

        return results

    def compare_strategies(
        self,
        prices: pd.Series,
        regimes: np.ndarray,
        strategies: Dict[str, Dict[int, float]],
        **kwargs,
    ) -> pd.DataFrame:
        """Compare multiple trading strategies.

        Args:
            prices: Price series
            regimes: Regime predictions
            strategies: Dictionary of strategy names to position maps
            **kwargs: Additional arguments for backtest

        Returns:
            DataFrame comparing strategy metrics
        """
        results = []

        for name, position_map in strategies.items():
            metrics = self.evaluate_strategy(
                prices, regimes, position_map, **kwargs
            )

            results.append(
                {
                    "strategy": name,
                    "total_return": metrics["total_return"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "win_rate": metrics["win_rate"],
                    "profit_factor": metrics["profit_factor"],
                    "num_trades": metrics["num_trades"],
                }
            )

        return pd.DataFrame(results).set_index("strategy")

    def calculate_confidence_weighted_returns(
        self,
        returns: pd.Series,
        probabilities: np.ndarray,
        regimes: np.ndarray,
    ) -> pd.Series:
        """Calculate returns weighted by regime confidence.

        Args:
            returns: Return series
            probabilities: Regime probabilities
            regimes: Predicted regimes

        Returns:
            Confidence-weighted return series
        """
        # Get confidence in predicted regime
        confidence = probabilities[np.arange(len(regimes)), regimes]

        # Weight returns by confidence
        weighted_returns = returns.values[: len(confidence)] * confidence

        return pd.Series(
            weighted_returns, index=returns.index[: len(confidence)]
        )

    def regime_performance_summary(
        self, returns: pd.Series, regimes: np.ndarray
    ) -> pd.DataFrame:
        """Create summary statistics for each regime.

        Args:
            returns: Return series
            regimes: Regime labels

        Returns:
            DataFrame with regime statistics
        """
        regime_metrics = calculate_regime_based_returns(returns, regimes)

        summary_data = []
        for regime, metrics in regime_metrics.items():
            summary_data.append(
                {
                    "regime": regime,
                    "count": metrics["num_periods"],
                    "mean_return": metrics["mean_return"],
                    "volatility": metrics["volatility"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "win_rate": metrics["win_rate"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )

        return pd.DataFrame(summary_data).set_index("regime")

    def calculate_regime_timing_value(
        self,
        returns: pd.Series,
        true_regimes: np.ndarray,
        predicted_regimes: np.ndarray,
    ) -> float:
        """Calculate value added by regime timing.

        Args:
            returns: Return series
            true_regimes: True regime labels
            predicted_regimes: Predicted regime labels

        Returns:
            Timing value (excess return from correct predictions)
        """
        # Perfect foresight returns
        perfect_positions = np.array(
            [
                1.0 if r == 0 else (-1.0 if r == 1 else 0.0)
                for r in true_regimes[: len(returns)]
            ]
        )
        perfect_returns = (perfect_positions * returns.values).sum()

        # Predicted returns
        predicted_positions = np.array(
            [
                1.0 if r == 0 else (-1.0 if r == 1 else 0.0)
                for r in predicted_regimes[: len(returns)]
            ]
        )
        predicted_returns = (predicted_positions * returns.values).sum()

        # Buy and hold returns
        buy_hold_returns = returns.sum()

        # Timing value is excess over buy-and-hold
        timing_value = predicted_returns - buy_hold_returns

        # As percentage of perfect timing
        perfect_excess = perfect_returns - buy_hold_returns
        if perfect_excess != 0:
            timing_efficiency = timing_value / perfect_excess
        else:
            timing_efficiency = 0.0

        return timing_efficiency

    def plot_performance(
        self,
        prices: pd.Series,
        regimes: np.ndarray,
        position_map: Optional[Dict[int, float]] = None,
        show: bool = True,
    ) -> Figure:
        """Plot performance analysis.

        Args:
            prices: Price series
            regimes: Regime predictions
            position_map: Optional position map for strategy
            show: Whether to display the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot 1: Prices with regime overlay
        ax1 = axes[0]
        ax1.plot(prices.index, prices.values, "b-", linewidth=1)
        ax1.set_ylabel("Price")
        ax1.set_title("Price with Regime Overlay")
        ax1.grid(True, alpha=0.3)

        # Add regime coloring
        regime_colors = ["green", "red", "gray", "blue", "orange"]
        for i in range(len(prices)):
            if i < len(regimes):
                ax1.axvspan(
                    prices.index[max(0, i - 1)],
                    prices.index[min(i, len(prices) - 1)],
                    alpha=0.2,
                    color=regime_colors[regimes[i] % len(regime_colors)],
                )

        # Plot 2: Returns by regime
        ax2 = axes[1]
        returns = prices.pct_change().dropna()

        for regime in np.unique(regimes):
            regime_mask = regimes[: len(returns)] == regime
            regime_returns = returns[regime_mask]
            ax2.scatter(
                regime_returns.index,
                regime_returns.values,
                label=f"Regime {regime}",
                alpha=0.6,
                color=regime_colors[regime % len(regime_colors)],
            )

        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Returns")
        ax2.set_title("Returns by Regime")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Cumulative returns if strategy provided
        ax3 = axes[2]

        if position_map is not None:
            backtest = backtest_regime_strategy(prices, regimes, position_map)
            ax3.plot(
                backtest["cumulative_returns"].index,
                backtest["cumulative_returns"].values,
                "b-",
                label="Strategy",
                linewidth=2,
            )

            # Add buy-and-hold
            buy_hold = (1 + returns).cumprod() * 10000
            ax3.plot(
                buy_hold.index,
                buy_hold.values,
                "gray--",
                label="Buy & Hold",
                linewidth=1,
            )

            ax3.set_ylabel("Portfolio Value")
            ax3.set_title("Strategy Performance")
            ax3.legend()
        else:
            # Just show cumulative returns
            cum_returns = (1 + returns).cumprod()
            ax3.plot(cum_returns.index, cum_returns.values, "b-", linewidth=2)
            ax3.set_ylabel("Cumulative Returns")
            ax3.set_title("Cumulative Performance")

        ax3.set_xlabel("Date")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if show:
            plt.show()

        return fig
