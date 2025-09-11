"""Compatibility backtester used by examples.

Provides a simple class interface `RegimeBacktester` that wraps the
functional backtesting utilities in `performance.py`.
"""

from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

from .performance import backtest_regime_strategy


class RegimeBacktester:
    """Lightweight wrapper to backtest regime-based strategies.

    Parameters mirror the usage in examples for minimal changes.
    """

    def __init__(
        self,
        model=None,
        transaction_cost: float = 0.001,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.model = model
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

    def backtest_strategy(
        self,
        features: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        strategy_weights: Optional[Dict[int, float]] = None,
        prices: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Backtest given `strategy_weights` across predicted regimes.

        - If `prices` is not provided, constructs a price series from `returns`.
        - Uses the attached `model` to predict regimes from `features`.
        """
        if self.model is None:
            raise ValueError("RegimeBacktester requires a fitted `model` instance")

        # Predict regimes
        regimes = self.model.predict(features)

        # Build price series
        if prices is None:
            if returns is None:
                raise ValueError(
                    "Either `prices` or `returns` must be provided to backtest"
                )
            ret = pd.Series(returns).fillna(0)
            prices = (1.0 + ret).cumprod() * float(self.initial_capital)
        else:
            prices = pd.Series(prices)

        # Default: no position if not specified
        position_map = strategy_weights or {}

        results = backtest_regime_strategy(
            prices=prices,
            regimes=np.asarray(regimes),
            position_map=position_map,
            transaction_cost=self.transaction_cost,
            initial_capital=self.initial_capital,
        )

        # Alias for examples expecting 'portfolio_values'
        results["portfolio_values"] = results.get("cumulative_returns")
        return results

    def compare_strategies(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        strategies: Dict[str, Dict[int, float]],
        prices: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compare multiple strategies using the attached model's regimes.

        Builds a price series if not provided, predicts regimes from features,
        runs each strategy via backtest_regime_strategy, and returns a summary
        DataFrame similar to PerformanceEvaluator.compare_strategies.
        """
        import pandas as pd

        # Prepare prices
        if prices is None:
            ret = pd.Series(returns).fillna(0)
            prices = (1.0 + ret).cumprod() * float(self.initial_capital)
        else:
            prices = pd.Series(prices)

        # Predict regimes
        if self.model is None:
            raise ValueError("RegimeBacktester requires a fitted `model` instance")
        regimes = np.asarray(self.model.predict(features))

        rows = []
        for name, position_map in strategies.items():
            res = backtest_regime_strategy(
                prices=prices,
                regimes=regimes,
                position_map=position_map,
                transaction_cost=self.transaction_cost,
                initial_capital=self.initial_capital,
            )
            rows.append(
                {
                    "strategy": name,
                    "total_return": res["total_return"],
                    "sharpe_ratio": res["sharpe_ratio"],
                    "max_drawdown": res["max_drawdown"],
                    "win_rate": res["win_rate"],
                    "profit_factor": res["profit_factor"],
                    "num_trades": res["num_trades"],
                }
            )

        return pd.DataFrame(rows).set_index("strategy")
