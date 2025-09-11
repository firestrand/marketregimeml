#!/usr/bin/env python3
"""
Transition Analysis

Analyze regime transition patterns on real market data.
Requires ALPHAVANTAGE_API_KEY in your environment.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from marketregimeml.models import HMMRegimeDetector
from marketregimeml.data.loaders import KrakenDataLoader
from marketregimeml.evaluation.performance import calculate_regime_transition_returns


def load_data(symbol: str = "BTC/USD", lookback_days: int = 900) -> pd.DataFrame:
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    return loader.fetch_ohlcv(symbol, "1d", start, end)


def prepare_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=ohlcv.index)
    feats["returns"] = ohlcv["close"].pct_change()
    feats["volatility"] = (ohlcv["high"] / ohlcv["low"] - 1)
    feats["momentum"] = ohlcv["close"].pct_change(5)
    feats = feats.dropna()
    return feats


def main():
    print("=" * 60)
    print("Regime Transition Analysis (real data)")
    print("=" * 60)

    ohlcv = load_data("BTC/USD", 1000)
    feats = prepare_features(ohlcv)

    model = HMMRegimeDetector(n_regimes=3, covariance_type="diag", random_state=42)
    model.fit(feats)
    regimes = model.predict(feats)

    print("\nTransition matrix (counts):")
    analysis = model.analyze_regime_transitions(regimes)
    tm = analysis["transition_matrix"]
    for i, row in enumerate(tm):
        print(f"{i}: " + "  ".join(f"{int(x):4d}" for x in row))

    print("\nTransition probabilities:")
    tp = analysis["transition_probabilities"]
    for i, row in enumerate(tp):
        print(f"{i}: " + "  ".join(f"{x:0.3f}" for x in row))

    rets = feats["returns"].reindex(feats.index)
    trans_summary = calculate_regime_transition_returns(rets, regimes, periods_forward=5)
    if not trans_summary.empty:
        print("\nForward 5-day returns by transition:")
        print(trans_summary)
    else:
        print("\nNo transitions detected for the chosen window.")


if __name__ == "__main__":
    main()
