#!/usr/bin/env python3
"""
Model Comparison

Compare multiple regime detection models on real market data.
Requires ALPHAVANTAGE_API_KEY in your environment.
"""

from datetime import datetime, timedelta
import pandas as pd

from marketregimeml.models import HMMRegimeDetector, GMMRegimeDetector
from marketregimeml.data.loaders import KrakenDataLoader
from marketregimeml.evaluation.evaluator import ModelEvaluator


def load_data(symbol: str = "BTC/USD", lookback_days: int = 800) -> pd.DataFrame:
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    return loader.fetch_ohlcv(symbol, "1d", start, end)


def prepare_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=ohlcv.index)
    feats["returns"] = ohlcv["close"].pct_change()
    feats["volatility"] = (ohlcv["high"] / ohlcv["low"] - 1)
    feats["momentum"] = ohlcv["close"].pct_change(10)
    return feats.dropna()


def main():
    print("=" * 60)
    print("Regime Model Comparison (real data)")
    print("=" * 60)

    ohlcv = load_data("BTC/USD", 900)
    feats = prepare_features(ohlcv)

    models = [
        HMMRegimeDetector(n_regimes=3, covariance_type="diag", random_state=42),
        GMMRegimeDetector(n_regimes=3, covariance_type="full", random_state=42),
    ]

    for m in models:
        m.fit(feats)

    evaluator = ModelEvaluator(models[0])
    comp = evaluator.compare_models(models, feats)
    display = comp[[
        "model_type",
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_index",
        "avg_confidence",
        "regime_quality_index",
    ]].sort_values("regime_quality_index", ascending=False)
    print(display.round(3))


if __name__ == "__main__":
    main()
