"""Ensemble composition ablation across multiple timeframes for a given asset.

Focus: BTC/USD intraday (M5, M15, M30, H1). Aggregates M15/M30/H1 from M5 if available.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from marketregimeml.benchmarks.ensemble_composition_ablation import (
    test_ensemble_combinations,
)
from marketregimeml.benchmarks.reporting import write_ensemble_ablation_markdown
from marketregimeml.data.loaders.kraken import KrakenDataLoader
from marketregimeml.benchmarks.real_data_benchmark import RealDataLoader
from marketregimeml.data.storage.duckdb_store import DuckDBStore


def _prepare_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    # Reuse RealDataLoader's feature prep for consistency
    return RealDataLoader()._prepare_features(ohlcv)


def _load_btc_timeframe(timeframe: str, bars: int) -> pd.DataFrame | None:
    """Load BTC/USD OHLCV for timeframe using Kraken; try DuckDB first when possible."""
    store = DuckDBStore()
    # Direct read if present in DuckDB with matching timeframe
    try:
        df = store.read_ohlcv("BTC/USD", timeframe)
        if not df.empty and len(df) >= bars:
            return _prepare_features(df.iloc[-bars:])
    except Exception:
        pass

    # Otherwise fetch from Kraken
    loader = KrakenDataLoader()
    end = datetime.utcnow()
    # Approx start based on timeframe granularity
    if timeframe == "H1":
        start = end - timedelta(hours=bars * 2)
    elif timeframe in ("M5", "M15", "M30"):
        # For minute timeframes, try aggregating from stored M5 if available
        base_tf = "M5"
        try:
            agg_tf = timeframe
            base = store.read_ohlcv("BTC/USD", base_tf)
            if not base.empty:
                agg = store.read_ohlcv_aggregated("BTC/USD", base_tf, agg_tf)
                if not agg.empty:
                    return _prepare_features(agg.iloc[-bars:])
        except Exception:
            pass
        # Fall back: attempt to fetch directly at requested timeframe (may be limited)
        start = end - timedelta(days=365 * 2)
    else:
        start = end - timedelta(days=365 * 2)

    ohlcv = loader.fetch_ohlcv("BTC/USD", timeframe=timeframe, start=start, end=end)
    # Optionally store
    try:
        store.write_ohlcv(ohlcv, "BTC/USD", timeframe)
    except Exception:
        pass
    return _prepare_features(ohlcv)


def run_btc_multi_tf_ablation(
    timeframes: List[str] | None = None, n_regimes: int = 3
) -> Dict[str, Dict]:
    if timeframes is None:
        timeframes = ["M5", "M15", "M30", "H1"]

    # Estimate bars to cover ~2 years
    days_2y = 365 * 2
    bars_map = {
        "H1": int(days_2y * 24),
        "M30": int(days_2y * 24 * 2),
        "M15": int(days_2y * 24 * 4),
        "M5": int(days_2y * 24 * 12),
    }

    datasets: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        features = _load_btc_timeframe(tf, bars_map.get(tf, 1000))
        if features is not None and len(features) > 100:
            datasets[f"BTC/USD ({tf})"] = features

    all_results: Dict[str, Dict] = {}
    for name, feats in datasets.items():
        print(f"\n=== Dataset: {name} | samples={len(feats)} ===")
        results = test_ensemble_combinations(
            feats, n_regimes=n_regimes, include_unsupervised=True, seeds=[0, 1]
        )
        all_results[name] = results

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"report/ensemble_ablation_btc_multi_tf_{ts}.md"
    try:
        write_ensemble_ablation_markdown(all_results, out)
        print(f"Markdown report saved to {out}")
    except Exception as e:
        print(f"Markdown export skipped: {e}")

    return all_results


if __name__ == "__main__":
    run_btc_multi_tf_ablation()
