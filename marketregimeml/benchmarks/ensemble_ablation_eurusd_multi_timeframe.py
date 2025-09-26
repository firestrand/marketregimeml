"""Ensemble composition ablation across EUR/USD intraday timeframes using DuckDB aggregation.

Assumes M5 OHLCV is stored in DuckDB; derives M15/M30/H1 via read_ohlcv_aggregated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd

from marketregimeml.benchmarks.ensemble_composition_ablation import (
    test_ensemble_combinations,
)
from marketregimeml.benchmarks.reporting import write_ensemble_ablation_markdown
from marketregimeml.data.storage.duckdb_store import DuckDBStore
from marketregimeml.benchmarks.real_data_benchmark import RealDataLoader


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    return RealDataLoader()._prepare_features(df)


def run_eurusd_multi_tf_ablation(n_regimes: int = 3, seeds: List[int] | None = None) -> Dict[str, Dict]:
    if seeds is None:
        seeds = [0, 1]

    store = DuckDBStore()
    datasets: Dict[str, pd.DataFrame] = {}

    # Base M5
    m5 = store.read_ohlcv("EUR_USD", "M5")
    if m5 is not None and not m5.empty:
        datasets["EUR/USD (M5)"] = _prep(m5)
    else:
        print("No EUR/USD M5 found in DuckDB. Please run scripts/fetch_oanda_intraday.py EUR_USD M5 <days> first.")

    # Aggregated timeframes from M5
    for tf in ["M15", "M30", "H1"]:
        try:
            agg = store.read_ohlcv_aggregated("EUR_USD", "M5", tf)
            if not agg.empty:
                datasets[f"EUR/USD ({tf})"] = _prep(agg)
        except Exception as e:
            print(f"Skipping {tf}: {e}")

    all_results: Dict[str, Dict] = {}
    for name, feats in datasets.items():
        print(f"\n=== Dataset: {name} | samples={len(feats)} ===")
        results = test_ensemble_combinations(
            feats, n_regimes=n_regimes, include_unsupervised=True, seeds=seeds
        )
        all_results[name] = results

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"report/ensemble_ablation_eurusd_multi_tf_{ts}.md"
    try:
        write_ensemble_ablation_markdown(all_results, out)
        print(f"Markdown report saved to {out}")
    except Exception as e:
        print(f"Markdown export skipped: {e}")

    return all_results


if __name__ == "__main__":
    run_eurusd_multi_tf_ablation()

