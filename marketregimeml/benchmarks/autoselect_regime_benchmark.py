"""Auto-select n_regimes benchmark and comparison.

Loads intraday FX (EUR/USD M5) from DuckDB (preferred) or OANDA, compares
auto-selected regime count (AIC/BIC/Silhouette) vs fixed counts, and reports
RQI metrics.
"""

from datetime import datetime, timedelta
from typing import Dict

import pandas as pd

from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.data.storage.duckdb_store import DuckDBStore
from marketregimeml.data.loaders.oanda import OANDADataLoader
from marketregimeml.models import HMMRegimeDetector, GMMRegimeDetector


def load_eurusd_m5(limit_days: int = 730) -> pd.DataFrame:
    try:
        store = DuckDBStore()
        df = store.read_ohlcv("EUR_USD", "M5")
        if not df.empty:
            # Limit to last limit_days
            if len(df) > 0:
                cutoff = df.index.max() - pd.Timedelta(days=limit_days)
                df = df[df.index >= cutoff]
            return df
    except Exception:
        pass

    # Fallback to OANDA if DB missing
    loader = OANDADataLoader(store_to_duckdb=True)
    end = datetime.utcnow()
    start = end - timedelta(days=limit_days)
    return loader.fetch_ohlcv("EUR_USD", "M5", start, end)


def compare_autoselect(
    features: pd.DataFrame, model_name: str, criteria: str = "aic"
) -> Dict:
    metrics = RegimeMetrics()

    if model_name == "HMM":
        base = HMMRegimeDetector(n_regimes=3, random_state=0)
    elif model_name == "GMM":
        base = GMMRegimeDetector(n_regimes=3, random_state=0)
    else:
        raise ValueError("Unsupported model")

    # Auto-select
    base.set_params(auto_optimize_regimes=True, min_regimes=2, max_regimes=9)
    opt = base.optimize_regime_count(features, criteria=criteria)
    best_model = opt.get("best_model", None)
    optimal_n = opt.get("optimal_regimes", base.n_regimes)

    results: Dict[str, float] = {"optimal_n": float(optimal_n)}

    if best_model is None:
        best_model = base
        best_model.fit(features)

    # Score optimized
    preds = best_model.predict(features)
    prob = best_model.predict_proba(features)
    results["rqi_auto"] = metrics.regime_quality_index(features.values, preds, prob)

    # Score baselines (fixed n in {2,3,4,5})
    fixed_ns = [2, 3, 4, 5]
    for n in fixed_ns:
        if model_name == "HMM":
            m = HMMRegimeDetector(n_regimes=n, random_state=0)
        else:
            m = GMMRegimeDetector(n_regimes=n, random_state=0)
        m.fit(features)
        p = m.predict(features)
        pr = m.predict_proba(features)
        results[f"rqi_fixed_{n}"] = metrics.regime_quality_index(features.values, p, pr)

    return results


def run():
    print("Auto-select n_regimes benchmark (EUR/USD M5 ~2y)")
    ohlcv = load_eurusd_m5(730)
    features = pd.DataFrame(
        {
            "returns": ohlcv["close"].pct_change(),
            "vol": ohlcv["high"] / ohlcv["low"] - 1,
        }
    ).dropna()

    results = {}
    for model in ["HMM", "GMM"]:
        for criteria in ["aic", "bic", "silhouette", "calinski_harabasz"]:
            print(f"\n{model} criteria={criteria}")
            res = compare_autoselect(features, model, criteria)
            results[(model, criteria)] = res
            print(
                f"optimal_n={int(res['optimal_n'])}, RQI_auto={res['rqi_auto']:.1f}, "
                + ", ".join(
                    [f"n={n}:{res[f'rqi_fixed_{n}']:.1f}" for n in [2, 3, 4, 5]]
                )
            )

    # Save markdown summary
    lines = ["# Auto-select n_regimes Benchmark (EUR/USD M5 ~2y)", ""]
    for (model, criteria), res in results.items():
        lines.append(f"## {model} ({criteria.upper()})")
        lines.append(
            f"- optimal_n: {int(res['optimal_n'])}\n- RQI(auto): {res['rqi_auto']:.1f}\n"
            + "- RQI(fixed): "
            + ", ".join([f"n={n}:{res[f'rqi_fixed_{n}']:.1f}" for n in [2, 3, 4, 5]])
        )
        lines.append("")

    path = "report/autoselect_regimes_m5_2y.md"
    import os

    os.makedirs("report", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved markdown summary to {path}")


if __name__ == "__main__":
    run()
