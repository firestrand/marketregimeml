"""Shared reporting utilities for ablation studies.

Provides compact Markdown summaries to keep benchmark outputs consistent.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def write_text(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def _md_h2(title: str) -> str:
    return f"\n\n## {title}\n\n"


def _md_code(text: str) -> str:
    return f"`{text}`"


def _format_table(df: pd.DataFrame) -> str:
    # Pandas to_markdown is convenient but may not always be available in minimal envs
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback: simple pipe table
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines)


def write_regime_ablation_markdown(
    all_results: Dict[str, Dict[int, Dict]], out_path: str
) -> None:
    """Create a Markdown summary for regime-count ablation results.

    all_results structure: {dataset_name: {n_regimes: {model_name: metrics_dict}}}
    """
    parts = ["# Regime Count Ablation Summary"]

    # Per-dataset summaries
    for dataset_name, dataset_results in all_results.items():
        parts.append(_md_h2(dataset_name))

        # Build RQI table
        rows = []
        model_names = set()
        for n_regimes, models in dataset_results.items():
            row = {"n_regimes": n_regimes}
            for mname, m in models.items():
                model_names.add(mname)
                val = m.get("rqi_mean", m.get("rqi", 0.0))
                row[f"{mname}_RQI"] = round(float(val), 2)
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("n_regimes")
        parts.append(_format_table(df))

        # Best per model
        best_lines = ["\nBest n_regimes per model:"]
        for mname in sorted(model_names):
            col = f"{mname}_RQI"
            if col in df.columns:
                idx = df[col].idxmax()
                best_lines.append(
                    f"- {mname}: n_regimes={int(df.loc[idx, 'n_regimes'])} (RQI={df.loc[idx, col]:.2f})"
                )
        parts.append("\n".join(best_lines))

    # Overall average RQI by n_regimes
    parts.append(_md_h2("Overall"))
    agg: Dict[int, list] = {}
    for dataset_results in all_results.values():
        for n_regimes, models in dataset_results.items():
            agg.setdefault(n_regimes, [])
            for m in models.values():
                rv = m.get("rqi_mean", m.get("rqi", 0.0))
                if rv:
                    agg[n_regimes].append(float(rv))
    summary = sorted(((n, np.mean(v)) for n, v in agg.items() if v), key=lambda x: x[0])
    if summary:
        df_sum = pd.DataFrame(summary, columns=["n_regimes", "avg_RQI"])  # type: ignore[arg-type]
        parts.append("\nAverage RQI across all models and datasets:\n")
        parts.append(_format_table(df_sum))

        best_n = max(summary, key=lambda kv: kv[1])[0]
        parts.append(f"\n✅ Optimal n_regimes: {_md_code(best_n)}")

    write_text(out_path, "\n".join(parts))


def write_ensemble_ablation_markdown(
    all_results: Dict[str, Dict[str, Dict]], out_path: str
) -> None:
    """Create a Markdown summary for ensemble-composition ablation results.

    all_results structure: {dataset_name: {config_name: metrics_dict}}
    """
    parts = ["# Ensemble Composition Ablation Summary"]

    for dataset_name, dataset_results in all_results.items():
        parts.append(_md_h2(dataset_name))
        # Sort by RQI
        sorted_results = sorted(
            dataset_results.items(), key=lambda x: x[1].get("rqi", 0.0), reverse=True
        )
        top = sorted_results[:10]
        df = pd.DataFrame(
            [
                {
                    "configuration": name,
                    "n_regimes": int(m.get("n_regimes", 0)),
                    "RQI": round(float(m.get("rqi", 0.0)), 2),
                    "type": m.get("type", ""),
                    "silhouette_pre": round(float(m.get("silhouette_pre", 0.0)), 3),
                    "silhouette": round(float(m.get("silhouette", 0.0)), 3),
                    "separation_pre": round(float(m.get("separation_pre", 0.0)), 3),
                    "separation": round(float(m.get("separation", 0.0)), 3),
                    "strategy_ready": bool(m.get("strategy_ready", False)),
                }
                for name, m in top
            ]
        )
        parts.append("Top configurations:\n")
        parts.append(_format_table(df))

        # Best ready configurations (top 5)
        ready = [
            (name, m)
            for name, m in sorted(
                dataset_results.items(),
                key=lambda x: x[1].get("rqi", 0.0),
                reverse=True,
            )
            if m.get("strategy_ready", False)
        ][:5]
        if ready:
            parts.append("\nBest ready configurations (top 5):\n")
            df_ready = pd.DataFrame(
                [
                    {
                        "configuration": n,
                        "n_regimes": int(m.get("n_regimes", 0)),
                        "RQI": round(float(m.get("rqi", 0.0)), 2),
                        "silhouette": round(float(m.get("silhouette", 0.0)), 3),
                        "separation": round(float(m.get("separation", 0.0)), 3),
                    }
                    for n, m in ready
                ]
            )
            parts.append(_format_table(df_ready))

        # Triage: configurations with negative silhouette (top 5)
        neg = [
            (name, m)
            for name, m in sorted(
                dataset_results.items(), key=lambda x: x[1].get("silhouette", 0.0)
            )
            if m.get("silhouette", 0.0) < 0
        ][:5]
        if neg:
            parts.append("\nTriage: configurations with negative silhouette (top 5):\n")
            df_neg = pd.DataFrame(
                [
                    {
                        "configuration": n,
                        "n_regimes": int(m.get("n_regimes", 0)),
                        "silhouette": round(float(m.get("silhouette", 0.0)), 3),
                        "separation": round(float(m.get("separation", 0.0)), 3),
                        "RQI": round(float(m.get("rqi", 0.0)), 1),
                    }
                    for n, m in neg
                ]
            )
            parts.append(_format_table(df_neg))

        # Closest-to-ready (not ready), top 5 by silhouette
        not_ready = [
            (name, m)
            for name, m in sorted(
                dataset_results.items(),
                key=lambda x: x[1].get("silhouette", 0.0),
                reverse=True,
            )
            if not m.get("strategy_ready", False)
        ][:5]
        if not_ready:
            parts.append("\nClosest-to-ready (not ready; top 5 by silhouette):\n")
            df_nr = pd.DataFrame(
                [
                    {
                        "configuration": n,
                        "n_regimes": int(m.get("n_regimes", 0)),
                        "silhouette": round(float(m.get("silhouette", 0.0)), 3),
                        "separation": round(float(m.get("separation", 0.0)), 3),
                        "persistence": round(float(m.get("persistence", 0.0)), 3),
                        "min_prop": round(float(m.get("min_cluster_prop", 0.0)), 3),
                        "RQI": round(float(m.get("rqi", 0.0)), 1),
                    }
                    for n, m in not_ready
                ]
            )
            parts.append(_format_table(df_nr))

    write_text(out_path, "\n".join(parts))
