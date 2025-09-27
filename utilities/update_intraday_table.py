#!/usr/bin/env python
"""Update docs/intraday_recommendations.md Top‑Ready table with exact metrics.

Parses latest regime_ablation_results_*.txt and ensemble markdowns to extract
post‑smoothing silhouette and separation for HMM(2) and the BTC H1 ensemble pick.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
DOC_PATH = ROOT / "docs" / "intraday_recommendations.md"


def latest(pattern: str) -> Optional[Path]:
    files = sorted(glob.glob(str(REPORT_DIR / pattern)))
    return Path(files[-1]) if files else None


def parse_regime_results(path: Path) -> Dict[str, Dict[int, Dict[str, Dict[str, float]]]]:
    # results[dataset][n_regimes][model] = metrics
    results: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {}
    dataset = None
    nreg = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Dataset section header style: 'EUR/USD (M5):'
        if line.endswith(":") and "(" in line and ")" in line and not line.startswith("n_regimes="):
            dataset = line[:-1]
            results.setdefault(dataset, {})
            continue
        if line.startswith("n_regimes=") and dataset:
            try:
                nreg = int(line.split("=", 1)[1].rstrip(":"))
                results[dataset].setdefault(nreg, {})
            except Exception:
                nreg = None
            continue
        # Model metric lines: 'HMM: RQImean=..., ..., Stab=xx.x%, Sep=a.bcd, Sil=a.bcd, ...'
        if dataset and nreg and ":" in line:
            name, rest = line.split(":", 1)
            name = name.strip()
            # Extract Sil and Sep and Stab% and MinProp
            m_sil = re.search(r"Sil=([\-0-9\.]+)", rest)
            m_sep = re.search(r"Sep=([\-0-9\.]+)", rest)
            m_stab = re.search(r"Stab=([0-9\.]+)%", rest)
            m_minp = re.search(r"MinProp=([0-9\.]+)", rest)
            if m_sil and m_sep:
                results[dataset][nreg][name] = {
                    "silhouette": float(m_sil.group(1)),
                    "separation": float(m_sep.group(1)),
                    "persistence": float(m_stab.group(1)) / 100.0 if m_stab else 0.0,
                    "min_prop": float(m_minp.group(1)) if m_minp else 0.0,
                }
    return results


def parse_ensemble_md(path: Path, dataset_name: str, config_contains: str) -> Optional[Tuple[float, float]]:
    # Parse the dataset section and its Top configurations table rows
    text = path.read_text(encoding="utf-8")
    # Find section start
    idx = text.find(f"## {dataset_name}")
    if idx == -1:
        return None
    section = text[idx:]
    # Find Top configurations table
    tbl_idx = section.find("Top configurations:")
    if tbl_idx == -1:
        return None
    tbl = section[tbl_idx:]
    # Extract markdown table rows (lines starting with '|')
    rows = [ln for ln in tbl.splitlines() if ln.strip().startswith("|")]
    if len(rows) < 3:
        return None
    headers = [h.strip() for h in rows[0].strip("| ").split("|")]
    # Build row dicts
    best = None
    for r in rows[2:]:
        cols = [c.strip() for c in r.strip("| ").split("|")]
        row = dict(zip(headers, cols))
        cfg = row.get("configuration", "")
        if config_contains in cfg:
            try:
                sil = float(row.get("silhouette", "0").replace("~", "").split()[0])
                sep = float(row.get("separation", "0").replace("~", "").split()[0])
                best = (sil, sep)
                break
            except Exception:
                continue
    return best


def parse_best_ready_configs(path: Path, dataset_name: str) -> Optional[Dict[str, str]]:
    """Parse 'Best ready configurations' table for a dataset and return the first row.

    Returns a dict with keys: configuration, n_regimes, RQI, silhouette, separation
    """
    text = path.read_text(encoding="utf-8")
    idx = text.find(f"## {dataset_name}")
    if idx == -1:
        return None
    section = text[idx:]
    tag = "Best ready configurations (top 5):"
    t_idx = section.find(tag)
    if t_idx == -1:
        return None
    tbl = section[t_idx:]
    rows = [ln for ln in tbl.splitlines() if ln.strip().startswith("|")]
    if len(rows) < 3:
        return None
    headers = [h.strip() for h in rows[0].strip("| ").split("|")]
    # Take first data row
    cols = [c.strip() for c in rows[2].strip("| ").split("|")]
    row = dict(zip(headers, cols))
    return row


def update_doc(values: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
    md = DOC_PATH.read_text(encoding="utf-8")
    # Replace rows in the Top‑Ready table using simple regex per line
    def repl(line: str, key: str, sil: float, sep: float) -> str:
        return re.sub(r"\|\s*" + re.escape(key) + r"\s*\|([\s\S]*?)\|\s*[-~0-9\.\(\)\s]+\|\s*[-~0-9\.\+\(\)\s]+\|",
                      f"| {key} |\\1| {sil:.3f} | {sep:.3f} |", line)

    lines = md.splitlines()
    new_lines = []
    for ln in lines:
        if ln.startswith("| EUR/USD | M5 ") and "HMM" in ln:
            sil, sep = values.get("EUR/USD", {}).get("M5", (None, None)) or (None, None)
            if sil is not None:
                parts = ln.split("|")
                parts[-4] = f" {sil:.3f} "
                parts[-3] = f" {sep:.3f} "
                ln = "|".join(parts)
        if ln.startswith("| EUR/USD | M15 ") and "HMM" in ln:
            sil, sep = values.get("EUR/USD", {}).get("M15", (None, None)) or (None, None)
            if sil is not None:
                parts = ln.split("|")
                parts[-4] = f" {sil:.3f} "
                parts[-3] = f" {sep:.3f} "
                ln = "|".join(parts)
        if ln.startswith("| EUR/USD | M30 ") and "HMM" in ln:
            sil, sep = values.get("EUR/USD", {}).get("M30", (None, None)) or (None, None)
            if sil is not None:
                parts = ln.split("|")
                parts[-4] = f" {sil:.3f} "
                parts[-3] = f" {sep:.3f} "
                ln = "|".join(parts)
        if ln.startswith("| EUR/USD | H1 ") and "HMM" in ln:
            sil, sep = values.get("EUR/USD", {}).get("H1", (None, None)) or (None, None)
            if sil is not None:
                parts = ln.split("|")
                parts[-4] = f" {sil:.3f} "
                parts[-3] = f" {sep:.3f} "
                ln = "|".join(parts)
        if ln.startswith("| BTC/USD | M5 ") and "HMM" in ln:
            sil, sep = values.get("BTC/USD", {}).get("M5", (None, None)) or (None, None)
            if sil is not None:
                parts = ln.split("|")
                parts[-4] = f" {sil:.3f} "
                parts[-3] = f" {sep:.3f} "
                ln = "|".join(parts)
        if ln.startswith("| BTC/USD | H1 ") and "Voting" in ln:
            sil, sep = values.get("BTC/USD", {}).get("H1_ENSEMBLE", (None, None)) or (None, None)
            if sil is not None:
                parts = ln.split("|")
                parts[-4] = f" {sil:.3f} "
                parts[-3] = f" {sep:.3f} "
                ln = "|".join(parts)
        new_lines.append(ln)
    DOC_PATH.write_text("\n".join(new_lines), encoding="utf-8")

    # Now update Best‑Ready Per Timeframe section using ensemble 'best ready' rows
    btc_md = latest("ensemble_ablation_btc_multi_tf_*.md")
    eur_md = latest("ensemble_ablation_eurusd_multi_tf_*.md")
    best_rows: List[str] = []
    headers = [
        "Instrument",
        "Timeframe",
        "Configuration",
        "n_regimes",
        "RQI",
        "silhouette",
        "separation",
        "Source",
    ]
    if btc_md:
        for tf in ["M5", "M15", "M30", "H1"]:
            ds = f"BTC/USD ({tf})"
            row = parse_best_ready_configs(btc_md, ds)
            if row:
                best_rows.append(
                    "| BTC/USD | {tf} | {cfg} | {nr} | {rqi} | {sil} | {sep} | {src} |".format(
                        tf=tf,
                        cfg=row.get("configuration", ""),
                        nr=row.get("n_regimes", ""),
                        rqi=row.get("RQI", ""),
                        sil=row.get("silhouette", ""),
                        sep=row.get("separation", ""),
                        src=btc_md.name,
                    )
                )
    if eur_md:
        for tf in ["M5", "M15", "M30", "H1"]:
            ds = f"EUR/USD ({tf})"
            row = parse_best_ready_configs(eur_md, ds)
            if row:
                best_rows.append(
                    "| EUR/USD | {tf} | {cfg} | {nr} | {rqi} | {sil} | {sep} | {src} |".format(
                        tf=tf,
                        cfg=row.get("configuration", ""),
                        nr=row.get("n_regimes", ""),
                        rqi=row.get("RQI", ""),
                        sil=row.get("silhouette", ""),
                        sep=row.get("separation", ""),
                        src=eur_md.name,
                    )
                )

    doc = DOC_PATH.read_text(encoding="utf-8")
    start_tag = "<!-- AUTO-BEST-READY-START -->"
    end_tag = "<!-- AUTO-BEST-READY-END -->"
    if start_tag in doc and end_tag in doc and best_rows:
        head = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
        table = head + "\n".join(best_rows) + "\n"
        new_doc = re.sub(
            start_tag + r"[\s\S]*?" + end_tag,
            start_tag + "\n" + table + end_tag,
            doc,
            flags=re.M,
        )
        DOC_PATH.write_text(new_doc, encoding="utf-8")


def main() -> None:
    reg_path = latest("regime_ablation_results_*.txt")
    if not reg_path:
        print("No regime_ablation_results_*.txt found under report/")
        return
    reg = parse_regime_results(reg_path)

    vals: Dict[str, Dict[str, Tuple[float, float]]] = {}
    # EUR/USD TFs from HMM at n=2
    for tf in ["M5", "M15", "M30", "H1"]:
        ds = f"EUR/USD ({tf})"
        sil_sep = None
        try:
            m = reg[ds][2]["HMM"]
            sil_sep = (m["silhouette"], m["separation"])  # type: ignore[index]
        except Exception:
            pass
        if sil_sep:
            vals.setdefault("EUR/USD", {})[tf] = sil_sep

    # BTC HMM M5
    for tf in ["M5"]:
        ds = f"BTC/USD ({tf})"
        try:
            m = reg[ds][2]["HMM"]
            vals.setdefault("BTC/USD", {})[tf] = (m["silhouette"], m["separation"])  # type: ignore[index]
        except Exception:
            pass

    # BTC H1 ensemble from after‑denoiser report
    ens_path = latest("ensemble_ablation_btc_multi_tf_*185009*.md") or latest("ensemble_ablation_btc_multi_tf_*.md")
    if ens_path:
        best = parse_ensemble_md(ens_path, "BTC/USD (H1)", "Ensemble(XGBoost+RandomForest):voting")
        if best:
            vals.setdefault("BTC/USD", {})["H1_ENSEMBLE"] = best

    update_doc(vals)
    print("Updated:", DOC_PATH)


if __name__ == "__main__":
    main()
