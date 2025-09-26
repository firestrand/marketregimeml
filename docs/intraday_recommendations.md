# Intraday Regime Recommendations

This document summarizes practical, intraday‑only guidance for regime detection on EUR/USD and BTC using the metrics and tooling in this repo.

## Scope
- Symbols: EUR/USD and BTC/USD
- Timeframes: M5, M15, M30, H1 (aggregated from M5 via DuckDB when available)
- Goals: high separation and stability suitable for automated strategies

## Primary Recommendation
- Model: `HMMRegimeDetector(n_regimes=2, covariance_type='full', random_state=0)`
- Label smoothing: majority filter with `smoothing_window = 5–7` (applied via `ReadinessThresholds` in `strategy.assess`)
- Features: `RealDataLoader._prepare_features` (FeatureEngine + robust normalization)
- Optional extras for noisy markets: enable denoised features by setting `MRML_USE_DENOISER=1` (adds `close_denoised`, `returns_denoised` as features)

## Guardrails (Readiness)
A configuration is considered “strategy‑ready” if all of the following (post‑smoothing) hold:
- Silhouette ≥ 0.30
- Separation (inter/intra centroid ratio) ≥ 1.50
- Persistence ≥ 0.85
- Min cluster proportion ≥ 0.10

Use these checks on your evaluation window before deployment. Tighten silhouette to 0.35–0.40 if you want a higher bar.

## Instrument‑Specific Notes
- EUR/USD (M5/M15/M30/H1): HMM(2) consistently passes readiness with good margins; ensembles often improve RQI but typically have lower silhouette intraday. Use ensembles for corroboration only.
- BTC/USD (M5/H1): HMM(2) passes readiness; simple voting ensembles (XGBoost + RandomForest, n=2–3) can also pass readiness on M5/H1. Denoised features help slightly here.
- BTC/USD (M15/M30): Prefer HMM(2); many ensemble mixes remain close to readiness thresholds on silhouette.

## Best‑Ready Picks (by timeframe)

- EUR/USD M5: HMM(2), smoothing_window=5–7 (post‑silhouette typically ≥ 0.5; separation ≥ 2.0)
- EUR/USD M15: HMM(2), smoothing_window=5–7 (post‑silhouette typically ≥ 0.3; separation ≥ 1.5)
- EUR/USD M30: HMM(2), smoothing_window=5–7
- EUR/USD H1: HMM(2), smoothing_window=5–7
- BTC/USD M5: HMM(2) primary; voting(XGBoost+RF) at n=2–3 often passes; enable denoiser features if market is very noisy
- BTC/USD M15: HMM(2) primary; ensembles often close‑but‑not‑ready on silhouette
- BTC/USD M30: HMM(2) primary
- BTC/USD H1: HMM(2) primary; voting(XGBoost+RF) at n=2–3 also strong (post‑silhouette around ~0.32–0.34; separation ~3.3+)

## Summary Table

| Instrument | Timeframe | Recommended | n_regimes | Smoothing | Typical Post Silhouette | Typical Post Separation | Ready |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EUR/USD | M5 | HMM | 2 | 5–7 | ≥ 0.50 | ≥ 2.0 | Yes |
| EUR/USD | M15 | HMM | 2 | 5–7 | ≥ 0.30 | ≥ 1.5 | Yes |
| EUR/USD | M30 | HMM | 2 | 5–7 | ≥ 0.35 | ≥ 1.7 | Yes |
| EUR/USD | H1 | HMM | 2 | 5–7 | ≥ 0.40 | ≥ 2.0 | Yes |
| BTC/USD | M5 | HMM; Voting(XGB+RF) | 2–3 | 5–7 | ~0.30–0.34 (ensemble), higher for HMM | ~3.0–3.4 (ensemble), ≥ 2.0 HMM | Yes* |
| BTC/USD | M15 | HMM | 2 | 5–7 | ≥ 0.30 | ≥ 1.5 | Yes |
| BTC/USD | M30 | HMM | 2 | 5–7 | ≥ 0.32 | ≥ 1.7 | Yes |
| BTC/USD | H1 | HMM; Voting(XGB+RF) | 2–3 | 5–7 | ~0.32–0.34 (ensemble), higher for HMM | ~3.3+ (ensemble), ≥ 2.0 HMM | Yes* |

Notes:
- “Yes*” for ensembles indicates readiness can be achieved post‑smoothing; gate on thresholds per your window.
- Enable denoised features (MRML_USE_DENOISER=1) if your window is particularly noisy (helps most on BTC M5/H1).

## Top‑Ready Configs (latest run)

| Instrument | Timeframe | Configuration | n_regimes | Post Silhouette | Post Separation | Persistence | Min Prop | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EUR/USD | M5 | HMM(full) | 2 | 0.54 | 2.33 | ≥ 0.85 | ~0.19 | regime_ablation_summary_20250924_171139.md |
| EUR/USD | M15 | HMM(full) | 2 | ≥ 0.30 | ≥ 1.50 | ≥ 0.85 | ≥ 0.10 | regime_ablation_summary_20250924_171139.md |
| EUR/USD | M30 | HMM(full) | 2 | ≥ 0.35 | ≥ 1.70 | ≥ 0.85 | ≥ 0.10 | regime_ablation_summary_20250924_171139.md |
| EUR/USD | H1 | HMM(full) | 2 | ≥ 0.40 | ≥ 2.00 | ≥ 0.85 | ≥ 0.10 | regime_ablation_summary_20250924_171139.md |
| BTC/USD | M5 | HMM(full) | 2 | ≥ 0.30 | ≥ 1.50 | ≥ 0.85 | ≥ 0.10 | regime_ablation_summary_20250924_171139.md |
| BTC/USD | H1 | Voting(XGB+RF) | 2–3 | ~0.32–0.34 | ~3.3+ | ≥ 0.85 | ≥ 0.10 | ensemble_ablation_btc_multi_tf_20250924_185009.md |

Notes:
- Exact numeric values per timeframe are available in the linked Markdown reports (see “Best ready configurations” sections in the ensemble reports and each HMM row in the regime‑count summary). The table above highlights the best‑ready configurations and indicative metrics captured during the latest runs.

## Ensemble Guidance (Secondary)
- Strategy: simple voting
- Mix: XGBoost + RandomForest (optionally add HMM/GMM for 3‑model voting)
- Regime counts: 2–3 only
- Gate deployment on the same readiness thresholds using post‑smoothing metrics

## Reproduce and Compare
- Intraday regime‑count ablation (with/without denoiser):
  - Without denoiser: `python -m marketregimeml.benchmarks.regime_ablation_study`
  - With denoiser: `MRML_USE_DENOISER=1 python -m marketregimeml.benchmarks.regime_ablation_study`
  - Reports (examples):
    - `report/regime_ablation_summary_20250924_151413.md` (no denoiser)
    - `report/regime_ablation_summary_20250924_171139.md` (denoiser on)

- Intraday ensemble ablations (before/after denoiser):
  - BTC: `python -m marketregimeml.benchmarks.ensemble_ablation_multi_timeframe`
    - `report/ensemble_ablation_btc_multi_tf_20250924_140810.md` (before)
    - `report/ensemble_ablation_btc_multi_tf_20250924_185009.md` (after)
  - EUR/USD: `python -m marketregimeml.benchmarks.ensemble_ablation_eurusd_multi_timeframe`
    - `report/ensemble_ablation_eurusd_multi_tf_20250924_110341.md` (before)
    - `report/ensemble_ablation_eurusd_multi_tf_20250924_212230.md` (after)

## Production Checklist
- Train/evaluate on your intraday window with FeatureEngine + robust normalization
- Use HMM(2) as the baseline; apply smoothing_window 5–7 in assessment
- Confirm readiness (silhouette, separation, persistence, min cluster proportion)
- (Optional) Compare with XGBoost+RF voting ensembles; deploy only if readiness passes on the same window
- Monitor drift alerts on silhouette (<0.30), separation (<1.50), and min cluster proportion (<0.10)

## Rationale
- Intraday regimes benefit from parsimonious structure; n=2–3 consistently outperform higher counts in separation, silhouette, and stability.
- Label smoothing increases persistence and clarifies cluster boundaries without obscuring price dynamics.
- Denoised features can add robustness in highly noisy conditions without replacing raw inputs, preserving signal integrity.

## Best‑Ready Per Timeframe (latest run)

<!-- AUTO-BEST-READY-START -->
Pending.
<!-- AUTO-BEST-READY-END -->
