"""Ablation study for optimal n_regimes parameter."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

from marketregimeml.models import (
    HMMRegimeDetector,
    GMMRegimeDetector,
    GARCHRegimeDetector,
    XGBoostRegimeClassifier,
    SVMRegimeClassifier,
    RandomForestRegimeClassifier,
    EnsembleRegimeDetector,
)
from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.evaluation import metrics_functions as mf
from marketregimeml.evaluation import strategy as strat
from marketregimeml.evaluation.strategy import ReadinessThresholds
from marketregimeml.benchmarks.reporting import write_regime_ablation_markdown
from marketregimeml.benchmarks.real_data_benchmark import RealDataLoader
from marketregimeml.data.storage.duckdb_store import DuckDBStore
from marketregimeml.data.loaders.kraken import KrakenDataLoader
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def test_regime_counts(
    features: pd.DataFrame,
    regime_counts: List[int] = [2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 21, 25, 29, 33],
    seeds: List[int] = [0, 1, 2],
) -> Dict:
    """Test different n_regimes values with stability sweep across seeds."""
    results = {}
    metrics = RegimeMetrics()
    
    for n_regimes in regime_counts:
        print(f"\nTesting n_regimes={n_regimes}")
        model_results = {}
        
        def build_models(seed: int) -> Dict[str, object]:
            models: Dict[str, object] = {
                "HMM": HMMRegimeDetector(n_regimes=n_regimes, random_state=seed),
                "GMM": GMMRegimeDetector(n_regimes=n_regimes, random_state=seed),
                "GARCH": GARCHRegimeDetector(n_regimes=n_regimes, random_state=seed),
                "RandomForest": RandomForestRegimeClassifier(
                    n_regimes=n_regimes, n_estimators=300, random_state=seed, class_weight='balanced'
                ),
                "SVM (RBF)": SVMRegimeClassifier(
                    n_regimes=n_regimes, kernel='rbf', probability=True, random_state=seed, class_weight='balanced'
                ),
            }
            # Optional XGBoost
            try:
                _xgb = XGBoostRegimeClassifier(
                    n_regimes=n_regimes, n_estimators=200, random_state=seed
                )
                models["XGBoost"] = _xgb
            except Exception:
                pass
            # Ensembles
            fast_svm = EnsembleRegimeDetector(
                models=[
                    SVMRegimeClassifier(
                        n_regimes=n_regimes, kernel='rbf', probability=True, random_state=seed
                    ),
                    SVMRegimeClassifier(
                        n_regimes=n_regimes, kernel='linear', probability=True, random_state=seed
                    ),
                ],
                n_regimes=n_regimes,
                strategy='voting',
            )
            models["Fast SVM Ensemble"] = fast_svm
            # Top3 ensemble only if XGBoost available
            if "XGBoost" in models:
                top3 = EnsembleRegimeDetector(
                    models=[
                        models["XGBoost"],
                        SVMRegimeClassifier(
                            n_regimes=n_regimes, kernel='rbf', probability=True, random_state=seed
                        ),
                        RandomForestRegimeClassifier(
                            n_regimes=n_regimes, n_estimators=300, random_state=seed
                        ),
                    ],
                    n_regimes=n_regimes,
                    strategy='voting',
                )
                models["Top3 Ensemble"] = top3
            return models

        # Stability sweep across seeds for each model name
        model_names = list(build_models(seeds[0]).keys())
        for model_name in model_names:
            rqi_scores: List[float] = []
            regimes_runs: List[np.ndarray] = []
            proba_runs: List[np.ndarray] = []
            try:
                for seed in seeds:
                    model = build_models(seed)[model_name]
                    model.fit(features)
                    regimes = model.predict(features)
                    probabilities = model.predict_proba(features)
                    regimes_runs.append(regimes)
                    proba_runs.append(probabilities)
                    rqi_scores.append(
                        metrics.regime_quality_index(
                            features.values, regimes, probabilities
                        )
                    )

                # Compute cross-seed stability (pairwise ARI/NMI scores)
                pair_aris: List[float] = []
                pair_nmis: List[float] = []
                for i in range(len(regimes_runs)):
                    for j in range(i + 1, len(regimes_runs)):
                        pair_aris.append(
                            adjusted_rand_score(regimes_runs[i], regimes_runs[j])
                        )
                        pair_nmis.append(
                            normalized_mutual_info_score(
                                regimes_runs[i], regimes_runs[j]
                            )
                        )

                # Use the last run's artifacts for component metrics
                regimes = regimes_runs[-1]
                probabilities = proba_runs[-1]
                stability = metrics.regime_stability(regimes)
                dist = metrics.regime_distribution(regimes)
                conf = metrics.confidence_metrics(probabilities)
                temporal = metrics.temporal_consistency_metrics(regimes)
                # Strategy-oriented assessment (DRY via strategy.assess)
                thresholds = ReadinessThresholds(smoothing_window=5)
                assess = strat.assess(
                    features=features.values,
                    regimes=regimes,
                    probabilities=probabilities,
                    persistence=float(stability["persistence"]),
                    thresholds=thresholds,
                )

                model_results[model_name] = {
                    "rqi_mean": float(np.mean(rqi_scores)),
                    "rqi_std": float(np.std(rqi_scores)),
                    "seed_ari_mean": float(np.mean(pair_aris)) if pair_aris else 1.0,
                    "seed_nmi_mean": float(np.mean(pair_nmis)) if pair_nmis else 1.0,
                    "stability_pct": float(stability["persistence"] * 100.0),
                    "avg_duration": float(stability["avg_duration"]),
                    "separation": float(assess["separation"]),
                    "silhouette": float(assess["silhouette"]),
                    "davies_bouldin": float(assess["davies_bouldin"]),
                    "calinski_harabasz": float(assess["calinski_harabasz"]),
                    "normalized_entropy": float(dist["normalized_entropy"]),
                    "avg_confidence": float(conf["avg_confidence"]),
                    "avg_margin": float(conf["avg_margin"]),
                    "avg_local_consistency": float(
                        temporal["avg_local_consistency"]
                    ),
                    "autocorr_lag1": float(temporal["autocorr_lag1"]),
                    "min_cluster_prop": float(assess["min_cluster_prop"]),
                    "n_regimes_observed": int(stability["n_regimes_observed"]),
                    "strategy_ready": bool(assess["strategy_ready"]),
                }

                print(
                    f"  {model_name}: RQI(mean)={np.mean(rqi_scores):.1f}±{np.std(rqi_scores):.1f}, Seed ARI={np.mean(pair_aris) if pair_aris else 1.0:.3f}"
                )
            except Exception as e:
                print(f"  {model_name}: Failed - {e}")
                model_results[model_name] = {
                    "rqi_mean": 0.0,
                    "rqi_std": 0.0,
                    "seed_ari_mean": 0.0,
                    "seed_nmi_mean": 0.0,
                }
        
        results[n_regimes] = model_results
    
    return results


def run_ablation_study():
    """Run comprehensive ablation study."""
    print("=" * 80)
    print("REGIME COUNT ABLATION STUDY")
    print("Testing optimal n_regimes parameter on real market data")
    print("=" * 80)
    
    # Load data
    loader = RealDataLoader()
    store = DuckDBStore()

    # Compute approximate limits for ~2 years
    days_2y = 365 * 2
    limit_m5 = int(days_2y * (24 * 60 / 5))  # 2y of 5-min bars

    # Helper to build features from OHLCV using the same logic as loader
    def build_features_from_ohlcv(df):
        if df is None or len(df) == 0:
            return None
        return loader._prepare_features(df)  # reuse internal helper for consistency

    datasets = {}
    # EUR/USD M5 base and aggregates
    eurusd_m5 = loader.load_forex_data("EUR_USD", "M5", limit_m5)
    datasets["EUR/USD (M5)"] = eurusd_m5
    for tf, div in [("M15", 3), ("M30", 6), ("H1", 12)]:
        try:
            agg = store.read_ohlcv_aggregated("EUR_USD", "M5", tf)
            if not agg.empty:
                datasets[f"EUR/USD ({tf})"] = build_features_from_ohlcv(agg.iloc[-limit_m5 // div :])
        except Exception:
            pass

    # BTC intraday: ensure M5 exists, then aggregate to M15/M30/H1
    try:
        btc_m5 = store.read_ohlcv("BTC/USD", "M5")
        if btc_m5.empty:
            # fetch and store M5 if missing
            kl = KrakenDataLoader()
            end = datetime.utcnow()
            start = end - timedelta(minutes=limit_m5 * 5)
            fetched = kl.fetch_ohlcv("BTC/USD", timeframe="M5", start=start, end=end)
            if not fetched.empty:
                store.write_ohlcv(fetched, "BTC/USD", "M5")
                btc_m5 = store.read_ohlcv("BTC/USD", "M5")
        if not btc_m5.empty:
            datasets["BTC/USD (M5)"] = build_features_from_ohlcv(btc_m5.iloc[-limit_m5:])
            for tf, div in [("M15", 3), ("M30", 6), ("H1", 12)]:
                try:
                    agg = store.read_ohlcv_aggregated("BTC/USD", "M5", tf)
                    if not agg.empty:
                        datasets[f"BTC/USD ({tf})"] = build_features_from_ohlcv(agg.iloc[-limit_m5 // div :])
                except Exception:
                    pass
    except Exception:
        pass
    
    all_results = {}
    
    for dataset_name, data in datasets.items():
        if data is None:
            print(f"\nSkipping {dataset_name} - failed to load")
            continue
            
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Samples: {len(data)}")
        print(f"{'='*60}")
        
        # Test different regime counts with stability sweep
        results = test_regime_counts(
            data,
            regime_counts=[2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 21, 25, 29, 33],
            seeds=[0, 1, 2],
        )
        all_results[dataset_name] = results
    
    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        print("-" * 40)
        
        # Create comparison table
        df_data = []
        for n_regimes, models in dataset_results.items():
            row = {"n_regimes": n_regimes}
            for model_name, m in models.items():
                rqi_val = m.get("rqi_mean", m.get("rqi", 0.0))
                row[f"{model_name}_RQI"] = rqi_val
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Find best n_regimes for each model
        print("\nBest n_regimes per model:")
        for model_name in ["HMM", "XGBoost", "SVM (RBF)", "Fast SVM Ensemble", "Top3 Ensemble"]:
            col_name = f"{model_name}_RQI"
            if col_name in df.columns:
                best_idx = df[col_name].idxmax()
                best_n = df.loc[best_idx, "n_regimes"]
                best_rqi = df.loc[best_idx, col_name]
                print(f"  {model_name}: n_regimes={best_n} (RQI={best_rqi:.1f})")
    
    # Overall conclusion
    print("\n" + "=" * 80)
    print("CONCLUSIONS:")
    print("=" * 80)
    
    # Calculate average RQI for each n_regimes across all datasets and models
    regime_totals = {k: [] for k in [2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 21, 25, 29, 33]}
    
    for dataset_results in all_results.values():
        for n_regimes, models in dataset_results.items():
            for m in models.values():
                rqi_val = m.get("rqi_mean", m.get("rqi", 0.0))
                if rqi_val > 0:
                    regime_totals[n_regimes].append(rqi_val)
    
    print("\nAverage RQI across all models and datasets:")
    for n_regimes, rqis in regime_totals.items():
        if rqis:
            avg_rqi = np.mean(rqis)
            print(f"  n_regimes={n_regimes}: {avg_rqi:.1f} (n={len(rqis)} tests)")
    
    # Find overall winner
    best_n = max(regime_totals.keys(), key=lambda k: np.mean(regime_totals[k]) if regime_totals[k] else 0)
    print(f"\n✅ OPTIMAL: n_regimes={best_n} provides best overall performance")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"regime_ablation_results_{timestamp}.txt"
    
    with open(results_file, "w") as f:
        f.write("REGIME COUNT ABLATION STUDY RESULTS\n")
        f.write("=" * 80 + "\n\n")
        for dataset_name, dataset_results in all_results.items():
            f.write(f"{dataset_name}:\n")
            f.write("-" * 40 + "\n")
            for n_regimes, models in dataset_results.items():
                f.write(f"\nn_regimes={n_regimes}:\n")
                for model_name, m in models.items():
                    rqm = m.get("rqi_mean", m.get("rqi", 0.0))
                    rqs = m.get("rqi_std", 0.0)
                    f.write(
                        (
                            "  {name}: RQImean={rqm:.1f}±{rqs:.1f}, SeedARI={ari:.3f}, SeedNMI={nmi:.3f}, "
                            "Stab={stab:.1f}%, Sep={sep:.3f}, Sil={sil:.3f}, DBI={dbi:.3f}, CH={ch:.1f}, MinProp={mp:.2f}, "
                            "Conf={conf:.3f}, EntN={ent:.3f}, LCons={lcons:.3f}, AR1={ar1:.3f}, Ready={ready}\n"
                        ).format(
                            name=model_name,
                            rqm=rqm,
                            rqs=rqs,
                            ari=m.get("seed_ari_mean", 0.0),
                            nmi=m.get("seed_nmi_mean", 0.0),
                            stab=m.get("stability_pct", 0.0),
                            sep=m.get("separation", 0.0),
                            sil=m.get("silhouette", 0.0),
                            dbi=m.get("davies_bouldin", 0.0),
                            ch=m.get("calinski_harabasz", 0.0),
                            mp=m.get("min_cluster_prop", 0.0),
                            conf=m.get("avg_confidence", 0.0),
                            ent=m.get("normalized_entropy", 0.0),
                            lcons=m.get("avg_local_consistency", 0.0),
                            ar1=m.get("autocorr_lag1", 0.0),
                            ready=m.get("strategy_ready", False),
                        )
                    )
            f.write("\n")
    
    print(f"\nResults saved to {results_file}")

    # Save Markdown report alongside text results (for human scanning)
    md_path = f"report/regime_ablation_summary_{timestamp}.md"
    try:
        write_regime_ablation_markdown(all_results, md_path)
        print(f"Markdown report saved to {md_path}")
    except Exception as e:
        print(f"Markdown export skipped: {e}")


if __name__ == "__main__":
    run_ablation_study()
