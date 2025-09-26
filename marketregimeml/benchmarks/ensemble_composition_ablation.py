"""Ablation study for ensemble composition."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from itertools import combinations

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
from marketregimeml.benchmarks.real_data_benchmark import RealDataLoader
from marketregimeml.benchmarks.reporting import write_ensemble_ablation_markdown


def test_ensemble_combinations(
    features: pd.DataFrame,
    n_regimes: int = 3,
    include_unsupervised: bool = True,
    seeds: List[int] = [0, 1, 2],
) -> Dict:
    """Test different ensemble compositions and strategies with stability sweep."""
    results = {}
    metrics = RegimeMetrics()
    
    # Model factory honoring seeds
    def build_base_models(seed: int):
        models = {
            "HMM": HMMRegimeDetector(n_regimes=n_regimes, random_state=seed),
            "GMM": GMMRegimeDetector(n_regimes=n_regimes, random_state=seed),
            "GARCH": GARCHRegimeDetector(n_regimes=n_regimes, random_state=seed),
            "RandomForest": RandomForestRegimeClassifier(
                n_regimes=n_regimes, n_estimators=300, random_state=seed, class_weight='balanced'
            ),
            "SVM_RBF": SVMRegimeClassifier(
                n_regimes=n_regimes, kernel='rbf', probability=True, random_state=seed, class_weight='balanced'
            ),
            "SVM_Linear": SVMRegimeClassifier(
                n_regimes=n_regimes, kernel='linear', probability=True, random_state=seed, class_weight='balanced'
            ),
        }
        try:
            models["XGBoost"] = XGBoostRegimeClassifier(
                n_regimes=n_regimes, n_estimators=200, random_state=seed
            )
        except Exception:
            pass
        return models
    
    # Test individual models first
    print("\nTesting individual models (stability across seeds)...")
    model_names = list(build_base_models(seeds[0]).keys())
    for model_name in model_names:
        try:
            rqis = []
            preds = []
            for seed in seeds:
                model = build_base_models(seed)[model_name]
                model.fit(features)
                regimes = model.predict(features)
                probabilities = model.predict_proba(features)
                rqis.append(metrics.regime_quality_index(features.values, regimes, probabilities))
                preds.append(regimes)
            # Cross-seed ARI
            from sklearn.metrics import adjusted_rand_score
            pair_aris = []
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    pair_aris.append(adjusted_rand_score(preds[i], preds[j]))
            # Strategy assessment using last run's regimes
            last_regimes = preds[-1]
            # Recompute probabilities for last seed to assess consistency
            _last_model = build_base_models(seeds[-1])[model_name]
            _last_model.fit(features)
            last_prob = _last_model.predict_proba(features)
            stab = metrics.regime_stability(last_regimes)
            assess_pre = strat.assess(
                features=features.values,
                regimes=last_regimes,
                probabilities=last_prob,
                persistence=float(stab.get("persistence", 0.0)),
                thresholds=ReadinessThresholds(smoothing_window=0),
            )
            assess = strat.assess(
                features=features.values,
                regimes=last_regimes,
                probabilities=last_prob,
                persistence=float(stab.get("persistence", 0.0)),
                thresholds=ReadinessThresholds(smoothing_window=5),
            )
            results[model_name] = {
                "rqi": float(np.mean(rqis)),
                "rqi_std": float(np.std(rqis)),
                "seed_ari_mean": float(np.mean(pair_aris)) if pair_aris else 1.0,
                "type": "individual",
                "n_regimes": int(n_regimes),
                "separation_pre": float(assess_pre["separation"]),
                "silhouette_pre": float(assess_pre["silhouette"]),
                "separation": float(assess["separation"]),
                "silhouette": float(assess["silhouette"]),
                "calinski_harabasz": float(assess["calinski_harabasz"]),
                "davies_bouldin": float(assess["davies_bouldin"]),
                "min_cluster_prop": float(assess.get("min_cluster_prop", 0.0)),
                "persistence": float(stab.get("persistence", 0.0)),
                "strategy_ready": bool(assess["strategy_ready"]),
            }
            print(
                f"  {model_name}: RQImean={np.mean(rqis):.1f}±{np.std(rqis):.1f}, Seed ARI={np.mean(pair_aris) if pair_aris else 1.0:.3f}"
            )
        except Exception as e:
            print(f"  {model_name}: Failed - {e}")
            results[model_name] = {"rqi": 0, "type": "individual"}
    
    # Test 2-model ensembles
    print("\nTesting 2-model ensembles...")
    ml_models = ["XGBoost", "RandomForest", "SVM_RBF", "SVM_Linear"]
    ensemble_candidates = (["HMM", "GMM", "GARCH"] + ml_models) if include_unsupervised else ml_models

    for combo in combinations(ensemble_candidates, 2):
        combo_name = f"Ensemble({'+'.join(combo)})"
        try:
            # Evaluate multiple ensemble strategies
            for strategy in ["voting", "weighted_voting", "bayesian"]:
                name = f"{combo_name}:{strategy}"
                rqis = []
                preds = []
                for seed in seeds:
                    models = [build_base_models(seed)[n] for n in combo]
                    ensemble = EnsembleRegimeDetector(
                        models=models, n_regimes=n_regimes, strategy=strategy
                    )
                    if strategy == "weighted_voting":
                        ensemble.weights = [1.0 / len(models)] * len(models)
                    ensemble.fit(features)
                    if strategy == "weighted_voting":
                        try:
                            ensemble.optimize_weights(features, validation_split=0.2)
                        except Exception:
                            pass
                    regimes = ensemble.predict(features)
                    probabilities = ensemble.predict_proba(features)
                    rqis.append(metrics.regime_quality_index(features.values, regimes, probabilities))
                    preds.append(regimes)
                # Cross-seed ARI
                from sklearn.metrics import adjusted_rand_score
                pair_aris = []
                for i in range(len(preds)):
                    for j in range(i + 1, len(preds)):
                        pair_aris.append(adjusted_rand_score(preds[i], preds[j]))
                # Strategy-oriented assessment on last run
                last_regimes = preds[-1]
                last_prob = ensemble.predict_proba(features)
                stab = metrics.regime_stability(last_regimes)
                assess_pre = strat.assess(
                    features=features.values,
                    regimes=last_regimes,
                    probabilities=last_prob,
                    persistence=float(stab.get("persistence", 0.0)),
                    thresholds=ReadinessThresholds(smoothing_window=0),
                )
                assess = strat.assess(
                    features=features.values,
                    regimes=last_regimes,
                    probabilities=last_prob,
                    persistence=float(stab.get("persistence", 0.0)),
                    thresholds=ReadinessThresholds(smoothing_window=5),
                )
                results[name] = {
                    "rqi": float(np.mean(rqis)),
                    "rqi_std": float(np.std(rqis)),
                    "seed_ari_mean": float(np.mean(pair_aris)) if pair_aris else 1.0,
                    "type": "2-model",
                    "n_regimes": int(n_regimes),
                    "separation_pre": float(assess_pre["separation"]),
                    "silhouette_pre": float(assess_pre["silhouette"]),
                    "separation": float(assess["separation"]),
                    "silhouette": float(assess["silhouette"]),
                    "calinski_harabasz": float(assess["calinski_harabasz"]),
                    "davies_bouldin": float(assess["davies_bouldin"]),
                    "min_cluster_prop": float(assess.get("min_cluster_prop", 0.0)),
                    "persistence": float(stab.get("persistence", 0.0)),
                    "strategy_ready": bool(assess["strategy_ready"]),
                }
                print(
                    f"  {name}: RQImean={np.mean(rqis):.1f}±{np.std(rqis):.1f}, Seed ARI={np.mean(pair_aris) if pair_aris else 1.0:.3f}, "
                    f"Sep={assess['separation']:.3f}, Sil={assess['silhouette']:.3f}, CH={assess['calinski_harabasz']:.1f}, DBI={assess['davies_bouldin']:.3f}"
                )
        except Exception as e:
            print(f"  {combo_name}: Failed - {e}")
            results[combo_name] = {"rqi": 0, "type": "2-model"}
    
    # Test 3-model ensembles
    print("\nTesting 3-model ensembles...")
    for combo in combinations(ensemble_candidates, 3):
        combo_name = f"Ensemble({'+'.join(combo)})"
        try:
            for strategy in ["voting", "weighted_voting", "bayesian"]:
                name = f"{combo_name}:{strategy}"
                rqis = []
                preds = []
                for seed in seeds:
                    models = [build_base_models(seed)[n] for n in combo]
                    ensemble = EnsembleRegimeDetector(
                        models=models, n_regimes=n_regimes, strategy=strategy
                    )
                    if strategy == "weighted_voting":
                        ensemble.weights = [1.0 / len(models)] * len(models)
                    ensemble.fit(features)
                    if strategy == "weighted_voting":
                        try:
                            ensemble.optimize_weights(features, validation_split=0.2)
                        except Exception:
                            pass
                    regimes = ensemble.predict(features)
                    probabilities = ensemble.predict_proba(features)
                    rqis.append(metrics.regime_quality_index(features.values, regimes, probabilities))
                    preds.append(regimes)
                from sklearn.metrics import adjusted_rand_score
                pair_aris = []
                for i in range(len(preds)):
                    for j in range(i + 1, len(preds)):
                        pair_aris.append(adjusted_rand_score(preds[i], preds[j]))
                last_regimes = preds[-1]
                last_prob = ensemble.predict_proba(features)
                stab = metrics.regime_stability(last_regimes)
                assess_pre = strat.assess(
                    features=features.values,
                    regimes=last_regimes,
                    probabilities=last_prob,
                    persistence=float(stab.get("persistence", 0.0)),
                    thresholds=ReadinessThresholds(smoothing_window=0),
                )
                assess = strat.assess(
                    features=features.values,
                    regimes=last_regimes,
                    probabilities=last_prob,
                    persistence=float(stab.get("persistence", 0.0)),
                    thresholds=ReadinessThresholds(smoothing_window=5),
                )
                results[name] = {
                    "rqi": float(np.mean(rqis)),
                    "rqi_std": float(np.std(rqis)),
                    "seed_ari_mean": float(np.mean(pair_aris)) if pair_aris else 1.0,
                    "type": "3-model",
                    "n_regimes": int(n_regimes),
                    "separation_pre": float(assess_pre["separation"]),
                    "silhouette_pre": float(assess_pre["silhouette"]),
                    "separation": float(assess["separation"]),
                    "silhouette": float(assess["silhouette"]),
                    "calinski_harabasz": float(assess["calinski_harabasz"]),
                    "davies_bouldin": float(assess["davies_bouldin"]),
                    "min_cluster_prop": float(assess.get("min_cluster_prop", 0.0)),
                    "persistence": float(stab.get("persistence", 0.0)),
                    "strategy_ready": bool(assess["strategy_ready"]),
                }
                print(
                    f"  {name}: RQImean={np.mean(rqis):.1f}±{np.std(rqis):.1f}, Seed ARI={np.mean(pair_aris) if pair_aris else 1.0:.3f}, "
                    f"Sep={assess['separation']:.3f}, Sil={assess['silhouette']:.3f}, CH={assess['calinski_harabasz']:.1f}, DBI={assess['davies_bouldin']:.3f}"
                )
        except Exception as e:
            print(f"  {combo_name}: Failed - {e}")
            results[combo_name] = {"rqi": 0, "type": "3-model"}
    
    # Test 4-model ensemble (all ML models) with seeds
    print("\nTesting 4-model ensemble...")
    ensemble_name = "Ensemble(All_ML)"
    try:
        for strategy in ["voting", "weighted_voting", "bayesian"]:
            name = f"{ensemble_name}:{strategy}"
            rqis = []
            preds = []
            for seed in seeds:
                models = [
                    XGBoostRegimeClassifier(n_regimes=n_regimes, n_estimators=200, random_state=seed),
                    RandomForestRegimeClassifier(n_regimes=n_regimes, n_estimators=300, random_state=seed),
                    SVMRegimeClassifier(n_regimes=n_regimes, kernel='rbf', probability=True, random_state=seed),
                    SVMRegimeClassifier(n_regimes=n_regimes, kernel='linear', probability=True, random_state=seed),
                ]
                ensemble = EnsembleRegimeDetector(models=models, n_regimes=n_regimes, strategy=strategy)
                if strategy == "weighted_voting":
                    ensemble.weights = [1.0 / len(models)] * len(models)
                ensemble.fit(features)
                if strategy == "weighted_voting":
                    try:
                        ensemble.optimize_weights(features, validation_split=0.2)
                    except Exception:
                        pass
                regimes = ensemble.predict(features)
                probabilities = ensemble.predict_proba(features)
                rqis.append(metrics.regime_quality_index(features.values, regimes, probabilities))
                preds.append(regimes)
            from sklearn.metrics import adjusted_rand_score
            pair_aris = []
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    pair_aris.append(adjusted_rand_score(preds[i], preds[j]))
            last_regimes = preds[-1]
            last_prob = ensemble.predict_proba(features)
            stab = metrics.regime_stability(last_regimes)
            assess_pre = strat.assess(
                features=features.values,
                regimes=last_regimes,
                probabilities=last_prob,
                persistence=float(stab.get("persistence", 0.0)),
                thresholds=ReadinessThresholds(smoothing_window=0),
            )
            assess = strat.assess(
                features=features.values,
                regimes=last_regimes,
                probabilities=last_prob,
                persistence=float(stab.get("persistence", 0.0)),
                thresholds=ReadinessThresholds(smoothing_window=5),
            )
            results[name] = {
                "rqi": float(np.mean(rqis)),
                "rqi_std": float(np.std(rqis)),
                "seed_ari_mean": float(np.mean(pair_aris)) if pair_aris else 1.0,
                "type": "4-model",
                "n_regimes": int(n_regimes),
                "separation_pre": float(assess_pre["separation"]),
                "silhouette_pre": float(assess_pre["silhouette"]),
                "separation": float(assess["separation"]),
                "silhouette": float(assess["silhouette"]),
                "calinski_harabasz": float(assess["calinski_harabasz"]),
                "davies_bouldin": float(assess["davies_bouldin"]),
                "min_cluster_prop": float(assess.get("min_cluster_prop", 0.0)),
                "persistence": float(stab.get("persistence", 0.0)),
                "strategy_ready": bool(assess["strategy_ready"]),
            }
            print(
                f"  {name}: RQImean={np.mean(rqis):.1f}±{np.std(rqis):.1f}, Seed ARI={np.mean(pair_aris) if pair_aris else 1.0:.3f}, "
                f"Sep={assess['separation']:.3f}, Sil={assess['silhouette']:.3f}, CH={assess['calinski_harabasz']:.1f}, DBI={assess['davies_bouldin']:.3f}"
            )
    except Exception as e:
        print(f"  {ensemble_name}: Failed - {e}")
        results[ensemble_name] = {"rqi": 0, "type": "4-model"}
    
    return results


def run_ensemble_ablation():
    """Run comprehensive ensemble composition ablation study."""
    print("=" * 80)
    print("ENSEMBLE COMPOSITION ABLATION STUDY")
    print("Finding optimal ensemble combinations")
    print("=" * 80)
    
    # Load data (intraday FX focus)
    loader = RealDataLoader()
    
    days_2y = 365 * 2
    limit_m5 = int(days_2y * (24 * 60 / 5))
    limit_m1 = int(days_2y * (24 * 60))

    datasets = {
        "EUR/USD (M5)": loader.load_forex_data("EUR_USD", "M5", limit_m5),
    }
    
    all_results = {}
    
    for dataset_name, data in datasets.items():
        if data is None:
            print(f"\nSkipping {dataset_name} - failed to load")
            continue
            
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Samples: {len(data)}")
        print(f"{'='*60}")
        
        # Test different ensemble combinations (include unsupervised models too)
        results = test_ensemble_combinations(
            data, n_regimes=3, include_unsupervised=True, seeds=[0, 1, 2]
        )
        all_results[dataset_name] = results
    
    # Print summary
    print("\n" + "=" * 80)
    print("ENSEMBLE ABLATION SUMMARY")
    print("=" * 80)
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        print("-" * 40)
        
        # Sort by RQI
        sorted_results = sorted(dataset_results.items(), key=lambda x: x[1]["rqi"], reverse=True)
        
        # Show top 10
        print("\nTop 10 configurations:")
        for i, (name, metrics) in enumerate(sorted_results[:10], 1):
            print(f"  {i}. {name}: RQI={metrics['rqi']:.1f} ({metrics['type']})")
        
        # Best by type
        print("\nBest by ensemble size:")
        for ensemble_type in ["individual", "2-model", "3-model", "4-model"]:
            type_results = [(n, m) for n, m in sorted_results if m["type"] == ensemble_type]
            if type_results:
                best = type_results[0]
                print(f"  {ensemble_type}: {best[0]} (RQI={best[1]['rqi']:.1f})")
    
    # Overall conclusions
    print("\n" + "=" * 80)
    print("OVERALL CONCLUSIONS:")
    print("=" * 80)
    
    # Calculate average RQI for each configuration across all datasets
    config_totals = {}
    
    for dataset_results in all_results.values():
        for name, metrics in dataset_results.items():
            if name not in config_totals:
                config_totals[name] = []
            if metrics["rqi"] > 0:  # Exclude failed runs
                config_totals[name].append(metrics["rqi"])
    
    # Calculate averages and sort
    config_averages = {}
    for name, rqis in config_totals.items():
        if rqis:
            config_averages[name] = np.mean(rqis)
    
    sorted_averages = sorted(config_averages.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 configurations across all datasets:")
    for i, (name, avg_rqi) in enumerate(sorted_averages[:5], 1):
        n_tests = len(config_totals[name])
        print(f"  {i}. {name}: {avg_rqi:.1f} average RQI (n={n_tests} datasets)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ensemble_ablation_results_{timestamp}.txt"
    
    with open(results_file, "w") as f:
        f.write("ENSEMBLE COMPOSITION ABLATION STUDY RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for dataset_name, dataset_results in all_results.items():
            f.write(f"{dataset_name}:\n")
            f.write("-" * 40 + "\n")
            
            sorted_results = sorted(dataset_results.items(), key=lambda x: x[1]["rqi"], reverse=True)
            for name, metrics in sorted_results:
                f.write(f"  {name}: RQI={metrics['rqi']:.1f} ({metrics['type']})\n")
            f.write("\n")
    
    print(f"\nResults saved to {results_file}")
    # Save Markdown summary alongside text
    md_path = f"report/ensemble_ablation_summary_{timestamp}.md"
    try:
        write_ensemble_ablation_markdown(all_results, md_path)
        print(f"Markdown report saved to {md_path}")
    except Exception as e:
        print(f"Markdown export skipped: {e}")
    
    # Final recommendation
    print("\n" + "=" * 80)
    print("✅ RECOMMENDATION:")
    best_config = sorted_averages[0]
    print(f"   Best ensemble: {best_config[0]}")
    print(f"   Average RQI: {best_config[1]:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    run_ensemble_ablation()
