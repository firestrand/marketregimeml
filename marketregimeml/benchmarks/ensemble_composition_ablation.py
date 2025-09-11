"""Ablation study for ensemble composition."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
from itertools import combinations

from marketregimeml.models import (
    HMMRegimeDetector,
    GMMRegimeDetector,
    GARCHRegimeDetector,
    XGBoostRegimeClassifier,
    SVMRegimeClassifier,
    RandomForestRegimeClassifier,
    EnsembleRegimeDetector
)
from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.benchmarks.real_data_benchmark import RealDataLoader


def test_ensemble_combinations(features: pd.DataFrame, n_regimes: int = 3) -> Dict:
    """Test different ensemble compositions."""
    results = {}
    metrics = RegimeMetrics()
    
    # Available models for ensemble
    base_models = {
        "HMM": HMMRegimeDetector(n_regimes=n_regimes, random_state=42),
        "GMM": GMMRegimeDetector(n_regimes=n_regimes, random_state=42),
        "GARCH": GARCHRegimeDetector(n_regimes=n_regimes, random_state=42),
        "XGBoost": XGBoostRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42),
        "RandomForest": RandomForestRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42),
        "SVM_RBF": SVMRegimeClassifier(n_regimes=n_regimes, kernel='rbf', probability=True, random_state=42),
        "SVM_Linear": SVMRegimeClassifier(n_regimes=n_regimes, kernel='linear', probability=True, random_state=42)
    }
    
    # Test individual models first
    print("\nTesting individual models...")
    for model_name, model in base_models.items():
        try:
            model.fit(features)
            regimes = model.predict(features)
            probabilities = model.predict_proba(features)
            rqi = metrics.regime_quality_index(features.values, regimes, probabilities)
            results[model_name] = {"rqi": rqi, "type": "individual"}
            print(f"  {model_name}: RQI={rqi:.1f}")
        except Exception as e:
            print(f"  {model_name}: Failed - {e}")
            results[model_name] = {"rqi": 0, "type": "individual"}
    
    # Test 2-model ensembles
    print("\nTesting 2-model ensembles...")
    ml_models = ["XGBoost", "RandomForest", "SVM_RBF", "SVM_Linear"]
    
    for combo in combinations(ml_models, 2):
        ensemble_name = f"Ensemble({'+'.join(combo)})"
        try:
            models = []
            for model_name in combo:
                if model_name == "XGBoost":
                    models.append(XGBoostRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42))
                elif model_name == "RandomForest":
                    models.append(RandomForestRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42))
                elif model_name == "SVM_RBF":
                    models.append(SVMRegimeClassifier(n_regimes=n_regimes, kernel='rbf', probability=True, random_state=42))
                elif model_name == "SVM_Linear":
                    models.append(SVMRegimeClassifier(n_regimes=n_regimes, kernel='linear', probability=True, random_state=42))
            
            ensemble = EnsembleRegimeDetector(models=models, n_regimes=n_regimes, strategy='voting')
            ensemble.fit(features)
            regimes = ensemble.predict(features)
            probabilities = ensemble.predict_proba(features)
            rqi = metrics.regime_quality_index(features.values, regimes, probabilities)
            results[ensemble_name] = {"rqi": rqi, "type": "2-model"}
            print(f"  {ensemble_name}: RQI={rqi:.1f}")
        except Exception as e:
            print(f"  {ensemble_name}: Failed - {e}")
            results[ensemble_name] = {"rqi": 0, "type": "2-model"}
    
    # Test 3-model ensembles
    print("\nTesting 3-model ensembles...")
    for combo in combinations(ml_models, 3):
        ensemble_name = f"Ensemble({'+'.join(combo)})"
        try:
            models = []
            for model_name in combo:
                if model_name == "XGBoost":
                    models.append(XGBoostRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42))
                elif model_name == "RandomForest":
                    models.append(RandomForestRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42))
                elif model_name == "SVM_RBF":
                    models.append(SVMRegimeClassifier(n_regimes=n_regimes, kernel='rbf', probability=True, random_state=42))
                elif model_name == "SVM_Linear":
                    models.append(SVMRegimeClassifier(n_regimes=n_regimes, kernel='linear', probability=True, random_state=42))
            
            ensemble = EnsembleRegimeDetector(models=models, n_regimes=n_regimes, strategy='voting')
            ensemble.fit(features)
            regimes = ensemble.predict(features)
            probabilities = ensemble.predict_proba(features)
            rqi = metrics.regime_quality_index(features.values, regimes, probabilities)
            results[ensemble_name] = {"rqi": rqi, "type": "3-model"}
            print(f"  {ensemble_name}: RQI={rqi:.1f}")
        except Exception as e:
            print(f"  {ensemble_name}: Failed - {e}")
            results[ensemble_name] = {"rqi": 0, "type": "3-model"}
    
    # Test 4-model ensemble (all ML models)
    print("\nTesting 4-model ensemble...")
    ensemble_name = "Ensemble(All_ML)"
    try:
        models = [
            XGBoostRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42),
            RandomForestRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42),
            SVMRegimeClassifier(n_regimes=n_regimes, kernel='rbf', probability=True, random_state=42),
            SVMRegimeClassifier(n_regimes=n_regimes, kernel='linear', probability=True, random_state=42)
        ]
        
        ensemble = EnsembleRegimeDetector(models=models, n_regimes=n_regimes, strategy='voting')
        ensemble.fit(features)
        regimes = ensemble.predict(features)
        probabilities = ensemble.predict_proba(features)
        rqi = metrics.regime_quality_index(features.values, regimes, probabilities)
        results[ensemble_name] = {"rqi": rqi, "type": "4-model"}
        print(f"  {ensemble_name}: RQI={rqi:.1f}")
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
    
    # Load data
    loader = RealDataLoader()
    
    datasets = {
        "SPY (Stocks)": loader.load_spy(500),
        "EUR/USD (Forex)": loader.load_forex_data("EUR_USD", "D", 500),
        "BTC/USD (Crypto)": loader.load_crypto("BTC/USD", 500)
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
        
        # Test different ensemble combinations
        results = test_ensemble_combinations(data, n_regimes=3)
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
    
    # Final recommendation
    print("\n" + "=" * 80)
    print("✅ RECOMMENDATION:")
    best_config = sorted_averages[0]
    print(f"   Best ensemble: {best_config[0]}")
    print(f"   Average RQI: {best_config[1]:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    run_ensemble_ablation()