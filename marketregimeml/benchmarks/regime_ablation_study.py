"""Ablation study for optimal n_regimes parameter."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

from marketregimeml.models import (
    HMMRegimeDetector,
    XGBoostRegimeClassifier,
    SVMRegimeClassifier,
    RandomForestRegimeClassifier,
    EnsembleRegimeDetector
)
from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.benchmarks.real_data_benchmark import RealDataLoader


def test_regime_counts(features: pd.DataFrame, regime_counts: List[int] = [2, 3, 4, 5, 7]) -> Dict:
    """Test different n_regimes values."""
    results = {}
    metrics = RegimeMetrics()
    
    for n_regimes in regime_counts:
        print(f"\nTesting n_regimes={n_regimes}")
        model_results = {}
        
        # Test individual models
        models = {
            "HMM": HMMRegimeDetector(n_regimes=n_regimes, random_state=42),
            "XGBoost": XGBoostRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42),
            "SVM (RBF)": SVMRegimeClassifier(n_regimes=n_regimes, kernel='rbf', probability=True, random_state=42),
            "Fast SVM Ensemble": EnsembleRegimeDetector(
                models=[
                    SVMRegimeClassifier(n_regimes=n_regimes, kernel='rbf', probability=True, random_state=42),
                    SVMRegimeClassifier(n_regimes=n_regimes, kernel='linear', probability=True, random_state=42)
                ],
                n_regimes=n_regimes,
                strategy='voting'
            ),
            "Top3 Ensemble": EnsembleRegimeDetector(
                models=[
                    XGBoostRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42),
                    SVMRegimeClassifier(n_regimes=n_regimes, kernel='rbf', probability=True, random_state=42),
                    RandomForestRegimeClassifier(n_regimes=n_regimes, n_estimators=100, random_state=42)
                ],
                n_regimes=n_regimes,
                strategy='voting'
            )
        }
        
        for model_name, model in models.items():
            try:
                # Fit and predict
                model.fit(features)
                regimes = model.predict(features)
                probabilities = model.predict_proba(features)
                
                # Calculate RQI
                rqi = metrics.regime_quality_index(features.values, regimes, probabilities)
                stability = metrics.regime_stability(regimes)
                
                model_results[model_name] = {
                    "rqi": rqi,
                    "stability": stability['persistence'] * 100,
                    "avg_duration": stability['avg_duration']
                }
                
                print(f"  {model_name}: RQI={rqi:.1f}, Stability={stability['persistence']*100:.1f}%")
                
            except Exception as e:
                print(f"  {model_name}: Failed - {e}")
                model_results[model_name] = {"rqi": 0, "stability": 0, "avg_duration": 0}
        
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
        
        # Test different regime counts
        results = test_regime_counts(data, [2, 3, 4, 5, 7])
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
            for model_name, metrics in models.items():
                row[f"{model_name}_RQI"] = metrics["rqi"]
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
    regime_totals = {2: [], 3: [], 4: [], 5: [], 7: []}
    
    for dataset_results in all_results.values():
        for n_regimes, models in dataset_results.items():
            for metrics in models.values():
                if metrics["rqi"] > 0:  # Exclude failed runs
                    regime_totals[n_regimes].append(metrics["rqi"])
    
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
                for model_name, metrics in models.items():
                    f.write(f"  {model_name}: RQI={metrics['rqi']:.1f}, Stability={metrics['stability']:.1f}%\n")
            f.write("\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    run_ablation_study()