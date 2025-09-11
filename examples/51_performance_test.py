#!/usr/bin/env python
"""Simple and robust performance test of regime detection models."""

import sys
import time
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector
from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.data.loaders import KrakenDataLoader

# Suppress warnings
warnings.filterwarnings('ignore')


def create_simple_dataset(n_samples: int = 300) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Create a dataset from real BTC/USD data with basic features."""
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=600)
    ohlcv = loader.fetch_ohlcv("BTC/USD", "1d", start, end)
    returns = ohlcv['close'].pct_change().dropna()
    returns = returns.iloc[-n_samples:]
    
    features = pd.DataFrame(index=returns.index)
    features['returns'] = returns
    features['abs_returns'] = returns.abs()
    features['squared_returns'] = returns.values ** 2
    features['rolling_mean_10'] = returns.rolling(10, min_periods=1).mean()
    features['rolling_std_10'] = returns.rolling(10, min_periods=1).std()
    features = features.dropna()
    
    # No true labels for real data; provide single-class placeholder to avoid supervised metrics
    true_regimes = np.zeros(len(features), dtype=int)
    
    return features, returns.loc[features.index], true_regimes


def test_single_model(
    model_class,
    model_params: Dict[str, Any],
    features: pd.DataFrame,
    returns: pd.Series,
    true_regimes: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """Test a single model and return metrics."""
    
    print(f"  Testing {model_name}...")
    
    try:
        start_time = time.time()
        
        # Create and fit model
        model = model_class(**model_params)
        model.fit(features)
        
        # Get predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        fit_time = time.time() - start_time
        
        # Basic validation
        unique_predictions = np.unique(predictions)
        n_predicted_regimes = len(unique_predictions)
        
        print(f"    Fit time: {fit_time:.2f}s")
        print(f"    Predicted regimes: {sorted(unique_predictions)}")
        print(f"    Number of regime changes: {np.sum(np.diff(predictions) != 0)}")
        
        # Calculate metrics
        results = {
            'model_name': model_name,
            'success': True,
            'fit_time': fit_time,
            'n_predicted_regimes': n_predicted_regimes,
            'n_regime_changes': np.sum(np.diff(predictions) != 0)
        }
        
        # Only calculate metrics if we have multiple regimes
        if n_predicted_regimes > 1:
            # Clustering metrics
            silhouette = RegimeMetrics.silhouette_coefficient(features.values, predictions)
            results['silhouette_score'] = silhouette
            
            # Supervised metrics (if meaningful)
            if len(np.unique(true_regimes)) > 1:
                ari = RegimeMetrics.adjusted_rand_index(true_regimes, predictions)
                nmi = RegimeMetrics.normalized_mutual_info(true_regimes, predictions)
                results['adjusted_rand_index'] = ari
                results['normalized_mutual_info'] = nmi
            
            # Regime stability
            stability = RegimeMetrics.regime_stability(predictions)
            results['transition_rate'] = stability['transition_rate']
            results['avg_duration'] = stability['avg_duration']
            results['persistence'] = stability['persistence']
            
            # Confidence metrics
            confidence = RegimeMetrics.confidence_metrics(probabilities)
            results['avg_confidence'] = confidence['avg_confidence']
            results['avg_entropy'] = confidence['avg_entropy']
            
            # Regime Quality Index
            rqi = RegimeMetrics.regime_quality_index(
                features.values, predictions, probabilities, returns.values
            )
            results['regime_quality_index'] = rqi
            
            # Financial metrics (commented out - FinancialMetrics removed from library)
            # regime_returns = FinancialMetrics.regime_returns(returns, predictions)
            # 
            # # Extract regime-specific Sharpe ratios
            # for regime_name, stats in regime_returns.items():
            #     results[f'{regime_name}_sharpe'] = stats['sharpe_ratio']
            #     results[f'{regime_name}_mean_return'] = stats['mean_return']
            #     results[f'{regime_name}_win_rate'] = stats['win_rate']
            
        else:
            print(f"    ⚠️  Only found {n_predicted_regimes} regime(s) - limited metrics available")
            results.update({
                'silhouette_score': 0,
                'adjusted_rand_index': 0,
                'normalized_mutual_info': 0,
                'transition_rate': 0,
                'avg_duration': len(features),
                'persistence': 1.0,
                'avg_confidence': 1.0,
                'avg_entropy': 0,
                'regime_quality_index': 0
            })
        
        print(f"    ✅ Success - RQI: {results.get('regime_quality_index', 0):.1f}")
        return results
        
    except Exception as e:
        print(f"    ❌ Failed: {str(e)}")
        return {
            'model_name': model_name,
            'success': False,
            'error': str(e)
        }


def main():
    """Run simple performance test."""
    
    print("🧪 Simple Regime Detection Performance Test")
    print("=" * 60)
    
    # Create dataset
    print("\n📊 Creating test dataset...")
    features, returns, true_regimes = create_simple_dataset()
    
    print(f"   Dataset size: {len(features)} samples")
    print(f"   Features: {list(features.columns)}")
    print(f"   True regimes: {np.unique(true_regimes)}")
    print(f"   Regime distribution: {np.bincount(true_regimes)}")
    
    # Test models
    models_to_test = [
        ('HMM-Full', HMMRegimeDetector, {'n_regimes': 3, 'covariance_type': 'full', 'random_state': 42}),
        ('HMM-Diag', HMMRegimeDetector, {'n_regimes': 3, 'covariance_type': 'diag', 'random_state': 42}),
        ('GMM-Full', GMMRegimeDetector, {'n_regimes': 3, 'covariance_type': 'full', 'random_state': 42}),
        ('GMM-Tied', GMMRegimeDetector, {'n_regimes': 3, 'covariance_type': 'tied', 'random_state': 42}),
        ('GMM-Auto', GMMRegimeDetector, {'n_regimes': 3, 'random_state': 42})
    ]
    
    print(f"\n🔬 Testing {len(models_to_test)} models...")
    print("-" * 40)
    
    results = []
    
    for model_name, model_class, model_params in models_to_test:
        result = test_single_model(
            model_class, model_params, features, returns, true_regimes, model_name
        )
        results.append(result)
    
    # Analyze results
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    print(f"Successful tests: {len(successful_results)}")
    print(f"Failed tests: {len(failed_results)}")
    
    if failed_results:
        print(f"\nFailed models:")
        for result in failed_results:
            print(f"  ❌ {result['model_name']}: {result.get('error', 'Unknown error')}")
    
    if successful_results:
        print(f"\n🏆 MODEL PERFORMANCE RANKING")
        print("-" * 40)
        
        # Sort by Regime Quality Index
        successful_results.sort(key=lambda x: x.get('regime_quality_index', 0), reverse=True)
        
        print(f"{'Rank':<4} {'Model':<12} {'RQI':<6} {'ARI':<6} {'Sil':<6} {'Time':<6}")
        print("-" * 45)
        
        for i, result in enumerate(successful_results, 1):
            rqi = result.get('regime_quality_index', 0)
            ari = result.get('adjusted_rand_index', 0)
            sil = result.get('silhouette_score', 0)
            time_val = result.get('fit_time', 0)
            
            print(f"{i:<4} {result['model_name']:<12} {rqi:<6.1f} {ari:<6.3f} {sil:<6.3f} {time_val:<6.2f}")
        
        # Best model analysis
        best_result = successful_results[0]
        print(f"\n🥇 BEST MODEL: {best_result['model_name']}")
        print("-" * 30)
        print(f"   Regime Quality Index: {best_result.get('regime_quality_index', 0):.1f}")
        print(f"   Adjusted Rand Index: {best_result.get('adjusted_rand_index', 0):.4f}")
        print(f"   Silhouette Score: {best_result.get('silhouette_score', 0):.4f}")
        print(f"   Fit Time: {best_result.get('fit_time', 0):.2f}s")
        print(f"   Regime Changes: {best_result.get('n_regime_changes', 0)}")
        print(f"   Persistence: {best_result.get('persistence', 0):.3f}")
        
        # Regime-specific performance
        regime_keys = [k for k in best_result.keys() if '_mean_return' in k]
        if regime_keys:
            print(f"\n   Regime Performance:")
            for key in regime_keys:
                regime_name = key.replace('_mean_return', '')
                mean_ret = best_result.get(key, 0)
                sharpe = best_result.get(f'{regime_name}_sharpe', 0)
                win_rate = best_result.get(f'{regime_name}_win_rate', 0)
                print(f"     {regime_name}: Return={mean_ret:.4f}, Sharpe={sharpe:.3f}, WinRate={win_rate:.1%}")
        
        # Summary statistics
        print(f"\n📈 AGGREGATE STATISTICS")
        print("-" * 30)
        
        avg_rqi = np.mean([r.get('regime_quality_index', 0) for r in successful_results])
        avg_ari = np.mean([r.get('adjusted_rand_index', 0) for r in successful_results])
        avg_time = np.mean([r.get('fit_time', 0) for r in successful_results])
        
        print(f"   Average RQI: {avg_rqi:.1f}")
        print(f"   Average ARI: {avg_ari:.4f}")
        print(f"   Average Fit Time: {avg_time:.2f}s")
        
        # Count regime detection success
        multi_regime_models = len([r for r in successful_results if r.get('n_predicted_regimes', 1) > 1])
        print(f"   Models detecting multiple regimes: {multi_regime_models}/{len(successful_results)}")
        
    print(f"\n✅ Performance test completed!")
    
    # Save results
    if successful_results:
        df = pd.DataFrame(successful_results)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_performance_test_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to: {filename}")


if __name__ == "__main__":
    main()
