"""
Clean example of market regime detection using MarketRegimeML.

This example demonstrates the focused API for regime detection only.
No trading, backtesting, or portfolio management - just pure regime identification.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Core regime detection imports
from marketregimeml.models import (
    HMMRegimeDetector, 
    GMMRegimeDetector,
    EnsembleRegimeDetector
)
from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.evaluation.evaluator import ModelEvaluator
from marketregimeml.data.loaders import KrakenDataLoader


def load_market_data(symbol: str = "BTC/USD", timeframe: str = "1d", lookback_days: int = 800) -> pd.DataFrame:
    """Load OHLCV data from Kraken."""
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    return loader.fetch_ohlcv(symbol, timeframe, start, end)


def prepare_features(data):
    """Prepare features for regime detection."""
    features = pd.DataFrame({
        # Return-based features
        'returns': data['close'].pct_change(),
        'log_returns': np.log(data['close'] / data['close'].shift(1)),
        
        # Volatility features  
        'realized_vol': data['close'].pct_change().rolling(20).std(),
        'parkinson_vol': np.sqrt(
            np.log(data['high'] / data['low'])**2 / (4 * np.log(2))
        ),
        
        # Price momentum
        'momentum_5d': data['close'].pct_change(5),
        'momentum_20d': data['close'].pct_change(20),
        
        # Volume features
        'volume_change': data['volume'].pct_change(),
        'volume_ma_ratio': data['volume'] / data['volume'].rolling(20).mean(),
        
        # Range features
        'daily_range': (data['high'] - data['low']) / data['close'],
        'body_range': np.abs(data['close'] - data['open']) / data['close']
    }).dropna()
    
    return features


def detect_regimes_example():
    """Example 1: Basic regime detection."""
    print("🔍 Example 1: Basic Regime Detection")
    print("=" * 50)
    
    # Load market data
    data = load_market_data("BTC/USD", "1d", 900)
    features = prepare_features(data)
    
    # Detect regimes using HMM
    detector = HMMRegimeDetector(n_regimes=3, random_state=42)
    detector.fit(features)
    
    # Predict regimes
    predicted_regimes = detector.predict(features)
    probabilities = detector.predict_proba(features)
    
    # Evaluate regime quality
    metrics = RegimeMetrics()
    
    # Clustering quality
    silhouette = metrics.silhouette_coefficient(features.values, predicted_regimes)
    print(f"Silhouette Score: {silhouette:.3f}")
    
    # Regime stability
    stability = metrics.regime_stability(predicted_regimes)
    print(f"Average Regime Duration: {stability['avg_duration']:.1f} days")
    print(f"Transition Rate: {stability['transition_rate']:.3f}")
    print(f"Persistence: {stability['persistence']:.3f}")
    
    # Overall quality index
    quality = metrics.regime_quality_index(
        features.values, predicted_regimes, probabilities
    )
    print(f"Regime Quality Index: {quality:.1f}/100")
    
    return detector, predicted_regimes, features


def compare_models_example():
    """Example 2: Compare multiple regime detection models."""
    print("\n🏆 Example 2: Model Comparison")
    print("=" * 50)
    
    # Load data
    data = load_market_data("BTC/USD", "1d", 600)
    features = prepare_features(data)
    
    # Create multiple models
    models = [
        HMMRegimeDetector(n_regimes=3, random_state=42),
        GMMRegimeDetector(n_regimes=3, random_state=42)
    ]
    
    # Fit all models
    for model in models:
        model.fit(features)
    
    # Compare using ModelEvaluator
    evaluator = ModelEvaluator(models[0])
    comparison = evaluator.compare_models(models, features)
    
    print("Model Comparison Results:")
    print(comparison[['model_type', 'silhouette_score', 'regime_quality_index']].round(3))
    
    # Identify best model
    best_idx = comparison['regime_quality_index'].idxmax()
    best_model = comparison.loc[best_idx, 'model_type']
    best_score = comparison.loc[best_idx, 'regime_quality_index']
    
    print(f"\nBest Model: {best_model} (Quality: {best_score:.1f}/100)")
    
    return comparison


def ensemble_example():
    """Example 3: Ensemble regime detection."""
    print("\n🎯 Example 3: Ensemble Regime Detection")
    print("=" * 50)
    
    # Load data
    data = load_market_data("BTC/USD", "1d", 800)
    features = prepare_features(data)
    
    # Create ensemble with multiple base models
    ensemble = EnsembleRegimeDetector(
        models=[
            HMMRegimeDetector(n_regimes=3, random_state=42),
            GMMRegimeDetector(n_regimes=3, random_state=43)
        ],
        strategy='voting',  # Simple majority voting
        random_state=42
    )
    
    # Fit ensemble
    ensemble.fit(features)
    
    # Predict regimes
    regimes = ensemble.predict(features)
    probabilities = ensemble.predict_proba(features)
    
    # Evaluate ensemble performance
    metrics = RegimeMetrics()
    quality = metrics.regime_quality_index(
        features.values, regimes, probabilities
    )
    
    # Analyze regime characteristics
    distribution = metrics.regime_distribution(regimes)
    temporal = metrics.temporal_consistency_metrics(regimes)
    
    print(f"Ensemble Quality Index: {quality:.1f}/100")
    print(f"Regime Distribution Entropy: {distribution['entropy']:.3f}")
    print(f"Local Temporal Consistency: {temporal['avg_local_consistency']:.3f}")
    print(f"Average Run Length: {temporal['avg_run_length']:.1f}")
    
    return ensemble, regimes


def regime_analysis_example():
    """Example 4: Detailed regime analysis."""
    print("\n📊 Example 4: Detailed Regime Analysis")
    print("=" * 50)
    
    # Get regimes from previous example
    detector, regimes, features = detect_regimes_example()
    
    # Comprehensive evaluation
    evaluator = ModelEvaluator(detector)
    results = evaluator.evaluate_model(features)
    
    # Generate detailed report
    report = evaluator.generate_report(features)
    
    # Print key sections of the report
    print("\nDetailed Regime Analysis:")
    print("-" * 30)
    
    # Regime distribution
    regime_dist = results['regime_analysis']['distribution']
    print("Regime Distribution:")
    for i, (count, prop) in enumerate(zip(regime_dist['regime_counts'], regime_dist['regime_proportions'])):
        print(f"  Regime {i}: {count} periods ({prop:.1%})")
    
    # Feature statistics by regime
    feature_stats = results['regime_analysis']['feature_statistics']
    print("\nFeature Characteristics by Regime:")
    for regime_id, stats in feature_stats.items():
        print(f"  {regime_id}: {stats['n_samples']} samples ({stats['proportion']:.1%})")
        print(f"    Avg returns: {stats['feature_means'].get('returns', 0):.4f}")
        print(f"    Avg volatility: {stats['feature_means'].get('realized_vol', 0):.4f}")
    
    return results


def main():
    """Run all regime detection examples."""
    print("🎯 MarketRegimeML - Focused Regime Detection Examples")
    print("=" * 60)
    print("Pure regime detection - no trading, no backtesting")
    print()
    
    # Run examples
    detect_regimes_example()
    compare_models_example()
    ensemble_example()
    regime_analysis_example()
    
    print("\n" + "=" * 60)
    print("✅ Examples complete! MarketRegimeML focuses purely on regime detection.")
    print("🎯 Use these regimes as inputs to your own trading/analysis systems.")


if __name__ == "__main__":
    main()
