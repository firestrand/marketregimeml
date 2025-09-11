#!/usr/bin/env python
"""Quick demonstration of regime metrics on real BTC/USD data (Kraken)."""

import pandas as pd
from datetime import datetime, timedelta

from marketregimeml.models.gmm import GMMRegimeDetector
from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.data.loaders import KrakenDataLoader


def quick_demo():
    """Quick demonstration of performance metrics."""
    print("=" * 60)
    print("REGIME DETECTION PERFORMANCE METRICS - QUICK DEMO")
    print("=" * 60)
    
    # Load BTC/USD OHLCV
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=600)
    ohlcv = loader.fetch_ohlcv("BTC/USD", "1d", start, end)
    
    # Build lightweight features
    features = pd.DataFrame(index=ohlcv.index)
    features['returns'] = ohlcv['close'].pct_change()
    features['volatility'] = (ohlcv['high'] / ohlcv['low'] - 1)
    features['momentum'] = ohlcv['close'].pct_change(10)
    features = features.dropna()
    print(f"Loaded {len(features)} feature rows")
    
    # Fit GMM model
    print("\nFitting GMM model...")
    model = GMMRegimeDetector(n_regimes=3, random_state=42)
    model.fit(features)
    
    # Get predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    
    print("Model fitted successfully!")
    
    # Demonstrate key metrics
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS RESULTS")
    print("=" * 60)
    
    # 1. Clustering Quality Metrics
    print("\n1. CLUSTERING QUALITY METRICS:")
    print("-" * 30)
    
    silhouette = RegimeMetrics.silhouette_coefficient(features.values, predictions)
    print(f"Silhouette Score: {silhouette:.4f}")
    print("  → Measures how well-separated the regimes are")
    print("  → Range: [-1, 1], higher is better")
    
    db_index = RegimeMetrics.davies_bouldin_index(features.values, predictions)
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print("  → Measures average similarity between clusters")
    print("  → Lower values indicate better clustering")
    
    ch_index = RegimeMetrics.calinski_harabasz_index(features.values, predictions)
    print(f"Calinski-Harabasz Index: {ch_index:.4f}")
    print("  → Ratio of between-cluster to within-cluster variance")
    print("  → Higher values indicate better clustering")
    
    # 2. Regime Characteristics
    print("\n2. REGIME CHARACTERISTICS:")
    print("-" * 30)
    
    stability = RegimeMetrics.regime_stability(predictions)
    print(f"Transition Rate: {stability['transition_rate']:.4f}")
    print(f"Average Duration: {stability['avg_duration']:.2f} days")
    print(f"Persistence: {stability['persistence']:.4f}")
    print("  → Lower transition rate = more stable regimes")
    
    distribution = RegimeMetrics.regime_distribution(predictions)
    print(f"Distribution Entropy: {distribution['normalized_entropy']:.4f}")
    print(f"Gini Coefficient: {distribution['gini_coefficient']:.4f}")
    print("  → Higher entropy = more balanced regime distribution")
    
    # 3. Confidence Metrics
    print("\n3. CONFIDENCE METRICS:")
    print("-" * 30)
    
    confidence = RegimeMetrics.confidence_metrics(probabilities)
    print(f"Average Confidence: {confidence['avg_confidence']:.4f}")
    print(f"Average Margin: {confidence['avg_margin']:.4f}")
    print(f"Normalized Entropy: {confidence['normalized_entropy']:.4f}")
    print("  → Higher confidence = more certain predictions")
    
    # Summary
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ MODEL PERFORMANCE:")
    print(f"   • Clustering Quality (Silhouette): {silhouette:.4f}")
    print(f"   • Regime Stability: {stability['persistence']:.4f}")
    print(f"   • Prediction Confidence: {confidence['avg_confidence']:.4f}")
    
    return {
        'silhouette': silhouette,
        'stability': stability,
        'confidence': confidence,
    }


if __name__ == "__main__":
    try:
        results = quick_demo()
        print(f"\n✅ Quick demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
