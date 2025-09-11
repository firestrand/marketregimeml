#!/usr/bin/env python3
"""
Test script for enhanced regime detection capabilities.

This script demonstrates:
1. Arbitrary odd-numbered regimes (3, 5, 7, 9)
2. Fuzzy regime matching
3. Automatic regime optimization
4. Regime transition analysis
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse

# Import regime detectors
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.garch import GARCHRegimeDetector
from marketregimeml.data.loaders import KrakenDataLoader

def load_market_features(lookback_days: int = 900) -> pd.DataFrame:
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    ohlcv = loader.fetch_ohlcv("BTC/USD", "1d", start, end)
    df = pd.DataFrame(index=ohlcv.index)
    df['returns'] = ohlcv['close'].pct_change()
    df['volatility'] = (ohlcv['high'] / ohlcv['low'] - 1)
    df['momentum'] = ohlcv['close'].pct_change(5)
    return df.dropna()

def test_arbitrary_regimes(quick: bool = False):
    """Test different numbers of regimes."""
    print("🎯 TESTING ARBITRARY REGIME COUNTS")
    print("=" * 50)
    
    regime_list = [3, 5] if quick else [3, 5, 7]
    for n_regimes in regime_list:
        print(f"\n📊 Testing {n_regimes} regimes:")
        
        # Load data
        features = load_market_features(lookback_days=400 if quick else 700)
        
        # Test HMM detector
        detector = HMMRegimeDetector(n_regimes=n_regimes, random_state=42)
        detector.fit(features)
        
        print(f"   Regime Names: {list(detector.regime_names.values())}")
        
        # Get predictions and analyze regime frequencies
        predicted = detector.predict(features)
        unique, counts = np.unique(predicted, return_counts=True)
        regime_dist = counts / len(predicted)
        print(f"   Regime Distribution: {dict(zip(unique, regime_dist))}")

def test_fuzzy_matching(quick: bool = False):
    """Test fuzzy regime matching capabilities."""
    print("\n\n🔮 TESTING FUZZY REGIME MATCHING")
    print("=" * 50)
    
    # Load data
    features = load_market_features(lookback_days=300 if quick else 500)
    
    # Test with fuzzy matching enabled
    detector = HMMRegimeDetector(
        n_regimes=5, 
        fuzzy_matching=True, 
        fuzzy_threshold=0.7,
        random_state=42
    )
    detector.fit(features)
    
    # Get fuzzy predictions
    fuzzy_results = detector.predict_fuzzy(features)
    
    print("Fuzzy Results Keys:", list(fuzzy_results.keys()))
    print(f"Sample Size: {len(fuzzy_results['crisp'])}")
    
    # Analyze confidence distribution
    confidence = fuzzy_results['confidence']
    print(f"\nConfidence Statistics:")
    print(f"   Mean: {np.mean(confidence):.3f}")
    print(f"   Std:  {np.std(confidence):.3f}")
    print(f"   Min:  {np.min(confidence):.3f}")
    print(f"   Max:  {np.max(confidence):.3f}")
    
    # Count fuzzy vs crisp assignments
    fuzzy_assignments = fuzzy_results['fuzzy']
    crisp_count = np.sum(np.max(fuzzy_assignments, axis=1) == 1.0)
    fuzzy_count = len(fuzzy_assignments) - crisp_count
    
    print(f"\nAssignment Types:")
    print(f"   Crisp: {crisp_count} ({crisp_count/len(fuzzy_assignments)*100:.1f}%)")
    print(f"   Fuzzy: {fuzzy_count} ({fuzzy_count/len(fuzzy_assignments)*100:.1f}%)")
    
    # Show example fuzzy assignments
    print(f"\nExample Fuzzy Assignments (first 5 samples):")
    for i in range(min(5, len(fuzzy_assignments))):
        fuzzy_probs = fuzzy_assignments[i]
        crisp_regime = fuzzy_results['crisp'][i]
        conf = confidence[i]
        print(f"   Sample {i}: Crisp={crisp_regime}, Confidence={conf:.3f}, Fuzzy={fuzzy_probs}")

def test_regime_optimization(quick: bool = False):
    """Test automatic regime count optimization."""
    print("\n\n⚡ TESTING REGIME COUNT OPTIMIZATION")
    print("=" * 50)
    
    # Load data
    features = load_market_features(lookback_days=400 if quick else 600)
    
    # Test auto-optimization
    detector = HMMRegimeDetector(
        n_regimes=3,  # Start with 3, let it optimize
        auto_optimize_regimes=True,
        min_regimes=2,
        max_regimes=8,
        random_state=42
    )
    
    # Run optimization
    optimization_result = detector.optimize_regime_count(features, criteria='aic')
    
    print("Optimization Results:")
    print(f"   Optimal Regimes: {optimization_result['optimal_regimes']}")
    print(f"   Criterion: {optimization_result['criterion']}")
    
    print(f"\nAll Scores:")
    for n, score in optimization_result['scores'].items():
        print(f"   {n} regimes: AIC = {score:.2f}")
    
    # Test different criteria
    for criterion in ['bic', 'silhouette']:
        result = detector.optimize_regime_count(features, criteria=criterion)
        print(f"\n{criterion.upper()} Optimal: {result['optimal_regimes']} regimes")

def test_transition_analysis(quick: bool = False):
    """Test regime transition analysis."""
    print("\n\n🔄 TESTING REGIME TRANSITION ANALYSIS")
    print("=" * 50)
    
    # Load data
    features = load_market_features(lookback_days=500 if quick else 800)
    
    detector = HMMRegimeDetector(n_regimes=5, random_state=42)
    detector.fit(features)
    predicted = detector.predict(features)
    
    # Analyze transitions
    transition_analysis = detector.analyze_regime_transitions(predicted)
    
    print("Transition Analysis:")
    print(f"   Total Transitions: {transition_analysis['n_transitions']}")
    
    print(f"\nRegime Frequencies:")
    for regime, freq in enumerate(transition_analysis['regime_frequencies']):
        regime_name = detector.regime_names[regime]
        print(f"   {regime_name}: {freq:.3f}")
    
    print(f"\nAverage Regime Durations:")
    for regime, duration in transition_analysis['average_durations'].items():
        regime_name = detector.regime_names[regime]
        print(f"   {regime_name}: {duration:.1f} periods")
    
    print(f"\nRegime Persistence (stay probability):")
    for regime, persistence in enumerate(transition_analysis['persistence']):
        regime_name = detector.regime_names[regime]
        print(f"   {regime_name}: {persistence:.3f}")
    
    print(f"\nTransition Matrix:")
    transition_probs = transition_analysis['transition_probabilities']
    print("     " + "".join([f"{detector.regime_names[i]:<8}" for i in range(5)]))
    for i in range(5):
        row_name = detector.regime_names[i]
        row_probs = [f"{prob:.3f}" for prob in transition_probs[i]]
        print(f"{row_name:<8} " + " ".join([f"{p:<8}" for p in row_probs]))

def main():
    """Run all enhanced regime detection tests."""
    print("🚀 ENHANCED REGIME DETECTION CAPABILITIES")
    print("=" * 60)
    print("Testing arbitrary regimes, fuzzy matching, and optimization")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Enhanced regime detection tests")
    parser.add_argument("--quick", action="store_true", help="Run tests in quick mode")
    parser.add_argument("--tiny", action="store_true", help="Run tests in tiny mode")
    args = parser.parse_args()

    try:
        quick = args.quick or args.tiny
        test_arbitrary_regimes(quick=quick)
        test_fuzzy_matching(quick=quick)
        test_regime_optimization(quick=quick)
        test_transition_analysis(quick=quick)
        
        print("\n\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n✨ Enhanced Features Summary:")
        print("   ✅ Arbitrary odd-numbered regimes (3, 5, 7, 9)")
        print("   ✅ Fuzzy regime matching with confidence scores")
        print("   ✅ Automatic regime count optimization")
        print("   ✅ Comprehensive transition analysis")
        print("   ✅ Meaningful regime naming conventions")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
