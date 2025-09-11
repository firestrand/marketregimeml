"""
Demonstration of Advanced HMM Features for Market Regime Detection

This example showcases the new HMM enhancements:
1. BIC/AIC-based automatic selection of optimal number of regimes
2. Interpretable regime labeling (Bull/Bear/Sideways)
3. Comprehensive regime statistics and analysis
4. Model selection criteria comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.data.loaders import KrakenDataLoader


def load_market_features(lookback_days: int = 900) -> pd.DataFrame:
    """Load BTC/USD from Kraken and construct features used in this demo."""
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    ohlcv = loader.fetch_ohlcv("BTC/USD", "1d", start, end)

    df = pd.DataFrame(index=ohlcv.index)
    df['returns'] = ohlcv['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volume'] = ohlcv['volume'].pct_change()
    df['spread'] = (ohlcv['high'] / ohlcv['low'] - 1)

    # Rolling features
    df['sma_5'] = df['returns'].rolling(5, min_periods=1).mean()
    df['std_20'] = df['returns'].rolling(20, min_periods=1).std()
    return df.dropna()


def demonstrate_bic_aic_optimization():
    """Demonstrate automatic selection of optimal number of regimes."""
    print("=" * 80)
    print("1. BIC/AIC-Based Optimization for Optimal Number of Regimes")
    print("=" * 80)
    
    # Load real data features
    features = load_market_features(lookback_days=800)
    
    # Use only core features for optimization
    core_features = features[['returns', 'volatility']]
    
    # Initialize detector
    detector = HMMRegimeDetector(random_state=42)
    
    # Find optimal number of regimes using BIC
    print("\nOptimizing number of regimes using BIC criterion...")
    bic_results = detector.optimize_n_regimes(
        core_features,
        min_regimes=2,
        max_regimes=5,
        criterion='bic',
        n_init=5
    )
    
    print(f"\nOptimization Results (BIC):")
    print(f"  Optimal number of regimes: {bic_results['optimal_n_regimes']}")
    print(f"  Optimal BIC score: {bic_results['optimal_score']:.2f}")
    
    # Display all tested configurations
    print("\n  Tested configurations:")
    for n_reg, bic, aic in zip(bic_results['n_regimes'], 
                                bic_results['bic'], 
                                bic_results['aic']):
        print(f"    {n_reg} regimes: BIC={bic:.2f}, AIC={aic:.2f}")
    
    # Compare with AIC
    print("\nOptimizing using AIC criterion for comparison...")
    detector_aic = HMMRegimeDetector(random_state=42)
    aic_results = detector_aic.optimize_n_regimes(
        core_features,
        min_regimes=2,
        max_regimes=5,
        criterion='aic',
        n_init=5
    )
    
    print(f"\nOptimization Results (AIC):")
    print(f"  Optimal number of regimes: {aic_results['optimal_n_regimes']}")
    print(f"  Optimal AIC score: {aic_results['optimal_score']:.2f}")
    
    return detector, core_features


def demonstrate_regime_labeling(detector, features):
    """Demonstrate interpretable regime labeling."""
    print("\n" + "=" * 80)
    print("2. Interpretable Regime Labeling")
    print("=" * 80)
    
    # Method 1: Volatility-Return based labeling
    print("\nMethod 1: Volatility-Return Based Labeling")
    labels_vol_ret = detector.label_regimes(features, method='volatility_return')
    print("  Regime Labels:")
    for regime_id, label in labels_vol_ret.items():
        print(f"    Regime {regime_id}: {label}")
    
    # Method 2: Return Quantile based labeling
    print("\nMethod 2: Return Quantile Based Labeling")
    labels_quantile = detector.label_regimes(features, method='return_quantile')
    print("  Regime Labels:")
    for regime_id, label in labels_quantile.items():
        print(f"    Regime {regime_id}: {label}")
    
    # Method 3: Custom/Default labeling
    print("\nMethod 3: Custom/Default Labeling")
    labels_custom = detector.label_regimes(features, method='custom')
    print("  Regime Labels:")
    for regime_id, label in labels_custom.items():
        print(f"    Regime {regime_id}: {label}")
    
    # Use volatility-return for further analysis
    detector.label_regimes(features, method='volatility_return')
    
    return detector


def demonstrate_regime_statistics(detector, features):
    """Demonstrate comprehensive regime statistics."""
    print("\n" + "=" * 80)
    print("3. Comprehensive Regime Statistics")
    print("=" * 80)
    
    # Get regime predictions
    regimes = detector.predict(features)
    
    # Get detailed statistics
    stats = detector.get_regime_statistics(features, regimes)
    
    print("\nRegime Statistics Summary:")
    print("-" * 60)
    
    # Display key statistics for each regime
    for _, row in stats.iterrows():
        print(f"\n{row['regime_name']} (ID: {row['regime_id']}):")
        print(f"  Proportion: {row['proportion']:.2%}")
        print(f"  Count: {row['count']} observations")
        print(f"  Avg Duration: {row.get('avg_duration', 'N/A'):.1f} periods")
        print(f"  Returns: μ={row['returns_mean']:.4f}, σ={row['returns_std']:.4f}")
        print(f"  Volatility: μ={row['volatility_mean']:.4f}, σ={row['volatility_std']:.4f}")
        print(f"  Skewness: {row.get('returns_skew', 'N/A'):.3f}")
        print(f"  Kurtosis: {row.get('returns_kurtosis', 'N/A'):.3f}")
    
    # Transition Matrix Analysis
    print("\n" + "-" * 60)
    print("Transition Matrix:")
    trans_mat = detector.model.transmat_
    print("\n       ", end="")
    for i in range(len(trans_mat)):
        regime_name = detector.regime_names.get(i, f"R{i}")
        print(f"{regime_name:>15}", end="")
    print()
    
    for i in range(len(trans_mat)):
        regime_name = detector.regime_names.get(i, f"R{i}")
        print(f"{regime_name:>7}", end="")
        for j in range(len(trans_mat)):
            print(f"{trans_mat[i, j]:>15.3f}", end="")
        print()
    
    # Stability Analysis
    print("\n" + "-" * 60)
    print("Regime Stability Analysis:")
    stability_score = np.mean(np.diag(trans_mat))
    print(f"  Overall Stability Score: {stability_score:.3f}")
    print(f"  (Higher values indicate more persistent regimes)")
    
    for i in range(len(trans_mat)):
        regime_name = detector.regime_names.get(i, f"R{i}")
        persistence = trans_mat[i, i]
        expected_duration = 1 / (1 - persistence) if persistence < 1 else np.inf
        print(f"  {regime_name}: Persistence={persistence:.3f}, Expected Duration={expected_duration:.1f}")
    
    return stats, regimes


def demonstrate_model_selection():
    """Demonstrate different model selection criteria."""
    print("\n" + "=" * 80)
    print("4. Model Selection Criteria Comparison")
    print("=" * 80)
    
    # Load real data features
    features = load_market_features(lookback_days=500)
    core_features = features[['returns', 'volatility']]
    
    # Test different selection criteria
    criteria = ['score', 'bic', 'aic', 'stability']
    results = {}
    
    for criterion in criteria:
        print(f"\nFitting with selection criterion: {criterion}")
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)
        detector.fit(core_features, n_init=5, select_best=criterion)
        
        # Calculate metrics
        regimes = detector.predict(core_features)
        n_transitions = np.sum(np.diff(regimes) != 0)
        stability = np.mean(np.diag(detector.model.transmat_))
        
        results[criterion] = {
            'log_likelihood': detector.diagnostics['log_likelihood'],
            'aic': detector.diagnostics['aic'],
            'bic': detector.diagnostics['bic'],
            'n_transitions': n_transitions,
            'stability': stability
        }
        
        print(f"  Log-likelihood: {results[criterion]['log_likelihood']:.2f}")
        print(f"  AIC: {results[criterion]['aic']:.2f}")
        print(f"  BIC: {results[criterion]['bic']:.2f}")
        print(f"  Transitions: {results[criterion]['n_transitions']}")
        print(f"  Stability: {results[criterion]['stability']:.3f}")
    
    # Compare results
    print("\n" + "-" * 60)
    print("Comparison Summary:")
    print("  Best log-likelihood:", max(results.items(), key=lambda x: x[1]['log_likelihood'])[0])
    print("  Best AIC:", min(results.items(), key=lambda x: x[1]['aic'])[0])
    print("  Best BIC:", min(results.items(), key=lambda x: x[1]['bic'])[0])
    print("  Most stable:", max(results.items(), key=lambda x: x[1]['stability'])[0])


def visualize_regimes(detector, features, regimes):
    """Visualize detected regimes."""
    print("\n" + "=" * 80)
    print("5. Regime Visualization")
    print("=" * 80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot 1: Returns with regime coloring
    ax1 = axes[0]
    for regime_id in np.unique(regimes):
        mask = regimes == regime_id
        regime_name = detector.regime_names.get(regime_id, f"Regime {regime_id}")
        ax1.scatter(np.where(mask)[0], features['returns'][mask], 
                   s=10, label=regime_name, alpha=0.6)
    ax1.set_ylabel('Returns')
    ax1.set_title('Market Returns by Regime')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility
    ax2 = axes[1]
    for regime_id in np.unique(regimes):
        mask = regimes == regime_id
        ax2.scatter(np.where(mask)[0], features['volatility'][mask], 
                   s=10, alpha=0.6)
    ax2.set_ylabel('Volatility')
    ax2.set_title('Market Volatility by Regime')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime sequence
    ax3 = axes[2]
    ax3.plot(regimes, drawstyle='steps-post', linewidth=2)
    ax3.set_ylabel('Regime')
    ax3.set_xlabel('Time')
    ax3.set_title('Regime Sequence')
    ax3.set_yticks(range(detector.n_regimes))
    ax3.set_yticklabels([detector.regime_names.get(i, f"R{i}") 
                         for i in range(detector.n_regimes)])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hmm_regimes_visualization.png', dpi=100, bbox_inches='tight')
    print("\nVisualization saved as 'hmm_regimes_visualization.png'")
    
    # Print regime probabilities summary
    probabilities = detector.predict_proba(features)
    print("\nRegime Probability Statistics:")
    for i in range(detector.n_regimes):
        regime_name = detector.regime_names.get(i, f"Regime {i}")
        avg_prob = probabilities[:, i].mean()
        max_prob = probabilities[:, i].max()
        print(f"  {regime_name}: Avg={avg_prob:.3f}, Max={max_prob:.3f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("ADVANCED HMM FEATURES DEMONSTRATION")
    print("Market Regime Detection with Enhanced Capabilities")
    print("=" * 80)
    
    # 1. Demonstrate BIC/AIC optimization
    detector, features = demonstrate_bic_aic_optimization()
    
    # 2. Demonstrate regime labeling
    detector = demonstrate_regime_labeling(detector, features)
    
    # 3. Demonstrate regime statistics
    stats, regimes = demonstrate_regime_statistics(detector, features)
    
    # 4. Demonstrate model selection criteria
    demonstrate_model_selection()
    
    # 5. Visualize results
    try:
        visualize_regimes(detector, features, regimes)
    except Exception as e:
        print(f"\nVisualization skipped: {e}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. BIC/AIC optimization automatically finds the optimal number of regimes")
    print("2. Interpretable labeling makes regime analysis more intuitive")
    print("3. Comprehensive statistics provide deep insights into regime behavior")
    print("4. Multiple selection criteria allow for flexible model selection")
    print("5. The enhanced HMM detector is production-ready for market analysis")
    print("\nFor more information, see the documentation and test files.")


if __name__ == "__main__":
    main()
