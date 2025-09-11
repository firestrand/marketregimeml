#!/usr/bin/env python3
"""
Advanced Ensemble Example for MarketRegimeML

This script demonstrates advanced ensemble techniques including:
- Multiple ensemble strategies
- Weight optimization
- Cross-validation
- Model diversity analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

from marketregimeml.models import (
    HMMRegimeDetector,
    GMMRegimeDetector, 
    GARCHRegimeDetector,
    RandomForestRegimeClassifier,
    LSTMRegimeDetector,
    EnsembleRegimeDetector
)
from marketregimeml.evaluation import RegimeEvaluator, RegimeBacktester
from marketregimeml.data.loaders import KrakenDataLoader


def load_market_data(symbol: str = "BTC/USD", timeframe: str = "1d", lookback_days: int = 1200):
    """Load real BTC/USD data with Kraken and prepare features."""
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    ohlcv = loader.fetch_ohlcv(symbol, timeframe, start, end)

    # Build features
    feats = pd.DataFrame(index=ohlcv.index)
    feats['returns'] = ohlcv['close'].pct_change()
    feats['volatility'] = (ohlcv['high'] / ohlcv['low'] - 1)
    feats['volume'] = ohlcv['volume'].pct_change()
    feats['momentum'] = ohlcv['close'].pct_change(5)
    feats['rsi'] = calculate_rsi(feats['returns'])
    feats['ma_ratio'] = calculate_ma_ratio(feats['returns'])
    feats = feats.dropna()
    return feats


def calculate_rsi(returns, period=14):
    """Calculate RSI indicator."""
    delta = pd.Series(returns)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ma_ratio(returns, short=10, long=30):
    """Calculate moving average ratio."""
    prices = (1 + returns).cumprod()
    ma_short = prices.rolling(window=short).mean()
    ma_long = prices.rolling(window=long).mean()
    return ma_short / ma_long - 1


def test_ensemble_strategies(features, true_regimes):
    """Test different ensemble strategies."""
    
    print("\n" + "=" * 60)
    print("Testing Ensemble Strategies")
    print("=" * 60)
    
    # Create diverse base models
    base_models = [
        HMMRegimeDetector(n_regimes=3, covariance_type='diag'),
        HMMRegimeDetector(n_regimes=3, covariance_type='full'),
        GMMRegimeDetector(n_regimes=3, covariance_type='full'),
        GARCHRegimeDetector(n_regimes=3, threshold_method='quantile'),
        RandomForestRegimeClassifier(n_regimes=3, n_estimators=50)
    ]
    
    strategies = ['voting', 'weighted_voting', 'bayesian', 'consensus', 'stacking']
    results = {}
    
    evaluator = RegimeEvaluator()
    
    for strategy in strategies:
        print(f"\nTesting {strategy.upper()} strategy...")
        
        # Create ensemble
        ensemble = EnsembleRegimeDetector(
            models=base_models.copy(),  # Use copy to avoid reusing fitted models
            strategy=strategy,
            n_regimes=3,
            consensus_threshold=0.6,
            random_state=42
        )
        
        # Fit and predict
        ensemble.fit(features)
        predictions = ensemble.predict(features)
        
        # Evaluate
        metrics = evaluator.evaluate_model(
            ensemble, 
            features,
            true_regimes
        )
        
        # Get ensemble-specific metrics
        avg_agreement = ensemble.diagnostics.get('avg_agreement', 0)
        
        results[strategy] = {
            'ari': metrics.get('adjusted_rand_index', float('nan')),
            'silhouette': metrics['silhouette_score'],
            'stability': metrics['regime_stability'],
            'agreement': avg_agreement,
            'ensemble': ensemble
        }
        
        if 'adjusted_rand_index' in metrics:
            print(f"  ARI: {metrics['adjusted_rand_index']:.3f}")
        print(f"  Silhouette: {metrics['silhouette_score']:.3f}")
        print(f"  Stability: {metrics['regime_stability']:.3f}")
        print(f"  Model Agreement: {avg_agreement:.3f}")
    
    return results


def optimize_ensemble_weights(features, true_regimes):
    """Demonstrate weight optimization for ensemble."""
    
    print("\n" + "=" * 60)
    print("Optimizing Ensemble Weights")
    print("=" * 60)
    
    # Create ensemble with weighted voting
    models = [
        HMMRegimeDetector(n_regimes=3, init_method='kmeans'),
        GMMRegimeDetector(n_regimes=3),
        GARCHRegimeDetector(n_regimes=3),
        RandomForestRegimeClassifier(n_regimes=3)
    ]
    # For very small datasets, remove GARCH (needs >= 30 obs in validation window)
    if len(features) < 180:
        models = [m for m in models if not isinstance(m, GARCHRegimeDetector)]
    
    ensemble = EnsembleRegimeDetector(
        models=models,
        strategy='weighted_voting',
        n_regimes=3,
        random_state=42
    )
    
    # Fit with equal weights
    print("\nInitial performance (equal weights):")
    ensemble.fit(features)
    initial_pred = ensemble.predict(features)
    
    evaluator = RegimeEvaluator()
    initial_metrics = evaluator.evaluate_model(ensemble, features, true_regimes)
    if 'adjusted_rand_index' in initial_metrics:
        print(f"  ARI: {initial_metrics['adjusted_rand_index']:.3f}")
    print(f"  Initial weights: {ensemble.weights}")
    
    # Optimize weights
    print("\nOptimizing weights...")
    optimal_weights = ensemble.optimize_weights(features)
    
    print(f"  Optimized weights:")
    for i, item in enumerate(ensemble.models):
        # items may be (name, model)
        model_cls = item[1].__class__.__name__ if isinstance(item, tuple) else item.__class__.__name__
        print(f"    {model_cls}: {optimal_weights[i]:.3f}")
    
    # Apply optimized weights
    ensemble.weights = optimal_weights
    optimized_pred = ensemble.predict(features)
    
    optimized_metrics = evaluator.evaluate_model(ensemble, features, true_regimes)
    print(f"\nOptimized performance:")
    if 'adjusted_rand_index' in optimized_metrics:
        print(f"  ARI: {optimized_metrics['adjusted_rand_index']:.3f}")
    
    improvement = 0.0
    if 'adjusted_rand_index' in initial_metrics and 'adjusted_rand_index' in optimized_metrics and initial_metrics['adjusted_rand_index'] != 0:
        improvement = (
            (optimized_metrics['adjusted_rand_index'] - initial_metrics['adjusted_rand_index']) /
            abs(initial_metrics['adjusted_rand_index']) * 100
        )
        print(f"  Improvement: {improvement:+.1f}%")
    
    return ensemble, optimal_weights, improvement


def analyze_model_diversity(features):
    """Analyze diversity among ensemble models."""
    
    print("\n" + "=" * 60)
    print("Analyzing Model Diversity")
    print("=" * 60)
    
    # Create diverse models
    models = [
        HMMRegimeDetector(n_regimes=3, init_method='kmeans'),
        HMMRegimeDetector(n_regimes=3, init_method='random'),
        GMMRegimeDetector(n_regimes=3, covariance_type='full'),
        GMMRegimeDetector(n_regimes=3, covariance_type='diag'),
        GARCHRegimeDetector(n_regimes=3),
        RandomForestRegimeClassifier(n_regimes=3)
    ]
    
    # Fit all models and get predictions
    predictions = []
    model_names = []
    
    for model in models:
        model.fit(features)
        pred = model.predict(features)
        predictions.append(pred)
        suffix = getattr(model, 'init_method', getattr(model, 'covariance_type', 'default'))
        model_names.append(model.__class__.__name__ + "_" + str(suffix))
    
    # Calculate pairwise agreement
    n_models = len(models)
    agreement_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                # Calculate adjusted rand index
                from sklearn.metrics import adjusted_rand_score
                agreement_matrix[i, j] = adjusted_rand_score(
                    predictions[i], predictions[j]
                )
    
    # Print agreement matrix
    print("\nPairwise Model Agreement (ARI):")
    print(" " * 25 + "  ".join([f"{i:>3}" for i in range(n_models)]))
    for i, name in enumerate(model_names):
        print(f"{name[:22]:22} ", end="")
        for j in range(n_models):
            print(f"{agreement_matrix[i, j]:5.2f}", end="")
        print()
    
    # Calculate diversity metrics
    avg_agreement = (agreement_matrix.sum() - n_models) / (n_models * (n_models - 1))
    diversity = 1 - avg_agreement
    
    print(f"\nAverage Agreement: {avg_agreement:.3f}")
    print(f"Diversity Score: {diversity:.3f}")
    
    # Create ensemble with these models
    ensemble = EnsembleRegimeDetector(
        models=models,
        strategy='weighted_voting',
        n_regimes=3
    )
    ensemble.fit(features)
    
    # Get model contributions
    contributions = ensemble.get_model_contributions(features)
    print("\nModel Contributions to Ensemble:")
    print(contributions)
    
    return agreement_matrix, model_names, diversity


def cross_validate_ensemble(features):
    """Perform cross-validation on ensemble models."""
    
    print("\n" + "=" * 60)
    print("Cross-Validating Ensemble")
    print("=" * 60)
    
    # Create ensemble
    ensemble = EnsembleRegimeDetector(
        n_regimes=3,
        strategy='weighted_voting',
        random_state=42
    )
    
    # Fit ensemble
    ensemble.fit(features[:500])  # Use first half for training
    
    # Cross-validate
    print("\nPerforming 5-fold cross-validation...")
    cv_results = ensemble.cross_validate_models(features[:500], n_splits=5)
    
    print("\nCross-Validation Results (Log-Likelihood):")
    print("-" * 50)
    
    # Sort by mean score
    sorted_results = sorted(cv_results.items(), key=lambda x: x[1], reverse=True)
    for model_name, mean_score in sorted_results:
        print(f"{model_name:25} Mean: {mean_score:8.2f}")
    
    # Check if ensemble beats best individual
    ensemble_score = float('-inf')  # not available in this CV summary
    best_individual = max(v for k, v in cv_results.items())
    
    if ensemble_score > best_individual:
        improvement = (ensemble_score - best_individual) / abs(best_individual) * 100
        print(f"\n✓ Ensemble outperforms best individual by {improvement:.1f}%")
    else:
        print("\n✗ Ensemble does not outperform best individual model")
    
    return cv_results


def backtest_ensemble_strategy(features, ensemble):
    """Backtest trading strategy using ensemble predictions."""
    
    print("\n" + "=" * 60)
    print("Backtesting Ensemble Strategy")
    print("=" * 60)
    
    # Define multiple strategies
    strategies = {
        'Conservative': {0: 0.2, 1: 0.6, 2: 1.0},
        'Balanced': {0: 0.3, 1: 1.0, 2: 1.5},
        'Aggressive': {0: 0.0, 1: 1.0, 2: 2.0},
        'Adaptive': {0: 0.5, 1: 1.0, 2: 1.3}  # Less extreme
    }
    
    backtester = RegimeBacktester(
        model=ensemble,
        transaction_cost=0.001,
        initial_capital=100000,
        risk_free_rate=0.02
    )
    
    # Test each strategy
    results = {}
    for name, weights in strategies.items():
        result = backtester.backtest_strategy(
            features=features,
            returns=features['returns'],
            strategy_weights=weights,
            rebalance_frequency='W'
        )
        results[name] = result
        
        print(f"\n{name} Strategy:")
        print(f"  Return: {result['total_return']:.2%}")
        print(f"  Sharpe: {result['sharpe_ratio']:.2f}")
        print(f"  Max DD: {result['max_drawdown']:.2%}")
        print(f"  Win Rate: {result['win_rate']:.1%}")
    
    # Compare strategies
    comparison = backtester.compare_strategies(
        features=features,
        returns=features['returns'],
        strategies=strategies
    )
    
    print("\n" + "-" * 50)
    print("Strategy Comparison Summary:")
    print(comparison[['total_return', 'sharpe_ratio', 'max_drawdown']].round(3))
    
    # Find best strategy
    best_strategy = comparison['sharpe_ratio'].idxmax()
    print(f"\nBest Strategy (by Sharpe): {best_strategy}")
    
    return results, comparison, best_strategy


def visualize_ensemble_results(features, ensemble, true_regimes):
    """Create comprehensive visualization of ensemble results."""
    
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    
    predictions = ensemble.predict(features)
    probabilities = ensemble.predict_proba(features)
    
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Regime predictions comparison
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(features.index, true_regimes, 'b-', alpha=0.5, label='True Regimes')
    ax1.plot(features.index, predictions, 'r-', alpha=0.5, label='Ensemble Predictions')
    ax1.fill_between(features.index, true_regimes, alpha=0.3)
    ax1.fill_between(features.index, predictions, alpha=0.3)
    ax1.set_ylabel('Regime')
    ax1.set_title('True vs Predicted Regimes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime probabilities
    ax2 = plt.subplot(4, 1, 2)
    colors = ['red', 'yellow', 'green']
    labels = ['Bear', 'Neutral', 'Bull']
    for i in range(3):
        ax2.fill_between(features.index, 0, probabilities[:, i], 
                        color=colors[i], alpha=0.5, label=labels[i])
    ax2.set_ylabel('Probability')
    ax2.set_title('Regime Probabilities from Ensemble')
    ax2.legend()
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model agreement over time
    ax3 = plt.subplot(4, 1, 3)
    
    # Calculate rolling agreement
    window = 20
    agreement_scores = []
    
    for i in range(window, len(predictions)):
        window_pred = predictions[i-window:i]
        # Simple agreement metric: std of predictions
        unique, counts = np.unique(window_pred, return_counts=True)
        if len(unique) > 0:
            agreement = max(counts) / window
        else:
            agreement = 1.0
        agreement_scores.append(agreement)
    
    agreement_index = features.index[window:]
    ax3.plot(agreement_index, agreement_scores, 'g-', linewidth=1)
    ax3.fill_between(agreement_index, agreement_scores, alpha=0.3)
    ax3.set_ylabel('Agreement Score')
    ax3.set_title(f'Model Agreement (rolling {window}-day window)')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Returns by regime
    ax4 = plt.subplot(4, 1, 4)
    returns = features['returns'].values
    
    for regime in range(3):
        mask = predictions == regime
        regime_returns = returns[mask]
        x_pos = features.index[mask]
        ax4.scatter(x_pos, regime_returns, c=colors[regime], 
                   alpha=0.5, s=10, label=labels[regime])
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_ylabel('Returns')
    ax4.set_xlabel('Date')
    ax4.set_title('Returns by Predicted Regime')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_ensemble_results.png', dpi=100)
    print("  ✓ Saved visualization to 'advanced_ensemble_results.png'")
    
    return fig


def main():
    """Run the advanced ensemble demonstration."""
    
    print("=" * 60)
    print("Advanced Ensemble Techniques Demonstration")
    print("=" * 60)
    
    # Args
    parser = argparse.ArgumentParser(description="Advanced ensemble methods demo")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode (smaller dataset)")
    parser.add_argument("--tiny", action="store_true", help="Run in tiny mode (very small dataset)")
    args = parser.parse_args()

    # Load real data
    print("\nLoading market data (Kraken BTC/USD)...")
    try:
        lookback = 1400
        if args.tiny:
            lookback = 120
        elif args.quick:
            lookback = 600
        features = load_market_data("BTC/USD", "1d", lookback)
    except Exception:
        print("  ✗ Could not load data from Kraken.")
        raise
    print(f"  Loaded {len(features)} samples with {features.shape[1]} features")

    # Determine slice sizes for quick mode
    train_n = 500
    if args.tiny:
        train_n = 100
    elif args.quick:
        train_n = 300
    
    # Test ensemble strategies
    strategy_results = test_ensemble_strategies(features[:train_n], None)
    
    # Find best strategy
    best_strategy = max(strategy_results.items(), 
                       key=lambda x: x[1]['ari'])
    print(f"\nBest ensemble strategy: {best_strategy[0].upper()}")
    
    # Optimize weights
    optimized_ensemble, weights, improvement = optimize_ensemble_weights(
        features[:train_n], None
    )
    
    # Analyze diversity
    agreement_matrix, model_names, diversity = analyze_model_diversity(features[:train_n])
    
    # Cross-validate
    cv_results = cross_validate_ensemble(features[:train_n])
    
    # Backtest strategies
    # Backtest on remainder; if too small, choose last 40% of data
    backtest_start = train_n if train_n < len(features) else int(len(features) * 0.6)
    if len(features) - backtest_start < 10:
        print("\nSkipping backtest due to insufficient remainder length.")
        best_strategy_name = "N/A"
    else:
        backtest_results, strategy_comparison, best_strategy_name = backtest_ensemble_strategy(
            features[backtest_start:],
            optimized_ensemble
        )
    
    # Visualize results using training window; no true labels available
    try:
        visualize_ensemble_results(
            features[:train_n],
            best_strategy[1]['ensemble'],
            np.zeros(len(features[:train_n]))
        )
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Advanced Ensemble Analysis Complete!")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"1. Best ensemble strategy: {best_strategy[0]}")
    print(f"2. Weight optimization improved ARI by {improvement:.1f}%")
    print(f"3. Model diversity score: {diversity:.3f}")
    print(f"4. Best trading strategy: {best_strategy_name}")
    print("\nNext steps:")
    print("- Try with real market data")
    print("- Add more sophisticated base models (LSTM, Transformer)")
    print("- Implement online learning for adaptive ensembles")
    print("- Develop custom ensemble strategies")


if __name__ == "__main__":
    main()
