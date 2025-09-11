#!/usr/bin/env python3
"""
Quick Start Example for MarketRegimeML

This script demonstrates the basic usage of the MarketRegimeML library
for detecting market regimes in financial data using real BTC/USD data
from Kraken (public API, no key required).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import MarketRegimeML components
from marketregimeml.models import HMMRegimeDetector, EnsembleRegimeDetector
from marketregimeml.evaluation import RegimeEvaluator, RegimeBacktester
from marketregimeml.data.loaders import KrakenDataLoader


def load_market_data(symbol: str = "BTC/USD", timeframe: str = "1d", lookback_days: int = 600) -> pd.DataFrame:
    """Load real market data from Kraken (BTC/USD)."""
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    return loader.fetch_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)


def prepare_features(ohlcv_data):
    """Prepare features for regime detection."""
    features = pd.DataFrame(index=ohlcv_data.index)
    
    # Calculate returns
    features['returns'] = ohlcv_data['close'].pct_change()
    
    # Calculate volatility (high-low spread)
    features['volatility'] = (ohlcv_data['high'] / ohlcv_data['low'] - 1)
    
    # Calculate volume changes
    features['volume_change'] = ohlcv_data['volume'].pct_change()
    
    # Calculate momentum
    features['momentum'] = ohlcv_data['close'] / ohlcv_data['close'].shift(5) - 1
    
    # Remove NaN values
    features = features.dropna()
    
    return features


def main():
    """Run the quickstart demonstration."""
    
    print("=" * 60)
    print("MarketRegimeML Quick Start Example")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading market data (Kraken BTC/USD)...")
    try:
        ohlcv_data = load_market_data("BTC/USD", "1d", 800)
    except Exception as e:
        print(f"   ✗ Could not load real market data from Kraken: {e}")
        raise
    print(f"   Loaded {len(ohlcv_data)} records of market data")
    
    # Step 2: Prepare features
    print("\n2. Preparing features...")
    features = prepare_features(ohlcv_data)
    print(f"   Created {features.shape[1]} features")
    print(f"   Feature columns: {list(features.columns)}")
    
    # No ground truth labels for real data
    true_regimes_aligned = None
    
    # Step 3: Create and train HMM model
    print("\n3. Training Hidden Markov Model...")
    hmm_model = HMMRegimeDetector(
        n_regimes=3,
        covariance_type='diag',
        random_state=42
    )
    hmm_model.fit(features)
    print("   ✓ Model trained successfully")
    
    # Step 4: Predict regimes
    print("\n4. Detecting market regimes...")
    predicted_regimes = hmm_model.predict(features)
    regime_probabilities = hmm_model.predict_proba(features)
    
    # Calculate regime distribution
    unique, counts = np.unique(predicted_regimes, return_counts=True)
    print("   Regime distribution:")
    regime_names = ['Bear', 'Neutral', 'Bull']
    for regime, count in zip(unique, counts):
        pct = count / len(predicted_regimes) * 100
        print(f"     {regime_names[regime]}: {count} periods ({pct:.1f}%)")
    
    # Step 5: Analyze transitions
    print("\n5. Analyzing regime transitions...")
    transitions = hmm_model.analyze_regime_transitions(features)
    print("   Transition matrix:")
    print("   " + " " * 10 + "  ".join([f"{name:>8}" for name in regime_names]))
    for i, from_regime in enumerate(regime_names):
        row = transitions['transition_matrix'][i]
        print(f"   {from_regime:8} " + "  ".join([f"{p:8.3f}" for p in row]))
    
    # Step 6: Evaluate model
    print("\n6. Evaluating model performance...")
    evaluator = RegimeEvaluator()
    metrics = evaluator.evaluate_model(
        model=hmm_model,
        features=features,
        true_regimes=true_regimes_aligned
    )
    
    print(f"   Silhouette Score: {metrics['silhouette_score']:.3f}")
    if 'adjusted_rand_index' in metrics:
        print(f"   Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
    print(f"   Regime Stability: {metrics['regime_stability']:.3f}")
    
    # Step 7: Backtest a simple strategy
    print("\n7. Backtesting regime-based strategy...")
    
    # Define position weights for each regime
    strategy_weights = {
        0: 0.3,   # 30% exposure in bear market
        1: 1.0,   # 100% exposure in neutral market
        2: 1.5    # 150% exposure in bull market
    }
    
    backtester = RegimeBacktester(
        model=hmm_model,
        transaction_cost=0.001,
        initial_capital=100000
    )
    
    backtest_results = backtester.backtest_strategy(
        features=features,
        returns=features['returns'],
        strategy_weights=strategy_weights
    )
    
    print(f"   Total Return: {backtest_results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    print(f"   Win Rate: {backtest_results['win_rate']:.1%}")
    
    # Step 8: Try ensemble approach
    print("\n8. Creating ensemble model...")
    from marketregimeml.models import GMMRegimeDetector
    
    ensemble = EnsembleRegimeDetector(
        models=[
            HMMRegimeDetector(n_regimes=3, init_method='kmeans'),
            HMMRegimeDetector(n_regimes=3, init_method='random'),
            GMMRegimeDetector(n_regimes=3, covariance_type='full')
        ],
        strategy='voting',
        n_regimes=3,
        random_state=42
    )
    
    ensemble.fit(features)
    ensemble_regimes = ensemble.predict(features)
    
    print(f"   Model agreement: {ensemble.diagnostics['avg_agreement']:.1%}")
    
    # Step 9: Visualize results
    print("\n9. Creating visualizations...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Price with regime coloring
    ax1 = axes[0]
    colors = ['red', 'yellow', 'green']
    for regime in range(3):
        mask = predicted_regimes == regime
        indices = features.index[mask]
        prices = ohlcv_data.loc[indices, 'close']
        ax1.scatter(indices, prices, c=colors[regime], alpha=0.6, s=10, 
                   label=regime_names[regime])
    ax1.plot(features.index, ohlcv_data.loc[features.index, 'close'], 
             'k-', alpha=0.3, linewidth=0.5)
    ax1.set_ylabel('Price')
    ax1.set_title('Market Regimes Detected by HMM')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime probabilities
    ax2 = axes[1]
    for regime in range(3):
        ax2.plot(features.index, regime_probabilities[:, regime], 
                label=regime_names[regime], alpha=0.7)
    ax2.set_ylabel('Probability')
    ax2.set_title('Regime Probabilities Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Strategy performance
    ax3 = axes[2]
    portfolio_values = backtest_results['portfolio_values']
    ax3.plot(portfolio_values.index, portfolio_values.values, 'b-', label='Strategy')
    
    # Add buy-and-hold benchmark
    buy_hold_returns = features['returns'].fillna(0)
    buy_hold_values = 100000 * (1 + buy_hold_returns).cumprod()
    ax3.plot(buy_hold_values.index, buy_hold_values.values, 'gray', 
             alpha=0.5, label='Buy & Hold')
    
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.set_xlabel('Date')
    ax3.set_title('Strategy Performance vs Buy & Hold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regime_detection_results.png', dpi=100)
    print("   ✓ Saved visualization to 'regime_detection_results.png'")
    
    # Summary
    print("\n" + "=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Try with real market data using OANDA or Alpha Vantage loaders")
    print("2. Experiment with different models (GARCH, LSTM, etc.)")
    print("3. Optimize hyperparameters for your specific use case")
    print("4. Implement more sophisticated trading strategies")
    print("5. Use fuzzy matching for uncertain regime boundaries")
    print("\nFor more examples, see the documentation and other example scripts.")


if __name__ == "__main__":
    main()
