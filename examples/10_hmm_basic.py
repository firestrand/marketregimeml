#!/usr/bin/env python3
"""
Basic HMM Regime Detection
===========================
Simple example of using Hidden Markov Models for regime detection.

This example demonstrates:
- Creating and training an HMM model
- Detecting market regimes
- Analyzing regime transitions
- Visualizing results

Requirements:
- marketregimeml package
- pandas, numpy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.data.loaders import KrakenDataLoader


def load_market_data(symbol: str = "BTC/USD", timeframe: str = "1d", lookback_days: int = 900) -> pd.DataFrame:
    """Load real BTC/USD OHLCV from Kraken and build basic features."""
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    ohlcv = loader.fetch_ohlcv(symbol, timeframe, start, end)
    df = pd.DataFrame(index=ohlcv.index)
    df['price'] = ohlcv['close']
    df['returns'] = ohlcv['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df = df.dropna()
    return df


def train_hmm_model(features, n_regimes=3):
    """Train HMM model on features."""
    print(f"\nTraining HMM with {n_regimes} regimes...")
    
    # Initialize HMM model
    model = HMMRegimeDetector(
        n_regimes=n_regimes,
        covariance_type='full',
        n_iter=100,
        random_state=42
    )
    
    # Fit the model
    model.fit(features)
    
    print("Training complete!")
    
    return model


def analyze_regimes(model, features, data):
    """Analyze detected regimes."""
    # Predict regimes
    regimes = model.predict(features)
    regime_probs = model.predict_proba(features)
    
    # Add predictions to data
    data['predicted_regime'] = regimes
    
    # Calculate regime statistics
    print("\n" + "=" * 60)
    print("Regime Analysis")
    print("=" * 60)
    
    for regime in range(model.n_regimes):
        regime_data = data[data['predicted_regime'] == regime]
        if len(regime_data) > 0:
            print(f"\nRegime {regime}:")
            print(f"  Periods: {len(regime_data)} ({100*len(regime_data)/len(data):.1f}%)")
            print(f"  Avg Return: {regime_data['returns'].mean():.4f}")
            print(f"  Volatility: {regime_data['returns'].std():.4f}")
            print(f"  Return/Risk: {regime_data['returns'].mean()/regime_data['returns'].std():.2f}")
    
    # Analyze transitions
    print("\n" + "=" * 60)
    print("Transition Analysis")
    print("=" * 60)
    
    analysis = model.analyze_regime_transitions(regimes)
    transition_matrix = analysis['transition_matrix']
    print("\nTransition Matrix:")
    print("(rows=from, columns=to)")
    print(pd.DataFrame(transition_matrix).round(3))
    
    # Calculate average regime duration
    regime_changes = np.diff(regimes) != 0
    n_transitions = np.sum(regime_changes)
    avg_duration = len(regimes) / (n_transitions + 1)
    print(f"\nAverage regime duration: {avg_duration:.1f} periods")
    print(f"Total transitions: {n_transitions}")
    
    return regimes, regime_probs


def plot_results(data, regimes, regime_probs):
    """Visualize regime detection results."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Price with regime coloring
    ax1 = axes[0]
    colors = ['green', 'red', 'blue']
    for regime in range(3):
        mask = regimes == regime
        ax1.scatter(data.index[mask], data['price'][mask], 
                   c=colors[regime], alpha=0.3, s=10, label=f'Regime {regime}')
    ax1.plot(data.index, data['price'], 'k-', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel('Price')
    ax1.set_title('Price Series with Detected Regimes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Returns
    ax2 = axes[1]
    ax2.plot(data.index, data['returns'], 'b-', linewidth=0.5)
    ax2.set_ylabel('Returns')
    ax2.set_title('Daily Returns')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Rolling volatility
    ax3 = axes[2]
    ax3.plot(data.index, data['volatility'], 'r-', linewidth=1)
    ax3.set_ylabel('Volatility')
    ax3.set_title('20-Day Rolling Volatility')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Regime probabilities
    ax4 = axes[3]
    for regime in range(3):
        ax4.plot(data.index, regime_probs[:, regime], 
                label=f'Regime {regime}', alpha=0.7)
    ax4.set_ylabel('Probability')
    ax4.set_xlabel('Date')
    ax4.set_title('Regime Probabilities')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration."""
    print("=" * 60)
    print("Basic HMM Regime Detection")
    print("=" * 60)
    
    # Load real data
    print("\n1. Loading BTC/USD data from Kraken...")
    data = load_market_data("BTC/USD", "1d", 1000)
    print(f"   Loaded {len(data)} periods of data")
    
    # Prepare features for HMM
    print("\n2. Preparing features...")
    features = data[['returns', 'volatility']].values
    print(f"   Feature shape: {features.shape}")
    
    # Train HMM model
    model = train_hmm_model(features, n_regimes=3)
    
    # Analyze regimes
    regimes, regime_probs = analyze_regimes(model, features, data)
    
    # No ground truth available for real data
    
    # Model diagnostics
    print("\n" + "=" * 60)
    print("Model Diagnostics")
    print("=" * 60)
    
    diagnostics = model.get_diagnostics()
    print(f"\nLog-likelihood: {diagnostics.get('log_likelihood', 'N/A'):.2f}")
    print(f"AIC: {diagnostics.get('aic', 'N/A'):.2f}")
    print(f"BIC: {diagnostics.get('bic', 'N/A'):.2f}")
    
    # Plot results
    print("\n3. Plotting results...")
    plot_results(data, regimes, regime_probs)
    
    return model, data, regimes


if __name__ == "__main__":
    model, data, regimes = main()
    print("\n" + "=" * 60)
    print("HMM regime detection complete!")
    print("=" * 60)
