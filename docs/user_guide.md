# User Guide

## Introduction

MarketRegimeML helps you identify and analyze market regimes - distinct market states characterized by different statistical properties. Common regimes include:

- **Bull Market**: Upward trending with positive returns
- **Bear Market**: Downward trending with negative returns  
- **Neutral/Sideways**: Range-bound with mean reversion
- **High Volatility**: Crisis or uncertainty periods
- **Low Volatility**: Calm, trending markets

This guide walks through practical examples of using the library.

## Installation and Setup

### 1. Install the Library

```bash
# Basic installation
pip install marketregimeml

# Development installation with all features
pip install marketregimeml[dev,notebook]
```

### 2. Set Up API Keys

Create a `.env` file in your project root:

```bash
# For FOREX data (OANDA)
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_oanda_account_id

# For stock data (Alpha Vantage)
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
```

### 3. Import Required Modules

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import models
from marketregimeml.models import (
    HMMRegimeDetector,
    GMMRegimeDetector,
    EnsembleRegimeDetector
)

# Import data loaders
from marketregimeml.data.loaders import (
    OANDADataLoader,
    AlphaVantageDataLoader  
)

# Import evaluation tools
from marketregimeml.evaluation import (
    RegimeEvaluator,
    RegimeBacktester
)
```

## Loading Market Data

### FOREX Data with OANDA

```python
# Initialize OANDA loader
oanda = OANDADataLoader(
    api_key="your_key",
    account_id="your_account",
    practice=True  # Use practice account
)

# Fetch EUR/USD daily data
eurusd_data = oanda.fetch_ohlcv(
    symbol="EUR_USD",
    timeframe="D",
    start="2023-01-01",
    end="2023-12-31"
)

print(f"Loaded {len(eurusd_data)} days of EUR/USD data")
print(eurusd_data.head())
```

### Stock Data with Alpha Vantage

```python
# Initialize Alpha Vantage loader
av_loader = AlphaVantageDataLoader(api_key="your_key")

# Fetch Apple stock data
aapl_data = av_loader.fetch_ohlcv(
    symbol="AAPL",
    timeframe="daily",
    outputsize="full"  # Get full history
)

print(f"Loaded {len(aapl_data)} days of AAPL data")
```

### Loading Multiple Symbols

```python
# Load multiple currency pairs
symbols = ["EUR_USD", "GBP_USD", "USD_JPY"]

multi_data = {}
for symbol in symbols:
    multi_data[symbol] = oanda.fetch_ohlcv(
        symbol=symbol,
        timeframe="H4",
        limit=500
    )

# Combine into single DataFrame
combined = pd.DataFrame({
    symbol: data['close'] 
    for symbol, data in multi_data.items()
})
```

## Feature Engineering

### Basic Features

```python
def prepare_features(ohlcv_data):
    """Prepare basic features for regime detection."""
    
    features = pd.DataFrame(index=ohlcv_data.index)
    
    # Returns
    features['returns'] = ohlcv_data['close'].pct_change()
    
    # Volatility (High-Low)
    features['volatility'] = (
        ohlcv_data['high'] / ohlcv_data['low'] - 1
    )
    
    # Volume changes
    if 'volume' in ohlcv_data.columns:
        features['volume_change'] = ohlcv_data['volume'].pct_change()
    
    # Price momentum
    features['momentum'] = (
        ohlcv_data['close'] / ohlcv_data['close'].shift(5) - 1
    )
    
    # Remove NaN values
    features = features.dropna()
    
    return features

# Prepare features
features = prepare_features(eurusd_data)
print(f"Feature shape: {features.shape}")
```

### Advanced Features

```python
def add_technical_indicators(ohlcv_data, features):
    """Add technical indicators as features."""
    
    close = ohlcv_data['close']
    high = ohlcv_data['high']
    low = ohlcv_data['low']
    
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    features['bb_upper'] = (close - (sma + 2*std)) / close
    features['bb_lower'] = ((sma - 2*std) - close) / close
    
    # ATR (Average True Range)
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features['atr'] = tr.rolling(window=14).mean() / close
    
    return features.dropna()

# Add technical indicators
features = add_technical_indicators(eurusd_data, features)
```

## Basic Regime Detection

### Using Hidden Markov Model (HMM)

```python
# Create and fit HMM detector
hmm_detector = HMMRegimeDetector(
    n_regimes=3,
    covariance_type='diag',
    init_method='kmeans'  # Initialize with k-means
)

# Fit the model
hmm_detector.fit(features)

# Predict regimes
regimes = hmm_detector.predict(features)

# Get regime probabilities
probabilities = hmm_detector.predict_proba(features)

# Analyze results
unique, counts = np.unique(regimes, return_counts=True)
print("\nRegime Distribution:")
for regime, count in zip(unique, counts):
    pct = count / len(regimes) * 100
    print(f"  Regime {regime}: {count} periods ({pct:.1f}%)")

# Get transition matrix
transitions = hmm_detector.analyze_regime_transitions(features)
print("\nTransition Matrix:")
print(transitions['transition_matrix'])
```

### Using Gaussian Mixture Model (GMM)

```python
# Create and fit GMM detector
gmm_detector = GMMRegimeDetector(
    n_regimes=3,
    covariance_type='full',
    random_state=42
)

# Fit and predict
gmm_detector.fit(features)
gmm_regimes = gmm_detector.predict(features)

# Compare with HMM
agreement = np.mean(regimes == gmm_regimes)
print(f"HMM-GMM Agreement: {agreement:.1%}")
```

## Advanced Features

### Fuzzy Regime Matching

Handle uncertain regime boundaries:

```python
# Enable fuzzy matching
fuzzy_detector = HMMRegimeDetector(
    n_regimes=3,
    fuzzy_matching=True,
    fuzzy_threshold=0.7  # 70% confidence required
)

fuzzy_detector.fit(features)

# Get fuzzy predictions
fuzzy_regimes, confidence = fuzzy_detector.predict_fuzzy(features)

# Identify uncertain periods
uncertain_mask = confidence < 0.7
uncertain_pct = uncertain_mask.sum() / len(confidence) * 100

print(f"Uncertain classifications: {uncertain_pct:.1f}%")

# Handle uncertain periods
regime_signals = np.where(
    uncertain_mask,
    -1,  # No position when uncertain
    fuzzy_regimes
)
```

### Automatic Regime Optimization

Find the optimal number of regimes:

```python
# Auto-optimize regime count
auto_detector = HMMRegimeDetector(
    auto_optimize_regimes=True,
    min_regimes=2,
    max_regimes=7,
    random_state=42
)

auto_detector.fit(features)

print(f"Optimal number of regimes: {auto_detector.n_regimes}")
print(f"BIC score: {auto_detector.information_criteria(features)['bic']:.2f}")

# Test different regime counts manually
results = []
for n in range(2, 8):
    detector = HMMRegimeDetector(n_regimes=n)
    detector.fit(features)
    ic = detector.information_criteria(features)
    results.append({
        'n_regimes': n,
        'aic': ic['aic'],
        'bic': ic['bic']
    })

results_df = pd.DataFrame(results)
print("\nInformation Criteria by Regime Count:")
print(results_df)
```

## Ensemble Methods

### Creating an Ensemble

```python
# Create diverse models
models = [
    HMMRegimeDetector(n_regimes=3, init_method='kmeans'),
    HMMRegimeDetector(n_regimes=3, init_method='random'),
    GMMRegimeDetector(n_regimes=3, covariance_type='full'),
    GMMRegimeDetector(n_regimes=3, covariance_type='diag')
]

# Create ensemble with voting
ensemble = EnsembleRegimeDetector(
    models=models,
    strategy='voting',
    n_regimes=3
)

# Fit ensemble
ensemble.fit(features)

# Get ensemble predictions
ensemble_regimes = ensemble.predict(features)

# Check model agreement
print(f"Average model agreement: {ensemble.diagnostics['avg_agreement']:.1%}")
```

### Weighted Ensemble with Optimization

```python
# Create ensemble with weighted voting
weighted_ensemble = EnsembleRegimeDetector(
    n_regimes=3,
    strategy='weighted_voting',
    models=[
        HMMRegimeDetector(n_regimes=3),
        GMMRegimeDetector(n_regimes=3),
        GARCHRegimeDetector(n_regimes=3)
    ]
)

# Fit ensemble
weighted_ensemble.fit(features)

# Optimize weights based on likelihood
optimal_weights = weighted_ensemble.optimize_weights(
    features,
    criterion='log_likelihood'
)

print("Optimized Model Weights:")
for i, weight in enumerate(optimal_weights):
    model_name = weighted_ensemble.models[i].__class__.__name__
    print(f"  {model_name}: {weight:.3f}")

# Apply optimized weights
weighted_ensemble.weights = optimal_weights
optimized_regimes = weighted_ensemble.predict(features)
```

## Model Evaluation

### Computing Evaluation Metrics

```python
# Create evaluator
evaluator = RegimeEvaluator()

# Evaluate single model
hmm_metrics = evaluator.evaluate_model(
    model=hmm_detector,
    features=features
)

print("HMM Model Metrics:")
print(f"  Silhouette Score: {hmm_metrics['silhouette_score']:.3f}")
print(f"  Calinski-Harabasz: {hmm_metrics['calinski_harabasz_score']:.1f}")
print(f"  Regime Stability: {hmm_metrics['regime_stability']:.3f}")
print(f"  BIC: {hmm_metrics['bic']:.1f}")
```

### Comparing Multiple Models

```python
# Compare models
models_to_compare = {
    'HMM': hmm_detector,
    'GMM': gmm_detector,
    'Ensemble': ensemble
}

comparison_results = {}
for name, model in models_to_compare.items():
    metrics = evaluator.evaluate_model(model, features)
    comparison_results[name] = metrics

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_results).T
print("\nModel Comparison:")
print(comparison_df[['silhouette_score', 'bic', 'regime_stability']])
```

## Backtesting Trading Strategies

### Simple Regime-Based Strategy

```python
# Define strategy weights for each regime
# Regime 0: Conservative (bear market)
# Regime 1: Neutral
# Regime 2: Aggressive (bull market)
strategy_weights = {
    0: 0.3,   # 30% exposure in bear market
    1: 1.0,   # 100% exposure in neutral
    2: 1.5    # 150% exposure in bull market (leveraged)
}

# Create backtester
backtester = RegimeBacktester(
    model=hmm_detector,
    transaction_cost=0.001,  # 0.1% transaction cost
    initial_capital=100000
)

# Run backtest
results = backtester.backtest_strategy(
    features=features,
    returns=features['returns'],
    strategy_weights=strategy_weights,
    rebalance_frequency='W'  # Weekly rebalancing
)

print("\nBacktest Results:")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Annualized Return: {results['annualized_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

### Comparing Multiple Strategies

```python
# Define multiple strategies
strategies = {
    'Conservative': {0: 0.0, 1: 0.5, 2: 1.0},
    'Balanced': {0: 0.3, 1: 1.0, 2: 1.3},
    'Aggressive': {0: 0.5, 1: 1.0, 2: 2.0},
    'Buy_and_Hold': {0: 1.0, 1: 1.0, 2: 1.0}
}

# Compare strategies
comparison = backtester.compare_strategies(
    features=features,
    returns=features['returns'],
    strategies=strategies
)

print("\nStrategy Comparison:")
print(comparison[['total_return', 'sharpe_ratio', 'max_drawdown']])

# Find best strategy
best_strategy = comparison['sharpe_ratio'].idxmax()
print(f"\nBest Strategy: {best_strategy}")
```

### Walk-Forward Analysis

```python
# Rolling backtest with periodic refitting
rolling_results = backtester.rolling_backtest(
    features=features,
    returns=features['returns'],
    strategy_weights=strategy_weights,
    window_size=60,  # 60-period training window
    step_size=20      # Refit every 20 periods
)

# Analyze rolling performance
returns = [r['total_return'] for r in rolling_results]
sharpes = [r['sharpe_ratio'] for r in rolling_results]

print(f"\nRolling Backtest Summary:")
print(f"Average Return: {np.mean(returns):.2%}")
print(f"Return Std Dev: {np.std(returns):.2%}")
print(f"Average Sharpe: {np.mean(sharpes):.2f}")
print(f"Sharpe Std Dev: {np.std(sharpes):.2f}")
```

## Visualization

### Plotting Regimes

```python
import matplotlib.pyplot as plt

def plot_regimes_with_price(dates, prices, regimes):
    """Plot price with regime coloring."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot price with regime coloring
    colors = ['red', 'yellow', 'green']
    for regime in range(3):
        mask = regimes == regime
        ax1.scatter(dates[mask], prices[mask], 
                   c=colors[regime], alpha=0.6, s=10,
                   label=f'Regime {regime}')
    
    ax1.plot(dates, prices, 'k-', alpha=0.3, linewidth=0.5)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.set_title('Market Regimes')
    
    # Plot regime sequence
    ax2.plot(dates, regimes, 'b-', linewidth=1)
    ax2.fill_between(dates, 0, regimes, alpha=0.3)
    ax2.set_ylabel('Regime')
    ax2.set_xlabel('Date')
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_yticks([0, 1, 2])
    
    plt.tight_layout()
    plt.show()

# Plot regimes
plot_regimes_with_price(
    features.index,
    eurusd_data.loc[features.index, 'close'],
    regimes
)
```

### Plotting Transition Matrix

```python
def plot_transition_matrix(transition_matrix, regime_names=None):
    """Plot regime transition matrix as heatmap."""
    
    if regime_names is None:
        regime_names = [f'Regime {i}' for i in range(len(transition_matrix))]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set_xticks(range(len(regime_names)))
    ax.set_yticks(range(len(regime_names)))
    ax.set_xticklabels(regime_names)
    ax.set_yticklabels(regime_names)
    
    # Add values to cells
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix)):
            text = ax.text(j, i, f'{transition_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black')
    
    ax.set_xlabel('To Regime')
    ax.set_ylabel('From Regime')
    ax.set_title('Regime Transition Probabilities')
    
    plt.tight_layout()
    plt.show()

# Plot transition matrix
transitions = hmm_detector.analyze_regime_transitions(features)
plot_transition_matrix(
    transitions['transition_matrix'],
    regime_names=['Bear', 'Neutral', 'Bull']
)
```

## Production Deployment

### Creating a Regime Monitor

```python
class RegimeMonitor:
    """Monitor regimes in production."""
    
    def __init__(self, model, lookback_window=100):
        self.model = model
        self.lookback_window = lookback_window
        self.regime_history = []
        self.alert_callbacks = []
    
    def update(self, new_data):
        """Update with new market data."""
        
        # Prepare features
        features = prepare_features(new_data.tail(self.lookback_window))
        
        # Predict current regime
        current_regime = self.model.predict(features.tail(1))[0]
        
        # Get regime probability
        proba = self.model.predict_proba(features.tail(1))[0]
        confidence = proba.max()
        
        # Store in history
        self.regime_history.append({
            'timestamp': new_data.index[-1],
            'regime': current_regime,
            'confidence': confidence,
            'probabilities': proba
        })
        
        # Check for regime change
        if len(self.regime_history) > 1:
            prev_regime = self.regime_history[-2]['regime']
            if current_regime != prev_regime:
                self._trigger_alerts(prev_regime, current_regime, confidence)
        
        return current_regime, confidence
    
    def _trigger_alerts(self, old_regime, new_regime, confidence):
        """Trigger alerts on regime change."""
        
        alert = {
            'timestamp': self.regime_history[-1]['timestamp'],
            'old_regime': old_regime,
            'new_regime': new_regime,
            'confidence': confidence
        }
        
        for callback in self.alert_callbacks:
            callback(alert)
    
    def add_alert_callback(self, callback):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)

# Example usage
monitor = RegimeMonitor(hmm_detector)

def on_regime_change(alert):
    print(f"REGIME CHANGE at {alert['timestamp']}:")
    print(f"  {alert['old_regime']} → {alert['new_regime']}")
    print(f"  Confidence: {alert['confidence']:.1%}")

monitor.add_alert_callback(on_regime_change)

# Simulate real-time updates
for i in range(len(eurusd_data) - 100):
    window = eurusd_data.iloc[i:i+100]
    regime, confidence = monitor.update(window)
```

### Saving and Loading Models

```python
import pickle

# Save fitted model
with open('hmm_regime_model.pkl', 'wb') as f:
    pickle.dump(hmm_detector, f)

# Load model
with open('hmm_regime_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Verify loaded model works
test_prediction = loaded_model.predict(features.tail(10))
print(f"Loaded model predictions: {test_prediction}")
```

## Best Practices

### 1. Feature Selection

```python
# Test feature importance
from sklearn.feature_selection import mutual_info_regression

# Calculate mutual information
regimes = hmm_detector.predict(features)
mi_scores = []

for col in features.columns:
    mi = mutual_info_regression(
        features[[col]], 
        regimes,
        random_state=42
    )[0]
    mi_scores.append((col, mi))

# Sort by importance
mi_scores.sort(key=lambda x: x[1], reverse=True)

print("Feature Importance (Mutual Information):")
for feature, score in mi_scores:
    print(f"  {feature}: {score:.3f}")

# Select top features
top_features = [f[0] for f in mi_scores[:3]]
features_selected = features[top_features]
```

### 2. Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

cv_scores = []
for train_idx, test_idx in tscv.split(features):
    # Split data
    train_features = features.iloc[train_idx]
    test_features = features.iloc[test_idx]
    
    # Fit model on train
    cv_model = HMMRegimeDetector(n_regimes=3)
    cv_model.fit(train_features)
    
    # Evaluate on test
    test_likelihood = cv_model._compute_log_likelihood(test_features)
    cv_scores.append(test_likelihood)

print(f"CV Log-Likelihood: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
```

### 3. Handling Different Market Conditions

```python
def adaptive_regime_detection(data, market_condition='normal'):
    """Adapt regime detection to market conditions."""
    
    if market_condition == 'crisis':
        # Standard regimes during crisis
        detector = HMMRegimeDetector(
            n_regimes=3,
            covariance_type='full'
        )
    elif market_condition == 'trending':
        # Fewer regimes in trending markets
        detector = HMMRegimeDetector(
            n_regimes=2,
            init_method='volatility'
        )
    else:  # normal
        detector = HMMRegimeDetector(
            n_regimes=3,
            init_method='kmeans'
        )
    
    features = prepare_features(data)
    detector.fit(features)
    
    return detector

# Detect market condition
volatility = features['volatility'].rolling(20).mean()
current_vol = volatility.iloc[-1]
hist_vol = volatility.mean()

if current_vol > hist_vol * 2:
    condition = 'crisis'
elif features['returns'].rolling(20).mean().iloc[-1] > 0.001:
    condition = 'trending'
else:
    condition = 'normal'

print(f"Market condition: {condition}")
adaptive_model = adaptive_regime_detection(eurusd_data, condition)
```

## Troubleshooting

### Common Issues and Solutions

1. **Model doesn't converge**
```python
# Increase iterations and adjust tolerance
detector = HMMRegimeDetector(
    n_regimes=3,
    n_iter=200,  # Increase iterations
    convergence_tol=1e-4  # Relax tolerance
)
```

2. **Unstable regime assignments**
```python
# Use ensemble for stability
stable_ensemble = EnsembleRegimeDetector(
    n_regimes=3,
    strategy='consensus',
    consensus_threshold=0.7  # Require 70% agreement
)
```

3. **Poor backtesting performance**
```python
# Add minimum holding period
results = backtester.backtest_strategy(
    features=features,
    returns=returns,
    strategy_weights=strategy_weights,
    min_position_hold=5,  # Hold for at least 5 periods
    rebalance_frequency='W'  # Less frequent rebalancing
)
```

4. **API rate limits**
```python
# Use caching
loader = OANDADataLoader(
    api_key="your_key",
    account_id="your_account",
    cache_dir="./cache",  # Enable disk caching
    cache_expire_hours=24  # Cache for 24 hours
)
```

## Next Steps

1. **Explore Advanced Models**: Try deep learning models (LSTM, Transformer) for complex patterns
2. **Custom Features**: Implement domain-specific features for your market
3. **Multi-Asset Strategies**: Detect regimes across multiple assets simultaneously
4. **Real-time Integration**: Connect to live data feeds for production trading
5. **Risk Management**: Implement position sizing based on regime confidence

For more details, see the [API Reference](api_reference.md) and [Model Comparison Guide](model_comparison.md).