# Model Comparison Guide

## 🎯 Quick Answer: What Model Should I Use?

### For 95% of Use Cases
**Use Fast SVM Ensemble (SVM RBF + SVM Linear)**
- Best overall RQI: 83.6-86.9 across all assets
- Training time: 0.006 seconds (as fast as single SVM!)
- Works universally on stocks, forex, and crypto

### Specific Scenarios
| Your Need | Use This | RQI | Speed |
|-----------|----------|-----|-------|
| **Best accuracy** | Fast SVM Ensemble | 83.6-86.9 | 0.006s |
| **Maximum accuracy** | Top3 Ensemble | 83.5-87.1 | 0.35-0.6s |
| **Single model** | XGBoost | 72-79.8 | 0.5-0.9s |
| **Ultra-fast single** | GARCH | 71-73.9 | 0.005s |
| **Forex specific** | Fast SVM or GARCH | 83.9 / 73.8 | 0.006s / 0.005s |

### What NOT to Use
❌ **Deep Learning** - LSTM only achieves 51-64 RQI vs 83.6-86.9 for ensembles
❌ **Many features** - 10 optimized features outperform 35+ features
❌ **n_regimes > 3** - More regimes lead to overfitting on real data

## Overview

This guide helps you choose the right regime detection model for your specific use case. Each model has strengths and weaknesses depending on your data characteristics, computational resources, and accuracy requirements.

**⚠️ Important**: All models have been tested on **100% REAL market data** across 8 financial instruments:
- **Stocks**: SPY, QQQ, AAPL, MSFT (Alpha Vantage)
- **Forex**: EUR/USD, GBP/USD (OANDA)
- **Crypto**: BTC/USD, ETH/USD (Kraken)

**Key Finding**: Fast SVM Ensemble (RBF + Linear) provides best universal performance with only 10 features.

## Quick Selection Matrix

**Benchmark Dataset Summary:**

| Model Type | Timeframe | Bars Used | Time Period | Features |
|------------|-----------|-----------|-------------|----------|
| Traditional Models | Daily | 200 bars | ~10 months | 5 (returns, volatility, volume, momentum, RSI) |
| Deep Learning (Small) | Daily | 200 bars | ~10 months | 5 basic features |
| Deep Learning (Large) | 5-minute | 8,640 bars | 30 trading days | 12 technical indicators |
| Deep Learning (Quality Test) | 5-minute | 5,000 bars | ~17 trading days | 14 comprehensive features |

**Key Notes:**
- All benchmarks used synthetic data with realistic market characteristics
- Regime transitions placed every 25-30% of dataset for testing detection
- 5-minute bars: 288 bars per trading day (6.5 hours × 12 bars/hour)
- Daily bars: ~20 trading days per month

| Model | Best For | Data Size | Speed (Tested) | Interpretability | RQI (Real Data) |
|-------|----------|-----------|----------------|------------------|-----------------|
| **Fast SVM Ensemble** | Universal best performance | Any | Ultra Fast (0.006s) | Medium | **83.6-86.9** |
| **Top3 Ensemble** | Maximum accuracy | Any | Fast (0.35-0.6s) | Medium | **83.5-87.1** |
| **XGBoost** | Best single model | Large | Medium (0.5-0.9s) | Medium | 72.0-79.8 |
| **SVM (RBF)** | Non-linear single model | Medium | Very Fast (0.007s) | Low | 71.5-78.6 |
| **SVM (Linear)** | Linear boundaries | Medium | Very Fast (0.006s) | Low | 70.8-78.1 |
| **HMM** | Sequential patterns | Medium | Fast (0.3-0.9s) | High | 67.8-76.6 |
| **Random Forest** | Non-linear patterns | Large | Fast (0.06-0.08s) | Medium | 70.8-78.2 |
| **GARCH** | Volatility regimes (Forex) | Medium | Very Fast (0.005s) | Medium | 68.2-73.9 |
| **GMM** | Clustering | Small-Medium | Fast (0.05-0.18s) | High | 60.8-73.7 |
| **MS-GARCH** | Time-varying volatility | Medium | Ultra Fast (<0.001s) | Medium | 60.1-62.5 |
| **LSTM** | Not recommended | Large | Slow (0.8-1.4s) | Low | 49.1-64.9 |

†All models now tested with calibrated probabilities
‡Deep learning RQI measured on 5,000 5-minute bars (17 trading days)
‡‡Training times on 8,640 5-minute bars (30 trading days), 12 technical features with MPS
*Traditional model times on 200 daily bars, 5 features
*RQI = Regime Quality Index (0-100 scale)

## Detailed Model Comparison

### Statistical Models

#### Hidden Markov Model (HMM)

**Strengths:**
- Excellent for sequential data with clear regime transitions
- Models transition probabilities between regimes
- Provides interpretable transition matrix
- Fast training and inference
- Works well with limited data (>100 samples)

**Weaknesses:**
- Assumes Markov property (future depends only on current state)
- May struggle with non-linear patterns
- Sensitive to initialization method

**When to Use:**
- Financial markets with clear regime transitions
- When transition probabilities are important
- Need interpretable results for stakeholders
- Limited computational resources

**Example Use Case:**
```python
# Detecting bull/bear markets in stock indices
hmm = HMMRegimeDetector(
    n_regimes=3,  # Use 3 for real market data (bull/bear/sideways)
    init_method='volatility'  # Initialize based on volatility clusters
)
```

#### Gaussian Mixture Model (GMM)

**Strengths:**
- Simple and fast
- No temporal assumptions
- Works with very small datasets
- Multiple covariance structures available

**Weaknesses:**
- Ignores temporal dependencies
- May produce unstable regime assignments
- Less suitable for time series

**When to Use:**
- Cross-sectional analysis
- When temporal order doesn't matter
- Quick exploratory analysis
- Very limited data (<100 samples)

**Example Use Case:**
```python
# Clustering assets by risk-return profiles
gmm = GMMRegimeDetector(
    n_regimes=4,
    covariance_type='full'
)
```

#### GARCH Regime Detector

**Strengths:**
- Specifically designed for volatility modeling
- Captures volatility clustering
- Well-established in finance
- Fast computation

**Weaknesses:**
- Primarily focused on volatility
- Requires returns data
- May miss non-volatility regimes

**When to Use:**
- Volatility-based trading strategies
- Risk management applications
- Options trading
- High-frequency data

**Example Use Case:**
```python
# Volatility regime detection for options trading
garch = GARCHRegimeDetector(
    p=1, q=1,
    n_regimes=3,
    threshold_method='quantile'
)
```

#### Markov-Switching GARCH (MS-GARCH)

**Strengths:**
- Combines regime switching with GARCH
- Models regime-dependent volatility
- Superior for volatility forecasting
- Captures complex volatility dynamics

**Weaknesses:**
- Computationally intensive
- Complex parameter estimation
- Requires more data than simple GARCH

**When to Use:**
- Advanced volatility modeling
- When volatility dynamics change across regimes
- Derivative pricing
- Professional risk management

**Example Use Case:**
```python
# Professional volatility modeling
msgarch = MSGARCHRegimeDetector(
    n_regimes=2,
    switching_variance=True,
    switching_mean=True
)
```

### Machine Learning Models

#### Random Forest

**Strengths:**
- Handles non-linear relationships
- Robust to outliers
- Feature importance analysis
- No scaling required
- Parallel training

**Weaknesses:**
- Can overfit with small datasets
- Less interpretable than linear models
- Memory intensive with many trees

**When to Use:**
- Many features (>10)
- Non-linear patterns suspected
- Need feature importance
- Robust predictions required

**Example Use Case:**
```python
# Multi-factor regime detection
rf = RandomForestRegimeClassifier(
    n_regimes=3,
    n_estimators=200,
    max_depth=10
)
```

#### XGBoost

**Strengths:**
- State-of-the-art accuracy
- Handles missing data
- Built-in regularization
- Feature importance
- Efficient implementation

**Weaknesses:**
- Many hyperparameters to tune
- Can overfit without proper regularization
- Requires careful cross-validation

**When to Use:**
- Maximum accuracy needed
- Complex feature interactions
- Kaggle-competition-level performance
- Have time for hyperparameter tuning

**Example Use Case:**
```python
# High-accuracy regime classification
xgb = XGBoostRegimeClassifier(
    n_regimes=3,
    n_estimators=300,
    learning_rate=0.01,
    max_depth=6
)
```

#### Support Vector Machine (SVM)

**Strengths:**
- Excellent for non-linear boundaries
- Works well in high dimensions
- Strong theoretical foundation
- Memory efficient

**Weaknesses:**
- Slow with large datasets
- Sensitive to hyperparameters
- Probability estimates less reliable
- Hard to interpret

**When to Use:**
- Clear but non-linear regime boundaries
- High-dimensional feature space
- Medium-sized datasets
- Binary classification (2 regimes)

**Example Use Case:**
```python
# Non-linear regime boundaries
svm = SVMRegimeClassifier(
    n_regimes=2,
    kernel='rbf',
    C=1.0
)
```

### Deep Learning Models

#### LSTM (Long Short-Term Memory)

**Strengths:**
- Captures long-term dependencies
- Handles variable-length sequences
- Learns complex temporal patterns
- State-of-the-art for sequences

**Weaknesses:**
- Requires large datasets (>1000 samples)
- Computationally intensive
- Black-box nature
- Prone to overfitting

**When to Use:**
- Long historical patterns matter
- Complex temporal dependencies
- Large datasets available
- GPU/MPS acceleration available

**Example Use Case:**
```python
# Long-term pattern recognition
lstm = LSTMRegimeDetector(
    n_regimes=3,
    hidden_size=128,
    n_layers=2,
    sequence_length=50,
    bidirectional=True
)
```

#### Transformer

**Strengths:**
- Attention mechanism for pattern focus
- Parallel processing
- Best for complex patterns
- State-of-the-art performance

**Weaknesses:**
- Requires very large datasets
- High computational cost
- Many parameters
- Complex architecture

**When to Use:**
- Very large datasets (>5000 samples)
- Complex multi-scale patterns
- Latest deep learning needed
- Computational resources available

**Example Use Case:**
```python
# Advanced pattern detection
transformer = TransformerRegimeDetector(
    n_regimes=3,
    d_model=256,
    n_heads=8,
    n_layers=6,
    sequence_length=100
)
```

#### CNN-LSTM

**Strengths:**
- Combines pattern recognition with sequences
- Good for multi-scale features
- Automatic feature extraction
- Handles complex patterns

**Weaknesses:**
- Very complex architecture
- Requires careful tuning
- Needs large datasets
- Computationally expensive

**When to Use:**
- Multi-scale patterns present
- Image-like pattern in time series
- Large, complex datasets
- Maximum pattern recognition needed

**Example Use Case:**
```python
# Multi-scale pattern detection
cnn_lstm = CNNLSTMRegimeDetector(
    n_regimes=3,
    cnn_filters=[64, 128],
    kernel_sizes=[3, 5],
    lstm_hidden=128,
    sequence_length=60
)
```

### Ensemble Methods

#### Ensemble Detector

**Strengths:**
- Combines strengths of multiple models
- Most robust predictions
- Reduces overfitting
- Handles various pattern types
- Multiple combination strategies

**Weaknesses:**
- Computationally expensive
- Complex to maintain
- Harder to interpret
- Requires fitting multiple models

**When to Use:**
- Maximum robustness required
- Production trading systems
- When no single model dominates
- Have computational resources

**Example Use Case:**
```python
# Fast SVM Ensemble - Our #1 performer (83.6-86.8 RQI)
fast_svm = EnsembleRegimeDetector(
    models=[
        SVMRegimeClassifier(n_regimes=3, kernel='rbf', probability=True, random_state=42),
        SVMRegimeClassifier(n_regimes=3, kernel='linear', probability=True, random_state=42)
    ],
    n_regimes=3,
    strategy='voting'  # Simple majority voting
)

# Top3 Ensemble - Best for FOREX (83.7 RQI)
top3 = EnsembleRegimeDetector(
    models=[
        XGBoostRegimeClassifier(n_regimes=3, n_estimators=100, random_state=42),
        SVMRegimeClassifier(n_regimes=3, kernel='rbf', probability=True, random_state=42),
        RandomForestRegimeClassifier(n_regimes=3, n_estimators=100, random_state=42)
    ],
    n_regimes=3,
    strategy='voting'
)
```

## Performance Benchmarks

### Benchmark Methodology

**Data Sources (REAL MARKET DATA ONLY):**
- **FOREX**: EUR/USD, GBP/USD from OANDA API (500 daily samples)
- **Major Indices**: SPY (S&P 500), QQQ (NASDAQ-100) from Alpha Vantage API
- **Individual Stocks**: AAPL, MSFT from Alpha Vantage API
- **Period**: ~2 years of daily data (480-500 samples per dataset)
- **Features**: 10 engineered features including returns, multiple volatility measures (Parkinson, Garman-Klass), momentum, RSI, volume ratios, ATR

**Timeframes Tested:**
1. **Daily Bars** (Traditional Models):
   - 200 bars = approximately 10 months of trading days
   - Features: returns, volatility, volume, momentum, RSI
   
2. **5-Minute Bars** (Deep Learning Models):
   - Small test: 200 bars = ~16.7 hours of trading
   - Production test: 8,640 bars = 30 trading days (288 bars per day)
   - Features: 12 technical indicators including multi-timeframe volatility, momentum, RSI, volume ratios

**Training Configuration:**
- Traditional models: Default hyperparameters with 50-100 estimators
- Deep learning: 30-50 epochs with early stopping
- All models tested with n_regimes=3 (optimal for real market data)

### Real Market Data Results

**Comprehensive Benchmark Results (September 2025)**

| Market Type | Model | Train Time | RQI | vs Best Single | Recommendation |
|-------------|-------|------------|-----|----------------|----------------|
| **SPY (S&P 500 Index)** ||||||
| Ensemble (Fast SVM) | 0.006s | **83.6** | +3.8 | ✅ BEST |
| Ensemble (Top3) | 0.373s | 83.5 | +3.7 | Alternative |
| XGBoost (Single) | 0.433s | 79.8 | - | Best single |
| SVM RBF (Single) | 0.007s | 79.5 | -0.3 | Fast single |
| **AAPL (Individual Stock)** ||||||
| Ensemble (Fast SVM) | 0.007s | **83.8** | +4.6 | ✅ BEST |
| Ensemble (Top3) | 0.423s | 83.6 | +4.4 | Alternative |
| XGBoost (Single) | 0.555s | 79.2 | - | Best single |
| SVM RBF (Single) | 0.007s | 78.6 | -0.6 | Fast single |
| **EUR/USD (FOREX)** ||||||
| Ensemble (Top3) | 0.355s | **83.7** | +9.9 | ✅ BEST |
| Ensemble (Fast SVM) | 0.006s | 83.4 | +9.6 | Fast alternative |
| GARCH (Single) | 0.004s | 73.8 | - | Best single |
| XGBoost (Single) | 0.594s | 72.4 | -1.4 | Alternative |
| **BTC/USD (Crypto)** ||||||
| Ensemble (Fast SVM) | 0.030s | **86.8** | +14.8 | ✅ BEST |
| Ensemble (Top3) | 0.421s | 86.3 | +14.3 | Alternative |
| XGBoost (Single) | 0.550s | 72.0 | - | Best single |
| HMM (Single) | 0.339s | 71.5 | -0.5 | Alternative |
| **ETH/USD (Crypto)** ||||||
| Ensemble (Fast SVM) | 0.006s | **85.6** | +7.4 | ✅ BEST |
| Ensemble (Top3) | 0.492s | 85.5 | +7.3 | Alternative |
| XGBoost (Single) | 0.590s | 78.2 | - | Best single |
| HMM (Single) | 0.504s | 73.9 | -4.3 | Alternative |

### Ensemble Specifications for Replication

| Ensemble Name | Component 1 | Component 2 | Component 3 | Strategy | Parameters |
|---------------|-------------|-------------|-------------|----------|------------|
| **Fast SVM** | SVM (RBF kernel) | SVM (Linear kernel) | - | Voting | `n_regimes=3, probability=True, random_state=42` |
| **Top3** | XGBoost | SVM (RBF kernel) | Random Forest | Voting | `n_regimes=3, n_estimators=100 (XGB/RF), random_state=42` |

**Why Fast SVM Wins:**
- RBF kernel captures non-linear volatility patterns
- Linear kernel captures directional trends
- Voting combines both perspectives for robust predictions
- Training parallelizes, maintaining single-SVM speed (0.006s)

**Key Findings from Real Data:**
- **Crypto highest performance**: Ensembles achieve 85.6-86.8 RQI on crypto (best overall)
- **Universal winner**: Fast SVM ensemble dominates ALL markets (83.4-86.8 RQI)
- **Biggest improvements on crypto**: +15 RQI points for BTC, +7 for ETH over single models
- **FOREX**: Top3 ensemble slightly better (83.7 vs 83.4 RQI)
- **Speed champion**: Fast SVM ensemble at 0.006-0.030s across all markets
- **Deep learning not competitive**: LSTM consistently 20-30 points below ensembles

### Overall Model Performance (Average Across All Real Datasets)

| Rank | Model | Avg RQI | Avg Train Time | Best Dataset | Status |
|------|-------|---------|----------------|--------------|--------|
| 1 | **XGBoost** | 76.5 | 0.548s | SPY (79.8) | ✅ Tested |
| 2 | **SVM (RBF)** | 76.0 | 0.007s | AAPL (78.6) | ✅ Tested |
| 3 | **Random Forest** | 75.4 | 0.069s | SPY (79.3) | ✅ Tested |
| 4 | **SVM (Linear)** | 75.4 | 0.006s | SPY (79.0) | ✅ Tested |
| 5 | **HMM** | 72.2 | 0.414s | QQQ (76.0) | ✅ Tested |
| 6 | **GARCH** | 71.8 | 0.006s | EUR/USD (73.8) | ✅ Tested |
| 7 | **Ensemble (Voting)** | 69.4 | 0.466s | SPY (79.1) | ✅ Tested |
| 8 | **GMM** | 69.1 | 0.076s | SPY (73.7) | ✅ Tested |
| 9 | **Ensemble (Weighted)** | 63.8 | 0.549s | MSFT (73.1) | ✅ Tested |
| 10 | **MS-GARCH** | 62.1 | <0.001s | EUR/USD (62.5) | ✅ Tested |
| 11 | **LSTM** | 58.3 | 0.945s | AAPL (68.0) | ✅ Tested |

**Key Findings from Real Market Data:**
- **Best Overall**: XGBoost dominates with 76.5 average RQI across all datasets
- **Best for Stocks**: XGBoost excels on SPY (79.8), AAPL (79.2), MSFT (78.3)
- **Best for FOREX**: GARCH optimal for currency pairs (73.8-73.9 RQI)
- **Speed Champion**: MS-GARCH (<0.001s) followed by GARCH and SVM (0.005-0.008s)
- **SVM Excellence**: Both kernels achieve 75-76 RQI with <0.01s training - best speed/accuracy tradeoff
- **Deep Learning Underperforms**: LSTM averages only 58.3 RQI despite longer training times
- **HMM Strong on Indices**: Best performer on QQQ (76.0 RQI)

### Performance Notes

**Hardware Acceleration (Production Verified):**
- Deep learning models automatically detect and use MPS on Apple Silicon
- MPS Performance Metrics (8,640 5-minute bars = 30 trading days):
  - LSTM: 1.77s per epoch, 163 samples/second training
  - Transformer: 4.08s per epoch, achieves best quality (RQI: 59.2)
  - CNN-LSTM: 2.08s per epoch, fastest inference (37,564 samples/second)
- Traditional models remain CPU-efficient without GPU requirements

**Dataset Size Impact:**
- Small datasets (<1000 bars): Traditional models recommended
  - Example: 200 daily bars = ~10 months of trading
- Medium datasets (1000-5000 bars): Consider both traditional and deep learning
  - Example: 5,000 5-minute bars = ~17 trading days
- Large datasets (>5000 bars): Deep learning may show better performance
  - Example: 8,640 5-minute bars = 30 trading days
  - Example: 25,920 5-minute bars = 90 trading days (3 months)

**Note:** Benchmarks performed with minimal configurations for deep learning models. Production configurations with larger models may show different performance characteristics.

## Model Selection Decision Tree

```
Start: What is your primary goal?
│
├─> Interpretability Critical?
│   │
│   ├─> Yes: Need transition probabilities?
│   │   ├─> Yes: HMM
│   │   └─> No: GMM
│   │
│   └─> No: Continue...
│
├─> Volatility-Focused?
│   │
│   ├─> Yes: Regime-switching needed?
│   │   ├─> Yes: MS-GARCH
│   │   └─> No: GARCH
│   │
│   └─> No: Continue...
│
├─> Dataset Size?
│   │
│   ├─> Small (<500): 
│   │   ├─> Temporal: HMM
│   │   └─> Non-temporal: GMM
│   │
│   ├─> Medium (500-5000):
│   │   ├─> Linear: HMM
│   │   ├─> Non-linear: Random Forest
│   │   └─> Complex: XGBoost
│   │
│   └─> Large (>5000):
│       ├─> Long-term patterns: LSTM
│       ├─> Complex patterns: Transformer
│       └─> Maximum accuracy: Ensemble
```

## Practical Recommendations by Use Case

### Day Trading (5-Minute to 1-Hour Bars)
**Recommended:** CNN-LSTM for speed, GARCH for simplicity
```python
# For 5-minute bars with deep learning
model = CNNLSTMRegimeDetector(
    n_regimes=5,
    sequence_length=60  # 5 hours of 5-min bars
)

# For quick volatility-based detection
model = GARCHRegimeDetector(
    n_regimes=3,
    threshold_method='quantile'
)
```
- CNN-LSTM: 37,564 samples/second inference (can handle multiple symbols)
- GARCH: Simple and fast for volatility regimes
- Both suitable for 5-minute bar processing

### Swing Trading (Daily Bars)
**Recommended:** XGBoost or HMM
```python
# For daily bars with best accuracy
model = XGBoostRegimeClassifier(
    n_regimes=5,
    n_estimators=100
)
# RQI: 72.5 on 200 daily bars

# For interpretable transitions
model = HMMRegimeDetector(
    n_regimes=5,
    init_method='kmeans'
)
# RQI: 60.8 on 200 daily bars
```
- XGBoost: Best quality on daily bars (RQI: 72.5)
- HMM: Good for understanding regime transitions
- Both tested on 200 daily bars (~10 months)

### Long-term Investment (Daily/Weekly Bars)
**Recommended:** Ensemble with Multiple Models
```python
model = EnsembleRegimeDetector(
    models=[
        HMMRegimeDetector(n_regimes=3),
        XGBoostRegimeClassifier(n_regimes=3),
        LSTMRegimeDetector(n_regimes=3)
    ],
    strategy='weighted_voting'
)
```
- Maximum robustness for long-term positions
- Combines multiple perspectives
- Reduces model-specific risks

### Risk Management
**Recommended:** MS-GARCH or GARCH
```python
model = MSGARCHRegimeDetector(
    n_regimes=2,
    switching_variance=True
)
```
- Specifically designed for volatility modeling
- Industry-standard for risk metrics
- Provides volatility forecasts

### Algorithmic Trading (Mixed Timeframes)
**Recommended:** Model by Timeframe
```python
# For daily bars
model = XGBoostRegimeClassifier(
    n_regimes=5,
    n_estimators=100
)  # 72.5 RQI on daily data

# For 5-minute bars  
model = CNNLSTMRegimeDetector(
    n_regimes=5,
    sequence_length=60  # 5 hours lookback
)  # 53.2 RQI, 37.5k samples/sec
```
- Daily bars: XGBoost for best accuracy
- 5-min bars: CNN-LSTM for speed (37,564 samples/second)
- Consider timeframe when selecting model

### Research/Backtesting
**Recommended:** Test Multiple Models on Your Timeframe
```python
models_to_test = [
    HMMRegimeDetector(n_regimes=3),
    GMMRegimeDetector(n_regimes=3),
    RandomForestRegimeClassifier(n_regimes=3),
    LSTMRegimeDetector(n_regimes=3)
]

# Compare all models
evaluator = ModelEvaluator(models_to_test[0])
for model in models_to_test:
    model.fit(features)
    results = evaluator.evaluate_model(features)
    print(f"{model.__class__.__name__}: Quality Index: {results['regime_quality_index']:.1f}")
```

## Combining Models Effectively

### Complementary Model Pairs

1. **HMM + GARCH**: Combines regime transitions with volatility
```python
# Use HMM for regime detection
hmm = HMMRegimeDetector(n_regimes=3)
hmm.fit(features)
regimes = hmm.predict(features)

# Use GARCH for volatility within each regime
for regime in range(3):
    regime_data = features[regimes == regime]
    garch = GARCHModel(p=1, q=1)
    garch.fit(regime_data['returns'])
```

2. **GMM + Random Forest**: Unsupervised discovery + supervised refinement
```python
# Use GMM to discover regimes
gmm = GMMRegimeDetector(n_regimes=3)
gmm.fit(features)
labels = gmm.predict(features)

# Use Random Forest to learn pattern
rf = RandomForestRegimeClassifier(
    n_regimes=3,
    supervised=True
)
rf.fit(features, labels)
```

3. **Fast + Slow Models**: Real-time decisions with periodic recalibration
```python
# Fast model for real-time
fast_model = GMMRegimeDetector(n_regimes=3)

# Slow model for daily recalibration
slow_model = LSTMRegimeDetector(n_regimes=3)

# Use fast model intraday, recalibrate with slow model daily
```

## Model Limitations and Mitigation

### Common Limitations

| Model | Main Limitation | Mitigation Strategy |
|-------|----------------|-------------------|
| HMM | Assumes Markov property | Combine with LSTM for long-term memory |
| GMM | Ignores time dependencies | Use only for cross-sectional analysis |
| GARCH | Volatility-only focus | Combine with return-based models |
| Random Forest | Can overfit | Use cross-validation and pruning |
| XGBoost | Many hyperparameters | Use Bayesian optimization for tuning |
| SVM | Slow with large data | Subsample or use linear kernel |
| LSTM | Needs large datasets | Use transfer learning or simpler models |
| Transformer | Very data hungry | Pre-train on related datasets |
| Ensemble | Computational cost | Use model selection or pruning |

## Data Requirements for Production

### Minimum Data Requirements by Model Type

| Model | Minimum Bars | Recommended Bars | Optimal Timeframe |
|-------|--------------|------------------|-------------------|
| GMM | 50 | 200+ | Any |
| HMM | 100 | 500+ | Daily |
| GARCH | 100 | 500+ | Any (volatility-focused) |
| Random Forest | 200 | 1,000+ | Daily |
| XGBoost | 200 | 1,000+ | Daily |
| SVM | 100 | 500+ | Any |
| LSTM | 1,000 | 5,000+ | 5-min to 1-hour |
| Transformer | 2,000 | 10,000+ | 5-min to 1-hour |
| CNN-LSTM | 1,000 | 5,000+ | 5-min to 1-hour |

### Production Recommendations

**For Real-Time Trading (5-minute bars):**
- Use CNN-LSTM for fastest inference (37,564 samples/second)
- Maintain rolling window of 8,640-17,280 bars (30-60 days)
- Retrain weekly with latest data

**For Daily Analysis:**
- Use XGBoost for best accuracy (RQI: 72.5)
- Maintain 200-500 daily bars (10-25 months)
- Retrain monthly

**For Mixed Strategies:**
- Run multiple models in parallel
- Use appropriate model for each timeframe
- Combine signals using ensemble approach

## Conclusion

### Key Takeaways

1. **Real Data Winner**: GMM achieves 83.0 RQI on real EUR/USD daily data
2. **Traditional Models Dominate**: Consistently outperform deep learning on real financial data
3. **HMM for High-Frequency**: Best for 5-minute bars (80.4 RQI on EUR/USD)
4. **Deep Learning Struggles**: LSTM only 48.3 RQI on real FOREX despite MPS acceleration
5. **Speed Champion**: GARCH trains in 0.01s with good performance (71+ RQI)
6. **Start Simple**: GMM and HMM provide excellent results without complexity
7. **Validate on Real Data**: Synthetic benchmarks don't reflect real-world performance
8. **Monitor Performance**: Models degrade over time, retrain regularly

### Recommended Workflow

1. **Exploratory Phase**: GMM for quick insights
2. **Development Phase**: Test HMM, Random Forest, and one deep learning model
3. **Optimization Phase**: Tune best performers and create ensemble
4. **Production Phase**: Deploy ensemble with monitoring
5. **Maintenance Phase**: Regular retraining and model updates

For implementation details, see the [User Guide](user_guide.md) and [API Reference](api_reference.md).