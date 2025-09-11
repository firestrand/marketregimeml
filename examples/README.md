# MarketRegimeML Examples

This directory contains example scripts demonstrating the **focused regime detection capabilities** of MarketRegimeML.

**Note**: This library focuses purely on regime detection. No trading strategies, backtesting, or portfolio management examples are included.

## 🎯 Core Examples

### 1. 01_quickstart.py
**Basic introduction to regime detection**:
- Loading real market data
- Simple feature preparation
- Training HMM model
- Basic evaluation metrics

```bash
python 01_quickstart.py
```

### 2. 14_ensemble_methods.py
**Ensemble regime detection**:
- Multiple model combination strategies
- Consensus-based regime identification
- Model diversity analysis

```bash
python 14_ensemble_methods.py [--quick]
```
 
### 3. 42_transition_analysis.py
Analyze transition matrices and forward returns after regime changes using real data.
```bash
python 42_transition_analysis.py
```

### 4. 43_model_comparison.py
Compare HMM vs GMM on real data with `ModelEvaluator`.
```bash
python 43_model_comparison.py
```

Note on long-running examples: `14_ensemble_methods.py` and `41_regime_optimization.py` can take a while on full history.
- Use `--quick` for a smaller window.
- Use `--tiny` for a very small window intended for CI/timeboxed runs.

## 📊 What These Examples Demonstrate

### ✅ **Regime Detection Focus**
- Identifying market regimes using various algorithms
- Evaluating regime stability and persistence
- Measuring clustering quality and temporal consistency
- Comparing model performance for regime detection

### ❌ **What's NOT Included**
- Trading strategy implementation
- Backtesting frameworks
- Portfolio management
- Risk management systems
- Financial performance metrics

## 🚀 Running the Examples

### Prerequisites

1. Install MarketRegimeML in a clean virtual environment:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pandas matplotlib scikit-learn hmmlearn python-dotenv
```

2. Set up API keys (for real data examples):
```bash
# Create .env file
OANDA_API_KEY=your_key
OANDA_ACCOUNT_ID=your_account
ALPHAVANTAGE_API_KEY=your_key
```

3. No API key needed for core examples (Kraken public BTC/USD data). Optional integrations (OANDA, Alpha Vantage, Binance) require their own credentials if you use those examples.

### Example Output

Each example generates:
- Console output with regime detection metrics
- Regime quality assessments
- Model comparison results
- Temporal analysis of regime transitions

## 📖 Learning Path

1. Start with `01_quickstart.py`  
2. Explore `14_ensemble_methods.py`  
3. Deep dive into `42_transition_analysis.py` and `43_model_comparison.py`
4. **Work with your own data** - Apply to real market data

## 🔧 Common Patterns

### Loading Data
```python
from marketregimeml.data.loaders import OANDADataLoader

loader = OANDADataLoader(api_key="your_key", account_id="your_account")
data = loader.fetch_ohlcv("EUR_USD", "D", limit=500)
```

### Preparing Features for Regime Detection
```python
features = pd.DataFrame({
    'returns': data['close'].pct_change(),
    'volatility': data['high'] / data['low'] - 1,
    'momentum': data['close'].pct_change(20),
    'volume_change': data['volume'].pct_change()
}).dropna()
```

### Detecting Regimes
```python
from marketregimeml.models import HMMRegimeDetector

detector = HMMRegimeDetector(n_regimes=3)
detector.fit(features)
regimes = detector.predict(features)
probabilities = detector.predict_proba(features)
```

### Evaluating Regime Quality
```python
from marketregimeml.evaluation.metrics import RegimeMetrics

metrics = RegimeMetrics()
quality = metrics.regime_quality_index(features.values, regimes, probabilities)
stability = metrics.regime_stability(regimes)
```

### Comparing Models
See `43_model_comparison.py` for a full example using real data and `ModelEvaluator`.

## 💡 Use Cases for Regime Detection

1. **Market Research**: Identify regime patterns in different markets/timeframes
2. **Risk Management**: Detect regime changes for position sizing in external systems
3. **Feature Engineering**: Generate regime-based features for other trading models
4. **Market Analysis**: Understand market state transitions and persistence
5. **Strategy Input**: Use regime information as input to separate trading systems

## 🤝 Contributing Examples

We welcome new examples! Please ensure they use real data sources (no synthetic generation) and focus on regime detection.

## 🎯 Philosophy

These examples demonstrate that **effective regime detection doesn't require complex trading frameworks**. By focusing solely on identifying market regimes, this library provides a clean, reliable foundation for your own analysis and trading systems.

**Regime detection is the input, not the entire solution.**
