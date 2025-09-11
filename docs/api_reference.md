# API Reference

## Table of Contents

- [Regime Count Recommendations](#regime-count-recommendations)
- [Base Classes](#base-classes)
- [Statistical Models](#statistical-models)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Models](#deep-learning-models)
- [Ensemble Methods](#ensemble-methods)
- [Evaluation](#evaluation)
- [Data Loaders](#data-loaders)
- [Configuration](#configuration)

## Regime Count Recommendations

Based on comprehensive benchmarks using 100% real market data, **we recommend 5 regimes as the optimal default** across asset classes:

### Optimal Regime Counts by Asset Class

| Asset Class | Recommended Count | Performance Improvement | Best Models |
|-------------|------------------|------------------------|-------------|
| **Cryptocurrency** | 5-7 regimes | +10% RQI | HMM, RandomForest |
| **Stocks** | 5 regimes | +16% RQI | RandomForest, HMM |
| **Forex** | 3-5 regimes | +2% RQI | XGBoost, HMM |

### Key Insights

- **5 regimes** provide optimal balance between model complexity and performance
- **Higher regime counts (5-7)** capture more nuanced market states
- **Traditional 3-regime models** underperform across crypto and stock markets
- **Asset-specific optimization** recommended for production systems

**All MarketRegimeML models now default to 5 regimes based on these findings.**

## Base Classes

### BaseRegimeDetector

Abstract base class for all regime detection models.

```python
class BaseRegimeDetector(ABC)
```

#### Parameters
- `n_regimes` (int, default=5): Number of market regimes to detect (2-9). **Default updated to 5 based on benchmark optimization**
- `fuzzy_matching` (bool, default=False): Enable fuzzy regime boundaries
- `fuzzy_threshold` (float, default=0.7): Confidence threshold for fuzzy matching
- `auto_optimize_regimes` (bool, default=False): Automatically optimize regime count
- `min_regimes` (int, default=2): Minimum regimes for optimization
- `max_regimes` (int, default=9): Maximum regimes for optimization
- `random_state` (int, optional): Random seed for reproducibility

#### Methods

##### fit(features, **kwargs)
Fit the model to market data.

**Parameters:**
- `features` (pd.DataFrame): Market features with shape (n_samples, n_features)
- `**kwargs`: Additional model-specific parameters

**Returns:**
- `self`: Fitted model instance

##### predict(features)
Predict regime labels for each time period.

**Parameters:**
- `features` (pd.DataFrame): Market features

**Returns:**
- `np.ndarray`: Regime labels (0 to n_regimes-1)

##### predict_proba(features)
Get regime probabilities for each time period.

**Parameters:**
- `features` (pd.DataFrame): Market features

**Returns:**
- `np.ndarray`: Probabilities with shape (n_samples, n_regimes)

##### predict_fuzzy(features)
Get fuzzy regime predictions with confidence scores.

**Parameters:**
- `features` (pd.DataFrame): Market features

**Returns:**
- `tuple`: (regime_labels, confidence_scores)

##### analyze_regime_transitions(features)
Analyze regime transition patterns.

**Parameters:**
- `features` (pd.DataFrame): Market features

**Returns:**
- `dict`: Contains 'transition_matrix', 'steady_state', 'mean_duration'

##### optimize_regime_count(features, criterion='bic')
Find optimal number of regimes.

**Parameters:**
- `features` (pd.DataFrame): Market features
- `criterion` (str): 'aic' or 'bic'

**Returns:**
- `int`: Optimal number of regimes

---

## Statistical Models

### HMMRegimeDetector

Hidden Markov Model for sequential regime detection.

```python
from marketregimeml.models import HMMRegimeDetector

detector = HMMRegimeDetector(
    n_regimes=3,
    covariance_type='diag',
    n_iter=100,
    init_method='kmeans'
)
```

#### Parameters
- `n_regimes` (int): Number of hidden states
- `covariance_type` (str): 'spherical', 'diag', 'full', 'tied'
- `n_iter` (int): Maximum EM iterations
- `init_method` (str): 'kmeans', 'random', 'volatility'
- `convergence_tol` (float): Convergence tolerance

#### Example
```python
# Detect market regimes with HMM
hmm = HMMRegimeDetector(n_regimes=3, init_method='volatility')
hmm.fit(features)
regimes = hmm.predict(features)
transition_matrix = hmm.get_transition_matrix()
```

### GMMRegimeDetector

Gaussian Mixture Model for clustering-based regime detection.

```python
from marketregimeml.models import GMMRegimeDetector

detector = GMMRegimeDetector(
    n_regimes=3,
    covariance_type='full',
    max_iter=100
)
```

#### Parameters
- `n_regimes` (int): Number of Gaussian components
- `covariance_type` (str): 'spherical', 'diag', 'full', 'tied'
- `max_iter` (int): Maximum EM iterations
- `init_params` (str): 'kmeans' or 'random'
- `reg_covar` (float): Regularization for covariance

### GARCHRegimeDetector

GARCH-based volatility regime detection.

```python
from marketregimeml.models import GARCHRegimeDetector

detector = GARCHRegimeDetector(
    p=1,
    q=1,
    n_regimes=3,
    threshold_method='quantile'
)
```

#### Parameters
- `p` (int): GARCH lag order
- `q` (int): ARCH lag order
- `n_regimes` (int): Number of volatility regimes
- `threshold_method` (str): 'quantile', 'kmeans', 'jenks'
- `vol_target` (bool): Use volatility targeting

### MSGARCHRegimeDetector

Markov-Switching GARCH for regime-dependent volatility.

```python
from marketregimeml.models import MSGARCHRegimeDetector

detector = MSGARCHRegimeDetector(
    n_regimes=2,
    switching_variance=True,
    switching_mean=False
)
```

#### Parameters
- `n_regimes` (int): Number of volatility regimes
- `switching_variance` (bool): Allow variance to switch
- `switching_mean` (bool): Allow mean to switch
- `max_iter` (int): Maximum iterations for optimization

---

## Machine Learning Models

### RandomForestRegimeClassifier

Ensemble tree-based regime classification.

```python
from marketregimeml.models import RandomForestRegimeClassifier

classifier = RandomForestRegimeClassifier(
    n_regimes=3,
    n_estimators=100,
    supervised=False
)
```

#### Parameters
- `n_regimes` (int): Number of regimes
- `n_estimators` (int): Number of trees
- `max_depth` (int, optional): Maximum tree depth
- `supervised` (bool): Use supervised learning if labels available
- `feature_importance_threshold` (float): Minimum feature importance

#### Methods
- `get_feature_importance()`: Get feature importance scores
- `plot_feature_importance()`: Visualize feature importance

### XGBoostRegimeClassifier

Gradient boosting for regime classification.

```python
from marketregimeml.models import XGBoostRegimeClassifier

classifier = XGBoostRegimeClassifier(
    n_regimes=3,
    n_estimators=100,
    learning_rate=0.1
)
```

#### Parameters
- `n_regimes` (int): Number of regimes
- `n_estimators` (int): Number of boosting rounds
- `learning_rate` (float): Boosting learning rate
- `max_depth` (int): Maximum tree depth
- `enable_categorical` (bool): Support categorical features

### SVMRegimeClassifier

Support Vector Machine for non-linear regime boundaries.

```python
from marketregimeml.models import SVMRegimeClassifier

classifier = SVMRegimeClassifier(
    n_regimes=3,
    kernel='rbf',
    supervised=False
)
```

#### Parameters
- `n_regimes` (int): Number of regimes
- `kernel` (str): 'linear', 'poly', 'rbf', 'sigmoid'
- `C` (float): Regularization parameter
- `gamma` (str or float): Kernel coefficient
- `supervised` (bool): Use supervised learning

---

## Deep Learning Models

All deep learning models support Apple Silicon MPS acceleration.

### LSTMRegimeDetector

Long Short-Term Memory for temporal patterns.

```python
from marketregimeml.models import LSTMRegimeDetector

detector = LSTMRegimeDetector(
    n_regimes=3,
    hidden_size=64,
    n_layers=2,
    sequence_length=20
)
```

#### Parameters
- `n_regimes` (int): Number of regimes
- `hidden_size` (int): LSTM hidden state size
- `n_layers` (int): Number of LSTM layers
- `sequence_length` (int): Input sequence length
- `dropout` (float): Dropout probability
- `bidirectional` (bool): Use bidirectional LSTM
- `learning_rate` (float): Adam optimizer learning rate
- `batch_size` (int): Training batch size
- `n_epochs` (int): Training epochs

#### Device Support
```python
# Automatically uses MPS on Apple Silicon
detector = LSTMRegimeDetector(n_regimes=3)
print(f"Using device: {detector.device}")  # mps, cuda, or cpu
```

### TransformerRegimeDetector

Attention-based regime detection.

```python
from marketregimeml.models import TransformerRegimeDetector

detector = TransformerRegimeDetector(
    n_regimes=3,
    d_model=128,
    n_heads=8,
    n_layers=4
)
```

#### Parameters
- `n_regimes` (int): Number of regimes
- `d_model` (int): Model dimension
- `n_heads` (int): Number of attention heads
- `n_layers` (int): Number of transformer layers
- `sequence_length` (int): Input sequence length
- `dropout` (float): Dropout probability
- `learning_rate` (float): Adam optimizer learning rate

### CNNLSTMRegimeDetector

Convolutional feature extraction with LSTM.

```python
from marketregimeml.models import CNNLSTMRegimeDetector

detector = CNNLSTMRegimeDetector(
    n_regimes=3,
    cnn_filters=[32, 64],
    kernel_sizes=[3, 3],
    lstm_hidden=64
)
```

#### Parameters
- `n_regimes` (int): Number of regimes
- `cnn_filters` (list): Number of filters per CNN layer
- `kernel_sizes` (list): Kernel size per CNN layer
- `lstm_hidden` (int): LSTM hidden size
- `sequence_length` (int): Input sequence length

---

## Ensemble Methods

### EnsembleRegimeDetector

Combine multiple models for robust predictions.

```python
from marketregimeml.models import EnsembleRegimeDetector

ensemble = EnsembleRegimeDetector(
    n_regimes=3,
    models=None,  # Uses default model set
    strategy='weighted_voting',
    consensus_threshold=0.6
)
```

#### Parameters
- `n_regimes` (int): Number of regimes
- `models` (list, optional): Custom list of detector instances
- `strategy` (str): Ensemble strategy
  - `'voting'`: Simple majority voting
  - `'weighted_voting'`: Performance-weighted voting
  - `'bayesian'`: Bayesian model averaging
  - `'consensus'`: Require agreement threshold
  - `'stacking'`: Meta-learning with LogisticRegression
- `weights` (array-like, optional): Manual model weights
- `consensus_threshold` (float): Agreement threshold for consensus

#### Methods

##### optimize_weights(features, criterion='log_likelihood')
Optimize model weights based on performance.

**Parameters:**
- `features` (pd.DataFrame): Training features
- `criterion` (str): 'log_likelihood', 'aic', 'bic'

**Returns:**
- `np.ndarray`: Optimized weights

##### cross_validate_models(features, n_splits=5)
Cross-validate all models in ensemble.

**Parameters:**
- `features` (pd.DataFrame): Training features
- `n_splits` (int): Number of CV folds

**Returns:**
- `dict`: Cross-validation scores per model

##### get_model_contributions(features)
Analyze each model's contribution to ensemble.

**Parameters:**
- `features` (pd.DataFrame): Features for analysis

**Returns:**
- `pd.DataFrame`: Model contribution statistics

#### Example
```python
# Create ensemble with custom models
from marketregimeml.models import (
    HMMRegimeDetector,
    GMMRegimeDetector,
    LSTMRegimeDetector
)

models = [
    HMMRegimeDetector(n_regimes=3),
    GMMRegimeDetector(n_regimes=3),
    LSTMRegimeDetector(n_regimes=3)
]

ensemble = EnsembleRegimeDetector(
    models=models,
    strategy='weighted_voting'
)

# Fit and optimize weights
ensemble.fit(features)
optimal_weights = ensemble.optimize_weights(features)

# Get predictions with model agreement
predictions = ensemble.predict(features)
agreement = ensemble.diagnostics['avg_agreement']
```

---

## Feature Engineering

### ComprehensiveFeatures

All-in-one feature engineering class combining multiple feature types optimized for regime detection.

```python
from marketregimeml.features import ComprehensiveFeatures

feature_engine = ComprehensiveFeatures()
```

#### Methods

##### create_features(ohlcv, feature_sets=None, windows=None)
Create comprehensive feature set from OHLCV data.

**Parameters:**
- `ohlcv` (pd.DataFrame): OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
- `feature_sets` (list, optional): Feature categories to include. 
  Options: ['price', 'returns', 'volatility', 'technical', 'statistical', 'entropy', 'microstructure']
  Default: all feature sets
- `windows` (list, optional): Window sizes for rolling features. Default: [5, 10, 20, 50]

**Returns:**
- `pd.DataFrame`: Feature matrix with 20-37+ engineered features

##### create_ml_optimized_features(ohlcv, max_features=50)
Create feature set optimized for machine learning models.

**Parameters:**
- `ohlcv` (pd.DataFrame): OHLCV data
- `max_features` (int): Maximum number of features to return

**Returns:**
- `pd.DataFrame`: ML-optimized features with variance-based selection

##### create_regime_optimized_features(ohlcv, n_regimes=3)
Create features optimized for specific number of regimes.

**Parameters:**
- `ohlcv` (pd.DataFrame): OHLCV data  
- `n_regimes` (int): Number of regimes to optimize for

**Returns:**
- `pd.DataFrame`: Regime-specific optimized features

#### Example
```python
# Create all features
features = feature_engine.create_features(
    ohlcv_data,
    feature_sets=['returns', 'volatility', 'technical'],
    windows=[10, 20]
)

# ML-optimized features for tree-based models
ml_features = feature_engine.create_ml_optimized_features(
    ohlcv_data, 
    max_features=30
)

# Features optimized for 5-regime detection
regime_features = feature_engine.create_regime_optimized_features(
    ohlcv_data,
    n_regimes=5
)
```

### Feature Categories

#### TechnicalIndicators
- RSI (7, 14, 21 periods)
- MACD (line, signal, histogram)
- Bollinger Bands (position, width)
- Stochastic (K, D)
- ATR (14 period)

#### VolatilityFeatures
- Parkinson estimator
- Garman-Klass estimator  
- Yang-Zhang estimator
- Volatility ratios

#### StatisticalFeatures
- Autocorrelation (multiple lags)
- Hurst exponent
- Jarque-Bera test statistic

#### EntropyFeatures
- Sample entropy
- Approximate entropy
- Permutation entropy

#### MicrostructureFeatures
- Volume ratios and trends
- Price efficiency
- Spread measures
- Amihud illiquidity

---

## Evaluation

### RegimeEvaluator

Comprehensive model evaluation metrics.

```python
from marketregimeml.evaluation import RegimeEvaluator

evaluator = RegimeEvaluator()
metrics = evaluator.evaluate_model(
    model=detector,
    features=features,
    true_regimes=None  # Optional ground truth
)
```

#### Methods

##### evaluate_model(model, features, true_regimes=None)
Compute all evaluation metrics.

**Returns dict with:**
- `silhouette_score`: Cluster separation quality
- `calinski_harabasz_score`: Ratio of between-cluster to within-cluster variance
- `davies_bouldin_score`: Average similarity between clusters
- `log_likelihood`: Model likelihood
- `aic`: Akaike Information Criterion
- `bic`: Bayesian Information Criterion
- `regime_stability`: Persistence of regimes
- `transition_entropy`: Randomness of transitions

##### compare_models(models, features, true_regimes=None)
Compare multiple models.

**Returns:**
- `pd.DataFrame`: Metrics comparison table

### RegimeBacktester

Backtest trading strategies based on regimes.

```python
from marketregimeml.evaluation import RegimeBacktester

backtester = RegimeBacktester(
    model=detector,
    transaction_cost=0.001,
    initial_capital=100000
)
```

#### Parameters
- `model`: Fitted regime detector
- `transaction_cost` (float): Cost per trade (fraction)
- `initial_capital` (float): Starting capital
- `risk_free_rate` (float): Risk-free rate for Sharpe ratio

#### Methods

##### backtest_strategy(features, returns, strategy_weights, **kwargs)
Run strategy backtest.

**Parameters:**
- `features` (pd.DataFrame): Market features
- `returns` (pd.Series): Asset returns
- `strategy_weights` (dict): Position weights per regime
- `rebalance_frequency` (str): 'D', 'W', 'M'
- `min_position_hold` (int): Minimum holding periods
- `lookback_window` (int, optional): Rolling window for refitting

**Returns dict with:**
- `total_return`: Cumulative return
- `annualized_return`: Annualized return
- `sharpe_ratio`: Risk-adjusted return
- `max_drawdown`: Maximum peak-to-trough loss
- `calmar_ratio`: Return/MaxDrawdown ratio
- `win_rate`: Percentage of positive returns
- `portfolio_values`: Time series of portfolio value

##### compare_strategies(features, returns, strategies)
Compare multiple strategies.

**Parameters:**
- `strategies` (dict): Named strategy definitions

**Returns:**
- `pd.DataFrame`: Strategy comparison metrics

##### rolling_backtest(features, returns, strategy_weights, window_size, step_size)
Walk-forward analysis.

**Returns:**
- `list`: Results for each window

---

## Data Loaders

### OANDADataLoader

FOREX data from OANDA.

```python
from marketregimeml.data.loaders import OANDADataLoader

loader = OANDADataLoader(
    api_key="your_key",
    account_id="your_account",
    practice=True
)

# Fetch OHLCV data
data = loader.fetch_ohlcv(
    symbol="EUR_USD",
    timeframe="H4",
    start="2023-01-01",
    end="2023-12-31"
)
```

#### Methods
- `fetch_ohlcv()`: Get OHLCV data
- `fetch_multiple()`: Get multiple symbols
- `get_available_symbols()`: List available pairs
- `validate_symbol()`: Check symbol validity

### AlphaVantageDataLoader

Stock data from Alpha Vantage.

```python
from marketregimeml.data.loaders import AlphaVantageDataLoader

loader = AlphaVantageDataLoader(api_key="your_key")

# Fetch daily stock data
data = loader.fetch_ohlcv(
    symbol="AAPL",
    timeframe="daily",
    outputsize="full"
)
```

#### Rate Limiting
- Automatic rate limit handling
- Two-tier caching (memory + disk)
- Parquet storage for efficiency

---

## Configuration

### ConfigLoader

YAML-based configuration management.

```python
from marketregimeml.config import ConfigLoader

config = ConfigLoader()

# Get market configuration
forex_config = config.get_market_config('forex')

# Get feature definitions
features = config.get_feature_config()

# Get model parameters
model_params = config.get_regime_config('hmm')
```

#### Configuration Files
- `config/markets.yaml`: Data sources and symbols
- `config/features.yaml`: Feature engineering
- `config/regimes.yaml`: Model parameters

#### Environment Variables
```python
# Automatically loaded from .env
config = ConfigLoader()
oanda_key = config.get_env('OANDA_API_KEY')
```

---

## Utilities

### Feature Engineering

```python
from marketregimeml.features import FeatureEngineer

engineer = FeatureEngineer()

# Add technical indicators
features = engineer.add_technical_indicators(
    data,
    indicators=['rsi', 'macd', 'bollinger']
)

# Add volatility measures
features = engineer.add_volatility_features(
    data,
    methods=['yang_zhang', 'garman_klass']
)

# Add entropy features
features = engineer.add_entropy_features(
    data,
    types=['approximate', 'sample', 'permutation']
)
```

### Visualization

```python
from marketregimeml.visualization import RegimePlotter

plotter = RegimePlotter()

# Plot regime transitions
plotter.plot_regimes(
    dates=data.index,
    regimes=predictions,
    prices=data['close']
)

# Plot transition matrix
plotter.plot_transition_matrix(
    transition_matrix,
    regime_names=['Bear', 'Neutral', 'Bull']
)

# Plot regime statistics
plotter.plot_regime_statistics(
    model=detector,
    features=features
)
```

---

## Error Handling

All methods raise appropriate exceptions:

- `ValueError`: Invalid parameters or data
- `NotFittedError`: Model used before fitting
- `DataValidationError`: Data fails validation
- `APIError`: External API issues

Example:
```python
from marketregimeml.exceptions import NotFittedError

try:
    predictions = detector.predict(features)
except NotFittedError:
    print("Model must be fitted first!")
    detector.fit(features)
    predictions = detector.predict(features)
```

---

## Best Practices

1. **Data Preparation**
   - Ensure no missing values in features
   - Standardize features if needed
   - Use appropriate sequence length for deep learning

2. **Model Selection**
   - HMM: Sequential data with clear transitions
   - GMM: When regime order doesn't matter
   - GARCH: Volatility-focused analysis
   - Deep Learning: Large datasets (>1000 samples)
   - Ensemble: When robustness is critical

3. **Hyperparameter Tuning**
   - Use cross-validation for parameter selection
   - Monitor overfitting with validation metrics
   - Consider regime count optimization

4. **Production Deployment**
   - Cache API calls to avoid rate limits
   - Use batch predictions for efficiency
   - Monitor model drift over time
   - Implement proper error handling