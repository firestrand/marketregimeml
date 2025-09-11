#!/usr/bin/env python3
"""
Feature Preparation for Regime Detection
=========================================
Basic feature engineering for market regime detection.

This example demonstrates:
- Creating price-based features (returns, volatility)
- Adding technical indicators
- Computing statistical features
- Preparing features for regime models

Requirements:
- marketregimeml package
- pandas
- ALPHAVANTAGE_API_KEY set in environment
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from marketregimeml.data.loaders import KrakenDataLoader


def load_market_data(symbol: str = "BTC/USD", timeframe: str = "1d", lookback_days: int = 800) -> pd.DataFrame:
    """Load OHLCV data from Kraken (BTC/USD)."""
    loader = KrakenDataLoader()
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    return loader.fetch_ohlcv(symbol, timeframe, start, end)


def prepare_basic_features(data):
    """Prepare basic price-based features."""
    features = pd.DataFrame(index=data.index)
    
    # Price returns
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # Price ratios
    features['high_low_ratio'] = data['high'] / data['low'] - 1
    features['close_open_ratio'] = data['close'] / data['open'] - 1
    
    # Volume features
    features['volume_change'] = data['volume'].pct_change()
    features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    
    # Rolling statistics
    features['returns_std_20'] = features['returns'].rolling(20).std()
    features['returns_mean_20'] = features['returns'].rolling(20).mean()
    features['returns_skew_20'] = features['returns'].rolling(20).skew()
    features['returns_kurt_20'] = features['returns'].rolling(20).kurt()
    
    return features


def add_technical_indicators(data):
    """Add simple technical indicators without extra deps."""
    import numpy as np
    features = pd.DataFrame(index=data.index)

    # RSI
    close = data['close']
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features['rsi'] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_histogram'] = macd - signal

    # Bollinger Bands (20, 2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    features['bb_upper'] = upper
    features['bb_lower'] = lower
    features['bb_position'] = (close - lower) / (upper - lower)

    # OBV
    direction = close.diff().fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    features['obv'] = (direction * data['volume']).cumsum()

    return features


# Removed advanced volatility/statistical features to avoid extra deps


def combine_all_features(data):
    """Combine all feature types into a single DataFrame."""
    print("Preparing basic features...")
    basic = prepare_basic_features(data)
    
    print("Adding technical indicators...")
    technical = add_technical_indicators(data)
    
    # Combine features
    all_features = pd.concat([basic, technical], axis=1)
    
    # Remove rows with NaN values
    all_features = all_features.dropna()
    
    return all_features


def main():
    """Main demonstration."""
    print("=" * 60)
    print("Feature Preparation for Regime Detection")
    print("=" * 60)
    
    # Load OHLCV data
    print("\n1. Loading OHLCV data (Kraken BTC/USD)...")
    try:
        data = load_market_data("BTC/USD", "1d", 900)
    except Exception:
        print("   ✗ Could not load data from Kraken.")
        raise
    print(f"   Loaded {len(data)} periods of data")
    
    # Prepare features
    print("\n2. Preparing features...")
    features = combine_all_features(data)
    
    print(f"\n3. Feature summary:")
    print(f"   Total features: {len(features.columns)}")
    print(f"   Samples after cleaning: {len(features)}")
    
    print("\n4. Feature categories:")
    
    # Group features by type
    basic_cols = [c for c in features.columns if 'returns' in c or 'ratio' in c or 'volume' in c]
    tech_cols = [c for c in features.columns if any(x in c for x in ['rsi', 'macd', 'bb_', 'stochastic', 'obv'])]
    vol_cols = [c for c in features.columns if 'vol' in c and c not in basic_cols]
    stat_cols = []
    
    print(f"   Basic features: {len(basic_cols)}")
    print(f"   Technical indicators: {len(tech_cols)}")
    print(f"   Volatility measures: {len(vol_cols)}")
    print(f"   Statistical features: {len(stat_cols)}")
    
    print("\n5. Feature statistics:")
    print(features.describe().round(3))
    
    print("\n6. Correlation matrix (top correlations):")
    corr = features.corr(numeric_only=True).abs()
    # Get top correlations (excluding diagonal)
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    print("\n   Highest correlations:")
    for feat1, feat2, corr_val in corr_pairs[:5]:
        print(f"   {feat1[:20]:20} <-> {feat2[:20]:20}: {corr_val:.3f}")
    
    print("\n7. Ready for regime detection!")
    print("   These features can now be used with any regime detection model.")
    
    # Example of feature selection
    print("\n8. Example: Selecting key features for regime detection")
    key_features = ['returns', 'returns_std_20', 'rsi', 'macd_histogram', 'bb_position']
    
    selected = features[key_features].dropna()
    print(f"   Selected {len(key_features)} key features")
    print(f"   Final dataset shape: {selected.shape}")
    
    return features


if __name__ == "__main__":
    features = main()
    print("\n" + "=" * 60)
    print("Feature preparation complete!")
    print("=" * 60)
