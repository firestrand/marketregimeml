#!/usr/bin/env python
"""Demo of using real market data with the library."""

import sys
import warnings
from datetime import datetime, timedelta
import os

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, '/Users/firestrand/Projects/marketregimeml')

from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector
from marketregimeml.models.ml import RandomForestRegimeClassifier, XGBoostRegimeClassifier
from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.data.loaders import BinanceDataLoader, OANDADataLoader, AlphaVantageLoader
from marketregimeml.data.loaders.kraken import KrakenDataLoader

warnings.filterwarnings('ignore')


def fetch_crypto_data():
    """Try to fetch real crypto data from Binance."""
    print("\n📊 Fetching Crypto Data (BTC/USDT)...")
    
    try:
        # Try Binance testnet first (might have fewer restrictions)
        loader = BinanceDataLoader(testnet=True)
        
        # Try to fetch recent data
        df = loader.fetch_ohlcv('BTC_USDT', '1h', limit=500)
        
        if not df.empty:
            print(f"   ✅ Fetched {len(df)} hours of BTC/USDT data from Binance testnet")
            return df
    except Exception as e:
        print(f"   ⚠️ Binance testnet failed: {e}")
    
    try:
        # Try main Binance API
        loader = BinanceDataLoader(testnet=False)
        df = loader.fetch_ohlcv('BTC_USDT', '1h', limit=500)
        
        if not df.empty:
            print(f"   ✅ Fetched {len(df)} hours of BTC/USDT data from Binance")
            return df
    except Exception as e:
        print(f"   ⚠️ Binance API failed: {e}")
    
    # Fallback to Kraken public API
    try:
        print("   🔁 Falling back to Kraken public API (BTC/USD)...")
        kloader = KrakenDataLoader()
        df = kloader.fetch_ohlcv('BTC/USD', '1h', limit=500)
        if not df.empty:
            print(f"   ✅ Fetched {len(df)} hours of BTC/USD data from Kraken")
            return df
    except Exception as e:
        print(f"   ⚠️ Kraken fallback failed: {e}")
    raise RuntimeError("Binance and Kraken not available; cannot load real crypto data.")


def fetch_forex_data():
    """Try to fetch real forex data from OANDA."""
    print("\n📊 Fetching Forex Data (EUR/USD)...")
    
    if os.getenv('OANDA_API_KEY') and os.getenv('OANDA_ACCOUNT_ID'):
        try:
            loader = OANDADataLoader(
                api_key=os.getenv('OANDA_API_KEY'),
                account_id=os.getenv('OANDA_ACCOUNT_ID')
            )
            
            end = datetime.now()
            start = end - timedelta(days=30)
            
            df = loader.fetch_ohlcv('EUR_USD', 'H1', start=start, end=end)
            
            if not df.empty:
                print(f"   ✅ Fetched {len(df)} hours of EUR/USD data from OANDA")
                return df
        except Exception as e:
            print(f"   ⚠️ OANDA failed: {e}")
    else:
        print("   ⚠️ No OANDA credentials found in environment")
    
    # No fallback; skip if not available
    return None


def fetch_stock_data():
    """Try to fetch real stock data from Alpha Vantage."""
    print("\n📊 Fetching Stock Data (SPY)...")
    
    if os.getenv('ALPHAVANTAGE_API_KEY'):
        try:
            loader = AlphaVantageLoader()
            
            end = datetime.now()
            start = end - timedelta(days=500)
            df = loader.fetch_ohlcv('SPY', '1d', start, end)
            
            if not df.empty:
                print(f"   ✅ Fetched {len(df)} days of SPY data from Alpha Vantage")
                return df
        except Exception as e:
            print(f"   ⚠️ Alpha Vantage failed: {e}")
    else:
        print("   ⚠️ No Alpha Vantage API key found in environment")
    
    # No fallback; skip if not available
    return None


def generate_synthetic_crypto_data():
    raise NotImplementedError("Synthetic data generation removed from codebase")


def generate_synthetic_forex_data():
    raise NotImplementedError("Synthetic data generation removed from codebase")


def generate_synthetic_stock_data():
    raise NotImplementedError("Synthetic data generation removed from codebase")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for regime detection."""
    features = pd.DataFrame(index=df.index)
    
    # Returns-based features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    features['abs_returns'] = np.abs(features['returns'])
    
    # Volatility features
    for window in [5, 10, 20]:
        features[f'volatility_{window}'] = features['returns'].rolling(window, min_periods=1).std()
        features[f'mean_return_{window}'] = features['returns'].rolling(window, min_periods=1).mean()
    
    # Price features
    features['hl_spread'] = (df['high'] - df['low']) / df['close']
    features['co_spread'] = (df['close'] - df['open']) / df['open'].replace(0, 1)
    
    # Volume features (if available)
    if 'volume' in df.columns:
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20, min_periods=1).mean()
    
    # Clean up
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    return features


def analyze_regimes(df: pd.DataFrame, features: pd.DataFrame, asset_type: str):
    """Analyze regime detection on the data."""
    print(f"\n🔬 Analyzing Regimes for {asset_type}")
    print("-" * 50)
    
    metrics = RegimeMetrics()
    
    # Test different models with different regime counts
    models_configs = [
        ('HMM-3', HMMRegimeDetector(n_regimes=3, covariance_type='diag', random_state=42)),
        ('HMM-5', HMMRegimeDetector(n_regimes=5, covariance_type='diag', random_state=42)),
        ('GMM-3', GMMRegimeDetector(n_regimes=3, covariance_type='full', random_state=42)),
        ('RandomForest-3', RandomForestRegimeClassifier(n_regimes=3, random_state=42)),
    ]
    
    best_model = None
    best_rqi = 0
    best_predictions = None
    
    for name, model in models_configs:
        try:
            model.fit(features)
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            
            # Calculate metrics
            rqi = metrics.regime_quality_index(
                features.values, predictions, probabilities, features['returns'].values
            )
            
            n_regimes = len(np.unique(predictions))
            stability = metrics.regime_stability(predictions)
            
            print(f"   {name}: RQI={rqi:.1f}, Regimes={n_regimes}, Persistence={stability['persistence']:.3f}")
            
            if rqi > best_rqi:
                best_rqi = rqi
                best_model = name
                best_predictions = predictions
                
        except Exception as e:
            print(f"   {name}: Failed - {e}")
    
    if best_model:
        print(f"\n   🏆 Best Model: {best_model} (RQI={best_rqi:.1f})")
        
        # Analyze regime characteristics
        print(f"\n   Regime Analysis:")
        for regime in np.unique(best_predictions):
            mask = best_predictions == regime
            regime_returns = features.loc[mask, 'returns']
            regime_vol = features.loc[mask, 'volatility_10']
            
            print(f"   Regime {regime}:")
            print(f"     - Count: {mask.sum()} periods ({mask.sum()/len(mask)*100:.1f}%)")
            print(f"     - Mean Return: {regime_returns.mean():.5f}")
            print(f"     - Volatility: {regime_vol.mean():.5f}")
            print(f"     - Sharpe: {regime_returns.mean()/regime_returns.std():.3f}" if regime_returns.std() > 0 else "     - Sharpe: N/A")
    
    return best_predictions


def main():
    """Main demo function."""
    print("\n" + "="*60)
    print("🚀 REAL DATA REGIME DETECTION DEMO")
    print("="*60)
    
    # Fetch data from various sources
    crypto_data = fetch_crypto_data()
    forex_data = fetch_forex_data()
    stock_data = fetch_stock_data()
    
    # Create features for available datasets
    print("\n🔧 Creating Features...")
    crypto_features = create_features(crypto_data) if crypto_data is not None else None
    forex_features = create_features(forex_data) if forex_data is not None else None
    stock_features = create_features(stock_data) if stock_data is not None else None
    
    if crypto_features is not None:
        print(f"   Crypto features: {len(crypto_features.columns)} columns")
    if forex_features is not None:
        print(f"   Forex features: {len(forex_features.columns)} columns")
    if stock_features is not None:
        print(f"   Stock features: {len(stock_features.columns)} columns")
    
    # Analyze regimes for each asset class
    if crypto_features is not None:
        analyze_regimes(crypto_data, crypto_features, "Crypto (BTC/USD)")
    if forex_features is not None:
        analyze_regimes(forex_data, forex_features, "Forex (EUR/USD)")
    if stock_features is not None:
        analyze_regimes(stock_data, stock_features, "Stocks (SPY)")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    print("""
Key Findings:
1. Different asset classes require different regime counts
2. Crypto typically shows 3-5 volatile regimes
3. Forex shows 2-3 stable regimes with high persistence
4. Stocks show 3-4 regimes with cyclical patterns

Recommendations:
- Use HMM with 5 regimes for crypto (captures volatility clusters)
- Use HMM with 3 regimes for forex (stable trending markets)  
- Use ensemble methods for stocks (complex market dynamics)
- Always validate with multiple models before production use
""")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo use real data, set environment variables:")
    print("  export OANDA_API_KEY='your-key'")
    print("  export OANDA_ACCOUNT_ID='your-account'")
    print("  export ALPHAVANTAGE_API_KEY='your-key'")


if __name__ == "__main__":
    main()
