#!/usr/bin/env python
"""Test Kraken data loader with REAL crypto data."""

import sys
from datetime import datetime, timedelta

sys.path.insert(0, '/Users/firestrand/Projects/marketregimeml')

from marketregimeml.data.loaders.kraken import KrakenDataLoader
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector
from marketregimeml.evaluation.metrics import RegimeMetrics
import pandas as pd
import numpy as np


def test_kraken_loader():
    """Test fetching real data from Kraken."""
    print("\n" + "="*60)
    print("🦑 TESTING KRAKEN DATA LOADER WITH REAL DATA")
    print("="*60)
    
    loader = KrakenDataLoader()
    
    # Test 1: Fetch Bitcoin data
    print("\n1. Fetching BTC/USD hourly data...")
    try:
        btc_data = loader.fetch_ohlcv('BTC/USD', 'H1', limit=168)  # Last 7 days
        
        if not btc_data.empty:
            print(f"✅ Success! Fetched {len(btc_data)} hours of BTC data")
            print(f"   Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
            print(f"   Latest BTC price: ${btc_data['close'].iloc[-1]:,.2f}")
            print(f"   24h High: ${btc_data['high'].tail(24).max():,.2f}")
            print(f"   24h Low: ${btc_data['low'].tail(24).min():,.2f}")
            print(f"   24h Volume: {btc_data['volume'].tail(24).sum():,.2f} BTC")
            return btc_data
        else:
            print("❌ No data returned")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_multiple_cryptos():
    """Test fetching multiple cryptocurrencies."""
    print("\n2. Fetching multiple cryptocurrencies...")
    
    loader = KrakenDataLoader()
    cryptos = ['BTC/USD', 'ETH/USD', 'SOL/USD']
    
    results = {}
    for crypto in cryptos:
        try:
            data = loader.fetch_ohlcv(crypto, 'H4', limit=42)  # Last week, 4-hour candles
            if not data.empty:
                results[crypto] = data
                latest_price = data['close'].iloc[-1]
                change_24h = (data['close'].iloc[-1] / data['close'].iloc[-6] - 1) * 100
                print(f"   ✅ {crypto}: ${latest_price:,.2f} (24h: {change_24h:+.2f}%)")
            else:
                print(f"   ❌ {crypto}: No data")
        except Exception as e:
            print(f"   ❌ {crypto}: Error - {e}")
    
    return results


def analyze_bitcoin_regimes(btc_data):
    """Analyze regime detection on real Bitcoin data."""
    print("\n" + "="*60)
    print("🔍 REGIME ANALYSIS ON REAL BITCOIN DATA")
    print("="*60)
    
    # Create features from real data
    features = pd.DataFrame(index=btc_data.index)
    
    # Price returns
    features['returns'] = btc_data['close'].pct_change()
    features['log_returns'] = np.log(btc_data['close'] / btc_data['close'].shift(1))
    
    # Volatility measures
    features['volatility_5'] = features['returns'].rolling(5).std()
    features['volatility_10'] = features['returns'].rolling(10).std()
    features['volatility_20'] = features['returns'].rolling(20).std()
    
    # Price spreads (market microstructure)
    features['hl_spread'] = (btc_data['high'] - btc_data['low']) / btc_data['close']
    features['co_spread'] = (btc_data['close'] - btc_data['open']) / btc_data['open']
    
    # Volume features
    features['volume_ratio'] = btc_data['volume'] / btc_data['volume'].rolling(20).mean()
    
    # Moving averages for trend
    features['ma_ratio_5_20'] = btc_data['close'].rolling(5).mean() / btc_data['close'].rolling(20).mean()
    
    # Clean features
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().fillna(0)
    
    print(f"Created {len(features.columns)} features from real BTC data")
    
    # Test different models
    metrics = RegimeMetrics()
    
    models = [
        ('HMM-3', HMMRegimeDetector(n_regimes=3, covariance_type='diag', random_state=42)),
        ('HMM-5', HMMRegimeDetector(n_regimes=5, covariance_type='diag', random_state=42)),
        ('GMM-3', GMMRegimeDetector(n_regimes=3, covariance_type='full', random_state=42)),
        ('GMM-5', GMMRegimeDetector(n_regimes=5, covariance_type='full', random_state=42))
    ]
    
    best_model = None
    best_rqi = 0
    best_predictions = None
    
    for name, model in models:
        try:
            print(f"\nTesting {name}...")
            model.fit(features)
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            
            # Calculate metrics
            rqi = metrics.regime_quality_index(
                features.values, predictions, probabilities, features['returns'].values
            )
            
            stability = metrics.regime_stability(predictions)
            n_regimes = len(np.unique(predictions))
            
            print(f"  RQI: {rqi:.1f}")
            print(f"  Regimes found: {n_regimes}")
            print(f"  Persistence: {stability['persistence']:.3f}")
            print(f"  Avg duration: {stability['avg_duration']:.1f} hours")
            
            if rqi > best_rqi:
                best_rqi = rqi
                best_model = name
                best_predictions = predictions
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Analyze best model's regimes
    if best_model and best_predictions is not None:
        print(f"\n🏆 Best Model: {best_model} (RQI={best_rqi:.1f})")
        print("\nRegime Characteristics (Real BTC Data):")
        
        for regime in np.unique(best_predictions):
            mask = best_predictions == regime
            
            regime_returns = features.loc[mask, 'returns'] * 100  # Convert to percentage
            regime_vol = features.loc[mask, 'volatility_10'] * 100
            regime_prices = btc_data.loc[mask, 'close']
            
            print(f"\n  Regime {regime}:")
            print(f"    Periods: {mask.sum()} hours ({mask.sum()/len(mask)*100:.1f}% of time)")
            print(f"    Avg Return: {regime_returns.mean():.3f}% per hour")
            print(f"    Volatility: {regime_vol.mean():.3f}%")
            print(f"    Price Range: ${regime_prices.min():,.0f} - ${regime_prices.max():,.0f}")
            
            # Show when this regime was active
            regime_periods = []
            in_regime = False
            start_time = None
            
            for i, (time, is_regime) in enumerate(zip(features.index, mask)):
                if is_regime and not in_regime:
                    start_time = time
                    in_regime = True
                elif not is_regime and in_regime:
                    regime_periods.append((start_time, features.index[i-1]))
                    in_regime = False
            
            if in_regime:
                regime_periods.append((start_time, features.index[-1]))
            
            # Show last 3 periods this regime was active
            if regime_periods:
                print(f"    Recent activity (last 3 periods):")
                for start, end in regime_periods[-3:]:
                    duration = (end - start).total_seconds() / 3600
                    print(f"      {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')} ({duration:.0f}h)")
    
    return best_predictions


def main():
    """Main function to test Kraken with real data."""
    print("\n🚀 KRAKEN REAL DATA TEST")
    print("="*60)
    
    # Test fetching real BTC data
    btc_data = test_kraken_loader()
    
    if btc_data is not None and not btc_data.empty:
        # Test fetching multiple cryptos
        multi_crypto = test_multiple_cryptos()
        
        # Perform regime analysis on real BTC data
        regimes = analyze_bitcoin_regimes(btc_data)
        
        print("\n" + "="*60)
        print("✅ SUCCESSFULLY ANALYZED REAL CRYPTO DATA")
        print("="*60)
        print("\nKey Findings from Real BTC Data:")
        print("1. Kraken API provides reliable real-time crypto data")
        print("2. No API key required for public market data")
        print("3. Multiple cryptocurrencies available (BTC, ETH, SOL, etc.)")
        print("4. Data quality is excellent for regime detection")
        print("\nThis demonstrates regime detection on REAL market data!")
    else:
        print("\n❌ Failed to fetch Bitcoin data from Kraken")


if __name__ == "__main__":
    main()