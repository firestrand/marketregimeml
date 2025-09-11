#!/usr/bin/env python
"""Test all data sources with REAL data using .env credentials."""

import sys
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from marketregimeml.data.loaders import (
    OANDADataLoader,
    AlphaVantageLoader,
    KrakenDataLoader,
)
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.evaluation.metrics import RegimeMetrics
import pandas as pd
import numpy as np


def test_oanda_real_data():
    """Test OANDA with real forex data."""
    print("\n" + "="*60)
    print("💱 TESTING OANDA WITH REAL FOREX DATA")
    print("="*60)
    
    # Skip if loader not available
    if OANDADataLoader is None:
        print("❌ OANDA loader not available (missing dependency)")
        return None

    # Check if credentials are loaded
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    
    print(f"API Key found: {'Yes' if api_key else 'No'}")
    print(f"Account ID found: {'Yes' if account_id else 'No'}")
    
    if not api_key or not account_id:
        print("❌ Credentials not found in .env")
        return None
    
    try:
        # Initialize loader (it will use env variables internally)
        loader = OANDADataLoader()
        
        # Fetch EUR/USD data for last 24 hours
        end = datetime.now()
        start = end - timedelta(hours=48)
        
        print(f"\nFetching EUR/USD from {start} to {end}...")
        df = loader.fetch_ohlcv('EUR_USD', 'H1', start_date=start, end_date=end)
        
        if not df.empty:
            print(f"✅ Success! Fetched {len(df)} hours of EUR/USD data")
            print(f"   Latest EUR/USD: {df['close'].iloc[-1]:.5f}")
            print(f"   24h High: {df['high'].tail(24).max():.5f}")
            print(f"   24h Low: {df['low'].tail(24).min():.5f}")
            print(f"   Avg Volume: {df['volume'].mean():.0f}")
            return df
        else:
            print("❌ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_alphavantage_real_data():
    """Test Alpha Vantage with real stock data."""
    print("\n" + "="*60)
    print("📈 TESTING ALPHA VANTAGE WITH REAL STOCK DATA")
    print("="*60)
    
    # Skip if loader not available
    if AlphaVantageLoader is None:
        print("❌ Alpha Vantage loader not available (missing dependency)")
        return None

    # Check if API key is loaded
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    
    print(f"API Key found: {'Yes' if api_key else 'No'}")
    
    if not api_key:
        print("❌ API key not found in .env")
        return None
    
    try:
        # Initialize loader (it will use env variable internally)
        loader = AlphaVantageLoader()
        
        print("\nFetching SPY daily data...")
        end = datetime.now()
        start = end - timedelta(days=30)
        df = loader.fetch_ohlcv('SPY', 'D', start_date=start, end_date=end)  # Last 30 days
        
        if not df.empty:
            print(f"✅ Success! Fetched {len(df)} days of SPY data")
            print(f"   Latest SPY close: ${df['close'].iloc[-1]:.2f}")
            print(f"   30-day High: ${df['high'].max():.2f}")
            print(f"   30-day Low: ${df['low'].min():.2f}")
            print(f"   Avg Volume: {df['volume'].mean():,.0f}")
            
            # Calculate simple metrics
            returns = df['close'].pct_change()
            print(f"   30-day Return: {(df['close'].iloc[-1]/df['close'].iloc[0] - 1)*100:.2f}%")
            print(f"   Daily Volatility: {returns.std()*100:.2f}%")
            return df
        else:
            print("❌ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Note: Alpha Vantage has strict rate limits (5/min, 500/day)")
        return None


def test_kraken_real_data():
    """Test Kraken with real crypto data."""
    print("\n" + "="*60)
    print("🪙 TESTING KRAKEN WITH REAL CRYPTO DATA")
    print("="*60)
    
    try:
        loader = KrakenDataLoader()
        
        print("\nFetching BTC/USD hourly data...")
        df = loader.fetch_ohlcv('BTC/USD', 'H1', limit=48)  # Last 48 hours
        
        if not df.empty:
            print(f"✅ Success! Fetched {len(df)} hours of BTC data")
            print(f"   Latest BTC price: ${df['close'].iloc[-1]:,.2f}")
            print(f"   48h High: ${df['high'].max():,.2f}")
            print(f"   48h Low: ${df['low'].min():,.2f}")
            print(f"   Total Volume: {df['volume'].sum():.2f} BTC")
            return df
        else:
            print("❌ No data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def quick_regime_analysis(data, asset_name):
    """Quick regime analysis on real data."""
    if data is None or data.empty:
        return
    
    print(f"\n🔍 Quick Regime Analysis for {asset_name}")
    print("-" * 40)
    
    # Simple features
    features = pd.DataFrame(index=data.index)
    features['returns'] = data['close'].pct_change()
    features['volatility'] = features['returns'].rolling(5, min_periods=1).std()
    features['volume_norm'] = data['volume'] / data['volume'].rolling(10, min_periods=1).mean()
    features = features.fillna(0)
    
    # Quick HMM fit
    model = HMMRegimeDetector(n_regimes=3, covariance_type='diag', random_state=42)
    model.fit(features)
    predictions = model.predict(features)
    
    # Show regime distribution
    unique, counts = np.unique(predictions, return_counts=True)
    for regime, count in zip(unique, counts):
        pct = count / len(predictions) * 100
        mask = predictions == regime
        avg_return = features.loc[mask, 'returns'].mean() * 100
        print(f"   Regime {regime}: {pct:.1f}% of time, Avg return: {avg_return:.3f}%")


def main():
    """Test all data sources with real data."""
    print("\n" + "🚀 "*20)
    print("TESTING ALL DATA SOURCES WITH REAL DATA")
    print("🚀 "*20)
    
    # Test each data source
    forex_data = test_oanda_real_data()
    stock_data = test_alphavantage_real_data()
    crypto_data = test_kraken_real_data()
    
    # Quick analysis on successful data
    if forex_data is not None:
        quick_regime_analysis(forex_data, "EUR/USD")
    
    if stock_data is not None:
        quick_regime_analysis(stock_data, "SPY")
    
    if crypto_data is not None:
        quick_regime_analysis(crypto_data, "BTC/USD")
    
    # Summary
    print("\n" + "="*60)
    print("📊 DATA SOURCE STATUS SUMMARY")
    print("="*60)
    
    status = []
    if forex_data is not None:
        status.append("✅ OANDA (Forex): Working - EUR/USD data retrieved")
    else:
        status.append("❌ OANDA (Forex): Failed")
    
    if stock_data is not None:
        status.append("✅ Alpha Vantage (Stocks): Working - SPY data retrieved")
    else:
        status.append("❌ Alpha Vantage (Stocks): Failed")
    
    if crypto_data is not None:
        status.append("✅ Kraken (Crypto): Working - BTC data retrieved")
    else:
        status.append("❌ Kraken (Crypto): Failed")
    
    for s in status:
        print(s)
    
    # Count working sources
    working = sum(1 for s in status if s.startswith("✅"))
    print(f"\n🎯 {working}/3 data sources working with REAL data")
    
    if working == 3:
        print("\n🎉 ALL DATA SOURCES WORKING! Ready for production use.")
    elif working > 0:
        print(f"\n⚠️ {working} data source(s) working. Check credentials for failed sources.")
    else:
        print("\n❌ No data sources working. Please check .env file and credentials.")


if __name__ == "__main__":
    main()
