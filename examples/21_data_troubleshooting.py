#!/usr/bin/env python
"""Test real data sources to identify and fix issues."""

import sys
import os
import requests
from datetime import datetime, timedelta

sys.path.insert(0, '/Users/firestrand/Projects/marketregimeml')

from marketregimeml.data.loaders import OANDADataLoader, AlphaVantageLoader


def test_binance_direct():
    """Test Binance API directly to debug the 451 error."""
    print("\n" + "="*60)
    print("TESTING BINANCE API DIRECTLY")
    print("="*60)
    
    # Test different Binance endpoints
    endpoints = [
        ("Binance.com", "https://api.binance.com/api/v3/ping"),
        ("Binance.com Time", "https://api.binance.com/api/v3/time"),
        ("Binance.com Exchange Info", "https://api.binance.com/api/v3/exchangeInfo?symbol=BTCUSDT"),
        ("Binance.us", "https://api.binance.us/api/v3/ping"),
        ("Binance.us Klines", "https://api.binance.us/api/v3/klines?symbol=BTCUSD&interval=1h&limit=1"),
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    for name, url in endpoints:
        try:
            print(f"\nTesting {name}...")
            print(f"URL: {url}")
            response = session.get(url, timeout=5)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print("✅ Success!")
                if "klines" in url.lower():
                    data = response.json()
                    if data:
                        print(f"Sample data: {data[0][:5]}...")  # First 5 fields
            else:
                print(f"❌ Failed: {response.status_code} - {response.text[:100]}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Try alternative crypto APIs
    print("\n" + "="*60)
    print("TESTING ALTERNATIVE CRYPTO DATA SOURCES")
    print("="*60)
    
    alternatives = [
        ("CoinGecko", "https://api.coingecko.com/api/v3/ping"),
        ("CoinGecko OHLC", "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1"),
        ("Kraken", "https://api.kraken.com/0/public/Time"),
        ("Kraken OHLC", "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=60"),
    ]
    
    for name, url in alternatives:
        try:
            print(f"\nTesting {name}...")
            print(f"URL: {url}")
            response = session.get(url, timeout=5)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print("✅ Success!")
                data = response.json()
                if "ohlc" in url.lower():
                    if "result" in data:  # Kraken format
                        for key in data["result"]:
                            if key != "last":
                                print(f"Sample OHLC: {data['result'][key][0]}")
                                break
                    elif isinstance(data, list) and data:  # CoinGecko format
                        print(f"Sample OHLC: {data[0]}")
        except Exception as e:
            print(f"❌ Error: {e}")


def test_oanda():
    """Test OANDA data loader."""
    print("\n" + "="*60)
    print("TESTING OANDA DATA LOADER")
    print("="*60)
    
    if OANDADataLoader is None:
        print("❌ OANDA loader not available (missing dependency)")
        return False

    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    
    if not api_key or not account_id:
        print("❌ OANDA credentials not found in environment")
        print("   Set these environment variables:")
        print("   export OANDA_API_KEY='your-api-key'")
        print("   export OANDA_ACCOUNT_ID='your-account-id'")
        print("\n   To get free OANDA API access:")
        print("   1. Sign up at https://www.oanda.com/")
        print("   2. Create a practice account")
        print("   3. Go to 'Manage API Access' to get your API key")
        return False
    
    try:
        loader = OANDADataLoader(api_key=api_key, account_id=account_id)
        
        # Test fetching recent EUR/USD data
        end = datetime.now()
        start = end - timedelta(hours=24)
        
        print(f"Fetching EUR_USD from {start} to {end}...")
        df = loader.fetch_ohlcv('EUR_USD', 'H1', start=start, end=end)
        
        if not df.empty:
            print(f"✅ Success! Fetched {len(df)} candles")
            print(f"Latest EUR/USD: {df['close'].iloc[-1]:.5f}")
            print(f"Data columns: {df.columns.tolist()}")
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_alphavantage():
    """Test Alpha Vantage data loader."""
    print("\n" + "="*60)
    print("TESTING ALPHA VANTAGE DATA LOADER")
    print("="*60)
    
    if AlphaVantageLoader is None:
        print("❌ Alpha Vantage loader not available (missing dependency)")
        return False

    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    
    if not api_key:
        print("❌ Alpha Vantage API key not found in environment")
        print("   Set this environment variable:")
        print("   export ALPHAVANTAGE_API_KEY='your-api-key'")
        print("\n   To get free Alpha Vantage API key:")
        print("   1. Visit https://www.alphavantage.co/support/#api-key")
        print("   2. Enter your email to get a free API key instantly")
        return False
    
    try:
        loader = AlphaVantageLoader(api_key=api_key)
        
        print("Fetching SPY daily data...")
        df = loader.fetch_ohlcv('SPY', 'D', limit=10)
        
        if not df.empty:
            print(f"✅ Success! Fetched {len(df)} days")
            print(f"Latest SPY close: ${df['close'].iloc[-1]:.2f}")
            print(f"Data columns: {df.columns.tolist()}")
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n   Note: Alpha Vantage has rate limits:")
        print("   - 5 API calls per minute")
        print("   - 500 API calls per day")
        return False


def main():
    """Run all tests."""
    print("\n🔍 TESTING REAL DATA SOURCES\n")
    
    # Test Binance and alternatives
    test_binance_direct()
    
    # Test OANDA
    oanda_works = test_oanda()
    
    # Test Alpha Vantage
    av_works = test_alphavantage()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not oanda_works:
        print("\n📌 OANDA: Need to set up credentials")
    else:
        print("\n✅ OANDA: Working")
    
    if not av_works:
        print("📌 Alpha Vantage: Need to set up API key")
    else:
        print("✅ Alpha Vantage: Working")
    
    print("\n📌 For crypto data:")
    print("   - Binance may be geo-restricted (451 error)")
    print("   - Try Binance.us if in the United States")
    print("   - Alternative: Use Kraken or CoinGecko APIs")
    
    print("\n💡 Next steps:")
    print("1. Set up missing API credentials")
    print("2. Implement Kraken or CoinGecko loader for crypto")
    print("3. Use working data sources for regime detection")


if __name__ == "__main__":
    main()
