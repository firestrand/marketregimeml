"""Unit tests for data loaders.

Following DRY, KISS, and SOLID principles with bottom-up testing.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime, timedelta
import requests

from marketregimeml.data.loaders.base import MarketDataLoader
from marketregimeml.data.loaders.alphavantage import AlphaVantageLoader
from marketregimeml.data.loaders.oanda import OANDADataLoader


class TestDataHelper:
    """DRY: Centralized test data generation for loaders."""
    
    @staticmethod
    def create_ohlcv_data(n_periods=100):
        """Create sample OHLCV data."""
        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
        
        # Generate realistic price data
        np.random.seed(42)
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n_periods)))
        
        # Generate OHLC from close
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0]
        
        # Generate volume
        volume = np.random.uniform(1e6, 1e7, n_periods)
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return df
    
    @staticmethod
    def create_api_response(data_format='alphavantage'):
        """Create mock API response."""
        if data_format == 'alphavantage':
            return {
                'Time Series (Daily)': {
                    '2024-01-10': {
                        '1. open': '100.0',
                        '2. high': '105.0',
                        '3. low': '99.0',
                        '4. close': '103.0',
                        '5. volume': '1000000'
                    },
                    '2024-01-09': {
                        '1. open': '98.0',
                        '2. high': '102.0',
                        '3. low': '97.0',
                        '4. close': '100.0',
                        '5. volume': '900000'
                    }
                }
            }
        elif data_format == 'oanda':
            return {
                'candles': [
                    {
                        'time': '2024-01-10T00:00:00Z',
                        'mid': {
                            'o': '100.0',
                            'h': '105.0',
                            'l': '99.0',
                            'c': '103.0'
                        },
                        'volume': 1000
                    },
                    {
                        'time': '2024-01-09T00:00:00Z',
                        'mid': {
                            'o': '98.0',
                            'h': '102.0',
                            'l': '97.0',
                            'c': '100.0'
                        },
                        'volume': 900
                    }
                ]
            }


class ConcreteDataLoader(MarketDataLoader):
    """Concrete implementation for testing abstract base class."""
    
    def fetch_ohlcv(self, symbol, timeframe='1d', limit=100):
        """Simple implementation returning test data."""
        return TestDataHelper.create_ohlcv_data(limit)
    
    def _fetch_ohlcv_impl(self, symbol, timeframe, limit):
        """Internal implementation."""
        return TestDataHelper.create_ohlcv_data(limit)
    
    def _validate_config(self):
        """Validate configuration."""
        return True
    
    def get_available_symbols(self):
        """Get available symbols."""
        return ['TEST', 'TEST1', 'TEST2', 'TEST3']
    
    def get_available_timeframes(self):
        """Get available timeframes."""
        return ['1m', '5m', '1h', '1d', '1w']


class TestMarketDataLoader:
    """Test base MarketDataLoader functionality."""
    
    @pytest.fixture
    def loader(self):
        """Create concrete loader instance."""
        return ConcreteDataLoader()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        return TestDataHelper.create_ohlcv_data()
    
    def test_initialization(self, loader):
        """Test loader initialization."""
        assert loader is not None
        assert hasattr(loader, 'fetch_ohlcv')
    
    def test_fetch_ohlcv(self, loader):
        """Test fetching OHLCV data."""
        data = loader.fetch_ohlcv('TEST', '1d', 50)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_validate_ohlcv(self, loader, sample_data):
        """Test OHLCV validation."""
        # Valid data should pass
        is_valid = loader.validate_ohlcv(sample_data)
        assert is_valid
        
        # Invalid data: high < low
        invalid_data = sample_data.copy()
        invalid_data.loc[invalid_data.index[0], 'high'] = 50
        invalid_data.loc[invalid_data.index[0], 'low'] = 100
        
        is_valid = loader.validate_ohlcv(invalid_data)
        assert not is_valid
    
    def test_validate_missing_columns(self, loader):
        """Test validation with missing columns."""
        incomplete_data = pd.DataFrame({
            'open': [100, 101],
            'close': [101, 102]
        })
        
        is_valid = loader.validate_ohlcv(incomplete_data)
        assert not is_valid
    
    def test_validate_negative_prices(self, loader, sample_data):
        """Test validation with negative prices."""
        invalid_data = sample_data.copy()
        invalid_data.loc[invalid_data.index[0], 'close'] = -10
        
        is_valid = loader.validate_ohlcv(invalid_data)
        assert not is_valid
    
    def test_standardize_ohlcv(self, loader, sample_data):
        """Test OHLCV standardization."""
        # Add extra columns
        sample_data['extra'] = 1
        
        standardized = loader.standardize_ohlcv(sample_data)
        
        assert 'extra' not in standardized.columns
        assert all(col in standardized.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert standardized.index.name == 'date' or isinstance(standardized.index, pd.DatetimeIndex)
    
    def test_resample_ohlcv(self, loader):
        """Test OHLCV resampling."""
        # Create hourly data
        hourly_data = TestDataHelper.create_ohlcv_data(24*7)  # 1 week hourly
        hourly_data.index = pd.date_range(end=datetime.now(), periods=24*7, freq='H')
        
        # Resample to daily
        daily_data = loader.resample_ohlcv(hourly_data, 'D')
        
        assert len(daily_data) <= 7  # Should have at most 7 days
        assert daily_data['volume'].sum() <= hourly_data['volume'].sum() * 1.01  # Allow small floating point error
    
    def test_fetch_multiple(self, loader):
        """Test fetching multiple symbols."""
        symbols = ['TEST1', 'TEST2', 'TEST3']
        
        data = loader.fetch_multiple(symbols, '1d', 10)
        
        assert isinstance(data, dict)
        assert len(data) == 3
        assert all(symbol in data for symbol in symbols)
        assert all(isinstance(df, pd.DataFrame) for df in data.values())


class TestAlphaVantageLoader:
    """Test Alpha Vantage data loader."""
    
    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        with patch.dict('os.environ', {'ALPHAVANTAGE_API_KEY': 'test_key'}):
            return AlphaVantageLoader()
    
    @patch('requests.get')
    def test_fetch_ohlcv_success(self, mock_get, loader):
        """Test successful data fetch."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = TestDataHelper.create_api_response('alphavantage')
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        data = loader.fetch_ohlcv('AAPL', 'daily', 2)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_fetch_ohlcv_api_error(self, mock_get, loader):
        """Test API error handling."""
        mock_get.side_effect = requests.RequestException("API Error")
        
        with pytest.raises(Exception):
            loader.fetch_ohlcv('AAPL', 'daily', 100)
    
    @patch('requests.get')
    def test_rate_limiting(self, mock_get, loader):
        """Test rate limiting."""
        mock_response = Mock()
        mock_response.json.return_value = {'Note': 'API call frequency limit'}
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception):
            loader.fetch_ohlcv('AAPL', 'daily', 100)
    
    def test_invalid_api_key(self):
        """Test invalid API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError):
                AlphaVantageLoader()


class TestOANDADataLoader:
    """Test OANDA data loader."""
    
    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        with patch.dict('os.environ', {
            'OANDA_API_KEY': 'test_key',
            'OANDA_ACCOUNT_ID': 'test_account'
        }):
            return OANDADataLoader(environment='practice')
    
    @patch('requests.get')
    def test_fetch_ohlcv_success(self, mock_get, loader):
        """Test successful data fetch."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = TestDataHelper.create_api_response('oanda')
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        data = loader.fetch_ohlcv('EUR_USD', 'D', 2)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_granularity_conversion(self, loader):
        """Test timeframe to granularity conversion."""
        # Test various timeframe conversions
        assert loader._get_granularity('1m') == 'M1'
        assert loader._get_granularity('5m') == 'M5'
        assert loader._get_granularity('1h') == 'H1'
        assert loader._get_granularity('1d') == 'D'
        assert loader._get_granularity('1w') == 'W'
    
    @patch('requests.get')
    def test_fetch_multiple_pairs(self, mock_get, loader):
        """Test fetching multiple currency pairs."""
        mock_response = Mock()
        mock_response.json.return_value = TestDataHelper.create_api_response('oanda')
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
        data = loader.fetch_multiple(pairs, 'D', 2)
        
        assert isinstance(data, dict)
        assert len(data) == 3
        assert all(pair in data for pair in pairs)
    
    def test_environment_url(self):
        """Test environment URL selection."""
        with patch.dict('os.environ', {
            'OANDA_API_KEY': 'key',
            'OANDA_ACCOUNT_ID': 'acc'
        }):
            # Practice environment
            loader_practice = OANDADataLoader(environment='practice')
            assert 'practice' in loader_practice.base_url
            
            # Live environment
            loader_live = OANDADataLoader(environment='live')
            assert 'trade' in loader_live.base_url


class TestDataLoaderIntegration:
    """Integration tests for data loaders."""
    
    def test_loader_compatibility(self):
        """Test that different loaders return compatible data."""
        # Create mock loaders
        loader1 = ConcreteDataLoader()
        
        # Get data from loader
        data1 = loader1.fetch_ohlcv('TEST', '1d', 50)
        
        # Validate data structure
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        assert all(col in data1.columns for col in required_columns)
        assert isinstance(data1.index, pd.DatetimeIndex)
        
        # Validate data
        assert loader1.validate_ohlcv(data1)
    
    def test_data_concatenation(self):
        """Test concatenating data from multiple timeframes."""
        loader = ConcreteDataLoader()
        
        # Get data for different periods
        data1 = loader.fetch_ohlcv('TEST', '1d', 30)
        data2 = loader.fetch_ohlcv('TEST', '1d', 30)
        
        # Shift data2 index to avoid overlap
        data2.index = data2.index + timedelta(days=30)
        
        # Concatenate
        combined = pd.concat([data1, data2])
        
        assert len(combined) == 60
        assert loader.validate_ohlcv(combined)


class TestDataLoaderEdgeCases:
    """Test edge cases for data loaders."""
    
    def test_empty_symbol_list(self):
        """Test with empty symbol list."""
        loader = ConcreteDataLoader()
        
        data = loader.fetch_multiple([], '1d', 100)
        
        assert isinstance(data, dict)
        assert len(data) == 0
    
    def test_invalid_timeframe(self):
        """Test with invalid timeframe."""
        loader = ConcreteDataLoader()
        
        # Should handle gracefully
        data = loader.fetch_ohlcv('TEST', 'invalid', 100)
        assert isinstance(data, pd.DataFrame)
    
    def test_zero_limit(self):
        """Test with zero limit."""
        loader = ConcreteDataLoader()
        
        data = loader.fetch_ohlcv('TEST', '1d', 0)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 0
    
    def test_very_large_limit(self):
        """Test with very large limit."""
        loader = ConcreteDataLoader()
        
        # Should handle gracefully
        data = loader.fetch_ohlcv('TEST', '1d', 1000000)
        
        assert isinstance(data, pd.DataFrame)
        # Implementation may cap at reasonable limit
        assert len(data) <= 1000000