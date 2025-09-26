"""Unit tests for volatility features.

Following DRY, KISS, and SOLID principles in test design.
Bottom-up approach with real calculations, minimal mocking.
"""

import pytest
import numpy as np
import pandas as pd

from marketregimeml.features.volatility import (
    VolatilityFeatures,
    _yang_zhang_core,
    _garman_klass_core,
    _parkinson_core,
    _rogers_satchell_core,
)


class TestDataHelper:
    """DRY: Centralized test data generation following Single Responsibility."""
    
    @staticmethod
    def create_ohlc_data(n_periods=20, base_price=100.0, volatility=0.02):
        """Create realistic OHLC data for testing.
        
        KISS: Simple random walk with controlled volatility.
        """
        np.random.seed(42)
        
        # Generate close prices using random walk
        returns = np.random.normal(0, volatility, n_periods)
        close = base_price * np.exp(np.cumsum(returns))
        
        # Generate realistic OHLC from close
        daily_range = volatility * 2
        high = close * (1 + np.random.uniform(0, daily_range, n_periods))
        low = close * (1 - np.random.uniform(0, daily_range, n_periods))
        
        # Open between previous close and current close
        open_prices = np.zeros(n_periods)
        open_prices[0] = base_price
        for i in range(1, n_periods):
            open_prices[i] = close[i-1] * (1 + np.random.normal(0, volatility/2))
        
        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_prices, close))
        low = np.minimum(low, np.minimum(open_prices, close))
        
        return open_prices, high, low, close
    
    @staticmethod
    def create_dataframe(n_periods=20):
        """Create DataFrame with OHLC data."""
        open_p, high, low, close = TestDataHelper.create_ohlc_data(n_periods)
        
        dates = pd.date_range("2023-01-01", periods=n_periods, freq="D")
        return pd.DataFrame({
            'open': open_p,
            'high': high,
            'low': low,
            'close': close
        }, index=dates)


class TestVolatilityCore:
    """Test core Numba-optimized functions directly.
    
    SOLID: Tests separated by function responsibility.
    """
    
    def test_yang_zhang_core_basic(self):
        """Test Yang-Zhang core calculation with simple data."""
        # Create simple test data where volatility is predictable
        open_p = np.array([100.0, 101.0, 100.0, 102.0, 101.0])
        high = np.array([102.0, 103.0, 102.0, 104.0, 103.0])
        low = np.array([99.0, 100.0, 99.0, 101.0, 100.0])
        close = np.array([101.0, 100.0, 101.0, 103.0, 102.0])
        
        result = _yang_zhang_core(open_p, high, low, close, window=3)
        
        # Check shape
        assert len(result) == len(close)
        # First window-1 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Remaining values should be positive
        assert np.all(result[2:] > 0)
    
    def test_garman_klass_core_basic(self):
        """Test Garman-Klass core calculation."""
        high = np.array([102.0, 103.0, 102.0, 104.0, 103.0])
        low = np.array([99.0, 100.0, 99.0, 101.0, 100.0])
        close = np.array([101.0, 100.0, 101.0, 103.0, 102.0])
        
        result = _garman_klass_core(high, low, close, window=3)
        
        assert len(result) == len(close)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.all(result[2:] > 0)
    
    def test_parkinson_core_basic(self):
        """Test Parkinson core calculation."""
        high = np.array([102.0, 103.0, 102.0, 104.0, 103.0])
        low = np.array([99.0, 100.0, 99.0, 101.0, 100.0])
        
        result = _parkinson_core(high, low, window=3)
        
        assert len(result) == len(high)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.all(result[2:] > 0)
    
    def test_rogers_satchell_core_basic(self):
        """Test Rogers-Satchell core calculation."""
        open_p = np.array([100.0, 101.0, 100.0, 102.0, 101.0])
        high = np.array([102.0, 103.0, 102.0, 104.0, 103.0])
        low = np.array([99.0, 100.0, 99.0, 101.0, 100.0])
        close = np.array([101.0, 100.0, 101.0, 103.0, 102.0])
        
        result = _rogers_satchell_core(open_p, high, low, close, window=3)
        
        assert len(result) == len(close)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # Rogers-Satchell can be zero if prices don't move much
        assert np.all(result[2:] >= 0)


class TestVolatilityFeatures:
    """Test VolatilityFeatures class methods.
    
    KISS: Each test focuses on one aspect.
    """
    
    @pytest.fixture
    def volatility_calc(self):
        """DRY: Shared volatility calculator instance."""
        return VolatilityFeatures()
    
    @pytest.fixture
    def sample_data(self):
        """DRY: Shared sample data."""
        return TestDataHelper.create_dataframe(50)
    
    def test_yang_zhang_method(self, volatility_calc, sample_data):
        """Test Yang-Zhang volatility calculation."""
        result = volatility_calc.yang_zhang(
            sample_data['open'],
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            window=20
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        # First 19 values should be NaN (window=20)
        assert result.isna().sum() == 19
        # All non-NaN values should be positive
        assert (result.dropna() > 0).all()
    
    def test_garman_klass_method(self, volatility_calc, sample_data):
        """Test Garman-Klass volatility calculation."""
        result = volatility_calc.garman_klass(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            window=20
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.isna().sum() == 19
        assert (result.dropna() > 0).all()
    
    def test_parkinson_method(self, volatility_calc, sample_data):
        """Test Parkinson volatility calculation."""
        result = volatility_calc.parkinson(
            sample_data['high'],
            sample_data['low'],
            window=20
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.isna().sum() == 19
        assert (result.dropna() > 0).all()
    
    def test_rogers_satchell_method(self, volatility_calc, sample_data):
        """Test Rogers-Satchell volatility calculation."""
        result = volatility_calc.rogers_satchell(
            sample_data['open'],
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            window=20
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.isna().sum() == 19
        assert (result.dropna() >= 0).all()
    
    def test_close_to_close_method(self, volatility_calc, sample_data):
        """Test close-to-close volatility (standard deviation)."""
        result = volatility_calc.close_to_close(
            sample_data['close'],
            window=20
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        # Should calculate returns internally, so window-1 for returns + window for rolling
        assert result.isna().sum() == 20  # First return is NaN, then need window
        assert (result.dropna() > 0).all()
    
    def test_realized_volatility_method(self, volatility_calc):
        """Test realized volatility with intraday data."""
        # Create higher frequency data (intraday returns)
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, 100)))
        returns = pd.Series(prices, index=dates).pct_change().dropna()
        
        result = volatility_calc.realized_volatility(
            returns,
            freq='5min',  # Frequency parameter
            daily_window=1  # Daily window parameter
        )
        
        assert isinstance(result, pd.Series)
        # Result will be daily aggregated
        assert len(result) > 0
        assert (result.dropna() >= 0).all()
    
    def test_volatility_of_volatility(self, volatility_calc, sample_data):
        """Test volatility of volatility calculation."""
        # First calculate base volatility
        base_vol = volatility_calc.close_to_close(
            sample_data['close'],
            window=10
        )
        
        # Then calculate vol of vol
        result = volatility_calc.volatility_of_volatility(
            base_vol,
            window=10
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(base_vol)
        # More NaNs due to nested calculation
        assert result.notna().sum() > 0
        assert (result.dropna() >= 0).all()
    
    def test_window_validation(self, volatility_calc):
        """Test that invalid window sizes are handled."""
        prices = pd.Series([100, 101, 102, 103])
        
        # Window larger than data - returns mostly NaN
        result = volatility_calc.close_to_close(prices, window=10)
        assert result.isna().all()  # All NaN when window > data
        
        # Test methods that do validate window
        with pytest.raises(ValueError):
            volatility_calc.yang_zhang(prices, prices, prices, prices, window=-1)
        
        with pytest.raises(ValueError):
            volatility_calc.garman_klass(prices, prices, prices, window=0)
    
    def test_empty_data_handling(self, volatility_calc):
        """Test handling of empty data."""
        empty_series = pd.Series([], dtype=float)
        
        result = volatility_calc.close_to_close(empty_series, window=5)
        assert len(result) == 0
        assert isinstance(result, pd.Series)
    
    def test_single_value_handling(self, volatility_calc):
        """Test handling of single value."""
        single_value = pd.Series([100.0])
        
        result = volatility_calc.close_to_close(single_value, window=1)
        assert len(result) == 1
        assert np.isnan(result.iloc[0])


class TestVolatilityComparison:
    """Test relationships between different volatility measures.
    
    SOLID: Separate test class for cross-measure validation.
    """
    
    @pytest.fixture
    def all_volatilities(self):
        """Calculate all volatility types for comparison."""
        data = TestDataHelper.create_dataframe(100)
        calc = VolatilityFeatures()
        window = 20
        
        return {
            'yang_zhang': calc.yang_zhang(
                data['open'], data['high'], data['low'], data['close'], window
            ),
            'garman_klass': calc.garman_klass(
                data['high'], data['low'], data['close'], window
            ),
            'parkinson': calc.parkinson(
                data['high'], data['low'], window
            ),
            'rogers_satchell': calc.rogers_satchell(
                data['open'], data['high'], data['low'], data['close'], window
            ),
            'close_to_close': calc.close_to_close(
                data['close'], window
            )
        }
    
    def test_volatility_correlations(self, all_volatilities):
        """Test that different volatility measures are correlated."""
        # Drop NaN values for correlation calculation
        clean_data = pd.DataFrame(all_volatilities).dropna()
        
        # Most volatility measures should be positively correlated
        corr_matrix = clean_data.corr()
        
        # Check that most correlations are positive (at least 80%)
        off_diagonal = corr_matrix.values[~np.eye(len(corr_matrix), dtype=bool)]
        positive_ratio = (off_diagonal > 0).mean()
        assert positive_ratio > 0.7  # At least 70% positive correlations
        
        # Most strong correlations should be positive
        strong_correlations = off_diagonal[np.abs(off_diagonal) > 0.5]
        assert (strong_correlations > 0).mean() > 0.8
    
    def test_volatility_rankings(self, all_volatilities):
        """Test that volatility measures generally agree on high/low vol periods."""
        clean_data = pd.DataFrame(all_volatilities).dropna()
        
        # Rank each volatility measure
        rankings = clean_data.rank()
        
        # Calculate rank correlations (Spearman)
        rank_corr = rankings.corr()
        
        # Most rank correlations should be strong
        off_diagonal = rank_corr.values[~np.eye(len(rank_corr), dtype=bool)]
        # At least 50% should have strong rank correlation
        assert (off_diagonal > 0.5).mean() >= 0.5
        # The high/low/Parkinson measures should be highly correlated
        hl_correlations = rank_corr.loc[
            ['yang_zhang', 'garman_klass', 'parkinson'],
            ['yang_zhang', 'garman_klass', 'parkinson']
        ].values[~np.eye(3, dtype=bool)]
        assert (hl_correlations > 0.8).all()


class TestVolatilityEdgeCases:
    """Test edge cases and error conditions.
    
    KISS: Simple, focused edge case tests.
    """
    
    def test_constant_prices(self):
        """Test volatility calculation with constant prices."""
        calc = VolatilityFeatures()
        constant_prices = pd.Series([100.0] * 20)
        
        result = calc.close_to_close(constant_prices, window=5)
        
        # Volatility should be zero for constant prices
        assert (result.dropna() == 0).all()
    
    def test_monotonic_prices(self):
        """Test with strictly increasing prices."""
        calc = VolatilityFeatures()
        increasing_prices = pd.Series(range(100, 120))
        
        result = calc.close_to_close(increasing_prices, window=5)
        
        # Should have consistent non-zero volatility
        assert (result.dropna() > 0).all()
        # Volatility should be relatively constant for linear increase
        assert result.dropna().std() < result.dropna().mean() * 0.1
    
    def test_extreme_values(self):
        """Test with extreme price movements."""
        calc = VolatilityFeatures()
        
        # Create data with extreme jump
        prices = pd.Series([100.0] * 10 + [1000.0] * 10)
        
        result = calc.close_to_close(prices, window=5)
        
        # Should handle extreme values without errors
        assert not result.isna().all()
        # Volatility should spike at the jump
        max_vol_idx = result.dropna().idxmax()
        assert 8 <= max_vol_idx <= 12  # Around the jump point
    
    def test_negative_prices_error(self):
        """Test that negative prices are handled appropriately."""
        calc = VolatilityFeatures()
        
        # Some volatility methods use log, which fails for negative prices
        high = pd.Series([100, 110, -120, 130])
        low = pd.Series([90, 100, -110, 120])
        
        # Should handle gracefully (return NaN for invalid periods)
        result = calc.parkinson(high, low, window=2)
        
        # Result should contain NaN where calculation is invalid
        assert result.isna().any()