"""
Comprehensive tests for technical features module.

Following TDD, SOLID, DRY, and KISS principles.
Targeting 90%+ coverage for features/technical.py
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock
import warnings

warnings.filterwarnings("ignore")


class TestTechnicalIndicators:
    """Comprehensive test suite for TechnicalIndicators class."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create realistic OHLCV data for testing.

        Arrange: Generate synthetic but realistic market data
        """
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        # Generate realistic price series with trend
        base_price = 100
        trend = np.linspace(0, 20, n)  # Upward trend
        noise = np.random.normal(0, 2, n)
        close = base_price + trend + noise

        # Create OHLC from close with realistic relationships
        high = close + np.abs(np.random.normal(0, 1, n))
        low = close - np.abs(np.random.normal(0, 1, n))
        open_shift = np.roll(close, 1)
        open_shift[0] = close[0]

        # Volume with some correlation to price changes
        volume = (
            1000
            + np.abs(np.diff(np.append(close[0], close))) * 100
            + np.random.normal(0, 200, n)
        )
        volume = np.maximum(volume, 100)  # Ensure positive volume

        return pd.DataFrame(
            {
                "open": open_shift,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )

    @pytest.fixture
    def indicators(self):
        """Create TechnicalIndicators instance for testing."""
        from marketregimeml.features.technical import TechnicalIndicators

        return TechnicalIndicators()

    # RSI Tests
    def test_rsi_basic_calculation(self, indicators, sample_ohlcv):
        """Test RSI calculation with default parameters."""
        # Arrange
        prices = sample_ohlcv["close"]

        # Act
        rsi = indicators.rsi(prices)

        # Assert
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)
        assert rsi.index.equals(prices.index)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(0 <= x <= 100 for x in valid_rsi)
        # First 14 values should be NaN (default period)
        assert rsi.iloc[:13].isna().all()

    def test_rsi_custom_period(self, indicators, sample_ohlcv):
        """Test RSI with custom period."""
        # Arrange
        prices = sample_ohlcv["close"]
        period = 21

        # Act
        rsi = indicators.rsi(prices, period=period)

        # Assert
        # First (period-1) values should be NaN
        assert rsi.iloc[: period - 1].isna().all()
        # Should have valid values after period
        assert not rsi.iloc[period:].isna().all()

    def test_rsi_edge_cases(self, indicators):
        """Test RSI with edge cases."""
        # Arrange
        # Constant prices (no change)
        constant_prices = pd.Series([100] * 30)

        # Act
        rsi_constant = indicators.rsi(constant_prices)

        # Assert
        # For constant prices, RSI behavior can vary by implementation
        # Just ensure we get valid output
        valid_rsi = rsi_constant.dropna()
        if len(valid_rsi) > 0:
            assert all(0 <= x <= 100 for x in valid_rsi)  # Valid RSI range
            # Most implementations give 0 or 50 for constant prices
            unique_values = set(valid_rsi)
            assert len(unique_values) <= 2  # Should be consistent

    def test_rsi_insufficient_data(self, indicators):
        """Test RSI with insufficient data."""
        # Arrange
        short_series = pd.Series([100, 101, 99])  # Only 3 points

        # Act
        rsi = indicators.rsi(short_series, period=14)

        # Assert
        # Should return all NaN for insufficient data
        assert rsi.isna().all()

    # ATR Tests
    def test_atr_basic_calculation(self, indicators, sample_ohlcv):
        """Test ATR calculation."""
        # Arrange
        high = sample_ohlcv["high"]
        low = sample_ohlcv["low"]
        close = sample_ohlcv["close"]

        # Act
        atr = indicators.atr(high, low, close)

        # Assert
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(high)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert all(x >= 0 for x in valid_atr)
        # First value should be NaN (needs previous close)
        assert pd.isna(atr.iloc[0])

    def test_atr_custom_period(self, indicators, sample_ohlcv):
        """Test ATR with custom period."""
        # Arrange
        high, low, close = (
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )
        period = 21

        # Act
        atr = indicators.atr(high, low, close, period=period)

        # Assert
        # Should have valid values after period
        assert not atr.iloc[period:].isna().all()

    # Bollinger Bands Tests
    def test_bollinger_bands_basic(self, indicators, sample_ohlcv):
        """Test Bollinger Bands calculation."""
        # Arrange
        prices = sample_ohlcv["close"]

        # Act
        bb_upper, bb_middle, bb_lower = indicators.bollinger_bands(prices)

        # Assert
        assert isinstance(bb_upper, pd.Series)
        assert isinstance(bb_middle, pd.Series)
        assert isinstance(bb_lower, pd.Series)

        # All should have same length and index
        assert len(bb_upper) == len(bb_middle) == len(bb_lower) == len(prices)
        assert bb_upper.index.equals(bb_middle.index.equals(bb_lower.index))

        # Upper should be >= Middle >= Lower (where not NaN)
        valid_idx = ~(bb_upper.isna() | bb_middle.isna() | bb_lower.isna())
        if valid_idx.any():
            assert all(bb_upper[valid_idx] >= bb_middle[valid_idx])
            assert all(bb_middle[valid_idx] >= bb_lower[valid_idx])

    def test_bollinger_bands_custom_params(self, indicators, sample_ohlcv):
        """Test Bollinger Bands with custom parameters."""
        # Arrange
        prices = sample_ohlcv["close"]
        window = 10
        num_std = 1.5

        # Act
        bb_upper, bb_middle, bb_lower = indicators.bollinger_bands(
            prices, window=window, num_std=num_std
        )

        # Assert
        # First (window-1) values should be NaN
        assert bb_upper.iloc[: window - 1].isna().all()
        assert bb_middle.iloc[: window - 1].isna().all()
        assert bb_lower.iloc[: window - 1].isna().all()

    # MACD Tests
    def test_macd_basic(self, indicators, sample_ohlcv):
        """Test MACD calculation."""
        # Arrange
        prices = sample_ohlcv["close"]

        # Act
        macd_line, signal_line, histogram = indicators.macd(prices)

        # Assert
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

        # All should have same length
        assert (
            len(macd_line) == len(signal_line) == len(histogram) == len(prices)
        )

        # Histogram should equal MACD - Signal (where not NaN)
        valid_idx = ~(macd_line.isna() | signal_line.isna())
        if valid_idx.any():
            np.testing.assert_array_almost_equal(
                histogram[valid_idx],
                macd_line[valid_idx] - signal_line[valid_idx],
                decimal=10,
            )

    def test_macd_custom_periods(self, indicators, sample_ohlcv):
        """Test MACD with custom periods."""
        # Arrange
        prices = sample_ohlcv["close"]
        fast_period = 8
        slow_period = 21
        signal_period = 7

        # Act
        macd_line, signal_line, histogram = indicators.macd(
            prices,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

        # Assert
        # Should have valid values after slow_period + signal_period
        expected_nan_period = slow_period + signal_period - 2
        assert not signal_line.iloc[expected_nan_period:].isna().all()

    # Stochastic Tests
    def test_stochastic_basic(self, indicators, sample_ohlcv):
        """Test Stochastic oscillator calculation."""
        # Arrange
        high = sample_ohlcv["high"]
        low = sample_ohlcv["low"]
        close = sample_ohlcv["close"]

        # Act
        k_percent, d_percent = indicators.stochastic(high, low, close)

        # Assert
        assert isinstance(k_percent, pd.Series)
        assert isinstance(d_percent, pd.Series)

        # Should be between 0 and 100
        valid_k = k_percent.dropna()
        valid_d = d_percent.dropna()

        if len(valid_k) > 0:
            assert all(0 <= x <= 100 for x in valid_k)
        if len(valid_d) > 0:
            assert all(0 <= x <= 100 for x in valid_d)

    # Williams %R Tests
    def test_williams_r_basic(self, indicators, sample_ohlcv):
        """Test Williams %R calculation."""
        # Arrange
        high = sample_ohlcv["high"]
        low = sample_ohlcv["low"]
        close = sample_ohlcv["close"]

        # Act
        williams_r = indicators.williams_r(high, low, close)

        # Assert
        assert isinstance(williams_r, pd.Series)

        # Should be between -100 and 0
        valid_values = williams_r.dropna()
        if len(valid_values) > 0:
            assert all(-100 <= x <= 0 for x in valid_values)

    # CCI Tests
    def test_cci_basic(self, indicators, sample_ohlcv):
        """Test Commodity Channel Index calculation."""
        # Arrange
        high = sample_ohlcv["high"]
        low = sample_ohlcv["low"]
        close = sample_ohlcv["close"]

        # Act
        cci = indicators.cci(high, low, close)

        # Assert
        assert isinstance(cci, pd.Series)
        assert len(cci) == len(high)

        # CCI can range widely, but should be numeric
        valid_cci = cci.dropna()
        assert all(
            isinstance(x, (int, float)) and not pd.isna(x) for x in valid_cci
        )

    # OBV Tests
    def test_obv_basic(self, indicators, sample_ohlcv):
        """Test On-Balance Volume calculation."""
        # Arrange
        close = sample_ohlcv["close"]
        volume = sample_ohlcv["volume"]

        # Act
        obv = indicators.obv(close, volume)

        # Assert
        assert isinstance(obv, pd.Series)
        assert len(obv) == len(close)

        # OBV should be cumulative
        assert not obv.iloc[-1] == obv.iloc[1]  # Should change over time

    def test_obv_price_volume_relationship(self, indicators):
        """Test OBV price-volume relationship logic."""
        # Arrange
        close = pd.Series([100, 101, 100, 102, 98])  # Up, down, up, down
        volume = pd.Series([1000, 1000, 1000, 1000, 1000])

        # Act
        obv = indicators.obv(close, volume)

        # Assert
        # First value should be first volume
        assert obv.iloc[0] == volume.iloc[0]

        # Check cumulative logic
        expected = [1000, 2000, 1000, 2000, 1000]  # Up adds, down subtracts
        np.testing.assert_array_equal(obv, expected)

    # VWAP Tests
    def test_vwap_basic(self, indicators, sample_ohlcv):
        """Test Volume Weighted Average Price calculation."""
        # Arrange
        high = sample_ohlcv["high"]
        low = sample_ohlcv["low"]
        close = sample_ohlcv["close"]
        volume = sample_ohlcv["volume"]

        # Act
        vwap = indicators.vwap(high, low, close, volume)

        # Assert
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(close)

        # VWAP should be within reasonable range of prices
        price_range = [close.min(), close.max()]
        valid_vwap = vwap.dropna()

        if len(valid_vwap) > 0:
            # VWAP might be slightly outside price range but should be reasonable
            assert all(
                price_range[0] * 0.8 <= x <= price_range[1] * 1.2
                for x in valid_vwap
            )

    # Money Flow Index Tests
    def test_money_flow_index_basic(self, indicators, sample_ohlcv):
        """Test Money Flow Index calculation."""
        # Arrange
        high = sample_ohlcv["high"]
        low = sample_ohlcv["low"]
        close = sample_ohlcv["close"]
        volume = sample_ohlcv["volume"]

        # Act
        mfi = indicators.money_flow_index(high, low, close, volume)

        # Assert
        assert isinstance(mfi, pd.Series)

        # MFI should be between 0 and 100
        valid_mfi = mfi.dropna()
        if len(valid_mfi) > 0:
            assert all(0 <= x <= 100 for x in valid_mfi)

    # ADX Tests
    def test_adx_basic(self, indicators, sample_ohlcv):
        """Test Average Directional Index calculation."""
        # Arrange
        high = sample_ohlcv["high"]
        low = sample_ohlcv["low"]
        close = sample_ohlcv["close"]

        # Act
        adx = indicators.adx(high, low, close)

        # Assert
        assert isinstance(adx, pd.Series)

        # ADX should be between 0 and 100
        valid_adx = adx.dropna()
        if len(valid_adx) > 0:
            assert all(0 <= x <= 100 for x in valid_adx)

    # Ichimoku Cloud Tests
    def test_ichimoku_cloud_basic(self, indicators, sample_ohlcv):
        """Test Ichimoku Cloud calculation."""
        # Arrange
        high = sample_ohlcv["high"]
        low = sample_ohlcv["low"]
        close = sample_ohlcv["close"]

        # Act
        result = indicators.ichimoku_cloud(high, low, close)

        # Assert
        assert isinstance(result, dict)
        expected_keys = [
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
        ]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], pd.Series)
            assert len(result[key]) == len(close)

    # Edge Cases and Error Handling
    def test_empty_series_handling(self, indicators):
        """Test handling of empty series."""
        # Arrange
        empty_series = pd.Series([], dtype=float)

        # Act & Assert
        with pytest.raises((ValueError, IndexError)):
            indicators.rsi(empty_series)

    def test_single_value_series(self, indicators):
        """Test handling of single-value series."""
        # Arrange
        single_value = pd.Series([100.0])

        # Act
        rsi = indicators.rsi(single_value)

        # Assert
        assert len(rsi) == 1
        assert pd.isna(rsi.iloc[0])  # Should be NaN for insufficient data

    def test_nan_handling(self, indicators):
        """Test handling of NaN values in input."""
        # Arrange
        prices_with_nan = pd.Series([100, 101, np.nan, 103, 104])

        # Act
        rsi = indicators.rsi(prices_with_nan)

        # Assert
        # Should handle NaN gracefully (exact behavior depends on implementation)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices_with_nan)

    def test_negative_period_handling(self, indicators, sample_ohlcv):
        """Test handling of invalid period parameters."""
        # Arrange
        prices = sample_ohlcv["close"]

        # Act & Assert
        with pytest.raises(ValueError):
            indicators.rsi(prices, period=-1)

    def test_zero_period_handling(self, indicators, sample_ohlcv):
        """Test handling of zero period."""
        # Arrange
        prices = sample_ohlcv["close"]

        # Act & Assert
        with pytest.raises(ValueError):
            indicators.rsi(prices, period=0)

    # Integration Tests
    def test_multiple_indicators_same_data(self, indicators, sample_ohlcv):
        """Test multiple indicators on same dataset for consistency."""
        # Arrange
        high, low, close, volume = (
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )

        # Act
        rsi = indicators.rsi(close)
        atr = indicators.atr(high, low, close)
        obv = indicators.obv(close, volume)

        # Assert
        # All should have same length as input
        assert len(rsi) == len(atr) == len(obv) == len(close)

        # All should use same index
        assert rsi.index.equals(close.index)
        assert atr.index.equals(close.index)
        assert obv.index.equals(close.index)

    def test_indicator_chaining(self, indicators, sample_ohlcv):
        """Test using one indicator's output as input to another."""
        # Arrange
        close = sample_ohlcv["close"]

        # Act
        # Calculate RSI, then calculate RSI of RSI (RSI smoothing)
        rsi_primary = indicators.rsi(close, period=14)
        rsi_secondary = indicators.rsi(rsi_primary.dropna(), period=9)

        # Assert
        assert isinstance(rsi_secondary, pd.Series)
        # Secondary RSI should also be between 0-100
        valid_secondary = rsi_secondary.dropna()
        if len(valid_secondary) > 0:
            assert all(0 <= x <= 100 for x in valid_secondary)


class TestTechnicalIndicatorsPerformance:
    """Test performance characteristics of technical indicators."""

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        np.random.seed(42)
        n = 10000  # Large dataset
        dates = pd.date_range("2000-01-01", periods=n, freq="D")

        close = 100 + np.cumsum(np.random.normal(0, 1, n))
        high = close + np.abs(np.random.normal(0, 0.5, n))
        low = close - np.abs(np.random.normal(0, 0.5, n))
        open_prices = np.roll(close, 1)
        volume = np.random.lognormal(8, 1, n)

        return pd.DataFrame(
            {
                "open": open_prices,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )

    def test_rsi_performance(self, large_dataset):
        """Test RSI calculation performance on large dataset."""
        from marketregimeml.features.technical import TechnicalIndicators

        # Arrange
        indicators = TechnicalIndicators()
        close = large_dataset["close"]

        # Act & Assert (should complete without timeout)
        import time

        start_time = time.time()
        rsi = indicators.rsi(close)
        execution_time = time.time() - start_time

        # Should complete in reasonable time (less than 1 second for 10k points)
        assert execution_time < 1.0
        assert len(rsi) == len(close)
        assert not rsi.dropna().empty
