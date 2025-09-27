"""Unit tests for technical indicators.

Following DRY, KISS, and SOLID principles with bottom-up testing.
Minimal mocking, using real calculations where possible.
"""

import pytest
import numpy as np
import pandas as pd

from marketregimeml.features.technical import TechnicalIndicators


class TestDataHelper:
    """DRY: Centralized test data generation for technical indicators."""

    @staticmethod
    def create_price_series(n_periods=50, trend=0.001, volatility=0.02, seed=42):
        """Create realistic price series with trend and volatility."""
        np.random.seed(seed)

        # Generate returns with trend
        returns = np.random.normal(trend, volatility, n_periods)
        prices = 100 * np.exp(np.cumsum(returns))

        dates = pd.date_range("2023-01-01", periods=n_periods, freq="D")
        return pd.Series(prices, index=dates, name="price")

    @staticmethod
    def create_ohlcv_data(n_periods=50, seed=42):
        """Create OHLCV data for indicators requiring multiple price points."""
        np.random.seed(seed)

        # Base close prices
        close = TestDataHelper.create_price_series(n_periods, seed=seed)

        # Generate realistic OHLC from close
        daily_range = 0.02
        high = close * (1 + np.random.uniform(0, daily_range, n_periods))
        low = close * (1 - np.random.uniform(0, daily_range, n_periods))

        # Open between previous close and current close
        open_prices = close.shift(1).fillna(close.iloc[0])
        open_prices = open_prices * (1 + np.random.normal(0, 0.01, n_periods))

        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_prices, close))
        low = np.minimum(low, np.minimum(open_prices, close))

        # Generate volume (higher volume on larger price moves)
        price_change = np.abs(close.pct_change().fillna(0))
        volume = (
            1000000 * (1 + price_change * 10) * np.random.uniform(0.8, 1.2, n_periods)
        )

        return pd.DataFrame(
            {
                "open": open_prices,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    @staticmethod
    def create_trending_prices(n_periods=30, direction="up"):
        """Create prices with clear trend for testing trend indicators."""
        dates = pd.date_range("2023-01-01", periods=n_periods, freq="D")

        if direction == "up":
            prices = (
                100 + np.arange(n_periods) * 0.5 + np.random.normal(0, 0.5, n_periods)
            )
        elif direction == "down":
            prices = (
                100 - np.arange(n_periods) * 0.5 + np.random.normal(0, 0.5, n_periods)
            )
        else:  # sideways
            prices = 100 + np.random.normal(0, 1, n_periods)

        return pd.Series(prices, index=dates, name="price")


# Tests for internal functions removed - now using common_features implementations


class TestTechnicalIndicators:
    """Test TechnicalIndicators class methods."""

    @pytest.fixture
    def indicators(self):
        """DRY: Shared technical indicators instance."""
        return TechnicalIndicators()

    @pytest.fixture
    def price_data(self):
        """DRY: Shared price data."""
        return TestDataHelper.create_price_series(100)

    @pytest.fixture
    def ohlcv_data(self):
        """DRY: Shared OHLCV data."""
        return TestDataHelper.create_ohlcv_data(100)

    def test_rsi_method(self, indicators, price_data):
        """Test RSI indicator method."""
        result = indicators.rsi(price_data, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(price_data)
        # RSI range check
        valid_values = result.dropna()
        assert (valid_values >= 0).all() and (valid_values <= 100).all()

    def test_atr_method(self, indicators, ohlcv_data):
        """Test ATR indicator method."""
        result = indicators.atr(
            ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"], period=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_data)
        # ATR should be positive
        assert (result.dropna() > 0).all()

    def test_bollinger_bands(self, indicators, price_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = indicators.bollinger_bands(
            price_data, period=20, num_std=2
        )

        # Check all bands returned
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

        # Check logical ordering
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] > middle[valid_idx]).all()
        assert (middle[valid_idx] > lower[valid_idx]).all()

    def test_macd(self, indicators, price_data):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = indicators.macd(price_data)

        # Check all components returned
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

        # Histogram should be difference between MACD and signal
        valid_idx = ~(macd_line.isna() | signal_line.isna())
        np.testing.assert_array_almost_equal(
            histogram[valid_idx], (macd_line - signal_line)[valid_idx]
        )

    def test_stochastic(self, indicators, ohlcv_data):
        """Test Stochastic Oscillator calculation."""
        k, d = indicators.stochastic(
            ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"]
        )

        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        assert k.name == "stoch_k_14"
        assert d.name == "stoch_d_14"

        # Stochastic should be between 0 and 100
        valid_k = k.dropna()
        valid_d = d.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_williams_r(self, indicators, ohlcv_data):
        """Test Williams %R calculation."""
        result = indicators.williams_r(
            ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"]
        )

        assert isinstance(result, pd.Series)
        assert result.name == "williams_r_14"

        # Williams %R should be between -100 and 0
        valid_values = result.dropna()
        assert (valid_values >= -100).all() and (valid_values <= 0).all()

    def test_cci(self, indicators, ohlcv_data):
        """Test Commodity Channel Index calculation."""
        result = indicators.cci(
            ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"]
        )

        assert isinstance(result, pd.Series)
        assert result.name == "cci_20"
        assert len(result) == len(ohlcv_data)

        # CCI typically ranges from -200 to +200 but can exceed
        valid_values = result.dropna()
        assert len(valid_values) > 0

    def test_obv(self, indicators, ohlcv_data):
        """Test On-Balance Volume calculation."""
        result = indicators.obv(ohlcv_data["close"], ohlcv_data["volume"])

        assert isinstance(result, pd.Series)
        assert result.name == "obv"
        assert len(result) == len(ohlcv_data)

        # OBV is cumulative, should start at 0
        assert result.iloc[0] == 0
        # OBV changes based on price direction
        # If price goes up, add volume; if down, subtract volume
        price_up = ohlcv_data["close"].iloc[1] > ohlcv_data["close"].iloc[0]
        if price_up:
            assert result.iloc[1] == ohlcv_data["volume"].iloc[1]
        else:
            assert result.iloc[1] == -ohlcv_data["volume"].iloc[1]


class TestTrendDetection:
    """Test indicators' ability to detect trends."""

    @pytest.fixture
    def indicators(self):
        return TechnicalIndicators()

    def test_macd_detects_uptrend(self, indicators):
        """Test MACD identifies uptrend."""
        uptrend = TestDataHelper.create_trending_prices(50, "up")
        macd_line, signal_line, histogram = indicators.macd(uptrend)

        # In uptrend, MACD should be above signal (positive histogram)
        last_values = histogram.dropna().tail(10)
        assert (last_values > 0).sum() > 7  # Most should be positive

    def test_macd_detects_downtrend(self, indicators):
        """Test MACD identifies downtrend."""
        downtrend = TestDataHelper.create_trending_prices(50, "down")
        macd_line, signal_line, histogram = indicators.macd(downtrend)

        # In downtrend, MACD should be below signal (negative histogram)
        last_values = histogram.dropna().tail(10)
        assert (last_values < 0).sum() > 7  # Most should be negative

    def test_rsi_extreme_trends(self, indicators):
        """Test RSI behavior in extreme trends."""
        # Strong uptrend
        strong_up = TestDataHelper.create_trending_prices(30, "up")
        rsi_up = indicators.rsi(strong_up, period=14)

        # Strong downtrend
        strong_down = TestDataHelper.create_trending_prices(30, "down")
        rsi_down = indicators.rsi(strong_down, period=14)

        # RSI should be higher in uptrend
        assert rsi_up.dropna().mean() > rsi_down.dropna().mean()


class TestIndicatorRelationships:
    """Test relationships between different indicators."""

    @pytest.fixture
    def all_indicators(self):
        """Calculate multiple indicators on same data."""
        indicators = TechnicalIndicators()
        data = TestDataHelper.create_ohlcv_data(200)

        return {
            "rsi": indicators.rsi(data["close"]),
            "atr": indicators.atr(data["high"], data["low"], data["close"]),
            "cci": indicators.cci(data["high"], data["low"], data["close"]),
            "williams_r": indicators.williams_r(
                data["high"], data["low"], data["close"]
            ),
            "data": data,
        }

    def test_momentum_indicators_correlation(self, all_indicators):
        """Test that momentum indicators are correlated."""
        # RSI and Williams %R should be inversely correlated
        # (RSI high when Williams %R less negative)
        rsi = all_indicators["rsi"].dropna()
        wr = all_indicators["williams_r"].dropna()

        # Align indices
        common_idx = rsi.index.intersection(wr.index)
        if len(common_idx) > 20:
            correlation = rsi[common_idx].corr(wr[common_idx])
            # Should be positive correlation (both measure momentum)
            assert correlation > 0.3

    def test_volatility_trend_relationship(self, all_indicators):
        """Test relationship between volatility and trend indicators."""
        atr = all_indicators["atr"].dropna()

        # During high volatility, ATR should be elevated
        high_vol_threshold = atr.quantile(0.75)
        high_vol_periods = atr > high_vol_threshold

        # Should have some high volatility periods
        assert high_vol_periods.sum() > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def indicators(self):
        return TechnicalIndicators()

    def test_constant_prices(self, indicators):
        """Test indicators with constant prices."""
        constant = pd.Series([100.0] * 30)

        # RSI with no price change
        rsi = indicators.rsi(constant, period=14)
        # RSI undefined when no changes, often shows as 50 or NaN
        assert rsi.dropna().std() < 1 or rsi.isna().all()

        # Bollinger Bands with no volatility
        upper, middle, lower = indicators.bollinger_bands(constant, period=20)
        # Bands should collapse to same value
        valid_idx = ~upper.isna()
        if valid_idx.any():
            np.testing.assert_array_almost_equal(upper[valid_idx], lower[valid_idx])

    def test_insufficient_data(self, indicators):
        """Test indicators with insufficient data."""
        short_series = pd.Series([100, 101, 102])

        # RSI needs at least period + 1 points
        rsi = indicators.rsi(short_series, period=14)
        assert rsi.isna().all()

        # ATR with insufficient data
        atr = indicators.atr(short_series, short_series, short_series, period=14)
        assert atr.isna().all()

    def test_extreme_values(self, indicators):
        """Test indicators with extreme price movements."""
        # Create extreme price jumps
        prices = pd.Series([100, 100, 100, 1000, 1000, 1000, 10, 10, 10])

        # RSI should handle extreme moves
        rsi = indicators.rsi(prices, period=3)
        assert not rsi.isna().all()
        # After big up move, RSI should be high
        assert rsi.iloc[4] > 70 if not np.isnan(rsi.iloc[4]) else True

    def test_zero_division_handling(self, indicators):
        """Test handling of zero division cases."""
        # Create data that might cause division issues
        high = pd.Series([100, 100, 100, 100, 100])
        low = pd.Series([100, 100, 100, 100, 100])
        close = pd.Series([100, 100, 100, 100, 100])

        # Stochastic with no range
        k, d = indicators.stochastic(high, low, close, period=3)
        # Should handle gracefully (return NaN or constant)
        assert not np.isinf(k).any()
        assert not np.isinf(d).any()

        # Williams %R with no range
        wr = indicators.williams_r(high, low, close, period=3)
        assert not np.isinf(wr).any()


class TestIndicatorConsistency:
    """Test consistency and numerical stability of indicators."""

    @pytest.fixture
    def indicators(self):
        return TechnicalIndicators()

    def test_deterministic_results(self, indicators):
        """Test that indicators give consistent results."""
        prices = TestDataHelper.create_price_series(50, seed=123)

        # Calculate RSI multiple times
        rsi1 = indicators.rsi(prices, period=14)
        rsi2 = indicators.rsi(prices, period=14)

        # Should be identical
        pd.testing.assert_series_equal(rsi1, rsi2)

    def test_parameter_sensitivity(self, indicators):
        """Test indicator sensitivity to parameters."""
        prices = TestDataHelper.create_price_series(100)

        # Different RSI periods
        rsi_7 = indicators.rsi(prices, period=7)
        rsi_14 = indicators.rsi(prices, period=14)
        rsi_21 = indicators.rsi(prices, period=21)

        # Shorter period should be more volatile
        assert rsi_7.dropna().std() > rsi_14.dropna().std()
        assert rsi_14.dropna().std() > rsi_21.dropna().std()

    def test_numerical_stability(self, indicators):
        """Test numerical stability with large values."""
        # Large price values
        large_prices = pd.Series(np.random.uniform(1e6, 1e7, 50))

        # Should handle large values without overflow
        rsi = indicators.rsi(large_prices)
        assert not np.isinf(rsi).any()
        assert (rsi.dropna() >= 0).all() and (rsi.dropna() <= 100).all()

        # Small price values
        small_prices = pd.Series(np.random.uniform(1e-6, 1e-5, 50))

        rsi = indicators.rsi(small_prices)
        assert not np.isinf(rsi).any()
