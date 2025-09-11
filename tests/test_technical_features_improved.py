"""Improved tests for technical features to increase coverage."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from marketregimeml.features.technical import TechnicalIndicators


class TestTechnicalFeaturesImproved:
    """Comprehensive test suite for technical features."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        # Generate realistic price data
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))

        data = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(n) * 0.005),
                "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
                "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
                "close": close,
                "volume": np.random.randint(1000000, 10000000, n),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)

        return data

    @pytest.fixture
    def features(self):
        """Create TechnicalIndicators instance."""
        return TechnicalIndicators()

    def test_sma(self, features, sample_ohlcv):
        """Test Simple Moving Average."""
        sma_20 = features.sma(sample_ohlcv["close"], window=20)

        assert len(sma_20) == len(sample_ohlcv)
        assert sma_20.isna().sum() == 19  # First 19 values should be NaN
        assert not sma_20[20:].isna().any()  # No NaN after window

        # Test that SMA is actually averaging
        manual_sma = sample_ohlcv["close"].rolling(20).mean()
        pd.testing.assert_series_equal(sma_20, manual_sma, check_names=False)

    def test_ema(self, features, sample_ohlcv):
        """Test Exponential Moving Average."""
        ema_20 = features.ema(sample_ohlcv["close"], window=20)

        assert len(ema_20) == len(sample_ohlcv)
        assert not ema_20.isna().any()  # EMA should have no NaN values

        # EMA should be different from SMA
        sma_20 = features.sma(sample_ohlcv["close"], window=20)
        assert not np.allclose(ema_20[20:], sma_20[20:], equal_nan=True)

    def test_macd(self, features, sample_ohlcv):
        """Test MACD indicator."""
        macd_result = features.macd(sample_ohlcv["close"])

        assert "macd" in macd_result.columns
        assert "signal" in macd_result.columns
        assert "histogram" in macd_result.columns

        assert len(macd_result) == len(sample_ohlcv)

        # MACD histogram should be difference between MACD and signal
        expected_hist = macd_result["macd"] - macd_result["signal"]
        pd.testing.assert_series_equal(
            macd_result["histogram"], expected_hist, check_names=False
        )

    def test_rsi(self, features, sample_ohlcv):
        """Test RSI indicator."""
        rsi = features.rsi(sample_ohlcv["close"], window=14)

        assert len(rsi) == len(sample_ohlcv)

        # RSI should be between 0 and 100
        valid_rsi = rsi[~rsi.isna()]
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

        # Test extreme cases
        # All up moves should give high RSI
        up_prices = pd.Series(range(100))
        up_rsi = features.rsi(up_prices, window=14)
        assert up_rsi.iloc[-1] > 70  # Should be overbought

        # All down moves should give low RSI
        down_prices = pd.Series(range(100, 0, -1))
        down_rsi = features.rsi(down_prices, window=14)
        assert down_rsi.iloc[-1] < 30  # Should be oversold

    def test_bollinger_bands(self, features, sample_ohlcv):
        """Test Bollinger Bands."""
        bb = features.bollinger_bands(
            sample_ohlcv["close"], window=20, num_std=2
        )

        assert "middle" in bb.columns
        assert "upper" in bb.columns
        assert "lower" in bb.columns
        assert "bandwidth" in bb.columns
        assert "percent_b" in bb.columns

        assert len(bb) == len(sample_ohlcv)

        # Middle band should be SMA
        sma_20 = features.sma(sample_ohlcv["close"], window=20)
        pd.testing.assert_series_equal(bb["middle"], sma_20, check_names=False)

        # Upper band should be above middle, lower below
        valid_idx = ~bb["middle"].isna()
        assert (
            bb.loc[valid_idx, "upper"] >= bb.loc[valid_idx, "middle"]
        ).all()
        assert (
            bb.loc[valid_idx, "lower"] <= bb.loc[valid_idx, "middle"]
        ).all()

        # Bandwidth should be positive
        assert (bb.loc[valid_idx, "bandwidth"] > 0).all()

    def test_stochastic_oscillator(self, features, sample_ohlcv):
        """Test Stochastic Oscillator."""
        stoch = features.stochastic_oscillator(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            window=14,
        )

        assert "K" in stoch.columns
        assert "D" in stoch.columns

        # K and D should be between 0 and 100
        valid_k = stoch["K"][~stoch["K"].isna()]
        valid_d = stoch["D"][~stoch["D"].isna()]

        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()

    def test_atr(self, features, sample_ohlcv):
        """Test Average True Range."""
        atr = features.atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            window=14,
        )

        assert len(atr) == len(sample_ohlcv)

        # ATR should be positive
        valid_atr = atr[~atr.isna()]
        assert (valid_atr >= 0).all()

        # ATR should increase with volatility
        # Create high volatility data
        high_vol = sample_ohlcv.copy()
        high_vol["high"] = high_vol["close"] * 1.05
        high_vol["low"] = high_vol["close"] * 0.95

        high_vol_atr = features.atr(
            high_vol["high"], high_vol["low"], high_vol["close"], window=14
        )

        # High volatility ATR should be larger
        assert high_vol_atr.mean() > atr.mean()

    def test_obv(self, features, sample_ohlcv):
        """Test On-Balance Volume."""
        obv = features.obv(sample_ohlcv["close"], sample_ohlcv["volume"])

        assert len(obv) == len(sample_ohlcv)
        assert not obv.isna().any()

        # First value should be first volume
        assert obv.iloc[0] == sample_ohlcv["volume"].iloc[0]

        # Test cumulative nature
        price_changes = sample_ohlcv["close"].diff()
        volume_signs = np.where(
            price_changes > 0, 1, np.where(price_changes < 0, -1, 0)
        )
        signed_volume = sample_ohlcv["volume"] * volume_signs
        expected_obv = signed_volume.cumsum()
        expected_obv.iloc[0] = sample_ohlcv["volume"].iloc[0]

        pd.testing.assert_series_equal(obv, expected_obv, check_names=False)

    def test_vwap(self, features, sample_ohlcv):
        """Test Volume Weighted Average Price."""
        vwap = features.vwap(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )

        assert len(vwap) == len(sample_ohlcv)
        assert not vwap.isna().any()

        # VWAP should be within high/low range
        assert (vwap >= sample_ohlcv["low"]).all()
        assert (vwap <= sample_ohlcv["high"]).all()

    def test_pivot_points(self, features, sample_ohlcv):
        """Test Pivot Points."""
        pivots = features.pivot_points(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )

        expected_cols = ["pivot", "r1", "r2", "r3", "s1", "s2", "s3"]
        for col in expected_cols:
            assert col in pivots.columns

        # Check relationships
        # R1 > Pivot > S1
        assert (pivots["r1"] > pivots["pivot"]).all()
        assert (pivots["pivot"] > pivots["s1"]).all()

        # R2 > R1, R3 > R2
        assert (pivots["r2"] > pivots["r1"]).all()
        assert (pivots["r3"] > pivots["r2"]).all()

        # S1 > S2 > S3
        assert (pivots["s1"] > pivots["s2"]).all()
        assert (pivots["s2"] > pivots["s3"]).all()

    def test_ichimoku_cloud(self, features, sample_ohlcv):
        """Test Ichimoku Cloud."""
        ichimoku = features.ichimoku_cloud(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )

        expected_cols = [
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
        ]
        for col in expected_cols:
            assert col in ichimoku.columns

        assert len(ichimoku) == len(sample_ohlcv)

    def test_calculate_features(self, features, sample_ohlcv):
        """Test calculate_features main method."""
        result = features.calculate_features(sample_ohlcv)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)

        # Check that features were calculated
        feature_cols = [
            col
            for col in result.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        assert len(feature_cols) > 0

        # Test with custom indicators
        custom_result = features.calculate_features(
            sample_ohlcv, indicators=["sma", "rsi", "macd"]
        )

        # Should have SMA, RSI, and MACD columns
        assert any("sma" in col.lower() for col in custom_result.columns)
        assert any("rsi" in col.lower() for col in custom_result.columns)
        assert any("macd" in col.lower() for col in custom_result.columns)

    def test_adx(self, features, sample_ohlcv):
        """Test ADX indicator."""
        adx = features.adx(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            window=14,
        )

        assert len(adx) == len(sample_ohlcv)

        # ADX should be between 0 and 100
        valid_adx = adx[~adx.isna()]
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()

    def test_cci(self, features, sample_ohlcv):
        """Test Commodity Channel Index."""
        cci = features.cci(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            window=20,
        )

        assert len(cci) == len(sample_ohlcv)

        # CCI typically ranges from -200 to +200 but can exceed
        valid_cci = cci[~cci.isna()]
        assert valid_cci.std() > 0  # Should have variation

    def test_mfi(self, features, sample_ohlcv):
        """Test Money Flow Index."""
        mfi = features.mfi(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
            window=14,
        )

        assert len(mfi) == len(sample_ohlcv)

        # MFI should be between 0 and 100
        valid_mfi = mfi[~mfi.isna()]
        assert (valid_mfi >= 0).all()
        assert (valid_mfi <= 100).all()

    def test_williams_r(self, features, sample_ohlcv):
        """Test Williams %R."""
        williams = features.williams_r(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            window=14,
        )

        assert len(williams) == len(sample_ohlcv)

        # Williams %R should be between -100 and 0
        valid_williams = williams[~williams.isna()]
        assert (valid_williams >= -100).all()
        assert (valid_williams <= 0).all()

    def test_roc(self, features, sample_ohlcv):
        """Test Rate of Change."""
        roc = features.roc(sample_ohlcv["close"], window=10)

        assert len(roc) == len(sample_ohlcv)

        # Manually calculate ROC for verification
        expected_roc = (
            (sample_ohlcv["close"] - sample_ohlcv["close"].shift(10))
            / sample_ohlcv["close"].shift(10)
            * 100
        )

        pd.testing.assert_series_equal(roc, expected_roc, check_names=False)

    def test_feature_names(self, features):
        """Test getting feature names."""
        names = features.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)

        # Common indicators should be in the list
        expected_indicators = ["sma", "ema", "rsi", "macd", "bollinger_bands"]
        for indicator in expected_indicators:
            assert any(indicator in name.lower() for name in names)

    def test_edge_cases(self, features):
        """Test edge cases and error handling."""
        # Test with small data
        small_data = pd.Series([100, 101, 99, 102, 98])

        # SMA with window larger than data
        sma = features.sma(small_data, window=10)
        assert sma.isna().all()  # All NaN when window > data length

        # RSI with constant prices
        constant_prices = pd.Series([100] * 20)
        rsi = features.rsi(constant_prices, window=14)
        # RSI should be around 50 for no change
        assert 45 <= rsi.iloc[-1] <= 55

        # Test with NaN values
        data_with_nan = pd.Series([100, np.nan, 102, 103, np.nan, 105, 106])
        sma = features.sma(data_with_nan, window=3)
        assert len(sma) == len(data_with_nan)
