"""Technical indicators for market analysis.

This module provides specialized technical indicators not available
in common_features. For standard indicators like RSI, Bollinger Bands,
MACD, and ATR, use marketregimeml.features.common_features.
"""

from typing import Tuple, Optional

import numpy as np
import pandas as pd

from marketregimeml.features.base import BaseFeatureCalculator
from marketregimeml.features.common_features import TechnicalFeatures
from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)

# Export the existing class
__all__ = ["TechnicalIndicators"]


class TechnicalIndicators(BaseFeatureCalculator):
    """Technical indicator calculators for market analysis.

    This class provides specialized technical indicators.
    For standard indicators (RSI, Bollinger, MACD, ATR),
    use TechnicalFeatures from common_features module.
    """

    def __init__(self):
        """Initialize technical indicator calculator."""
        logger.debug("TechnicalIndicators initialized")
        self._common_features = TechnicalFeatures()

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical features.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data

        Returns
        -------
        pd.DataFrame
            Technical indicator features
        """
        features = pd.DataFrame(index=data.index)

        # Add specialized indicators only
        if "close" in data.columns and "volume" in data.columns:
            features["obv"] = self.obv(data["close"], data["volume"])

        if all(col in data.columns for col in ["high", "low", "close"]):
            features["cci"] = self.cci(data["high"], data["low"], data["close"])
            features["williams_r"] = self.williams_r(
                data["high"], data["low"], data["close"]
            )

        if all(col in data.columns for col in ["high", "low", "close", "volume"]):
            features["mfi"] = self.money_flow_index(
                data["high"], data["low"], data["close"], data["volume"]
            )
            features["vwap"] = self.vwap(
                data["high"], data["low"], data["close"], data["volume"]
            )

        return features

    # Delegate standard indicators to common_features
    def rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI - delegates to TechnicalFeatures."""
        return self._common_features.calculate_rsi(prices, period)

    def atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate ATR - delegates to TechnicalFeatures."""
        return self._common_features.calculate_atr(high, low, close, period)

    def bollinger_bands(
        self, prices: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands - delegates to TechnicalFeatures."""
        middle, upper, lower = self._common_features.calculate_bollinger_bands(
            prices, period, num_std
        )
        # Return in the same order as original (upper, middle, lower)
        return upper, middle, lower

    def macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD - delegates to TechnicalFeatures."""
        return self._common_features.calculate_macd(prices, fast, slow, signal)

    def stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator.

        Momentum indicator comparing close to price range.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Lookback period
            smooth_k: Smoothing for %K
            smooth_d: Smoothing for %D

        Returns:
            Tuple of (%K, %D)
        """
        # Calculate raw stochastic
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)

        # Smooth %K
        k = k_raw.rolling(window=smooth_k).mean()

        # Calculate %D (signal line)
        d = k.rolling(window=smooth_d).mean()

        k.name = f"stoch_k_{period}"
        d.name = f"stoch_d_{period}"

        return k, d

    def williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate Williams %R.

        Similar to Stochastic but inverted scale.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Lookback period

        Returns:
            Williams %R series
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        wr.name = f"williams_r_{period}"

        return wr

    def cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """Calculate Commodity Channel Index.

        Measures deviation from average price.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: CCI period

        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )

        cci = (typical_price - sma) / (0.015 * mad)
        cci.name = f"cci_{period}"

        return cci

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume.

        Volume-based indicator for trend confirmation.

        Args:
            close: Close price series
            volume: Volume series

        Returns:
            OBV series
        """
        obv = np.where(
            close > close.shift(1),
            volume,
            np.where(close < close.shift(1), -volume, 0),
        )
        obv = pd.Series(obv, index=close.index).cumsum()
        obv.name = "obv"

        return obv

    def vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: Optional[int] = None,
    ) -> pd.Series:
        """Calculate Volume Weighted Average Price.

        Average price weighted by volume.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: Rolling period (None for cumulative)

        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3

        if period:
            cum_vol = volume.rolling(window=period).sum()
            cum_vol_price = (typical_price * volume).rolling(window=period).sum()
        else:
            cum_vol = volume.cumsum()
            cum_vol_price = (typical_price * volume).cumsum()

        vwap = cum_vol_price / cum_vol
        vwap.name = f"vwap_{period}" if period else "vwap"

        return vwap

    def money_flow_index(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate Money Flow Index.

        Volume-weighted RSI, ranges 0-100.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: MFI period

        Returns:
            MFI series
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        # Determine positive and negative money flow
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)

        # Calculate money flow ratio
        positive_mf = (
            pd.Series(positive_flow, index=close.index).rolling(window=period).sum()
        )
        negative_mf = (
            pd.Series(negative_flow, index=close.index).rolling(window=period).sum()
        )

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        mfi.name = f"mfi_{period}"

        return mfi

    def adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate Average Directional Index.

        Measures trend strength (not direction).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ADX period

        Returns:
            ADX series
        """
        # Calculate directional movement
        plus_dm = np.where(
            (high - high.shift(1)) > (low.shift(1) - low),
            np.maximum(high - high.shift(1), 0),
            0,
        )
        minus_dm = np.where(
            (low.shift(1) - low) > (high - high.shift(1)),
            np.maximum(low.shift(1) - low, 0),
            0,
        )

        # Calculate ATR
        atr_val = self.atr(high, low, close, period)

        # Calculate directional indicators
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr_val
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr_val

        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        adx.name = f"adx_{period}"

        return adx

    def ichimoku_cloud(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        conversion: int = 9,
        base: int = 26,
        span_b: int = 52,
        displacement: int = 26,
    ) -> dict:
        """Calculate Ichimoku Cloud components.

        Comprehensive trend-following system.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            conversion: Conversion line period
            base: Base line period
            span_b: Span B period
            displacement: Cloud displacement

        Returns:
            Dictionary with Ichimoku components
        """
        # Conversion Line (Tenkan-sen)
        conv_high = high.rolling(window=conversion).max()
        conv_low = low.rolling(window=conversion).min()
        conversion_line = (conv_high + conv_low) / 2

        # Base Line (Kijun-sen)
        base_high = high.rolling(window=base).max()
        base_low = low.rolling(window=base).min()
        base_line = (base_high + base_low) / 2

        # Leading Span A (Senkou Span A)
        span_a = ((conversion_line + base_line) / 2).shift(displacement)

        # Leading Span B (Senkou Span B)
        span_b_high = high.rolling(window=span_b).max()
        span_b_low = low.rolling(window=span_b).min()
        span_b_line = ((span_b_high + span_b_low) / 2).shift(displacement)

        # Lagging Span (Chikou Span)
        lagging_span = close.shift(-displacement)

        return {
            "conversion_line": conversion_line,
            "base_line": base_line,
            "span_a": span_a,
            "span_b": span_b_line,
            "lagging_span": lagging_span,
        }
