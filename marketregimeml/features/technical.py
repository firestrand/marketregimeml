"""Technical indicators for market analysis."""

from typing import Tuple, Optional

import numpy as np
import pandas as pd

# Optional Numba acceleration: fall back to no-op if unavailable
try:  # pragma: no cover - optional dependency
    from numba import jit
except Exception:  # Fallback: define no-op decorator
    def jit(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        return wrapper

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)

# Export the existing class
__all__ = ["TechnicalIndicators"]


@jit(nopython=True, cache=True)
def _rsi_core(prices, period):
    """Numba-optimized RSI calculation."""
    n = len(prices)
    result = np.empty(n)
    result[:] = np.nan

    if n < period + 1:
        return result

    # Calculate price changes
    deltas = np.diff(prices)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Initial average gain/loss
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss != 0:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))
    else:
        result[period] = 100

    # Calculate RSI for remaining periods using EMA
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss != 0:
            rs = avg_gain / avg_loss
            result[i + 1] = 100 - (100 / (1 + rs))
        else:
            result[i + 1] = 100

    return result


@jit(nopython=True, cache=True)
def _atr_core(high, low, close, period):
    """Numba-optimized ATR calculation."""
    n = len(close)
    result = np.empty(n)
    result[:] = np.nan

    if n < period + 1:
        return result

    # Calculate True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Initial ATR
    result[period - 1] = np.mean(tr[:period])

    # Calculate ATR using EMA
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


class TechnicalIndicators:
    """Technical indicator calculators for market analysis."""

    def __init__(self):
        """Initialize technical indicator calculator."""
        logger.debug("TechnicalIndicators initialized")

    def rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index.

        RSI measures momentum, ranges from 0-100.
        Above 70: Overbought
        Below 30: Oversold

        Args:
            prices: Price series (typically close)
            period: RSI period

        Returns:
            RSI series
        """
        result = _rsi_core(prices.values, period)
        return pd.Series(result, index=prices.index, name=f"rsi_{period}")

    def atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate Average True Range.

        ATR measures volatility using full price range.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period

        Returns:
            ATR series
        """
        result = _atr_core(high.values, low.values, close.values, period)
        return pd.Series(result, index=close.index, name=f"atr_{period}")

    def bollinger_bands(
        self, prices: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands.

        Price bands based on moving average and standard deviation.

        Args:
            prices: Price series (typically close)
            period: MA period
            num_std: Number of standard deviations

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        upper.name = f"bb_upper_{period}"
        middle.name = f"bb_middle_{period}"
        lower.name = f"bb_lower_{period}"

        return upper, middle, lower

    def macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence).

        Trend-following momentum indicator.

        Args:
            prices: Price series (typically close)
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        macd_line.name = "macd"
        signal_line.name = "macd_signal"
        histogram.name = "macd_histogram"

        return macd_line, signal_line, histogram

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
            cum_vol_price = (
                (typical_price * volume).rolling(window=period).sum()
            )
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
        positive_flow = np.where(
            typical_price > typical_price.shift(1), money_flow, 0
        )
        negative_flow = np.where(
            typical_price < typical_price.shift(1), money_flow, 0
        )

        # Calculate money flow ratio
        positive_mf = (
            pd.Series(positive_flow, index=close.index)
            .rolling(window=period)
            .sum()
        )
        negative_mf = (
            pd.Series(negative_flow, index=close.index)
            .rolling(window=period)
            .sum()
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
        plus_di = (
            100 * pd.Series(plus_dm).rolling(window=period).mean() / atr_val
        )
        minus_di = (
            100 * pd.Series(minus_dm).rolling(window=period).mean() / atr_val
        )

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
