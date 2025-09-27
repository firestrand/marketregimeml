"""Common feature calculation patterns.

Consolidates frequently used feature engineering patterns.
Following DRY principle to reduce code duplication.
"""

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd


class PriceFeatures:
    """Common price-based feature calculations."""

    @staticmethod
    def calculate_returns(
        prices: pd.Series,
        method: str = "simple",
        periods: int = 1,
    ) -> pd.Series:
        """Calculate returns from prices.

        Parameters
        ----------
        prices : pd.Series
            Price series
        method : str
            'simple' or 'log' returns
        periods : int
            Number of periods for return calculation

        Returns
        -------
        pd.Series
            Returns series
        """
        if method == "simple":
            return prices.pct_change(periods)
        elif method == "log":
            return np.log(prices / prices.shift(periods))
        else:
            raise ValueError(f"Unknown return method: {method}")

    @staticmethod
    def calculate_volatility(
        returns: pd.Series,
        window: int = 20,
        annualize: bool = False,
        trading_days: int = 252,
    ) -> pd.Series:
        """Calculate rolling volatility.

        Parameters
        ----------
        returns : pd.Series
            Returns series
        window : int
            Window size
        annualize : bool
            Whether to annualize
        trading_days : int
            Trading days per year

        Returns
        -------
        pd.Series
            Volatility series
        """
        vol = returns.rolling(window=window).std()
        if annualize:
            vol = vol * np.sqrt(trading_days)
        return vol

    @staticmethod
    def calculate_price_momentum(
        prices: pd.Series,
        short_window: int = 10,
        long_window: int = 30,
    ) -> pd.Series:
        """Calculate price momentum.

        Parameters
        ----------
        prices : pd.Series
            Price series
        short_window : int
            Short moving average window
        long_window : int
            Long moving average window

        Returns
        -------
        pd.Series
            Momentum indicator
        """
        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()
        return (short_ma - long_ma) / long_ma


class StatisticalFeatures:
    """Common statistical feature calculations."""

    @staticmethod
    def calculate_rolling_zscore(
        data: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Calculate rolling z-score.

        Parameters
        ----------
        data : pd.Series
            Data series
        window : int
            Window size

        Returns
        -------
        pd.Series
            Z-score series
        """
        mean = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        return (data - mean) / std.replace(0, np.nan)

    @staticmethod
    def calculate_rolling_quantile(
        data: pd.Series,
        window: int = 20,
        quantile: float = 0.5,
    ) -> pd.Series:
        """Calculate rolling quantile.

        Parameters
        ----------
        data : pd.Series
            Data series
        window : int
            Window size
        quantile : float
            Quantile to calculate (0-1)

        Returns
        -------
        pd.Series
            Quantile series
        """
        return data.rolling(window=window).quantile(quantile)

    @staticmethod
    def calculate_rolling_skewness(
        data: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Calculate rolling skewness.

        Parameters
        ----------
        data : pd.Series
            Data series
        window : int
            Window size

        Returns
        -------
        pd.Series
            Skewness series
        """
        return data.rolling(window=window).skew()

    @staticmethod
    def calculate_rolling_kurtosis(
        data: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Calculate rolling kurtosis.

        Parameters
        ----------
        data : pd.Series
            Data series
        window : int
            Window size

        Returns
        -------
        pd.Series
            Kurtosis series
        """
        return data.rolling(window=window).kurt()

    @staticmethod
    def calculate_autocorrelation(
        data: pd.Series,
        lag: int = 1,
        window: Optional[int] = None,
    ) -> Union[float, pd.Series]:
        """Calculate autocorrelation.

        Parameters
        ----------
        data : pd.Series
            Data series
        lag : int
            Lag for autocorrelation
        window : int, optional
            If provided, calculates rolling autocorrelation

        Returns
        -------
        float or pd.Series
            Autocorrelation value(s)
        """
        if window is None:
            return data.autocorr(lag=lag)
        else:
            return data.rolling(window=window).apply(
                lambda x: pd.Series(x).autocorr(lag=lag), raw=False
            )


class RegimeFeatures:
    """Common regime-related feature calculations."""

    @staticmethod
    def calculate_regime_duration(
        regimes: np.ndarray,
    ) -> pd.Series:
        """Calculate duration in current regime.

        Parameters
        ----------
        regimes : np.ndarray
            Regime sequence

        Returns
        -------
        pd.Series
            Duration in current regime
        """
        duration = np.zeros_like(regimes, dtype=float)
        current_regime = regimes[0]
        current_duration = 1

        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                current_regime = regimes[i]
                current_duration = 1
            duration[i] = current_duration

        return pd.Series(duration)

    @staticmethod
    def calculate_regime_frequency(
        regimes: np.ndarray,
        window: int = 50,
    ) -> pd.DataFrame:
        """Calculate rolling regime frequencies.

        Parameters
        ----------
        regimes : np.ndarray
            Regime sequence
        window : int
            Window size

        Returns
        -------
        pd.DataFrame
            Regime frequencies
        """
        unique_regimes = np.unique(regimes)
        frequencies = pd.DataFrame()

        for regime in unique_regimes:
            regime_indicator = (regimes == regime).astype(float)
            frequencies[f"regime_{regime}_freq"] = (
                pd.Series(regime_indicator).rolling(window=window, min_periods=1).mean()
            )

        return frequencies

    @staticmethod
    def calculate_transition_features(
        regimes: np.ndarray,
        window: int = 50,
    ) -> pd.DataFrame:
        """Calculate regime transition features.

        Parameters
        ----------
        regimes : np.ndarray
            Regime sequence
        window : int
            Window size

        Returns
        -------
        pd.DataFrame
            Transition features
        """
        features = pd.DataFrame()

        # Transition indicator
        transitions = np.diff(regimes, prepend=regimes[0]) != 0
        features["transition"] = transitions.astype(float)

        # Rolling transition rate
        features["transition_rate"] = (
            features["transition"].rolling(window=window, min_periods=1).mean()
        )

        # Time since last transition
        time_since_transition = np.zeros_like(regimes, dtype=float)
        last_transition = 0
        for i in range(len(regimes)):
            if i > 0 and regimes[i] != regimes[i - 1]:
                last_transition = i
            time_since_transition[i] = i - last_transition

        features["time_since_transition"] = time_since_transition

        return features


class TechnicalFeatures:
    """Common technical indicator calculations."""

    @staticmethod
    def calculate_rsi(
        prices: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Calculate Relative Strength Index.

        Parameters
        ----------
        prices : pd.Series
            Price series
        window : int
            Window size

        Returns
        -------
        pd.Series
            RSI values
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands.

        Parameters
        ----------
        prices : pd.Series
            Price series
        window : int
            Window size
        num_std : float
            Number of standard deviations

        Returns
        -------
        tuple
            (middle_band, upper_band, lower_band)
        """
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return middle, upper, lower

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator.

        Parameters
        ----------
        prices : pd.Series
            Price series
        fast_period : int
            Fast EMA period
        slow_period : int
            Slow EMA period
        signal_period : int
            Signal EMA period

        Returns
        -------
        tuple
            (macd_line, signal_line, histogram)
        """
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Calculate Average True Range.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        window : int
            Window size

        Returns
        -------
        pd.Series
            ATR values
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr


def combine_features(*feature_dfs: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple feature DataFrames.

    Parameters
    ----------
    *feature_dfs : pd.DataFrame
        Feature DataFrames to combine

    Returns
    -------
    pd.DataFrame
        Combined features
    """
    return pd.concat(feature_dfs, axis=1)


def select_non_correlated_features(
    features: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Select features with correlation below threshold.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix
    threshold : float
        Correlation threshold

    Returns
    -------
    pd.DataFrame
        Selected features
    """
    corr_matrix = features.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [
        column for column in upper_tri.columns if any(upper_tri[column] > threshold)
    ]
    return features.drop(columns=to_drop)
