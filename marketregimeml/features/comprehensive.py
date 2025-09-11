"""Comprehensive feature engineering for regime detection."""

from typing import Optional, List
import numpy as np
import pandas as pd
from marketregimeml.features.technical import TechnicalIndicators
from marketregimeml.features.volatility import VolatilityFeatures
from marketregimeml.features.statistical import StatisticalFeatures
from marketregimeml.features.entropy import EntropyFeatures
from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class ComprehensiveFeatures:
    """Comprehensive feature engineering combining all feature types."""

    def __init__(self):
        """Initialize feature extractors."""
        self.technical = TechnicalIndicators()
        self.volatility = VolatilityFeatures()
        self.statistical = StatisticalFeatures()
        self.entropy = EntropyFeatures()

    def create_features(
        self,
        ohlcv: pd.DataFrame,
        feature_sets: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data.

        Args:
            ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            feature_sets: List of feature sets to include.
                         Options: ['price', 'returns', 'volatility', 'technical',
                                  'statistical', 'entropy', 'microstructure']
                         Default: all
            windows: List of window sizes for rolling features
                    Default: [5, 10, 20, 50]

        Returns:
            DataFrame with comprehensive features
        """
        if len(ohlcv) == 0:
            return pd.DataFrame()

        if feature_sets is None:
            feature_sets = [
                "price",
                "returns",
                "volatility",
                "technical",
                "statistical",
                "entropy",
                "microstructure",
            ]

        if windows is None:
            windows = [5, 10, 20, 50]

        features = pd.DataFrame(index=ohlcv.index)

        # Add each feature set using dedicated methods (KISS principle)
        if "price" in feature_sets:
            features = self._add_price_features(features, ohlcv, windows)

        if "returns" in feature_sets:
            features = self._add_returns_features(features, ohlcv, windows)

        if "volatility" in feature_sets:
            features = self._add_volatility_features(features, ohlcv, windows)

        if "technical" in feature_sets:
            features = self._add_technical_features(features, ohlcv)

        if "statistical" in feature_sets:
            features = self._add_statistical_features(features, ohlcv)

        if "entropy" in feature_sets:
            features = self._add_entropy_features(features, ohlcv)

        if "microstructure" in feature_sets and "volume" in ohlcv.columns:
            features = self._add_microstructure_features(features, ohlcv)

        # Clean up features
        features = self._clean_features(features)

        logger.info(f"Created {len(features.columns)} features")

        return features

    def _add_price_features(
        self, features: pd.DataFrame, ohlcv: pd.DataFrame, windows: List[int]
    ) -> pd.DataFrame:
        """Add price-based features (KISS - single responsibility)."""
        logger.debug("Creating price features")

        features["price_momentum_5"] = ohlcv["close"].pct_change(5)
        features["price_momentum_20"] = ohlcv["close"].pct_change(20)

        # Price position in range
        price_min = ohlcv["close"].rolling(20).min()
        price_max = ohlcv["close"].rolling(20).max()
        features["price_position"] = (ohlcv["close"] - price_min) / (
            price_max - price_min + 1e-10
        )

        # Price relative to moving averages
        for window in windows:
            ma = ohlcv["close"].rolling(window).mean()
            features[f"price_to_ma_{window}"] = (
                ohlcv["close"] / (ma + 1e-10) - 1
            )

        return features

    def _add_returns_features(
        self, features: pd.DataFrame, ohlcv: pd.DataFrame, windows: List[int]
    ) -> pd.DataFrame:
        """Add returns-based features (KISS - single responsibility)."""
        logger.debug("Creating returns features")

        # Basic returns
        features["returns"] = ohlcv["close"].pct_change()
        features["log_returns"] = np.log(
            ohlcv["close"] / ohlcv["close"].shift(1)
        )
        features["abs_returns"] = np.abs(features["returns"])
        features["squared_returns"] = features["returns"] ** 2

        # Rolling statistics
        for window in windows:
            features[f"returns_mean_{window}"] = (
                features["returns"].rolling(window).mean()
            )
            features[f"returns_std_{window}"] = (
                features["returns"].rolling(window).std()
            )
            features[f"returns_skew_{window}"] = (
                features["returns"].rolling(window).skew()
            )
            features[f"returns_kurt_{window}"] = (
                features["returns"].rolling(window).kurt()
            )

        return features

    def _add_volatility_features(
        self, features: pd.DataFrame, ohlcv: pd.DataFrame, windows: List[int]
    ) -> pd.DataFrame:
        """Add volatility features (KISS - single responsibility)."""
        logger.debug("Creating volatility features")

        # Various volatility measures
        for window in [10, 20]:
            features[f"parkinson_{window}"] = self.volatility.parkinson(
                ohlcv["high"], ohlcv["low"], window=window
            )
            features[f"garman_klass_{window}"] = self.volatility.garman_klass(
                ohlcv["high"], ohlcv["low"], ohlcv["close"], window=window
            )
            features[f"yang_zhang_{window}"] = self.volatility.yang_zhang(
                ohlcv["open"],
                ohlcv["high"],
                ohlcv["low"],
                ohlcv["close"],
                window=window,
            )

        # Volatility ratios (only if returns features exist)
        if "returns_std_5" in features and "returns_std_20" in features:
            features["vol_ratio_5_20"] = features["returns_std_5"] / (
                features["returns_std_20"] + 1e-10
            )

        if "returns_std_10" in features and "returns_std_50" in features:
            features["vol_ratio_10_50"] = features["returns_std_10"] / (
                features["returns_std_50"] + 1e-10
            )

        return features

    def _add_technical_features(
        self, features: pd.DataFrame, ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """Add technical indicator features (KISS - single responsibility)."""
        logger.debug("Creating technical features")

        # RSI at multiple timeframes
        for period in [7, 14, 21]:
            features[f"rsi_{period}"] = self.technical.rsi(
                ohlcv["close"], period=period
            )

        # MACD
        macd, signal, histogram = self.technical.macd(ohlcv["close"])
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_histogram"] = histogram

        # Bollinger Bands
        for period in [10, 20]:
            upper, middle, lower = self.technical.bollinger_bands(
                ohlcv["close"], period=period
            )
            features[f"bb_position_{period}"] = (ohlcv["close"] - lower) / (
                upper - lower + 1e-10
            )
            features[f"bb_width_{period}"] = (upper - lower) / (middle + 1e-10)

        # Stochastic
        k, d = self.technical.stochastic(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        features["stoch_k"] = k
        features["stoch_d"] = d

        # ATR
        features["atr_14"] = self.technical.atr(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )

        return features

    def _add_statistical_features(
        self, features: pd.DataFrame, ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """Add statistical features (KISS - single responsibility)."""
        logger.debug("Creating statistical features")

        if "returns" not in features:
            features["returns"] = ohlcv["close"].pct_change()

        # Autocorrelation
        for lag in [1, 5, 10]:
            features[f"autocorr_lag_{lag}"] = (
                features["returns"]
                .rolling(20)
                .apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
                )
            )

        # Hurst exponent
        features["hurst_20"] = (
            features["returns"]
            .rolling(20)
            .apply(
                lambda x: self._calculate_hurst(x) if len(x) == 20 else np.nan
            )
        )

        # Jarque-Bera test statistic
        features["jarque_bera_20"] = (
            features["returns"]
            .rolling(20)
            .apply(
                lambda x: self._jarque_bera_stat(x) if len(x) == 20 else np.nan
            )
        )

        return features

    def _add_entropy_features(
        self, features: pd.DataFrame, ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """Add entropy features (KISS - single responsibility)."""
        logger.debug("Creating entropy features")

        # Various entropy measures - use rolling methods with small windows
        features["sample_entropy"] = (
            self.entropy.sample_entropy(ohlcv["close"], window=20).iloc[-1]
            if len(ohlcv) >= 20
            else 0
        )
        features["approx_entropy"] = (
            self.entropy.approximate_entropy(ohlcv["close"], window=20).iloc[
                -1
            ]
            if len(ohlcv) >= 20
            else 0
        )
        features["permutation_entropy"] = (
            self.entropy.permutation_entropy(ohlcv["close"], window=20).iloc[
                -1
            ]
            if len(ohlcv) >= 20
            else 0
        )

        return features

    def _add_microstructure_features(
        self, features: pd.DataFrame, ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """Add market microstructure features (KISS - single responsibility)."""
        logger.debug("Creating microstructure features")

        # Volume features
        features["volume_ratio"] = ohlcv["volume"] / (
            ohlcv["volume"].rolling(20).mean() + 1e-10
        )
        features["volume_trend"] = ohlcv["volume"].rolling(5).mean() / (
            ohlcv["volume"].rolling(20).mean() + 1e-10
        )

        # Price efficiency
        features["price_efficiency"] = np.abs(
            ohlcv["close"] - ohlcv["open"]
        ) / (ohlcv["high"] - ohlcv["low"] + 1e-10)

        # Spreads
        features["hl_spread"] = (ohlcv["high"] - ohlcv["low"]) / (
            ohlcv["close"] + 1e-10
        )
        features["co_spread"] = (ohlcv["close"] - ohlcv["open"]) / (
            ohlcv["open"] + 1e-10
        )

        # Amihud illiquidity (if returns exist)
        if "returns" in features:
            features["amihud_illiquidity"] = np.abs(features["returns"]) / (
                ohlcv["volume"] + 1e-10
            )

        return features

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features (DRY - reusable cleaning logic)."""
        # Replace infinities with NaN
        features = features.replace([np.inf, -np.inf], np.nan)

        # Forward fill then fill remaining with 0
        features = features.ffill().fillna(0)

        return features

    def create_ml_optimized_features(
        self, ohlcv: pd.DataFrame, max_features: int = 50
    ) -> pd.DataFrame:
        """
        Create optimized feature set specifically for ML models.

        Focuses on features that work well with tree-based models and
        neural networks, avoiding highly correlated features.

        Args:
            ohlcv: OHLCV DataFrame
            max_features: Maximum number of features to return

        Returns:
            DataFrame with optimized features for ML
        """
        # Create comprehensive features
        all_features = self.create_features(
            ohlcv,
            feature_sets=[
                "returns",
                "volatility",
                "technical",
                "microstructure",
            ],
            windows=[5, 10, 20],  # Fewer windows to reduce feature count
        )

        # Select most important features based on variance and low correlation
        # This is a simplified version - in production you'd use feature selection
        feature_variance = all_features.var()
        top_features = feature_variance.nlargest(max_features).index

        selected_features = all_features[top_features]

        # Add interaction features for top features
        if len(selected_features.columns) < max_features:
            # Add some interaction terms
            if (
                "returns" in selected_features
                and "returns_std_10" in selected_features
            ):
                selected_features["returns_vol_interaction"] = (
                    selected_features["returns"]
                    * selected_features["returns_std_10"]
                )

            if "rsi_14" in selected_features:
                selected_features["rsi_extreme"] = (
                    (selected_features["rsi_14"] > 70)
                    | (selected_features["rsi_14"] < 30)
                ).astype(int)

        logger.info(
            f"Selected {len(selected_features.columns)} ML-optimized features"
        )

        return selected_features

    def create_regime_optimized_features(
        self, ohlcv: pd.DataFrame, n_regimes: int = 3
    ) -> pd.DataFrame:
        """
        Create features optimized for specific number of regimes.

        Args:
            ohlcv: OHLCV DataFrame
            n_regimes: Number of regimes to optimize for

        Returns:
            DataFrame with regime-optimized features
        """
        features = pd.DataFrame(index=ohlcv.index)

        # For 3 regimes: focus on clear bull/bear/sideways indicators
        if n_regimes == 3:
            # Trend indicators
            features["returns"] = ohlcv["close"].pct_change()
            features["ma_trend"] = (
                ohlcv["close"] / ohlcv["close"].rolling(20).mean() - 1
            )
            features["momentum"] = ohlcv["close"].pct_change(10)

            # Volatility for regime transitions
            features["volatility"] = features["returns"].rolling(10).std()
            features["vol_change"] = features["volatility"].pct_change(5)

            # Market structure
            features["rsi"] = self.technical.rsi(ohlcv["close"], period=14)

        # For 5-7 regimes: more granular features
        elif n_regimes >= 5:
            # Multiple timeframe features
            for window in [5, 10, 20, 50]:
                features[f"returns_{window}"] = ohlcv["close"].pct_change(
                    window
                )
                features[f"volatility_{window}"] = (
                    ohlcv["close"].pct_change().rolling(window).std()
                )

            # Detailed technical indicators
            features["rsi_7"] = self.technical.rsi(ohlcv["close"], period=7)
            features["rsi_14"] = self.technical.rsi(ohlcv["close"], period=14)
            features["rsi_21"] = self.technical.rsi(ohlcv["close"], period=21)

            # Multiple volatility measures
            features["parkinson"] = self.volatility.parkinson(
                ohlcv["high"], ohlcv["low"], window=10
            )
            features["garman_klass"] = self.volatility.garman_klass(
                ohlcv["high"], ohlcv["low"], ohlcv["close"], window=10
            )

            # Microstructure for granular regime detection
            if "volume" in ohlcv:
                features["volume_profile"] = (
                    ohlcv["volume"] / ohlcv["volume"].rolling(50).mean()
                )
                features["price_efficiency"] = np.abs(
                    ohlcv["close"] - ohlcv["open"]
                ) / (ohlcv["high"] - ohlcv["low"] + 1e-10)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0)

        logger.info(
            f"Created {len(features.columns)} features optimized for {n_regimes} regimes"
        )

        return features

    def _calculate_hurst(self, ts: pd.Series) -> float:
        """Simplified Hurst exponent calculation."""
        if len(ts) < 20:
            return 0.5
        try:
            lags = range(2, min(20, len(ts)))
            tau = [
                np.std(np.subtract(ts[lag:].values, ts[:-lag].values))
                for lag in lags
            ]
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]
        except Exception:
            return 0.5

    def _jarque_bera_stat(self, ts: pd.Series) -> float:
        """Calculate Jarque-Bera test statistic."""
        n = len(ts)
        if n < 4:
            return 0
        skew = ts.skew()
        kurt = ts.kurt()
        jb = (n / 6) * (skew**2 + 0.25 * (kurt - 3) ** 2)
        return jb
