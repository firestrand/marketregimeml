"""Main feature engineering engine."""

from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

from marketregimeml.features.volatility import VolatilityFeatures
from marketregimeml.features.entropy import EntropyFeatures
from marketregimeml.features.statistical import StatisticalFeatures
from marketregimeml.features.technical import TechnicalIndicators
from marketregimeml.utils.logging import get_logger
from marketregimeml.config_loader import config_loader

logger = get_logger(__name__)


class FeatureEngine:
    """Main feature engineering engine for market data.

    Orchestrates computation of various feature types including
    volatility, entropy, statistical, and technical indicators.
    """

    def __init__(
        self,
        feature_config: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True,
    ):
        """Initialize feature engine.

        Args:
            feature_config: Feature configuration dictionary
            enable_caching: Whether to cache computed features
        """
        # Load configuration
        if feature_config is None:
            try:
                feature_config = config_loader.get_feature_config()
            except Exception:
                # Default configuration if config loader fails
                feature_config = self._get_default_config()

        self.config = feature_config

        # Initialize feature calculators
        self.volatility = VolatilityFeatures()
        self.entropy = EntropyFeatures()
        self.statistical = StatisticalFeatures()
        self.technical = TechnicalIndicators()

        # Feature cache
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None

        # Scalers for normalization
        self.scalers = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature configuration."""
        return {
            "volatility": {
                "enabled": True,
                "indicators": ["yang_zhang", "garman_klass"],
                "window": 20,
            },
            "entropy": {
                "enabled": True,
                "measures": ["shannon", "approximate"],
                "window": 30,
            },
            "statistical": {
                "enabled": True,
                "features": ["mean", "std", "skew"],
                "window": 20,
            },
            "technical": {
                "enabled": True,
                "indicators": ["rsi", "macd"],
                "params": {},
            },
        }

    def compute_features(
        self,
        data: pd.DataFrame,
        feature_types: Optional[List[str]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute selected features from market data.

        Args:
            data: OHLCV market data
            feature_types: List of feature types to compute
            use_cache: Whether to use cached features
            **kwargs: Additional parameters

        Returns:
            DataFrame with computed features
        """
        # Validate input data
        if not self._validate_data(data):
            raise ValueError("Invalid input data")

        # Default to all feature types
        if feature_types is None:
            feature_types = [
                "volatility",
                "entropy",
                "statistical",
                "technical",
            ]

        # Check cache
        cached_result = self._get_cached_features(
            data, feature_types, use_cache
        )
        if cached_result is not None:
            return cached_result

        # Compute features
        features = self._compute_selected_features(data, feature_types, kwargs)

        # Cache and return
        self._cache_features_if_enabled(
            data, feature_types, features, use_cache
        )
        return features

    def _get_cached_features(
        self, data: pd.DataFrame, feature_types: List[str], use_cache: bool
    ) -> Optional[pd.DataFrame]:
        """Get cached features if available."""
        if not (use_cache and self.enable_caching):
            return None

        cache_key = self._get_cache_key(data, feature_types)
        if cache_key in self.cache:
            logger.info("Using cached features")
            return self.cache[cache_key]

        return None

    def _compute_selected_features(
        self, data: pd.DataFrame, feature_types: List[str], kwargs: dict
    ) -> pd.DataFrame:
        """Compute all selected feature types."""
        features = data.copy()

        # Feature computation mapping
        feature_computers = {
            "volatility": self._compute_volatility_features,
            "entropy": self._compute_entropy_features,
            "statistical": self._compute_statistical_features,
            "technical": self._compute_technical_features,
        }

        for feature_type in feature_types:
            if self._is_feature_type_enabled(feature_type):
                computed_features = feature_computers.get(feature_type)
                if computed_features:
                    new_features = computed_features(data, **kwargs)
                    features = pd.concat([features, new_features], axis=1)

        return features

    def _is_feature_type_enabled(self, feature_type: str) -> bool:
        """Check if a feature type is enabled in config."""
        return self.config.get(feature_type, {}).get("enabled", True)

    def _cache_features_if_enabled(
        self,
        data: pd.DataFrame,
        feature_types: List[str],
        features: pd.DataFrame,
        use_cache: bool,
    ) -> None:
        """Cache computed features if caching is enabled."""
        if use_cache and self.enable_caching:
            cache_key = self._get_cache_key(data, feature_types)
            self.cache[cache_key] = features

    def _validate_data(self, data: Any) -> bool:
        """Validate input data.

        Args:
            data: Input data to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, pd.DataFrame):
            return False

        # Check for required columns
        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                return False

        # Check for sufficient data
        if len(data) < 2:
            logger.error("Insufficient data points")
            return False

        return True

    def _get_cache_key(
        self, data: pd.DataFrame, feature_types: List[str]
    ) -> str:
        """Generate cache key for feature set.

        Args:
            data: Input data
            feature_types: Feature types

        Returns:
            Cache key string
        """
        # Create hash from data shape and feature types
        key_data = {
            "shape": data.shape,
            "columns": list(data.columns),
            "index_range": (str(data.index[0]), str(data.index[-1])),
            "feature_types": sorted(feature_types),
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    def _compute_volatility_features(
        self, data: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Compute volatility features.

        Args:
            data: OHLCV data
            **kwargs: Additional parameters

        Returns:
            DataFrame with volatility features
        """
        vol_config = self.config.get("volatility", {})
        indicators = vol_config.get("indicators", ["yang_zhang"])
        window = kwargs.get("window", vol_config.get("window", 20))

        vol_features = pd.DataFrame(index=data.index)

        for indicator in indicators:
            if indicator == "yang_zhang":
                vol = self.volatility.yang_zhang(
                    data["open"],
                    data["high"],
                    data["low"],
                    data["close"],
                    window,
                )
                vol_features[f"yang_zhang_{window}"] = vol

            elif indicator == "garman_klass":
                vol = self.volatility.garman_klass(
                    data["high"], data["low"], data["close"], window
                )
                vol_features[f"garman_klass_{window}"] = vol

            elif indicator == "parkinson":
                vol = self.volatility.parkinson(
                    data["high"], data["low"], window
                )
                vol_features[f"parkinson_{window}"] = vol

            elif indicator == "rogers_satchell":
                vol = self.volatility.rogers_satchell(
                    data["open"],
                    data["high"],
                    data["low"],
                    data["close"],
                    window,
                )
                vol_features[f"rogers_satchell_{window}"] = vol

        return vol_features

    def _compute_entropy_features(
        self, data: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Compute entropy features.

        Args:
            data: OHLCV data
            **kwargs: Additional parameters

        Returns:
            DataFrame with entropy features
        """
        entropy_config = self.config.get("entropy", {})
        measures = entropy_config.get("measures", ["shannon"])
        window = kwargs.get("window", entropy_config.get("window", 30))

        entropy_features = pd.DataFrame(index=data.index)

        for measure in measures:
            if measure == "shannon":
                entropy = self.entropy.shannon_entropy(data["close"], window)
                entropy_features[f"shannon_entropy_{window}"] = entropy

            elif measure == "approximate":
                entropy = self.entropy.approximate_entropy(
                    data["close"], window
                )
                entropy_features[f"approximate_entropy_{window}"] = entropy

            elif measure == "sample":
                entropy = self.entropy.sample_entropy(data["close"], window)
                entropy_features[f"sample_entropy_{window}"] = entropy

            elif measure == "permutation":
                entropy = self.entropy.permutation_entropy(
                    data["close"], window
                )
                entropy_features[f"permutation_entropy_{window}"] = entropy

        return entropy_features

    def _compute_statistical_features(
        self, data: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Compute statistical features.

        Args:
            data: OHLCV data
            **kwargs: Additional parameters

        Returns:
            DataFrame with statistical features
        """
        stat_config = self.config.get("statistical", {})
        features = stat_config.get("features", ["mean", "std"])
        window = kwargs.get("window", stat_config.get("window", 20))

        stat_features = pd.DataFrame(index=data.index)

        # Use close prices for statistical features
        close_prices = data["close"]

        for feature in features:
            if feature == "mean":
                stat_features[f"mean_{window}"] = (
                    self.statistical.rolling_mean(close_prices, window)
                )

            elif feature == "std":
                stat_features[f"std_{window}"] = self.statistical.rolling_std(
                    close_prices, window
                )

            elif feature == "skew":
                stat_features[f"skew_{window}"] = (
                    self.statistical.rolling_skewness(close_prices, window)
                )

            elif feature == "kurtosis":
                stat_features[f"kurtosis_{window}"] = (
                    self.statistical.rolling_kurtosis(close_prices, window)
                )

            elif feature == "min":
                stat_features[f"min_{window}"] = self.statistical.rolling_min(
                    close_prices, window
                )

            elif feature == "max":
                stat_features[f"max_{window}"] = self.statistical.rolling_max(
                    close_prices, window
                )

        return stat_features

    def _compute_technical_features(
        self, data: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Compute technical indicators.

        Args:
            data: OHLCV data
            **kwargs: Additional parameters

        Returns:
            DataFrame with technical indicators
        """
        tech_config = self.config.get("technical", {})
        indicators = tech_config.get("indicators", ["rsi"])
        params = tech_config.get("params", {})

        tech_features = pd.DataFrame(index=data.index)

        for indicator in indicators:
            if indicator == "rsi":
                window = params.get("rsi_window", 14)
                rsi = self.technical.rsi(data["close"], window)
                tech_features[f"rsi_{window}"] = rsi

            elif indicator == "macd":
                fast = params.get("macd_fast", 12)
                slow = params.get("macd_slow", 26)
                signal = params.get("macd_signal", 9)
                macd_line, signal_line, histogram = self.technical.macd(
                    data["close"], fast, slow, signal
                )
                tech_features["macd"] = macd_line
                tech_features["macd_signal"] = signal_line
                tech_features["macd_histogram"] = histogram

            elif indicator == "bollinger_bands":
                window = params.get("bb_window", 20)
                num_std = params.get("bb_std", 2)
                upper, middle, lower = self.technical.bollinger_bands(
                    data["close"], window, num_std
                )
                tech_features["bb_upper"] = upper
                tech_features["bb_middle"] = middle
                tech_features["bb_lower"] = lower

            elif indicator == "sma":
                window = params.get("sma_window", 20)
                sma = self.technical.sma(data["close"], window)
                tech_features[f"sma_{window}"] = sma

            elif indicator == "ema":
                window = params.get("ema_window", 20)
                ema = self.technical.ema(data["close"], window)
                tech_features[f"ema_{window}"] = ema

        return tech_features

    def _handle_missing_values(
        self, data: pd.DataFrame, method: str = "forward_fill", **kwargs
    ) -> pd.DataFrame:
        """Handle missing values in features.

        Args:
            data: Feature data
            method: Method to handle missing values
            **kwargs: Additional parameters

        Returns:
            DataFrame with missing values handled
        """
        if method == "forward_fill":
            return data.fillna(method="ffill")

        elif method == "backward_fill":
            return data.fillna(method="bfill")

        elif method == "interpolate":
            return data.interpolate(method="linear")

        elif method == "drop":
            return data.dropna()

        elif method == "mean":
            return data.fillna(data.mean())

        elif method == "median":
            return data.fillna(data.median())

        else:
            logger.warning(f"Unknown method {method}, using forward fill")
            return data.fillna(method="ffill")

    def normalize_features(
        self,
        features: pd.DataFrame,
        method: str = "standard",
        exclude_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Normalize features.

        Args:
            features: Feature DataFrame
            method: Normalization method
            exclude_cols: Columns to exclude from normalization

        Returns:
            Normalized features
        """
        if exclude_cols is None:
            exclude_cols = ["open", "high", "low", "close", "volume"]

        # Separate columns to normalize and preserve
        cols_to_normalize = [
            col for col in features.columns if col not in exclude_cols
        ]
        preserved_cols = [
            col for col in features.columns if col in exclude_cols
        ]

        if not cols_to_normalize:
            return features

        # Select scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            logger.warning(
                f"Unknown normalization method {method}, using standard"
            )
            scaler = StandardScaler()

        # Fit and transform
        normalized_data = scaler.fit_transform(features[cols_to_normalize])

        # Store scaler for potential inverse transform
        self.scalers[method] = scaler

        # Create result DataFrame
        result = pd.DataFrame(
            normalized_data, columns=cols_to_normalize, index=features.index
        )

        # Add back preserved columns
        for col in preserved_cols:
            result[col] = features[col]

        return result

    def select_features(
        self,
        features: pd.DataFrame,
        method: str = "variance",
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        target: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Select most important features.

        Args:
            features: Feature DataFrame
            method: Selection method
            k: Number of features to select
            threshold: Threshold for selection
            target: Target variable for supervised selection

        Returns:
            Selected features
        """
        # Separate preserved and feature columns
        preserve_cols, feature_cols = self._separate_columns(features)

        # Use strategy pattern for feature selection
        selection_params = {
            "features": features,
            "feature_cols": feature_cols,
            "k": k,
            "threshold": threshold,
            "target": target,
        }

        selected_cols = self._apply_selection_method(method, selection_params)

        # Return selected features plus preserved columns
        return features[preserve_cols + selected_cols]

    def _separate_columns(
        self, features: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Separate preserved OHLCV columns from feature columns."""
        preserve_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = [
            col for col in features.columns if col not in preserve_cols
        ]
        return preserve_cols, feature_cols

    def _apply_selection_method(
        self, method: str, params: Dict[str, Any]
    ) -> List[str]:
        """Apply the specified feature selection method."""
        selectors = {
            "variance": self._select_by_variance,
            "correlation": self._select_by_correlation,
            "kbest": self._select_kbest,
        }

        selector = selectors.get(method, self._select_top_variance)
        return selector(params)

    def _select_by_variance(self, params: Dict[str, Any]) -> List[str]:
        """Select features by variance threshold."""
        features = params["features"]
        feature_cols = params["feature_cols"]
        threshold = params.get("threshold") or 0.01

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(features[feature_cols].fillna(0))

        return [
            feature_cols[i]
            for i in range(len(feature_cols))
            if selector.get_support()[i]
        ]

    def _select_by_correlation(self, params: Dict[str, Any]) -> List[str]:
        """Remove highly correlated features."""
        features = params["features"]
        feature_cols = params["feature_cols"]
        threshold = params.get("threshold") or 0.9

        # Calculate correlation matrix
        corr_matrix = features[feature_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        return [col for col in feature_cols if col not in to_drop]

    def _select_kbest(self, params: Dict[str, Any]) -> List[str]:
        """Select k best features using statistical test."""
        features = params["features"]
        feature_cols = params["feature_cols"]
        target = params.get("target")
        k = params.get("k") or 10

        if target is None:
            # Fall back to variance selection if no target
            return self._select_top_variance(params)

        selector = SelectKBest(f_classif, k=k)
        selector.fit(features[feature_cols].fillna(0), target)

        return [
            feature_cols[i]
            for i in range(len(feature_cols))
            if selector.get_support()[i]
        ]

    def _select_top_variance(self, params: Dict[str, Any]) -> List[str]:
        """Select top k features by variance."""
        features = params["features"]
        feature_cols = params["feature_cols"]
        k = params.get("k") or 10

        variances = features[feature_cols].var()
        return variances.nlargest(k).index.tolist()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary of feature importance scores
        """
        # This would typically use a trained model
        # For now, return empty dict as placeholder
        return {}

    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate computed features.

        Args:
            features: Feature DataFrame

        Returns:
            Validation results
        """
        results = {
            "missing_values": {},
            "infinite_values": {},
            "constant_features": [],
            "highly_correlated": [],
        }

        # Check missing values
        missing = features.isnull().sum()
        results["missing_values"] = missing[missing > 0].to_dict()

        # Check infinite values
        inf_mask = np.isinf(features.select_dtypes(include=[np.number]))
        infinite = inf_mask.sum()
        results["infinite_values"] = infinite[infinite > 0].to_dict()

        # Check constant features
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if features[col].nunique() == 1:
                results["constant_features"].append(col)

        # Check highly correlated features
        corr_matrix = features[numeric_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        for col in upper.columns:
            high_corr = upper[col][upper[col] > 0.95]
            if not high_corr.empty:
                for idx in high_corr.index:
                    results["highly_correlated"].append(
                        (col, idx, high_corr[idx])
                    )

        return results

    def save_feature_config(self, filepath: str) -> None:
        """Save feature configuration to file.

        Args:
            filepath: Path to save configuration
        """
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Feature configuration saved to {filepath}")

    def clear_cache(self) -> None:
        """Clear feature cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Feature cache cleared")

    def compute_all_features(
        self, data: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Compute all available features.

        Args:
            data: OHLCV data
            **kwargs: Additional parameters

        Returns:
            DataFrame with all features
        """
        return self.compute_features(
            data,
            feature_types=[
                "volatility",
                "entropy",
                "statistical",
                "technical",
            ],
            **kwargs,
        )
