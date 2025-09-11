"""Feature computation engine following Single Responsibility Principle."""

from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

from marketregimeml.features.volatility import VolatilityFeatures
from marketregimeml.features.entropy import EntropyFeatures
from marketregimeml.features.statistical import StatisticalFeatures
from marketregimeml.features.technical import TechnicalIndicators
from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureComputer:
    """Computes features using specialized calculators.

    Single Responsibility: Orchestrates feature computation from different modules.
    """

    def __init__(self):
        """Initialize feature computer with all calculators."""
        self.volatility = VolatilityFeatures()
        self.entropy = EntropyFeatures()
        self.statistical = StatisticalFeatures()
        self.technical = TechnicalIndicators()
        logger.debug("FeatureComputer initialized with all calculators")

    def _get_volatility_calculator(self, indicator: str):
        """Get the volatility calculator function and required columns.

        Args:
            indicator: Name of volatility indicator

        Returns:
            Tuple of (calculator_function, required_columns)
        """
        calculators = {
            "yang_zhang": (
                lambda d, w: self.volatility.yang_zhang(
                    d["open"], d["high"], d["low"], d["close"], w
                ),
                ["open", "high", "low", "close"],
            ),
            "garman_klass": (
                lambda d, w: self.volatility.garman_klass(
                    d["high"], d["low"], d["close"], w
                ),
                ["high", "low", "close"],
            ),
            "parkinson": (
                lambda d, w: self.volatility.parkinson(d["high"], d["low"], w),
                ["high", "low"],
            ),
            "rogers_satchell": (
                lambda d, w: self.volatility.rogers_satchell(
                    d["open"], d["high"], d["low"], d["close"], w
                ),
                ["open", "high", "low", "close"],
            ),
            "close_to_close": (
                lambda d, w: self.volatility.close_to_close(d["close"], w),
                ["close"],
            ),
        }
        return calculators.get(indicator, (None, []))

    def _compute_single_volatility(
        self,
        indicator: str,
        data: pd.DataFrame,
        window: int,
        config: Dict[str, Any],
    ) -> Optional[pd.Series]:
        """Compute a single volatility indicator.

        Args:
            indicator: Indicator name
            data: Market data
            window: Rolling window size
            config: Configuration dictionary

        Returns:
            Computed feature or None if failed
        """
        try:
            if indicator == "garch":
                # Special handling for GARCH
                if "close" not in data.columns:
                    return None
                returns = data["close"].pct_change().dropna()
                garch_params = config.get("garch_params", {})
                return self.volatility.garch_volatility(
                    returns, **garch_params
                )

            # Get calculator and required columns
            calculator, required_cols = self._get_volatility_calculator(
                indicator
            )

            if calculator is None:
                logger.warning(f"Unknown volatility indicator: {indicator}")
                return None

            # Check if required columns are available
            if not all(col in data.columns for col in required_cols):
                logger.debug(
                    f"Missing columns for {indicator}: {required_cols}"
                )
                return None

            # Calculate the feature
            return calculator(data, window)

        except Exception as e:
            logger.warning(f"{indicator} volatility calculation failed: {e}")
            return None

    def compute_volatility_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute volatility-based features.

        Args:
            data: OHLCV market data
            config: Volatility feature configuration
            **kwargs: Additional parameters

        Returns:
            DataFrame with volatility features
        """
        if config is None:
            config = {
                "indicators": ["yang_zhang", "garman_klass", "close_to_close"],
                "window": kwargs.get("window", 20),
            }

        window = config.get("window", 20)
        indicators = config.get("indicators", ["close_to_close"])

        try:
            # Compute all requested indicators
            features = []
            for indicator in indicators:
                feature = self._compute_single_volatility(
                    indicator, data, window, config
                )
                if feature is not None:
                    features.append(feature)

            # Combine all features
            if features:
                result = pd.concat(features, axis=1)
                logger.debug(f"Computed {len(features)} volatility features")
                return result
            else:
                logger.warning("No volatility features could be computed")
                return pd.DataFrame(index=data.index)

        except Exception as e:
            logger.error(f"Error computing volatility features: {e}")
            return pd.DataFrame(index=data.index)

    def compute_entropy_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute entropy-based features.

        Args:
            data: Market data
            config: Entropy feature configuration
            **kwargs: Additional parameters

        Returns:
            DataFrame with entropy features
        """
        config = self._get_entropy_config(config, kwargs)

        # Extract data for entropy calculation
        returns_data = self._extract_returns_for_entropy(data)
        if returns_data is None:
            return pd.DataFrame(index=data.index)

        # Compute entropy features
        features = self._compute_entropy_measures(
            returns_data,
            config["measures"],
            config["window"],
            config.get("params", {}),
        )

        return self._combine_features(features, data.index)

    def _get_entropy_config(
        self, config: Optional[Dict[str, Any]], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get entropy configuration with defaults."""
        if config is None:
            return {
                "measures": ["shannon", "approximate", "sample"],
                "window": kwargs.get("window", 50),
            }
        return config

    def _extract_returns_for_entropy(
        self, data: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Extract returns data for entropy calculation."""
        if "close" in data.columns:
            return data["close"].pct_change().dropna()

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return data[numeric_cols[0]].pct_change().dropna()

        logger.warning("No numeric data found for entropy calculation")
        return None

    def _compute_entropy_measures(
        self,
        returns_data: pd.Series,
        measures: List[str],
        window: int,
        params: Dict[str, Any],
    ) -> List[pd.Series]:
        """Compute requested entropy measures."""
        features = []

        # Entropy calculator mapping
        calculators = {
            "shannon": lambda: self.entropy.shannon_entropy(
                returns_data, window, params.get("bins", 10)
            ),
            "approximate": lambda: self.entropy.approximate_entropy(
                returns_data, params.get("m", 2), params.get("r", 0.2)
            ),
            "sample": lambda: self.entropy.sample_entropy(
                returns_data, params.get("m", 2), params.get("r", 0.2)
            ),
            "permutation": lambda: self.entropy.permutation_entropy(
                returns_data, params.get("order", 3), params.get("delay", 1)
            ),
            "spectral": lambda: self.entropy.spectral_entropy(
                returns_data, window
            ),
        }

        for measure in measures:
            if measure in calculators:
                try:
                    features.append(calculators[measure]())
                except Exception as e:
                    logger.debug(f"Failed to compute {measure} entropy: {e}")

        return features

    def _combine_features(
        self, features: List[pd.Series], index: pd.Index
    ) -> pd.DataFrame:
        """Combine feature series into a DataFrame."""
        if not features:
            return pd.DataFrame(index=index)

        try:
            result = pd.concat(features, axis=1)
            result = result.reindex(index)
            return result.fillna(method="ffill").fillna(method="bfill")
        except Exception as e:
            logger.error(f"Error combining entropy features: {e}")
            return pd.DataFrame(index=index)

    def compute_statistical_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute statistical features.

        Args:
            data: Market data
            config: Statistical feature configuration
            **kwargs: Additional parameters

        Returns:
            DataFrame with statistical features
        """
        config = self._get_statistical_config(config, kwargs)

        # Extract returns data
        returns_data = self._extract_returns_for_stats(data)
        if returns_data is None:
            return pd.DataFrame(index=data.index)

        # Compute requested features
        features = self._compute_statistical_measures(
            returns_data, config["features"], config["window"], config
        )

        return self._combine_features(features, data.index)

    def _get_statistical_config(
        self, config: Optional[Dict[str, Any]], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get statistical configuration with defaults."""
        if config is None:
            return {
                "features": ["rolling_stats", "autocorrelation", "hurst"],
                "window": kwargs.get("window", 20),
            }
        return config

    def _extract_returns_for_stats(
        self, data: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Extract returns data for statistical calculation."""
        if "close" in data.columns:
            return data["close"].pct_change().dropna()

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return data[numeric_cols[0]].pct_change().dropna()

        logger.warning("No numeric data found for statistical calculation")
        return None

    def _compute_statistical_measures(
        self,
        returns_data: pd.Series,
        feature_types: List[str],
        window: int,
        config: Dict[str, Any],
    ) -> List[pd.Series]:
        """Compute requested statistical measures."""
        features = []

        # Statistical calculator mapping
        calculators = {
            "rolling_stats": lambda: self._compute_rolling_stats(
                returns_data, window
            ),
            "autocorrelation": lambda: self._compute_autocorrelations(
                returns_data, window, config.get("autocorr_lags", [1, 5, 10])
            ),
            "hurst": lambda: [
                self.statistical.hurst_exponent(
                    returns_data, config.get("hurst_window", 100)
                )
            ],
            "jarque_bera": lambda: [
                self.statistical.jarque_bera_stat(returns_data, window)
            ],
        }

        for feature_type in feature_types:
            if feature_type in calculators:
                try:
                    result = calculators[feature_type]()
                    features.extend(
                        result if isinstance(result, list) else [result]
                    )
                except Exception as e:
                    logger.debug(f"Failed to compute {feature_type}: {e}")

        return features

    def _compute_rolling_stats(
        self, returns_data: pd.Series, window: int
    ) -> List[pd.Series]:
        """Compute rolling statistics."""
        rolling_stats = self.statistical.rolling_statistics(
            returns_data, window
        )
        return [rolling_stats[col] for col in rolling_stats.columns]

    def _compute_autocorrelations(
        self, returns_data: pd.Series, window: int, lags: List[int]
    ) -> List[pd.Series]:
        """Compute autocorrelations for multiple lags."""
        return [
            self.statistical.autocorrelation(returns_data, window, lag)
            for lag in lags
        ]

    def _get_technical_calculator(self, indicator: str):
        """Get the technical indicator calculator and required columns.

        Args:
            indicator: Name of technical indicator

        Returns:
            Tuple of (calculator_function, required_columns, param_extractor)
        """
        calculators = {
            "rsi": (
                lambda d, p: self.technical.rsi(
                    d["close"], p.get("rsi_period", 14)
                ),
                ["close"],
                lambda p: {"rsi_period": p.get("rsi_period", 14)},
            ),
            "macd": (
                lambda d, p: self.technical.macd(
                    d["close"],
                    p.get("macd_fast", 12),
                    p.get("macd_slow", 26),
                    p.get("macd_signal", 9),
                ),
                ["close"],
                lambda p: {
                    "macd_fast": p.get("macd_fast", 12),
                    "macd_slow": p.get("macd_slow", 26),
                    "macd_signal": p.get("macd_signal", 9),
                },
            ),
            "bollinger": (
                lambda d, p: self.technical.bollinger_bands(
                    d["close"], p.get("bb_period", 20), p.get("bb_std", 2.0)
                ),
                ["close"],
                lambda p: {
                    "bb_period": p.get("bb_period", 20),
                    "bb_std": p.get("bb_std", 2.0),
                },
            ),
            "stochastic": (
                lambda d, p: self.technical.stochastic(
                    d["high"], d["low"], d["close"], p.get("stoch_period", 14)
                ),
                ["high", "low", "close"],
                lambda p: {"stoch_period": p.get("stoch_period", 14)},
            ),
            "atr": (
                lambda d, p: self.technical.atr(
                    d["high"], d["low"], d["close"], p.get("atr_period", 14)
                ),
                ["high", "low", "close"],
                lambda p: {"atr_period": p.get("atr_period", 14)},
            ),
        }
        return calculators.get(indicator, (None, [], lambda p: {}))

    def _compute_single_technical(
        self, indicator: str, data: pd.DataFrame, params: Dict[str, Any]
    ) -> List[pd.Series]:
        """Compute a single technical indicator.

        Args:
            indicator: Indicator name
            data: Market data
            params: Parameters dictionary

        Returns:
            List of computed features (some indicators return multiple)
        """
        try:
            calculator, required_cols, param_extractor = (
                self._get_technical_calculator(indicator)
            )

            if calculator is None:
                logger.warning(f"Unknown technical indicator: {indicator}")
                return []

            # Check if required columns are available
            if not all(col in data.columns for col in required_cols):
                logger.debug(
                    f"Missing columns for {indicator}: {required_cols}"
                )
                return []

            # Extract parameters
            indicator_params = param_extractor(params)

            # Calculate the feature
            result = calculator(data, indicator_params)

            # Handle indicators that return multiple values
            if isinstance(result, tuple):
                return list(result)
            else:
                return [result]

        except Exception as e:
            logger.warning(f"{indicator} calculation failed: {e}")
            return []

    def compute_technical_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute technical indicator features.

        Args:
            data: OHLCV market data
            config: Technical feature configuration
            **kwargs: Additional parameters

        Returns:
            DataFrame with technical features
        """
        if config is None:
            config = {"indicators": ["rsi", "macd", "bollinger"], "params": {}}

        indicators = config.get("indicators", ["rsi"])
        params = config.get("params", {})

        try:
            # Compute all requested indicators
            features = []
            for indicator in indicators:
                indicator_features = self._compute_single_technical(
                    indicator, data, params
                )
                features.extend(indicator_features)

            # Combine all features
            if features:
                result = pd.concat(features, axis=1)
                logger.debug(f"Computed {len(features)} technical features")
                return result
            else:
                logger.warning("No technical features could be computed")
                return pd.DataFrame(index=data.index)

        except Exception as e:
            logger.error(f"Error computing technical features: {e}")
            return pd.DataFrame(index=data.index)

    def compute_all_features(
        self,
        data: pd.DataFrame,
        feature_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute all available features.

        Args:
            data: Market data
            feature_config: Complete feature configuration
            **kwargs: Additional parameters

        Returns:
            DataFrame with all computed features
        """
        feature_config = self._get_default_feature_config(feature_config)
        all_features = self._compute_enabled_features(
            data, feature_config, kwargs
        )
        return self._combine_all_features(all_features, data.index)

    def _get_default_feature_config(
        self, config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get default feature configuration if not provided."""
        if config is None:
            return {
                "volatility": {"enabled": True},
                "entropy": {"enabled": True},
                "statistical": {"enabled": True},
                "technical": {"enabled": True},
            }
        return config

    def _compute_enabled_features(
        self,
        data: pd.DataFrame,
        feature_config: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> List[pd.DataFrame]:
        """Compute all enabled feature types."""
        all_features = []

        # Feature computation mapping
        feature_computers = {
            "volatility": self.compute_volatility_features,
            "entropy": self.compute_entropy_features,
            "statistical": self.compute_statistical_features,
            "technical": self.compute_technical_features,
        }

        for feature_type, compute_func in feature_computers.items():
            if self._is_feature_enabled(feature_config, feature_type):
                features = self._safe_compute_features(
                    compute_func,
                    data,
                    feature_config.get(feature_type),
                    feature_type,
                    **kwargs,
                )
                if features is not None and not features.empty:
                    all_features.append(features)

        return all_features

    def _is_feature_enabled(
        self, config: Dict[str, Any], feature_type: str
    ) -> bool:
        """Check if a feature type is enabled."""
        return config.get(feature_type, {}).get("enabled", True)

    def _safe_compute_features(
        self,
        compute_func,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]],
        feature_type: str,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Safely compute features with error handling."""
        try:
            return compute_func(data, config, **kwargs)
        except Exception as e:
            logger.warning(
                f"{feature_type.capitalize()} features computation failed: {e}"
            )
            return None

    def _combine_all_features(
        self, feature_list: List[pd.DataFrame], index: pd.Index
    ) -> pd.DataFrame:
        """Combine all computed features into a single DataFrame."""
        if not feature_list:
            logger.warning("No features could be computed")
            return pd.DataFrame(index=index)

        try:
            result = pd.concat(feature_list, axis=1)
            logger.info(f"Computed {len(result.columns)} total features")
            return result
        except Exception as e:
            logger.error(f"Error combining features: {e}")
            return pd.DataFrame(index=index)

    def _get_required_columns(
        self, feature_type: str, indicators: List[str]
    ) -> List[str]:
        """Get required columns for specific feature types and indicators."""
        requirements = {
            "volatility": {
                "yang_zhang": ["open", "high", "low", "close"],
                "garman_klass": ["high", "low", "close"],
                "parkinson": ["high", "low"],
                "rogers_satchell": ["open", "high", "low", "close"],
                "close_to_close": ["close"],
                "garch": ["close"],
            },
            "technical": {
                "rsi": ["close"],
                "macd": ["close"],
                "bollinger": ["close"],
                "stochastic": ["high", "low", "close"],
                "atr": ["high", "low", "close"],
            },
        }

        required = set()
        for indicator in indicators:
            if (
                feature_type in requirements
                and indicator in requirements[feature_type]
            ):
                required.update(requirements[feature_type][indicator])

        return list(required)

    def _check_available_columns(
        self, data: pd.DataFrame, required: List[str]
    ) -> List[str]:
        """Check which required columns are available in data."""
        return [col for col in required if col in data.columns]
