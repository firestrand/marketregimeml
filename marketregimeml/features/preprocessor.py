"""Feature preprocessing following Single Responsibility Principle."""

from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class FeaturePreprocessor:
    """Handles feature preprocessing including normalization and missing values.

    Single Responsibility: Feature data preprocessing and transformation.
    """

    def __init__(self):
        """Initialize feature preprocessor."""
        self.scalers = {}
        self.imputers = {}
        logger.debug("FeaturePreprocessor initialized")

    def _apply_imputation_strategy(
        self, features_clean: pd.DataFrame, strategy: str, **kwargs
    ) -> pd.DataFrame:
        """Apply a specific imputation strategy.

        Args:
            features_clean: DataFrame to apply imputation to
            strategy: Imputation strategy name
            **kwargs: Strategy-specific parameters

        Returns:
            DataFrame with imputation applied
        """
        # Strategy mapping to handler methods
        strategy_handlers = {
            "drop": self._handle_drop_strategy,
            "interpolate": self._handle_interpolate_strategy,
            "mean": lambda df, **kw: self._handle_sklearn_imputer(
                df, "mean", **kw
            ),
            "median": lambda df, **kw: self._handle_sklearn_imputer(
                df, "median", **kw
            ),
            "knn": self._handle_knn_imputer,
            "forward_fill": lambda df, **kw: df.fillna(method="ffill"),
            "backward_fill": lambda df, **kw: df.fillna(method="bfill"),
        }

        if strategy in strategy_handlers:
            return strategy_handlers[strategy](features_clean, **kwargs)
        else:
            logger.warning(
                f"Unknown missing value strategy: {strategy}, using interpolation"
            )
            return (
                features_clean.interpolate()
                .fillna(method="ffill")
                .fillna(method="bfill")
            )

    def _handle_drop_strategy(
        self, features_clean: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Handle drop strategy for missing values."""
        original_len = len(features_clean)
        features_clean = features_clean.dropna()
        logger.info(
            f"Dropped {original_len - len(features_clean)} rows with missing values"
        )
        return features_clean

    def _handle_interpolate_strategy(
        self, features_clean: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Handle interpolation strategy for missing values."""
        method = kwargs.get("method", "linear")
        limit = kwargs.get("limit", None)
        features_clean = features_clean.interpolate(method=method, limit=limit)
        features_clean = features_clean.fillna(method="ffill").fillna(
            method="bfill"
        )
        logger.debug(f"Applied interpolation with method: {method}")
        return features_clean

    def _handle_sklearn_imputer(
        self, features_clean: pd.DataFrame, strategy: str, **kwargs
    ) -> pd.DataFrame:
        """Handle sklearn SimpleImputer strategies."""
        imputer_key = f"{strategy}_imputer"
        numeric_cols = features_clean.select_dtypes(
            include=[np.number]
        ).columns

        if imputer_key not in self.imputers:
            self.imputers[imputer_key] = SimpleImputer(strategy=strategy)
            self.imputers[imputer_key].fit(features_clean[numeric_cols])

        features_clean[numeric_cols] = self.imputers[imputer_key].transform(
            features_clean[numeric_cols]
        )
        logger.debug(f"Applied {strategy} imputation")
        return features_clean

    def _handle_knn_imputer(
        self, features_clean: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Handle KNN imputation strategy."""
        n_neighbors = kwargs.get("n_neighbors", 5)
        imputer_key = f"knn_imputer_{n_neighbors}"
        numeric_cols = features_clean.select_dtypes(
            include=[np.number]
        ).columns

        if imputer_key not in self.imputers:
            self.imputers[imputer_key] = KNNImputer(n_neighbors=n_neighbors)
            self.imputers[imputer_key].fit(features_clean[numeric_cols])

        features_clean[numeric_cols] = self.imputers[imputer_key].transform(
            features_clean[numeric_cols]
        )
        logger.debug(f"Applied KNN imputation with {n_neighbors} neighbors")
        return features_clean

    def handle_missing_values(
        self, features: pd.DataFrame, strategy: str = "interpolate", **kwargs
    ) -> pd.DataFrame:
        """Handle missing values in feature data.

        Args:
            features: Features DataFrame with potential missing values
            strategy: Missing value handling strategy
            **kwargs: Additional parameters for specific strategies

        Returns:
            DataFrame with missing values handled
        """
        if features.empty:
            return features

        try:
            features_clean = features.copy()
            features_clean = self._apply_imputation_strategy(
                features_clean, strategy, **kwargs
            )

            # Log final missing value status
            remaining_nas = features_clean.isna().sum().sum()
            if remaining_nas > 0:
                logger.warning(
                    f"{remaining_nas} missing values remain after {strategy} strategy"
                )
            else:
                logger.debug("All missing values handled successfully")

            return features_clean

        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return features  # Return original features on error

    def _get_or_create_scaler(self, method: str, fit: bool, **kwargs):
        """Get or create a scaler for the given method.

        Args:
            method: Normalization method
            fit: Whether to create a new scaler
            **kwargs: Scaler parameters

        Returns:
            Tuple of (scaler_key, scaler_instance)
        """
        scaler_key = f"{method}_scaler"

        # Factory pattern for scaler creation
        scaler_factories = {
            "standard": lambda: StandardScaler(),
            "robust": lambda: RobustScaler(
                quantile_range=kwargs.get("quantile_range", (25.0, 75.0))
            ),
            "minmax": lambda: MinMaxScaler(
                feature_range=kwargs.get("feature_range", (0, 1))
            ),
        }

        if method not in scaler_factories:
            logger.warning(
                f"Unknown normalization method: {method}, using standard"
            )
            method = "standard"
            scaler_key = "standard_scaler"

        if scaler_key not in self.scalers or fit:
            self.scalers[scaler_key] = scaler_factories.get(
                method, scaler_factories["standard"]
            )()

        return scaler_key, self.scalers[scaler_key]

    def normalize_features(
        self,
        features: pd.DataFrame,
        method: str = "standard",
        fit: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Normalize features using specified method.

        Args:
            features: Features DataFrame to normalize
            method: Normalization method ('standard', 'robust', 'minmax')
            fit: Whether to fit scaler (True) or use existing (False)
            **kwargs: Additional parameters for scalers

        Returns:
            Normalized features DataFrame
        """
        if features.empty:
            return features

        try:
            features_norm = features.copy()
            numeric_cols = features_norm.select_dtypes(
                include=[np.number]
            ).columns

            if len(numeric_cols) == 0:
                logger.warning("No numeric columns found for normalization")
                return features_norm

            # Get or create scaler
            scaler_key, scaler = self._get_or_create_scaler(
                method, fit, **kwargs
            )

            # Fit scaler if needed
            if fit:
                scaler.fit(features_norm[numeric_cols])

            # Apply normalization
            features_norm[numeric_cols] = scaler.transform(
                features_norm[numeric_cols]
            )
            logger.debug(
                f"Applied {method} normalization to {len(numeric_cols)} columns"
            )

            return features_norm

        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return features  # Return original features on error

    def remove_outliers(
        self, features: pd.DataFrame, method: str = "iqr", **kwargs
    ) -> pd.DataFrame:
        """Remove outliers from features.

        Args:
            features: Features DataFrame
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            **kwargs: Additional parameters for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        if features.empty:
            return features

        try:
            features_clean = features.copy()
            numeric_cols = features_clean.select_dtypes(
                include=[np.number]
            ).columns

            if len(numeric_cols) == 0:
                return features_clean

            if method == "iqr":
                # Interquartile Range method
                multiplier = kwargs.get("multiplier", 1.5)

                for col in numeric_cols:
                    Q1 = features_clean[col].quantile(0.25)
                    Q3 = features_clean[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR

                    outliers = (features_clean[col] < lower_bound) | (
                        features_clean[col] > upper_bound
                    )
                    features_clean.loc[outliers, col] = np.nan

                logger.debug(
                    f"Applied IQR outlier removal with multiplier {multiplier}"
                )

            elif method == "zscore":
                # Z-score method
                threshold = kwargs.get("threshold", 3.0)

                for col in numeric_cols:
                    z_scores = np.abs(
                        (features_clean[col] - features_clean[col].mean())
                        / features_clean[col].std()
                    )
                    outliers = z_scores > threshold
                    features_clean.loc[outliers, col] = np.nan

                logger.debug(
                    f"Applied Z-score outlier removal with threshold {threshold}"
                )

            elif method == "isolation_forest":
                # Isolation Forest method (requires sklearn)
                try:
                    from sklearn.ensemble import IsolationForest

                    contamination = kwargs.get("contamination", 0.1)

                    iso_forest = IsolationForest(
                        contamination=contamination, random_state=42
                    )
                    outliers = (
                        iso_forest.fit_predict(features_clean[numeric_cols])
                        == -1
                    )

                    features_clean.loc[outliers] = np.nan
                    logger.debug(
                        f"Applied Isolation Forest outlier removal with contamination {contamination}"
                    )

                except ImportError:
                    logger.warning(
                        "sklearn not available, falling back to IQR method"
                    )
                    return self.remove_outliers(
                        features, method="iqr", **kwargs
                    )

                    from sklearn.ensemble import IsolationForest

                    contamination = kwargs.get("contamination", 0.1)

                    iso_forest = IsolationForest(
                        contamination=contamination, random_state=42
                    )
                    outliers = (
                        iso_forest.fit_predict(features_clean[numeric_cols])
                        == -1
                    )

                    features_clean.loc[outliers] = np.nan
                    logger.debug(
                        f"Applied Isolation Forest outlier removal with contamination {contamination}"
                    )

                except ImportError:
                    logger.warning(
                        "sklearn not available, falling back to IQR method"
                    )
                    return self.remove_outliers(
                        features, method="iqr", **kwargs
                    )

            else:
                logger.warning(
                    f"Unknown outlier removal method: {method}, using IQR"
                )
                return self.remove_outliers(features, method="iqr", **kwargs)

            # Handle NaN values created by outlier removal
            features_clean = self.handle_missing_values(
                features_clean, strategy="interpolate"
            )

            return features_clean

        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return features  # Return original features on error

    def clip_extreme_values(
        self,
        features: pd.DataFrame,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ) -> pd.DataFrame:
        """Clip extreme values to specified percentiles.

        Args:
            features: Features DataFrame
            lower_percentile: Lower percentile for clipping (0-100)
            upper_percentile: Upper percentile for clipping (0-100)

        Returns:
            DataFrame with clipped values
        """
        if features.empty:
            return features

        try:
            features_clipped = features.copy()
            numeric_cols = features_clipped.select_dtypes(
                include=[np.number]
            ).columns

            for col in numeric_cols:
                lower_bound = features_clipped[col].quantile(
                    lower_percentile / 100
                )
                upper_bound = features_clipped[col].quantile(
                    upper_percentile / 100
                )

                features_clipped[col] = features_clipped[col].clip(
                    lower=lower_bound, upper=upper_bound
                )

            logger.debug(
                f"Clipped values to {lower_percentile}-{upper_percentile} percentile range"
            )
            return features_clipped

        except Exception as e:
            logger.error(f"Error clipping extreme values: {e}")
            return features

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about fitted preprocessors.

        Returns:
            Dictionary with preprocessor information
        """
        info = {
            "scalers": {},
            "imputers": {},
            "fitted_scalers": list(self.scalers.keys()),
            "fitted_imputers": list(self.imputers.keys()),
        }

        # Get scaler information
        for name, scaler in self.scalers.items():
            scaler_info = {"type": type(scaler).__name__}

            if hasattr(scaler, "mean_"):
                scaler_info["feature_count"] = len(scaler.mean_)
            elif hasattr(scaler, "center_"):
                scaler_info["feature_count"] = len(scaler.center_)
            elif hasattr(scaler, "min_"):
                scaler_info["feature_count"] = len(scaler.min_)

            info["scalers"][name] = scaler_info

        # Get imputer information
        for name, imputer in self.imputers.items():
            imputer_info = {
                "type": type(imputer).__name__,
                "strategy": getattr(imputer, "strategy", "unknown"),
            }

            if hasattr(imputer, "statistics_"):
                imputer_info["feature_count"] = len(imputer.statistics_)

            info["imputers"][name] = imputer_info

        return info

    def reset_preprocessors(self) -> None:
        """Reset all fitted preprocessors."""
        self.scalers.clear()
        self.imputers.clear()
        logger.info("All preprocessors reset")
