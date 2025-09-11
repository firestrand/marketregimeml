"""Centralized default configurations for all models following DRY principle."""

from typing import Dict, Any


class ModelDefaults:
    """Centralized default parameters for all model types.

    Single source of truth for model defaults following DRY principle.
    """

    # Common parameters across all models
    COMMON = {
        "n_regimes": 3,
        "random_state": None,
        "verbose": True,
    }

    # HMM specific defaults
    HMM = {
        **COMMON,
        "covariance_type": "full",
        "n_iter": 100,
        "tol": 1e-4,
        "init_method": "kmeans",
        "n_init": 10,
    }

    # GMM specific defaults
    GMM = {
        **COMMON,
        "covariance_type": "full",
        "n_init": 10,
        "max_iter": 100,
        "tol": 1e-3,
        "init_params": "kmeans",
        "warm_start": False,
    }

    # GARCH specific defaults
    GARCH = {
        **COMMON,
        "p": 1,
        "q": 1,
        "vol_threshold": 0.5,
        "window": 252,
        "min_obs": 100,
    }

    # Random Forest defaults
    RANDOM_FOREST = {
        **COMMON,
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
    }

    # XGBoost defaults
    XGBOOST = {
        **COMMON,
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "objective": "multi:softprob",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "gpu_id": -1,
    }

    # SVM defaults
    SVM = {
        **COMMON,
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "class_weight": None,
        "max_iter": -1,
    }

    # Deep Learning common defaults
    DEEP_LEARNING_COMMON = {
        **COMMON,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "early_stopping": True,
        "patience": 10,
        "device": "cpu",
    }

    # LSTM defaults
    LSTM = {
        **DEEP_LEARNING_COMMON,
        "hidden_size": 64,
        "num_layers": 2,
        "bidirectional": False,
    }

    # Transformer defaults
    TRANSFORMER = {
        **DEEP_LEARNING_COMMON,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.1,
    }

    # CNN-LSTM defaults
    CNN_LSTM = {
        **DEEP_LEARNING_COMMON,
        "conv_channels": [32, 64],
        "kernel_sizes": [3, 3],
        "pool_sizes": [2, 2],
        "lstm_hidden": 64,
        "lstm_layers": 2,
    }

    # Ensemble defaults
    ENSEMBLE = {
        **COMMON,
        "models": ["hmm", "gmm", "random_forest"],
        "voting": "soft",
        "weights": None,
        "fit_params": None,
    }

    # Adaptive model defaults
    ADAPTIVE = {
        **COMMON,
        "models": ["hmm", "gmm", "xgboost"],
        "window_size": 100,
        "update_frequency": 20,
        "performance_metric": "log_likelihood",
        "weight_decay": 0.9,
    }

    @classmethod
    def get_defaults(cls, model_type: str) -> Dict[str, Any]:
        """Get default configuration for a specific model type.

        Args:
            model_type: Type of model (e.g., 'hmm', 'gmm', 'lstm')

        Returns:
            Dictionary of default parameters
        """
        model_type = model_type.upper().replace("-", "_")

        if hasattr(cls, model_type):
            return getattr(cls, model_type).copy()
        else:
            # Return common defaults for unknown types
            return cls.COMMON.copy()

    @classmethod
    def merge_with_user_config(
        cls, model_type: str, user_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge user configuration with defaults.

        Args:
            model_type: Type of model
            user_config: User-provided configuration

        Returns:
            Merged configuration with defaults
        """
        defaults = cls.get_defaults(model_type)

        # Deep merge user config into defaults
        merged = defaults.copy()
        if user_config:
            merged.update(user_config)

        return merged


class FeatureDefaults:
    """Centralized default parameters for feature engineering."""

    VOLATILITY = {
        "enabled": True,
        "indicators": ["yang_zhang", "garman_klass", "close_to_close"],
        "window": 20,
        "min_periods": 10,
    }

    ENTROPY = {
        "enabled": True,
        "measures": ["shannon", "approximate", "sample"],
        "window": 50,
        "bins": 10,
        "m": 2,
        "r": 0.2,
    }

    STATISTICAL = {
        "enabled": True,
        "features": ["rolling_stats", "autocorrelation", "hurst"],
        "window": 20,
        "autocorr_lags": [1, 5, 10],
        "quantiles": [0.25, 0.75],
    }

    TECHNICAL = {
        "enabled": True,
        "indicators": ["rsi", "macd", "bollinger"],
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2.0,
    }

    PREPROCESSING = {
        "missing_values": "interpolate",
        "normalization": "standard",
        "outlier_removal": False,
        "outlier_method": "iqr",
        "outlier_threshold": 1.5,
    }

    SELECTION = {
        "enabled": False,
        "method": "variance",
        "threshold": 0.01,
        "k_best": 20,
    }

    @classmethod
    def get_feature_defaults(cls) -> Dict[str, Any]:
        """Get all feature engineering defaults.

        Returns:
            Dictionary of all feature defaults
        """
        return {
            "volatility": cls.VOLATILITY.copy(),
            "entropy": cls.ENTROPY.copy(),
            "statistical": cls.STATISTICAL.copy(),
            "technical": cls.TECHNICAL.copy(),
            "preprocessing": cls.PREPROCESSING.copy(),
            "selection": cls.SELECTION.copy(),
        }


class ValidationDefaults:
    """Centralized default parameters for validation."""

    CROSS_VALIDATION = {
        "n_splits": 5,
        "test_size": 0.2,
        "gap": 0,
        "max_train_size": None,
    }

    WALK_FORWARD = {
        "n_splits": 5,
        "train_period": 252,
        "test_period": 63,
        "step": 21,
        "expanding": False,
    }

    PURGED_CV = {
        "n_splits": 5,
        "purge_gap": 10,
        "embargo": 0.01,
    }

    REGIME_AWARE = {
        "n_splits": 5,
        "min_regime_samples": 20,
        "stratify": True,
    }
