"""Configuration loader utility for MarketRegimeML."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration from YAML files."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration loader.

        Args:
            config_dir: Path to configuration directory. If None, uses default.
        """
        if config_dir is None:
            # Find project root (where setup.py is located)
            current_dir = Path(__file__).parent.parent
            config_dir = current_dir / "config"

        self.config_dir = Path(config_dir) if config_dir else None
        # Don't require config directory to exist

        # Load environment variables
        load_dotenv()

        # Cache for loaded configurations
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_default_config(self, config_name: str) -> Dict[str, Any]:
        """Get default configuration when no config file exists.
        Args:
            config_name: Name of configuration
        Returns:
            Default configuration dictionary
        """
        defaults = {
            "features": {
                "volatility": {"enabled": True, "window": 20},
                "entropy": {"enabled": True, "window": 30},
                "statistical": {"enabled": True, "window": 20},
                "technical": {"enabled": True},
            },
            "markets": {
                "data_sources": {},
                "cache": {"enabled": True, "ttl": 3600},
            },
            "regimes": {
                "regime_definitions": {
                    "n_regimes": 5,
                    "regimes": {},
                },  # New optimized default
                "models": {
                    "hmm": {"enabled": True, "parameters": {"n_regimes": 5}},
                    "gmm": {"enabled": True, "parameters": {"n_regimes": 5}},
                },
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }
        return defaults.get(config_name, {})

    def load(self, config_name: str, reload: bool = False) -> Dict[str, Any]:
        """Load configuration from YAML file or return defaults.

        Args:
            config_name: Name of configuration file (without .yaml extension)
            reload: Force reload even if cached

        Returns:
            Configuration dictionary
        """
        if not reload and config_name in self._cache:
            return self._cache[config_name]

        # Return defaults if no config directory
        if self.config_dir is None or not self.config_dir.exists():
            return self._get_default_config(config_name)
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            return self._get_default_config(config_name)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Process environment variables
        config = self._process_env_vars(config)

        # Cache the configuration
        self._cache[config_name] = config

        return config

    def _extract_env_var_name(self, config_str: str) -> Optional[str]:
        """Extract environment variable name from config string.

        Args:
            config_str: Configuration string that might contain env var reference

        Returns:
            Environment variable name if found, None otherwise
        """
        if config_str.startswith("${") and config_str.endswith("}"):
            return config_str[2:-1]
        return None

    def _resolve_env_var(self, env_var: str) -> Optional[str]:
        """Resolve environment variable value.

        Args:
            env_var: Environment variable name

        Returns:
            Environment variable value or None
        """
        value = os.getenv(env_var)
        if value is None and env_var.endswith("_ENV"):
            # Try to resolve indirect reference
            actual_var_name = env_var.replace("_ENV", "")
            return os.getenv(actual_var_name)
        return value

    def _process_string_config(self, config: str) -> Any:
        """Process a string configuration value for environment variables.

        Args:
            config: String configuration value

        Returns:
            Processed value
        """
        env_var = self._extract_env_var_name(config)
        if env_var:
            return self._resolve_env_var(env_var)

        # Check if it's a reference to env variable name (not a value)
        if config.endswith("_env") or config.endswith("_ENV"):
            return config

        return config

    def _process_env_vars(self, config: Any) -> Any:
        """Recursively process environment variables in configuration.

        Replaces strings like '${ENV_VAR}' with actual environment variable values.

        Args:
            config: Configuration data

        Returns:
            Processed configuration
        """
        if isinstance(config, dict):
            return {k: self._process_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._process_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._process_string_config(config)
        return config

    def get_market_config(self) -> Dict[str, Any]:
        """Get market configuration.

        Returns
        -------
        Dict[str, Any]
            Market configuration dictionary
        """
        return self.load("markets")

    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature configuration.

        Returns
        -------
        Dict[str, Any]
            Feature configuration dictionary
        """
        return self.load("features")

    def get_regime_config(self) -> Dict[str, Any]:
        """Get regime configuration.

        Returns
        -------
        Dict[str, Any]
            Regime configuration dictionary
        """
        return self.load("regimes")

    def get_data_source_config(self, source: str) -> Dict[str, Any]:
        """Get configuration for specific data source.

        Parameters
        ----------
        source : str
            Data source name (e.g., 'oanda', 'alphavantage')

        Returns
        -------
        Dict[str, Any]
            Data source configuration
        """
        market_config = self.get_market_config()
        if "data_sources" not in market_config:
            raise ValueError("No data sources configured")

        if source not in market_config["data_sources"]:
            raise ValueError(f"Data source not configured: {source}")

        return market_config["data_sources"][source]

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration.

        Returns
        -------
        Dict[str, Any]
            Cache configuration dictionary
        """
        market_config = self.get_market_config()
        return market_config.get("cache", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.

        Returns
        -------
        Dict[str, Any]
            Logging configuration dictionary
        """
        market_config = self.get_market_config()
        return market_config.get("logging", {})

    def reload_all(self) -> None:
        """Reload all cached configurations."""
        self._cache.clear()


# Global configuration loader instance
config_loader = ConfigLoader()
