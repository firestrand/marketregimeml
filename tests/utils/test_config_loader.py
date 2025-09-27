"""Tests for configuration loader."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from marketregimeml.config_loader import ConfigLoader


class TestConfigLoader:
    """Test suite for ConfigLoader."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary configuration directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            # Create test configuration files
            markets_config = {
                "markets": {
                    "test": {
                        "symbols": ["TEST1", "TEST2"],
                        "timeframes": ["1h", "1d"],
                    }
                },
                "data_sources": {
                    "test_source": {"api_key": "test_key", "timeout": 30}
                },
                "cache": {"enabled": True, "type": "parquet"},
                "logging": {"level": "INFO"},
            }

            features_config = {
                "feature_sets": {
                    "volatility": {
                        "enabled": True,
                        "features": [
                            {
                                "name": "historical_volatility",
                                "windows": [20, 50],
                            }
                        ],
                    }
                }
            }

            regimes_config = {
                "regime_definitions": {
                    "n_regimes": 3,
                    "regimes": {
                        0: {"name": "Low Vol", "color": "green"},
                        1: {"name": "Normal", "color": "blue"},
                        2: {"name": "High Vol", "color": "red"},
                    },
                }
            }

            # Write configuration files
            with open(config_dir / "markets.yaml", "w") as f:
                yaml.dump(markets_config, f)

            with open(config_dir / "features.yaml", "w") as f:
                yaml.dump(features_config, f)

            with open(config_dir / "regimes.yaml", "w") as f:
                yaml.dump(regimes_config, f)

            yield config_dir

    def test_init_with_custom_dir(self, temp_config_dir):
        """Test initialization with custom config directory."""
        loader = ConfigLoader(str(temp_config_dir))
        assert loader.config_dir == temp_config_dir

    def test_init_with_invalid_dir(self):
        """Test initialization with invalid config directory."""
        # ConfigLoader allows non-existent directories
        loader = ConfigLoader("/nonexistent/path")
        assert loader.config_dir == Path("/nonexistent/path")

    def test_load_markets_config(self, temp_config_dir):
        """Test loading markets configuration."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.load("markets")

        assert "markets" in config
        assert "test" in config["markets"]
        assert config["markets"]["test"]["symbols"] == ["TEST1", "TEST2"]

    def test_load_features_config(self, temp_config_dir):
        """Test loading features configuration."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.load("features")

        assert "feature_sets" in config
        assert "volatility" in config["feature_sets"]
        assert config["feature_sets"]["volatility"]["enabled"] is True

    def test_load_regimes_config(self, temp_config_dir):
        """Test loading regimes configuration."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.load("regimes")

        assert "regime_definitions" in config
        assert config["regime_definitions"]["n_regimes"] == 3

    def test_load_with_cache(self, temp_config_dir):
        """Test that configurations are cached."""
        loader = ConfigLoader(str(temp_config_dir))

        # Load config twice
        config1 = loader.load("markets")
        config2 = loader.load("markets")

        # Should be the same object (cached)
        assert config1 is config2

    def test_load_with_reload(self, temp_config_dir):
        """Test force reload of configuration."""
        loader = ConfigLoader(str(temp_config_dir))

        # Load config
        _ = loader.load("markets")  # Initial load to populate cache

        # Modify the file
        config_path = temp_config_dir / "markets.yaml"
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        data["markets"]["test"]["symbols"].append("TEST3")
        with open(config_path, "w") as f:
            yaml.dump(data, f)

        # Load without reload (should get cached version)
        config2 = loader.load("markets")
        assert len(config2["markets"]["test"]["symbols"]) == 2

        # Load with reload
        config3 = loader.load("markets", reload=True)
        assert len(config3["markets"]["test"]["symbols"]) == 3

    def test_load_nonexistent_config(self, temp_config_dir):
        """Test loading non-existent configuration file."""
        loader = ConfigLoader(str(temp_config_dir))

        # Returns empty dict for non-existent configs
        config = loader.load("nonexistent")
        assert config == {}

    def test_get_market_config(self, temp_config_dir):
        """Test get_market_config helper."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.get_market_config()

        assert "markets" in config
        assert "data_sources" in config

    def test_get_feature_config(self, temp_config_dir):
        """Test get_feature_config helper."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.get_feature_config()

        assert "feature_sets" in config

    def test_get_regime_config(self, temp_config_dir):
        """Test get_regime_config helper."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.get_regime_config()

        assert "regime_definitions" in config

    def test_get_data_source_config(self, temp_config_dir):
        """Test get_data_source_config."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.get_data_source_config("test_source")

        assert config["api_key"] == "test_key"
        assert config["timeout"] == 30

    def test_get_data_source_config_invalid(self, temp_config_dir):
        """Test get_data_source_config with invalid source."""
        loader = ConfigLoader(str(temp_config_dir))

        with pytest.raises(ValueError, match="Data source not configured"):
            loader.get_data_source_config("invalid_source")

    def test_get_cache_config(self, temp_config_dir):
        """Test get_cache_config."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.get_cache_config()

        assert config["enabled"] is True
        assert config["type"] == "parquet"

    def test_get_logging_config(self, temp_config_dir):
        """Test get_logging_config."""
        loader = ConfigLoader(str(temp_config_dir))
        config = loader.get_logging_config()

        assert config["level"] == "INFO"

    def test_reload_all(self, temp_config_dir):
        """Test reload_all method."""
        loader = ConfigLoader(str(temp_config_dir))

        # Load some configs
        loader.load("markets")
        loader.load("features")

        assert len(loader._cache) == 2

        # Reload all
        loader.reload_all()

        assert len(loader._cache) == 0

    def test_process_env_vars(self, temp_config_dir, monkeypatch):
        """Test environment variable processing."""
        # Set environment variable
        monkeypatch.setenv("TEST_API_KEY", "secret_key_123")

        # Create config with env var reference
        config_path = temp_config_dir / "test_env.yaml"
        config_data = {
            "api": {"key": "${TEST_API_KEY}", "url": "https://api.example.com"}
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader(str(temp_config_dir))
        config = loader.load("test_env")

        assert config["api"]["key"] == "secret_key_123"
        assert config["api"]["url"] == "https://api.example.com"

    def test_process_env_vars_nested(self, temp_config_dir, monkeypatch):
        """Test environment variable processing in nested structures."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")

        config_path = temp_config_dir / "test_nested.yaml"
        config_data = {
            "database": {
                "connections": [
                    {"host": "${DB_HOST}", "port": "${DB_PORT}"},
                    {"host": "backup.example.com", "port": "5433"},
                ]
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader(str(temp_config_dir))
        config = loader.load("test_nested")

        assert config["database"]["connections"][0]["host"] == "localhost"
        assert config["database"]["connections"][0]["port"] == "5432"
        assert (
            config["database"]["connections"][1]["host"]
            == "backup.example.com"
        )
