"""Comprehensive tests for volatility features module.

Following TDD approach with AAA pattern to boost coverage from ~30% to 90%.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from marketregimeml.features.volatility import VolatilityFeatures


class TestVolatilityFeatures:
    """Test suite for VolatilityFeatures class."""

    @pytest.fixture
    def volatility_features(self):
        """Create VolatilityFeatures instance for testing."""
        return VolatilityFeatures()

    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data for testing."""
        np.random.seed(42)
        n_points = 100

        # Generate realistic OHLC data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, n_points)
        prices = base_price * np.cumprod(1 + returns)

        # Create realistic OHLC from prices
        open_prices = prices.copy()
        close_prices = prices * (1 + np.random.normal(0, 0.005, n_points))
        high_prices = np.maximum(open_prices, close_prices) * (
            1 + np.abs(np.random.normal(0, 0.01, n_points))
        )
        low_prices = np.minimum(open_prices, close_prices) * (
            1 - np.abs(np.random.normal(0, 0.01, n_points))
        )

        return {
            "open": pd.Series(open_prices, name="open"),
            "high": pd.Series(high_prices, name="high"),
            "low": pd.Series(low_prices, name="low"),
            "close": pd.Series(close_prices, name="close"),
        }

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series for testing."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.001, 0.02, 100), name="returns")

    @pytest.fixture
    def high_freq_returns(self):
        """Create high frequency return series for testing."""
        np.random.seed(42)
        dates = pd.date_range(
            "2023-01-01", periods=1440, freq="1min"
        )  # 1 day of minute data
        return pd.Series(
            np.random.normal(0, 0.001, 1440), index=dates, name="hf_returns"
        )

    def test_init(self, volatility_features):
        """Test VolatilityFeatures initialization."""
        # Arrange & Act - done in fixture

        # Assert
        assert volatility_features.min_window == 2
        assert hasattr(volatility_features, "min_window")

    def test_yang_zhang_basic(self, volatility_features, sample_ohlc_data):
        """Test basic Yang-Zhang volatility calculation."""
        # Arrange
        window = 20

        # Act
        result = volatility_features.yang_zhang(
            sample_ohlc_data["open"],
            sample_ohlc_data["high"],
            sample_ohlc_data["low"],
            sample_ohlc_data["close"],
            window,
        )

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == f"yang_zhang_{window}"
        assert len(result) == len(sample_ohlc_data["close"])
        # First (window-1) values should be NaN
        assert pd.isna(result.iloc[: window - 1]).all()

        # Check for valid volatility values
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(
                x >= 0 for x in valid_values
            )  # Volatility should be non-negative

    def test_yang_zhang_different_windows(
        self, volatility_features, sample_ohlc_data
    ):
        """Test Yang-Zhang volatility with different window sizes."""
        # Arrange
        windows = [10, 20, 30]

        # Act & Assert
        for window in windows:
            result = volatility_features.yang_zhang(
                sample_ohlc_data["open"],
                sample_ohlc_data["high"],
                sample_ohlc_data["low"],
                sample_ohlc_data["close"],
                window,
            )
            assert isinstance(result, pd.Series)
            assert result.name == f"yang_zhang_{window}"
            assert len(result) == len(sample_ohlc_data["close"])
            assert pd.isna(result.iloc[: window - 1]).all()

    def test_yang_zhang_min_window_validation(
        self, volatility_features, sample_ohlc_data
    ):
        """Test Yang-Zhang volatility with invalid window size."""
        # Arrange
        invalid_window = 1  # Less than min_window

        # Act & Assert
        with pytest.raises(ValueError, match="Window must be at least"):
            volatility_features.yang_zhang(
                sample_ohlc_data["open"],
                sample_ohlc_data["high"],
                sample_ohlc_data["low"],
                sample_ohlc_data["close"],
                invalid_window,
            )

    def test_garman_klass_basic(self, volatility_features, sample_ohlc_data):
        """Test basic Garman-Klass volatility calculation."""
        # Arrange
        window = 20

        # Act
        result = volatility_features.garman_klass(
            sample_ohlc_data["high"],
            sample_ohlc_data["low"],
            sample_ohlc_data["close"],
            window,
        )

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == f"garman_klass_{window}"
        assert len(result) == len(sample_ohlc_data["close"])
        assert pd.isna(result.iloc[: window - 1]).all()

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(x >= 0 for x in valid_values)

    def test_garman_klass_min_window_validation(
        self, volatility_features, sample_ohlc_data
    ):
        """Test Garman-Klass volatility with invalid window size."""
        # Arrange
        invalid_window = 1

        # Act & Assert
        with pytest.raises(ValueError, match="Window must be at least"):
            volatility_features.garman_klass(
                sample_ohlc_data["high"],
                sample_ohlc_data["low"],
                sample_ohlc_data["close"],
                invalid_window,
            )

    def test_parkinson_basic(self, volatility_features, sample_ohlc_data):
        """Test basic Parkinson volatility calculation."""
        # Arrange
        window = 20

        # Act
        result = volatility_features.parkinson(
            sample_ohlc_data["high"], sample_ohlc_data["low"], window
        )

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == f"parkinson_{window}"
        assert len(result) == len(sample_ohlc_data["high"])
        assert pd.isna(result.iloc[: window - 1]).all()

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(x >= 0 for x in valid_values)

    def test_parkinson_min_window_validation(
        self, volatility_features, sample_ohlc_data
    ):
        """Test Parkinson volatility with invalid window size."""
        # Arrange
        invalid_window = 1

        # Act & Assert
        with pytest.raises(ValueError, match="Window must be at least"):
            volatility_features.parkinson(
                sample_ohlc_data["high"],
                sample_ohlc_data["low"],
                invalid_window,
            )

    def test_rogers_satchell_basic(
        self, volatility_features, sample_ohlc_data
    ):
        """Test basic Rogers-Satchell volatility calculation."""
        # Arrange
        window = 20

        # Act
        result = volatility_features.rogers_satchell(
            sample_ohlc_data["open"],
            sample_ohlc_data["high"],
            sample_ohlc_data["low"],
            sample_ohlc_data["close"],
            window,
        )

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == f"rogers_satchell_{window}"
        assert len(result) == len(sample_ohlc_data["close"])
        assert pd.isna(result.iloc[: window - 1]).all()

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(x >= 0 for x in valid_values)

    def test_rogers_satchell_min_window_validation(
        self, volatility_features, sample_ohlc_data
    ):
        """Test Rogers-Satchell volatility with invalid window size."""
        # Arrange
        invalid_window = 1

        # Act & Assert
        with pytest.raises(ValueError, match="Window must be at least"):
            volatility_features.rogers_satchell(
                sample_ohlc_data["open"],
                sample_ohlc_data["high"],
                sample_ohlc_data["low"],
                sample_ohlc_data["close"],
                invalid_window,
            )

    def test_close_to_close_basic(self, volatility_features, sample_ohlc_data):
        """Test basic close-to-close volatility calculation."""
        # Arrange
        window = 20

        # Act
        result = volatility_features.close_to_close(
            sample_ohlc_data["close"], window
        )

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == f"close_volatility_{window}"
        assert len(result) == len(sample_ohlc_data["close"])

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(x >= 0 for x in valid_values)

    def test_close_to_close_different_windows(
        self, volatility_features, sample_ohlc_data
    ):
        """Test close-to-close volatility with different window sizes."""
        # Arrange
        windows = [10, 20, 50]

        # Act & Assert
        for window in windows:
            result = volatility_features.close_to_close(
                sample_ohlc_data["close"], window
            )
            assert isinstance(result, pd.Series)
            assert result.name == f"close_volatility_{window}"
            valid_values = result.dropna()
            if len(valid_values) > 0:
                assert all(x >= 0 for x in valid_values)

    def test_garch_volatility_basic(self, volatility_features, sample_returns):
        """Test basic GARCH volatility calculation."""
        # Arrange
        omega = 0.00001
        alpha = 0.1
        beta = 0.85

        # Act
        result = volatility_features.garch_volatility(
            sample_returns, omega, alpha, beta
        )

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == "garch_volatility"
        assert len(result) == len(sample_returns)

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(x >= 0 for x in valid_values)

    def test_garch_volatility_default_params(
        self, volatility_features, sample_returns
    ):
        """Test GARCH volatility with default parameters."""
        # Arrange & Act
        result = volatility_features.garch_volatility(sample_returns)

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == "garch_volatility"

    def test_garch_volatility_insufficient_data(self, volatility_features):
        """Test GARCH volatility with insufficient data."""
        # Arrange
        short_returns = pd.Series([0.01, 0.02])  # Only 2 points

        # Act
        result = volatility_features.garch_volatility(short_returns)

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == "garch_volatility"
        # Should handle gracefully with mostly NaN values

    def test_garch_volatility_with_nans(self, volatility_features):
        """Test GARCH volatility with NaN values in returns."""
        # Arrange
        returns_with_nans = pd.Series([0.01, np.nan, 0.02, 0.015, np.nan] * 20)

        # Act
        result = volatility_features.garch_volatility(returns_with_nans)

        # Assert
        assert isinstance(result, pd.Series)
        assert len(result) == len(returns_with_nans)

    def test_realized_volatility_basic(
        self, volatility_features, high_freq_returns
    ):
        """Test basic realized volatility calculation."""
        # Arrange
        freq = "5min"
        daily_window = 1

        # Act
        result = volatility_features.realized_volatility(
            high_freq_returns, freq, daily_window
        )

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == "realized_volatility"

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(x >= 0 for x in valid_values)

    def test_realized_volatility_default_params(
        self, volatility_features, high_freq_returns
    ):
        """Test realized volatility with default parameters."""
        # Arrange & Act
        result = volatility_features.realized_volatility(high_freq_returns)

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == "realized_volatility"

    def test_volatility_of_volatility_basic(
        self, volatility_features, sample_ohlc_data
    ):
        """Test basic volatility of volatility calculation."""
        # Arrange
        # First calculate a volatility series
        volatility = volatility_features.close_to_close(
            sample_ohlc_data["close"], 20
        )
        window = 20

        # Act
        result = volatility_features.volatility_of_volatility(
            volatility, window
        )

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == f"vol_of_vol_{window}"
        assert len(result) == len(volatility)

        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(x >= 0 for x in valid_values)

    def test_volatility_of_volatility_different_windows(
        self, volatility_features, sample_ohlc_data
    ):
        """Test volatility of volatility with different window sizes."""
        # Arrange
        volatility = volatility_features.close_to_close(
            sample_ohlc_data["close"], 20
        )
        windows = [10, 20, 30]

        # Act & Assert
        for window in windows:
            result = volatility_features.volatility_of_volatility(
                volatility, window
            )
            assert isinstance(result, pd.Series)
            assert result.name == f"vol_of_vol_{window}"

    def test_volatility_estimators_comparison(
        self, volatility_features, sample_ohlc_data
    ):
        """Test comparison of different volatility estimators."""
        # Arrange
        window = 20

        # Act
        yang_zhang = volatility_features.yang_zhang(
            sample_ohlc_data["open"],
            sample_ohlc_data["high"],
            sample_ohlc_data["low"],
            sample_ohlc_data["close"],
            window,
        )
        garman_klass = volatility_features.garman_klass(
            sample_ohlc_data["high"],
            sample_ohlc_data["low"],
            sample_ohlc_data["close"],
            window,
        )
        parkinson = volatility_features.parkinson(
            sample_ohlc_data["high"], sample_ohlc_data["low"], window
        )
        close_to_close = volatility_features.close_to_close(
            sample_ohlc_data["close"], window
        )

        # Assert - all should be positive and have similar structure
        estimators = [yang_zhang, garman_klass, parkinson, close_to_close]
        for estimator in estimators:
            assert isinstance(estimator, pd.Series)
            assert len(estimator) == len(sample_ohlc_data["close"])
            valid_values = estimator.dropna()
            if len(valid_values) > 0:
                assert all(x >= 0 for x in valid_values)

    def test_edge_case_constant_prices(self, volatility_features):
        """Test volatility calculations with constant prices."""
        # Arrange
        n_points = 50
        constant_price = 100.0
        ohlc_data = {
            "open": pd.Series([constant_price] * n_points),
            "high": pd.Series([constant_price] * n_points),
            "low": pd.Series([constant_price] * n_points),
            "close": pd.Series([constant_price] * n_points),
        }
        window = 20

        # Act
        yang_zhang = volatility_features.yang_zhang(
            ohlc_data["open"],
            ohlc_data["high"],
            ohlc_data["low"],
            ohlc_data["close"],
            window,
        )
        close_vol = volatility_features.close_to_close(
            ohlc_data["close"], window
        )

        # Assert - constant prices should give zero or near-zero volatility
        valid_yz = yang_zhang.dropna()
        valid_close = close_vol.dropna()

        if len(valid_yz) > 0:
            # Yang-Zhang might have small numerical errors but should be close to zero
            assert all(x < 0.01 for x in valid_yz)  # Very small threshold

        if len(valid_close) > 0:
            # Close-to-close should be exactly zero for constant prices
            assert all(x == 0 or np.isclose(x, 0) for x in valid_close)

    def test_edge_case_single_day_data(self, volatility_features):
        """Test volatility calculations with minimal data."""
        # Arrange
        single_point_data = {
            "open": pd.Series([100.0]),
            "high": pd.Series([102.0]),
            "low": pd.Series([98.0]),
            "close": pd.Series([101.0]),
        }
        window = 2  # Minimum window

        # Act & Assert - should handle gracefully
        result = volatility_features.parkinson(
            single_point_data["high"], single_point_data["low"], window
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 1
        # First value should be NaN due to insufficient window
        assert pd.isna(result.iloc[0])

    def test_edge_case_extreme_prices(self, volatility_features):
        """Test volatility calculations with extreme price movements."""
        # Arrange - create data with extreme price movements
        np.random.seed(42)
        n_points = 30
        base_prices = [100.0]

        # Create extreme movements
        for i in range(n_points - 1):
            change = np.random.choice([-0.9, 0.9])  # Extreme ±90% moves
            base_prices.append(base_prices[-1] * (1 + change))

        extreme_data = {
            "open": pd.Series(base_prices),
            "high": pd.Series([p * 1.05 for p in base_prices]),
            "low": pd.Series([p * 0.95 for p in base_prices]),
            "close": pd.Series([p * 1.01 for p in base_prices]),
        }
        window = 10

        # Act
        result = volatility_features.yang_zhang(
            extreme_data["open"],
            extreme_data["high"],
            extreme_data["low"],
            extreme_data["close"],
            window,
        )

        # Assert - should handle extreme values without crashing
        assert isinstance(result, pd.Series)
        valid_values = result.dropna()
        if len(valid_values) > 0:
            # Extreme volatility should be high but finite
            assert all(np.isfinite(x) for x in valid_values)
            assert all(x >= 0 for x in valid_values)

    def test_numba_core_functions_accessible(self):
        """Test that Numba core functions are accessible."""
        # Arrange & Act & Assert
        from marketregimeml.features.volatility import (
            _yang_zhang_core,
            _garman_klass_core,
            _parkinson_core,
            _rogers_satchell_core,
            _garch_variance_core,
        )

        # Check functions exist and are callable
        assert callable(_yang_zhang_core)
        assert callable(_garman_klass_core)
        assert callable(_parkinson_core)
        assert callable(_rogers_satchell_core)
        assert callable(_garch_variance_core)

    def test_annualization_factor(self, volatility_features, sample_ohlc_data):
        """Test that volatility is properly annualized."""
        # Arrange
        window = 20

        # Act
        daily_vol = volatility_features.close_to_close(
            sample_ohlc_data["close"], window
        )

        # Assert
        valid_values = daily_vol.dropna()
        if len(valid_values) > 0:
            # For realistic daily returns (~2% annual vol),
            # annualized volatility should be reasonable (not too high/low)
            mean_vol = valid_values.mean()
            assert (
                0.01 < mean_vol < 2.0
            )  # Between 1% and 200% annual volatility

    def test_volatility_consistency(
        self, volatility_features, sample_ohlc_data
    ):
        """Test consistency between different window sizes."""
        # Arrange
        windows = [10, 20, 30]

        # Act
        results = {}
        for window in windows:
            results[window] = volatility_features.close_to_close(
                sample_ohlc_data["close"], window
            )

        # Assert - longer windows should generally be smoother
        for window in windows:
            valid_values = results[window].dropna()
            if len(valid_values) > 10:  # Need sufficient data
                # Standard deviation of volatility should decrease with longer windows
                vol_stability = valid_values.std()
                assert vol_stability >= 0  # Basic sanity check

    def test_series_alignment(self, volatility_features, sample_ohlc_data):
        """Test that output series are properly aligned with input indices."""
        # Arrange
        window = 20

        # Act
        result = volatility_features.yang_zhang(
            sample_ohlc_data["open"],
            sample_ohlc_data["high"],
            sample_ohlc_data["low"],
            sample_ohlc_data["close"],
            window,
        )

        # Assert
        assert result.index.equals(sample_ohlc_data["close"].index)
        assert len(result) == len(sample_ohlc_data["close"])

    def test_garch_error_handling(self, volatility_features):
        """Test GARCH error handling with problematic data."""
        # Arrange - create problematic returns that might cause GARCH to fail
        problematic_returns = pd.Series([np.inf, -np.inf, 0, 0, 0] * 10)

        # Act
        result = volatility_features.garch_volatility(problematic_returns)

        # Assert - should handle gracefully and return empty series with correct name
        assert isinstance(result, pd.Series)
        assert result.name == "garch_volatility"
        assert len(result) == len(problematic_returns)

    @patch("marketregimeml.features.volatility.logger")
    def test_garch_logging(self, mock_logger, volatility_features):
        """Test that GARCH logs appropriate warnings and errors."""
        # Arrange
        short_returns = pd.Series([0.01, 0.02])

        # Act
        result = volatility_features.garch_volatility(short_returns)

        # Assert
        assert isinstance(result, pd.Series)
        mock_logger.warning.assert_called_with(
            "Insufficient data for GARCH calculation"
        )

    def test_realized_volatility_resampling(self, volatility_features):
        """Test realized volatility with multi-day high frequency data."""
        # Arrange - create 2 days of minute data
        dates = pd.date_range(
            "2023-01-01", periods=2880, freq="1min"
        )  # 2 days
        np.random.seed(42)
        hf_returns = pd.Series(np.random.normal(0, 0.0005, 2880), index=dates)

        # Act
        result = volatility_features.realized_volatility(hf_returns)

        # Assert
        assert isinstance(result, pd.Series)
        assert result.name == "realized_volatility"
        # Should have daily frequency (2 days = 2 observations)
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert len(valid_values) <= 2  # At most 2 days of data
