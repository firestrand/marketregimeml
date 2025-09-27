"""Tests for base feature calculator."""

import numpy as np
import pandas as pd
import pytest

from marketregimeml.features.base import BaseFeatureCalculator


class ConcreteFeatureCalculator(BaseFeatureCalculator):
    """Concrete implementation for testing abstract base class."""

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simple implementation for testing."""
        return pd.DataFrame({"feature": data.iloc[:, 0] * 2}, index=data.index)


class TestBaseFeatureCalculator:
    """Test suite for BaseFeatureCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create concrete calculator instance."""
        return ConcreteFeatureCalculator()

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = 100 + np.random.randn(100).cumsum()
        return pd.Series(prices, index=dates, name="price")

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV dataframe."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        data = {
            "open": 100 + np.random.randn(50).cumsum(),
            "high": 105 + np.random.randn(50).cumsum(),
            "low": 95 + np.random.randn(50).cumsum(),
            "close": 100 + np.random.randn(50).cumsum(),
            "volume": np.random.randint(1000, 10000, 50),
        }
        return pd.DataFrame(data, index=dates)

    def test_validate_window_valid(self, calculator):
        """Test window validation with valid inputs."""
        # Should not raise
        calculator.validate_window(10)
        calculator.validate_window(5, min_window=2)
        calculator.validate_window(10, min_window=5, max_window=20)

    def test_validate_window_invalid_type(self, calculator):
        """Test window validation with invalid type."""
        with pytest.raises(TypeError, match="Window must be integer"):
            calculator.validate_window(10.5)
        with pytest.raises(TypeError, match="Window must be integer"):
            calculator.validate_window("10")

    def test_validate_window_too_small(self, calculator):
        """Test window validation with too small window."""
        with pytest.raises(ValueError, match="Window must be >= 2"):
            calculator.validate_window(1)
        with pytest.raises(ValueError, match="Window must be >= 5"):
            calculator.validate_window(3, min_window=5)

    def test_validate_window_too_large(self, calculator):
        """Test window validation with too large window."""
        with pytest.raises(ValueError, match="Window must be <= 100"):
            calculator.validate_window(150, max_window=100)

    def test_validate_data_valid(self, calculator, sample_df, sample_data):
        """Test data validation with valid data."""
        # Should not raise for DataFrame
        calculator.validate_data(sample_df)
        # Should not raise for Series
        calculator.validate_data(sample_data)

    def test_validate_data_none(self, calculator):
        """Test data validation with None."""
        with pytest.raises(ValueError, match="Data cannot be None"):
            calculator.validate_data(None)

    def test_validate_data_empty(self, calculator):
        """Test data validation with empty dataframe."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            calculator.validate_data(empty_df)

        empty_series = pd.Series()
        with pytest.raises(ValueError, match="Data cannot be empty"):
            calculator.validate_data(empty_series)

    def test_validate_data_invalid_type(self, calculator):
        """Test data validation with invalid type."""
        with pytest.raises(TypeError, match="Data must be Series or DataFrame"):
            calculator.validate_data([1, 2, 3])

    def test_handle_nan_drop(self, calculator, sample_data):
        """Test NaN handling with drop method."""
        data_with_nan = sample_data.copy()
        data_with_nan.iloc[10:15] = np.nan

        result = calculator.handle_nan(data_with_nan, method="drop")
        assert not result.isna().any()
        assert len(result) == len(data_with_nan) - 5

    def test_handle_nan_fill(self, calculator, sample_data):
        """Test NaN handling with fill method."""
        data_with_nan = sample_data.copy()
        data_with_nan.iloc[10:12] = np.nan

        result = calculator.handle_nan(data_with_nan, method="fill", fill_value=999)
        assert not result.isna().any()
        assert result.iloc[10] == 999
        assert result.iloc[11] == 999

    def test_handle_nan_forward(self, calculator, sample_data):
        """Test NaN handling with forward fill."""
        data_with_nan = sample_data.copy()
        data_with_nan.iloc[10] = np.nan

        result = calculator.handle_nan(data_with_nan, method="forward")
        assert not result.isna().any()
        assert result.iloc[10] == sample_data.iloc[9]

    def test_handle_nan_backward(self, calculator, sample_data):
        """Test NaN handling with backward fill."""
        data_with_nan = sample_data.copy()
        data_with_nan.iloc[10] = np.nan

        result = calculator.handle_nan(data_with_nan, method="backward")
        assert not result.isna().any()
        assert result.iloc[10] == sample_data.iloc[11]

    def test_handle_nan_invalid_method(self, calculator, sample_data):
        """Test NaN handling with invalid method."""
        with pytest.raises(ValueError, match="Unknown NaN handling method"):
            calculator.handle_nan(sample_data, method="invalid")

    def test_rolling_operation_mean(self, calculator, sample_data):
        """Test rolling mean operation."""
        window = 10
        result = calculator.rolling_operation(sample_data, window, "mean")

        assert len(result) == len(sample_data)
        assert result.iloc[: window - 1].isna().all()
        assert not result.iloc[window:].isna().any()

        # Check a specific value
        expected = sample_data.iloc[10:20].mean()
        np.testing.assert_almost_equal(result.iloc[19], expected)

    def test_rolling_operation_std(self, calculator, sample_data):
        """Test rolling standard deviation."""
        window = 10
        result = calculator.rolling_operation(sample_data, window, "std")

        assert len(result) == len(sample_data)
        assert result.iloc[: window - 1].isna().all()
        assert not result.iloc[window:].isna().any()
        assert (result.iloc[window:] > 0).all()  # Std should be positive

    def test_rolling_operation_various(self, calculator, sample_data):
        """Test various rolling operations."""
        window = 5
        operations = ["median", "min", "max", "skew", "kurt"]

        for op in operations:
            result = calculator.rolling_operation(sample_data, window, op)
            assert len(result) == len(sample_data)
            assert result.iloc[: window - 1].isna().all()

    def test_rolling_operation_invalid(self, calculator, sample_data):
        """Test rolling operation with invalid operation."""
        with pytest.raises(ValueError, match="Unknown rolling operation"):
            calculator.rolling_operation(sample_data, 10, "invalid_op")

    def test_ewm_operation_mean(self, calculator, sample_data):
        """Test exponentially weighted mean."""
        result = calculator.ewm_operation(sample_data, span=10, operation="mean")

        assert len(result) == len(sample_data)
        assert not result.isna().any()  # EWM doesn't produce NaN at start

        # EWM should give more weight to recent values
        # The last EWM value should be closer to the last data point than the first
        last_diff = abs(result.iloc[-1] - sample_data.iloc[-1])
        # Check that EWM smooths the data
        assert last_diff < sample_data.std()

    def test_ewm_operation_std(self, calculator, sample_data):
        """Test exponentially weighted standard deviation."""
        result = calculator.ewm_operation(sample_data, span=10, operation="std")

        assert len(result) == len(sample_data)
        # First value might be NaN for std
        assert not result.iloc[1:].isna().any()
        assert (result.iloc[1:] >= 0).all()  # Std should be non-negative

    def test_calculate_returns_simple(self, calculator, sample_data):
        """Test simple returns calculation."""
        returns = calculator.calculate_returns(sample_data, method="simple")

        assert len(returns) == len(sample_data)
        assert returns.iloc[0] != returns.iloc[0]  # First should be NaN

        # Check specific calculation
        expected = (sample_data.iloc[10] - sample_data.iloc[9]) / sample_data.iloc[9]
        np.testing.assert_almost_equal(returns.iloc[10], expected)

    def test_calculate_returns_log(self, calculator, sample_data):
        """Test log returns calculation."""
        returns = calculator.calculate_returns(sample_data, method="log")

        assert len(returns) == len(sample_data)
        assert returns.iloc[0] != returns.iloc[0]  # First should be NaN

        # Check specific calculation
        expected = np.log(sample_data.iloc[10] / sample_data.iloc[9])
        np.testing.assert_almost_equal(returns.iloc[10], expected)

    def test_calculate_returns_multiperiod(self, calculator, sample_data):
        """Test multi-period returns."""
        periods = 5
        returns = calculator.calculate_returns(
            sample_data, method="simple", periods=periods
        )

        assert len(returns) == len(sample_data)
        assert returns.iloc[:periods].isna().all()

        # Check specific calculation
        expected = (sample_data.iloc[10] - sample_data.iloc[5]) / sample_data.iloc[5]
        np.testing.assert_almost_equal(returns.iloc[10], expected)

    def test_calculate_returns_invalid_method(self, calculator, sample_data):
        """Test returns with invalid method."""
        with pytest.raises(ValueError, match="Unknown return method"):
            calculator.calculate_returns(sample_data, method="invalid")

    def test_normalize_zscore(self, calculator, sample_data):
        """Test z-score normalization."""
        normalized = calculator.normalize(sample_data, method="zscore")

        assert len(normalized) == len(sample_data)
        np.testing.assert_almost_equal(normalized.mean(), 0, decimal=10)
        np.testing.assert_almost_equal(normalized.std(), 1, decimal=10)

    def test_normalize_minmax(self, calculator, sample_data):
        """Test min-max normalization."""
        normalized = calculator.normalize(sample_data, method="minmax")

        assert len(normalized) == len(sample_data)
        np.testing.assert_almost_equal(normalized.min(), 0)
        np.testing.assert_almost_equal(normalized.max(), 1)

    def test_normalize_robust(self, calculator, sample_data):
        """Test robust normalization."""
        normalized = calculator.normalize(sample_data, method="robust")

        assert len(normalized) == len(sample_data)
        # Median should be close to 0
        assert abs(normalized.median()) < 0.1

    def test_normalize_rolling_zscore(self, calculator, sample_data):
        """Test rolling z-score normalization."""
        window = 20
        normalized = calculator.normalize(sample_data, method="zscore", window=window)

        assert len(normalized) == len(sample_data)
        assert normalized.iloc[: window - 1].isna().all()

        # Check local normalization in a window
        window_data = sample_data.iloc[30 : 30 + window]
        expected = (
            sample_data.iloc[30 + window - 1] - window_data.mean()
        ) / window_data.std()
        np.testing.assert_almost_equal(normalized.iloc[30 + window - 1], expected)

    def test_normalize_rolling_minmax(self, calculator, sample_data):
        """Test rolling min-max normalization."""
        window = 20
        normalized = calculator.normalize(sample_data, method="minmax", window=window)

        assert len(normalized) == len(sample_data)
        assert normalized.iloc[: window - 1].isna().all()

        # Values should be between 0 and 1 in each window
        for i in range(window, len(normalized)):
            window_norm = normalized.iloc[i - window + 1 : i + 1]
            assert window_norm.min() >= -0.01  # Small tolerance for float precision
            assert window_norm.max() <= 1.01

    def test_normalize_invalid_method(self, calculator, sample_data):
        """Test normalization with invalid method."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            calculator.normalize(sample_data, method="invalid")

    def test_calculate_features_implementation(self, calculator):
        """Test that concrete implementation works."""
        data = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        result = calculator.calculate_features(data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data)
        assert "feature" in result.columns
        np.testing.assert_array_equal(
            result["feature"].values, data["value"].values * 2
        )
