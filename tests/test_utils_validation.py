"""Unit tests for validation utilities.

Bottom-up testing approach with real data, minimal mocking.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from marketregimeml.utils.validation import (
    OHLCVValidator,
    ParameterValidator,
    DataValidator,
    RegimeValidator,
    validate_fitted,
    validate_input,
)


class TestOHLCVValidator:
    """Test OHLCV data validation."""
    
    def test_validate_valid_data(self):
        """Test validation of valid OHLCV data."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [103, 104, 105, 106, 107],
            "volume": [1000, 1100, 1200, 1300, 1400]
        }, index=dates)
        
        assert OHLCVValidator.validate(df) is True
    
    def test_validate_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            OHLCVValidator.validate(df)
    
    def test_validate_missing_columns(self):
        """Test validation fails for missing required columns."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            # Missing low, close, volume
        }, index=dates)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            OHLCVValidator.validate(df)
    
    def test_validate_no_datetime_index(self):
        """Test validation fails without DatetimeIndex."""
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [95, 96],
            "close": [103, 104],
            "volume": [1000, 1100]
        })  # Regular integer index
        
        with pytest.raises(ValueError, match="must have DatetimeIndex"):
            OHLCVValidator.validate(df)
    
    def test_validate_nan_in_ohlc(self):
        """Test validation fails with NaN in OHLC data."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "open": [100, np.nan, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [103, 104, 105],
            "volume": [1000, 1100, 1200]
        }, index=dates)
        
        with pytest.raises(ValueError, match="NaN values found in OHLC"):
            OHLCVValidator.validate(df)
    
    def test_validate_nan_in_volume_allowed(self):
        """Test that NaN in volume is allowed."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [103, 104, 105],
            "volume": [1000, np.nan, 1200]  # NaN in volume
        }, index=dates)
        
        assert OHLCVValidator.validate(df) is True
    
    def test_validate_negative_prices(self):
        """Test validation fails with negative prices."""
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        df = pd.DataFrame({
            "open": [100, -101],  # Negative price
            "high": [105, 106],
            "low": [95, 96],
            "close": [103, 104],
            "volume": [1000, 1100]
        }, index=dates)
        
        with pytest.raises(ValueError, match="Prices cannot be negative"):
            OHLCVValidator.validate(df)
    
    def test_validate_zero_prices(self):
        """Test validation fails with zero prices."""
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [95, 0],  # Zero price
            "close": [103, 104],
            "volume": [1000, 1100]
        }, index=dates)
        
        with pytest.raises(ValueError, match="Prices cannot be zero"):
            OHLCVValidator.validate(df)
    
    def test_validate_high_low_consistency(self):
        """Test validation fails when high < low."""
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 96],  # High < Low
            "low": [95, 97],
            "close": [103, 104],
            "volume": [1000, 1100]
        }, index=dates)
        
        with pytest.raises(ValueError, match="High must be >= Low"):
            OHLCVValidator.validate(df)
    
    def test_validate_high_open_close_consistency(self):
        """Test validation fails when high < open or close."""
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 100],  # High < Open
            "low": [95, 96],
            "close": [103, 99],
            "volume": [1000, 1100]
        }, index=dates)
        
        with pytest.raises(ValueError, match="High must be >= Open and Close"):
            OHLCVValidator.validate(df)
    
    def test_validate_low_open_close_consistency(self):
        """Test validation fails when low > open or close."""
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [95, 105],  # Low > Open and Close
            "close": [103, 104],
            "volume": [1000, 1100]
        }, index=dates)
        
        with pytest.raises(ValueError, match="Low must be <= Open and Close"):
            OHLCVValidator.validate(df)
    
    def test_standardize_column_names(self):
        """Test standardizing column names to lowercase."""
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        df = pd.DataFrame({
            "Open": [100, 101],
            "HIGH": [105, 106],
            "Low": [95, 96],
            "CLOSE": [103, 104],
            "Volume": [1000, 1100]
        }, index=dates)
        
        standardized = OHLCVValidator.standardize(df)
        
        expected_columns = ["open", "high", "low", "close", "volume"]
        assert list(standardized.columns) == expected_columns
        assert standardized["open"].tolist() == [100, 101]
    
    def test_standardize_sorts_by_date(self):
        """Test that standardize sorts by date."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        # Create unsorted DataFrame
        df = pd.DataFrame({
            "open": [102, 100, 101],
            "high": [107, 105, 106],
            "low": [97, 95, 96],
            "close": [105, 103, 104],
            "volume": [1200, 1000, 1100]
        }, index=[dates[2], dates[0], dates[1]])
        
        standardized = OHLCVValidator.standardize(df)
        
        # Should be sorted by date
        assert standardized.index[0] == dates[0]
        assert standardized.index[1] == dates[1]
        assert standardized.index[2] == dates[2]
        assert standardized["open"].tolist() == [100, 101, 102]


class TestParameterValidator:
    """Test parameter validation."""
    
    def test_validate_n_regimes_valid(self):
        """Test validation of valid n_regimes."""
        assert ParameterValidator.validate_n_regimes(3) is True
        assert ParameterValidator.validate_n_regimes(5) is True
        assert ParameterValidator.validate_n_regimes(2) is True
    
    def test_validate_n_regimes_invalid(self):
        """Test validation of invalid n_regimes."""
        with pytest.raises(ValueError, match="must be an integer >= 2"):
            ParameterValidator.validate_n_regimes(1)
        
        with pytest.raises(ValueError, match="must be an integer >= 2"):
            ParameterValidator.validate_n_regimes(0)
        
        with pytest.raises(ValueError, match="must be an integer >= 2"):
            ParameterValidator.validate_n_regimes(-1)
        
        with pytest.raises(ValueError, match="must be an integer >= 2"):
            ParameterValidator.validate_n_regimes(2.5)
        
        with pytest.raises(ValueError, match="must be an integer >= 2"):
            ParameterValidator.validate_n_regimes("3")
    
    def test_validate_random_state_valid(self):
        """Test validation of valid random_state."""
        assert ParameterValidator.validate_random_state(None) is True
        assert ParameterValidator.validate_random_state(42) is True
        assert ParameterValidator.validate_random_state(0) is True
        assert ParameterValidator.validate_random_state(np.random.RandomState(42)) is True
    
    def test_validate_random_state_invalid(self):
        """Test validation of invalid random_state."""
        with pytest.raises(ValueError, match="random_state must be"):
            ParameterValidator.validate_random_state(-1)
        
        with pytest.raises(ValueError, match="random_state must be"):
            ParameterValidator.validate_random_state(3.14)
        
        with pytest.raises(ValueError, match="random_state must be"):
            ParameterValidator.validate_random_state("42")
    
    def test_validate_covariance_type_valid(self):
        """Test validation of valid covariance types."""
        valid_types = ["full", "diag", "tied", "spherical"]
        for cov_type in valid_types:
            assert ParameterValidator.validate_covariance_type(cov_type) is True
    
    def test_validate_covariance_type_invalid(self):
        """Test validation of invalid covariance types."""
        with pytest.raises(ValueError, match="covariance_type must be one of"):
            ParameterValidator.validate_covariance_type("invalid")
        
        with pytest.raises(ValueError, match="covariance_type must be one of"):
            ParameterValidator.validate_covariance_type("Full")  # Case sensitive
        
        with pytest.raises(ValueError, match="covariance_type must be one of"):
            ParameterValidator.validate_covariance_type(None)
    
    def test_validate_features_valid(self):
        """Test validation of valid features."""
        # NumPy array
        features = np.array([[1, 2], [3, 4], [5, 6]])
        assert ParameterValidator.validate_features(features) is True
        
        # DataFrame
        features = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["a", "b"])
        assert ParameterValidator.validate_features(features) is True
    
    def test_validate_features_invalid(self):
        """Test validation of invalid features."""
        # Empty array
        with pytest.raises(ValueError, match="Features cannot be empty"):
            ParameterValidator.validate_features(np.array([]))
        
        # 1D array
        with pytest.raises(ValueError, match="Features must be 2D"):
            ParameterValidator.validate_features(np.array([1, 2, 3]))
        
        # Invalid type
        with pytest.raises(ValueError, match="Features must be numpy array or DataFrame"):
            ParameterValidator.validate_features([[1, 2], [3, 4]])


class TestDataValidator:
    """Test data validation utilities."""
    
    def test_check_consistent_length_valid(self):
        """Test checking consistent length with valid data."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        assert DataValidator.check_consistent_length(X, y) is True
    
    def test_check_consistent_length_invalid(self):
        """Test checking consistent length with mismatched data."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1])  # Different length
        
        with pytest.raises(ValueError, match="Inconsistent number of samples"):
            DataValidator.check_consistent_length(X, y)
    
    def test_check_no_missing_data_valid(self):
        """Test checking for missing data with valid data."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        assert DataValidator.check_no_missing_data(X) is True
        
        df = pd.DataFrame(X, columns=["a", "b"])
        assert DataValidator.check_no_missing_data(df) is True
    
    def test_check_no_missing_data_invalid(self):
        """Test checking for missing data with NaN values."""
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        
        with pytest.raises(ValueError, match="Data contains NaN values"):
            DataValidator.check_no_missing_data(X)
        
        df = pd.DataFrame(X, columns=["a", "b"])
        with pytest.raises(ValueError, match="Data contains NaN values"):
            DataValidator.check_no_missing_data(df)
    
    def test_check_sufficient_data_valid(self):
        """Test checking for sufficient data with valid size."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        assert DataValidator.check_sufficient_data(X, min_samples=3) is True
    
    def test_check_sufficient_data_invalid(self):
        """Test checking for sufficient data with insufficient size."""
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="Insufficient data"):
            DataValidator.check_sufficient_data(X, min_samples=5)


class TestRegimeValidator:
    """Test regime validation utilities."""
    
    def test_validate_regimes_valid(self):
        """Test validation of valid regime labels."""
        regimes = np.array([0, 1, 2, 0, 1, 2])
        assert RegimeValidator.validate_regimes(regimes, n_regimes=3) is True
    
    def test_validate_regimes_invalid_range(self):
        """Test validation with regimes outside valid range."""
        regimes = np.array([0, 1, 2, 3])  # 3 is outside range for n_regimes=3
        
        with pytest.raises(ValueError, match="Regime labels must be in range"):
            RegimeValidator.validate_regimes(regimes, n_regimes=3)
        
        regimes = np.array([-1, 0, 1])  # -1 is negative
        with pytest.raises(ValueError, match="Regime labels must be in range"):
            RegimeValidator.validate_regimes(regimes, n_regimes=3)
    
    def test_validate_regimes_wrong_shape(self):
        """Test validation with wrong shape."""
        regimes = np.array([[0, 1], [2, 0]])  # 2D array
        
        with pytest.raises(ValueError, match="Regime labels must be 1D"):
            RegimeValidator.validate_regimes(regimes, n_regimes=3)
    
    def test_validate_transition_matrix_valid(self):
        """Test validation of valid transition matrix."""
        matrix = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]
        ])
        
        assert RegimeValidator.validate_transition_matrix(matrix, n_regimes=3) is True
    
    def test_validate_transition_matrix_wrong_shape(self):
        """Test validation with wrong shaped matrix."""
        matrix = np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ])
        
        with pytest.raises(ValueError, match="must be 3x3"):
            RegimeValidator.validate_transition_matrix(matrix, n_regimes=3)
    
    def test_validate_transition_matrix_invalid_probs(self):
        """Test validation with invalid probabilities."""
        # Rows don't sum to 1
        matrix = np.array([
            [0.7, 0.2, 0.2],  # Sums to 1.1
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]
        ])
        
        with pytest.raises(ValueError, match="Each row must sum to 1"):
            RegimeValidator.validate_transition_matrix(matrix, n_regimes=3)
        
        # Negative probability
        matrix = np.array([
            [0.7, 0.4, -0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]
        ])
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            RegimeValidator.validate_transition_matrix(matrix, n_regimes=3)


class TestValidationDecorators:
    """Test validation decorator functions."""
    
    def test_validate_fitted_with_fitted_model(self):
        """Test validate_fitted with fitted model."""
        class Model:
            def __init__(self):
                self.is_fitted = True
            
            @validate_fitted
            def predict(self, X):
                return X * 2
        
        model = Model()
        result = model.predict(5)
        assert result == 10
    
    def test_validate_fitted_with_unfitted_model(self):
        """Test validate_fitted with unfitted model."""
        class Model:
            def __init__(self):
                self.is_fitted = False
            
            @validate_fitted
            def predict(self, X):
                return X * 2
        
        model = Model()
        with pytest.raises(ValueError, match="Model is not fitted"):
            model.predict(5)
    
    def test_validate_input_decorator(self):
        """Test validate_input decorator."""
        @validate_input
        def process_data(X):
            return X.sum()
        
        # Valid input
        X = np.array([[1, 2], [3, 4]])
        result = process_data(X)
        assert result == 10
        
        # Invalid input with NaN
        X = np.array([[1, 2], [np.nan, 4]])
        with pytest.raises(ValueError, match="Invalid input"):
            process_data(X)