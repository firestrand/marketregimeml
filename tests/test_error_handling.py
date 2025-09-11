"""Unit tests for error handling utilities.

Tests are written to match the actual implementation without mocking,
following a bottom-up approach for real unit testing.
"""

import pytest
import logging
import time
from marketregimeml.utils.error_handling import (
    MarketRegimeMLError,
    DataValidationError,
    ModelNotFittedError,
    FeatureComputationError,
    ModelFittingError,
    ConfigurationError,
    safe_execute,
    validate_and_execute,
    ErrorHandler,
    retry_on_failure,
    ensure_fitted,
    log_execution_time,
)


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_base_exception(self):
        """Test base MarketRegimeMLError."""
        with pytest.raises(MarketRegimeMLError):
            raise MarketRegimeMLError("Test error")
    
    def test_data_validation_error(self):
        """Test DataValidationError is subclass of base."""
        error = DataValidationError("Invalid data")
        assert isinstance(error, MarketRegimeMLError)
        assert str(error) == "Invalid data"
    
    def test_model_not_fitted_error(self):
        """Test ModelNotFittedError."""
        error = ModelNotFittedError("Model not fitted")
        assert isinstance(error, MarketRegimeMLError)
        assert str(error) == "Model not fitted"
    
    def test_feature_computation_error(self):
        """Test FeatureComputationError."""
        error = FeatureComputationError("Feature failed")
        assert isinstance(error, MarketRegimeMLError)
    
    def test_model_fitting_error(self):
        """Test ModelFittingError."""
        error = ModelFittingError("Fitting failed")
        assert isinstance(error, MarketRegimeMLError)
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Bad config")
        assert isinstance(error, MarketRegimeMLError)


class TestSafeExecute:
    """Test safe_execute decorator."""
    
    def test_successful_execution(self):
        """Test safe execution of successful function."""
        @safe_execute
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        assert result == 5
    
    def test_with_default_on_error(self):
        """Test returning default value on error."""
        @safe_execute(default_return=42)
        def failing_func():
            raise ValueError("Test error")
        
        result = failing_func()
        assert result == 42
    
    def test_with_custom_error_message(self):
        """Test custom error message."""
        @safe_execute(error_message="Custom error", default_return=None)
        def failing_func():
            raise ValueError("Test error")
        
        result = failing_func()
        assert result is None
    
    def test_raise_on_error(self):
        """Test re-raising exceptions when configured."""
        @safe_execute(raise_on_error=True)
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_func()
    
    def test_with_args_and_kwargs(self):
        """Test passing arguments through decorator."""
        @safe_execute
        def multiply(x, y, power=1):
            return (x * y) ** power
        
        result = multiply(2, 3, power=2)
        assert result == 36
    
    def test_logging_levels(self, caplog):
        """Test different logging levels."""
        @safe_execute(log_level="error", default_return=None)
        def failing_func():
            raise ValueError("Test error")
        
        with caplog.at_level(logging.ERROR):
            failing_func()
            assert "Error in failing_func" in caplog.text


class TestValidateAndExecute:
    """Test validate_and_execute decorator."""
    
    def test_successful_validation(self):
        """Test successful validation and execution."""
        def validator(*args, **kwargs):
            # Check first positional arg
            return args[0] > 0
        
        @validate_and_execute(validator)
        def double(x):
            return x * 2
        
        result = double(5)
        assert result == 10
    
    def test_failed_validation(self):
        """Test that validation failure raises error."""
        def validator(*args, **kwargs):
            return args[0] > 0
        
        @validate_and_execute(validator)
        def double(x):
            return x * 2
        
        with pytest.raises(DataValidationError, match="Validation failed"):
            double(-5)
    
    def test_custom_error_class(self):
        """Test using custom error class."""
        def validator(*args, **kwargs):
            return False
        
        @validate_and_execute(
            validator, 
            error_class=ConfigurationError,
            error_message="Config error"
        )
        def process(x):
            return x
        
        with pytest.raises(ConfigurationError, match="Config error"):
            process(5)


class TestErrorHandler:
    """Test ErrorHandler class."""
    
    def test_initialization(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler()
        assert handler.error_counts == {}
        assert handler.logger is not None
    
    def test_handle_error_with_raise(self):
        """Test handling error with raise_error=True."""
        handler = ErrorHandler()
        error = ValueError("Test error")
        
        with pytest.raises(ValueError):
            handler.handle_error(error, "test_context", raise_error=True)
        
        # Error type is tracked, not context
        assert handler.error_counts["ValueError"] == 1
    
    def test_handle_error_without_raise(self):
        """Test handling error without raising."""
        handler = ErrorHandler()
        error = ValueError("Test error")
        
        result = handler.handle_error(
            error, "test_context", 
            raise_error=False, 
            default_return="default_value"
        )
        
        assert result == "default_value"
        assert handler.error_counts["ValueError"] == 1
    
    def test_multiple_error_types(self):
        """Test tracking different error types."""
        handler = ErrorHandler()
        
        handler.handle_error(ValueError("e1"), "ctx1", raise_error=False)
        handler.handle_error(TypeError("e2"), "ctx2", raise_error=False)
        handler.handle_error(ValueError("e3"), "ctx3", raise_error=False)
        
        assert handler.error_counts["ValueError"] == 2
        assert handler.error_counts["TypeError"] == 1
    
    def test_get_error_summary(self):
        """Test getting error summary."""
        handler = ErrorHandler()
        
        handler.handle_error(ValueError("e1"), "ctx", raise_error=False)
        handler.handle_error(TypeError("e2"), "ctx", raise_error=False)
        
        summary = handler.get_error_summary()
        assert summary["ValueError"] == 1
        assert summary["TypeError"] == 1
    
    def test_reset_error_counts(self):
        """Test resetting error counts."""
        handler = ErrorHandler()
        
        handler.handle_error(ValueError("e"), "ctx", raise_error=False)
        assert handler.error_counts["ValueError"] == 1
        
        handler.reset_error_counts()
        assert handler.error_counts == {}


class TestRetryOnFailure:
    """Test retry_on_failure decorator."""
    
    def test_successful_on_first_try(self):
        """Test function that succeeds on first try."""
        call_count = 0
        
        @retry_on_failure(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_then_success(self):
        """Test function that fails then succeeds."""
        call_count = 0
        
        @retry_on_failure(max_retries=3, backoff_factor=0.01)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Not yet")
            return "success"
        
        result = eventually_successful()
        assert result == "success"
        assert call_count == 2
    
    def test_max_retries_exceeded(self):
        """Test that max retries raises error."""
        call_count = 0
        
        @retry_on_failure(max_retries=2, backoff_factor=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_fails()
        
        assert call_count == 3  # Initial + 2 retries
    
    def test_specific_exceptions(self):
        """Test retrying only specific exceptions."""
        call_count = 0
        
        @retry_on_failure(
            max_retries=3, 
            backoff_factor=0.01,
            exceptions=(ValueError,)
        )
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Wrong type")
        
        # Should not retry TypeError
        with pytest.raises(TypeError):
            raises_type_error()
        
        assert call_count == 1  # No retries
    
    def test_exponential_backoff(self):
        """Test exponential backoff timing."""
        call_times = []
        
        @retry_on_failure(max_retries=2, backoff_factor=0.01)
        def track_timing():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Retry")
            return "done"
        
        result = track_timing()
        assert result == "done"
        assert len(call_times) == 3
        
        # Check that delays are increasing
        if len(call_times) == 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            # Second delay should be longer (exponential backoff)
            assert delay2 > delay1


class TestEnsureFitted:
    """Test ensure_fitted decorator."""
    
    def test_with_fitted_model(self):
        """Test with properly fitted model."""
        class Model:
            def __init__(self):
                self.is_fitted = True
            
            @ensure_fitted
            def predict(self, X):
                return X * 2
        
        model = Model()
        result = model.predict(5)
        assert result == 10
    
    def test_with_unfitted_model(self):
        """Test with unfitted model."""
        class Model:
            def __init__(self):
                self.is_fitted = False
            
            @ensure_fitted
            def predict(self, X):
                return X * 2
        
        model = Model()
        with pytest.raises(ModelNotFittedError, match="Model must be fitted"):
            model.predict(5)
    
    def test_missing_is_fitted_attribute(self):
        """Test when is_fitted attribute doesn't exist."""
        class Model:
            @ensure_fitted
            def predict(self, X):
                return X * 2
        
        model = Model()
        with pytest.raises(ModelNotFittedError, match="Model must be fitted"):
            model.predict(5)
    
    def test_preserves_method_name(self):
        """Test that decorator preserves method metadata."""
        class Model:
            def __init__(self):
                self.is_fitted = True
            
            @ensure_fitted
            def predict(self, X):
                """Predict method."""
                return X * 2
        
        model = Model()
        assert model.predict.__name__ == "predict"
        assert "Predict method" in model.predict.__doc__


class TestLogExecutionTime:
    """Test log_execution_time decorator."""
    
    def test_execution_time_logging(self, caplog):
        """Test that execution time is logged."""
        @log_execution_time
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        with caplog.at_level(logging.DEBUG):
            result = slow_function()
            assert result == "done"
            assert "slow_function executed in" in caplog.text
            assert "seconds" in caplog.text
    
    def test_with_arguments(self, caplog):
        """Test decorator with function arguments."""
        @log_execution_time
        def add(a, b):
            return a + b
        
        with caplog.at_level(logging.DEBUG):
            result = add(2, 3)
            assert result == 5
            assert "add executed in" in caplog.text
    
    def test_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @log_execution_time
        def documented_function():
            """This is a documented function."""
            return "result"
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."
    
    def test_with_exception(self, caplog):
        """Test timing even when function raises exception."""
        @log_execution_time
        def failing_function():
            time.sleep(0.01)
            raise ValueError("Failed")
        
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError):
                failing_function()
            # Timing should still be logged
            assert "failing_function executed in" in caplog.text