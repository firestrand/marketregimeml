"""Centralized error handling utilities following DRY and consistency principles."""

import logging
from typing import Any, Callable, Optional, TypeVar
from functools import wraps
import traceback

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MarketRegimeMLError(Exception):
    """Base exception for all MarketRegimeML errors."""


class DataValidationError(MarketRegimeMLError):
    """Raised when data validation fails."""


class ModelNotFittedError(MarketRegimeMLError):
    """Raised when trying to use an unfitted model."""


class FeatureComputationError(MarketRegimeMLError):
    """Raised when feature computation fails."""


class ModelFittingError(MarketRegimeMLError):
    """Raised when model fitting fails."""


class ConfigurationError(MarketRegimeMLError):
    """Raised when configuration is invalid."""


def safe_execute(
    func: Callable[..., T],
    default_return: Optional[T] = None,
    error_message: Optional[str] = None,
    raise_on_error: bool = False,
    log_level: str = "warning",
) -> Callable[..., Optional[T]]:
    """Decorator for safe function execution with consistent error handling.

    Args:
        func: Function to wrap
        default_return: Value to return on error
        error_message: Custom error message
        raise_on_error: Whether to re-raise exceptions
        log_level: Logging level for errors

    Returns:
        Wrapped function with error handling
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Optional[T]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Create detailed error message
            msg = error_message or f"Error in {func.__name__}"
            full_msg = f"{msg}: {str(e)}"

            # Log at appropriate level
            log_func = getattr(logger, log_level, logger.warning)
            log_func(full_msg)

            # Log traceback for debugging
            logger.debug(f"Traceback: {traceback.format_exc()}")

            if raise_on_error:
                raise

            return default_return

    return wrapper


def validate_and_execute(
    validation_func: Callable[[Any], bool],
    error_class: type = DataValidationError,
    error_message: str = "Validation failed",
) -> Callable:
    """Decorator that validates inputs before execution.

    Args:
        validation_func: Function to validate inputs
        error_class: Exception class to raise on validation failure
        error_message: Error message for validation failure

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate inputs
            if not validation_func(*args, **kwargs):
                raise error_class(error_message)

            # Execute function
            return func(*args, **kwargs)

        return wrapper

    return decorator


class ErrorHandler:
    """Centralized error handler for consistent error management."""

    def __init__(self, logger_name: Optional[str] = None):
        """Initialize error handler.

        Args:
            logger_name: Name for the logger
        """
        self.logger = logging.getLogger(logger_name or __name__)
        self.error_counts = {}

    def handle_error(
        self,
        error: Exception,
        context: str,
        raise_error: bool = False,
        default_return: Any = None,
    ) -> Any:
        """Handle an error with consistent logging and tracking.

        Args:
            error: The exception that occurred
            context: Context where error occurred
            raise_error: Whether to re-raise the error
            default_return: Value to return if not raising

        Returns:
            default_return value or raises exception
        """
        # Track error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log error with context
        self.logger.warning(f"{context}: {error_type}: {str(error)}")
        self.logger.debug(f"Traceback: {traceback.format_exc()}")

        if raise_error:
            raise error

        return default_return

    def get_error_summary(self) -> dict:
        """Get summary of all errors handled.

        Returns:
            Dictionary with error counts by type
        """
        return self.error_counts.copy()

    def reset_error_counts(self):
        """Reset error tracking."""
        self.error_counts.clear()


def retry_on_failure(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator to retry function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor for exponential backoff
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time

            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2**attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {wait_time:.1f}s: {str(e)}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts"
                        )

            raise last_exception

        return wrapper

    return decorator


def ensure_fitted(method: Callable) -> Callable:
    """Decorator to ensure model is fitted before calling method.

    Args:
        method: Method to wrap

    Returns:
        Wrapped method with fitted check
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "is_fitted") or not self.is_fitted:
            raise ModelNotFittedError(
                f"Model must be fitted before calling {method.__name__}"
            )
        return method(self, *args, **kwargs)

    return wrapper


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with timing
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        import time

        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}"
            )
            raise

    return wrapper
