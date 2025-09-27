"""Common utilities for MarketRegimeML.

Consolidates frequently used patterns to reduce duplication.
Following DRY (Don't Repeat Yourself) principle.
"""

from typing import Union, Optional, Any, Dict, List
import numpy as np
import pandas as pd
from functools import wraps
import warnings


class ParameterValidator:
    """Common parameter validation patterns."""

    @staticmethod
    def validate_positive(value: Union[int, float], name: str) -> None:
        """Validate that a parameter is positive.

        Parameters
        ----------
        value : int or float
            Value to validate
        name : str
            Parameter name for error message

        Raises
        ------
        ValueError
            If value is not positive
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def validate_non_negative(value: Union[int, float], name: str) -> None:
        """Validate that a parameter is non-negative.

        Parameters
        ----------
        value : int or float
            Value to validate
        name : str
            Parameter name for error message

        Raises
        ------
        ValueError
            If value is negative
        """
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    @staticmethod
    def validate_in_range(
        value: Union[int, float],
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "value",
    ) -> None:
        """Validate that a parameter is within a range.

        Parameters
        ----------
        value : int or float
            Value to validate
        min_val : float, optional
            Minimum allowed value
        max_val : float, optional
            Maximum allowed value
        name : str
            Parameter name for error message

        Raises
        ------
        ValueError
            If value is outside the range
        """
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")

    @staticmethod
    def validate_window_size(window: int, data_length: int) -> None:
        """Validate window size for rolling operations.

        Parameters
        ----------
        window : int
            Window size
        data_length : int
            Length of data

        Raises
        ------
        ValueError
            If window is invalid
        """
        if window <= 0:
            raise ValueError(f"Window size must be positive, got {window}")
        if window > data_length:
            raise ValueError(
                f"Window size ({window}) cannot exceed data length ({data_length})"
            )


class DataNormalizer:
    """Common data normalization patterns."""

    @staticmethod
    def normalize_returns(
        returns: Union[pd.Series, np.ndarray],
        method: str = "zscore",
    ) -> Union[pd.Series, np.ndarray]:
        """Normalize returns using specified method.

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Returns to normalize
        method : str
            Normalization method ('zscore', 'minmax', 'robust')

        Returns
        -------
        pd.Series or np.ndarray
            Normalized returns
        """
        if method == "zscore":
            mean = np.nanmean(returns)
            std = np.nanstd(returns)
            if std > 0:
                return (returns - mean) / std
            return returns - mean

        elif method == "minmax":
            min_val = np.nanmin(returns)
            max_val = np.nanmax(returns)
            if max_val > min_val:
                return (returns - min_val) / (max_val - min_val)
            return np.zeros_like(returns)

        elif method == "robust":
            # Use median and MAD for robust normalization
            median = np.nanmedian(returns)
            mad = np.nanmedian(np.abs(returns - median))
            if mad > 0:
                return (returns - median) / (1.4826 * mad)
            return returns - median

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def normalize_features(
        features: pd.DataFrame,
        method: str = "zscore",
        feature_range: tuple = (0, 1),
    ) -> pd.DataFrame:
        """Normalize feature matrix.

        Parameters
        ----------
        features : pd.DataFrame
            Features to normalize
        method : str
            Normalization method
        feature_range : tuple
            Range for minmax normalization

        Returns
        -------
        pd.DataFrame
            Normalized features
        """
        normalized = features.copy()

        for col in normalized.columns:
            if method == "zscore":
                mean = normalized[col].mean()
                std = normalized[col].std()
                if std > 0:
                    normalized[col] = (normalized[col] - mean) / std
            elif method == "minmax":
                min_val = normalized[col].min()
                max_val = normalized[col].max()
                if max_val > min_val:
                    normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
                    # Scale to feature_range
                    normalized[col] = (
                        normalized[col] * (feature_range[1] - feature_range[0])
                        + feature_range[0]
                    )

        return normalized


class RollingWindow:
    """Common rolling window operations."""

    @staticmethod
    def apply_rolling(
        data: Union[pd.Series, pd.DataFrame],
        window: int,
        func: callable,
        min_periods: Optional[int] = None,
        center: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Apply function over rolling window.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Data to process
        window : int
            Window size
        func : callable
            Function to apply
        min_periods : int, optional
            Minimum periods required
        center : bool
            Whether to center the window

        Returns
        -------
        pd.Series or pd.DataFrame
            Result of rolling operation
        """
        if min_periods is None:
            min_periods = window

        return data.rolling(
            window=window,
            min_periods=min_periods,
            center=center
        ).apply(func, raw=True)

    @staticmethod
    def rolling_zscore(
        data: pd.Series,
        window: int,
        min_periods: Optional[int] = None,
    ) -> pd.Series:
        """Calculate rolling z-score.

        Parameters
        ----------
        data : pd.Series
            Data series
        window : int
            Window size
        min_periods : int, optional
            Minimum periods required

        Returns
        -------
        pd.Series
            Rolling z-scores
        """
        if min_periods is None:
            min_periods = window

        rolling = data.rolling(window=window, min_periods=min_periods)
        mean = rolling.mean()
        std = rolling.std()

        # Avoid division by zero
        std = std.replace(0, np.nan)
        return (data - mean) / std


class DataValidator:
    """Common data validation patterns."""

    @staticmethod
    def check_no_missing(
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        name: str = "data",
    ) -> None:
        """Check for missing values.

        Parameters
        ----------
        data : pd.DataFrame, pd.Series, or np.ndarray
            Data to check
        name : str
            Name for error message

        Raises
        ------
        ValueError
            If missing values are found
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if data.isna().any().any() if isinstance(data, pd.DataFrame) else data.isna().any():
                n_missing = data.isna().sum().sum() if isinstance(data, pd.DataFrame) else data.isna().sum()
                raise ValueError(f"{name} contains {n_missing} missing values")
        else:
            if np.isnan(data).any():
                n_missing = np.isnan(data).sum()
                raise ValueError(f"{name} contains {n_missing} missing values")

    @staticmethod
    def check_sufficient_data(
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        min_samples: int = 20,
        name: str = "data",
    ) -> None:
        """Check for sufficient data.

        Parameters
        ----------
        data : pd.DataFrame, pd.Series, or np.ndarray
            Data to check
        min_samples : int
            Minimum required samples
        name : str
            Name for error message

        Raises
        ------
        ValueError
            If insufficient data
        """
        n_samples = len(data)
        if n_samples < min_samples:
            raise ValueError(
                f"{name} has insufficient samples ({n_samples} < {min_samples})"
            )

    @staticmethod
    def check_no_constants(
        data: Union[pd.DataFrame, pd.Series],
        name: str = "data",
    ) -> None:
        """Check for constant columns/values.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Data to check
        name : str
            Name for error message

        Warns
        -----
        UserWarning
            If constant values are found
        """
        if isinstance(data, pd.DataFrame):
            constant_cols = data.columns[data.std() == 0].tolist()
            if constant_cols:
                warnings.warn(
                    f"{name} contains constant columns: {constant_cols}",
                    UserWarning
                )
        elif isinstance(data, pd.Series):
            if data.std() == 0:
                warnings.warn(f"{name} is constant", UserWarning)


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    fill_value: float = 0.0,
) -> Union[float, np.ndarray]:
    """Safe division handling zero denominators.

    Parameters
    ----------
    numerator : float or np.ndarray
        Numerator
    denominator : float or np.ndarray
        Denominator
    fill_value : float
        Value to use when denominator is zero

    Returns
    -------
    float or np.ndarray
        Result of division
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        if isinstance(result, np.ndarray):
            result[~np.isfinite(result)] = fill_value
        elif not np.isfinite(result):
            result = fill_value
    return result


def create_lagged_features(
    data: pd.DataFrame,
    n_lags: int,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create lagged features for time series.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    n_lags : int
        Number of lags
    columns : List[str], optional
        Columns to lag. If None, uses all columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with original and lagged features
    """
    if columns is None:
        columns = data.columns.tolist()

    result = data.copy()

    for col in columns:
        for lag in range(1, n_lags + 1):
            result[f"{col}_lag{lag}"] = data[col].shift(lag)

    return result


def calculate_rolling_correlation(
    x: pd.Series,
    y: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Calculate rolling correlation between two series.

    Parameters
    ----------
    x : pd.Series
        First series
    y : pd.Series
        Second series
    window : int
        Window size
    min_periods : int, optional
        Minimum periods required

    Returns
    -------
    pd.Series
        Rolling correlations
    """
    if min_periods is None:
        min_periods = window

    return x.rolling(window=window, min_periods=min_periods).corr(y)