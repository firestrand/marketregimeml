"""Base feature calculator with common utilities.

Provides shared functionality for all feature calculators to eliminate
code duplication and ensure consistency.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseFeatureCalculator(ABC):
    """Base class for all feature calculators with common utilities."""

    @staticmethod
    def validate_window(
        window: int, min_window: int = 2, max_window: Optional[int] = None
    ) -> None:
        """Validate window size parameter.

        Parameters
        ----------
        window : int
            Window size to validate
        min_window : int, default=2
            Minimum allowed window size
        max_window : int, optional
            Maximum allowed window size

        Raises
        ------
        ValueError
            If window is invalid
        """
        if not isinstance(window, int):
            raise TypeError(f"Window must be integer, got {type(window)}")
        if window < min_window:
            raise ValueError(f"Window must be >= {min_window}, got {window}")
        if max_window is not None and window > max_window:
            raise ValueError(f"Window must be <= {max_window}, got {window}")

    @staticmethod
    def validate_data(data: Union[pd.Series, pd.DataFrame]) -> None:
        """Validate input data.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Data to validate

        Raises
        ------
        ValueError
            If data is invalid
        """
        if data is None:
            raise ValueError("Data cannot be None")
        if isinstance(data, pd.Series):
            if len(data) == 0:
                raise ValueError("Data cannot be empty")
        elif isinstance(data, pd.DataFrame):
            if len(data) == 0 or len(data.columns) == 0:
                raise ValueError("DataFrame cannot be empty")
        else:
            raise TypeError(f"Data must be Series or DataFrame, got {type(data)}")

    @staticmethod
    def handle_nan(
        data: Union[pd.Series, pd.DataFrame],
        method: str = "drop",
        fill_value: float = 0,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Handle NaN values in data.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Data with potential NaN values
        method : str, default='drop'
            Method to handle NaN: 'drop', 'fill', 'forward', 'backward'
        fill_value : float, default=0
            Value to use when method='fill'

        Returns
        -------
        pd.Series or pd.DataFrame
            Data with NaN values handled
        """
        if method == "drop":
            return data.dropna()
        elif method == "fill":
            return data.fillna(fill_value)
        elif method == "forward":
            return data.fillna(method="ffill")
        elif method == "backward":
            return data.fillna(method="bfill")
        else:
            raise ValueError(f"Unknown NaN handling method: {method}")

    @staticmethod
    def rolling_operation(
        data: pd.Series,
        window: int,
        operation: str,
        min_periods: Optional[int] = None,
        **kwargs,
    ) -> pd.Series:
        """Perform rolling window operation.

        Parameters
        ----------
        data : pd.Series
            Input data
        window : int
            Window size
        operation : str
            Operation to perform: 'mean', 'std', 'var', 'min', 'max', 'sum'
        min_periods : int, optional
            Minimum number of observations required
        **kwargs
            Additional arguments for the operation

        Returns
        -------
        pd.Series
            Result of rolling operation
        """
        BaseFeatureCalculator.validate_window(window)

        if min_periods is None:
            min_periods = window

        rolling = data.rolling(window=window, min_periods=min_periods)

        if operation == "mean":
            return rolling.mean(**kwargs)
        elif operation == "std":
            return rolling.std(**kwargs)
        elif operation == "var":
            return rolling.var(**kwargs)
        elif operation == "min":
            return rolling.min(**kwargs)
        elif operation == "max":
            return rolling.max(**kwargs)
        elif operation == "sum":
            return rolling.sum(**kwargs)
        elif operation == "median":
            return rolling.median(**kwargs)
        elif operation == "skew":
            return rolling.skew(**kwargs)
        elif operation == "kurt":
            return rolling.kurt(**kwargs)
        elif operation == "quantile":
            q = kwargs.get("q", 0.5)
            return rolling.quantile(q)
        else:
            raise ValueError(f"Unknown rolling operation: {operation}")

    @staticmethod
    def ewm_operation(
        data: pd.Series,
        span: Optional[int] = None,
        alpha: Optional[float] = None,
        operation: str = "mean",
        **kwargs,
    ) -> pd.Series:
        """Perform exponentially weighted operation.

        Parameters
        ----------
        data : pd.Series
            Input data
        span : int, optional
            Span for EWM
        alpha : float, optional
            Smoothing factor
        operation : str, default='mean'
            Operation to perform: 'mean', 'std', 'var'
        **kwargs
            Additional arguments

        Returns
        -------
        pd.Series
            Result of EWM operation
        """
        if span is not None:
            ewm = data.ewm(span=span, **kwargs)
        elif alpha is not None:
            ewm = data.ewm(alpha=alpha, **kwargs)
        else:
            raise ValueError("Either span or alpha must be provided")

        if operation == "mean":
            return ewm.mean()
        elif operation == "std":
            return ewm.std()
        elif operation == "var":
            return ewm.var()
        else:
            raise ValueError(f"Unknown EWM operation: {operation}")

    @staticmethod
    def calculate_returns(
        prices: pd.Series, method: str = "simple", periods: int = 1
    ) -> pd.Series:
        """Calculate returns from prices.

        Parameters
        ----------
        prices : pd.Series
            Price series
        method : str, default='simple'
            'simple' or 'log' returns
        periods : int, default=1
            Number of periods for return calculation

        Returns
        -------
        pd.Series
            Returns series
        """
        if method == "simple":
            return prices.pct_change(periods)
        elif method == "log":
            return np.log(prices / prices.shift(periods))
        else:
            raise ValueError(f"Unknown return method: {method}")

    @staticmethod
    def normalize(
        data: pd.Series, method: str = "zscore", window: Optional[int] = None
    ) -> pd.Series:
        """Normalize data.

        Parameters
        ----------
        data : pd.Series
            Data to normalize
        method : str, default='zscore'
            Normalization method: 'zscore', 'minmax', 'robust'
        window : int, optional
            Window for rolling normalization

        Returns
        -------
        pd.Series
            Normalized data
        """
        if window is not None:
            if method == "zscore":
                mean = data.rolling(window).mean()
                std = data.rolling(window).std()
                return (data - mean) / std.replace(0, np.nan)
            elif method == "minmax":
                min_val = data.rolling(window).min()
                max_val = data.rolling(window).max()
                return (data - min_val) / (max_val - min_val).replace(0, np.nan)
            elif method == "robust":
                median = data.rolling(window).median()
                mad = (data - median).abs().rolling(window).median()
                return (data - median) / (1.4826 * mad).replace(0, np.nan)
        else:
            if method == "zscore":
                return (data - data.mean()) / data.std()
            elif method == "minmax":
                return (data - data.min()) / (data.max() - data.min())
            elif method == "robust":
                median = data.median()
                mad = (data - median).abs().median()
                return (data - median) / (1.4826 * mad)

        raise ValueError(f"Unknown normalization method: {method}")

    @abstractmethod
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from input data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Calculated features
        """

    def validate_and_prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare data for feature calculation.

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Validated and prepared data
        """
        self.validate_data(data)
        # Remove any infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        return data
