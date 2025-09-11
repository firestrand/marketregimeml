"""
Shared validation utilities for MarketRegimeML.

This module provides centralized validation logic to ensure DRY principles
and consistent validation across the entire codebase.
"""

from typing import Optional, Union, Any
from functools import wraps

import numpy as np
import pandas as pd


class OHLCVValidator:
    """Validator for OHLCV (Open, High, Low, Close, Volume) data.

    Centralizes all OHLCV validation logic to avoid duplication
    across data loaders.
    """

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    @classmethod
    def validate(cls, df: pd.DataFrame) -> bool:
        """Validate OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate

        Returns
        -------
        bool
            True if valid

        Raises
        ------
        ValueError
            If validation fails with specific error message
        """
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("DataFrame is empty")

        # Check for required columns
        missing_columns = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Check for NaN values in OHLC (volume can have NaN)
        ohlc_columns = ["open", "high", "low", "close"]
        if df[ohlc_columns].isna().any().any():
            raise ValueError("NaN values found in OHLC data")

        # Check for negative prices
        if (df[ohlc_columns] < 0).any().any():
            raise ValueError("Prices cannot be negative")

        # Check for zero prices
        if (df[ohlc_columns] == 0).any().any():
            raise ValueError("Prices cannot be zero")

        # Check high/low consistency
        if (df["high"] < df["low"]).any():
            raise ValueError("High must be >= Low for all records")

        # Check high >= open/close
        if ((df["high"] < df["open"]) | (df["high"] < df["close"])).any():
            raise ValueError("High must be >= Open and Close for all records")

        # Check low <= open/close
        if ((df["low"] > df["open"]) | (df["low"] > df["close"])).any():
            raise ValueError("Low must be <= Open and Close for all records")

        return True

    @classmethod
    def standardize(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize OHLCV DataFrame format.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to standardize

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame
        """
        # Create a copy to avoid modifying original
        df = df.copy()

        # Standardize column names
        column_mapping = {
            "Open": "open",
            "o": "open",
            "O": "open",
            "High": "high",
            "h": "high",
            "H": "high",
            "Low": "low",
            "l": "low",
            "L": "low",
            "Close": "close",
            "c": "close",
            "C": "close",
            "Volume": "volume",
            "v": "volume",
            "V": "volume",
            "vol": "volume",
            "Vol": "volume",
        }

        # Apply column mapping
        df = df.rename(columns=column_mapping)

        # Ensure column names are lowercase
        df.columns = df.columns.str.lower()

        # Add volume if missing (set to 0)
        if "volume" not in df.columns:
            df["volume"] = 0

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
                df = df.drop("date", axis=1)
            elif "datetime" in df.columns:
                df.index = pd.to_datetime(df["datetime"])
                df = df.drop("datetime", axis=1)
            elif "timestamp" in df.columns:
                df.index = pd.to_datetime(df["timestamp"])
                df = df.drop("timestamp", axis=1)
            else:
                # Try to convert existing index
                df.index = pd.to_datetime(df.index)

        # Select only OHLCV columns if they exist
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [col for col in ohlcv_cols if col in df.columns]
        df = df[available_cols]

        # Sort by index
        df = df.sort_index()

        # Remove duplicate timestamps (keep last)
        df = df[~df.index.duplicated(keep="last")]

        # Convert to float64 for consistency
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


class ParameterValidator:
    """Validator for common model parameters.

    Ensures consistent parameter validation across all models.
    """

    @staticmethod
    def validate_n_regimes(n_regimes: Any) -> int:
        """Validate n_regimes parameter.

        Parameters
        ----------
        n_regimes : Any
            Number of regimes

        Returns
        -------
        int
            Validated n_regimes

        Raises
        ------
        TypeError
            If n_regimes is not an integer
        ValueError
            If n_regimes is out of valid range
        """
        if not isinstance(n_regimes, int):
            raise TypeError("n_regimes must be an integer")

        if n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")

        if n_regimes > 10:
            raise ValueError("n_regimes must be <= 10 for practical purposes")

        return n_regimes

    @staticmethod
    def validate_random_state(
        random_state: Optional[Union[int, np.random.RandomState]],
    ) -> Optional[Union[int, np.random.RandomState]]:
        """Validate random_state parameter.

        Parameters
        ----------
        random_state : int, RandomState, or None
            Random state for reproducibility

        Returns
        -------
        int, RandomState, or None
            Validated random state

        Raises
        ------
        TypeError
            If random_state is invalid type
        """
        if random_state is None:
            return None

        if isinstance(random_state, int):
            return random_state

        if isinstance(random_state, np.random.RandomState):
            return random_state

        raise TypeError("random_state must be int, RandomState, or None")

    @staticmethod
    def validate_covariance_type(covariance_type: str) -> str:
        """Validate covariance_type parameter.

        Parameters
        ----------
        covariance_type : str
            Type of covariance matrix

        Returns
        -------
        str
            Validated covariance type

        Raises
        ------
        ValueError
            If covariance_type is invalid
        """
        valid_types = ["full", "diag", "spherical", "tied"]

        if covariance_type not in valid_types:
            raise ValueError(
                f"Invalid covariance_type: {covariance_type}. "
                f"Must be one of {valid_types}"
            )

        return covariance_type

    @staticmethod
    def validate_features(
        features: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Validate and convert features to numpy array.

        Parameters
        ----------
        features : array-like
            Feature matrix

        Returns
        -------
        np.ndarray
            Validated features as numpy array

        Raises
        ------
        ValueError
            If features are invalid
        """
        # Convert to numpy array if needed
        if isinstance(features, pd.DataFrame):
            features = features.values
        elif not isinstance(features, np.ndarray):
            features = np.array(features)

        # Check if empty
        if features.size == 0:
            raise ValueError("Features cannot be empty")

        # Ensure 2D
        if features.ndim == 1:
            raise ValueError(
                "Features must be 2D array (n_samples, n_features)"
            )

        if features.ndim != 2:
            raise ValueError(f"Features must be 2D, got {features.ndim}D")

        return features


class DataValidator:
    """General data validation utilities."""

    @staticmethod
    def check_consistent_length(*arrays) -> bool:
        """Check that all arrays have the same length.

        Parameters
        ----------
        *arrays : array-like
            Arrays to check

        Returns
        -------
        bool
            True if all have same length

        Raises
        ------
        ValueError
            If lengths are inconsistent
        """
        if not arrays:
            return True

        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"Arrays must have the same length. Got lengths: {lengths}"
            )

        return True

    @staticmethod
    def check_no_missing_data(data: np.ndarray) -> bool:
        """Check that data has no missing values.

        Parameters
        ----------
        data : np.ndarray
            Data to check

        Returns
        -------
        bool
            True if no missing data

        Raises
        ------
        ValueError
            If data contains NaN or inf
        """
        if np.isnan(data).any():
            raise ValueError("Data contains NaN values")

        if np.isinf(data).any():
            raise ValueError("Data contains infinite values")

        return True

    @staticmethod
    def check_sufficient_data(n_samples: int, min_samples: int = 50) -> bool:
        """Check if there are sufficient data points.

        Parameters
        ----------
        n_samples : int
            Number of samples
        min_samples : int
            Minimum required samples

        Returns
        -------
        bool
            True if sufficient data

        Raises
        ------
        ValueError
            If insufficient data
        """
        if n_samples < min_samples:
            raise ValueError(
                f"Insufficient data: {n_samples} samples, "
                f"minimum required: {min_samples}"
            )

        return True


def validate_fitted(method):
    """Decorator to check if model is fitted before prediction.

    Parameters
    ----------
    method : callable
        Method to wrap

    Returns
    -------
    callable
        Wrapped method
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "is_fitted") or not self.is_fitted:
            raise ValueError(
                f"Model must be fitted before calling {method.__name__}"
            )
        return method(self, *args, **kwargs)

    return wrapper


def validate_input(features_2d: bool = False, no_nan: bool = False):
    """Decorator to validate method inputs.

    Parameters
    ----------
    features_2d : bool
        Check if features are 2D
    no_nan : bool
        Check for NaN values

    Returns
    -------
    callable
        Decorator function
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, X, *args, **kwargs):
            if features_2d:
                if isinstance(X, pd.DataFrame):
                    X = X.values
                if X.ndim != 2:
                    raise ValueError("Features must be 2D")

            if no_nan:
                if isinstance(X, pd.DataFrame):
                    if X.isna().any().any():
                        raise ValueError("Data contains NaN values")
                elif np.isnan(X).any():
                    raise ValueError("Data contains NaN values")

            return method(self, X, *args, **kwargs)

        return wrapper

    return decorator


class RegimeValidator:
    """Validator for regime-specific data."""

    @staticmethod
    def validate_regimes(regimes: np.ndarray) -> np.ndarray:
        """Validate regime labels.

        Parameters
        ----------
        regimes : np.ndarray
            Regime labels

        Returns
        -------
        np.ndarray
            Validated regime labels

        Raises
        ------
        ValueError
            If regimes are invalid
        """
        regimes = np.asarray(regimes)

        if regimes.size == 0:
            raise ValueError("Regimes cannot be empty")

        if regimes.ndim != 1:
            raise ValueError("Regimes must be 1D array")

        # Check for valid regime labels (integers starting from 0)
        unique_regimes = np.unique(regimes)
        expected_regimes = np.arange(len(unique_regimes))

        if not np.array_equal(np.sort(unique_regimes), expected_regimes):
            raise ValueError(
                f"Regime labels must be consecutive integers starting from 0. "
                f"Got: {unique_regimes}"
            )

        return regimes

    @staticmethod
    def validate_transition_matrix(
        matrix: np.ndarray, n_regimes: int
    ) -> np.ndarray:
        """Validate regime transition matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Transition matrix
        n_regimes : int
            Expected number of regimes

        Returns
        -------
        np.ndarray
            Validated transition matrix

        Raises
        ------
        ValueError
            If matrix is invalid
        """
        matrix = np.asarray(matrix)

        if matrix.shape != (n_regimes, n_regimes):
            raise ValueError(
                f"Transition matrix must be {n_regimes}x{n_regimes}, "
                f"got {matrix.shape}"
            )

        # Check that rows sum to 1 (valid probability distribution)
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-5):
            raise ValueError("Transition matrix rows must sum to 1")

        # Check for valid probabilities
        if (matrix < 0).any() or (matrix > 1).any():
            raise ValueError(
                "Transition probabilities must be between 0 and 1"
            )

        return matrix
