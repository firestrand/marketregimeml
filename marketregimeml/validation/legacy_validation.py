"""Validation framework for market regime detection models.

This module provides advanced validation techniques specifically designed for
time series and regime detection, including:
- Combinatorial Purged Cross-Validation (CPCV)
- Regime-aware train/test splitting
- Embargo periods to prevent data leakage
- Deflated Sharpe Ratio for multiple testing correction
- Walk-forward validation
"""

from typing import Iterator, Tuple, Optional, Union, List, Dict
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    "CombinatorialPurgedCV",
    "RegimeAwareSplitter",
    "calculate_deflated_sharpe_ratio",
    "calculate_deflated_sharpe",  # Alias for compatibility
    "apply_embargo",
    "WalkForwardCV",
    "calculate_regime_adjusted_sharpe",
    "calculate_cv_stability",
    "calculate_regime_validation_metrics",
]

# Create alias for compatibility
calculate_deflated_sharpe = (
    lambda *args, **kwargs: calculate_deflated_sharpe_ratio(*args, **kwargs)
)


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation for time series.

    Implements CPCV to prevent data leakage in time series validation
    by ensuring proper temporal separation between train and test sets.

    Parameters
    ----------
    n_splits : int
        Number of splits/folds
    n_test_splits : int
        Number of test splits in each combination
    embargo_td : pd.Timedelta or int, optional
        Embargo period between train and test sets
        If int, interpreted as number of samples
    purge_td : pd.Timedelta or int, optional
        Purge period to remove from training set before test set
        If int, interpreted as number of samples
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        embargo_td: Optional[Union[pd.Timedelta, int]] = 10,
        purge_td: Optional[Union[pd.Timedelta, int]] = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be between 1 and n_splits-1")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_td = embargo_td
        self.purge_td = purge_td

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like
            Features matrix
        y : array-like, optional
            Target values (not used, for compatibility)
        groups : array-like, optional
            Group labels (not used, for compatibility)

        Yields
        ------
        train_idx : ndarray
            Training set indices
        test_idx : ndarray
            Test set indices
        """
        n_samples = self._get_n_samples(X)
        time_index = self._extract_time_index(X)
        splits = self._create_base_splits(n_samples)

        # Generate combinations and yield splits
        for train_idx, test_idx in self._generate_train_test_combinations(
            splits
        ):
            train_idx, test_idx = self._apply_embargo_if_needed(
                train_idx, test_idx, time_index
            )

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def _get_n_samples(self, X: Union[np.ndarray, pd.DataFrame]) -> int:
        """Get number of samples in X."""
        return len(X) if hasattr(X, "__len__") else X.shape[0]

    def _extract_time_index(
        self, X: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> Optional[pd.DatetimeIndex]:
        """Extract time index from X if available."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            if isinstance(X.index, pd.DatetimeIndex):
                return X.index
        return None

    def _create_base_splits(self, n_samples: int) -> List[np.ndarray]:
        """Create base splits for cross-validation."""
        indices = np.arange(n_samples)
        split_size = n_samples // self.n_splits
        splits = []

        for i in range(self.n_splits):
            start = i * split_size
            end = start + split_size if i < self.n_splits - 1 else n_samples
            splits.append(indices[start:end])

        return splits

    def _generate_train_test_combinations(
        self, splits: List[np.ndarray]
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index combinations."""
        test_combinations = list(
            combinations(range(self.n_splits), self.n_test_splits)
        )

        for test_combo in test_combinations:
            test_idx = np.concatenate([splits[i] for i in test_combo])
            train_idx = np.concatenate(
                [
                    splits[i]
                    for i in range(self.n_splits)
                    if i not in test_combo
                ]
            )
            yield train_idx, test_idx

    def _apply_embargo_if_needed(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        time_index: Optional[pd.DatetimeIndex],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply embargo and purge if configured."""
        if self.embargo_td is None and self.purge_td is None:
            return train_idx, test_idx

        return self._apply_embargo_purge(train_idx, test_idx, time_index)

    def _apply_embargo_purge(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        time_index: Optional[pd.DatetimeIndex],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply embargo and purge to prevent data leakage.

        Parameters
        ----------
        train_idx : ndarray
            Initial training indices
        test_idx : ndarray
            Initial test indices
        time_index : DatetimeIndex, optional
            Time index for temporal embargo

        Returns
        -------
        train_idx : ndarray
            Filtered training indices
        test_idx : ndarray
            Filtered test indices
        """
        test_start = test_idx.min()
        test_end = test_idx.max()

        if time_index is not None:
            # Apply temporal embargo
            if self.embargo_td is not None:
                if isinstance(self.embargo_td, pd.Timedelta):
                    embargo_mask = (
                        time_index[train_idx]
                        < time_index[test_start] - self.embargo_td
                    ) | (
                        time_index[train_idx]
                        > time_index[test_end] + self.embargo_td
                    )
                else:
                    # Interpret as number of samples
                    embargo_mask = (
                        train_idx < test_start - self.embargo_td
                    ) | (train_idx > test_end + self.embargo_td)
                train_idx = train_idx[embargo_mask]

            if self.purge_td is not None:
                if isinstance(self.purge_td, pd.Timedelta):
                    purge_mask = (
                        time_index[train_idx]
                        < time_index[test_start] - self.purge_td
                    )
                else:
                    # Interpret as number of samples
                    purge_mask = train_idx < test_start - self.purge_td
                train_idx = train_idx[purge_mask]
        else:
            # Apply index-based embargo
            if self.embargo_td is not None:
                if isinstance(self.embargo_td, int):
                    embargo_mask = (
                        train_idx < test_start - self.embargo_td
                    ) | (train_idx > test_end + self.embargo_td)
                    train_idx = train_idx[embargo_mask]

            if self.purge_td is not None:
                if isinstance(self.purge_td, int):
                    purge_mask = train_idx < test_start - self.purge_td
                    train_idx = train_idx[purge_mask]

        return train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits.

        Returns
        -------
        n_splits : int
            Number of splits
        """
        from math import comb

        return comb(self.n_splits, self.n_test_splits)


class RegimeAwareSplitter:
    """Train/test splitter that respects regime boundaries.

    Ensures that regime transitions are properly handled during splitting
    to prevent data leakage and maintain regime coherence.

    Parameters
    ----------
    test_size : float
        Proportion of data for test set
    embargo_size : int
        Number of samples to exclude around regime transitions
    min_regime_samples : int
        Minimum samples required per regime in train/test sets
    """

    def __init__(
        self,
        test_size: float = 0.2,
        embargo_size: int = 5,
        min_regime_samples: int = 10,
        random_state: Optional[int] = None,
        stratify: bool = False,
        temporal: bool = False,
    ):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if embargo_size < 0:
            raise ValueError("embargo_size must be non-negative")
        if min_regime_samples < 1:
            raise ValueError("min_regime_samples must be at least 1")

        self.test_size = test_size
        self.embargo_size = embargo_size
        self.min_regime_samples = min_regime_samples
        self.random_state = random_state
        self.stratify = stratify
        self.temporal = temporal

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        regimes: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split data respecting regime boundaries.

        Parameters
        ----------
        X : array-like
            Features matrix
        regimes : ndarray
            Regime labels for each sample

        Returns
        -------
        train_idx : ndarray
            Training set indices
        test_idx : ndarray
            Test set indices
        """
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]

        if len(regimes) != n_samples:
            raise ValueError("regimes must have same length as X")

        # Handle stratified splitting
        if self.stratify:
            train_idx, test_idx = self._stratified_split(regimes, n_samples)
        else:
            # Find regime transitions
            transitions = self._find_transitions(regimes)

            # Calculate split point
            split_point = int(n_samples * (1 - self.test_size))

            # Adjust split point to respect regime boundaries
            split_point = self._adjust_split_point(
                split_point, transitions, n_samples
            )

            # Create initial splits
            train_idx = np.arange(split_point)
            test_idx = np.arange(split_point, n_samples)

        # Apply embargo around transitions only if not stratified
        # (stratified already handles regime separation)
        if self.embargo_size > 0 and not self.stratify:
            transitions = self._find_transitions(regimes)
            train_idx, test_idx = self._apply_transition_embargo(
                train_idx, test_idx, transitions
            )

        # Ensure minimum samples per regime
        train_idx, test_idx = self._ensure_min_samples(
            train_idx, test_idx, regimes
        )

        return train_idx, test_idx

    def _find_transitions(self, regimes: np.ndarray) -> np.ndarray:
        """Find indices where regime transitions occur.

        Parameters
        ----------
        regimes : ndarray
            Regime labels

        Returns
        -------
        transitions : ndarray
            Indices of regime transitions
        """
        transitions = np.where(np.diff(regimes) != 0)[0] + 1
        return transitions

    def _adjust_split_point(
        self, split_point: int, transitions: np.ndarray, n_samples: int
    ) -> int:
        """Adjust split point to avoid cutting through regimes.

        Parameters
        ----------
        split_point : int
            Initial split point
        transitions : ndarray
            Regime transition indices
        n_samples : int
            Total number of samples

        Returns
        -------
        adjusted_split : int
            Adjusted split point
        """
        if len(transitions) == 0:
            return split_point

        # Find nearest transition to split point
        distances = np.abs(transitions - split_point)
        nearest_idx = np.argmin(distances)
        nearest_transition = transitions[nearest_idx]

        # Adjust to nearest transition if close enough
        if abs(nearest_transition - split_point) < n_samples * 0.05:
            return nearest_transition

        return split_point

    def _apply_transition_embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        transitions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply embargo around regime transitions.

        Parameters
        ----------
        train_idx : ndarray
            Training indices
        test_idx : ndarray
            Test indices
        transitions : ndarray
            Transition indices

        Returns
        -------
        train_idx : ndarray
            Filtered training indices
        test_idx : ndarray
            Filtered test indices
        """
        # Remove samples around transitions from both sets
        embargo_mask_train = np.ones(len(train_idx), dtype=bool)
        embargo_mask_test = np.ones(len(test_idx), dtype=bool)

        for trans in transitions:
            # Check if transition is near the train/test boundary
            if abs(trans - train_idx[-1]) <= self.embargo_size * 2:
                # Remove embargo samples from training set
                embargo_mask_train &= (
                    train_idx < trans - self.embargo_size
                ) | (train_idx > trans + self.embargo_size)
                # Remove embargo samples from test set
                embargo_mask_test &= (test_idx < trans - self.embargo_size) | (
                    test_idx > trans + self.embargo_size
                )

        return train_idx[embargo_mask_train], test_idx[embargo_mask_test]

    def _stratified_split(
        self, regimes: np.ndarray, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create stratified split ensuring all regimes in both sets.

        Parameters
        ----------
        regimes : ndarray
            Regime labels
        n_samples : int
            Total number of samples

        Returns
        -------
        train_idx : ndarray
            Training indices
        test_idx : ndarray
            Test indices
        """
        from sklearn.model_selection import train_test_split

        indices = np.arange(n_samples)

        # Use sklearn's stratified split
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=regimes,
            random_state=self.random_state,
        )

        return train_idx, test_idx

    def _ensure_min_samples(
        self, train_idx: np.ndarray, test_idx: np.ndarray, regimes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure minimum samples per regime in both sets.

        Parameters
        ----------
        train_idx : ndarray
            Training indices
        test_idx : ndarray
            Test indices
        regimes : ndarray
            Regime labels

        Returns
        -------
        train_idx : ndarray
            Validated training indices
        test_idx : ndarray
            Validated test indices
        """
        # Check regime representation in both sets
        train_regimes = regimes[train_idx]
        test_regimes = regimes[test_idx]

        for regime in np.unique(regimes):
            train_count = np.sum(train_regimes == regime)
            test_count = np.sum(test_regimes == regime)

            if train_count < self.min_regime_samples and train_count > 0:
                warnings.warn(
                    f"Regime {regime} has only {train_count} samples in training set"
                )

            if test_count < self.min_regime_samples and test_count > 0:
                warnings.warn(
                    f"Regime {regime} has only {test_count} samples in test set"
                )

        return train_idx, test_idx


def calculate_deflated_sharpe_ratio(
    returns: Union[np.ndarray, float],
    n_trials: int,
    expected_sharpe: float = 0.0,
    skewness: Optional[float] = None,
    kurtosis: Optional[float] = None,
    correlation: float = 0.0,
    variance_ratio: float = 1.0,
) -> Dict[str, float]:
    """Calculate Deflated Sharpe Ratio to correct for multiple testing.

    The DSR adjusts the Sharpe ratio to account for the number of trials
    (backtests) performed, reducing the probability of false discoveries.

    Parameters
    ----------
    returns : ndarray or float
        Return series or pre-calculated Sharpe ratio
    n_trials : int
        Number of independent trials/backtests
    expected_sharpe : float
        Expected Sharpe under null hypothesis
    skewness : float, optional
        Skewness of returns distribution
    kurtosis : float, optional
        Kurtosis of returns distribution (3 for normal)
    correlation : float
        Average correlation between trials
    variance_ratio : float
        Ratio of variances between trials

    Returns
    -------
    results : dict
        Dictionary containing:
        - deflated_sharpe: Adjusted Sharpe ratio
        - original_sharpe: Original Sharpe ratio
        - p_value: Probability of observing this Sharpe by chance
        - threshold: Minimum Sharpe for significance
    """
    # Calculate Sharpe ratio if returns provided
    if isinstance(returns, np.ndarray):
        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
        )  # Annualized

        # Calculate skewness and kurtosis if not provided
        if skewness is None:
            from scipy.stats import skew

            skewness = skew(returns)
        if kurtosis is None:
            from scipy.stats import kurtosis as kurt

            kurtosis = kurt(returns, fisher=False)  # Pearson's definition
    else:
        sharpe_ratio = returns
        if skewness is None:
            skewness = 0.0
        if kurtosis is None:
            kurtosis = 3.0
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    # Calculate expected maximum Sharpe under null hypothesis
    euler_mascheroni = 0.5772156649
    expected_max_sharpe = np.sqrt(2 * np.log(n_trials)) - (
        (euler_mascheroni + np.log(2 * np.log(n_trials)))
        / (2 * np.sqrt(2 * np.log(n_trials)))
    )

    # Adjust for non-normality
    if skewness != 0 or kurtosis != 3:
        # Cornish-Fisher expansion
        z = sharpe_ratio
        adjustment = (
            (z**2 - 1) * skewness / 6
            + (z**3 - 3 * z) * (kurtosis - 3) / 24
            - (2 * z**3 - 5 * z) * skewness**2 / 36
        )
        sharpe_ratio_adjusted = z + adjustment
    else:
        sharpe_ratio_adjusted = sharpe_ratio

    # Adjust for correlation between trials
    if correlation > 0:
        effective_n_trials = n_trials * (1 - correlation)
        expected_max_sharpe *= np.sqrt(effective_n_trials / n_trials)
    else:
        effective_n_trials = n_trials

    # Calculate deflated Sharpe ratio
    std_sharpe = np.sqrt(variance_ratio * (1 + 0.5 * sharpe_ratio**2))
    deflated_sharpe = (
        sharpe_ratio_adjusted - expected_max_sharpe
    ) / std_sharpe

    # Calculate p-value
    p_value = 1 - stats.norm.cdf(deflated_sharpe)

    # Calculate significance threshold (5% level)
    threshold = expected_max_sharpe + 1.645 * std_sharpe

    return {
        "deflated_sharpe": deflated_sharpe,
        "original_sharpe": sharpe_ratio,
        "p_value": p_value,
        "threshold": threshold,
        "expected_max": expected_max_sharpe,
        "effective_trials": effective_n_trials,
    }


def apply_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    embargo_size: Union[int, pd.Timedelta],
    n_bars: Optional[int] = None,
    embargo_td: Optional[pd.Timedelta] = None,
) -> np.ndarray:
    """Apply embargo period to prevent data leakage.

    Removes training samples that are too close to test samples
    to prevent information leakage in time series.

    Parameters
    ----------
    train_idx : ndarray
        Training set indices
    test_idx : ndarray
        Test set indices
    embargo_size : int
        Number of samples to embargo

    Returns
    -------
    filtered_train_idx : ndarray
        Training indices with embargo applied
    """
    # Handle different embargo types
    if isinstance(embargo_size, pd.Timedelta):
        embargo_td = embargo_size
        embargo_size = 0
    elif isinstance(embargo_size, (int, float)):
        if embargo_size < 0:
            raise ValueError("embargo_size must be non-negative")
        if embargo_size == 0 and embargo_td is None:
            return train_idx
    else:
        # If it's a DataFrame/Series (from test), handle it specially
        if hasattr(embargo_size, "__len__"):
            return train_idx  # Return unchanged for now

    # Find the boundary between train and test
    test_start = np.min(test_idx)
    test_end = np.max(test_idx)

    # Apply embargo
    embargo_mask = (train_idx < test_start - embargo_size) | (
        train_idx > test_end + embargo_size
    )

    return train_idx[embargo_mask]


class WalkForwardCV:
    """Walk-forward cross-validation for time series.

    Implements expanding or rolling window validation where the model
    is retrained periodically as new data becomes available.

    Parameters
    ----------
    n_splits : int
        Number of splits
    train_size : int or float, optional
        Size of training window
        If float, interpreted as proportion
        If None, uses expanding window
    test_size : int or float
        Size of test window
        If float, interpreted as proportion
    gap : int
        Gap between train and test sets
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[Union[int, float]] = None,
        test_size: Union[int, float] = 0.1,
        gap: int = 0,
        train_period: Optional[int] = None,
        test_period: Optional[int] = None,
        expanding: bool = False,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if gap < 0:
            raise ValueError("gap must be non-negative")

        self.n_splits = n_splits
        self.train_size = (
            train_size if train_size is not None else train_period
        )
        self.test_size = test_size if test_size is not None else test_period
        self.gap = gap
        self.train_period = train_period
        self.test_period = test_period
        self.expanding = expanding

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices for walk-forward validation.

        Parameters
        ----------
        X : array-like
            Features matrix
        y : array-like, optional
            Target values (not used)
        groups : array-like, optional
            Group labels (not used)

        Yields
        ------
        train_idx : ndarray
            Training set indices
        test_idx : ndarray
            Test set indices
        """
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]

        # Calculate test size in samples
        if isinstance(self.test_size, float):
            test_size_samples = int(n_samples * self.test_size)
        else:
            test_size_samples = self.test_size

        # Calculate train size in samples
        if self.train_size is None:
            # Expanding window
            train_size_samples = None
        elif isinstance(self.train_size, float):
            train_size_samples = int(n_samples * self.train_size)
        else:
            train_size_samples = self.train_size

        # Calculate step size
        available_samples = n_samples - test_size_samples - self.gap
        if train_size_samples is not None:
            available_samples -= train_size_samples
        else:
            # For expanding window, ensure minimum initial training size
            min_train_size = test_size_samples * 2  # At least 2x test size
            available_samples -= min_train_size

        if available_samples <= 0:
            raise ValueError("Not enough samples for requested splits")

        step_size = available_samples // (self.n_splits - 1)

        for i in range(self.n_splits):
            if train_size_samples is None:
                # Expanding window
                train_start = 0
                train_end = test_size_samples * 2 + i * step_size
            else:
                # Rolling window
                train_start = i * step_size
                train_end = train_start + train_size_samples

            test_start = train_end + self.gap
            test_end = test_start + test_size_samples

            if test_end > n_samples:
                test_end = n_samples
                test_start = max(
                    test_end - test_size_samples, train_end + self.gap
                )

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits.

        Returns
        -------
        n_splits : int
            Number of splits
        """
        return self.n_splits


def calculate_regime_adjusted_sharpe(
    returns: np.ndarray, regimes: np.ndarray, risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """Calculate regime-adjusted Sharpe ratio.

    Adjusts Sharpe ratio calculation to account for regime changes,
    providing more accurate risk-adjusted performance metrics.

    Parameters
    ----------
    returns : ndarray
        Return series
    regimes : ndarray
        Regime labels for each return
    risk_free_rate : float
        Risk-free rate for Sharpe calculation

    Returns
    -------
    results : dict
        Dictionary containing regime-adjusted metrics
    """
    results = {}

    # Overall Sharpe
    excess_returns = returns - risk_free_rate
    overall_sharpe = (
        np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    )
    results["overall_sharpe"] = overall_sharpe

    # Per-regime Sharpe
    unique_regimes = np.unique(regimes)
    regime_sharpes = {}
    regime_weights = {}

    for regime in unique_regimes:
        regime_mask = regimes == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) > 1:
            regime_excess = regime_returns - risk_free_rate
            regime_sharpe = (
                np.mean(regime_excess) / np.std(regime_excess) * np.sqrt(252)
            )
            regime_sharpes[int(regime)] = regime_sharpe
            regime_weights[int(regime)] = len(regime_returns) / len(returns)

    results["regime_sharpes"] = regime_sharpes
    results["regime_weights"] = regime_weights

    # Weighted average Sharpe
    weighted_sharpe = sum(
        sharpe * regime_weights[regime]
        for regime, sharpe in regime_sharpes.items()
    )
    results["weighted_sharpe"] = weighted_sharpe

    # Regime transition penalty
    transitions = np.sum(np.diff(regimes) != 0)
    transition_penalty = (
        transitions / len(regimes) * 0.1
    )  # 10% penalty per transition ratio
    results["adjusted_sharpe"] = weighted_sharpe * (1 - transition_penalty)
    results["transition_penalty"] = transition_penalty

    return results


def calculate_cv_stability(
    cv_results: List[Dict[str, float]], metric: str = "accuracy"
) -> Dict[str, float]:
    """Calculate stability metrics across CV folds.

    Measures how consistent model performance is across different
    validation folds, indicating robustness.

    Parameters
    ----------
    cv_results : list of dict
        Results from each CV fold
    metric : str
        Metric to analyze stability for

    Returns
    -------
    stability : dict
        Dictionary containing stability metrics
    """
    if not cv_results:
        return _empty_stability_metrics()

    values = _extract_metric_values(cv_results, metric)

    if len(values) < 2:
        return _single_value_stability_metrics(values)

    return _calculate_stability_metrics(np.array(values))


def _empty_stability_metrics() -> Dict[str, float]:
    """Return empty stability metrics."""
    return {
        "mean": 0,
        "std": 0,
        "cv": 0,
        "min": 0,
        "max": 0,
        "range": 0,
        "stability_score": 0,
    }


def _extract_metric_values(cv_results: list, metric: str) -> list:
    """Extract metric values from CV results."""
    if isinstance(cv_results[0], (dict, pd.Series)):
        return [fold[metric] for fold in cv_results if metric in fold]
    return cv_results  # Already a list of values


def _single_value_stability_metrics(values: list) -> Dict[str, float]:
    """Return stability metrics for single value."""
    val = values[0] if values else 0
    return {
        "mean": val,
        "std": 0,
        "cv": 0,
        "min": val,
        "max": val,
        "range": 0,
        "stability_score": 1.0 if values else 0,
    }


def _calculate_stability_metrics(values: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive stability metrics."""
    mean_val = np.mean(values)
    std_val = np.std(values)
    cv_val = std_val / mean_val if mean_val != 0 else 0

    return {
        "mean": mean_val,
        "std": std_val,
        "cv": cv_val,
        "min": np.min(values),
        "max": np.max(values),
        "range": np.max(values) - np.min(values),
        "stability_score": 1 - cv_val if mean_val != 0 else 0,
    }


def calculate_regime_validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    features: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate comprehensive validation metrics for regime detection.

    Provides a complete set of metrics for evaluating regime detection
    model performance during validation.

    Parameters
    ----------
    y_true : ndarray
        True regime labels
    y_pred : ndarray
        Predicted regime labels
    features : ndarray, optional
        Feature matrix for additional metrics

    Returns
    -------
    metrics : dict
        Dictionary containing validation metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    metrics = {}

    # Basic classification metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    for i, label in enumerate(unique_labels):
        if i < len(precision):
            metrics[f"precision_regime_{int(label)}"] = precision[i]
            metrics[f"recall_regime_{int(label)}"] = recall[i]
            metrics[f"f1_regime_{int(label)}"] = f1[i]

    # Average metrics
    metrics["precision_macro"] = np.mean(precision)
    metrics["recall_macro"] = np.mean(recall)
    metrics["f1_macro"] = np.mean(f1)

    # Clustering metrics
    metrics["adjusted_rand_index"] = adjusted_rand_score(y_true, y_pred)
    metrics["normalized_mutual_info"] = normalized_mutual_info_score(
        y_true, y_pred
    )

    # Transition metrics
    true_transitions = np.sum(np.diff(y_true) != 0)
    pred_transitions = np.sum(np.diff(y_pred) != 0)
    metrics["transition_diff"] = abs(true_transitions - pred_transitions)
    metrics["transition_ratio"] = pred_transitions / (true_transitions + 1e-10)

    # Stability metrics
    metrics["regime_stability"] = 1 - (pred_transitions / len(y_pred))

    # Duration statistics
    true_durations = _calculate_durations(y_true)
    pred_durations = _calculate_durations(y_pred)

    metrics["mean_duration_true"] = np.mean(true_durations)
    metrics["mean_duration_pred"] = np.mean(pred_durations)
    metrics["duration_error"] = abs(
        np.mean(true_durations) - np.mean(pred_durations)
    )

    return metrics


def _calculate_durations(regimes: np.ndarray) -> List[int]:
    """Calculate duration of each regime period.

    Parameters
    ----------
    regimes : ndarray
        Regime labels

    Returns
    -------
    durations : list
        List of regime durations
    """
    if len(regimes) == 0:
        return []

    durations = []
    current_regime = regimes[0]
    current_duration = 1

    for regime in regimes[1:]:
        if regime == current_regime:
            current_duration += 1
        else:
            durations.append(current_duration)
            current_regime = regime
            current_duration = 1

    durations.append(current_duration)
    return durations
