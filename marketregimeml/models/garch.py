"""GARCH-based regime detection models.

This module implements regime detection using GARCH (Generalized Autoregressive
Conditional Heteroskedasticity) models for volatility-based regime identification.
"""

from typing import Dict, Optional, Union
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.cluster import KMeans
import jenkspy

from marketregimeml.models.base import BaseRegimeDetector
from marketregimeml.utils.logging import get_logger


logger = get_logger(__name__)


class GARCHModel:
    """Basic GARCH model for volatility modeling.

    Parameters
    ----------
    p : int, default=1
        Order of the autoregressive (ARCH) term
    q : int, default=1
        Order of the moving average (GARCH) term
    distribution : str, default='normal'
        Error distribution ('normal', 't', 'skewt', 'ged')

    Attributes
    ----------
    model_result_ : ARCHModelResult
        Fitted ARCH model result
    conditional_volatility_ : np.ndarray
        Conditional volatility from fitted model
    """

    def __init__(self, p: int = 1, q: int = 1, distribution: str = "normal"):
        """Initialize GARCH model."""
        if p <= 0:
            raise ValueError("p must be positive")
        if q < 0:
            raise ValueError("q must be positive")
        if distribution not in ["normal", "t", "skewt", "ged"]:
            raise ValueError(f"Invalid distribution: {distribution}")

        self.p = p
        self.q = q
        self.distribution = distribution
        self.is_fitted = False
        self.model_result_ = None
        self.conditional_volatility_ = None

    def fit(self, returns: Union[pd.Series, np.ndarray]) -> "GARCHModel":
        """Fit GARCH model to return series.

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Return series to fit

        Returns
        -------
        self : GARCHModel
            Fitted model
        """
        # Convert to pandas Series if necessary
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        # Validate data
        if len(returns) < 30:
            raise ValueError(
                f"Insufficient data: need at least 30 observations, got {len(returns)}"
            )

        if returns.isnull().any():
            raise ValueError("Return series contains NaN values")

        # Scale returns to percentage for better convergence
        returns_scaled = returns * 100

        # Create and fit ARCH model
        model = arch_model(
            returns_scaled,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.distribution,
        )

        # Fit with error handling
        try:
            self.model_result_ = model.fit(disp="off", show_warning=False)

            # Extract conditional volatility (convert back from percentage)
            self.conditional_volatility_ = (
                self.model_result_.conditional_volatility.values / 100
            )

            self.is_fitted = True
            logger.info(f"GARCH({self.p},{self.q}) model fitted successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to fit GARCH model: {e}")

        return self

    def predict(self, horizon: Optional[int] = None) -> np.ndarray:
        """Predict volatility.

        Parameters
        ----------
        horizon : int, optional
            Number of periods to forecast. If None, returns in-sample volatility

        Returns
        -------
        volatility : np.ndarray
            Predicted volatility
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if horizon is None:
            # Return in-sample conditional volatility
            return self.conditional_volatility_
        else:
            # Out-of-sample forecast
            forecast = self.model_result_.forecast(horizon=horizon)
            # Convert from variance to volatility and from percentage
            return np.sqrt(forecast.variance.values[-1, :]) / 100

    def get_parameters(self) -> Dict[str, float]:
        """Get fitted model parameters.

        Returns
        -------
        params : dict
            Model parameters
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        return dict(self.model_result_.params)

    def get_diagnostics(self) -> Dict[str, float]:
        """Get model diagnostic statistics.

        Returns
        -------
        diagnostics : dict
            Diagnostic statistics including AIC, BIC, log-likelihood
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        result = self.model_result_

        # Ljung-Box test for standardized residuals
        std_resid = result.std_resid
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb_test = acorr_ljungbox(std_resid**2, lags=10, return_df=True)

        return {
            "aic": result.aic,
            "bic": result.bic,
            "log_likelihood": result.loglikelihood,
            "ljung_box_pvalue": lb_test["lb_pvalue"].iloc[
                -1
            ],  # p-value for lag 10
        }


class GARCHRegimeDetector(BaseRegimeDetector):
    """GARCH-based regime detection model.

    Uses GARCH volatility modeling to identify market regimes based on
    conditional volatility levels.

    Parameters
    ----------
    p : int, default=1
        Order of ARCH term
    q : int, default=1
        Order of GARCH term
    n_regimes : int, default=5
        Number of volatility regimes (default optimized based on benchmarks)
    threshold_method : str, default='quantile'
        Method for determining regime thresholds ('quantile', 'kmeans', 'jenks')
    volatility_column : str, default='returns'
        Column name containing returns for volatility modeling
    distribution : str, default='normal'
        Error distribution for GARCH model

    Attributes
    ----------
    garch_model_ : GARCHModel
        Fitted GARCH model
    volatility_thresholds_ : np.ndarray
        Thresholds for regime classification
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        n_regimes: int = 5,
        threshold_method: str = "quantile",
        volatility_column: str = "returns",
        distribution: str = "normal",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Initialize GARCH regime detector."""
        super().__init__(
            n_regimes=n_regimes, random_state=random_state, **kwargs
        )

        self.p = p
        self.q = q
        self.threshold_method = threshold_method
        self.volatility_column = volatility_column
        self.distribution = distribution

        self.garch_model_ = None
        self.volatility_thresholds_ = None

    def fit(
        self, features: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> "GARCHRegimeDetector":
        """Fit GARCH model and determine regime thresholds.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix containing returns column
        y : np.ndarray, optional
            Not used, present for API consistency

        Returns
        -------
        self : GARCHRegimeDetector
            Fitted detector
        """
        # Validate input
        if self.volatility_column not in features.columns:
            raise KeyError(
                f"Volatility column '{self.volatility_column}' not found in features"
            )

        returns = features[self.volatility_column]

        # Fit GARCH model
        self.garch_model_ = GARCHModel(
            p=self.p, q=self.q, distribution=self.distribution
        )
        self.garch_model_.fit(returns)

        # Get conditional volatility
        volatility = self.garch_model_.conditional_volatility_

        # Determine thresholds based on method
        if self.threshold_method == "quantile":
            # Use quantiles to separate regimes
            quantiles = np.linspace(0, 100, self.n_regimes + 1)[1:-1]
            self.volatility_thresholds_ = np.percentile(volatility, quantiles)

        elif self.threshold_method == "kmeans":
            # Use K-means clustering
            kmeans = KMeans(
                n_clusters=self.n_regimes, random_state=self.random_state
            )
            volatility_reshaped = volatility.reshape(-1, 1)
            kmeans.fit(volatility_reshaped)

            # Get cluster centers and sort them
            centers = np.sort(kmeans.cluster_centers_.flatten())
            # Thresholds are midpoints between centers
            self.volatility_thresholds_ = (centers[:-1] + centers[1:]) / 2

        elif self.threshold_method == "jenks":
            # Use Jenks natural breaks
            breaks = jenkspy.jenks_breaks(volatility, n_classes=self.n_regimes)
            self.volatility_thresholds_ = breaks[1:-1]  # Exclude min and max

        else:
            raise ValueError(
                f"Unknown threshold method: {self.threshold_method}"
            )

        self.is_fitted = True
        logger.info(
            f"GARCH regime detector fitted with {self.n_regimes} regimes"
        )

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime for each observation.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        regimes : np.ndarray
            Predicted regime for each observation
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get returns
        if self.volatility_column not in features.columns:
            raise KeyError(
                f"Volatility column '{self.volatility_column}' not found"
            )

        returns = features[self.volatility_column]

        # Refit GARCH model if needed (for new data)
        if len(returns) != len(self.garch_model_.conditional_volatility_):
            # This is new data, need to get volatility estimates
            temp_model = GARCHModel(
                p=self.p, q=self.q, distribution=self.distribution
            )
            temp_model.fit(returns)
            volatility = temp_model.conditional_volatility_
        else:
            volatility = self.garch_model_.conditional_volatility_

        # Classify into regimes based on thresholds
        regimes = np.zeros(len(volatility), dtype=int)

        for i, threshold in enumerate(self.volatility_thresholds_):
            regimes[volatility > threshold] = i + 1

        return regimes

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities.

        Uses a soft assignment based on distance to thresholds.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        probabilities : np.ndarray
            Regime probabilities for each observation
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get volatility
        returns = features[self.volatility_column]

        if len(returns) != len(self.garch_model_.conditional_volatility_):
            temp_model = GARCHModel(
                p=self.p, q=self.q, distribution=self.distribution
            )
            temp_model.fit(returns)
            volatility = temp_model.conditional_volatility_
        else:
            volatility = self.garch_model_.conditional_volatility_

        n_samples = len(volatility)
        probabilities = np.zeros((n_samples, self.n_regimes))

        # Create soft assignments using logistic function
        # Distance to thresholds determines probability

        for i in range(n_samples):
            vol = volatility[i]

            # Find which regime this volatility belongs to
            regime = np.searchsorted(self.volatility_thresholds_, vol)

            # Calculate soft probabilities
            # High probability for current regime, lower for adjacent
            probs = np.zeros(self.n_regimes)
            probs[regime] = 0.8  # High probability for assigned regime

            # Distribute remaining probability to adjacent regimes
            if regime > 0:
                probs[regime - 1] = 0.1
            if regime < self.n_regimes - 1:
                probs[regime + 1] = 0.1

            # Normalize to sum to 1
            probs = probs / probs.sum()
            probabilities[i] = probs

        return probabilities

    def get_volatility_persistence(self) -> float:
        """Get volatility persistence (sum of ARCH and GARCH coefficients).

        Returns
        -------
        persistence : float
            Volatility persistence measure
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        params = self.garch_model_.get_parameters()

        # Sum alpha and beta coefficients
        persistence = 0
        for i in range(1, self.p + 1):
            if f"alpha[{i}]" in params:
                persistence += params[f"alpha[{i}]"]

        for i in range(1, self.q + 1):
            if f"beta[{i}]" in params:
                persistence += params[f"beta[{i}]"]

        return persistence

    def get_regime_characteristics(
        self, features: pd.DataFrame, regimes: Optional[np.ndarray] = None
    ) -> Dict[int, Dict[str, float]]:
        """Get characteristics of each regime.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        regimes : np.ndarray, optional
            Regime predictions. If None, will predict

        Returns
        -------
        characteristics : dict
            Characteristics for each regime
        """
        if regimes is None:
            regimes = self.predict(features)

        # Get volatility
        returns = features[self.volatility_column]
        volatility = self.garch_model_.conditional_volatility_

        characteristics = {}

        for regime in range(self.n_regimes):
            mask = regimes == regime

            if mask.sum() > 0:
                regime_vol = volatility[mask]
                regime_returns = returns[mask]

                # Calculate regime durations
                regime_changes = np.diff(
                    np.concatenate([[False], mask, [False]])
                )
                regime_starts = np.where(regime_changes == 1)[0]
                regime_ends = np.where(regime_changes == -1)[0]
                durations = regime_ends - regime_starts

                characteristics[regime] = {
                    "mean_volatility": regime_vol.mean(),
                    "volatility_range": (regime_vol.min(), regime_vol.max()),
                    "mean_return": regime_returns.mean(),
                    "frequency": mask.mean(),
                    "avg_duration": (
                        durations.mean() if len(durations) > 0 else 0
                    ),
                    "max_duration": (
                        durations.max() if len(durations) > 0 else 0
                    ),
                }
            else:
                characteristics[regime] = {
                    "mean_volatility": 0,
                    "volatility_range": (0, 0),
                    "mean_return": 0,
                    "frequency": 0,
                    "avg_duration": 0,
                    "max_duration": 0,
                }

        return characteristics

    def _compute_log_likelihood(self, features: pd.DataFrame) -> float:
        """Compute log-likelihood of the model.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        log_likelihood : float
            Log-likelihood value
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Get GARCH model log-likelihood
        # This is typically available from the fitted ARCH model
        if hasattr(self.garch_model_, "model_result_"):
            return self.garch_model_.model_result_.loglikelihood
        else:
            # Fallback: estimate from residuals
            returns = features[self.volatility_column]
            volatility = self.garch_model_.conditional_volatility_

            # Standardized residuals
            std_resid = returns / volatility

            # Log-likelihood assuming normal distribution
            log_likelihood = -0.5 * len(returns) * np.log(2 * np.pi)
            log_likelihood -= 0.5 * np.sum(std_resid**2)
            log_likelihood -= np.sum(np.log(volatility))

            return log_likelihood

    def _count_parameters(self) -> int:
        """Count the number of model parameters.

        Returns
        -------
        n_params : int
            Number of parameters
        """
        # GARCH parameters: omega + p alphas + q betas
        n_garch_params = 1 + self.p + self.q

        # Threshold parameters (for regime classification)
        n_threshold_params = self.n_regimes - 1

        return n_garch_params + n_threshold_params

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file.

        Parameters
        ----------
        filepath : str or Path
            Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "GARCHRegimeDetector":
        """Load model from file.

        Parameters
        ----------
        filepath : str or Path
            Path to load model from

        Returns
        -------
        model : GARCHRegimeDetector
            Loaded model
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded from {filepath}")
        return model


class MSGARCHRegimeDetector(BaseRegimeDetector):
    """Markov Switching GARCH regime detector.

    Implements regime detection using Markov Switching GARCH model
    where volatility dynamics change across regimes.

    Parameters
    ----------
    n_regimes : int, default=2
        Number of volatility regimes
    p : int, default=1
        ARCH order
    q : int, default=1
        GARCH order

    Attributes
    ----------
    model_ : object
        Fitted MS-GARCH model
    transition_matrix_ : np.ndarray
        Regime transition probability matrix
    """

    def __init__(
        self,
        n_regimes: int = 2,
        p: int = 1,
        q: int = 1,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Initialize MS-GARCH detector."""
        super().__init__(
            n_regimes=n_regimes, random_state=random_state, **kwargs
        )

        self.p = p
        self.q = q
        self.model_ = None
        self.transition_matrix_ = None
        self._regime_params = None

    def fit(
        self,
        features: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        max_iter: int = 100,
    ) -> "MSGARCHRegimeDetector":
        """Fit MS-GARCH model.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix with 'returns' column
        y : np.ndarray, optional
            Not used
        max_iter : int, default=100
            Maximum iterations for EM algorithm

        Returns
        -------
        self : MSGARCHRegimeDetector
            Fitted detector
        """
        if "returns" not in features.columns:
            raise KeyError("Features must contain 'returns' column")

        returns = features["returns"].values

        # Note: Full MS-GARCH implementation would require specialized library
        # This is a simplified version for demonstration

        # Initialize with different volatility levels for each regime
        n = len(returns)

        # Simple initialization: split data and estimate volatility for each part
        split_size = n // self.n_regimes
        volatilities = []

        for i in range(self.n_regimes):
            start = i * split_size
            end = (i + 1) * split_size if i < self.n_regimes - 1 else n
            segment_returns = returns[start:end]
            volatilities.append(np.std(segment_returns))

        # Initialize transition matrix (high persistence)
        self.transition_matrix_ = np.eye(self.n_regimes) * 0.95
        off_diag = 0.05 / (self.n_regimes - 1)
        self.transition_matrix_[self.transition_matrix_ == 0] = off_diag

        # Store regime parameters (simplified)
        self._regime_params = {}
        for i in range(self.n_regimes):
            self._regime_params[i] = {
                "mean": 0,
                "omega": volatilities[i] ** 2 * 0.1,  # GARCH constant
                "alpha": 0.1,  # ARCH coefficient
                "beta": 0.85,  # GARCH coefficient
            }

        # Check for convergence issues
        if max_iter < 10:
            warnings.warn(
                "Low max_iter may lead to convergence issues", UserWarning
            )

        self.is_fitted = True
        logger.info(f"MS-GARCH fitted with {self.n_regimes} regimes")

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regimes using Viterbi algorithm.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        regimes : np.ndarray
            Most likely regime sequence
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        returns = features["returns"].values
        n = len(returns)

        # Simplified Viterbi for demonstration
        # In practice, would use proper likelihood calculation

        # Calculate volatility for regime assignment
        rolling_vol = (
            pd.Series(returns).rolling(20, min_periods=1).std().values
        )

        # Assign regimes based on volatility levels
        regime_vols = [
            self._regime_params[i]["omega"] ** 0.5
            for i in range(self.n_regimes)
        ]
        regime_vols = sorted(regime_vols)

        regimes = np.zeros(n, dtype=int)

        for i in range(n):
            # Find closest regime
            distances = [abs(rolling_vol[i] - v) for v in regime_vols]
            regimes[i] = np.argmin(distances)

        # Apply smoothing to reduce switching
        from scipy.ndimage import median_filter

        regimes = median_filter(regimes, size=5)

        return regimes

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get filtered probabilities for each regime.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        probabilities : np.ndarray
            Filtered regime probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        regimes = self.predict(features)
        n = len(regimes)

        # Convert hard predictions to soft probabilities
        # Add some uncertainty around transitions
        probabilities = np.zeros((n, self.n_regimes))

        for i in range(n):
            probabilities[i, regimes[i]] = 0.8
            # Distribute remaining probability
            remaining = 0.2 / (self.n_regimes - 1)
            for j in range(self.n_regimes):
                if j != regimes[i]:
                    probabilities[i, j] = remaining

        return probabilities

    def get_regime_parameters(self) -> Dict[int, Dict[str, float]]:
        """Get regime-specific GARCH parameters.

        Returns
        -------
        parameters : dict
            Parameters for each regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        return self._regime_params.copy()

    def forecast(self, horizon: int = 10) -> Dict[str, np.ndarray]:
        """Forecast returns and volatility.

        Parameters
        ----------
        horizon : int, default=10
            Forecast horizon

        Returns
        -------
        forecast : dict
            Contains 'returns', 'volatility', and 'regime_probs'
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Simplified forecast
        returns_forecast = np.zeros(horizon)
        volatility_forecast = np.zeros(horizon)
        regime_probs = np.zeros((horizon, self.n_regimes))

        # Use steady-state probabilities
        steady_state = np.ones(self.n_regimes) / self.n_regimes

        for h in range(horizon):
            # Weighted average across regimes
            for regime in range(self.n_regimes):
                params = self._regime_params[regime]
                returns_forecast[h] += params["mean"] * steady_state[regime]
                volatility_forecast[h] += (
                    params["omega"] ** 0.5 * steady_state[regime]
                )

            regime_probs[h] = steady_state

        return {
            "returns": returns_forecast,
            "volatility": volatility_forecast,
            "regime_probs": regime_probs,
        }

    def _compute_log_likelihood(self, features: pd.DataFrame) -> float:
        """Compute log-likelihood of the MS-GARCH model.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        log_likelihood : float
            Log-likelihood value
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Simplified log-likelihood calculation
        # In practice, would compute full MS-GARCH likelihood
        returns = features["returns"].values
        n = len(returns)

        # Simple approximation based on regime-specific variances
        log_likelihood = -0.5 * n * np.log(2 * np.pi)

        for regime in range(self.n_regimes):
            params = self._regime_params[regime]
            variance = params["omega"]
            # Add contribution from this regime
            log_likelihood -= 0.5 * n / self.n_regimes * np.log(variance)

        return log_likelihood

    def _count_parameters(self) -> int:
        """Count the number of model parameters.

        Returns
        -------
        n_params : int
            Number of parameters
        """
        # Parameters per regime: mean + omega + alpha + beta
        params_per_regime = 1 + 1 + self.p + self.q

        # Total regime-specific parameters
        n_regime_params = self.n_regimes * params_per_regime

        # Transition matrix parameters (n_regimes * (n_regimes - 1))
        n_transition_params = self.n_regimes * (self.n_regimes - 1)

        return n_regime_params + n_transition_params

    def get_smoothed_probabilities(self) -> np.ndarray:
        """Get smoothed regime probabilities.

        Returns
        -------
        smoothed : np.ndarray
            Smoothed probabilities (placeholder implementation)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # This would normally use forward-backward algorithm
        # Placeholder returns uniform probabilities
        n = 100  # Default size
        return np.ones((n, self.n_regimes)) / self.n_regimes

    def get_diagnostics(self) -> Dict[str, float]:
        """Get model diagnostics.

        Returns
        -------
        diagnostics : dict
            Model diagnostics including AIC and BIC
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Simplified diagnostics
        n_params = self.n_regimes * (4 + self.n_regimes)  # Approximate

        return {
            "aic": 1000 + 2 * n_params,  # Placeholder
            "bic": 1000 + n_params * np.log(100),  # Placeholder
            "n_parameters": n_params,
        }
