"""Gaussian Mixture Model for regime detection."""

from typing import Optional, Tuple
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from marketregimeml.models.base import BaseRegimeDetector
from marketregimeml.utils.logging import get_logger
from marketregimeml.utils.regime_utils import RegimeReorderingUtils

logger = get_logger(__name__)


class GMMRegimeDetector(BaseRegimeDetector):
    """Gaussian Mixture Model for market regime detection.

    Uses GMM to identify market regimes based on the distribution
    of features without considering temporal dependencies.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        covariance_type: str = "full",
        max_iter: int = 100,
        tol: float = 1e-3,
        n_init: int = 10,
        init_params: str = "kmeans",
        random_state: Optional[int] = None,
        warm_start: bool = False,
        reg_covar: float = 1e-6,
        verbose: int = 0,
        fuzzy_matching: bool = False,
        fuzzy_threshold: float = 0.7,
        auto_optimize_regimes: bool = False,
        min_regimes: int = 2,
        max_regimes: int = 9,
        **kwargs,
    ):
        """Initialize GMM regime detector.

        Args:
            n_regimes: Number of mixture components (regimes, default 5 based on benchmark optimization)
            covariance_type: Type of covariance matrix ('full', 'tied', 'diag', 'spherical')
            max_iter: Maximum number of EM iterations
            tol: Convergence tolerance
            n_init: Number of initializations to perform
            init_params: Initialization method ('kmeans', 'random', 'k-means++', 'random_from_data')
            random_state: Random seed
            warm_start: If True, use previous solution as initialization
            reg_covar: Regularization added to diagonal of covariance
            verbose: Verbosity level
        """
        super().__init__(
            n_regimes=n_regimes,
            random_state=random_state,
            fuzzy_matching=fuzzy_matching,
            fuzzy_threshold=fuzzy_threshold,
            auto_optimize_regimes=auto_optimize_regimes,
            min_regimes=min_regimes,
            max_regimes=max_regimes,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            init_params=init_params,
            warm_start=warm_start,
            reg_covar=reg_covar,
            verbose=verbose,
            **kwargs,
        )

        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_params = init_params
        self.warm_start = warm_start
        self.reg_covar = reg_covar
        self.verbose = verbose

        # Scaler for feature normalization
        self.scaler = StandardScaler()

        # Store model selection criteria
        self.aic_scores = []
        self.bic_scores = []

        # Optimal number of components analysis
        self.optimal_n_analysis = None

    def fit(
        self,
        features: pd.DataFrame,
        select_optimal_n: bool = False,
        max_components: int = 10,
    ) -> "GMMRegimeDetector":
        """Fit GMM model to features.

        Args:
            features: Feature matrix
            select_optimal_n: Whether to automatically select optimal number of components
            max_components: Maximum number of components to test

        Returns:
            Self for method chaining
        """
        # Convert to numpy and scale
        if hasattr(features, "values"):
            # DataFrame input
            X = features.values
            self.feature_names = list(features.columns)
        else:
            # Numpy array input
            X = features
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        X_scaled = self.scaler.fit_transform(X)

        # Store feature info
        self.n_features = X.shape[1]

        if select_optimal_n:
            # Find optimal number of components
            self._select_optimal_components(X_scaled, max_components)

        # Create and fit model
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            init_params=self.init_params,
            random_state=self.random_state,
            warm_start=self.warm_start,
            reg_covar=self.reg_covar,
            verbose=self.verbose,
        )

        # Fit model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.model.fit(X_scaled)

        # Check convergence
        if not self.model.converged_:
            logger.warning(
                f"GMM did not converge after {self.model.n_iter_} iterations"
            )

        # Store log-likelihood
        self.train_log_likelihood = self.model.score(X_scaled) * len(X_scaled)

        # Update diagnostics
        self._update_diagnostics(X_scaled)

        # Mark as fitted
        self.is_fitted = True
        self.fit_date = datetime.now()

        logger.info(
            f"GMM fitted with {self.n_regimes} regimes. "
            f"AIC: {self.diagnostics['aic']:.2f}, BIC: {self.diagnostics['bic']:.2f}"
        )

        return self

    def _select_optimal_components(
        self, X_scaled: np.ndarray, max_components: int = 10
    ):
        """Select optimal number of components using BIC.

        Args:
            X_scaled: Scaled feature matrix
            max_components: Maximum number of components to test
        """
        n_regimes_range = range(2, min(max_components + 1, len(X_scaled) // 10))

        aic_scores = []
        bic_scores = []

        for n_regimes_test in n_regimes_range:
            gmm = GaussianMixture(
                n_components=n_regimes_test,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                tol=self.tol,
                n_init=5,  # Fewer initializations for selection
                init_params=self.init_params,
                random_state=self.random_state,
                reg_covar=self.reg_covar,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                gmm.fit(X_scaled)

            aic = gmm.aic(X_scaled)
            bic = gmm.bic(X_scaled)

            aic_scores.append(aic)
            bic_scores.append(bic)

            logger.debug(f"n_regimes={n_regimes_test}: AIC={aic:.2f}, BIC={bic:.2f}")

        # Select based on BIC (more conservative)
        if len(bic_scores) > 0:
            optimal_idx = np.argmin(bic_scores)
            optimal_n = list(n_regimes_range)[optimal_idx]
        else:
            # No valid range to test
            optimal_n = self.n_regimes

        # Store analysis results
        self.optimal_n_analysis = {
            "n_regimes_range": list(n_regimes_range),
            "aic_scores": aic_scores,
            "bic_scores": bic_scores,
            "optimal_n": optimal_n,
            "selection_criterion": "BIC",
        }

        # Update n_regimes if different
        if optimal_n != self.n_regimes:
            logger.info(
                f"Optimal number of components: {optimal_n} (was {self.n_regimes})"
            )
            self.n_regimes = optimal_n
            self.regime_names = self._default_regime_names()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime labels.

        Args:
            features: Feature matrix

        Returns:
            Array of regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler.transform(X)

        # Special case for single regime
        if self.n_regimes == 1:
            return np.zeros(len(X_scaled), dtype=int)

        regimes = self.model.predict(X_scaled)

        # Reorder regimes by mean of first feature
        regimes = self._reorder_regimes(regimes, X_scaled)

        return regimes

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities.

        Args:
            features: Feature matrix

        Returns:
            Array of regime probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler.transform(X)

        # Special case for single regime
        if self.n_regimes == 1:
            return np.ones((len(X_scaled), 1))

        proba = self.model.predict_proba(X_scaled)

        # Reorder probabilities
        proba = self._reorder_probabilities(proba, X_scaled)

        return proba

    def _reorder_regimes(self, regimes: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Reorder regimes by mean of first feature.

        Args:
            regimes: Original regime labels
            features: Feature matrix

        Returns:
            Reordered regime labels
        """
        return RegimeReorderingUtils.reorder_regimes_by_feature_mean(
            regimes, features, self.n_regimes, feature_index=0
        )

    def _reorder_probabilities(
        self, proba: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """Reorder probability columns.

        Args:
            proba: Original probabilities
            features: Feature matrix

        Returns:
            Reordered probabilities
        """
        # Get regime assignments
        regimes = np.argmax(proba, axis=1)

        return RegimeReorderingUtils.reorder_probabilities(
            proba, regimes, features, self.n_regimes, feature_index=0
        )

    def _compute_log_likelihood(self, features: pd.DataFrame) -> float:
        """Compute log-likelihood of the data.

        Args:
            features: Feature matrix

        Returns:
            Log-likelihood value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler.transform(X)

        # GMM score returns average log-likelihood per sample
        return self.model.score(X_scaled) * len(X_scaled)

    def _count_parameters(self) -> int:
        """Count the number of model parameters.

        Returns:
            Number of parameters
        """
        n_params = 0

        # Mixing weights
        n_params += self.n_regimes - 1

        # Means
        n_params += self.n_regimes * self.n_features

        # Covariances
        if self.covariance_type == "full":
            n_params += self.n_regimes * self.n_features * (self.n_features + 1) // 2
        elif self.covariance_type == "diag":
            n_params += self.n_regimes * self.n_features
        elif self.covariance_type == "tied":
            n_params += self.n_features * (self.n_features + 1) // 2
        else:  # spherical
            n_params += self.n_regimes

        return n_params

    def _update_diagnostics(self, features: np.ndarray):
        """Update model diagnostics.

        Args:
            features: Scaled feature matrix
        """
        # Get predictions
        regimes = self.model.predict(features)
        proba = self.model.predict_proba(features)

        # Regime statistics
        self.diagnostics["regime_counts"] = np.bincount(
            regimes, minlength=self.n_regimes
        )
        self.diagnostics["regime_proportions"] = self.diagnostics[
            "regime_counts"
        ] / len(regimes)

        # Mixture weights
        self.diagnostics["weights"] = self.model.weights_

        # Component means
        self.diagnostics["means"] = self.model.means_

        # Component covariances
        if self.covariance_type == "full":
            self.diagnostics["covariances"] = self.model.covariances_
        elif self.covariance_type == "diag":
            self.diagnostics["covariances"] = self.model.covariances_
        elif self.covariance_type == "tied":
            self.diagnostics["covariances"] = self.model.covariances_
        else:  # spherical
            self.diagnostics["covariances"] = self.model.covariances_

        # Convergence info
        self.diagnostics["converged"] = self.model.converged_
        self.diagnostics["n_iter"] = self.model.n_iter_

        # Model selection criteria
        self.diagnostics["aic"] = self.model.aic(features)
        self.diagnostics["bic"] = self.model.bic(features)
        self.diagnostics["log_likelihood"] = self.model.score(features) * len(features)

        # Separation quality
        self._calculate_separation_metrics(features, regimes, proba)

        # Store optimal n analysis if performed
        if self.optimal_n_analysis:
            self.diagnostics["optimal_n_analysis"] = self.optimal_n_analysis

    def _calculate_separation_metrics(
        self, features: np.ndarray, regimes: np.ndarray, proba: np.ndarray
    ):
        """Calculate regime separation quality metrics.

        Args:
            features: Feature matrix
            regimes: Regime assignments
            proba: Regime probabilities
        """
        # Silhouette score
        from sklearn.metrics import silhouette_score

        if self.n_regimes > 1 and len(np.unique(regimes)) > 1:
            silhouette = silhouette_score(features, regimes)
            self.diagnostics["silhouette_score"] = silhouette
        else:
            self.diagnostics["silhouette_score"] = 0

        # Davies-Bouldin score (lower is better)
        from sklearn.metrics import davies_bouldin_score

        if self.n_regimes > 1 and len(np.unique(regimes)) > 1:
            db_score = davies_bouldin_score(features, regimes)
            self.diagnostics["davies_bouldin_score"] = db_score
        else:
            self.diagnostics["davies_bouldin_score"] = np.inf

        # Calinski-Harabasz score (higher is better)
        from sklearn.metrics import calinski_harabasz_score

        if self.n_regimes > 1 and len(np.unique(regimes)) > 1:
            ch_score = calinski_harabasz_score(features, regimes)
            self.diagnostics["calinski_harabasz_score"] = ch_score
        else:
            self.diagnostics["calinski_harabasz_score"] = 0

        # Average confidence (max probability)
        avg_confidence = np.mean(np.max(proba, axis=1))
        self.diagnostics["avg_confidence"] = avg_confidence

        # Entropy of assignments
        entropy = -np.mean(np.sum(proba * np.log(proba + 1e-10), axis=1))
        self.diagnostics["avg_entropy"] = entropy

    def sample(
        self, n_samples: int = 100, random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate samples from the fitted model.

        Args:
            n_samples: Number of samples to generate
            random_state: Random state for sampling

        Returns:
            Tuple of (features, regimes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")

        # Generate samples
        X_scaled, regimes = self.model.sample(n_samples)

        # Inverse transform features
        X = self.scaler.inverse_transform(X_scaled)

        # Create DataFrame
        features = pd.DataFrame(X, columns=self.feature_names)

        # Reorder regimes
        regimes = self._reorder_regimes(regimes, X_scaled)

        return features, regimes

    def get_mahalanobis_distance(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate Mahalanobis distance to each component.

        Args:
            features: Feature matrix

        Returns:
            DataFrame with distances to each component
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler.transform(X)

        distances = np.zeros((len(X_scaled), self.n_regimes))

        for i in range(self.n_regimes):
            mean = self.model.means_[i]

            if self.covariance_type == "full":
                cov = self.model.covariances_[i]
            elif self.covariance_type == "diag":
                cov = np.diag(self.model.covariances_[i])
            elif self.covariance_type == "tied":
                cov = self.model.covariances_
            else:  # spherical
                cov = self.model.covariances_[i] * np.eye(self.n_features)

            # Calculate Mahalanobis distance
            diff = X_scaled - mean
            inv_cov = np.linalg.inv(cov)

            for j in range(len(X_scaled)):
                d = diff[j]
                distances[j, i] = np.sqrt(np.dot(np.dot(d, inv_cov), d))

        # Create DataFrame
        distance_df = pd.DataFrame(
            distances,
            index=features.index,
            columns=[f"distance_regime_{i}" for i in range(self.n_regimes)],
        )

        return distance_df

    def get_outlier_scores(
        self, features: pd.DataFrame, threshold_percentile: float = 95
    ) -> pd.Series:
        """Calculate outlier scores based on likelihood.

        Args:
            features: Feature matrix
            threshold_percentile: Percentile for outlier threshold

        Returns:
            Series with outlier scores (higher = more outlier-like)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler.transform(X)

        # Get log-likelihood for each sample
        log_proba = self.model.score_samples(X_scaled)

        # Convert to outlier scores (negative log-likelihood)
        outlier_scores = -log_proba

        # Calculate threshold
        threshold = np.percentile(outlier_scores, threshold_percentile)

        # Create Series
        scores = pd.Series(outlier_scores, index=features.index, name="outlier_score")

        # Add threshold info
        scores.attrs["threshold"] = threshold
        scores.attrs["threshold_percentile"] = threshold_percentile

        return scores
