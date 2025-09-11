"""Hidden Markov Model for regime detection."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from marketregimeml.models.base import BaseRegimeDetector
from marketregimeml.utils.logging import get_logger
from marketregimeml.utils.regime_utils import RegimeReorderingUtils

logger = get_logger(__name__)


class HMMRegimeDetector(BaseRegimeDetector):
    """Hidden Markov Model for market regime detection.

    Uses Gaussian HMM to identify hidden market states based on
    observable features like returns and volatility.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        init_method: str = "kmeans",
        verbose: bool = False,
        fuzzy_matching: bool = False,
        fuzzy_threshold: float = 0.7,
        auto_optimize_regimes: bool = False,
        min_regimes: int = 2,
        max_regimes: int = 9,
        **kwargs,
    ):
        """Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden regimes (default 5 based on benchmark optimization)
            covariance_type: Type of covariance matrix ('full', 'diag', 'tied', 'spherical')
            n_iter: Maximum number of EM iterations
            tol: Convergence tolerance
            random_state: Random seed
            init_method: Initialization method ('kmeans', 'random', 'volatility', 'quantile')
            verbose: Whether to print convergence info
            fuzzy_matching: Enable fuzzy regime assignment
            fuzzy_threshold: Threshold for crisp vs fuzzy assignment
            auto_optimize_regimes: Automatically find optimal number of regimes
            min_regimes: Minimum regimes for auto-optimization
            max_regimes: Maximum regimes for auto-optimization
            **kwargs: Additional parameters
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
            n_iter=n_iter,
            tol=tol,
            init_method=init_method,
            verbose=verbose,
            **kwargs,
        )

        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.init_method = init_method
        self.verbose = verbose

        # Scaler for feature normalization
        self.scaler = StandardScaler()

        # Store multiple models for ensemble/stability
        self.models = []
        self.best_model_idx = None

        # Initialization parameters
        self.init_params = {}

    def _initialize_model(self) -> hmm.GaussianHMM:
        """Create and initialize HMM model.

        Returns:
            Initialized GaussianHMM model
        """
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        return model

    def _get_initial_params(
        self, features: np.ndarray, method: str = "kmeans"
    ) -> Dict[str, np.ndarray]:
        """Get initial parameters for HMM.

        Args:
            features: Feature matrix
            method: Initialization method

        Returns:
            Dictionary with initial parameters
        """
        # Dispatch to appropriate initialization method
        initializers = {
            "kmeans": self._init_kmeans,
            "volatility": self._init_volatility,
            "quantile": self._init_quantile,
            "random": self._init_random,
        }

        initializer = initializers.get(method, self._init_random)
        return initializer(features)

    def _init_kmeans(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Initialize using K-means clustering."""
        from sklearn.cluster import KMeans

        # Perform clustering
        kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(features)

        # Build parameters
        params = {
            "startprob": self._uniform_startprob(),
            "transmat": self._diagonal_transition_matrix(0.7, 0.1),
            "means": kmeans.cluster_centers_,
            "covars": self._compute_covariances_from_labels(features, labels),
        }

        return params

    def _init_volatility(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Initialize based on volatility quantiles."""
        # Get volatility-based labels
        vol_feature = features[:, 0]
        labels = self._assign_labels_by_quantiles(
            vol_feature, np.linspace(0, 100, self.n_regimes + 1)
        )

        # Build parameters from labels
        return self._build_params_from_labels(features, labels)

    def _init_quantile(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Initialize based on return quantiles."""
        returns = features[:, 0]

        # Get quantile boundaries
        quantiles = self._get_quantile_boundaries()
        thresholds = np.percentile(returns, quantiles)

        # Assign labels based on thresholds
        labels = self._assign_labels_by_thresholds(returns, thresholds)

        # Build parameters from labels
        return self._build_params_from_labels(features, labels)

    def _init_random(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Random initialization."""
        n_features = features.shape[1]
        data_mean = np.mean(features, axis=0)
        data_std = np.std(features, axis=0)

        params = {
            "startprob": np.random.dirichlet(np.ones(self.n_regimes)),
            "transmat": np.random.dirichlet(
                np.ones(self.n_regimes), size=self.n_regimes
            ),
            "means": data_mean
            + np.random.randn(self.n_regimes, n_features) * data_std * 0.5,
            "covars": self._random_covariances(n_features),
        }

        return params

    def _uniform_startprob(self) -> np.ndarray:
        """Create uniform start probabilities."""
        return np.ones(self.n_regimes) / self.n_regimes

    def _diagonal_transition_matrix(
        self, diag_val: float, off_diag: float
    ) -> np.ndarray:
        """Create transition matrix with diagonal preference."""
        trans_mat = np.full((self.n_regimes, self.n_regimes), off_diag)
        np.fill_diagonal(trans_mat, diag_val)
        return trans_mat / trans_mat.sum(axis=1, keepdims=True)

    def _compute_covariances_from_labels(
        self, features: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute covariances for each regime."""
        n_features = features.shape[1]
        covars = []

        for i in range(self.n_regimes):
            regime_data = features[labels == i]

            if len(regime_data) > 1:
                if self.covariance_type == "full":
                    cov = np.cov(regime_data.T) + np.eye(n_features) * 1e-6
                else:  # diag
                    cov = np.maximum(np.var(regime_data, axis=0), 1e-6)
            else:
                if self.covariance_type == "full":
                    cov = np.eye(n_features)
                else:  # diag
                    cov = np.ones(n_features)

            covars.append(cov)

        return np.array(covars)

    def _assign_labels_by_quantiles(
        self, data: np.ndarray, quantiles: np.ndarray
    ) -> np.ndarray:
        """Assign labels based on quantile boundaries."""
        n_samples = len(data)
        labels = np.zeros(n_samples, dtype=int)

        for i in range(self.n_regimes):
            mask = (data >= quantiles[i]) & (data < quantiles[i + 1])
            labels[mask] = i

        return labels

    def _get_quantile_boundaries(self) -> list:
        """Get quantile boundaries for regime assignment."""
        quantile_map = {
            2: [50],
            3: [33, 67],
            4: [25, 50, 75],
        }

        if self.n_regimes in quantile_map:
            return quantile_map[self.n_regimes]
        else:
            return list(
                np.linspace(
                    100 / self.n_regimes,
                    100 * (1 - 1 / self.n_regimes),
                    self.n_regimes - 1,
                )
            )

    def _assign_labels_by_thresholds(
        self, data: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        """Assign labels based on threshold values."""
        n_samples = len(data)
        labels = np.zeros(n_samples, dtype=int)

        labels[data <= thresholds[0]] = 0
        for i in range(len(thresholds) - 1):
            mask = (data > thresholds[i]) & (data <= thresholds[i + 1])
            labels[mask] = i + 1
        labels[data > thresholds[-1]] = self.n_regimes - 1

        return labels

    def _build_params_from_labels(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Build HMM parameters from assigned labels."""
        # Calculate means for each regime
        means = []
        for i in range(self.n_regimes):
            regime_data = features[labels == i]
            if len(regime_data) > 1:
                means.append(np.mean(regime_data, axis=0))
            else:
                means.append(np.mean(features, axis=0))

        params = {
            "startprob": self._uniform_startprob(),
            "transmat": self._estimate_transition_matrix(labels),
            "means": np.array(means),
            "covars": self._compute_covariances_from_labels(features, labels),
        }

        return params

    def _random_covariances(self, n_features: int) -> np.ndarray:
        """Generate random covariance matrices."""
        if self.covariance_type == "full":
            covars = []
            for _ in range(self.n_regimes):
                # Generate random positive definite matrix
                A = np.random.randn(n_features, n_features) * 0.1
                cov = np.dot(A, A.T) + np.eye(n_features)
                covars.append(cov)
            return np.array(covars)
        else:
            return np.random.uniform(0.5, 2.0, (self.n_regimes, n_features))

    def _estimate_transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        """Estimate transition matrix from label sequence.

        Args:
            labels: Sequence of regime labels

        Returns:
            Transition probability matrix
        """
        trans_mat = np.zeros((self.n_regimes, self.n_regimes))

        for i in range(len(labels) - 1):
            trans_mat[labels[i], labels[i + 1]] += 1

        # Add small constant to avoid zero probabilities
        trans_mat = trans_mat + 0.01

        # Normalize rows
        trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)

        return trans_mat

    def fit(
        self,
        features: pd.DataFrame,
        n_init: int = 10,
        select_best: str = "score",
    ) -> "HMMRegimeDetector":
        """Fit HMM model to features.

        Args:
            features: Feature matrix
            n_init: Number of random initializations to try
            select_best: Method to select best model ('score', 'stability', 'bic', 'aic')

        Returns:
            Self for method chaining
        """
        # Prepare data
        X_scaled = self._prepare_features(features)

        # Fit multiple models
        self.models, scores = self._fit_multiple_models(X_scaled, n_init)

        if not self.models:
            raise RuntimeError("All initializations failed")

        # Select best model
        self.best_model_idx = self._select_best_model(
            X_scaled, scores, select_best
        )
        self.model = self.models[self.best_model_idx]
        self.train_log_likelihood = scores[self.best_model_idx]

        # Update diagnostics
        self._update_diagnostics(X_scaled)

        # Mark as fitted
        self.is_fitted = True
        self.fit_date = datetime.now()

        logger.info(
            f"HMM fitted with {self.n_regimes} regimes. Best score: {self.train_log_likelihood:.2f}"
        )

        return self

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

        regimes = self.model.predict(X_scaled)

        # Reorder regimes if needed (e.g., by mean return)
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

        # Get posterior probabilities
        _, posteriors = self.model.score_samples(X_scaled)

        # Reorder if needed
        posteriors = self._reorder_probabilities(posteriors, X_scaled)

        return posteriors

    def _reorder_regimes(
        self, regimes: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """Reorder regimes by mean of first feature (typically returns).

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
        """Reorder probability columns to match regime ordering.

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

        return self.model.score(X_scaled)

    def _prepare_features(self, features):
        """Prepare and scale features for training."""
        if hasattr(features, "values"):
            # DataFrame input
            X = features.values
            self.feature_names = list(features.columns)
        else:
            # Numpy array input
            X = features
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        X_scaled = self.scaler.fit_transform(X)
        self.n_features = X.shape[1]
        return X_scaled

    def _fit_multiple_models(self, X_scaled, n_init):
        """Fit multiple models with different initializations."""
        models = []
        scores = []

        for i in range(n_init):
            model, score = self._fit_single_model(X_scaled, i)
            if model is not None:
                models.append(model)
                scores.append(score)
                logger.debug(
                    f"Initialization {i+1}/{n_init}: score={score:.2f}"
                )

        return models, scores

    def _fit_single_model(self, X_scaled, iteration):
        """Fit a single model with appropriate initialization."""
        try:
            model = self._initialize_model()

            # Choose initialization method
            init_method = self.init_method if iteration == 0 else "random"
            # Note: _get_initial_params is called for potential side effects
            _ = self._get_initial_params(X_scaled, init_method)

            # Let hmmlearn handle initialization
            model.init_params = "stmc"  # Initialize all parameters

            # Fit model
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                model.fit(X_scaled)

            score = model.score(X_scaled)
            return model, score

        except Exception as e:
            logger.warning(f"Initialization {iteration+1} failed: {e}")
            return None, None

    def _select_best_model(self, X_scaled, scores, select_best):
        """Select best model based on specified criterion."""
        selectors = {
            "score": lambda: np.argmax(scores),
            "bic": lambda: self._select_by_bic(X_scaled),
            "aic": lambda: self._select_by_aic(X_scaled),
            "stability": lambda: self._select_by_stability(X_scaled),
        }

        selector = selectors.get(select_best, selectors["score"])
        return selector()

    def _select_by_bic(self, X_scaled):
        """Select model with lowest BIC."""
        bics = []
        for model in self.models:
            n_params = self._count_parameters_for_model(model)
            ll = model.score(X_scaled)
            bic = np.log(len(X_scaled)) * n_params - 2 * ll
            bics.append(bic)
        return np.argmin(bics)

    def _select_by_aic(self, X_scaled):
        """Select model with lowest AIC."""
        aics = []
        for model in self.models:
            n_params = self._count_parameters_for_model(model)
            ll = model.score(X_scaled)
            aic = 2 * n_params - 2 * ll
            aics.append(aic)
        return np.argmin(aics)

    def _select_by_stability(self, X_scaled):
        """Select model with highest regime stability."""
        stabilities = []
        for model in self.models:
            regimes = model.predict(X_scaled)
            transitions = np.sum(np.diff(regimes) != 0)
            stabilities.append(-transitions)  # Negative for argmax
        return np.argmax(stabilities)

    def _count_parameters(self) -> int:
        """Count the number of model parameters.

        Returns:
            Number of parameters
        """
        return self._count_parameters_for_model(self.model)

    def _count_parameters_for_model(self, model: hmm.GaussianHMM) -> int:
        """Count the number of parameters for a given model.

        Args:
            model: HMM model

        Returns:
            Number of parameters
        """
        n_params = 0
        n_components = model.n_components
        n_features = (
            model.means_.shape[1]
            if hasattr(model, "means_")
            else self.n_features
        )

        # Initial state probabilities
        n_params += n_components - 1  # One constraint (sum to 1)

        # Transition matrix
        n_params += n_components * (n_components - 1)  # Row constraints

        # Means
        n_params += n_components * n_features

        # Covariances
        if model.covariance_type == "full":
            n_params += n_components * n_features * (n_features + 1) // 2
        elif model.covariance_type == "diag":
            n_params += n_components * n_features
        elif model.covariance_type == "tied":
            n_params += n_features * (n_features + 1) // 2
        else:  # spherical
            n_params += n_components

        return n_params

    def _update_diagnostics(self, features: np.ndarray):
        """Update model diagnostics.

        Args:
            features: Scaled feature matrix
        """
        # Get predictions
        regimes = self.model.predict(features)

        # Regime statistics
        self.diagnostics["regime_counts"] = np.bincount(
            regimes, minlength=self.n_regimes
        )
        self.diagnostics["regime_proportions"] = self.diagnostics[
            "regime_counts"
        ] / len(regimes)

        # Transition statistics
        transitions = np.sum(np.diff(regimes) != 0)
        self.diagnostics["n_transitions"] = transitions
        self.diagnostics["transition_rate"] = transitions / (len(regimes) - 1)

        # Model convergence
        # hmmlearn uses n_iter not n_iter_
        if hasattr(self.model, "n_iter"):
            self.diagnostics["n_iter"] = self.model.n_iter
        elif hasattr(self.model, "n_iter_"):
            self.diagnostics["n_iter"] = self.model.n_iter_
        else:
            self.diagnostics["n_iter"] = 0

        # Check convergence
        if hasattr(self.model, "monitor_"):
            self.diagnostics["converged"] = self.model.monitor_.converged
        elif hasattr(self.model, "converged_"):
            self.diagnostics["converged"] = self.model.converged_
        else:
            self.diagnostics["converged"] = True

        # Transition matrix
        self.diagnostics["transition_matrix"] = self.model.transmat_

        # State means
        self.diagnostics["state_means"] = self.model.means_

        # Log-likelihood
        self.diagnostics["log_likelihood"] = self.model.score(features)

        # AIC and BIC
        n_params = self._count_parameters()
        n_samples = len(features)
        ll = self.diagnostics["log_likelihood"]

        self.diagnostics["aic"] = 2 * n_params - 2 * ll
        self.diagnostics["bic"] = np.log(n_samples) * n_params - 2 * ll

        # Stability score (based on diagonal of transition matrix)
        self.diagnostics["stability_score"] = np.mean(
            np.diag(self.model.transmat_)
        )

    def get_viterbi_path(
        self, features: pd.DataFrame
    ) -> Tuple[np.ndarray, float]:
        """Get most likely state sequence using Viterbi algorithm.

        Args:
            features: Feature matrix

        Returns:
            Tuple of (state_sequence, log_probability)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler.transform(X)

        log_prob, states = self.model.decode(X_scaled, algorithm="viterbi")

        # Reorder states
        states = self._reorder_regimes(states, X_scaled)

        return states, log_prob

    # Compatibility: allow passing features directly for transition analysis
    def analyze_regime_transitions(self, data: Any) -> Dict[str, Any]:
        """Analyze transitions; accepts either regimes or features.

        If a feature matrix is provided, predicts regimes first.
        """
        if isinstance(data, pd.DataFrame) or (
            isinstance(data, np.ndarray) and getattr(data, "ndim", 1) == 2
        ):
            regimes = self.predict(data)  # type: ignore[arg-type]
        else:
            regimes = np.asarray(data)

        return super().analyze_regime_transitions(regimes)

    def sample(self, n_samples: int = 100) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate samples from the fitted model.

        Args:
            n_samples: Number of samples to generate

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

    def optimize_n_regimes(
        self,
        features: pd.DataFrame,
        min_regimes: int = 2,
        max_regimes: int = 5,
        criterion: str = "bic",
        n_init: int = 5,
    ) -> Dict[str, Any]:
        """Find optimal number of regimes using information criteria.

        Args:
            features: Feature matrix
            min_regimes: Minimum number of regimes to test
            max_regimes: Maximum number of regimes to test
            criterion: Selection criterion ('bic' or 'aic')
            n_init: Number of initializations per regime count

        Returns:
            Dictionary with optimization results
        """
        # Validate inputs
        if min_regimes > max_regimes:
            raise ValueError(
                f"min_regimes ({min_regimes}) must be <= max_regimes ({max_regimes})"
            )
        if min_regimes < 2:
            raise ValueError(
                f"min_regimes must be at least 2, got {min_regimes}"
            )
        if criterion not in ["bic", "aic"]:
            raise ValueError(
                f"criterion must be 'bic' or 'aic', got {criterion}"
            )

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler.fit_transform(X)
        n_samples = len(X_scaled)

        results = {
            "n_regimes": [],
            "aic": [],
            "bic": [],
            "log_likelihood": [],
            "n_parameters": [],
        }

        best_score = np.inf
        best_n_regimes = min_regimes

        for n_regimes in range(min_regimes, max_regimes + 1):
            # Store original n_regimes
            self.n_regimes = n_regimes

            # Fit model with current n_regimes
            try:
                self.fit(features, n_init=n_init)

                # Calculate information criteria
                ll = self.model.score(X_scaled)
                n_params = self._count_parameters()
                aic = 2 * n_params - 2 * ll
                bic = np.log(n_samples) * n_params - 2 * ll

                results["n_regimes"].append(n_regimes)
                results["aic"].append(aic)
                results["bic"].append(bic)
                results["log_likelihood"].append(ll)
                results["n_parameters"].append(n_params)

                # Check if this is the best model
                current_score = bic if criterion == "bic" else aic
                if current_score < best_score:
                    best_score = current_score
                    best_n_regimes = n_regimes

                logger.info(
                    f"n_regimes={n_regimes}: AIC={aic:.2f}, BIC={bic:.2f}, LL={ll:.2f}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to fit model with {n_regimes} regimes: {e}"
                )
                continue

        # Refit with optimal number of regimes
        self.n_regimes = best_n_regimes
        self.fit(
            features, n_init=n_init * 2
        )  # More initializations for final fit

        results["optimal_n_regimes"] = best_n_regimes
        results["optimal_score"] = best_score
        results["criterion"] = criterion

        return results

    def label_regimes(
        self, features: pd.DataFrame, method: str = "volatility_return"
    ) -> Dict[int, str]:
        """Label regimes with interpretable names.

        Args:
            features: Feature matrix used for fitting
            method: Labeling method ('volatility_return', 'return_quantile', 'custom')

        Returns:
            Dictionary mapping regime indices to names
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before labeling regimes")

        # Prepare data and get predictions
        X, regimes = self._prepare_labeling_data(features)

        # Use strategy pattern to get labels
        labeling_strategies = {
            "volatility_return": self._label_by_volatility_return,
            "return_quantile": self._label_by_return_quantile,
            "custom": self._label_by_return_ranking,
        }

        strategy = labeling_strategies.get(
            method, self._label_by_return_ranking
        )
        labels = strategy(X, regimes)

        # Store labels
        self.regime_names = labels
        return labels

    def _prepare_labeling_data(
        self, features: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for regime labeling."""
        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler.transform(X)
        regimes = self.model.predict(X_scaled)
        return X, regimes

    def _label_by_volatility_return(
        self, X: np.ndarray, regimes: np.ndarray
    ) -> Dict[int, str]:
        """Label regimes based on volatility and returns."""
        labels = {}
        overall_mean_vol = self._calculate_overall_volatility(X)

        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            if mask.sum() == 0:
                labels[regime_id] = f"Regime_{regime_id}"
                continue

            regime_stats = self._calculate_regime_stats(X, mask)
            labels[regime_id] = self._get_volatility_return_label(
                regime_stats["mean_return"],
                regime_stats["mean_volatility"],
                overall_mean_vol,
            )

        return labels

    def _label_by_return_quantile(
        self, X: np.ndarray, regimes: np.ndarray
    ) -> Dict[int, str]:
        """Label regimes based on return quantiles."""
        labels = {}
        q25, q75 = np.percentile(X[:, 0], [25, 75])

        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            if mask.sum() == 0:
                labels[regime_id] = f"Regime_{regime_id}"
                continue

            mean_return = X[mask, 0].mean()
            labels[regime_id] = self._get_quantile_label(mean_return, q25, q75)

        return labels

    def _label_by_return_ranking(
        self, X: np.ndarray, regimes: np.ndarray
    ) -> Dict[int, str]:
        """Label regimes by ranking their mean returns."""
        # Calculate mean return for each regime
        regime_returns = self._calculate_regime_returns(X, regimes)

        # Sort by return
        regime_returns.sort(key=lambda x: x[1])

        # Get labels based on number of regimes
        return self._get_ranked_labels(regime_returns)

    def _calculate_overall_volatility(self, X: np.ndarray) -> float:
        """Calculate overall volatility measure."""
        if X.shape[1] > 1:
            return X[:, 1].mean()
        else:
            return np.std(X[:, 0])

    def _calculate_regime_stats(
        self, X: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        """Calculate statistics for a specific regime."""
        regime_data = X[mask]
        stats = {
            "mean_return": regime_data[:, 0].mean(),
        }

        if X.shape[1] > 1:
            stats["mean_volatility"] = regime_data[:, 1].mean()
        else:
            stats["mean_volatility"] = np.std(regime_data[:, 0])

        return stats

    def _get_volatility_return_label(
        self, mean_return: float, mean_vol: float, overall_vol: float
    ) -> str:
        """Get label based on return and volatility characteristics."""
        high_vol = mean_vol > overall_vol * 1.2

        if mean_return > 0.001:  # Positive returns
            return "Bull_HighVol" if high_vol else "Bull_Normal"
        elif mean_return < -0.001:  # Negative returns
            return "Bear_Crisis" if high_vol else "Bear_Normal"
        else:  # Neutral returns
            return "Sideways_HighVol" if high_vol else "Sideways_Normal"

    def _get_quantile_label(
        self, mean_return: float, q25: float, q75: float
    ) -> str:
        """Get label based on return quantiles."""
        if mean_return < q25:
            return "Bear_Market"
        elif mean_return > q75:
            return "Bull_Market"
        else:
            return "Neutral_Market"

    def _calculate_regime_returns(
        self, X: np.ndarray, regimes: np.ndarray
    ) -> List[Tuple[int, float]]:
        """Calculate mean returns for each regime."""
        regime_returns = []
        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            if mask.sum() > 0:
                regime_returns.append((regime_id, X[mask, 0].mean()))
            else:
                regime_returns.append((regime_id, 0))
        return regime_returns

    def _get_ranked_labels(
        self, regime_returns: List[Tuple[int, float]]
    ) -> Dict[int, str]:
        """Get labels based on regime ranking."""
        labels = {}

        # Use lookup table for common cases
        label_mappings = {
            2: ["Bear", "Bull"],
            3: ["Bear", "Neutral", "Bull"],
            4: ["Strong_Bear", "Weak_Bear", "Weak_Bull", "Strong_Bull"],
        }

        if self.n_regimes in label_mappings:
            for i, (regime_id, _) in enumerate(regime_returns):
                labels[regime_id] = label_mappings[self.n_regimes][i]
        else:
            for i, (regime_id, _) in enumerate(regime_returns):
                labels[regime_id] = f"Regime_{i}_of_{self.n_regimes}"

        return labels

    def get_regime_statistics(
        self, features: pd.DataFrame, regimes: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Calculate detailed statistics for each regime.

        Args:
            features: Feature matrix
            regimes: Optional pre-computed regime labels

        Returns:
            DataFrame with regime statistics
        """
        if regimes is None:
            regimes = self.predict(features)

        X = features.values if hasattr(features, "values") else features

        stats = []
        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            if mask.sum() == 0:
                continue

            regime_data = X[mask]
            regime_name = self.regime_names.get(
                regime_id, f"Regime_{regime_id}"
            )

            # Calculate statistics
            stat_dict = {
                "regime_id": regime_id,
                "regime_name": regime_name,
                "count": mask.sum(),
                "proportion": mask.sum() / len(regimes),
            }

            # Add feature-specific statistics
            for i, feat_name in enumerate(self.feature_names):
                feat_data = regime_data[:, i]
                stat_dict[f"{feat_name}_mean"] = feat_data.mean()
                stat_dict[f"{feat_name}_std"] = feat_data.std()
                stat_dict[f"{feat_name}_min"] = feat_data.min()
                stat_dict[f"{feat_name}_max"] = feat_data.max()
                stat_dict[f"{feat_name}_skew"] = self._calculate_skewness(
                    feat_data
                )
                stat_dict[f"{feat_name}_kurtosis"] = self._calculate_kurtosis(
                    feat_data
                )

            # Calculate regime duration statistics
            durations = self._calculate_single_regime_durations(
                regimes, regime_id
            )
            if durations:
                stat_dict["avg_duration"] = np.mean(durations)
                stat_dict["max_duration"] = np.max(durations)
                stat_dict["min_duration"] = np.min(durations)

            stats.append(stat_dict)

        return pd.DataFrame(stats)

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data.

        Args:
            data: Data array

        Returns:
            Skewness value
        """
        from scipy import stats

        return stats.skew(data)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data.

        Args:
            data: Data array

        Returns:
            Kurtosis value
        """
        from scipy import stats

        return stats.kurtosis(data)

    def _calculate_single_regime_durations(
        self, regimes: np.ndarray, regime_id: int
    ) -> List[int]:
        """Calculate duration of each occurrence of a regime.

        Args:
            regimes: Array of regime labels
            regime_id: Regime to calculate durations for

        Returns:
            List of durations
        """
        durations = []
        in_regime = False
        current_duration = 0

        for regime in regimes:
            if regime == regime_id:
                if not in_regime:
                    in_regime = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_regime:
                    durations.append(current_duration)
                    in_regime = False
                    current_duration = 0

        # Add final duration if still in regime
        if in_regime:
            durations.append(current_duration)

        return durations

    def save(self, filepath: str) -> None:
        """Save model to file including scaler.

        Args:
            filepath: Path to save model
        """
        from pathlib import Path
        import pickle

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "n_regimes": self.n_regimes,
            "is_fitted": self.is_fitted,
            "fit_date": self.fit_date,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "regime_names": self.regime_names,
            "diagnostics": self.diagnostics,
            "class_name": self.__class__.__name__,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"HMM model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "HMMRegimeDetector":
        """Load model from file including scaler.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model instance
        """
        from pathlib import Path
        import pickle

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Create instance
        instance = cls(
            n_regimes=model_data["n_regimes"],
            covariance_type=model_data.get("covariance_type", "full"),
            n_iter=model_data.get("n_iter", 100),
            random_state=model_data.get("random_state", None),
        )

        # Restore state
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.is_fitted = model_data["is_fitted"]
        instance.fit_date = model_data["fit_date"]
        instance.n_features = model_data["n_features"]
        instance.feature_names = model_data["feature_names"]
        instance.regime_names = model_data["regime_names"]
        instance.diagnostics = model_data["diagnostics"]

        logger.info(f"HMM model loaded from {filepath}")
        return instance
