"""Machine learning based regime detection models.

This module implements regime detection using traditional machine learning
algorithms including Random Forest, XGBoost, and SVM.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
    TimeSeriesSplit,
)

from marketregimeml.models.base import BaseRegimeDetector
from marketregimeml.utils.logging import get_logger


logger = get_logger(__name__)

# Try to import XGBoost (optional dependency)
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning(
        "XGBoost not installed. XGBoostRegimeClassifier will not be available."
    )


class RandomForestRegimeClassifier(BaseRegimeDetector):
    """Random Forest based regime classifier.

    Can work in both supervised (with labels) and unsupervised
    (clustering first, then supervised) modes.

    Parameters
    ----------
    n_regimes : int, default=5
        Number of regimes to detect (default optimized based on benchmarks)
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int or None, default=None
        Maximum depth of trees
    min_samples_split : int, default=2
        Minimum samples required to split node
    min_samples_leaf : int, default=1
        Minimum samples required at leaf node
    max_features : str or int, default='sqrt'
        Number of features to consider for best split
    oob_score : bool, default=False
        Whether to use out-of-bag samples for score
    random_state : int or None, default=None
        Random seed for reproducibility

    Attributes
    ----------
    model_ : RandomForestClassifier
        Fitted Random Forest model
    feature_importances_ : np.ndarray
        Feature importance scores
    cluster_labels_ : np.ndarray or None
        Cluster labels if fitted in unsupervised mode
    oob_score_ : float or None
        Out-of-bag score if oob_score=True
    """

    def __init__(
        self,
        n_regimes: int = 5,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int] = "sqrt",
        oob_score: bool = False,
        random_state: Optional[int] = None,
        class_weight: Optional[Union[str, Dict]] = None,
        **kwargs,
    ):
        """Initialize Random Forest regime classifier."""
        super().__init__(n_regimes=n_regimes, random_state=random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.class_weight = class_weight

        self.model_ = None
        self.feature_importances_ = None
        self.cluster_labels_ = None
        self.oob_score_ = None
        self.scaler_ = None

    def fit(
        self, features: pd.DataFrame, y: Optional[np.ndarray] = None, **kwargs
    ) -> "RandomForestRegimeClassifier":
        """Fit Random Forest classifier.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Regime labels. If None, will use clustering first
        **kwargs : dict
            Additional parameters for fit

        Returns
        -------
        self : RandomForestRegimeClassifier
            Fitted classifier
        """
        # Store feature names
        if hasattr(features, "columns"):
            # DataFrame input
            self.feature_names = list(features.columns)
            X = features.values if hasattr(features, "values") else features
        else:
            # Numpy array input
            X = features
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.n_features = len(self.feature_names)

        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # If no labels provided, use clustering
        if y is None:
            logger.info("No labels provided, using KMeans clustering")
            kmeans = KMeans(
                n_clusters=self.n_regimes, random_state=self.random_state
            )
            self.cluster_labels_ = kmeans.fit_predict(X_scaled)
            y = self.cluster_labels_

        # Create Random Forest model
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            oob_score=self.oob_score,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight=self.class_weight,
        )

        # Fit the model
        self.model_.fit(X_scaled, y)

        # Store feature importances
        self.feature_importances_ = self.model_.feature_importances_

        # Store OOB score if available
        if self.oob_score:
            self.oob_score_ = self.model_.oob_score_
            logger.info(f"OOB Score: {self.oob_score_:.4f}")

        self.is_fitted = True
        logger.info(f"Random Forest fitted with {self.n_estimators} trees")

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime labels.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        predictions : np.ndarray
            Predicted regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        probabilities : np.ndarray
            Regime probabilities for each sample
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_proba(X_scaled)

    def get_feature_importance(
        self, normalize: bool = True
    ) -> Dict[str, float]:
        """Get feature importance scores.

        Parameters
        ----------
        normalize : bool, default=True
            Whether to normalize importances to sum to 1

        Returns
        -------
        importances : dict
            Feature names mapped to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        importances = self.feature_importances_.copy()

        if normalize:
            importances = importances / importances.sum()

        return dict(zip(self.feature_names, importances))

    def cross_validate(
        self,
        features: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> Dict[str, np.ndarray]:
        """Perform cross-validation.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Regime labels
        cv : int, default=5
            Number of folds
        scoring : str, default='accuracy'
            Scoring metric

        Returns
        -------
        scores : dict
            Cross-validation scores
        """
        if y is None:
            # Use clustering for labels
            kmeans = KMeans(
                n_clusters=self.n_regimes, random_state=self.random_state
            )
            y = kmeans.fit_predict(features)

        # Initialize scaler if not already done
        if not hasattr(self, "scaler_") or self.scaler_ is None:
            self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(features)

        # Use time series split for financial data
        tscv = TimeSeriesSplit(n_splits=cv)

        # Create model for CV
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )

        scores = cross_validate(
            model,
            X_scaled,
            y,
            cv=tscv,
            scoring=scoring,
            return_train_score=True,
        )

        return scores

    def tune_hyperparameters(
        self,
        features: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """Tune hyperparameters using grid search.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Regime labels
        param_grid : dict, optional
            Parameter grid for search
        cv : int, default=3
            Number of CV folds
        scoring : str, default='accuracy'
            Scoring metric

        Returns
        -------
        best_params : dict
            Best parameters found
        """
        if y is None:
            kmeans = KMeans(
                n_clusters=self.n_regimes, random_state=self.random_state
            )
            y = kmeans.fit_predict(features)

        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

        # Initialize scaler if not already done
        if not hasattr(self, "scaler_") or self.scaler_ is None:
            self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(features)

        base_model = RandomForestClassifier(
            random_state=self.random_state, n_jobs=-1
        )

        tscv = TimeSeriesSplit(n_splits=cv)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=tscv, scoring=scoring, n_jobs=-1
        )

        grid_search.fit(X_scaled, y)

        # Update model with best parameters
        self.model_ = grid_search.best_estimator_
        self.n_estimators = grid_search.best_params_.get(
            "n_estimators", self.n_estimators
        )
        self.max_depth = grid_search.best_params_.get(
            "max_depth", self.max_depth
        )
        self.min_samples_split = grid_search.best_params_.get(
            "min_samples_split", self.min_samples_split
        )
        self.min_samples_leaf = grid_search.best_params_.get(
            "min_samples_leaf", self.min_samples_leaf
        )

        self.feature_importances_ = self.model_.feature_importances_
        self.is_fitted = True

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")

        return grid_search.best_params_

    def _compute_log_likelihood(self, features: pd.DataFrame) -> float:
        """Compute pseudo log-likelihood for Random Forest.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        log_likelihood : float
            Pseudo log-likelihood based on prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Use average log probability as pseudo log-likelihood
        probas = self.predict_proba(features)
        predictions = self.predict(features)

        log_likelihood = 0
        for i, pred in enumerate(predictions):
            log_likelihood += np.log(probas[i, pred] + 1e-10)

        return log_likelihood

    def _count_parameters(self) -> int:
        """Count approximate number of parameters.

        Returns
        -------
        n_params : int
            Approximate parameter count
        """
        if not self.is_fitted:
            return 0

        # Approximate: number of trees * average tree size
        # This is a rough approximation
        n_params = self.n_estimators * 100  # Rough estimate

        return n_params

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
    def load(
        cls, filepath: Union[str, Path]
    ) -> "RandomForestRegimeClassifier":
        """Load model from file.

        Parameters
        ----------
        filepath : str or Path
            Path to load model from

        Returns
        -------
        model : RandomForestRegimeClassifier
            Loaded model
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded from {filepath}")
        return model


class XGBoostRegimeClassifier(BaseRegimeDetector):
    """XGBoost based regime classifier.

    Parameters
    ----------
    n_regimes : int, default=5
        Number of regimes
    n_estimators : int, default=100
        Number of boosting rounds
    learning_rate : float, default=0.1
        Learning rate
    max_depth : int, default=6
        Maximum tree depth
    subsample : float, default=1.0
        Subsample ratio of training data
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing trees
    tree_method : str, default='auto'
        Tree construction algorithm ('auto', 'exact', 'approx', 'hist', 'gpu_hist')
    gpu_id : int, default=-1
        GPU id to use (-1 for CPU)
    random_state : int or None, default=None
        Random seed
    """

    def __init__(
        self,
        n_regimes: int = 5,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        tree_method: str = "auto",
        gpu_id: int = -1,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Initialize XGBoost classifier."""
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        super().__init__(n_regimes=n_regimes, random_state=random_state)

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.gpu_id = gpu_id

        self.model_ = None
        self.scaler_ = None
        self.feature_importances_ = None

    def fit(
        self,
        features: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ) -> "XGBoostRegimeClassifier":
        """Fit XGBoost classifier.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Regime labels
        eval_set : list of tuples, optional
            Validation sets for early stopping
        early_stopping_rounds : int, optional
            Early stopping rounds

        Returns
        -------
        self : XGBoostRegimeClassifier
            Fitted classifier
        """
        # Store feature names
        if hasattr(features, "columns"):
            # DataFrame input
            self.feature_names = list(features.columns)
            X = features.values if hasattr(features, "values") else features
        else:
            # Numpy array input
            X = features
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.n_features = len(self.feature_names)

        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        if y is None:
            kmeans = KMeans(
                n_clusters=self.n_regimes, random_state=self.random_state
            )
            y = kmeans.fit_predict(X_scaled)

        # Prepare parameters
        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "tree_method": self.tree_method,
            "random_state": self.random_state,
            "objective": "multi:softprob",
            "num_class": self.n_regimes,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
        }

        if self.gpu_id >= 0:
            params["gpu_id"] = self.gpu_id

        # Add early_stopping_rounds to model params if provided
        if early_stopping_rounds is not None:
            params["early_stopping_rounds"] = early_stopping_rounds
            params["callbacks"] = [
                xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds, save_best=True
                )
            ]

        # Create and fit model
        self.model_ = xgb.XGBClassifier(**params)

        fit_params = {}
        if eval_set is not None:
            eval_set_scaled = [
                (self.scaler_.transform(X), y) for X, y in eval_set
            ]
            fit_params["eval_set"] = eval_set_scaled
            fit_params["verbose"] = False

        self.model_.fit(X_scaled, y, **fit_params)

        # Store feature importances
        self.feature_importances_ = self.model_.feature_importances_

        self.is_fitted = True
        logger.info(f"XGBoost fitted with {self.n_estimators} rounds")

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime labels.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        predictions : np.ndarray
            Predicted regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        probabilities : np.ndarray
            Regime probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_proba(X_scaled)

    def get_feature_importance(
        self, importance_type: str = "weight"
    ) -> Dict[str, float]:
        """Get feature importance scores.

        Parameters
        ----------
        importance_type : str, default='weight'
            Type of importance ('weight', 'gain', 'cover')

        Returns
        -------
        importances : dict
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Get importance based on type
        if importance_type == "weight":
            importances = self.model_.feature_importances_
        else:
            # Get from booster
            booster = self.model_.get_booster()
            importance_dict = booster.get_score(
                importance_type=importance_type
            )

            # Map to feature names
            importances = np.zeros(self.n_features)
            for i, fname in enumerate(self.feature_names):
                key = f"f{i}"
                if key in importance_dict:
                    importances[i] = importance_dict[key]

        # Normalize
        if importances.sum() > 0:
            importances = importances / importances.sum()

        return dict(zip(self.feature_names, importances))

    def tune_hyperparameters(
        self,
        features: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
    ) -> Dict[str, Any]:
        """Tune hyperparameters.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Regime labels
        param_grid : dict, optional
            Parameter grid
        cv : int, default=3
            Number of CV folds

        Returns
        -------
        best_params : dict
            Best parameters
        """
        if y is None:
            kmeans = KMeans(
                n_clusters=self.n_regimes, random_state=self.random_state
            )
            y = kmeans.fit_predict(features)

        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 6, 9],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }

        X_scaled = self.scaler_.fit_transform(features)

        base_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=self.n_regimes,
            use_label_encoder=False,
            random_state=self.random_state,
        )

        tscv = TimeSeriesSplit(n_splits=cv)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1
        )

        grid_search.fit(X_scaled, y)

        # Update model
        self.model_ = grid_search.best_estimator_
        self.is_fitted = True

        return grid_search.best_params_

    def _compute_log_likelihood(self, features: pd.DataFrame) -> float:
        """Compute pseudo log-likelihood."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        probas = self.predict_proba(features)
        predictions = self.predict(features)

        log_likelihood = 0
        for i, pred in enumerate(predictions):
            log_likelihood += np.log(probas[i, pred] + 1e-10)

        return log_likelihood

    def _count_parameters(self) -> int:
        """Count approximate number of parameters."""
        if not self.is_fitted:
            return 0

        # Approximate based on number of trees and nodes
        return self.n_estimators * 50  # Rough estimate


class SVMRegimeClassifier(BaseRegimeDetector):
    """Support Vector Machine based regime classifier.

    Parameters
    ----------
    n_regimes : int, default=5
        Number of regimes
    kernel : str, default='rbf'
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
    C : float, default=1.0
        Regularization parameter
    gamma : str or float, default='scale'
        Kernel coefficient
    degree : int, default=3
        Degree for polynomial kernel
    probability : bool, default=False
        Enable probability estimates
    class_weight : str or dict, default=None
        Class weight ('balanced' or dict)
    random_state : int or None, default=None
        Random seed
    """

    def __init__(
        self,
        n_regimes: int = 5,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        probability: bool = False,
        class_weight: Optional[Union[str, Dict]] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """Initialize SVM classifier."""
        super().__init__(n_regimes=n_regimes, random_state=random_state)

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.probability = probability
        self.class_weight = class_weight

        self.model_ = None
        self.scaler_ = None

    def fit(
        self, features: pd.DataFrame, y: Optional[np.ndarray] = None, **kwargs
    ) -> "SVMRegimeClassifier":
        """Fit SVM classifier.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Regime labels

        Returns
        -------
        self : SVMRegimeClassifier
            Fitted classifier
        """
        self.feature_names = list(features.columns)
        self.n_features = len(self.feature_names)

        # Standardize features (important for SVM)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(features)

        if y is None:
            kmeans = KMeans(
                n_clusters=self.n_regimes, random_state=self.random_state
            )
            y = kmeans.fit_predict(X_scaled)

        # Create SVM model
        self.model_ = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state,
            decision_function_shape="ovr",
        )

        # Fit the model
        self.model_.fit(X_scaled, y)

        self.is_fitted = True
        logger.info(f"SVM fitted with {self.kernel} kernel")

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime labels.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        predictions : np.ndarray
            Predicted regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        probabilities : np.ndarray
            Regime probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if not self.probability:
            raise ValueError("probability=True required for predict_proba")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_proba(X_scaled)

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        """Get decision function values.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix

        Returns
        -------
        decision : np.ndarray
            Decision function values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = features.values if hasattr(features, "values") else features
        X_scaled = self.scaler_.transform(X)
        return self.model_.decision_function(X_scaled)

    def get_support_vectors(self) -> Dict[str, Any]:
        """Get support vector information.

        Returns
        -------
        info : dict
            Support vector information
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        return {
            "n_support": self.model_.n_support_,
            "support_indices": self.model_.support_,
            "support_vectors": self.model_.support_vectors_,
        }

    def tune_hyperparameters(
        self,
        features: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
    ) -> Dict[str, Any]:
        """Tune hyperparameters.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Regime labels
        param_grid : dict, optional
            Parameter grid
        cv : int, default=3
            Number of CV folds

        Returns
        -------
        best_params : dict
            Best parameters
        """
        if y is None:
            kmeans = KMeans(
                n_clusters=self.n_regimes, random_state=self.random_state
            )
            y = kmeans.fit_predict(features)

        if param_grid is None:
            param_grid = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
            if self.kernel == "poly":
                param_grid["degree"] = [2, 3, 4]

        # Initialize scaler if not already done
        if not hasattr(self, "scaler_") or self.scaler_ is None:
            self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(features)

        base_model = SVC(
            kernel=self.kernel,
            probability=self.probability,
            random_state=self.random_state,
        )

        tscv = TimeSeriesSplit(n_splits=cv)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1
        )

        grid_search.fit(X_scaled, y)

        # Update model
        self.model_ = grid_search.best_estimator_
        self.C = grid_search.best_params_.get("C", self.C)
        self.gamma = grid_search.best_params_.get("gamma", self.gamma)

        self.is_fitted = True

        return grid_search.best_params_

    def _compute_log_likelihood(self, features: pd.DataFrame) -> float:
        """Compute pseudo log-likelihood."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Use decision function as proxy
        decision = self.decision_function(features)
        predictions = self.predict(features)

        # Simple pseudo log-likelihood
        log_likelihood = 0
        for i, pred in enumerate(predictions):
            # Use decision function value as confidence
            log_likelihood += decision[i, pred]

        return log_likelihood

    def _count_parameters(self) -> int:
        """Count number of support vectors as proxy for parameters."""
        if not self.is_fitted:
            return 0

        # Number of support vectors * features
        return len(self.model_.support_) * self.n_features
