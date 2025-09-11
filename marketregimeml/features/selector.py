"""Feature selection following Single Responsibility Principle."""

from typing import Dict, List, Optional, Any, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Handles feature selection and importance analysis.

    Single Responsibility: Feature selection and dimensionality reduction.
    """

    def __init__(self):
        """Initialize feature selector."""
        self.selectors = {}
        self.feature_importance_ = {}
        logger.debug("FeatureSelector initialized")

    def select_by_variance(
        self, features: pd.DataFrame, threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features by variance threshold.

        Args:
            features: Features DataFrame
            threshold: Variance threshold (features below this are removed)

        Returns:
            Tuple of (selected features, removed feature names)
        """
        if features.empty:
            return features, []

        try:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns for variance selection")
                return features, []

            selector_key = f"variance_{threshold}"
            if selector_key not in self.selectors:
                self.selectors[selector_key] = VarianceThreshold(
                    threshold=threshold
                )

            # Fit selector
            self.selectors[selector_key].fit(features[numeric_cols])
            selected_features = features[numeric_cols].loc[
                :, self.selectors[selector_key].get_support()
            ]

            # Add back non-numeric columns
            non_numeric_cols = features.select_dtypes(
                exclude=[np.number]
            ).columns
            if len(non_numeric_cols) > 0:
                selected_features = pd.concat(
                    [selected_features, features[non_numeric_cols]], axis=1
                )

            # Identify removed features
            removed_features = [
                col
                for col in numeric_cols
                if col not in selected_features.columns
            ]

            logger.info(
                f"Variance selection: kept {len(selected_features.columns)}, removed {len(removed_features)} features"
            )
            return selected_features, removed_features

        except Exception as e:
            logger.error(f"Error in variance selection: {e}")
            return features, []

    def select_k_best(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        k: Union[int, str] = 10,
        score_func: str = "f_classif",
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select k best features using statistical tests.

        Args:
            features: Features DataFrame
            target: Target variable for feature selection
            k: Number of features to select or 'all'
            score_func: Scoring function ('f_classif', 'f_regression', 'mutual_info_classif', 'mutual_info_regression')

        Returns:
            Tuple of (selected features, feature scores)
        """
        if features.empty:
            return features, {}

        try:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns for k-best selection")
                return features, {}

            # Select scoring function
            score_functions = {
                "f_classif": f_classif,
                "f_regression": f_regression,
                "mutual_info_classif": mutual_info_classif,
                "mutual_info_regression": mutual_info_regression,
            }

            if score_func not in score_functions:
                logger.warning(
                    f"Unknown score function: {score_func}, using f_classif"
                )
                score_func = "f_classif"

            # Create selector
            selector_key = f"kbest_{k}_{score_func}"
            if selector_key not in self.selectors:
                self.selectors[selector_key] = SelectKBest(
                    score_func=score_functions[score_func], k=k
                )

            # Fit and transform
            features_numeric = features[numeric_cols].fillna(
                features[numeric_cols].mean()
            )
            selected_array = self.selectors[selector_key].fit_transform(
                features_numeric, target
            )

            # Get selected feature names
            selected_mask = self.selectors[selector_key].get_support()
            selected_cols = numeric_cols[selected_mask]

            # Create selected features DataFrame
            selected_features = pd.DataFrame(
                selected_array, index=features.index, columns=selected_cols
            )

            # Add back non-numeric columns
            non_numeric_cols = features.select_dtypes(
                exclude=[np.number]
            ).columns
            if len(non_numeric_cols) > 0:
                selected_features = pd.concat(
                    [selected_features, features[non_numeric_cols]], axis=1
                )

            # Get feature scores
            scores = self.selectors[selector_key].scores_
            feature_scores = {
                col: score for col, score in zip(numeric_cols, scores)
            }

            logger.info(
                f"K-best selection: selected {len(selected_cols)} features using {score_func}"
            )
            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"Error in k-best selection: {e}")
            return features, {}

    def _prepare_data_for_importance(
        self, features: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
        """Prepare data for importance calculation.

        Args:
            features: Features DataFrame
            target: Target variable

        Returns:
            Tuple of (prepared features, prepared target, numeric columns)
        """
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame(), pd.Series(), pd.Index([])

        # Fill NaN values with mean
        X = features[numeric_cols].fillna(features[numeric_cols].mean())
        y = target.dropna()

        # Align X and y indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        return X, y, numeric_cols

    def _get_importance_model(self, target: pd.Series, method: str):
        """Get the appropriate model for importance calculation.

        Args:
            target: Target variable
            method: Method name ('random_forest', etc.)

        Returns:
            Scikit-learn model instance
        """
        is_classification = (
            target.dtype == "object"
            or len(target.unique()) < len(target) * 0.05
        )

        if method != "random_forest":
            logger.warning(f"Unknown method: {method}, using random_forest")
            method = "random_forest"

        if is_classification:
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)

    def _select_features_by_importance(
        self,
        feature_importances: Dict[str, float],
        top_k: Optional[int],
        threshold: Optional[float],
    ) -> List[str]:
        """Select features based on importance scores.

        Args:
            feature_importances: Dictionary of feature names to importance scores
            top_k: Number of top features to select
            threshold: Importance threshold

        Returns:
            List of selected feature names
        """
        if top_k is not None:
            sorted_features = sorted(
                feature_importances.items(), key=lambda x: x[1], reverse=True
            )
            return [name for name, _ in sorted_features[:top_k]]

        if threshold is not None:
            return [
                name
                for name, imp in feature_importances.items()
                if imp >= threshold
            ]

        # Default: use mean importance as threshold
        mean_importance = np.mean(list(feature_importances.values()))
        logger.info(f"Using mean importance threshold: {mean_importance:.4f}")
        return [
            name
            for name, imp in feature_importances.items()
            if imp >= mean_importance
        ]

    def select_by_importance(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        method: str = "random_forest",
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select features by importance from tree-based models.

        Args:
            features: Features DataFrame
            target: Target variable
            method: Method for importance calculation ('random_forest', 'extra_trees')
            top_k: Number of top features to select (mutually exclusive with threshold)
            threshold: Importance threshold for selection (mutually exclusive with top_k)

        Returns:
            Tuple of (selected features, feature importances)
        """
        if features.empty:
            return features, {}

        try:
            # Prepare data
            X, y, numeric_cols = self._prepare_data_for_importance(
                features, target
            )

            if len(X) == 0:
                logger.warning("No valid data for importance calculation")
                return features, {}

            # Get and fit model
            model = self._get_importance_model(y, method)
            model.fit(X, y)

            # Get feature importances
            importances = model.feature_importances_
            feature_importances = {
                col: imp for col, imp in zip(numeric_cols, importances)
            }

            # Select features based on importance
            selected_feature_names = self._select_features_by_importance(
                feature_importances, top_k, threshold
            )

            # Include non-numeric columns in result
            non_numeric_cols = list(
                features.select_dtypes(exclude=[np.number]).columns
            )
            all_selected_cols = selected_feature_names + non_numeric_cols
            selected_features = features[all_selected_cols]

            # Store importance for later use
            self.feature_importance_[method] = feature_importances

            logger.info(
                f"Importance selection: selected {len(selected_feature_names)} "
                f"features using {method}"
            )
            return selected_features, feature_importances

        except Exception as e:
            logger.error(f"Error in importance selection: {e}")
            return features, {}

    def recursive_feature_elimination(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        n_features: int = 10,
        step: int = 1,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Perform recursive feature elimination.

        Args:
            features: Features DataFrame
            target: Target variable
            n_features: Number of features to select
            step: Number of features to remove at each step

        Returns:
            Tuple of (selected features, feature rankings)
        """
        if features.empty:
            return features, {}

        try:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns for RFE selection")
                return features, {}

            # Prepare data
            X = features[numeric_cols].fillna(features[numeric_cols].mean())
            y = target.dropna()

            # Align X and y
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]

            if len(X) == 0:
                logger.warning("No common indices between features and target")
                return features, {}

            # Select base estimator
            if (
                y.dtype == "object" or len(y.unique()) < len(y) * 0.05
            ):  # Classification
                estimator = RandomForestClassifier(
                    n_estimators=50, random_state=42
                )
            else:  # Regression
                estimator = RandomForestRegressor(
                    n_estimators=50, random_state=42
                )

            # Create RFE selector
            selector_key = f"rfe_{n_features}_{step}"
            if selector_key not in self.selectors:
                self.selectors[selector_key] = RFE(
                    estimator=estimator,
                    n_features_to_select=n_features,
                    step=step,
                )

            # Fit and transform
            self.selectors[selector_key].fit(X, y)
            selected_mask = self.selectors[selector_key].get_support()
            selected_cols = numeric_cols[selected_mask]

            # Create selected features DataFrame
            selected_features = features[selected_cols]

            # Add back non-numeric columns
            non_numeric_cols = features.select_dtypes(
                exclude=[np.number]
            ).columns
            if len(non_numeric_cols) > 0:
                selected_features = pd.concat(
                    [selected_features, features[non_numeric_cols]], axis=1
                )

            # Get feature rankings
            rankings = self.selectors[selector_key].ranking_
            feature_rankings = {
                col: rank for col, rank in zip(numeric_cols, rankings)
            }

            logger.info(
                f"RFE selection: selected {len(selected_cols)} features"
            )
            return selected_features, feature_rankings

        except Exception as e:
            logger.error(f"Error in RFE selection: {e}")
            return features, {}

    def get_feature_importance(
        self, method: Optional[str] = None
    ) -> Dict[str, float]:
        """Get stored feature importance scores.

        Args:
            method: Specific method to retrieve, None for all

        Returns:
            Dictionary of feature importance scores
        """
        if method is None:
            return self.feature_importance_

        return self.feature_importance_.get(method, {})

    def _apply_selection_method(
        self,
        method_name: str,
        features: pd.DataFrame,
        target: pd.Series,
        method_params: Dict[str, Any],
    ) -> Tuple[set, Dict[str, Any]]:
        """Apply a single selection method.

        Args:
            method_name: Name of the selection method
            features: Features DataFrame
            target: Target variable
            method_params: Method parameters

        Returns:
            Tuple of (selected feature names, method results)
        """
        method_handlers = {
            "variance": lambda: self.select_by_variance(
                features, **method_params
            ),
            "k_best": lambda: self.select_k_best(
                features, target, **method_params
            ),
            "importance": lambda: self.select_by_importance(
                features, target, **method_params
            ),
        }

        if method_name not in method_handlers:
            logger.warning(f"Unknown selection method: {method_name}")
            return set(), {}

        # Apply the method
        if method_name == "variance":
            selected, removed = method_handlers[method_name]()
            return set(selected.columns), {"removed": removed}
        elif method_name == "k_best":
            selected, scores = method_handlers[method_name]()
            return set(selected.columns), {"scores": scores}
        elif method_name == "importance":
            selected, importances = method_handlers[method_name]()
            return set(selected.columns), {"importances": importances}

        return set(), {}

    def _combine_feature_sets(
        self, all_selections: Dict[str, set], voting: str, num_methods: int
    ) -> set:
        """Combine feature sets using voting strategy.

        Args:
            all_selections: Dictionary of method name to selected features
            voting: Voting strategy ('union', 'intersection', 'majority')
            num_methods: Total number of methods

        Returns:
            Final set of selected features
        """
        if voting == "union":
            final_features = set()
            for selected_set in all_selections.values():
                final_features.update(selected_set)
            return final_features

        elif voting == "intersection":
            if not all_selections:
                return set()
            final_features = next(iter(all_selections.values())).copy()
            for selected_set in all_selections.values():
                final_features.intersection_update(selected_set)
            return final_features

        elif voting == "majority":
            feature_votes = {}
            for selected_set in all_selections.values():
                for feature in selected_set:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1

            majority_threshold = num_methods / 2
            return {
                feature
                for feature, votes in feature_votes.items()
                if votes > majority_threshold
            }
        else:
            logger.warning(f"Unknown voting method: {voting}, using union")
            return self._combine_feature_sets(
                all_selections, "union", num_methods
            )

    def combine_selection_methods(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        methods: List[Dict[str, Any]],
        voting: str = "union",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Combine multiple feature selection methods.

        Args:
            features: Features DataFrame
            target: Target variable
            methods: List of method configurations
            voting: How to combine selections ('union', 'intersection', 'majority')

        Returns:
            Tuple of (selected features, selection results)
        """
        if features.empty:
            return features, {}

        try:
            all_selections = {}
            all_results = {}

            # Apply each method
            for method_config in methods:
                method_name = method_config["method"]
                method_params = method_config.get("params", {})

                selected_features, method_result = (
                    self._apply_selection_method(
                        method_name, features, target, method_params
                    )
                )

                if selected_features:
                    all_selections[method_name] = selected_features
                    all_results[method_name] = method_result

            # Check if any methods were applied
            if not all_selections:
                logger.warning("No valid selection methods applied")
                return features, {}

            # Combine selections using voting strategy
            final_features = self._combine_feature_sets(
                all_selections, voting, len(methods)
            )

            # Create final DataFrame
            selected_features = features[list(final_features)]

            # Compile results
            combined_results = {
                "methods_applied": list(all_selections.keys()),
                "voting_method": voting,
                "individual_results": all_results,
                "final_feature_count": len(final_features),
                "original_feature_count": len(features.columns),
            }

            logger.info(
                f"Combined selection: {len(final_features)} features from {len(features.columns)} original"
            )
            return selected_features, combined_results

        except Exception as e:
            logger.error(f"Error in combined selection: {e}")
            return features, {}

    def get_selector_info(self) -> Dict[str, Any]:
        """Get information about fitted selectors.

        Returns:
            Dictionary with selector information
        """
        info = {
            "fitted_selectors": list(self.selectors.keys()),
            "stored_importances": list(self.feature_importance_.keys()),
        }

        return info

    def reset_selectors(self) -> None:
        """Reset all fitted selectors."""
        self.selectors.clear()
        self.feature_importance_.clear()
        logger.info("All selectors reset")
