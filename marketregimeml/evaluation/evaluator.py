"""Model evaluation framework for regime detection."""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from marketregimeml.models.base import BaseRegimeDetector
from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluation framework for regime detection models."""

    def __init__(
        self,
        model: BaseRegimeDetector,
        regime_metrics: Optional[RegimeMetrics] = None,
    ):
        """Initialize model evaluator.

        Args:
            model: Regime detection model to evaluate
            regime_metrics: Regime-specific metrics calculator
        """
        self.model = model
        self.regime_metrics = regime_metrics or RegimeMetrics()

        # Store evaluation results
        self.evaluation_results = {}
        self.comparison_results = {}

    def evaluate_model(
        self, features: pd.DataFrame, true_regimes: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.

        Args:
            features: Feature matrix
            true_regimes: True regime labels for supervised evaluation

        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info(f"Evaluating {self.model.__class__.__name__}")

        if not self.model.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        results = {
            "model_info": {
                "model_type": self.model.__class__.__name__,
                "n_regimes": self.model.n_regimes,
                "n_features": self.model.n_features,
                "feature_names": self.model.feature_names,
                "fit_date": self.model.fit_date,
            }
        }

        # Get predictions
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        # Model diagnostics
        results["model_diagnostics"] = self.model.get_diagnostics()

        # Clustering quality metrics
        results["clustering_metrics"] = self._evaluate_clustering(
            features.values, predictions, probabilities
        )

        # Supervised evaluation (if true labels available)
        if true_regimes is not None:
            results["supervised_metrics"] = self._evaluate_supervised(
                true_regimes, predictions
            )

        # Regime characteristics
        results["regime_analysis"] = self._analyze_regimes(
            features, predictions, probabilities
        )

        # Temporal consistency
        results["temporal_metrics"] = (
            self.regime_metrics.temporal_consistency_metrics(predictions)
        )

        # Regime Quality Index
        results["regime_quality_index"] = (
            self.regime_metrics.regime_quality_index(
                features.values, predictions, probabilities
            )
        )

        self.evaluation_results = results
        return results

    def _evaluate_clustering(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate clustering quality."""
        return {
            "silhouette_score": self.regime_metrics.silhouette_coefficient(
                features, predictions
            ),
            "davies_bouldin_index": self.regime_metrics.davies_bouldin_index(
                features, predictions
            ),
            "calinski_harabasz_index": self.regime_metrics.calinski_harabasz_index(
                features, predictions
            ),
            **self.regime_metrics.confidence_metrics(probabilities),
        }

    def _evaluate_supervised(
        self, true_labels: np.ndarray, pred_labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate against true labels."""
        return {
            "adjusted_rand_index": self.regime_metrics.adjusted_rand_index(
                true_labels, pred_labels
            ),
            "normalized_mutual_info": self.regime_metrics.normalized_mutual_info(
                true_labels, pred_labels
            ),
            "adjusted_mutual_info": self.regime_metrics.adjusted_mutual_info(
                true_labels, pred_labels
            ),
        }

    def _analyze_regimes(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze regime characteristics."""
        regime_stats = {}

        # Get regime stability metrics
        regime_stats["stability"] = self.regime_metrics.regime_stability(
            predictions
        )

        # Get regime distribution
        regime_stats["distribution"] = self.regime_metrics.regime_distribution(
            predictions
        )

        # Feature statistics per regime
        feature_stats = {}
        for regime_id in np.unique(predictions):
            regime_mask = predictions == regime_id
            regime_features = features[regime_mask]

            feature_stats[f"regime_{regime_id}"] = {
                "n_samples": regime_mask.sum(),
                "proportion": regime_mask.mean(),
                "feature_means": regime_features.mean().to_dict(),
                "feature_stds": regime_features.std().to_dict(),
            }

        regime_stats["feature_statistics"] = feature_stats

        return regime_stats

    def cross_validate(
        self,
        features: pd.DataFrame,
        n_splits: int = 5,
        test_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Time series cross-validation.

        Args:
            features: Feature matrix
            n_splits: Number of CV splits
            test_size: Size of test set (if None, uses default)

        Returns:
            Cross-validation results
        """
        logger.info(f"Running {n_splits}-fold time series cross-validation")

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cv_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
            logger.debug(f"Processing fold {fold + 1}/{n_splits}")

            # Split data
            train_features = features.iloc[train_idx]
            test_features = features.iloc[test_idx]

            # Fit on train
            self.model.fit(train_features)

            # Evaluate on test
            test_predictions = self.model.predict(test_features)
            test_probabilities = self.model.predict_proba(test_features)

            # Calculate metrics
            fold_results = {
                "fold": fold,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "clustering_metrics": self._evaluate_clustering(
                    test_features.values, test_predictions, test_probabilities
                ),
                "regime_stability": self.regime_metrics.regime_stability(
                    test_predictions
                ),
                "regime_quality_index": self.regime_metrics.regime_quality_index(
                    test_features.values, test_predictions, test_probabilities
                ),
            }

            cv_results.append(fold_results)

        # Aggregate results
        aggregated = self._aggregate_cv_results(cv_results)

        return {"fold_results": cv_results, "aggregated_results": aggregated}

    def _aggregate_cv_results(self, cv_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        metrics_to_aggregate = [
            "clustering_metrics",
            "regime_stability",
            "regime_quality_index",
        ]

        aggregated = {}

        for metric_group in metrics_to_aggregate:
            if metric_group == "regime_quality_index":
                # Single value metric
                values = [fold[metric_group] for fold in cv_results]
                aggregated[metric_group] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
            else:
                # Dictionary of metrics
                group_metrics = {}
                first_fold = cv_results[0][metric_group]

                for key in first_fold.keys():
                    values = [fold[metric_group][key] for fold in cv_results]
                    group_metrics[key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                    }

                aggregated[metric_group] = group_metrics

        return aggregated

    def compare_models(
        self,
        models: List[BaseRegimeDetector],
        features: pd.DataFrame,
        true_regimes: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Compare multiple regime detection models.

        Args:
            models: List of models to compare
            features: Feature matrix
            true_regimes: True regime labels (optional)

        Returns:
            DataFrame with model comparison results
        """
        comparison_results = []

        for model in models:
            evaluator = ModelEvaluator(model, self.regime_metrics)
            results = evaluator.evaluate_model(
                features=features, true_regimes=true_regimes
            )

            # Flatten results for comparison
            flat_results = {
                "model_type": model.__class__.__name__,
                "n_regimes": model.n_regimes,
                "silhouette_score": results["clustering_metrics"][
                    "silhouette_score"
                ],
                "davies_bouldin_index": results["clustering_metrics"][
                    "davies_bouldin_index"
                ],
                "calinski_harabasz_index": results["clustering_metrics"][
                    "calinski_harabasz_index"
                ],
                "avg_confidence": results["clustering_metrics"][
                    "avg_confidence"
                ],
                "transition_rate": results["regime_analysis"]["stability"][
                    "transition_rate"
                ],
                "avg_duration": results["regime_analysis"]["stability"][
                    "avg_duration"
                ],
                "regime_quality_index": results["regime_quality_index"],
            }

            # Add supervised metrics if available
            if "supervised_metrics" in results:
                for key, value in results["supervised_metrics"].items():
                    flat_results[f"supervised_{key}"] = value

            comparison_results.append(flat_results)

        return pd.DataFrame(comparison_results)

    def generate_report(
        self, features: pd.DataFrame, output_file: Optional[str] = None
    ) -> str:
        """Generate comprehensive evaluation report.

        Args:
            features: Feature matrix
            output_file: Optional file path to save report

        Returns:
            Text report
        """
        if not self.evaluation_results:
            self.evaluate_model(features)

        results = self.evaluation_results

        report_lines = [
            "=" * 80,
            "REGIME DETECTION MODEL EVALUATION REPORT",
            "=" * 80,
            "",
            f"Model: {results['model_info']['model_type']}",
            f"Number of Regimes: {results['model_info']['n_regimes']}",
            f"Number of Features: {results['model_info']['n_features']}",
            f"Fit Date: {results['model_info']['fit_date']}",
            "",
            "-" * 80,
            "CLUSTERING QUALITY METRICS",
            "-" * 80,
            f"Silhouette Score: {results['clustering_metrics']['silhouette_score']:.4f}",
            f"Davies-Bouldin Index: {results['clustering_metrics']['davies_bouldin_index']:.4f}",
            f"Calinski-Harabasz Index: {results['clustering_metrics']['calinski_harabasz_index']:.2f}",
            "",
            "-" * 80,
            "PREDICTION CONFIDENCE",
            "-" * 80,
            f"Average Confidence: {results['clustering_metrics']['avg_confidence']:.4f}",
            f"Confidence Std Dev: {results['clustering_metrics']['confidence_std']:.4f}",
            f"Average Entropy: {results['clustering_metrics']['avg_entropy']:.4f}",
            f"Average Margin: {results['clustering_metrics']['avg_margin']:.4f}",
            "",
            "-" * 80,
            "REGIME STABILITY",
            "-" * 80,
            f"Transition Rate: {results['regime_analysis']['stability']['transition_rate']:.4f}",
            f"Average Duration: {results['regime_analysis']['stability']['avg_duration']:.2f}",
            f"Duration Std Dev: {results['regime_analysis']['stability']['duration_std']:.2f}",
            f"Persistence: {results['regime_analysis']['stability']['persistence']:.4f}",
            "",
            "-" * 80,
            "REGIME DISTRIBUTION",
            "-" * 80,
            f"Number of Regimes Observed: {results['regime_analysis']['distribution']['regime_counts'].shape[0]}",
            f"Entropy: {results['regime_analysis']['distribution']['entropy']:.4f}",
            f"Gini Coefficient: {results['regime_analysis']['distribution']['gini_coefficient']:.4f}",
            "",
            "-" * 80,
            "TEMPORAL CONSISTENCY",
            "-" * 80,
            f"Local Consistency: {results['temporal_metrics']['avg_local_consistency']:.4f}",
            f"Autocorrelation (Lag 1): {results['temporal_metrics']['autocorr_lag1']:.4f}",
            "",
            "-" * 80,
            "OVERALL QUALITY",
            "-" * 80,
            f"Regime Quality Index: {results['regime_quality_index']:.2f}/100",
            "",
            "=" * 80,
        ]

        # Add supervised metrics if available
        if "supervised_metrics" in results:
            report_lines.extend(
                [
                    "",
                    "-" * 80,
                    "SUPERVISED EVALUATION METRICS",
                    "-" * 80,
                    f"Adjusted Rand Index: {results['supervised_metrics']['adjusted_rand_index']:.4f}",
                    f"Normalized Mutual Info: {results['supervised_metrics']['normalized_mutual_info']:.4f}",
                    f"Adjusted Mutual Info: {results['supervised_metrics']['adjusted_mutual_info']:.4f}",
                ]
            )

        report = "\n".join(report_lines)

        # Save if output file specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")

        return report
