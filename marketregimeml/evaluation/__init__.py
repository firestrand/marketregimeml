"""Evaluation framework for MarketRegimeML.

Provides comprehensive metrics and evaluation tools for regime detection models.
Includes lightweight compatibility wrappers for examples.
"""

from typing import Optional, Dict, Any

from marketregimeml.evaluation.metrics import RegimeMetrics
from marketregimeml.evaluation.evaluator import ModelEvaluator

# Expose functional metrics for backward compatibility
try:
    from marketregimeml.evaluation.metrics_functions import (
        regime_accuracy,
        regime_transition_matrix,
        regime_duration_stats,
        regime_stability,
        adjusted_rand_index,
        normalized_mutual_info,
        silhouette_score_regimes,
        davies_bouldin_score_regimes,
        calinski_harabasz_score_regimes,
        regime_separation_score,
        regime_purity_score,
        regime_consistency_score,
        regime_prediction_metrics,
    )
except ImportError:
    # Some environments may not include all functional metrics
    pass

# Optional enhanced metrics
try:  # pragma: no cover - optional dependency
    from marketregimeml.evaluation.enhanced_metrics import (
        EnhancedRegimeMetrics,
    )

    _ENHANCED_METRICS_AVAILABLE = True
except Exception:  # ImportError or missing extras
    EnhancedRegimeMetrics = None  # type: ignore
    _ENHANCED_METRICS_AVAILABLE = False


class RegimeEvaluator:
    """Compatibility wrapper exposing a simple evaluate_model API used in examples.

    Internally delegates to ModelEvaluator, and returns a flat dictionary with
    commonly used metrics for easy printing in examples.
    """

    def evaluate_model(
        self,
        model,
        features,
        true_regimes=None,
    ) -> Dict[str, Any]:
        evaluator = ModelEvaluator(model)
        results = evaluator.evaluate_model(features, true_regimes)

        out: Dict[str, Any] = {
            "silhouette_score": results["clustering_metrics"].get("silhouette_score"),
            "davies_bouldin_index": results["clustering_metrics"].get(
                "davies_bouldin_index"
            ),
            "calinski_harabasz_index": results["clustering_metrics"].get(
                "calinski_harabasz_index"
            ),
            "avg_confidence": results["clustering_metrics"].get("avg_confidence"),
            # Use persistence (1 - transition_rate) as scalar stability index
            "regime_stability": results["regime_analysis"]["stability"].get(
                "persistence"
            ),
        }

        # Add supervised metrics if available
        if "supervised_metrics" in results:
            out.update(results["supervised_metrics"])  # adjusted_rand_index, etc.

        # Provide full results under a namespaced key for advanced use
        out["_full_results"] = results
        return out


__all__ = [
    "RegimeMetrics",
    "ModelEvaluator",
    "RegimeEvaluator",  # compatibility wrapper
]

# Add legacy/function exports if available
for _name in [
    "regime_accuracy",
    "regime_transition_matrix",
    "regime_duration_stats",
    "regime_stability",
    "adjusted_rand_index",
    "normalized_mutual_info",
    "silhouette_score_regimes",
    "davies_bouldin_score_regimes",
    "calinski_harabasz_score_regimes",
    "regime_separation_score",
    "regime_purity_score",
    "regime_consistency_score",
    "regime_prediction_metrics",
]:
    if _name in globals():
        __all__.append(_name)

if _ENHANCED_METRICS_AVAILABLE:
    __all__.append("EnhancedRegimeMetrics")
