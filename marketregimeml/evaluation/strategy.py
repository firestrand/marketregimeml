"""Strategy-oriented evaluation utilities (separation + readiness)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from marketregimeml.evaluation import metrics_functions as mf
from marketregimeml.utils.regime_utils import smooth_regimes_majority


@dataclass
class ReadinessThresholds:
    min_silhouette: float = 0.30
    min_separation: float = 1.50
    min_persistence: float = 0.85
    min_cluster_prop: float = 0.10
    # Optional smoothing to apply before evaluation (intraday often benefits)
    smoothing_window: int = 0


def assess(
    features: np.ndarray,
    regimes: np.ndarray,
    probabilities: np.ndarray,
    persistence: float,
    thresholds: ReadinessThresholds | None = None,
) -> Dict[str, float | bool]:
    """Compute separation metrics and strategy readiness flag.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    regimes : np.ndarray
        Predicted regime labels
    probabilities : np.ndarray
        Predicted regime probabilities
    persistence : float
        Persistence metric (1 - transition rate)
    thresholds : ReadinessThresholds | None
        Thresholds for readiness; default values used if None.
    """
    if thresholds is None:
        thresholds = ReadinessThresholds()

    # Optional smoothing to improve persistence and reduce noise
    regimes_eval = regimes
    if hasattr(thresholds, "smoothing_window"):
        # Backward-compatible: if thresholds carries this attr
        sw = getattr(thresholds, "smoothing_window")
        if sw and isinstance(sw, (int, float)) and sw >= 2:
            regimes_eval = smooth_regimes_majority(regimes, int(sw))

    sep = float(mf.regime_separation_score(features, regimes_eval))
    sil = float(mf.silhouette_score_regimes(features, regimes_eval))
    ch = float(mf.calinski_harabasz_score_regimes(features, regimes))
    dbi = float(mf.davies_bouldin_score_regimes(features, regimes))

    counts = np.bincount(regimes_eval)
    min_prop = float(counts.min() / len(regimes_eval)) if len(counts) else 0.0

    ready = (
        sil >= thresholds.min_silhouette
        and sep >= thresholds.min_separation
        and persistence >= thresholds.min_persistence
        and min_prop >= thresholds.min_cluster_prop
    )

    return {
        "separation": sep,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": dbi,
        "min_cluster_prop": min_prop,
        "strategy_ready": bool(ready),
    }
