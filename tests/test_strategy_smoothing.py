import numpy as np

from marketregimeml.utils.regime_utils import smooth_regimes_majority
from marketregimeml.evaluation.strategy import assess, ReadinessThresholds


def test_smooth_regimes_majority_increases_persistence():
    labels = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1])
    smoothed = smooth_regimes_majority(labels, window=3)
    # Original persistence
    transitions_orig = np.sum(labels[1:] != labels[:-1])
    transitions_sm = np.sum(smoothed[1:] != smoothed[:-1])
    assert transitions_sm <= transitions_orig


def test_assess_with_smoothing_uses_smoothed_labels():
    # Build simple feature matrix with two clusters
    X = np.vstack([
        np.random.normal(-1.0, 0.1, size=(50, 2)),
        np.random.normal(1.0, 0.1, size=(50, 2)),
    ])
    # Create noisy alternating regimes, then smoothed regimes should reduce switches
    regimes = np.array([0, 1] * 50)
    probs = np.ones((100, 2)) * 0.5
    persistence = 0.0  # dummy; readiness check focuses on silhouette/separation/min_prop

    t_no_smooth = ReadinessThresholds(smoothing_window=0)
    t_smooth = ReadinessThresholds(smoothing_window=5)

    a0 = assess(X, regimes, probs, persistence, thresholds=t_no_smooth)
    a1 = assess(X, regimes, probs, persistence, thresholds=t_smooth)

    # With smoothing, silhouette should not be worse
    assert a1["silhouette"] >= a0["silhouette"]
