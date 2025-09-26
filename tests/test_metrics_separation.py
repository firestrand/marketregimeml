import numpy as np
from marketregimeml.evaluation import metrics_functions as mf


def test_regime_separation_score_two_clusters():
    # Two well-separated clusters in 2D
    np.random.seed(0)
    a = np.random.normal(loc=0.0, scale=0.1, size=(100, 2))
    b = np.random.normal(loc=3.0, scale=0.1, size=(100, 2))
    X = np.vstack([a, b])
    y = np.array([0] * 100 + [1] * 100)

    sep = mf.regime_separation_score(X, y)
    sil = mf.silhouette_score_regimes(X, y)

    assert sep > 5.0  # strong inter/intra ratio
    assert sil > 0.8  # high silhouette for well-separated clusters


def test_regime_separation_score_single_cluster():
    X = np.random.normal(size=(50, 2))
    y = np.zeros(50, dtype=int)
    sep = mf.regime_separation_score(X, y)
    sil = mf.silhouette_score_regimes(X, y)
    assert sep == 0.0
    assert sil == 0.0

