import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from marketregimeml.models.gmm import GMMRegimeDetector


def test_optimize_guardrails_prefers_silhouette_over_aic_when_needed():
    # Two well-separated clusters -> silhouette should favor n=2
    X, _ = make_blobs(n_samples=400, centers=2, cluster_std=0.40, random_state=42)
    features = pd.DataFrame(X)

    model = GMMRegimeDetector(n_regimes=3, random_state=0)
    model.set_params(auto_optimize_regimes=True, min_regimes=2, max_regimes=9)
    res = model.optimize_regime_count(
        features, criteria="aic", min_silhouette=0.3, min_cluster_prop=0.40
    )
    assert int(res["optimal_regimes"]) == 2
