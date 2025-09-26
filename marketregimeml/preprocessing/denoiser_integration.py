"""Optional integration with financial-diffusion-denoiser.

Loads the denoiser from a local src path if available and exposes a tiny
adapter to denoise a 1D price series. This module is fully optional and
fails soft if the external project is unavailable.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import numpy as np


def _ensure_path_on_sys_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def get_denoiser(src_path: Optional[str] = None, **kwargs: Any):
    """Attempt to import and construct DiffusionDenoiser.

    Args:
        src_path: Path to the external project's src directory. If None, will
                  try env var FIN_DIFF_DENOISER_SRC, then a common default.
        **kwargs: Passed through to DiffusionDenoiser(...)

    Returns:
        Instance of DiffusionDenoiser or None if unavailable.
    """
    src = src_path or os.getenv(
        "FIN_DIFF_DENOISER_SRC",
        "/Users/firestrand/Projects/financial-diffusion-denoiser/src",
    )
    try:
        _ensure_path_on_sys_path(src)
        from financial_diffusion_denoiser.denoiser import DiffusionDenoiser

        return DiffusionDenoiser(**kwargs)
    except Exception:
        return None


def denoise_series(values: np.ndarray, steps: Optional[int] = None, **kwargs: Any) -> Optional[np.ndarray]:
    """Denoise a 1D price series with the external denoiser if available.

    Args:
        values: 1D numpy array of prices
        steps: Optional number of diffusion steps (adapter to API)
        **kwargs: Construct-time args for the denoiser

    Returns:
        Denoised 1D numpy array, or None if denoiser not available.
    """
    try:
        x = np.asarray(values).astype(float)
        if x.ndim != 1 or x.size == 0:
            return None
        dn = get_denoiser(**kwargs)
        if dn is None:
            return None
        # Fit simple stats (placeholder API) then denoise
        dn.fit(x)
        y = dn.denoise(x, steps=steps)
        return y.astype(float)
    except Exception:
        return None

