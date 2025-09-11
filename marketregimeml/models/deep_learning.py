"""Deep learning based regime detection models (optional).

This module provides deep learning detectors that rely on PyTorch. To keep
the package import-safe when PyTorch isn’t installed, we guard imports and
expose lightweight stubs that raise informative errors on use.
"""

from marketregimeml.models.base import BaseRegimeDetector
from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)

# Optional PyTorch dependency
try:  # pragma: no cover - optional dependency
    import torch  # type: ignore

    HAS_PYTORCH = True
    logger.debug("PyTorch detected for deep learning models")
except Exception:
    HAS_PYTORCH = False
    torch = None  # type: ignore
    logger.info(
        "PyTorch not installed; deep learning models are unavailable."
    )


class _RequiresPyTorch(BaseRegimeDetector):
    """Base stub that raises if PyTorch is unavailable."""

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not HAS_PYTORCH:
            raise ImportError(
                "PyTorch is required for deep learning models. "
                "Install it with: pip install torch"
            )
        super().__init__(
            n_regimes=kwargs.get("n_regimes", 3),
            random_state=kwargs.get("random_state", None),
        )


class LSTMRegimeDetector(_RequiresPyTorch):
    """LSTM-based regime detector (requires PyTorch)."""

    def __init__(self, n_regimes: int = 3, sequence_length: int = 30, **kwargs):
        super().__init__(n_regimes=n_regimes, random_state=kwargs.get("random_state"))
        # Full implementation intentionally omitted in this build.
        # This placeholder preserves import-time compatibility.
        raise ImportError(
            "LSTMRegimeDetector requires PyTorch and is not enabled in this build."
        )


class TransformerRegimeDetector(_RequiresPyTorch):
    """Transformer-based regime detector (requires PyTorch)."""

    def __init__(self, n_regimes: int = 3, sequence_length: int = 30, **kwargs):
        super().__init__(n_regimes=n_regimes, random_state=kwargs.get("random_state"))
        raise ImportError(
            "TransformerRegimeDetector requires PyTorch and is not enabled in this build."
        )


class CNNLSTMRegimeDetector(_RequiresPyTorch):
    """CNN-LSTM hybrid regime detector (requires PyTorch)."""

    def __init__(self, n_regimes: int = 3, sequence_length: int = 30, **kwargs):
        super().__init__(n_regimes=n_regimes, random_state=kwargs.get("random_state"))
        raise ImportError(
            "CNNLSTMRegimeDetector requires PyTorch and is not enabled in this build."
        )


__all__ = [
    "LSTMRegimeDetector",
    "TransformerRegimeDetector",
    "CNNLSTMRegimeDetector",
]

