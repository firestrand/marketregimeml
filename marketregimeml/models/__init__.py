"""Regime detection models for MarketRegimeML.

Provides various models for detecting market regimes including
Hidden Markov Models, Gaussian Mixture Models, GARCH models,
machine learning models, and ensemble methods.
"""

# Core models - always available
from marketregimeml.models.base import BaseRegimeDetector
from marketregimeml.models.hmm import HMMRegimeDetector
from marketregimeml.models.gmm import GMMRegimeDetector
from marketregimeml.models.garch import GARCHRegimeDetector, MSGARCHRegimeDetector

# Machine Learning models
from marketregimeml.models.ml import (
    RandomForestRegimeClassifier,
    XGBoostRegimeClassifier,
    SVMRegimeClassifier
)

# Ensemble models
from marketregimeml.models.ensemble import (
    EnsembleRegimeDetector,
    VotingEnsemble,
    StackingEnsemble,
    BaggingEnsemble,
    BoostingEnsemble
)

# Deep Learning models (optional, requires torch)
try:
    from marketregimeml.models.deep_learning import (
        LSTMRegimeDetector,
        TransformerRegimeDetector,
        CNNLSTMRegimeDetector
    )
    _DEEP_LEARNING_AVAILABLE = True
except Exception:
    LSTMRegimeDetector = None
    TransformerRegimeDetector = None
    CNNLSTMRegimeDetector = None
    _DEEP_LEARNING_AVAILABLE = False

__all__ = [
    # Base
    'BaseRegimeDetector',
    
    # Statistical models
    'HMMRegimeDetector',
    'GMMRegimeDetector',
    'GARCHRegimeDetector',
    'MSGARCHRegimeDetector',
    
    # ML models
    'RandomForestRegimeClassifier',
    'XGBoostRegimeClassifier',
    'SVMRegimeClassifier',
    
    # Ensemble models
    'EnsembleRegimeDetector',
    'VotingEnsemble',
    'StackingEnsemble', 
    'BaggingEnsemble',
    'BoostingEnsemble',
    
    # Deep Learning (when available)
    'LSTMRegimeDetector',
    'TransformerRegimeDetector',
    'CNNLSTMRegimeDetector',
]
