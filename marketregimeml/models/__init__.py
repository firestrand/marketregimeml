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

# Deep Learning models removed - were just stubs

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
]
