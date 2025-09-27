"""Feature engineering module for MarketRegimeML.

Provides advanced market features including volatility estimators,
entropy measures, and technical indicators.
"""

from marketregimeml.features.engine import FeatureEngine
from marketregimeml.features.volatility import VolatilityFeatures
from marketregimeml.features.entropy import EntropyFeatures
from marketregimeml.features.statistical import StatisticalFeatures
from marketregimeml.features.technical import TechnicalIndicators
from marketregimeml.features.base import BaseFeatureCalculator
from marketregimeml.features.common_features import (
    PriceFeatures,
    StatisticalFeatures as CommonStatisticalFeatures,
    RegimeFeatures,
    TechnicalFeatures,
)


__all__ = [
    "FeatureEngine",
    "VolatilityFeatures",
    "EntropyFeatures",
    "StatisticalFeatures",
    "TechnicalIndicators",
    "BaseFeatureCalculator",
    "PriceFeatures",
    "CommonStatisticalFeatures",
    "RegimeFeatures",
    "TechnicalFeatures",
]
