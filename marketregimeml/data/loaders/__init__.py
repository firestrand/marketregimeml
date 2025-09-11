"""Data loaders for MarketRegimeML.

Optional third-party integrations are imported lazily and may be None if the
extra dependency is not installed. This prevents import-time errors when users
work only with core functionality.
"""

from marketregimeml.data.loaders.base import MarketDataLoader

# Optional providers
try:  # pragma: no cover - optional extras
    from marketregimeml.data.loaders.alphavantage import AlphaVantageLoader
except Exception:  # ImportError or missing deps
    AlphaVantageLoader = None  # type: ignore

try:  # pragma: no cover
    from marketregimeml.data.loaders.oanda import OANDADataLoader
except Exception:
    OANDADataLoader = None  # type: ignore

try:  # pragma: no cover
    from marketregimeml.data.loaders.binance import BinanceDataLoader
except Exception:
    BinanceDataLoader = None  # type: ignore

try:  # pragma: no cover
    from marketregimeml.data.loaders.kraken import KrakenDataLoader
except Exception:
    KrakenDataLoader = None  # type: ignore

__all__ = [
    "MarketDataLoader",
    "AlphaVantageLoader",
    "OANDADataLoader",
    "BinanceDataLoader",
    "KrakenDataLoader",
]
