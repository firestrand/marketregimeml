"""Feature caching manager following Single Responsibility Principle."""

import hashlib
import json
from typing import Dict, List, Optional, Any

import pandas as pd

from marketregimeml.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureCacheManager:
    """Manages feature caching with optimized key generation and storage.

    Single Responsibility: Cache management for computed features.
    """

    def __init__(self, enable_caching: bool = True):
        """Initialize cache manager.

        Args:
            enable_caching: Whether to enable caching functionality
        """
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        logger.debug(
            f"FeatureCacheManager initialized with caching={'enabled' if enable_caching else 'disabled'}"
        )

    def get_cache_key(
        self, data: pd.DataFrame, feature_types: List[str], **kwargs
    ) -> str:
        """Generate cache key for feature computation.

        Args:
            data: Input data DataFrame
            feature_types: List of feature types to compute
            **kwargs: Additional parameters that affect computation

        Returns:
            Cache key string
        """
        if not self.enable_caching:
            return ""

        try:
            # Create hash based on data shape, feature types, and parameters
            cache_dict = {
                "data_shape": data.shape,
                "data_columns": list(data.columns),
                "feature_types": sorted(feature_types),
                "data_hash": hashlib.md5(
                    str(data.values).encode(), usedforsecurity=False
                ).hexdigest()[:8],
                "params": kwargs,
            }

            cache_string = json.dumps(cache_dict, sort_keys=True, default=str)
            return hashlib.md5(cache_string.encode(), usedforsecurity=False).hexdigest()[:16]

        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            return ""

    def get_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached features if available.

        Args:
            cache_key: Cache key for lookup

        Returns:
            Cached DataFrame if found, None otherwise
        """
        if not self.enable_caching or not cache_key or self.cache is None:
            return None

        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            return cached_result

        logger.debug(f"Cache miss for key: {cache_key[:8]}...")
        return None

    def cache_features(self, cache_key: str, features: pd.DataFrame) -> None:
        """Store computed features in cache.

        Args:
            cache_key: Cache key for storage
            features: Computed features DataFrame
        """
        if not self.enable_caching or not cache_key or self.cache is None:
            return

        try:
            self.cache[cache_key] = features.copy()
            logger.debug(
                f"Cached features for key: {cache_key[:8]}... (shape: {features.shape})"
            )
        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")

    def clear_cache(self) -> None:
        """Clear all cached features."""
        if self.cache is not None:
            cache_size = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared {cache_size} cached feature sets")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enable_caching or self.cache is None:
            return {"enabled": False, "size": 0, "memory_usage": 0}

        memory_usage = 0
        for cached_df in self.cache.values():
            if isinstance(cached_df, pd.DataFrame):
                memory_usage += cached_df.memory_usage(deep=True).sum()

        return {
            "enabled": True,
            "size": len(self.cache),
            "memory_usage": memory_usage,
            "cache_keys": list(self.cache.keys()),
        }

    def remove_cached_item(self, cache_key: str) -> bool:
        """Remove specific cached item.

        Args:
            cache_key: Key of item to remove

        Returns:
            True if item was removed, False if not found
        """
        if not self.enable_caching or not cache_key or self.cache is None:
            return False

        if cache_key in self.cache:
            del self.cache[cache_key]
            logger.debug(f"Removed cached item: {cache_key[:8]}...")
            return True

        return False
