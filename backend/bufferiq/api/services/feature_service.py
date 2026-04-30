"""Feature extraction service for real-time predictions."""

import hashlib
from typing import Dict, List, Optional

import numpy as np

from bufferiq.api.models.prediction import PredictionRequest
from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class FeatureService:
    """
    Service for extracting features from prediction requests.

    Integrates with Day 7 feature engineering modules.
    Implements feature caching for identical requests.
    """

    def __init__(self) -> None:
        """Initialize feature service."""
        self._cache: Dict[str, List[float]] = {}
        logger.info("FeatureService initialized")

    async def extract_features(
        self, request: PredictionRequest
    ) -> List[float]:
        """
        Extract features from prediction request.

        Args:
            request: Prediction request

        Returns:
            List of feature values

        Example:
            >>> features = await service.extract_features(request)
            >>> len(features)
            92
        """
        # Check cache
        cache_key = self._generate_cache_key(request)
        if cache_key in self._cache:
            logger.debug("Features found in cache")
            return self._cache[cache_key]

        # Extract features
        features = self._extract_features_impl(request)

        # Cache features
        self._cache[cache_key] = features

        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest 20%
            keys_to_remove = list(self._cache.keys())[:200]
            for key in keys_to_remove:
                del self._cache[key]

        return features

    def _extract_features_impl(
        self, request: PredictionRequest
    ) -> List[float]:
        """
        Internal feature extraction implementation.

        This is a simplified version. In production, integrate with
        Day 7 feature engineering modules.

        Args:
            request: Prediction request

        Returns:
            List of feature values
        """
        features = []

        # Content features
        content = request.content
        features.append(len(content))  # content_length
        features.append(content.count(" ") + 1)  # word_count
        features.append(content.count("!"))  # exclamation_count
        features.append(content.count("?"))  # question_count
        features.append(sum(1 for c in content if c.isupper()))  # caps_count
        features.append(content.count("#"))  # hashtag_count
        features.append(content.count("@"))  # mention_count

        # Platform features (one-hot encoded)
        platforms = ["linkedin", "twitter", "bluesky"]
        for platform in platforms:
            features.append(1.0 if request.platform == platform else 0.0)

        # Media features
        features.append(1.0 if request.has_media else 0.0)
        features.append(1.0 if request.has_link else 0.0)

        # Temporal features
        if request.scheduled_time:
            features.append(float(request.scheduled_time.hour))
            features.append(float(request.scheduled_time.weekday()))
            features.append(float(request.scheduled_time.day))
        else:
            features.extend([0.0, 0.0, 0.0])

        # Sentiment features (simplified - use proper NLP in production)
        positive_words = ["great", "awesome", "love", "excellent", "amazing"]
        negative_words = ["bad", "hate", "terrible", "awful", "poor"]

        content_lower = content.lower()
        features.append(
            sum(1 for word in positive_words if word in content_lower)
        )
        features.append(
            sum(1 for word in negative_words if word in content_lower)
        )

        # Pad to expected feature count (92 features)
        while len(features) < 92:
            features.append(0.0)

        return features[:92]  # Ensure exactly 92 features

    def _generate_cache_key(self, request: PredictionRequest) -> str:
        """
        Generate cache key for request.

        Args:
            request: Prediction request

        Returns:
            Cache key (MD5 hash)
        """
        # Create unique key from request content
        key_parts = [
            request.content,
            request.platform,
            str(request.scheduled_time) if request.scheduled_time else "",
            str(request.has_media),
            str(request.has_link),
        ]

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def extract_batch_features(
        self, requests: List[PredictionRequest]
    ) -> List[List[float]]:
        """
        Extract features for multiple requests.

        Args:
            requests: List of prediction requests

        Returns:
            List of feature lists
        """
        return [await self.extract_features(req) for req in requests]

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
        logger.info("Feature cache cleared")