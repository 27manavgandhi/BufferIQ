"""Feature extraction from audience data."""

from typing import Any, Dict

import numpy as np

from bufferiq.ml.segmentation.types import AudienceDataPoint, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import UnsupportedPlatformError


class FeatureExtractor:
    """Extract numerical features from audience data points."""

    def extract(
        self, data_point: AudienceDataPoint, platform: str
    ) -> Dict[str, float]:
        """
        Extract features from a single audience data point.

        Args:
            data_point: Audience member data
            platform: Platform type

        Returns:
            Dictionary of numerical features

        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        features: Dict[str, float] = {}

        # Activity features
        features["follower_count_log"] = float(np.log1p(data_point.follower_count))
        features["following_ratio"] = (
            data_point.follower_count / max(data_point.following_count, 1)
        )
        features["post_frequency"] = (
            data_point.post_count / max(data_point.account_age_days, 1)
        )
        features["avg_engagement_rate"] = data_point.avg_engagement_rate

        # Interaction type ratios
        total_interactions = max(sum(data_point.interaction_types.values()), 1)
        features["like_ratio"] = (
            data_point.interaction_types.get("likes", 0) / total_interactions
        )
        features["comment_ratio"] = (
            data_point.interaction_types.get("comments", 0) / total_interactions
        )
        features["share_ratio"] = (
            data_point.interaction_types.get("shares", 0) / total_interactions
        )
        features["click_ratio"] = (
            data_point.interaction_types.get("clicks", 0) / total_interactions
        )

        # Content preferences
        features["text_preference"] = float("text" in data_point.content_types_engaged)
        features["image_preference"] = float(
            "image" in data_point.content_types_engaged
        )
        features["video_preference"] = float(
            "video" in data_point.content_types_engaged
        )
        features["link_preference"] = float("link" in data_point.content_types_engaged)

        # Account characteristics
        features["account_age_normalized"] = min(
            data_point.account_age_days / 365.0, 10.0
        )
        features["is_verified"] = float(data_point.verified)
        features["topic_diversity"] = min(len(data_point.topics_engaged) / 10.0, 1.0)
        features["bio_keyword_count"] = float(len(data_point.bio_keywords))

        # Platform-specific features
        platform_features = self._extract_platform_features(data_point, platform)
        features.update(platform_features)

        return features

    def _extract_platform_features(
        self, data_point: AudienceDataPoint, platform: str
    ) -> Dict[str, float]:
        """Extract platform-specific features."""
        features: Dict[str, float] = {}

        if platform == "linkedin":
            features["professional_bio"] = float(
                any(
                    kw in data_point.bio_keywords
                    for kw in ["engineer", "manager", "director", "ceo", "founder"]
                )
            )
        elif platform == "twitter":
            features["retweet_behavior"] = (
                data_point.interaction_types.get("retweets", 0)
                / max(sum(data_point.interaction_types.values()), 1)
            )
        elif platform == "bluesky":
            features["federation_activity"] = float(data_point.follower_count > 100)

        return features