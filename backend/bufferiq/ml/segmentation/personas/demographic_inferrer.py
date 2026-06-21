"""Demographic inference from audience data."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bufferiq.ml.segmentation.types import AudienceDataPoint, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import UnsupportedPlatformError


class DemographicInferrer:
    """Infer demographic characteristics from audience data."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize demographic inferrer."""
        self.config = config or {}

    def infer(
        self, audience_members: List[AudienceDataPoint], platform: str
    ) -> Dict[str, Any]:
        """
        Infer demographics from audience members.

        Args:
            audience_members: List of audience members
            platform: Platform type

        Returns:
            Dictionary of inferred demographics

        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        if not audience_members:
            return self._empty_demographics()

        age_range = self._infer_age_range(audience_members, platform)
        location = self._infer_location(audience_members)
        language = self._infer_language(audience_members)
        verified_ratio = self._calculate_verified_ratio(audience_members)

        return {
            "age_range": age_range,
            "location": location,
            "language": language,
            "verified_ratio": verified_ratio,
        }

    def _infer_age_range(
        self, audience_members: List[AudienceDataPoint], platform: str
    ) -> Tuple[int, int]:
        """Infer age range based on platform and behavior."""
        platform_age_bases = {
            "linkedin": (25, 55),
            "twitter": (18, 45),
            "bluesky": (20, 40),
        }

        base_min, base_max = platform_age_bases.get(platform, (25, 55))

        # Adjust based on verified ratio
        verified_ratio = self._calculate_verified_ratio(audience_members)
        if verified_ratio > 0.5:
            base_min += 5

        return (base_min, base_max)

    def _infer_location(self, audience_members: List[AudienceDataPoint]) -> Optional[str]:
        """Infer primary location."""
        locations = [
            m.location for m in audience_members if m.location
        ]

        if not locations:
            return None

        # Return most common location
        from collections import Counter
        location_counts = Counter(locations)
        most_common = location_counts.most_common(1)

        return most_common[0][0] if most_common else None

    def _infer_language(self, audience_members: List[AudienceDataPoint]) -> str:
        """Infer primary language."""
        if not audience_members:
            return "en"

        languages = [m.language for m in audience_members]
        from collections import Counter
        lang_counts = Counter(languages)
        most_common = lang_counts.most_common(1)

        return most_common[0][0] if most_common else "en"

    def _calculate_verified_ratio(
        self, audience_members: List[AudienceDataPoint]
    ) -> float:
        """Calculate ratio of verified accounts."""
        if not audience_members:
            return 0.0

        verified_count = sum(1 for m in audience_members if m.verified)
        return verified_count / len(audience_members)

    def _empty_demographics(self) -> Dict[str, Any]:
        """Return empty demographics."""
        return {
            "age_range": (25, 55),
            "location": None,
            "language": "en",
            "verified_ratio": 0.0,
        }