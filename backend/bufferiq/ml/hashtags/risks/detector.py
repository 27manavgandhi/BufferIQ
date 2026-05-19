"""
Hashtag risk detector.

Detects risky hashtags and brand safety issues.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import json
import os

from bufferiq.ml.hashtags.extraction.extractor import SUPPORTED_PLATFORMS


@dataclass
class HashtagRisk:
    """Hashtag risk assessment."""

    hashtag: str
    risk_level: str  # "none", "low", "medium", "high", "critical"

    # Risk factors
    is_hijacked: bool = False
    is_controversial: bool = False
    is_spam: bool = False
    is_banned: bool = False
    is_nsfw: bool = False

    # Details
    risk_reasons: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    # Recommendations
    recommendation: str = "use"  # "use", "use_with_caution", "avoid"
    alternatives: List[str] = field(default_factory=list)


class HashtagRiskDetector:
    """
    Detect risky hashtags and brand safety issues.

    Identifies hijacked, controversial, spam,
    and banned hashtags.

    Example:
```python
        detector = HashtagRiskDetector()
        risk = detector.assess(
            hashtag="example",
            platform="linkedin"
        )

        if risk.risk_level != "none":
            print(f"⚠️  #{risk.hashtag} - Risk: {risk.risk_level}")
            print(f"Reasons:")
            for reason in risk.risk_reasons:
                print(f"  - {reason}")
            print(f"Recommendation: {risk.recommendation}")

            if risk.alternatives:
                print(f"Alternatives: {risk.alternatives}")
```
    """

    def __init__(self, banned_hashtags_path: Optional[str] = None) -> None:
        """
        Initialize risk detector.

        Args:
            banned_hashtags_path: Path to banned hashtags JSON
        """
        self.banned_hashtags = self._load_banned_hashtags(banned_hashtags_path)
        self.controversial_keywords = {
            "political",
            "religion",
            "controversial",
            # Add more as needed
        }

    def assess(
        self,
        hashtag: str,
        platform: str,
        check_realtime: bool = True,
    ) -> HashtagRisk:
        """
        Assess hashtag safety.

        Args:
            hashtag: Hashtag to check
            platform: Platform name
            check_realtime: Check real-time data

        Returns:
            Risk assessment

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Platform not supported: {platform}")

        hashtag_lower = hashtag.lower().lstrip("#")

        # Initialize risk
        risk = HashtagRisk(
            hashtag=hashtag_lower,
            risk_level="none",
            recommendation="use",
        )

        # Check if banned
        if hashtag_lower in self.banned_hashtags:
            risk.is_banned = True
            risk.risk_reasons.append("Hashtag is on banned list")
            risk.risk_level = "critical"
            risk.recommendation = "avoid"

        # Check for spam patterns
        if self._is_spam_pattern(hashtag_lower):
            risk.is_spam = True
            risk.risk_reasons.append("Matches spam pattern")
            risk.risk_level = max(risk.risk_level, "high", key=self._risk_order)
            risk.recommendation = "avoid"

        # Check controversial keywords
        if self._contains_controversial(hashtag_lower):
            risk.is_controversial = True
            risk.risk_reasons.append("Contains controversial keywords")
            risk.risk_level = max(risk.risk_level, "medium", key=self._risk_order)
            risk.recommendation = "use_with_caution"

        # Check NSFW
        if self._is_nsfw(hashtag_lower):
            risk.is_nsfw = True
            risk.risk_reasons.append("May contain NSFW content")
            risk.risk_level = max(risk.risk_level, "high", key=self._risk_order)
            risk.recommendation = "avoid"

        # Generate alternatives if risky
        if risk.risk_level not in ["none", "low"]:
            risk.alternatives = self._suggest_alternatives(hashtag_lower)

        return risk

    def _load_banned_hashtags(self, path: Optional[str]) -> Set[str]:
        """Load banned hashtags from JSON file."""
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    return set(data.get("banned", []))
            except Exception:
                pass

        # Default banned list
        return {
            "spam",
            "followback",
            "followme",
            "like4like",
            "follow4follow",
        }

    def _is_spam_pattern(self, hashtag: str) -> bool:
        """Check if hashtag matches spam patterns."""
        spam_patterns = [
            "follow",
            "like4",
            "f4f",
            "l4l",
            "followback",
            "followme",
        ]

        return any(pattern in hashtag for pattern in spam_patterns)

    def _contains_controversial(self, hashtag: str) -> bool:
        """Check if contains controversial keywords."""
        return any(keyword in hashtag for keyword in self.controversial_keywords)

    def _is_nsfw(self, hashtag: str) -> bool:
        """Check if potentially NSFW."""
        nsfw_keywords = {"nsfw", "adult", "xxx"}
        return any(keyword in hashtag for keyword in nsfw_keywords)

    def _suggest_alternatives(self, hashtag: str) -> List[str]:
        """Suggest alternative hashtags."""
        # Simple alternatives based on common substitutions
        alternatives = []

        # If spam, suggest professional versions
        if "follow" in hashtag:
            alternatives.append("networking")
            alternatives.append("connect")

        # Generic suggestions
        if not alternatives:
            alternatives.extend(["professional", "business", "industry"])

        return alternatives[:3]

    def _risk_order(self, risk_level: str) -> int:
        """Get numeric order for risk level comparison."""
        order = {
            "none": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }
        return order.get(risk_level, 0)