"""
Hashtag extractor with platform-specific rules.

Extracts and normalizes hashtags from content across different platforms.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class ExtractedHashtag:
    """Extracted hashtag with metadata."""

    hashtag: str  # Normalized (lowercase, no #)
    original: str  # Original casing with #
    position: int  # Position in text
    context: str  # Surrounding text (10 words)

    # Usage
    post_id: str
    platform: str
    created_at: datetime

    # Engagement
    engagement: int
    reach: Optional[int] = None
    impressions: Optional[int] = None


@dataclass
class HashtagExtractionResult:
    """Result of hashtag extraction."""

    hashtags: List[ExtractedHashtag]
    total_count: int
    unique_count: int
    duplicates: Dict[str, int]
    variants: Dict[str, List[str]] = field(default_factory=dict)

    # Patterns
    avg_position: float = 0.0
    placement_pattern: str = "mixed"

    # Platform stats
    platform_distribution: Dict[str, int] = field(default_factory=dict)


class HashtagExtractor:
    """
    Extract and normalize hashtags from content.

    Handles platform-specific extraction rules,
    variant detection, and normalization.

    Example:
```python
        extractor = HashtagExtractor()
        result = extractor.extract(
            text="Great #AI insights! #MachineLearning #ML",
            post_id="post123",
            platform="linkedin"
        )

        print(f"Found {result.unique_count} unique hashtags")
        for hashtag in result.hashtags:
            print(f"  {hashtag.hashtag} at position {hashtag.position}")
```
    """

    def __init__(self) -> None:
        """Initialize hashtag extractor."""
        # Platform-specific patterns
        self.patterns = {
            "linkedin": r"#[A-Za-z0-9_]+(?![A-Za-z0-9_])",
            "twitter": r"#[A-Za-z0-9_]+(?![A-Za-z0-9_])",
            "bluesky": r"#[A-Za-z0-9_]+(?![A-Za-z0-9_])",
        }

        # Common variants
        self.variant_map: Dict[str, List[str]] = {
            "ai": ["artificialintelligence", "aitech"],
            "ml": ["machinelearning", "mlai"],
            "seo": ["searchengineoptimization"],
            "socialmedia": ["sm", "smm"],
            "digitalmarketing": ["dm", "digimarketing"],
        }

    def extract(
        self,
        text: str,
        post_id: str,
        platform: str,
        engagement: int = 0,
        reach: Optional[int] = None,
        impressions: Optional[int] = None,
        created_at: Optional[datetime] = None,
    ) -> HashtagExtractionResult:
        """
        Extract hashtags from text.

        Args:
            text: Text to extract from
            post_id: Post identifier
            platform: Platform name
            engagement: Post engagement
            reach: Post reach
            impressions: Post impressions
            created_at: Post creation time

        Returns:
            Extraction result with hashtags

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        if created_at is None:
            created_at = datetime.now()

        # Extract hashtags using regex
        pattern = self.patterns[platform]
        matches = list(re.finditer(pattern, text))

        # Process each match
        hashtags: List[ExtractedHashtag] = []
        for match in matches:
            original = match.group(0)
            normalized = self.normalize(original)
            position = match.start()

            # Extract context (10 words around hashtag)
            context = self._extract_context(text, position, words=10)

            hashtag = ExtractedHashtag(
                hashtag=normalized,
                original=original,
                position=position,
                context=context,
                post_id=post_id,
                platform=platform,
                created_at=created_at,
                engagement=engagement,
                reach=reach,
                impressions=impressions,
            )
            hashtags.append(hashtag)

        # Calculate statistics
        total_count = len(hashtags)
        normalized_tags = [h.hashtag for h in hashtags]
        unique_count = len(set(normalized_tags))

        # Find duplicates
        counter = Counter(normalized_tags)
        duplicates = {tag: count for tag, count in counter.items() if count > 1}

        # Detect variants
        variants = self.detect_variants(normalized_tags)

        # Calculate average position
        avg_position = (
            sum(h.position for h in hashtags) / total_count if total_count > 0 else 0.0
        )

        # Determine placement pattern
        placement_pattern = self._determine_placement_pattern(hashtags, len(text))

        # Platform distribution
        platform_dist = {platform: total_count}

        return HashtagExtractionResult(
            hashtags=hashtags,
            total_count=total_count,
            unique_count=unique_count,
            duplicates=duplicates,
            variants=variants,
            avg_position=avg_position,
            placement_pattern=placement_pattern,
            platform_distribution=platform_dist,
        )

    def normalize(self, hashtag: str) -> str:
        """
        Normalize hashtag to canonical form.

        Args:
            hashtag: Raw hashtag (with or without #)

        Returns:
            Normalized hashtag (lowercase, no #)
        """
        # Remove # and lowercase
        normalized = hashtag.lstrip("#").lower()
        # Remove underscores
        normalized = normalized.replace("_", "")
        return normalized

    def detect_variants(self, hashtags: List[str]) -> Dict[str, List[str]]:
        """
        Detect hashtag variants.

        Args:
            hashtags: List of normalized hashtags

        Returns:
            Map of canonical -> variants
        """
        variants: Dict[str, List[str]] = {}

        # Check known variants
        for hashtag in hashtags:
            for canonical, variant_list in self.variant_map.items():
                if hashtag in variant_list or hashtag == canonical:
                    if canonical not in variants:
                        variants[canonical] = []
                    if hashtag not in variants[canonical]:
                        variants[canonical].append(hashtag)

        return variants

    def _extract_context(self, text: str, position: int, words: int = 10) -> str:
        """Extract surrounding context around position."""
        # Find word boundaries
        words_list = text.split()
        char_count = 0
        word_index = 0

        for i, word in enumerate(words_list):
            char_count += len(word) + 1  # +1 for space
            if char_count > position:
                word_index = i
                break

        # Get surrounding words
        start_idx = max(0, word_index - words // 2)
        end_idx = min(len(words_list), word_index + words // 2)

        context_words = words_list[start_idx:end_idx]
        return " ".join(context_words)

    def _determine_placement_pattern(
        self, hashtags: List[ExtractedHashtag], text_length: int
    ) -> str:
        """Determine hashtag placement pattern."""
        if not hashtags:
            return "none"

        positions = [h.position for h in hashtags]

        # Check if all at beginning (first 20%)
        if all(p < text_length * 0.2 for p in positions):
            return "beginning"

        # Check if all at end (last 20%)
        if all(p > text_length * 0.8 for p in positions):
            return "end"

        # Check if all in middle
        if all(text_length * 0.3 < p < text_length * 0.7 for p in positions):
            return "middle"

        return "mixed"