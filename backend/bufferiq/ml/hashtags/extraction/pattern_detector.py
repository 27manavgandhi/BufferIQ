"""
Hashtag usage pattern detector.

Analyzes patterns in hashtag usage across posts.
"""

from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

from bufferiq.ml.hashtags.extraction.extractor import ExtractedHashtag


@dataclass
class HashtagPattern:
    """Detected hashtag usage pattern."""

    pattern_type: str  # "frequent", "seasonal", "campaign", "burst"
    confidence: float  # 0-1
    description: str
    examples: List[str]
    metadata: Dict[str, float]


class HashtagPatternDetector:
    """
    Detect patterns in hashtag usage.

    Identifies frequent, seasonal, campaign, and burst patterns.

    Example:
```python
        detector = HashtagPatternDetector()
        patterns = detector.detect_patterns(extracted_hashtags)

        for pattern in patterns:
            print(f"{pattern.pattern_type}: {pattern.description}")
            print(f"  Confidence: {pattern.confidence:.2f}")
```
    """

    def __init__(self, min_frequency: int = 3) -> None:
        """
        Initialize pattern detector.

        Args:
            min_frequency: Minimum frequency to consider pattern
        """
        self.min_frequency = min_frequency

    def detect_patterns(
        self, hashtags: List[ExtractedHashtag]
    ) -> List[HashtagPattern]:
        """
        Detect usage patterns in hashtags.

        Args:
            hashtags: List of extracted hashtags

        Returns:
            List of detected patterns
        """
        patterns: List[HashtagPattern] = []

        # Frequency analysis
        frequency_pattern = self._detect_frequency_patterns(hashtags)
        if frequency_pattern:
            patterns.append(frequency_pattern)

        # Co-occurrence patterns
        cooccurrence_patterns = self._detect_cooccurrence_patterns(hashtags)
        patterns.extend(cooccurrence_patterns)

        # Timing patterns
        timing_pattern = self._detect_timing_patterns(hashtags)
        if timing_pattern:
            patterns.append(timing_pattern)

        return patterns

    def _detect_frequency_patterns(
        self, hashtags: List[ExtractedHashtag]
    ) -> HashtagPattern | None:
        """Detect frequently used hashtags."""
        if not hashtags:
            return None

        # Count hashtag frequencies
        freq_count: Dict[str, int] = defaultdict(int)
        for ht in hashtags:
            freq_count[ht.hashtag] += 1

        # Find high-frequency hashtags
        frequent = [
            (tag, count)
            for tag, count in freq_count.items()
            if count >= self.min_frequency
        ]

        if not frequent:
            return None

        # Sort by frequency
        frequent.sort(key=lambda x: x[1], reverse=True)

        # Calculate confidence based on consistency
        total_posts = len(set(ht.post_id for ht in hashtags))
        avg_frequency = sum(count for _, count in frequent) / len(frequent)
        confidence = min(1.0, avg_frequency / total_posts)

        return HashtagPattern(
            pattern_type="frequent",
            confidence=confidence,
            description=f"Found {len(frequent)} frequently used hashtags",
            examples=[tag for tag, _ in frequent[:5]],
            metadata={"avg_frequency": avg_frequency, "total_posts": total_posts},
        )

    def _detect_cooccurrence_patterns(
        self, hashtags: List[ExtractedHashtag]
    ) -> List[HashtagPattern]:
        """Detect hashtags that frequently appear together."""
        patterns: List[HashtagPattern] = []

        # Group by post
        post_hashtags: Dict[str, List[str]] = defaultdict(list)
        for ht in hashtags:
            post_hashtags[ht.post_id].append(ht.hashtag)

        # Find co-occurrences
        cooccurrence: Dict[tuple[str, str], int] = defaultdict(int)
        for post_tags in post_hashtags.values():
            unique_tags = list(set(post_tags))
            for i, tag1 in enumerate(unique_tags):
                for tag2 in unique_tags[i + 1 :]:
                    pair = tuple(sorted([tag1, tag2]))
                    cooccurrence[pair] += 1

        # Find significant co-occurrences
        significant = [
            (pair, count)
            for pair, count in cooccurrence.items()
            if count >= self.min_frequency
        ]

        if significant:
            significant.sort(key=lambda x: x[1], reverse=True)
            top_pair, top_count = significant[0]

            confidence = min(1.0, top_count / len(post_hashtags))

            pattern = HashtagPattern(
                pattern_type="cooccurrence",
                confidence=confidence,
                description=f"Hashtags often used together: {', '.join(top_pair)}",
                examples=list(top_pair),
                metadata={"cooccurrence_count": top_count},
            )
            patterns.append(pattern)

        return patterns

    def _detect_timing_patterns(
        self, hashtags: List[ExtractedHashtag]
    ) -> HashtagPattern | None:
        """Detect timing patterns in hashtag usage."""
        if not hashtags:
            return None

        # Group by hour of day
        hour_distribution: Dict[int, int] = defaultdict(int)
        for ht in hashtags:
            hour = ht.created_at.hour
            hour_distribution[hour] += 1

        if not hour_distribution:
            return None

        # Find peak hour
        peak_hour = max(hour_distribution, key=hour_distribution.get)  # type: ignore
        peak_count = hour_distribution[peak_hour]

        # Calculate confidence
        total = sum(hour_distribution.values())
        confidence = peak_count / total if total > 0 else 0.0

        return HashtagPattern(
            pattern_type="timing",
            confidence=confidence,
            description=f"Peak posting time: {peak_hour}:00",
            examples=[],
            metadata={"peak_hour": peak_hour, "peak_percentage": confidence},
        )