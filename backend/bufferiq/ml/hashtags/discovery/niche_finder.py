"""
Niche hashtag finder for low-competition opportunities.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class NicheOpportunity:
    """Niche hashtag opportunity."""

    hashtag: str
    volume: int  # Usage volume
    competition_score: float  # 0-100, lower is better
    effectiveness_score: float  # 0-100
    opportunity_score: float  # 0-100


class NicheHashtagFinder:
    """
    Find niche hashtag opportunities.

    Identifies hashtags with good engagement but lower competition.

    Example:
```python
        finder = NicheHashtagFinder()
        niches = finder.find_niches(
            seed_hashtag="ai",
            max_competition=30.0
        )

        for niche in niches:
            print(f"#{niche.hashtag}")
            print(f"  Competition: {niche.competition_score:.1f}")
            print(f"  Opportunity: {niche.opportunity_score:.1f}")
```
    """

    def __init__(self) -> None:
        """Initialize niche finder."""
        pass

    def find_niches(
        self,
        seed_hashtag: str,
        max_competition: float = 50.0,
        min_effectiveness: float = 70.0,
        limit: int = 20,
    ) -> List[NicheOpportunity]:
        """
        Find niche opportunities.

        Args:
            seed_hashtag: Seed hashtag for discovery
            max_competition: Maximum competition score
            min_effectiveness: Minimum effectiveness score
            limit: Maximum results

        Returns:
            List of niche opportunities
        """
        # Mock niche discovery
        # In production, query database for hashtags with:
        # - Lower usage volume (less competition)
        # - Good engagement rates (effective)
        # - Related to seed hashtag

        niches: List[NicheOpportunity] = []

        # Generate niche variants
        variants = [
            f"{seed_hashtag}tips",
            f"{seed_hashtag}101",
            f"{seed_hashtag}basics",
            f"{seed_hashtag}explained",
            f"{seed_hashtag}trends2024",
        ]

        for variant in variants[:limit]:
            # Mock scores
            volume = 150  # Lower volume = less competition
            competition = 25.0
            effectiveness = 85.0

            # Calculate opportunity score
            opportunity = self._calculate_opportunity_score(
                competition, effectiveness, volume
            )

            if competition <= max_competition and effectiveness >= min_effectiveness:
                niche = NicheOpportunity(
                    hashtag=variant,
                    volume=volume,
                    competition_score=competition,
                    effectiveness_score=effectiveness,
                    opportunity_score=opportunity,
                )
                niches.append(niche)

        # Sort by opportunity score
        niches.sort(key=lambda x: x.opportunity_score, reverse=True)

        return niches

    def _calculate_opportunity_score(
        self, competition: float, effectiveness: float, volume: int
    ) -> float:
        """
        Calculate opportunity score.

        Lower competition + higher effectiveness + moderate volume = higher opportunity

        Args:
            competition: Competition score (0-100)
            effectiveness: Effectiveness score (0-100)
            volume: Usage volume

        Returns:
            Opportunity score (0-100)
        """
        # Invert competition (lower is better)
        competition_factor = (100.0 - competition) / 100.0

        # Effectiveness factor
        effectiveness_factor = effectiveness / 100.0

        # Volume factor (prefer moderate volume, not too high or low)
        if 50 <= volume <= 500:
            volume_factor = 1.0
        elif volume < 50:
            volume_factor = 0.7  # Too low, might not be relevant
        else:
            volume_factor = 0.5  # Too high, more competition

        # Weighted combination
        opportunity = (
            competition_factor * 0.4 + effectiveness_factor * 0.4 + volume_factor * 0.2
        ) * 100.0

        return opportunity