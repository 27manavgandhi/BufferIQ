"""
Hashtag mixer for optimal combinations.

Creates balanced mixes of broad, niche, and branded hashtags.
"""

from typing import Dict, List


class HashtagMixer:
    """
    Mix hashtags optimally.

    Balances broad (high reach), niche (targeted), and branded hashtags.

    Example:
```python
        mixer = HashtagMixer()
        mix = mixer.create_mix(
            broad=["ai", "tech", "innovation"],
            niche=["aitips", "mlbasics"],
            branded=["mycompany"],
            total_count=5,
            platform="linkedin"
        )

        print(f"Optimal mix: {mix}")
```
    """

    def __init__(self) -> None:
        """Initialize mixer."""
        # Platform-specific mix ratios
        self.mix_ratios = {
            "linkedin": {"broad": 0.4, "niche": 0.4, "branded": 0.2},
            "twitter": {"broad": 0.5, "niche": 0.5, "branded": 0.0},
            "bluesky": {"broad": 0.5, "niche": 0.5, "branded": 0.0},
        }

    def create_mix(
        self,
        broad: List[str],
        niche: List[str],
        branded: List[str],
        total_count: int,
        platform: str,
    ) -> List[str]:
        """
        Create optimal hashtag mix.

        Args:
            broad: Broad hashtags
            niche: Niche hashtags
            branded: Branded hashtags
            total_count: Total hashtags to include
            platform: Platform name

        Returns:
            Mixed list of hashtags
        """
        ratios = self.mix_ratios.get(
            platform, {"broad": 0.5, "niche": 0.5, "branded": 0.0}
        )

        # Calculate counts
        broad_count = int(total_count * ratios["broad"])
        niche_count = int(total_count * ratios["niche"])
        branded_count = int(total_count * ratios["branded"])

        # Adjust to hit total
        remaining = total_count - (broad_count + niche_count + branded_count)
        if remaining > 0:
            broad_count += remaining

        # Build mix
        mix: List[str] = []
        mix.extend(broad[:broad_count])
        mix.extend(niche[:niche_count])
        mix.extend(branded[:branded_count])

        return mix[:total_count]

    def calculate_diversity_score(self, hashtags: List[str]) -> float:
        """
        Calculate diversity score of hashtag mix (0-100).

        Higher score = more diverse mix.

        Args:
            hashtags: List of hashtags

        Returns:
            Diversity score
        """
        if not hashtags:
            return 0.0

        # Calculate uniqueness
        unique_chars = set("".join(hashtags))
        avg_length = sum(len(h) for h in hashtags) / len(hashtags)

        # More unique characters and varied lengths = more diverse
        char_diversity = len(unique_chars) / (avg_length * len(hashtags))

        # Clamp and scale to 0-100
        diversity = min(1.0, char_diversity) * 100

        return diversity