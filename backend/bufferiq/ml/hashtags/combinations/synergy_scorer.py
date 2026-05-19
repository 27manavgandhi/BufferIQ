"""
Synergy scorer for hashtag pairs.

Scores how well hashtags work together.
"""

from typing import Dict, List, Tuple


class SynergyScorer:
    """
    Score synergy between hashtag pairs.

    Example:
```python
        scorer = SynergyScorer()

        synergy = scorer.calculate_synergy(
            hashtag1="ai",
            hashtag2="machinelearning",
            cooccurrence_count=50,
            individual_effectiveness1=85.0,
            individual_effectiveness2=80.0
        )

        print(f"Synergy score: {synergy:.1f}")
```
    """

    def calculate_synergy(
        self,
        hashtag1: str,
        hashtag2: str,
        cooccurrence_count: int,
        individual_effectiveness1: float,
        individual_effectiveness2: float,
    ) -> float:
        """
        Calculate synergy score (0-100).

        Args:
            hashtag1: First hashtag
            hashtag2: Second hashtag
            cooccurrence_count: Times used together
            individual_effectiveness1: Effectiveness of first
            individual_effectiveness2: Effectiveness of second

        Returns:
            Synergy score
        """
        # Base synergy from co-occurrence
        # Assume max co-occurrence of 100
        cooccurrence_factor = min(1.0, cooccurrence_count / 100.0)

        # Effectiveness factor (both should be effective)
        avg_effectiveness = (individual_effectiveness1 + individual_effectiveness2) / 200.0

        # Complementarity (different but related)
        complementarity = self._calculate_complementarity(hashtag1, hashtag2)

        # Weighted combination
        synergy = (
            cooccurrence_factor * 0.4 + avg_effectiveness * 0.4 + complementarity * 0.2
        ) * 100

        return synergy

    def _calculate_complementarity(self, hashtag1: str, hashtag2: str) -> float:
        """
        Calculate how complementary two hashtags are.

        Different but related = high complementarity.
        Too similar or unrelated = low complementarity.

        Args:
            hashtag1: First hashtag
            hashtag2: Second hashtag

        Returns:
            Complementarity score (0-1)
        """
        # Character overlap
        chars1 = set(hashtag1)
        chars2 = set(hashtag2)

        overlap = len(chars1 & chars2) / len(chars1 | chars2) if chars1 | chars2 else 0

        # Complementary if 20-60% overlap (related but different)
        if 0.2 <= overlap <= 0.6:
            complementarity = 1.0
        elif overlap < 0.2:
            complementarity = 0.5  # Too different
        else:
            complementarity = 0.3  # Too similar

        return complementarity

    def score_set(
        self,
        hashtags: List[str],
        effectiveness_scores: Dict[str, float],
    ) -> float:
        """
        Score synergy of entire hashtag set.

        Args:
            hashtags: List of hashtags
            effectiveness_scores: Effectiveness for each

        Returns:
            Overall synergy score (0-100)
        """
        if len(hashtags) < 2:
            return 50.0  # Neutral for single hashtag

        total_synergy = 0.0
        pair_count = 0

        # Calculate pairwise synergy
        for i, ht1 in enumerate(hashtags):
            for ht2 in hashtags[i + 1 :]:
                synergy = self.calculate_synergy(
                    hashtag1=ht1,
                    hashtag2=ht2,
                    cooccurrence_count=30,  # Mock
                    individual_effectiveness1=effectiveness_scores.get(ht1, 50.0),
                    individual_effectiveness2=effectiveness_scores.get(ht2, 50.0),
                )
                total_synergy += synergy
                pair_count += 1

        avg_synergy = total_synergy / pair_count if pair_count > 0 else 50.0

        return avg_synergy