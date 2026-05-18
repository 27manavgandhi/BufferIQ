"""Share of voice calculation."""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ShareOfVoiceCalculator:
    """
    Calculate share of voice metrics.

    Measures brand's visibility relative to competitors.
    """

    def calculate(
        self,
        user_volume: int,
        competitor_volumes: List[int],
    ) -> Dict[str, float]:
        """
        Calculate share of voice.

        Args:
            user_volume: User's content volume
            competitor_volumes: Competitor content volumes

        Returns:
            Share of voice metrics
        """
        total_volume = user_volume + sum(competitor_volumes)

        if total_volume == 0:
            return {
                "share_of_voice": 0.0,
                "rank": 1,
                "total_voices": 1,
            }

        # Calculate share
        share = (user_volume / total_volume) * 100

        # Calculate rank
        all_volumes = [user_volume] + competitor_volumes
        all_volumes.sort(reverse=True)
        rank = all_volumes.index(user_volume) + 1

        return {
            "share_of_voice": round(share, 2),
            "rank": rank,
            "total_voices": len(all_volumes),
            "user_volume": user_volume,
            "total_volume": total_volume,
        }

    def calculate_by_topic(
        self,
        user_topics: Dict[str, int],
        competitor_topics: List[Dict[str, int]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate share of voice by topic.

        Args:
            user_topics: User's topic volumes
            competitor_topics: Competitor topic volumes

        Returns:
            Share of voice by topic
        """
        topic_shares = {}

        # Get all unique topics
        all_topics = set(user_topics.keys())
        for comp_topics in competitor_topics:
            all_topics.update(comp_topics.keys())

        for topic in all_topics:
            user_vol = user_topics.get(topic, 0)
            comp_vols = [ct.get(topic, 0) for ct in competitor_topics]

            topic_shares[topic] = self.calculate(user_vol, comp_vols)

        return topic_shares

    def trend_analysis(
        self,
        current_share: float,
        previous_share: float,
    ) -> Dict[str, any]:
        """
        Analyze share of voice trend.

        Args:
            current_share: Current share percentage
            previous_share: Previous period share percentage

        Returns:
            Trend analysis
        """
        change = current_share - previous_share

        if abs(change) < 0.5:
            trend = "stable"
        elif change > 0:
            trend = "growing"
        else:
            trend = "declining"

        return {
            "current_share": current_share,
            "previous_share": previous_share,
            "change": round(change, 2),
            "trend": trend,
        }