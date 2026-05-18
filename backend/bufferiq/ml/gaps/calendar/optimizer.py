"""Calendar optimization."""

from typing import Any, Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CalendarOptimizer:
    """
    Optimize content calendar for balance and engagement.

    Adjusts scheduling for better distribution and timing.
    """

    def optimize(
        self,
        calendar_items: List[Dict[str, Any]],
        theme_weeks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Optimize calendar schedule.

        Args:
            calendar_items: Calendar items to optimize
            theme_weeks: Theme week definitions

        Returns:
            Optimized calendar items
        """
        # Group by week
        weeks: Dict[int, List[Dict[str, Any]]] = {}
        for item in calendar_items:
            week = item.get("week", 1)
            if week not in weeks:
                weeks[week] = []
            weeks[week].append(item)

        # Optimize each week
        optimized = []
        for week_num, items in weeks.items():
            # Find theme for this week
            week_theme = next(
                (t for t in theme_weeks if t.get("week") == week_num),
                None
            )

            # Apply theme if exists
            if week_theme:
                for item in items:
                    item["theme"] = week_theme.get("theme", "General")

            # Balance formats within week
            items = self._balance_formats(items)

            # Balance priorities
            items = self._balance_priorities(items)

            optimized.extend(items)

        # Sort by date
        optimized.sort(key=lambda x: x["date"])

        return optimized

    def _balance_formats(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Balance content formats within week."""
        # Ensure variety - no more than 2 of same format per week
        from collections import Counter

        formats = [item["format"] for item in items]
        format_counts = Counter(formats)

        # If imbalanced, swap some items
        for fmt, count in format_counts.items():
            if count > 2:
                # Find items with this format
                fmt_items = [i for i in items if i["format"] == fmt]

                # Keep top 2 by priority
                fmt_items.sort(key=lambda x: x.get("priority", 0), reverse=True)

                # Change format of lower priority items
                for item in fmt_items[2:]:
                    # Cycle to different format
                    formats_list = ["article", "tutorial", "listicle", "case-study"]
                    current_idx = formats_list.index(fmt) if fmt in formats_list else 0
                    item["format"] = formats_list[(current_idx + 1) % len(formats_list)]

        return items

    def _balance_priorities(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Balance priority distribution."""
        # Ensure high priority items are spread out
        items.sort(key=lambda x: x.get("priority", 0), reverse=True)

        # Reassign dates to spread high priority
        dates = [item["date"] for item in items]
        dates.sort()

        for i, item in enumerate(items):
            item["date"] = dates[i]

        return items

    def calculate_balance_score(
        self, calendar_items: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate calendar balance score (0-100).

        Args:
            calendar_items: Calendar items

        Returns:
            Balance score
        """
        if not calendar_items:
            return 0.0

        # Check format diversity
        formats = [item["format"] for item in calendar_items]
        format_diversity = len(set(formats)) / len(formats)

        # Check topic diversity
        topics = [item["topic"] for item in calendar_items]
        topic_diversity = len(set(topics)) / len(topics)

        # Check temporal spread
        dates = [datetime.fromisoformat(item["date"]) for item in calendar_items]
        dates.sort()

        # Calculate gaps between posts
        gaps = []
        for i in range(len(dates) - 1):
            gap = (dates[i + 1] - dates[i]).days
            gaps.append(gap)

        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            gap_std = (sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5
            temporal_balance = 1 / (1 + gap_std / avg_gap) if avg_gap > 0 else 0
        else:
            temporal_balance = 1.0

        # Combined score
        balance_score = (
            (format_diversity * 0.3) +
            (topic_diversity * 0.3) +
            (temporal_balance * 0.4)
        ) * 100

        return round(balance_score, 2)