"""Content calendar generation."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import logging

from bufferiq.ml.gaps.recommendations.generator import ContentRecommendation
from bufferiq.ml.gaps.calendar.optimizer import CalendarOptimizer
from bufferiq.ml.gaps.calendar.theme_planner import ThemePlanner

logger = logging.getLogger(__name__)


@dataclass
class ContentCalendar:
    """Generated content calendar."""

    start_date: datetime
    end_date: datetime
    total_pieces: int

    # Scheduled content
    calendar_items: List[Dict[str, Any]] = field(default_factory=list)

    # Balance metrics
    topic_distribution: Dict[str, int] = field(default_factory=dict)
    platform_distribution: Dict[str, int] = field(default_factory=dict)
    format_distribution: Dict[str, int] = field(default_factory=dict)

    # Optimization
    theme_weeks: List[Dict[str, Any]] = field(default_factory=list)
    posting_frequency: float = 3.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_pieces": self.total_pieces,
            "calendar_items": self.calendar_items,
            "topic_distribution": self.topic_distribution,
            "platform_distribution": self.platform_distribution,
            "format_distribution": self.format_distribution,
            "theme_weeks": self.theme_weeks,
            "posting_frequency": self.posting_frequency,
        }


class CalendarGenerator:
    """
    Generate optimized content calendars.

    Creates balanced publishing schedules with thematic organization.
    """

    def __init__(self):
        """Initialize calendar generator."""
        self.optimizer = CalendarOptimizer()
        self.theme_planner = ThemePlanner()

    def generate(
        self,
        recommendations: List[ContentRecommendation],
        weeks: int = 4,
        posts_per_week: int = 3,
        start_date: Optional[datetime] = None,
        platform: str = "linkedin",
    ) -> ContentCalendar:
        """
        Generate content calendar.

        Args:
            recommendations: Content recommendations
            weeks: Number of weeks
            posts_per_week: Target posts per week
            start_date: Calendar start date
            platform: Target platform

        Returns:
            Content calendar
        """
        if start_date is None:
            start_date = datetime.now()

        # Calculate end date
        end_date = start_date + timedelta(weeks=weeks)

        # Total pieces needed
        total_pieces = weeks * posts_per_week

        # Select recommendations
        selected_recs = recommendations[:total_pieces]

        # Generate theme weeks
        theme_weeks = self.theme_planner.plan_themes(selected_recs, weeks)

        # Schedule content
        calendar_items = self._schedule_content(
            selected_recs, start_date, weeks, posts_per_week, platform
        )

        # Optimize schedule
        optimized_items = self.optimizer.optimize(calendar_items, theme_weeks)

        # Calculate distributions
        topic_dist = self._calculate_topic_distribution(optimized_items)
        platform_dist = self._calculate_platform_distribution(optimized_items)
        format_dist = self._calculate_format_distribution(optimized_items)

        return ContentCalendar(
            start_date=start_date,
            end_date=end_date,
            total_pieces=len(optimized_items),
            calendar_items=optimized_items,
            topic_distribution=topic_dist,
            platform_distribution=platform_dist,
            format_distribution=format_dist,
            theme_weeks=theme_weeks,
            posting_frequency=float(posts_per_week),
        )

    def _schedule_content(
        self,
        recommendations: List[ContentRecommendation],
        start_date: datetime,
        weeks: int,
        posts_per_week: int,
        platform: str,
    ) -> List[Dict[str, Any]]:
        """Schedule content across calendar."""
        items = []

        # Posting days (Mon, Wed, Fri for 3/week)
        if posts_per_week == 3:
            posting_days = [0, 2, 4]  # Mon, Wed, Fri
        elif posts_per_week == 5:
            posting_days = [0, 1, 2, 3, 4]  # Weekdays
        else:
            # Distribute evenly
            posting_days = list(range(posts_per_week))

        current_date = start_date
        rec_index = 0

        for week in range(weeks):
            for day in posting_days:
                if rec_index >= len(recommendations):
                    break

                rec = recommendations[rec_index]

                # Calculate posting date
                post_date = current_date + timedelta(days=day)

                # Set optimal time (9 AM)
                post_date = post_date.replace(hour=9, minute=0, second=0)

                items.append({
                    "date": post_date.isoformat(),
                    "topic": rec.topic,
                    "title": rec.title_suggestions[0] if rec.title_suggestions else rec.topic,
                    "format": rec.recommended_format,
                    "platform": platform,
                    "priority": rec.priority_score,
                    "estimated_engagement": rec.estimated_engagement,
                    "week": week + 1,
                })

                rec_index += 1

            # Move to next week
            current_date += timedelta(weeks=1)

        return items

    def _calculate_topic_distribution(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate topic distribution."""
        distribution: Dict[str, int] = {}

        for item in items:
            topic = item["topic"]
            distribution[topic] = distribution.get(topic, 0) + 1

        return distribution

    def _calculate_platform_distribution(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate platform distribution."""
        distribution: Dict[str, int] = {}

        for item in items:
            platform = item["platform"]
            distribution[platform] = distribution.get(platform, 0) + 1

        return distribution

    def _calculate_format_distribution(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate format distribution."""
        distribution: Dict[str, int] = {}

        for item in items:
            fmt = item["format"]
            distribution[fmt] = distribution.get(fmt, 0) + 1

        return distribution