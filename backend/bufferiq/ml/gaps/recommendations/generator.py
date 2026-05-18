"""Content recommendation generation."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import logging
import hashlib

from bufferiq.ml.gaps.detection.detector import ContentGap
from bufferiq.ml.gaps.recommendations.title_suggester import TitleSuggester
from bufferiq.ml.gaps.recommendations.formatter import FormatRecommender

logger = logging.getLogger(__name__)


@dataclass
class ContentRecommendation:
    """Content creation recommendation."""

    recommendation_id: str
    topic: str
    title_suggestions: List[str] = field(default_factory=list)

    # Content details
    recommended_format: str = "article"
    suggested_length: str = "medium"
    key_points: List[str] = field(default_factory=list)
    suggested_angles: List[str] = field(default_factory=list)

    # Optimization
    optimal_platform: str = "linkedin"
    optimal_time: Optional[datetime] = None
    target_audience: str = "professionals"

    # Metrics
    priority_score: float = 50.0
    estimated_engagement: float = 100.0
    difficulty: str = "medium"
    time_to_create: str = "moderate"

    # Competitive context
    competitor_coverage: int = 0
    differentiation_angle: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommendation_id": self.recommendation_id,
            "topic": self.topic,
            "title_suggestions": self.title_suggestions,
            "recommended_format": self.recommended_format,
            "suggested_length": self.suggested_length,
            "key_points": self.key_points,
            "suggested_angles": self.suggested_angles,
            "optimal_platform": self.optimal_platform,
            "optimal_time": self.optimal_time.isoformat() if self.optimal_time else None,
            "target_audience": self.target_audience,
            "priority_score": self.priority_score,
            "estimated_engagement": self.estimated_engagement,
            "difficulty": self.difficulty,
            "time_to_create": self.time_to_create,
            "competitor_coverage": self.competitor_coverage,
            "differentiation_angle": self.differentiation_angle,
        }


class ContentRecommendationEngine:
    """
    Generate content recommendations from gap analysis.

    Produces specific, actionable content ideas with
    titles, angles, and optimization suggestions.

    Example:
```python
        engine = ContentRecommendationEngine()
        recommendations = engine.generate(
            gaps=gap_analysis.critical_gaps,
            count=10,
            user_profile=user_voice_profile
        )

        for rec in recommendations[:5]:
            print(f"\\n{rec.topic}")
            print(f"  Priority: {rec.priority_score:.1f}")
            print(f"  Titles:")
            for title in rec.title_suggestions:
                print(f"    - {title}")
            print(f"  Format: {rec.recommended_format}")
            print(f"  Estimated engagement: {rec.estimated_engagement:.0f}")
```
    """

    def __init__(self):
        """Initialize recommendation engine."""
        self.title_suggester = TitleSuggester()
        self.format_recommender = FormatRecommender()

    def generate(
        self,
        gaps: List[ContentGap],
        count: int = 20,
        user_profile: Optional[Any] = None,
        timeframe_days: int = 30,
    ) -> List[ContentRecommendation]:
        """
        Generate content recommendations.

        Args:
            gaps: Identified content gaps
            count: Number of recommendations
            user_profile: Optional user voice/style profile
            timeframe_days: Planning timeframe

        Returns:
            List of recommendations
        """
        recommendations = []

        for gap in gaps[:count]:
            rec = self._create_recommendation(gap, user_profile, timeframe_days)
            recommendations.append(rec)

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return recommendations[:count]

    def _create_recommendation(
        self,
        gap: ContentGap,
        user_profile: Optional[Any],
        timeframe_days: int,
    ) -> ContentRecommendation:
        """Create single recommendation from gap."""
        # Generate titles
        titles = self.title_suggester.generate_titles(
            topic=gap.topic, keywords=gap.keywords, count=5
        )

        # Recommend format
        format_rec = self.format_recommender.recommend(gap.topic, gap.keywords)

        # Generate key points
        key_points = self._generate_key_points(gap)

        # Determine optimal timing
        optimal_time = self._calculate_optimal_time(timeframe_days)

        # Calculate difficulty
        difficulty = self._assess_difficulty(gap)

        # Time to create
        time_to_create = self._estimate_creation_time(format_rec["format"])

        # Differentiation angle
        differentiation = self._generate_differentiation_angle(gap)

        # Generate ID
        rec_id = hashlib.sha256(gap.topic.encode()).hexdigest()[:16]

        return ContentRecommendation(
            recommendation_id=rec_id,
            topic=gap.topic,
            title_suggestions=titles,
            recommended_format=format_rec["format"],
            suggested_length=format_rec["length"],
            key_points=key_points,
            suggested_angles=gap.suggested_angles,
            optimal_platform=self._determine_platform(gap),
            optimal_time=optimal_time,
            target_audience=self._determine_audience(gap),
            priority_score=gap.priority_score,
            estimated_engagement=gap.estimated_engagement or 150.0,
            difficulty=difficulty,
            time_to_create=time_to_create,
            competitor_coverage=gap.competitor_coverage,
            differentiation_angle=differentiation,
        )

    def _generate_key_points(self, gap: ContentGap) -> List[str]:
        """Generate key points to cover."""
        key_points = [
            f"Introduction to {gap.topic}",
            f"Key benefits and applications",
            f"Common challenges and solutions",
            f"Best practices and recommendations",
            f"Future trends and outlook",
        ]

        return key_points[:4]

    def _calculate_optimal_time(self, timeframe_days: int) -> datetime:
        """Calculate optimal publication time."""
        # Distribute evenly over timeframe
        days_offset = timeframe_days // 4

        optimal = datetime.now() + timedelta(days=days_offset)

        # Adjust to weekday morning (9 AM)
        while optimal.weekday() >= 5:  # Weekend
            optimal += timedelta(days=1)

        optimal = optimal.replace(hour=9, minute=0, second=0, microsecond=0)

        return optimal

    def _assess_difficulty(self, gap: ContentGap) -> str:
        """Assess creation difficulty."""
        # Based on topic complexity and competition
        if gap.competitor_coverage >= 5:
            return "hard"
        elif gap.competitor_coverage >= 3:
            return "medium"
        else:
            return "easy"

    def _estimate_creation_time(self, content_format: str) -> str:
        """Estimate time to create content."""
        time_map = {
            "article": "moderate",
            "tutorial": "extensive",
            "listicle": "quick",
            "case_study": "extensive",
            "opinion": "quick",
            "how-to": "moderate",
        }

        return time_map.get(content_format, "moderate")

    def _determine_platform(self, gap: ContentGap) -> str:
        """Determine optimal platform."""
        # Default to LinkedIn for professional content
        return "linkedin"

    def _determine_audience(self, gap: ContentGap) -> str:
        """Determine target audience."""
        # Analyze keywords for audience clues
        if any(kw in ["beginner", "intro", "basics"] for kw in gap.keywords):
            return "beginners"
        elif any(kw in ["advanced", "expert", "deep"] for kw in gap.keywords):
            return "experts"
        else:
            return "professionals"

    def _generate_differentiation_angle(self, gap: ContentGap) -> str:
        """Generate differentiation angle."""
        angles = [
            f"Unique perspective on {gap.topic}",
            f"Practical approach to {gap.topic}",
            f"Data-driven insights on {gap.topic}",
            f"Real-world examples of {gap.topic}",
        ]

        # Pick based on gap characteristics
        if gap.competitor_coverage >= 5:
            return angles[0]  # Need unique perspective
        else:
            return angles[1]  # Practical is always good