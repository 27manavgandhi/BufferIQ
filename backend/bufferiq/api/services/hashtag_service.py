"""
Hashtag service layer.

Business logic for hashtag operations.
"""

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from bufferiq.ml.hashtags.intelligence.service import HashtagIntelligenceService


class HashtagService:
    """
    Service layer for hashtag operations.

    Provides business logic and orchestration for hashtag features.
    """

    def __init__(
        self,
        db_session: Session,
        intelligence_service: HashtagIntelligenceService,
    ) -> None:
        """
        Initialize hashtag service.

        Args:
            db_session: Database session
            intelligence_service: Hashtag intelligence service
        """
        self.db = db_session
        self.intelligence = intelligence_service

    async def analyze_with_cache(
        self,
        hashtag: str,
        platform: str,
        user_id: Optional[str] = None,
        cache_ttl: int = 3600,
    ) -> Dict[str, Any]:
        """
        Analyze hashtag with caching.

        Args:
            hashtag: Hashtag to analyze
            platform: Platform name
            user_id: Optional user context
            cache_ttl: Cache TTL in seconds

        Returns:
            Analysis results
        """
        # Check cache (if available)
        cache_key = f"hashtag:analysis:{platform}:{hashtag}"

        # For now, skip cache and call directly
        # In production, implement Redis caching

        analysis = await self.intelligence.analyze_hashtag(
            hashtag=hashtag,
            platform=platform,
            user_id=user_id,
        )

        return analysis

    async def batch_analyze(
        self,
        hashtags: List[str],
        platform: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple hashtags.

        Args:
            hashtags: List of hashtags
            platform: Platform name
            user_id: Optional user context

        Returns:
            List of analysis results
        """
        results = []

        for hashtag in hashtags:
            try:
                analysis = await self.intelligence.analyze_hashtag(
                    hashtag=hashtag,
                    platform=platform,
                    user_id=user_id,
                )
                results.append(analysis)
            except Exception as e:
                # Log error but continue
                results.append(
                    {
                        "hashtag": hashtag,
                        "error": str(e),
                        "success": False,
                    }
                )

        return results

    async def get_personalized_recommendations(
        self,
        content: str,
        platform: str,
        user_id: str,
        count: int = 5,
    ) -> List[str]:
        """
        Get personalized hashtag recommendations.

        Args:
            content: Content text
            platform: Platform name
            user_id: User identifier
            count: Number of recommendations

        Returns:
            List of recommended hashtags
        """
        # Get base recommendations
        recommendations = await self.intelligence.recommend_hashtags(
            content=content,
            platform=platform,
            user_id=user_id,
            count=count * 2,  # Get extra for filtering
        )

        # Validate safety
        validation = await self.intelligence.validate_hashtags(
            hashtags=recommendations,
            platform=platform,
        )

        # Filter out unsafe hashtags
        safe_recommendations = [
            hashtag
            for hashtag, risk in validation.items()
            if risk.risk_level in ["none", "low"]
        ]

        return safe_recommendations[:count]