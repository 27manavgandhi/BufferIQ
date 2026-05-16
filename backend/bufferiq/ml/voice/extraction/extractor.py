"""
Voice feature extraction from historical content.

Analyzes past posts to build comprehensive voice profile
with linguistic, syntactic, and stylistic characteristics.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

from sqlalchemy.orm import Session

from bufferiq.ml.voice.linguistic.lexical_analyzer import LexicalAnalyzer, LexicalMetrics
from bufferiq.ml.voice.linguistic.syntactic_analyzer import (
    SyntacticAnalyzer,
    SyntacticMetrics,
)
from bufferiq.ml.voice.stylistic.style_detector import (
    StyleDetector,
    StylisticFeatures,
)

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class VoiceFeatures:
    """Extracted voice features from content corpus."""
    
    lexical_profile: LexicalMetrics
    syntactic_profile: SyntacticMetrics
    stylistic_profile: StylisticFeatures
    temporal_evolution: Dict[str, any]
    platform_variations: Dict[str, Dict]
    confidence_score: float
    sample_size: int
    extraction_date: datetime


class VoiceExtractor:
    """
    Extract brand voice from historical content.
    
    Analyzes past posts to build comprehensive voice profile
    with linguistic, syntactic, and stylistic characteristics.
    
    Example:
```python
        extractor = VoiceExtractor(db_session)
        voice = await extractor.extract(
            user_id="user123",
            platform="linkedin",
            lookback_days=90
        )
        print(f"Voice extracted from {voice.sample_size} posts")
        print(f"Confidence: {voice.confidence_score:.2f}")
```
    """
    
    def __init__(
        self,
        db_session: Session,
        lexical_analyzer: Optional[LexicalAnalyzer] = None,
        syntactic_analyzer: Optional[SyntacticAnalyzer] = None,
        style_detector: Optional[StyleDetector] = None,
    ):
        """Initialize voice extractor."""
        self.db = db_session
        self.lexical_analyzer = lexical_analyzer or LexicalAnalyzer()
        self.syntactic_analyzer = syntactic_analyzer or SyntacticAnalyzer()
        self.style_detector = style_detector or StyleDetector()
    
    async def extract(
        self,
        user_id: str,
        platform: str,
        lookback_days: int = 90,
        min_posts: int = 20,
    ) -> VoiceFeatures:
        """
        Extract voice features from historical posts.
        
        Args:
            user_id: User identifier
            platform: Platform to analyze
            lookback_days: Days of history to analyze
            min_posts: Minimum posts required
        
        Returns:
            Extracted voice features
        
        Raises:
            ValueError: If platform not supported or insufficient posts
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        # Get historical posts (mock for now - would query database)
        posts = self._fetch_historical_posts(user_id, platform, lookback_days)
        
        if len(posts) < min_posts:
            raise ValueError(
                f"Insufficient posts for voice extraction. "
                f"Found {len(posts)}, minimum required: {min_posts}"
            )
        
        logger.info(
            f"Extracting voice from {len(posts)} posts "
            f"for user {user_id} on {platform}"
        )
        
        # Combine all text
        combined_text = " ".join(post["text"] for post in posts)
        
        # Extract features
        lexical = self.lexical_analyzer.analyze(combined_text)
        syntactic = self.syntactic_analyzer.analyze(combined_text)
        stylistic = self.style_detector.detect(combined_text)
        
        # Analyze temporal evolution
        temporal = self._analyze_temporal_evolution(posts)
        
        # Analyze platform variations
        platform_vars = self._analyze_platform_variations(posts)
        
        # Calculate confidence
        confidence = self._calculate_confidence(len(posts), min_posts)
        
        return VoiceFeatures(
            lexical_profile=lexical,
            syntactic_profile=syntactic,
            stylistic_profile=stylistic,
            temporal_evolution=temporal,
            platform_variations=platform_vars,
            confidence_score=confidence,
            sample_size=len(posts),
            extraction_date=datetime.utcnow(),
        )
    
    def _fetch_historical_posts(
        self, user_id: str, platform: str, lookback_days: int
    ) -> List[Dict]:
        """
        Fetch historical posts from database.
        
        Args:
            user_id: User ID
            platform: Platform
            lookback_days: Days to look back
        
        Returns:
            List of post dictionaries
        """
        # Mock implementation - would query actual database
        # In production, this would be:
        # cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        # query = self.db.query(Post).filter(
        #     Post.user_id == user_id,
        #     Post.platform == platform,
        #     Post.created_at >= cutoff_date
        # ).all()
        
        # Return mock data for now
        return [
            {
                "text": "Excited to announce our new product launch! 🚀",
                "created_at": datetime.utcnow() - timedelta(days=i),
                "platform": platform,
            }
            for i in range(30)
        ]
    
    def _analyze_temporal_evolution(self, posts: List[Dict]) -> Dict[str, any]:
        """
        Analyze how voice evolves over time.
        
        Args:
            posts: List of posts
        
        Returns:
            Temporal evolution metrics
        """
        # Sort by date
        sorted_posts = sorted(posts, key=lambda p: p["created_at"])
        
        # Split into early and recent
        mid_point = len(sorted_posts) // 2
        early_posts = sorted_posts[:mid_point]
        recent_posts = sorted_posts[mid_point:]
        
        # Analyze each period
        early_text = " ".join(p["text"] for p in early_posts)
        recent_text = " ".join(p["text"] for p in recent_posts)
        
        early_style = self.style_detector.detect(early_text)
        recent_style = self.style_detector.detect(recent_text)
        
        # Calculate drift
        formality_drift = recent_style.formality_score - early_style.formality_score
        
        return {
            "early_formality": early_style.formality_score,
            "recent_formality": recent_style.formality_score,
            "formality_drift": formality_drift,
            "has_evolved": abs(formality_drift) > 10,
        }
    
    def _analyze_platform_variations(self, posts: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze voice variations across platforms.
        
        Args:
            posts: List of posts
        
        Returns:
            Platform variation metrics
        """
        # Group by platform
        by_platform: Dict[str, List[str]] = {}
        for post in posts:
            platform = post.get("platform", "unknown")
            if platform not in by_platform:
                by_platform[platform] = []
            by_platform[platform].append(post["text"])
        
        # Analyze each platform
        variations = {}
        for platform, texts in by_platform.items():
            if len(texts) >= 5:  # Minimum for analysis
                combined = " ".join(texts)
                style = self.style_detector.detect(combined)
                variations[platform] = {
                    "formality": style.formality_score,
                    "style": style.style.value,
                    "sample_size": len(texts),
                }
        
        return variations
    
    def _calculate_confidence(self, sample_size: int, min_posts: int) -> float:
        """
        Calculate confidence score based on sample size.
        
        Args:
            sample_size: Number of posts analyzed
            min_posts: Minimum required
        
        Returns:
            Confidence score (0-1)
        """
        if sample_size < min_posts:
            return 0.0
        
        # Logarithmic scale: confidence increases with sample size
        # but with diminishing returns
        if sample_size >= 100:
            return 0.95
        elif sample_size >= 50:
            return 0.85
        elif sample_size >= 30:
            return 0.75
        else:
            return 0.6