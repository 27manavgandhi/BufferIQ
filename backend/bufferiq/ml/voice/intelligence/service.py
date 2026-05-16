"""
Voice intelligence orchestrator service.

Main service that coordinates all voice analysis modules.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from sqlalchemy.orm import Session

from bufferiq.ml.voice.extraction.extractor import VoiceExtractor
from bufferiq.ml.voice.profiler.builder import VoiceProfileBuilder, VoiceProfile
from bufferiq.ml.voice.consistency.scorer import VoiceConsistencyScorer
from bufferiq.ml.voice.drift.detector import VoiceDriftDetector
from bufferiq.ml.voice.recommendations.generator import VoiceRecommendationEngine
from bufferiq.ml.voice.validation.validator import VoiceValidator

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class VoiceIntelligenceService:
    """
    Main orchestrator for voice intelligence.
    
    Coordinates all voice analysis modules:
    - Voice extraction
    - Profile building
    - Consistency scoring
    - Drift detection
    - Recommendations
    - Multi-brand management
    
    Example:
```python
        service = VoiceIntelligenceService(
            db_session=session,
            cache=redis_client
        )
        
        # Extract and build profile
        profile = await service.build_voice_profile(
            brand_id="brand123",
            platform="linkedin"
        )
        
        # Analyze new content
        analysis = await service.analyze_content(
            text="New post content",
            brand_id="brand123",
            platform="linkedin"
        )
        
        print(f"Consistency: {analysis['consistency_score']}")
        print(f"Recommendations: {len(analysis['recommendations'])}")
```
    """
    
    def __init__(
        self,
        db_session: Session,
        cache: Optional[Any] = None,
        extractor: Optional[VoiceExtractor] = None,
        profiler: Optional[VoiceProfileBuilder] = None,
        scorer: Optional[VoiceConsistencyScorer] = None,
        drift_detector: Optional[VoiceDriftDetector] = None,
        recommender: Optional[VoiceRecommendationEngine] = None,
        validator: Optional[VoiceValidator] = None,
    ):
        """
        Initialize voice intelligence service.
        
        Args:
            db_session: Database session
            cache: Optional cache client (e.g., Redis)
            extractor: Optional voice extractor
            profiler: Optional profile builder
            scorer: Optional consistency scorer
            drift_detector: Optional drift detector
            recommender: Optional recommendation engine
            validator: Optional validator
        """
        self.db = db_session
        self.cache = cache
        
        # Initialize components
        self.extractor = extractor or VoiceExtractor(db_session)
        self.profiler = profiler or VoiceProfileBuilder()
        self.scorer = scorer or VoiceConsistencyScorer()
        self.drift_detector = drift_detector or VoiceDriftDetector(db_session)
        self.recommender = recommender or VoiceRecommendationEngine()
        self.validator = validator or VoiceValidator()
        
        # Profile cache
        self._profile_cache: Dict[str, VoiceProfile] = {}
        
        logger.info("Voice intelligence service initialized")
    
    async def build_voice_profile(
        self,
        brand_id: str,
        platform: str,
        lookback_days: int = 90,
        force_rebuild: bool = False,
    ) -> VoiceProfile:
        """
        Extract and build comprehensive voice profile.
        
        Args:
            brand_id: Brand identifier
            platform: Primary platform
            lookback_days: Days of history to analyze
            force_rebuild: Force rebuild even if cached
        
        Returns:
            Voice profile
        
        Raises:
            ValueError: If platform not supported or insufficient data
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        # Check cache
        cache_key = f"{brand_id}_{platform}"
        if not force_rebuild and cache_key in self._profile_cache:
            logger.info(f"Returning cached profile for {cache_key}")
            return self._profile_cache[cache_key]
        
        logger.info(f"Building voice profile for {brand_id} on {platform}")
        
        # Extract voice features
        voice_features = await self.extractor.extract(
            user_id=brand_id,
            platform=platform,
            lookback_days=lookback_days,
        )
        
        # Build profile
        profile = self.profiler.build(
            brand_id=brand_id,
            voice_features=voice_features,
            platform=platform,
        )
        
        # Cache profile
        self._profile_cache[cache_key] = profile
        
        logger.info(
            f"Profile built: {profile.profile_id}, "
            f"confidence: {profile.confidence:.2f}, "
            f"sample_size: {profile.sample_size}"
        )
        
        return profile
    
    async def analyze_content(
        self,
        text: str,
        brand_id: str,
        platform: str,
        return_recommendations: bool = True,
        return_validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Comprehensive voice analysis of content.
        
        Args:
            text: Content to analyze
            brand_id: Brand identifier
            platform: Target platform
            return_recommendations: Include recommendations
            return_validation: Include validation result
        
        Returns:
            Complete analysis results
        
        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        logger.info(f"Analyzing content for {brand_id} on {platform}")
        
        # Get or build profile
        profile = await self.build_voice_profile(brand_id, platform)
        
        # Score consistency
        consistency = self.scorer.score(text, profile, platform)
        
        # Build response
        analysis = {
            "text": text,
            "brand_id": brand_id,
            "platform": platform,
            "profile_id": profile.profile_id,
            "consistency_score": {
                "overall": consistency.overall_score,
                "lexical": consistency.lexical_consistency,
                "syntactic": consistency.syntactic_consistency,
                "stylistic": consistency.stylistic_consistency,
                "is_consistent": consistency.is_consistent,
                "severity": consistency.severity,
            },
            "metrics": {
                "cosine_similarity": consistency.cosine_similarity,
                "kl_divergence": consistency.kl_divergence,
            },
            "analyzed_at": datetime.utcnow().isoformat(),
        }
        
        # Add recommendations if requested
        if return_recommendations:
            recommendations = self.recommender.generate_recommendations(
                consistency, profile, platform
            )
            analysis["recommendations"] = [
                {
                    "type": r.type,
                    "priority": r.priority,
                    "current_value": r.current_value,
                    "suggested_value": r.suggested_value,
                    "reason": r.reason,
                    "impact_score": r.impact_score,
                    "examples": r.examples,
                }
                for r in recommendations
            ]
        
        # Add validation if requested
        if return_validation:
            validation = self.validator.validate(text, profile, platform)
            analysis["validation"] = {
                "passed": validation.passed,
                "score": validation.score,
                "threshold": validation.threshold,
                "issues": validation.issues,
                "warnings": validation.warnings,
                "suggestions": validation.suggestions,
            }
        
        logger.info(
            f"Analysis complete: consistency={consistency.overall_score:.1f}, "
            f"is_consistent={consistency.is_consistent}"
        )
        
        return analysis
    
    async def detect_drift(
        self,
        brand_id: str,
        platform: str,
        window_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Check for voice drift.
        
        Args:
            brand_id: Brand identifier
            platform: Platform to check
            window_days: Recent window size
        
        Returns:
            Drift analysis results
        
        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        logger.info(f"Detecting drift for {brand_id} on {platform}")
        
        # Detect drift
        drift_alert = await self.drift_detector.detect(
            brand_id=brand_id,
            platform=platform,
            window_days=window_days,
        )
        
        # Convert to dict
        result = {
            "brand_id": brand_id,
            "platform": platform,
            "drift_detected": drift_alert.drift_detected,
            "drift_score": drift_alert.drift_score,
            "drift_type": drift_alert.drift_type,
            "affected_dimensions": drift_alert.affected_dimensions,
            "severity": drift_alert.severity,
            "statistical_tests": {
                "t_statistic": drift_alert.t_statistic,
                "p_value": drift_alert.p_value,
                "confidence": drift_alert.confidence,
            },
            "likely_causes": drift_alert.likely_causes,
            "example_deviations": drift_alert.example_deviations,
            "checked_at": datetime.utcnow().isoformat(),
        }
        
        logger.info(
            f"Drift detection complete: detected={drift_alert.drift_detected}, "
            f"score={drift_alert.drift_score:.1f}, severity={drift_alert.severity}"
        )
        
        return result
    
    async def analyze_batch(
        self,
        contents: List[str],
        brand_id: str,
        platform: str,
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple pieces of content.
        
        Args:
            contents: List of content to analyze
            brand_id: Brand identifier
            platform: Target platform
        
        Returns:
            List of analysis results
        """
        logger.info(f"Analyzing batch of {len(contents)} items")
        
        results = []
        for content in contents:
            try:
                analysis = await self.analyze_content(
                    text=content,
                    brand_id=brand_id,
                    platform=platform,
                    return_recommendations=True,
                )
                results.append(analysis)
            except Exception as e:
                logger.error(f"Batch analysis error: {e}")
                results.append({
                    "text": content,
                    "error": str(e),
                    "analyzed_at": datetime.utcnow().isoformat(),
                })
        
        return results
    
    def get_cached_profile(self, brand_id: str, platform: str) -> Optional[VoiceProfile]:
        """
        Get cached profile if available.
        
        Args:
            brand_id: Brand identifier
            platform: Platform
        
        Returns:
            Cached profile or None
        """
        cache_key = f"{brand_id}_{platform}"
        return self._profile_cache.get(cache_key)
    
    def clear_cache(self, brand_id: Optional[str] = None) -> None:
        """
        Clear profile cache.
        
        Args:
            brand_id: Optional brand to clear (clears all if None)
        """
        if brand_id is None:
            self._profile_cache.clear()
            logger.info("Cleared entire profile cache")
        else:
            keys_to_remove = [k for k in self._profile_cache.keys() if k.startswith(brand_id)]
            for key in keys_to_remove:
                del self._profile_cache[key]
            logger.info(f"Cleared cache for brand {brand_id}")