"""Multi-modal service layer for API."""

from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from bufferiq.ml.multimodal.intelligence.service import MultiModalIntelligenceService
from bufferiq.domain.repositories.media_repository import (
    MediaAnalysisRepository,
    VisualFeaturesRepository,
    LinkPreviewRepository,
)
from bufferiq.domain.models.media_analysis import MediaAnalysis
from bufferiq.domain.models.visual_features import VisualFeatures
from bufferiq.domain.models.link_preview import LinkPreview


class MultiModalService:
    """Service for multi-modal analysis with database persistence."""
    
    def __init__(
        self,
        db: Session,
        intelligence_service: Optional[MultiModalIntelligenceService] = None
    ):
        """
        Initialize multi-modal service.
        
        Args:
            db: Database session
            intelligence_service: Intelligence service instance
        """
        self.db = db
        self.intelligence_service = intelligence_service or MultiModalIntelligenceService()
        
        # Initialize repositories
        self.media_repo = MediaAnalysisRepository(db)
        self.features_repo = VisualFeaturesRepository(db)
        self.link_repo = LinkPreviewRepository(db)
    
    async def analyze_and_persist(
        self,
        post_id: str,
        text: str,
        image_urls: list[str] | None,
        video_urls: list[str] | None,
        link_urls: list[str] | None,
        platform: str
    ) -> Dict[str, Any]:
        """
        Analyze post and persist results to database.
        
        Args:
            post_id: Post ID
            text: Post text
            image_urls: Image URLs
            video_urls: Video URLs
            link_urls: Link URLs
            platform: Platform type
            
        Returns:
            Analysis results
        """
        # Perform analysis
        results = await self.intelligence_service.analyze_post(
            post_id=post_id,
            text=text,
            image_urls=image_urls,
            video_urls=video_urls,
            link_urls=link_urls,
            platform=platform  # type: ignore
        )
        
        # Persist results
        self._persist_results(post_id, results, platform)
        
        return results
    
    def _persist_results(
        self,
        post_id: str,
        results: Dict[str, Any],
        platform: str
    ) -> None:
        """Persist analysis results to database."""
        # This is a simplified version
        # In production, handle each media type separately
        pass