"""Repository for media analysis operations."""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from bufferiq.domain.models.media_analysis import MediaAnalysis
from bufferiq.domain.models.visual_features import VisualFeatures
from bufferiq.domain.models.link_preview import LinkPreview


class MediaAnalysisRepository:
    """Repository for media analysis database operations."""
    
    def __init__(self, db: Session):
        """
        Initialize repository.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def create(self, media_analysis: MediaAnalysis) -> MediaAnalysis:
        """
        Create new media analysis record.
        
        Args:
            media_analysis: MediaAnalysis instance
            
        Returns:
            Created media analysis
        """
        self.db.add(media_analysis)
        self.db.commit()
        self.db.refresh(media_analysis)
        return media_analysis
    
    def get_by_id(self, analysis_id: str) -> Optional[MediaAnalysis]:
        """
        Get media analysis by ID.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            MediaAnalysis or None
        """
        return self.db.query(MediaAnalysis).filter(
            MediaAnalysis.id == analysis_id
        ).first()
    
    def get_by_post_id(self, post_id: str) -> List[MediaAnalysis]:
        """
        Get all media analyses for a post.
        
        Args:
            post_id: Post ID
            
        Returns:
            List of media analyses
        """
        return self.db.query(MediaAnalysis).filter(
            MediaAnalysis.post_id == post_id
        ).order_by(desc(MediaAnalysis.created_at)).all()
    
    def get_by_platform(
        self,
        platform: str,
        limit: int = 100
    ) -> List[MediaAnalysis]:
        """
        Get media analyses by platform.
        
        Args:
            platform: Platform type
            limit: Maximum number of results
            
        Returns:
            List of media analyses
        """
        return self.db.query(MediaAnalysis).filter(
            MediaAnalysis.platform == platform
        ).order_by(desc(MediaAnalysis.created_at)).limit(limit).all()
    
    def update(self, media_analysis: MediaAnalysis) -> MediaAnalysis:
        """
        Update media analysis.
        
        Args:
            media_analysis: MediaAnalysis instance
            
        Returns:
            Updated media analysis
        """
        self.db.commit()
        self.db.refresh(media_analysis)
        return media_analysis
    
    def delete(self, analysis_id: str) -> bool:
        """
        Delete media analysis.
        
        Args:
            analysis_id: Analysis ID
            
        Returns:
            True if deleted, False otherwise
        """
        analysis = self.get_by_id(analysis_id)
        if analysis:
            self.db.delete(analysis)
            self.db.commit()
            return True
        return False


class VisualFeaturesRepository:
    """Repository for visual features database operations."""
    
    def __init__(self, db: Session):
        """
        Initialize repository.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def create(self, visual_features: VisualFeatures) -> VisualFeatures:
        """
        Create visual features record.
        
        Args:
            visual_features: VisualFeatures instance
            
        Returns:
            Created visual features
        """
        self.db.add(visual_features)
        self.db.commit()
        self.db.refresh(visual_features)
        return visual_features
    
    def get_by_media_analysis_id(
        self,
        media_analysis_id: str
    ) -> Optional[VisualFeatures]:
        """
        Get visual features by media analysis ID.
        
        Args:
            media_analysis_id: Media analysis ID
            
        Returns:
            VisualFeatures or None
        """
        return self.db.query(VisualFeatures).filter(
            VisualFeatures.media_analysis_id == media_analysis_id
        ).first()


class LinkPreviewRepository:
    """Repository for link preview database operations."""
    
    def __init__(self, db: Session):
        """
        Initialize repository.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def create(self, link_preview: LinkPreview) -> LinkPreview:
        """
        Create link preview record.
        
        Args:
            link_preview: LinkPreview instance
            
        Returns:
            Created link preview
        """
        self.db.add(link_preview)
        self.db.commit()
        self.db.refresh(link_preview)
        return link_preview
    
    def get_by_media_analysis_id(
        self,
        media_analysis_id: str
    ) -> Optional[LinkPreview]:
        """
        Get link preview by media analysis ID.
        
        Args:
            media_analysis_id: Media analysis ID
            
        Returns:
            LinkPreview or None
        """
        return self.db.query(LinkPreview).filter(
            LinkPreview.media_analysis_id == media_analysis_id
        ).first()
    
    def get_by_url(self, url: str) -> Optional[LinkPreview]:
        """
        Get link preview by URL.
        
        Args:
            url: Link URL
            
        Returns:
            LinkPreview or None
        """
        return self.db.query(LinkPreview).filter(
            LinkPreview.url == url
        ).order_by(desc(LinkPreview.created_at)).first()