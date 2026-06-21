"""Multi-modal feature extraction."""

from typing import Dict, List, Optional, Any
import numpy as np

from bufferiq.ml.multimodal.types import (
    ImageAnalysisResult,
    VideoAnalysisResult,
    LinkPreviewAnalysis,
    PlatformType,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.multimodal.exceptions import (
    FeatureExtractionError,
    UnsupportedPlatformError,
)


class MultiModalFeatureExtractor:
    """Extract and combine features from multiple modalities."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names: List[str] = []
    
    def extract_features(
        self,
        text_features: Optional[Dict[str, Any]] = None,
        image_features: Optional[ImageAnalysisResult] = None,
        video_features: Optional[VideoAnalysisResult] = None,
        link_features: Optional[LinkPreviewAnalysis] = None,
        platform: PlatformType = "linkedin"
    ) -> np.ndarray:
        """
        Extract and fuse multi-modal features.
        
        Args:
            text_features: Text analysis features
            image_features: Image analysis features
            video_features: Video analysis features
            link_features: Link preview features
            platform: Platform type
            
        Returns:
            Fused feature vector
            
        Raises:
            UnsupportedPlatformError: If platform not supported
            FeatureExtractionError: If extraction fails
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)
        
        try:
            feature_vectors = []
            self.feature_names = []
            
            # Text features
            if text_features:
                text_vec = self._extract_text_features(text_features)
                feature_vectors.append(text_vec)
            
            # Image features
            if image_features:
                image_vec = self._extract_image_features(image_features)
                feature_vectors.append(image_vec)
            
            # Video features
            if video_features:
                video_vec = self._extract_video_features(video_features)
                feature_vectors.append(video_vec)
            
            # Link features
            if link_features:
                link_vec = self._extract_link_features(link_features)
                feature_vectors.append(link_vec)
            
            # Platform features
            platform_vec = self._extract_platform_features(platform)
            feature_vectors.append(platform_vec)
            
            if not feature_vectors:
                raise FeatureExtractionError("At least one modality required")
            
            # Concatenate all features
            fused = np.concatenate(feature_vectors)
            
            return fused
            
        except Exception as e:
            if isinstance(e, (UnsupportedPlatformError, FeatureExtractionError)):
                raise
            raise FeatureExtractionError(f"Feature extraction failed: {str(e)}")
    
    def _extract_text_features(self, text_features: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from text analysis."""
        features = []
        names = []
        
        # Text length
        text = text_features.get("text", "")
        features.append(len(text))
        names.append("text_length")
        
        # Word count
        word_count = len(text.split())
        features.append(word_count)
        names.append("word_count")
        
        # Has hashtags
        has_hashtags = 1.0 if "#" in text else 0.0
        features.append(has_hashtags)
        names.append("has_hashtags")
        
        # Has mentions
        has_mentions = 1.0 if "@" in text else 0.0
        features.append(has_mentions)
        names.append("has_mentions")
        
        # Has URL
        has_url = 1.0 if "http" in text.lower() else 0.0
        features.append(has_url)
        names.append("has_url")
        
        self.feature_names.extend(names)
        return np.array(features, dtype=np.float32)
    
    def _extract_image_features(self, image: ImageAnalysisResult) -> np.ndarray:
        """Extract numerical features from image analysis."""
        features = []
        names = []
        
        # Object counts
        features.append(len(image.objects))
        names.append("num_objects")
        
        # Text extracted
        features.append(len(image.text))
        names.append("num_text_elements")
        
        # Faces detected
        features.append(len(image.faces))
        names.append("num_faces")
        
        # Color diversity
        features.append(len(image.colors.dominant_colors))
        names.append("color_diversity")
        
        # Composition scores
        features.append(image.composition.rule_of_thirds)
        names.append("composition_thirds")
        
        features.append(image.composition.golden_ratio)
        names.append("composition_golden")
        
        features.append(image.composition.symmetry)
        names.append("composition_symmetry")
        
        features.append(image.composition.balance)
        names.append("composition_balance")
        
        # Aesthetic score (normalized)
        features.append(image.aesthetic_score / 100.0)
        names.append("aesthetic_score")
        
        # Brand presence
        features.append(1.0 if image.brand_elements else 0.0)
        names.append("has_brand")
        
        # Embedding sample (first 10 dimensions)
        embedding_sample = image.embeddings[:10].tolist()
        features.extend(embedding_sample)
        names.extend([f"img_embed_{i}" for i in range(10)])
        
        self.feature_names.extend(names)
        return np.array(features, dtype=np.float32)
    
    def _extract_video_features(self, video: VideoAnalysisResult) -> np.ndarray:
        """Extract numerical features from video analysis."""
        features = []
        names = []
        
        # Duration (normalized to 0-1, assuming max 300s)
        features.append(min(video.metadata.duration_seconds / 300.0, 1.0))
        names.append("video_duration_norm")
        
        # Resolution quality (normalized)
        resolution_score = min(video.metadata.resolution[0] / 1920.0, 1.0)
        features.append(resolution_score)
        names.append("video_resolution")
        
        # FPS quality
        fps_score = min(video.metadata.fps / 60.0, 1.0)
        features.append(fps_score)
        names.append("video_fps")
        
        # Has audio
        features.append(1.0 if video.metadata.has_audio else 0.0)
        names.append("has_audio")
        
        # Number of keyframes
        features.append(len(video.keyframes))
        names.append("num_keyframes")
        
        # Number of scenes
        features.append(len(video.scenes))
        names.append("num_scenes")
        
        # Engagement prediction
        features.append(video.engagement_prediction)
        names.append("video_engagement_pred")
        
        # Embedding sample
        embedding_sample = video.embeddings[:10].tolist()
        features.extend(embedding_sample)
        names.extend([f"vid_embed_{i}" for i in range(10)])
        
        self.feature_names.extend(names)
        return np.array(features, dtype=np.float32)
    
    def _extract_link_features(self, link: LinkPreviewAnalysis) -> np.ndarray:
        """Extract numerical features from link preview analysis."""
        features = []
        names = []
        
        # Quality scores (already 0-100, normalize to 0-1)
        features.append(link.quality_scores.title_quality / 100.0)
        names.append("link_title_quality")
        
        features.append(link.quality_scores.description_quality / 100.0)
        names.append("link_desc_quality")
        
        features.append(link.quality_scores.image_quality / 100.0)
        names.append("link_image_quality")
        
        features.append(link.quality_scores.overall_quality / 100.0)
        names.append("link_overall_quality")
        
        # CTR prediction
        features.append(link.ctr_prediction)
        names.append("link_ctr_pred")
        
        # Has metadata fields
        features.append(1.0 if link.metadata.title else 0.0)
        names.append("has_title")
        
        features.append(1.0 if link.metadata.description else 0.0)
        names.append("has_description")
        
        features.append(1.0 if link.metadata.image_url else 0.0)
        names.append("has_image")
        
        # Number of OG tags
        features.append(len(link.metadata.og_tags))
        names.append("num_og_tags")
        
        # Number of Twitter tags
        features.append(len(link.metadata.twitter_tags))
        names.append("num_twitter_tags")
        
        self.feature_names.extend(names)
        return np.array(features, dtype=np.float32)
    
    def _extract_platform_features(self, platform: PlatformType) -> np.ndarray:
        """Extract platform-specific features."""
        features = []
        names = []
        
        # One-hot encoding for platform
        platforms = ["linkedin", "twitter", "bluesky"]
        for p in platforms:
            features.append(1.0 if platform == p else 0.0)
            names.append(f"platform_{p}")
        
        self.feature_names.extend(names)
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return self.feature_names