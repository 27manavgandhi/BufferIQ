"""Type definitions for multi-modal analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import numpy as np

# Supported platforms
PlatformType = Literal["linkedin", "twitter", "bluesky"]
SUPPORTED_PLATFORMS: List[str] = ["linkedin", "twitter", "bluesky"]

# Media types
MediaType = Literal["image", "video", "link"]


@dataclass
class DetectedObject:
    """Detected object in image."""
    
    label: str
    confidence: float
    bounding_box: Dict[str, float]  # x, y, width, height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
        }


@dataclass
class ExtractedText:
    """Extracted text from image."""
    
    text: str
    confidence: float
    position: Dict[str, float]  # x, y, width, height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "position": self.position,
        }


@dataclass
class DetectedFace:
    """Detected face in image."""
    
    bounding_box: Dict[str, float]
    confidence: float
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bounding_box": self.bounding_box,
            "confidence": self.confidence,
            "emotion": self.emotion,
            "emotion_confidence": self.emotion_confidence,
        }


@dataclass
class ColorPalette:
    """Color palette from image."""
    
    dominant_colors: List[List[int]]  # RGB values
    color_percentages: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dominant_colors": self.dominant_colors,
            "color_percentages": self.color_percentages,
        }


@dataclass
class CompositionScores:
    """Composition quality scores."""
    
    rule_of_thirds: float
    golden_ratio: float
    symmetry: float
    balance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_of_thirds": self.rule_of_thirds,
            "golden_ratio": self.golden_ratio,
            "symmetry": self.symmetry,
            "balance": self.balance,
        }


@dataclass
class ImageAnalysisResult:
    """Result of image analysis."""
    
    objects: List[DetectedObject]
    text: List[ExtractedText]
    faces: List[DetectedFace]
    colors: ColorPalette
    composition: CompositionScores
    aesthetic_score: float  # 0-100
    brand_elements: List[str]
    embeddings: np.ndarray
    processing_time_ms: float
    platform: PlatformType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "text": [t.to_dict() for t in self.text],
            "faces": [f.to_dict() for f in self.faces],
            "colors": self.colors.to_dict(),
            "composition": self.composition.to_dict(),
            "aesthetic_score": self.aesthetic_score,
            "brand_elements": self.brand_elements,
            "embeddings_shape": self.embeddings.shape,
            "processing_time_ms": self.processing_time_ms,
            "platform": self.platform,
        }


@dataclass
class VideoMetadata:
    """Video metadata."""
    
    duration_seconds: float
    resolution: tuple[int, int]
    fps: float
    codec: str
    has_audio: bool
    file_size_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "duration_seconds": self.duration_seconds,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "codec": self.codec,
            "has_audio": self.has_audio,
            "file_size_mb": self.file_size_mb,
        }


@dataclass
class KeyFrame:
    """Extracted key frame."""
    
    timestamp: float
    frame_index: int
    thumbnail_url: str
    importance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "thumbnail_url": self.thumbnail_url,
            "importance_score": self.importance_score,
        }


@dataclass
class Scene:
    """Detected scene in video."""
    
    start_time: float
    end_time: float
    duration: float
    scene_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "scene_type": self.scene_type,
        }


@dataclass
class AudioFeatures:
    """Audio features from video."""
    
    duration_seconds: float
    sample_rate: int
    channels: int
    has_speech: bool
    music_detected: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "has_speech": self.has_speech,
            "music_detected": self.music_detected,
        }


@dataclass
class VideoAnalysisResult:
    """Result of video analysis."""
    
    metadata: VideoMetadata
    thumbnail_urls: List[str]
    keyframes: List[KeyFrame]
    scenes: List[Scene]
    audio_features: Optional[AudioFeatures]
    embeddings: np.ndarray
    engagement_prediction: float
    processing_time_ms: float
    platform: PlatformType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "thumbnail_urls": self.thumbnail_urls,
            "keyframes": [kf.to_dict() for kf in self.keyframes],
            "scenes": [s.to_dict() for s in self.scenes],
            "audio_features": self.audio_features.to_dict() if self.audio_features else None,
            "embeddings_shape": self.embeddings.shape,
            "engagement_prediction": self.engagement_prediction,
            "processing_time_ms": self.processing_time_ms,
            "platform": self.platform,
        }


@dataclass
class LinkMetadata:
    """Link metadata."""
    
    title: Optional[str]
    description: Optional[str]
    image_url: Optional[str]
    site_name: Optional[str]
    url: str
    og_tags: Dict[str, str] = field(default_factory=dict)
    twitter_tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "image_url": self.image_url,
            "site_name": self.site_name,
            "url": self.url,
            "og_tags": self.og_tags,
            "twitter_tags": self.twitter_tags,
        }


@dataclass
class QualityScores:
    """Quality scores for link preview."""
    
    title_quality: float  # 0-100
    description_quality: float  # 0-100
    image_quality: float  # 0-100
    overall_quality: float  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title_quality": self.title_quality,
            "description_quality": self.description_quality,
            "image_quality": self.image_quality,
            "overall_quality": self.overall_quality,
        }


@dataclass
class LinkPreviewAnalysis:
    """Result of link preview analysis."""
    
    metadata: LinkMetadata
    quality_scores: QualityScores
    ctr_prediction: float
    optimization_suggestions: List[str]
    platform: PlatformType
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "quality_scores": self.quality_scores.to_dict(),
            "ctr_prediction": self.ctr_prediction,
            "optimization_suggestions": self.optimization_suggestions,
            "platform": self.platform,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class MultiModalFeatures:
    """Fused multi-modal features."""
    
    feature_vector: np.ndarray
    feature_names: List[str]
    modalities_present: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_vector_shape": self.feature_vector.shape,
            "feature_names": self.feature_names,
            "modalities_present": self.modalities_present,
        }


@dataclass
class VisualQuality:
    """Visual quality assessment."""
    
    overall_score: float  # 0-100
    technical_quality: float  # Resolution, sharpness, etc.
    aesthetic_quality: float  # Composition, colors, etc.
    engagement_potential: float  # Predicted engagement impact
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "technical_quality": self.technical_quality,
            "aesthetic_quality": self.aesthetic_quality,
            "engagement_potential": self.engagement_potential,
        }


@dataclass
class EngagementPrediction:
    """Engagement prediction result."""
    
    predicted_engagement_rate: float
    confidence_interval: tuple[float, float]
    improvement_potential: float  # % improvement with optimization
    recommendation_priority: str  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_engagement_rate": self.predicted_engagement_rate,
            "confidence_interval": list(self.confidence_interval),
            "improvement_potential": self.improvement_potential,
            "recommendation_priority": self.recommendation_priority,
        }