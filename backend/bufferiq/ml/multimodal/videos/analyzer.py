"""Main video analyzer orchestrating all components."""

from typing import Dict, Any, Optional
import time

from bufferiq.ml.multimodal.types import (
    VideoAnalysisResult,
    PlatformType,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.multimodal.exceptions import UnsupportedPlatformError
from bufferiq.ml.multimodal.videos.metadata_extractor import MetadataExtractor
from bufferiq.ml.multimodal.videos.thumbnail_generator import ThumbnailGenerator
from bufferiq.ml.multimodal.videos.keyframe_extractor import KeyFrameExtractor
from bufferiq.ml.multimodal.videos.scene_detector import SceneDetector
from bufferiq.ml.multimodal.videos.audio_analyzer import AudioAnalyzer
from bufferiq.ml.multimodal.videos.embeddings import VideoEmbeddingGenerator


class VideoAnalyzer:
    """Comprehensive video analysis system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize video analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.metadata_extractor = MetadataExtractor()
        self.thumbnail_generator = ThumbnailGenerator()
        self.keyframe_extractor = KeyFrameExtractor()
        self.scene_detector = SceneDetector()
        self.audio_analyzer = AudioAnalyzer()
        self.embedding_generator = VideoEmbeddingGenerator()
    
    async def analyze(
        self,
        video_path: str,
        platform: PlatformType,
        extract_keyframes: bool = True,
        detect_scenes: bool = True,
        analyze_audio: bool = True
    ) -> VideoAnalysisResult:
        """
        Analyze video comprehensively.
        
        Args:
            video_path: Path to video file
            platform: Platform type (linkedin/twitter/bluesky)
            extract_keyframes: Whether to extract key frames
            detect_scenes: Whether to detect scenes
            analyze_audio: Whether to analyze audio
            
        Returns:
            Complete video analysis result
            
        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        # Validate platform
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)
        
        start_time = time.time()
        
        # Extract metadata
        metadata = self.metadata_extractor.extract(video_path)
        
        # Generate thumbnails
        thumbnails = self.thumbnail_generator.generate(video_path, count=3)
        
        # Extract key frames
        keyframes = (
            self.keyframe_extractor.extract(video_path)
            if extract_keyframes
            else []
        )
        
        # Detect scenes
        scenes = (
            self.scene_detector.detect(video_path)
            if detect_scenes
            else []
        )
        
        # Analyze audio
        audio_features = (
            self.audio_analyzer.analyze(video_path)
            if analyze_audio and metadata.has_audio
            else None
        )
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate(keyframes)
        
        # Predict engagement (simple heuristic for now)
        engagement = self._predict_engagement(metadata, keyframes, audio_features)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return VideoAnalysisResult(
            metadata=metadata,
            thumbnail_urls=thumbnails,
            keyframes=keyframes,
            scenes=scenes,
            audio_features=audio_features,
            embeddings=embeddings,
            engagement_prediction=engagement,
            processing_time_ms=processing_time_ms,
            platform=platform,
        )
    
    def _predict_engagement(
        self,
        metadata: Any,
        keyframes: list,
        audio_features: Any
    ) -> float:
        """
        Predict engagement based on video features.
        
        Args:
            metadata: Video metadata
            keyframes: Key frames
            audio_features: Audio features
            
        Returns:
            Engagement prediction (0-1)
        """
        score = 0.5  # Base score
        
        # Optimal duration (30-90 seconds)
        if 30 <= metadata.duration_seconds <= 90:
            score += 0.1
        
        # Good resolution
        if metadata.resolution[0] >= 1280:
            score += 0.1
        
        # Has audio
        if metadata.has_audio:
            score += 0.1
        
        # Good number of scenes (dynamic content)
        if keyframes and len(keyframes) >= 5:
            score += 0.1
        
        return min(score, 1.0)