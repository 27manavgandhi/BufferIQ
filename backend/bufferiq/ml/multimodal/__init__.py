"""
Multi-Modal Content Analysis System.

Provides comprehensive analysis of images, videos, and links in social media posts
to optimize engagement through visual content intelligence.
"""

from bufferiq.ml.multimodal.types import (
    ImageAnalysisResult,
    VideoAnalysisResult,
    LinkPreviewAnalysis,
    MultiModalFeatures,
    VisualQuality,
    EngagementPrediction,
)
from bufferiq.ml.multimodal.exceptions import (
    MultiModalError,
    UnsupportedPlatformError,
    MediaProcessingError,
    AnalysisError,
)

__all__ = [
    "ImageAnalysisResult",
    "VideoAnalysisResult",
    "LinkPreviewAnalysis",
    "MultiModalFeatures",
    "VisualQuality",
    "EngagementPrediction",
    "MultiModalError",
    "UnsupportedPlatformError",
    "MediaProcessingError",
    "AnalysisError",
]

__version__ = "1.0.0"