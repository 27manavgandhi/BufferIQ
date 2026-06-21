"""Audience segmentation module for BufferIQ."""

from bufferiq.ml.segmentation.types import (
    AudienceDataPoint,
    ProcessedAudienceFeatures,
    ClusteringResult,
    OptimalClusterConfig,
    PersonaProfile,
    SegmentSnapshot,
    SegmentEvolution,
    SegmentRecommendation,
)
from bufferiq.ml.segmentation.exceptions import (
    SegmentationError,
    UnsupportedPlatformError,
    InsufficientDataError,
    ClusteringError,
    PersonaGenerationError,
)

__all__ = [
    "AudienceDataPoint",
    "ProcessedAudienceFeatures",
    "ClusteringResult",
    "OptimalClusterConfig",
    "PersonaProfile",
    "SegmentSnapshot",
    "SegmentEvolution",
    "SegmentRecommendation",
    "SegmentationError",
    "UnsupportedPlatformError",
    "InsufficientDataError",
    "ClusteringError",
    "PersonaGenerationError",
]