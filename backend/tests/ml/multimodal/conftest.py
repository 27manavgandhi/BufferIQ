"""Pytest fixtures for multi-modal tests."""

import pytest
import numpy as np
from PIL import Image
import io

from bufferiq.ml.multimodal.types import (
    ImageAnalysisResult,
    VideoAnalysisResult,
    LinkPreviewAnalysis,
    DetectedObject,
    ExtractedText,
    DetectedFace,
    ColorPalette,
    CompositionScores,
    VideoMetadata,
    KeyFrame,
    Scene,
    AudioFeatures,
    LinkMetadata,
    QualityScores,
)


@pytest.fixture
def sample_image():
    """Create a sample PIL Image."""
    img = Image.new('RGB', (800, 600), color='red')
    return img


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes."""
    img = Image.new('RGB', (800, 600), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


@pytest.fixture
def sample_detected_object():
    """Create a sample detected object."""
    return DetectedObject(
        label="person",
        confidence=0.85,
        bounding_box={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}
    )


@pytest.fixture
def sample_extracted_text():
    """Create a sample extracted text."""
    return ExtractedText(
        text="Sample Text",
        confidence=0.9,
        position={"x": 0.5, "y": 0.5, "width": 0.2, "height": 0.05}
    )


@pytest.fixture
def sample_detected_face():
    """Create a sample detected face."""
    return DetectedFace(
        bounding_box={"x": 0.3, "y": 0.2, "width": 0.2, "height": 0.3},
        confidence=0.95,
        emotion="happy",
        emotion_confidence=0.87
    )


@pytest.fixture
def sample_color_palette():
    """Create a sample color palette."""
    return ColorPalette(
        dominant_colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        color_percentages=[0.5, 0.3, 0.2]
    )


@pytest.fixture
def sample_composition_scores():
    """Create sample composition scores."""
    return CompositionScores(
        rule_of_thirds=0.75,
        golden_ratio=0.68,
        symmetry=0.82,
        balance=0.71
    )


@pytest.fixture
def sample_image_analysis_result(
    sample_detected_object,
    sample_extracted_text,
    sample_detected_face,
    sample_color_palette,
    sample_composition_scores
):
    """Create a sample image analysis result."""
    return ImageAnalysisResult(
        objects=[sample_detected_object],
        text=[sample_extracted_text],
        faces=[sample_detected_face],
        colors=sample_color_palette,
        composition=sample_composition_scores,
        aesthetic_score=78.5,
        brand_elements=["logo"],
        embeddings=np.random.randn(512),
        processing_time_ms=123.45,
        platform="linkedin"
    )


@pytest.fixture
def sample_video_metadata():
    """Create sample video metadata."""
    return VideoMetadata(
        duration_seconds=45.0,
        resolution=(1920, 1080),
        fps=30.0,
        codec="h264",
        has_audio=True,
        file_size_mb=12.5
    )


@pytest.fixture
def sample_keyframe():
    """Create a sample keyframe."""
    return KeyFrame(
        timestamp=10.5,
        frame_index=315,
        thumbnail_url="/tmp/keyframe_315.jpg",
        importance_score=0.85
    )


@pytest.fixture
def sample_scene():
    """Create a sample scene."""
    return Scene(
        start_time=0.0,
        end_time=15.0,
        duration=15.0,
        scene_type="transition"
    )


@pytest.fixture
def sample_audio_features():
    """Create sample audio features."""
    return AudioFeatures(
        duration_seconds=45.0,
        sample_rate=44100,
        channels=2,
        has_speech=True,
        music_detected=False
    )


@pytest.fixture
def sample_video_analysis_result(
    sample_video_metadata,
    sample_keyframe,
    sample_scene,
    sample_audio_features
):
    """Create a sample video analysis result."""
    return VideoAnalysisResult(
        metadata=sample_video_metadata,
        thumbnail_urls=["/tmp/thumb_0.jpg", "/tmp/thumb_1.jpg", "/tmp/thumb_2.jpg"],
        keyframes=[sample_keyframe],
        scenes=[sample_scene],
        audio_features=sample_audio_features,
        embeddings=np.random.randn(512),
        engagement_prediction=0.72,
        processing_time_ms=456.78,
        platform="twitter"
    )


@pytest.fixture
def sample_link_metadata():
    """Create sample link metadata."""
    return LinkMetadata(
        title="Sample Article Title",
        description="This is a sample article description with some content.",
        image_url="https://example.com/image.jpg",
        site_name="Example Site",
        url="https://example.com/article",
        og_tags={"og:title": "Sample Article Title"},
        twitter_tags={"twitter:card": "summary_large_image"}
    )


@pytest.fixture
def sample_quality_scores():
    """Create sample quality scores."""
    return QualityScores(
        title_quality=85.0,
        description_quality=78.0,
        image_quality=92.0,
        overall_quality=85.0
    )


@pytest.fixture
def sample_link_preview_analysis(sample_link_metadata, sample_quality_scores):
    """Create a sample link preview analysis."""
    return LinkPreviewAnalysis(
        metadata=sample_link_metadata,
        quality_scores=sample_quality_scores,
        ctr_prediction=0.045,
        optimization_suggestions=["Add more descriptive title", "Optimize image size"],
        platform="bluesky",
        processing_time_ms=234.56
    )