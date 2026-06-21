"""Tests for multi-modal types."""

import pytest
import numpy as np

from bufferiq.ml.multimodal.types import (
    DetectedObject,
    ExtractedText,
    DetectedFace,
    ColorPalette,
    CompositionScores,
    SUPPORTED_PLATFORMS,
)


def test_supported_platforms():
    """Test supported platforms list."""
    assert SUPPORTED_PLATFORMS == ["linkedin", "twitter", "bluesky"]
    assert "facebook" not in SUPPORTED_PLATFORMS


def test_detected_object_to_dict(sample_detected_object):
    """Test DetectedObject serialization."""
    data = sample_detected_object.to_dict()
    
    assert data["label"] == "person"
    assert data["confidence"] == 0.85
    assert "bounding_box" in data
    assert data["bounding_box"]["x"] == 0.1


def test_extracted_text_to_dict(sample_extracted_text):
    """Test ExtractedText serialization."""
    data = sample_extracted_text.to_dict()
    
    assert data["text"] == "Sample Text"
    assert data["confidence"] == 0.9
    assert "position" in data


def test_detected_face_to_dict(sample_detected_face):
    """Test DetectedFace serialization."""
    data = sample_detected_face.to_dict()
    
    assert data["emotion"] == "happy"
    assert data["emotion_confidence"] == 0.87
    assert data["confidence"] == 0.95


def test_color_palette_to_dict(sample_color_palette):
    """Test ColorPalette serialization."""
    data = sample_color_palette.to_dict()
    
    assert len(data["dominant_colors"]) == 3
    assert len(data["color_percentages"]) == 3
    assert sum(data["color_percentages"]) == 1.0


def test_composition_scores_to_dict(sample_composition_scores):
    """Test CompositionScores serialization."""
    data = sample_composition_scores.to_dict()
    
    assert "rule_of_thirds" in data
    assert "golden_ratio" in data
    assert "symmetry" in data
    assert "balance" in data
    
    assert 0 <= data["rule_of_thirds"] <= 1
    assert 0 <= data["golden_ratio"] <= 1


def test_image_analysis_result_to_dict(sample_image_analysis_result):
    """Test ImageAnalysisResult serialization."""
    data = sample_image_analysis_result.to_dict()
    
    assert data["platform"] == "linkedin"
    assert data["aesthetic_score"] == 78.5
    assert len(data["objects"]) == 1
    assert len(data["text"]) == 1
    assert len(data["faces"]) == 1
    assert "embeddings_shape" in data


def test_video_analysis_result_to_dict(sample_video_analysis_result):
    """Test VideoAnalysisResult serialization."""
    data = sample_video_analysis_result.to_dict()
    
    assert data["platform"] == "twitter"
    assert data["engagement_prediction"] == 0.72
    assert len(data["thumbnail_urls"]) == 3
    assert data["metadata"]["duration_seconds"] == 45.0


def test_link_preview_analysis_to_dict(sample_link_preview_analysis):
    """Test LinkPreviewAnalysis serialization."""
    data = sample_link_preview_analysis.to_dict()
    
    assert data["platform"] == "bluesky"
    assert data["ctr_prediction"] == 0.045
    assert len(data["optimization_suggestions"]) == 2
    assert data["quality_scores"]["overall_quality"] == 85.0