"""Tests for image preprocessor."""

import pytest
from PIL import Image
import numpy as np

from bufferiq.ml.multimodal.images.preprocessor import ImagePreprocessor
from bufferiq.ml.multimodal.exceptions import MediaProcessingError


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = ImagePreprocessor()
    assert preprocessor.max_size == 2048
    assert preprocessor.target_size is None


def test_load_pil_image(sample_image):
    """Test loading PIL Image."""
    preprocessor = ImagePreprocessor()
    loaded = preprocessor.load_image(sample_image)
    
    assert isinstance(loaded, Image.Image)
    assert loaded.size == sample_image.size


def test_load_image_bytes(sample_image_bytes):
    """Test loading image from bytes."""
    preprocessor = ImagePreprocessor()
    loaded = preprocessor.load_image(sample_image_bytes)
    
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (800, 600)


def test_resize_large_image():
    """Test resizing large image."""
    preprocessor = ImagePreprocessor(max_size=1024)
    large_image = Image.new('RGB', (3000, 2000))
    
    resized = preprocessor.resize(large_image)
    
    assert max(resized.size) == 1024
    assert resized.size[0] / resized.size[1] == large_image.size[0] / large_image.size[1]


def test_resize_small_image():
    """Test that small images are not resized."""
    preprocessor = ImagePreprocessor(max_size=2048)
    small_image = Image.new('RGB', (800, 600))
    
    resized = preprocessor.resize(small_image)
    
    assert resized.size == small_image.size


def test_normalize_image(sample_image):
    """Test image normalization."""
    preprocessor = ImagePreprocessor()
    normalized = preprocessor.normalize(sample_image)
    
    assert isinstance(normalized, np.ndarray)
    assert normalized.ndim == 3
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_normalize_grayscale_image():
    """Test normalizing grayscale image."""
    preprocessor = ImagePreprocessor()
    gray_image = Image.new('L', (800, 600))
    
    normalized = preprocessor.normalize(gray_image)
    
    # Should be converted to RGB
    assert normalized.shape[2] == 3


def test_preprocess_pipeline(sample_image):
    """Test complete preprocessing pipeline."""
    preprocessor = ImagePreprocessor()
    image, normalized = preprocessor.preprocess(sample_image)
    
    assert isinstance(image, Image.Image)
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape[:2] == image.size[::-1]


def test_invalid_image_source():
    """Test loading invalid image source."""
    preprocessor = ImagePreprocessor()
    
    with pytest.raises(MediaProcessingError):
        preprocessor.load_image(12345)  # type: ignore