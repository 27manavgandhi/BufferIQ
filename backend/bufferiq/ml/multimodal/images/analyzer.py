"""Main image analyzer orchestrating all components."""

from typing import Dict, Any, Optional
import time
from PIL import Image

from bufferiq.ml.multimodal.types import (
    ImageAnalysisResult,
    PlatformType,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.multimodal.exceptions import UnsupportedPlatformError
from bufferiq.ml.multimodal.images.preprocessor import ImagePreprocessor
from bufferiq.ml.multimodal.images.object_detector import ObjectDetector
from bufferiq.ml.multimodal.images.ocr_extractor import OCRExtractor
from bufferiq.ml.multimodal.images.face_analyzer import FaceAnalyzer
from bufferiq.ml.multimodal.images.color_extractor import ColorExtractor
from bufferiq.ml.multimodal.images.composition_analyzer import CompositionAnalyzer
from bufferiq.ml.multimodal.images.aesthetic_scorer import AestheticScorer
from bufferiq.ml.multimodal.images.brand_detector import BrandDetector
from bufferiq.ml.multimodal.images.embeddings import ImageEmbeddingGenerator


class ImageAnalyzer:
    """Comprehensive image analysis system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize image analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.object_detector = ObjectDetector()
        self.ocr_extractor = OCRExtractor()
        self.face_analyzer = FaceAnalyzer()
        self.color_extractor = ColorExtractor()
        self.composition_analyzer = CompositionAnalyzer()
        self.aesthetic_scorer = AestheticScorer()
        self.brand_detector = BrandDetector()
        self.embedding_generator = ImageEmbeddingGenerator()
    
    async def analyze(
        self,
        image_source: str | bytes | Image.Image,
        platform: PlatformType,
        options: Optional[Dict[str, bool]] = None
    ) -> ImageAnalysisResult:
        """
        Analyze image comprehensively.
        
        Args:
            image_source: Image path, bytes, or PIL Image
            platform: Platform type (linkedin/twitter/bluesky)
            options: Analysis options
            
        Returns:
            Complete image analysis result
            
        Raises:
            UnsupportedPlatformError: If platform not supported
        """
        # Validate platform
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)
        
        start_time = time.time()
        
        # Default options
        if options is None:
            options = {}
        
        # Preprocess image
        image, _ = self.preprocessor.preprocess(image_source)
        
        # Run analyses
        objects = (
            self.object_detector.detect(image)
            if options.get("detect_objects", True)
            else []
        )
        
        text = (
            self.ocr_extractor.extract_text(image)
            if options.get("extract_text", True)
            else []
        )
        
        faces = (
            self.face_analyzer.detect_faces(image)
            if options.get("analyze_faces", True)
            else []
        )
        
        colors = (
            self.color_extractor.extract_palette(image)
            if options.get("extract_colors", True)
            else self.color_extractor.extract_palette(image)  # Always extract
        )
        
        composition = (
            self.composition_analyzer.analyze(image)
            if options.get("analyze_composition", True)
            else self.composition_analyzer.analyze(image)  # Always analyze
        )
        
        aesthetic_score = (
            self.aesthetic_scorer.score(image)
            if options.get("score_aesthetics", True)
            else self.aesthetic_scorer.score(image)  # Always score
        )
        
        brand_elements = (
            self.brand_detector.detect(image)
            if options.get("detect_brand", True)
            else []
        )
        
        # Generate embeddings (always)
        embeddings = self.embedding_generator.generate(image)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return ImageAnalysisResult(
            objects=objects,
            text=text,
            faces=faces,
            colors=colors,
            composition=composition,
            aesthetic_score=aesthetic_score,
            brand_elements=brand_elements,
            embeddings=embeddings,
            processing_time_ms=processing_time_ms,
            platform=platform,
        )