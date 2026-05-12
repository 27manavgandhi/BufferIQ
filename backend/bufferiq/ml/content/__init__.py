"""
Content Intelligence Module.

This module provides comprehensive content analysis including:
- Text preprocessing and feature extraction
- Sentiment and emotion analysis
- Topic modeling (NMF/LDA)
- Readability scoring
- Quality checking
- Content optimization
- Diversity analysis

Example:
```python
    from bufferiq.ml.content.intelligence import ContentIntelligenceService
    
    service = ContentIntelligenceService(db_session=session)
    result = await service.analyze_content(
        text="Great post about AI!",
        platform="linkedin"
    )
```
"""

from bufferiq.ml.content.preprocessing.text_cleaner import (
    TextCleaner,
    PreprocessedText,
)
from bufferiq.ml.content.sentiment.analyzer import (
    SentimentAnalyzer,
    SentimentResult,
    Sentiment,
)
from bufferiq.ml.content.topics.nmf_modeler import NMFTopicModeler
from bufferiq.ml.content.topics.lda_modeler import LDATopicModeler
from bufferiq.ml.content.readability.analyzer import (
    ReadabilityAnalyzer,
    ReadabilityScores,
)
from bufferiq.ml.content.quality.content_validator import (
    ContentQualityChecker,
    QualityReport,
)
from bufferiq.ml.content.optimizer.optimizer import (
    ContentOptimizer,
    OptimizationResult,
)
from bufferiq.ml.content.diversity.analyzer import (
    ContentDiversityAnalyzer,
    DiversityMetrics,
)
from bufferiq.ml.content.intelligence.service import ContentIntelligenceService

__all__ = [
    # Preprocessing
    "TextCleaner",
    "PreprocessedText",
    # Sentiment
    "SentimentAnalyzer",
    "SentimentResult",
    "Sentiment",
    # Topics
    "NMFTopicModeler",
    "LDATopicModeler",
    # Readability
    "ReadabilityAnalyzer",
    "ReadabilityScores",
    # Quality
    "ContentQualityChecker",
    "QualityReport",
    # Optimization
    "ContentOptimizer",
    "OptimizationResult",
    # Diversity
    "ContentDiversityAnalyzer",
    "DiversityMetrics",
    # Intelligence
    "ContentIntelligenceService",
]
