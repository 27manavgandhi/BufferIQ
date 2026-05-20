"""
Novelty detection module.

Detects novelty effects and time-based degradation in experiments.

Components:
    - NoveltyDetector: Detect novelty effects
    - TimeAnalyzer: Time-based analysis
    - InteractionDetector: Interaction effects

Example:
```python
    from bufferiq.ml.experiments.novelty import NoveltyDetector
    
    detector = NoveltyDetector()
    
    result = detector.detect(
        time_series_data=data,
        window_size=7
    )
```
"""

from bufferiq.ml.experiments.novelty.detector import NoveltyDetector
from bufferiq.ml.experiments.novelty.time_analyzer import TimeAnalyzer
from bufferiq.ml.experiments.novelty.interaction_detector import InteractionDetector

__all__ = [
    "NoveltyDetector",
    "TimeAnalyzer",
    "InteractionDetector",
]