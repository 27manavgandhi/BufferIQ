"""
Results analysis module.

Analyzes experiment results and determines winners.

Components:
    - ResultAnalyzer: Main analyzer
    - Segmentation: Segment analysis
    - WinnerSelector: Winner determination

Example:
```python
    from bufferiq.ml.experiments.results import ResultAnalyzer
    
    analyzer = ResultAnalyzer()
    
    result = analyzer.analyze(
        experiment_data=data,
        alpha=0.05
    )
```
"""

from bufferiq.ml.experiments.results.analyzer import ResultAnalyzer
from bufferiq.ml.experiments.results.segmentation import SegmentationAnalyzer
from bufferiq.ml.experiments.results.winner_selector import WinnerSelector

__all__ = [
    "ResultAnalyzer",
    "SegmentationAnalyzer",
    "WinnerSelector",
]