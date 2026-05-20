"""
Reporting module.

Generates experiment reports and visualizations.

Components:
    - ReportGenerator: Main report generator
    - Visualizer: Create visualizations
    - SummaryBuilder: Build summaries

Example:
```python
    from bufferiq.ml.experiments.reporting import ReportGenerator
    
    generator = ReportGenerator()
    
    report = generator.generate(experiment_result=result)
```
"""

from bufferiq.ml.experiments.reporting.generator import ReportGenerator
from bufferiq.ml.experiments.reporting.visualizer import ReportVisualizer
from bufferiq.ml.experiments.reporting.summary_builder import SummaryBuilder

__all__ = [
    "ReportGenerator",
    "ReportVisualizer",
    "SummaryBuilder",
]