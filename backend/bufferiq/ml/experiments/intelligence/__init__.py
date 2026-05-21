"""
Experiment intelligence service.

Main orchestrator for the experimentation framework.

Example:
```python
    from bufferiq.ml.experiments.intelligence import ExperimentIntelligenceService
    
    service = ExperimentIntelligenceService(db_session)
    
    experiment = await service.create_experiment(...)
```
"""

from bufferiq.ml.experiments.intelligence.service import ExperimentIntelligenceService

__all__ = [
    "ExperimentIntelligenceService",
]