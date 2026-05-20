"""
Assignment engine module.

Handles deterministic variant assignment with consistent
hashing and assignment logging.

Components:
    - AssignmentEngine: Main assignment logic
    - Bucketing: Hash-based bucketing
    - AssignmentLogger: Assignment tracking

Example:
```python
    from bufferiq.ml.experiments.assignment import AssignmentEngine
    
    engine = AssignmentEngine(db_session)
    
    assignment = engine.assign(
        experiment_config=config,
        user_id="user123"
    )
```
"""

from bufferiq.ml.experiments.assignment.engine import (
    AssignmentEngine,
    Assignment,
)
from bufferiq.ml.experiments.assignment.bucketing import HashBucketing
from bufferiq.ml.experiments.assignment.logger import AssignmentLogger

__all__ = [
    "AssignmentEngine",
    "Assignment",
    "HashBucketing",
    "AssignmentLogger",
]