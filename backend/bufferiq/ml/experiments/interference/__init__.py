"""
Interference detection module.

Detects network effects, spillover, and SUTVA violations.

Components:
    - InterferenceDetector: Main detector
    - NetworkAnalyzer: Network effect analysis
    - Mitigator: Mitigation strategies

Example:
```python
    from bufferiq.ml.experiments.interference import InterferenceDetector
    
    detector = InterferenceDetector()
    
    result = detector.detect(
        treatment_data=treatment,
        control_data=control,
        network_structure=network
    )
```
"""

from bufferiq.ml.experiments.interference.detector import InterferenceDetector
from bufferiq.ml.experiments.interference.network_analyzer import NetworkAnalyzer
from bufferiq.ml.experiments.interference.mitigator import InterferenceMitigator

__all__ = [
    "InterferenceDetector",
    "NetworkAnalyzer",
    "InterferenceMitigator",
]