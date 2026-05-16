"""
Multi-brand voice management module.

Manages multiple voice profiles for different brands.
"""

from bufferiq.ml.voice.brands.manager import MultiBrandVoiceManager
from bufferiq.ml.voice.brands.comparator import VoiceComparator

__all__ = [
    "MultiBrandVoiceManager",
    "VoiceComparator",
]