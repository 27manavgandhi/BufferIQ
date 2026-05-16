"""
Voice profiler module.

Builds and manages comprehensive voice profiles
with versioning and signature generation.
"""

from bufferiq.ml.voice.profiler.builder import VoiceProfile, VoiceProfileBuilder
from bufferiq.ml.voice.profiler.signature_generator import VoiceSignatureGenerator
from bufferiq.ml.voice.profiler.versioning import VoiceProfileVersioning

__all__ = [
    "VoiceProfile",
    "VoiceProfileBuilder",
    "VoiceSignatureGenerator",
    "VoiceProfileVersioning",
]