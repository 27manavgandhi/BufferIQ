"""Ensemble models for BufferIQ.

This module provides various ensemble techniques to combine multiple
models for improved performance and robustness.

Available Ensembles:
    - VotingEnsemble: Weighted voting (soft/hard)
    - StackingEnsemble: Multi-level stacking with meta-learner
    - BlendingEnsemble: Holdout-based blending
    - WeightedAverageEnsemble: Optimized weighted averaging

Utilities:
    - DiversityAnalyzer: Measure model diversity
    - ModelSelector: Intelligent model selection
    - WeightOptimizer: Optimize ensemble weights
    - EnsembleBuilder: Automated ensemble construction
    - PerformanceComparator: Compare ensemble vs base models
"""

from bufferiq.ml.ensemble.base import BaseEnsemble
from bufferiq.ml.ensemble.blending import BlendingEnsemble
from bufferiq.ml.ensemble.diversity_analyzer import DiversityAnalyzer
from bufferiq.ml.ensemble.ensemble_builder import EnsembleBuilder
from bufferiq.ml.ensemble.model_selector import ModelSelector
from bufferiq.ml.ensemble.performance_comparator import EnsemblePerformanceComparator
from bufferiq.ml.ensemble.stacking import StackingEnsemble
from bufferiq.ml.ensemble.voting import VotingEnsemble
from bufferiq.ml.ensemble.weight_optimizer import WeightOptimizer
from bufferiq.ml.ensemble.weighted_average import WeightedAverageEnsemble

__version__ = "1.0.0"

__all__ = [
    "BaseEnsemble",
    "VotingEnsemble",
    "StackingEnsemble",
    "BlendingEnsemble",
    "WeightedAverageEnsemble",
    "DiversityAnalyzer",
    "ModelSelector",
    "WeightOptimizer",
    "EnsembleBuilder",
    "EnsemblePerformanceComparator",
]
