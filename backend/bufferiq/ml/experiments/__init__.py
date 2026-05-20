"""
BufferIQ A/B Testing Framework & Experiment Engine.

Comprehensive experimentation system for data-driven content optimization.
Supports A/B tests, multivariate tests, sequential testing, and bandit algorithms.

Components:
    - Experiment Designer: Design tests with power analysis
    - Assignment Engine: Deterministic variant assignment
    - Statistical Analyzer: Hypothesis testing
    - Power Analyzer: Sample size calculation
    - Metrics Tracker: Performance tracking
    - Sequential Testing: Early stopping
    - Novelty Detector: Time-based effects
    - Bandit Optimizer: Thompson Sampling, UCB
    - Interference Detector: SUTVA validation
    - Result Analyzer: Winner determination
    - Experiment Monitor: Real-time monitoring
    - Report Generator: Comprehensive reports
    - Intelligence Service: Main orchestrator

Platform Support:
    - LinkedIn
    - Twitter
    - Bluesky

Example:
```python
    from bufferiq.ml.experiments import ExperimentIntelligenceService
    
    service = ExperimentIntelligenceService(db_session)
    
    # Create experiment
    experiment = await service.create_experiment(
        name="Headline Test",
        variants=[control, treatment],
        platform="linkedin",
        baseline_rate=0.05
    )
    
    # Assign user
    assignment = await service.assign_user(
        experiment_id=experiment.experiment_id,
        user_id="user123"
    )
    
    # Track metric
    await service.track_metric(
        experiment_id=experiment.experiment_id,
        user_id="user123",
        metric_type=MetricType.ENGAGEMENT_RATE,
        value=1.0
    )
    
    # Analyze results
    results = await service.analyze_experiment(
        experiment_id=experiment.experiment_id
    )
```

Author: BufferIQ Team
Created: Day 20
Version: 1.0.0
"""

from bufferiq.ml.experiments.design.designer import (
    ExperimentDesigner,
    ExperimentConfig,
    Variant,
    ExperimentType,
    MetricType,
)
from bufferiq.ml.experiments.design.sample_size_calculator import (
    SampleSizeCalculator,
)
from bufferiq.ml.experiments.assignment.engine import (
    AssignmentEngine,
    Assignment,
)
from bufferiq.ml.experiments.statistics.hypothesis_tester import (
    StatisticalAnalyzer,
    HypothesisTestResult,
    BayesianResult,
)
from bufferiq.ml.experiments.power.calculator import PowerAnalyzer
from bufferiq.ml.experiments.metrics.tracker import MetricsTracker
from bufferiq.ml.experiments.sequential.sprt import SequentialTester
from bufferiq.ml.experiments.novelty.detector import NoveltyDetector
from bufferiq.ml.experiments.bandits.thompson_sampling import (
    ThompsonSampling,
    BanditArm,
)
from bufferiq.ml.experiments.bandits.ucb import UCB
from bufferiq.ml.experiments.bandits.epsilon_greedy import EpsilonGreedy
from bufferiq.ml.experiments.interference.detector import InterferenceDetector
from bufferiq.ml.experiments.results.analyzer import ResultAnalyzer
from bufferiq.ml.experiments.monitoring.monitor import ExperimentMonitor
from bufferiq.ml.experiments.reporting.generator import ReportGenerator
from bufferiq.ml.experiments.intelligence.service import (
    ExperimentIntelligenceService,
)

__version__ = "1.0.0"

__all__ = [
    # Core types
    "ExperimentType",
    "MetricType",
    "ExperimentConfig",
    "Variant",
    "Assignment",
    "HypothesisTestResult",
    "BayesianResult",
    "BanditArm",
    # Design
    "ExperimentDesigner",
    "SampleSizeCalculator",
    # Assignment
    "AssignmentEngine",
    # Statistics
    "StatisticalAnalyzer",
    # Power
    "PowerAnalyzer",
    # Metrics
    "MetricsTracker",
    # Sequential
    "SequentialTester",
    # Novelty
    "NoveltyDetector",
    # Bandits
    "ThompsonSampling",
    "UCB",
    "EpsilonGreedy",
    # Interference
    "InterferenceDetector",
    # Results
    "ResultAnalyzer",
    # Monitoring
    "ExperimentMonitor",
    # Reporting
    "ReportGenerator",
    # Intelligence
    "ExperimentIntelligenceService",
]

# Supported platforms
SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]

# Statistical constants
DEFAULT_ALPHA = 0.05  # 5% Type I error rate
DEFAULT_POWER = 0.80  # 80% statistical power
DEFAULT_MDE = 0.10  # 10% minimum detectable effect