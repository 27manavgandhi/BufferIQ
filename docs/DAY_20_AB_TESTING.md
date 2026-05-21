# Day 20: A/B Testing Framework & Experiment Engine

## Overview

Complete A/B testing framework with statistical rigor, sequential testing, bandit algorithms, and comprehensive experiment management.

## Features

### Experiment Design
- A/B tests (2 variants)
- Multivariate tests (3+ variants)
- Power analysis and sample size calculation
- Traffic allocation control
- Platform validation (LinkedIn, Twitter, Bluesky)

### Statistical Analysis
- T-tests for continuous metrics
- Z-tests for proportions
- Mann-Whitney U (non-parametric)
- Bayesian analysis
- Effect size calculation (Cohen's d, Hedge's g, Cliff's delta)
- Confidence intervals

### Sequential Testing
- SPRT (Sequential Probability Ratio Test)
- Early stopping rules
- Group sequential designs
- Always-valid p-values

### Bandit Algorithms
- Thompson Sampling
- Upper Confidence Bound (UCB)
- Epsilon-Greedy
- Contextual bandits

### Advanced Features
- Novelty effect detection
- Interference detection (SUTVA validation)
- Sample ratio mismatch detection
- Anomaly detection
- Comprehensive reporting

## Quick Start

```python
from bufferiq.ml.experiments import ExperimentIntelligenceService
from bufferiq.ml.experiments.design.designer import Variant, MetricType

# Initialize service
service = ExperimentIntelligenceService(db_session)

# Create experiment
experiment = await service.create_experiment(
    name="Headline Test",
    description="Test AI-generated headlines",
    variants=[
        Variant("control", "Original", "Current headline", 0.5, {}, True),
        Variant("treatment", "AI Headline", "AI-powered", 0.5, {"version": "ai"})
    ],
    platform="linkedin",
    primary_metric=MetricType.ENGAGEMENT_RATE,
    baseline_rate=0.05,
    mde=0.10
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

if results['should_launch']:
    print(f"✓ Launch treatment! ({results['confidence']:.1%} confidence)")
```

## Architecture

### Module Structure

````
experiments/
├── design/          # Experiment configuration
├── assignment/      # User assignment
├── statistics/      # Statistical tests
├── power/           # Power analysis
├── metrics/         # Metric tracking
├── sequential/      # Sequential testing
├── novelty/         # Novelty detection
├── bandits/         # Bandit algorithms
├── interference/    # Interference detection
├── results/         # Result analysis
├── monitoring/      # Health monitoring
├── reporting/       # Report generation
└── intelligence/    # Main orchestrator
`````

## API Endpoints

### Create Experiment
`````http
POST /api/v1/experiments/create
`````

### Assign User
`````http
POST /api/v1/experiments/assign
`````

### Track Metric
`````http
POST /api/v1/experiments/track
`````

### Get Results
`````http
GET /api/v1/experiments/{experiment_id}/results
`````

## Statistical Guarantees

- **Type I Error (α)**: ≤ 0.05 (5% false positive rate)
- **Statistical Power**: ≥ 0.80 (80% power)
- **Minimum Detectable Effect**: Configurable (default 10%)
- **Sample Size Accuracy**: 95%+
- **Assignment Consistency**: 100%

## Platform Support

- ✅ LinkedIn
- ✅ Twitter
- ✅ Bluesky
- ❌ Facebook (not supported)
- ❌ Instagram (not supported)

## Testing

`````bash
# Run all tests
pytest tests/ml/experiments/ -v --cov=bufferiq/ml/experiments

# Expected: 450+ tests, 92%+ coverage, 0 failures
`````

## Performance

- Variant assignment: <100ms
- Statistical analysis: <500ms for 10K samples
- Sample size calculation: <50ms
- Memory usage: <800MB typical workload

## License

Proprietary - BufferIQ Team