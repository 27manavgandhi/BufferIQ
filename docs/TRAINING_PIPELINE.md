# Training Pipeline Documentation

## Overview

Day 8 training pipeline provides experiment tracking, model registry, checkpointing, and cross-validation.

## Components

### Training Pipeline

Main orchestrator for model training.

```python
from bufferiq.ml.training.pipeline import TrainingPipeline

config = TrainingPipelineConfig.from_yaml("config.yaml")
pipeline = TrainingPipeline(config, session)
results = await pipeline.run()
```

### Experiment Tracker

Track experiments with metrics and artifacts.

```python
from bufferiq.ml.training.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("experiment_001")
tracker.log_params({"learning_rate": 0.01})
tracker.log_metrics({"train_r2": 0.85})
```

### Model Registry

Register and manage trained models.

```python
from bufferiq.ml.training.model_registry import ModelRegistry

registry = ModelRegistry()
registry.register_model(
    model_path="model.joblib",
    version="1.0.0",
    metrics={"r2": 0.85}
)
```

### Checkpointing

Save best models during training.

```python
from bufferiq.ml.training.checkpoint import Checkpoint

checkpoint = Checkpoint(
    checkpoint_dir="checkpoints/",
    patience=10
)
```

## Usage

See `docs/MODEL_TRAINING.md` for complete guide.