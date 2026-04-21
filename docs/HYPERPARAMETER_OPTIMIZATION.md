# Hyperparameter Optimization Guide

## Overview

This document explains how to use BufferIQ's hyperparameter optimization system to improve model performance through systematic parameter tuning.

## Optimization Strategies

### 1. Grid Search

**When to use:** Small parameter spaces (<1000 combinations)

**Pros:**
- Exhaustive search guarantees finding best parameters in grid
- Deterministic and reproducible
- Easy to understand

**Cons:**
- Exponentially slow as parameters increase
- Inefficient for large spaces

**Example:**
```bash
python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_grid.yaml
```

### 2. Random Search

**When to use:** Large parameter spaces, limited time budget

**Pros:**
- More efficient than grid search
- Often finds good parameters quickly
- Can explore wide ranges

**Cons:**
- May miss optimal combination
- Non-deterministic (despite random seed)

**Example:**
```bash
python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_random.yaml
```

### 3. Bayesian Optimization

**When to use:** Expensive evaluations, want best performance

**Pros:**
- Intelligently explores parameter space
- Uses past trials to guide search
- Often best performance/trial ratio

**Cons:**
- Requires scikit-optimize
- More complex to configure

**Example:**
```bash
python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_bayesian.yaml
```

## Parameter Descriptions

### XGBoost Parameters

- **learning_rate** (0.01-0.3): Step size for weight updates. Lower = slower but more stable.
- **max_depth** (3-10): Maximum tree depth. Higher = more complex, risk overfitting.
- **n_estimators** (100-500): Number of trees. More trees = better fit but slower.
- **subsample** (0.6-1.0): Fraction of samples per tree. <1.0 adds randomness.
- **colsample_bytree** (0.6-1.0): Fraction of features per tree.
- **min_child_weight** (1-10): Minimum sum of instance weight in a leaf.
- **gamma** (0-0.5): Minimum loss reduction for split. Higher = more conservative.
- **reg_alpha** (0-10): L1 regularization. Higher = simpler models.
- **reg_lambda** (0-10): L2 regularization. Higher = simpler models.

### LightGBM Parameters

- **learning_rate** (0.01-0.3): Same as XGBoost.
- **num_leaves** (15-150): Max leaves per tree. More = complex.
- **n_estimators** (100-500): Number of trees.
- **feature_fraction** (0.6-1.0): Fraction of features per tree.
- **bagging_fraction** (0.6-1.0): Fraction of samples per tree.
- **bagging_freq** (0-10): Frequency of bagging (0=disabled).
- **min_child_samples** (5-30): Minimum samples in a leaf.
- **reg_alpha** (0-10): L1 regularization.
- **reg_lambda** (0-10): L2 regularization.

### RandomForest Parameters

- **n_estimators** (50-500): Number of trees.
- **max_depth** (5-30 or None): Maximum tree depth.
- **min_samples_split** (2-20): Minimum samples to split node.
- **min_samples_leaf** (1-10): Minimum samples in a leaf.
- **max_features** ('sqrt', 'log2', None): Features per split.
- **bootstrap** (True/False): Sample with replacement.

## Configuration Files

Create YAML config files with this structure:

```yaml
model_type: xgboost  # xgboost, lightgbm, random_forest
strategy: grid       # grid, random, bayesian

# For grid search:
search_space:
  learning_rate: [0.01, 0.05, 0.1]
  max_depth: [3, 5, 7]

# For random/bayesian search:
n_iter: 50  # Number of trials

cv_folds: 5
scoring: r2
n_jobs: -1
random_state: 42
output_dir: outputs/optimizations/my_optimization
```

## CLI Usage

### Run Optimization

```bash
# Grid search
python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_grid.yaml

# Dry run (validate config only)
python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_grid.yaml --dry-run
```

### View Best Parameters

```bash
python -m bufferiq.cli.optimize best-params --optimization-id xgboost_grid_20260412
```

### Compare Optimizations

```bash
python -m bufferiq.cli.optimize compare outputs/optimizations/xgboost_grid outputs/optimizations/xgboost_random
```

## Expected Performance

**Before Optimization (Day 9):**
- XGBoost: R² = 0.72
- LightGBM: R² = 0.71
- RandomForest: R² = 0.68

**After Optimization (Day 11 Target):**
- XGBoost: R² = 0.76+ (5-6% improvement)
- LightGBM: R² = 0.75+ (5-6% improvement)
- RandomForest: R² = 0.72+ (5-9% improvement)

## Output Files

Optimization creates these files in `output_dir`:

- **trials.json**: All trial parameters and scores
- **best_params.yaml**: Best parameters found
- **optimization_report.json**: Summary statistics

## Best Practices

1. **Start with random search** to explore parameter space efficiently
2. **Use Bayesian optimization** when you have time for 50+ trials
3. **Reserve grid search** for final tuning around known good values
4. **Always set random_state** for reproducibility
5. **Monitor overfitting** by comparing train vs validation scores
6. **Use cross-validation** (cv_folds=5) to avoid overfitting

## Troubleshooting

**Error: "scikit-optimize required"**
```bash
pip install scikit-optimize
```

**Error: "n_iter is required"**
Add `n_iter: 50` to your config for random/bayesian search.

**Slow optimization:**
- Reduce n_iter
- Reduce cv_folds (from 5 to 3)
- Reduce n_estimators in search space
- Use random search instead of grid search

**Poor results:**
- Widen parameter ranges
- Increase n_iter (for random/bayesian)
- Check for data quality issues
- Ensure sufficient training data

## Integration

with Training Pipeline
After optimization, use best parameters:

from bufferiq.ml.optimization.config_schema import OptimizationConfig
from bufferiq.ml.trainers.xgboost_trainer import XGBoostTrainer

# Load best params
config = OptimizationConfig.from_yaml("outputs/optimizations/xgboost_grid/best_params.yaml")

# Create trainer with best params
trainer = XGBoostTrainer()
trainer.build_model(config.best_params)

# Train model
trainer.train(X_train, y_train, X_val, y_val)

Next Steps

Day 12: Advanced optimization (Optuna, multi-objective)
Day 13: Ensemble models (combine optimized models)
Day 14: Model serving API (deploy optimized models)

### 6. Updates to Existing Files (3 files)

### backend/requirements.txt
```txt
# ... (existing dependencies)

# Day 11: Hyperparameter Optimization
scikit-optimize>=0.9.0
```

### backend/.gitignore

... (existing content)
## Day 11: Optimization outputs
outputs/optimizations/

### backend/Makefile
```makefile
# ... (existing targets)

# Day 11: Hyperparameter Optimization
.PHONY: optimize-xgboost-grid
optimize-xgboost-grid:
	python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_grid.yaml

.PHONY: optimize-xgboost-random
optimize-xgboost-random:
	python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_random.yaml

.PHONY: optimize-xgboost-bayesian
optimize-xgboost-bayesian:
	python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_bayesian.yaml

.PHONY: optimize-lightgbm
optimize-lightgbm:
	python -m bufferiq.cli.optimize run --config configs/optimization/lightgbm_grid.yaml

.PHONY: optimize-randomforest
optimize-randomforest:
	python -m bufferiq.cli.optimize run --config configs/optimization/randomforest_grid.yaml
```

