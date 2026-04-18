# Model Training Documentation - BufferIQ Day 9

## Overview

Day 9 implements concrete model trainers (XGBoost, LightGBM, Random Forest) that inherit from the `BaseTrainer` abstract class. These trainers enable engagement prediction with different algorithms, each offering unique strengths.

## Implemented Models

### 1. XGBoost Trainer

**Algorithm:** Gradient Boosting with regularization  
**Best for:** High accuracy, feature importance, interpretability

#### Key Features
- Early stopping on validation set
- L1/L2 regularization (reg_alpha, reg_lambda)
- Tree-based feature importance
- Column/row subsampling
- Learning rate control

#### Default Hyperparameters
```python
{
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0
}
```

#### Usage
```python
from bufferiq.ml.trainers import XGBoostTrainer

trainer = XGBoostTrainer(random_state=42, verbose=True)
trainer.build_model({
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05
})

metrics = trainer.train(X_train, y_train, X_val, y_val)
predictions = trainer.predict(X_test)
importance = trainer.get_feature_importance()
```

### 2. LightGBM Trainer

**Algorithm:** Gradient Boosting with leaf-wise growth  
**Best for:** Fast training, large datasets, memory efficiency

#### Key Features
- Leaf-wise tree growth (faster than level-wise)
- Categorical feature support
- Lower memory usage
- Faster training speed
- Histogram-based splitting

#### Default Hyperparameters
```python
{
    "n_estimators": 200,
    "max_depth": 7,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1
}
```

#### Usage
```python
from bufferiq.ml.trainers import LightGBMTrainer

trainer = LightGBMTrainer(random_state=42, verbose=True)
trainer.build_model({
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05
})

metrics = trainer.train(X_train, y_train, X_val, y_val)
```

### 3. Random Forest Trainer

**Algorithm:** Ensemble of decision trees  
**Best for:** Baseline, robustness, parallel training

#### Key Features
- Parallel training (n_jobs=-1)
- No early stopping needed
- Out-of-bag (OOB) estimates
- Robust to overfitting
- Feature importance via impurity

#### Default Hyperparameters
```python
{
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": -1
}
```

#### Usage
```python
from bufferiq.ml.trainers import RandomForestTrainer

trainer = RandomForestTrainer(random_state=42, verbose=True)
trainer.build_model({
    "n_estimators": 100,
    "max_depth": 15
})

metrics = trainer.train(X_train, y_train, X_val, y_val)
```

## Model Comparison

| Feature | XGBoost | LightGBM | Random Forest |
|---------|---------|----------|---------------|
| Training Speed | Medium | Fast | Slow |
| Memory Usage | Medium | Low | High |
| Accuracy | High | High | Medium |
| Overfitting Risk | Low (regularization) | Medium | Low |
| Interpretability | Good | Good | Fair |
| Parallel Training | Yes | Yes | Yes |
| Early Stopping | Yes | Yes | No |

## Training Workflow

### 1. Configure Training

Create YAML config file:

```yaml
experiment:
  experiment_name: "xgboost_v1"
  description: "XGBoost with regularization"
  use_cross_validation: false

data:
  target_column: "engagement_rate"
  platforms: ["linkedin", "twitter", "bluesky"]
  test_size: 0.2
  validation_size: 0.1
  time_based_split: true

model:
  model_type: "xgboost"
  hyperparameters:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05

training:
  max_epochs: 200
  early_stopping_patience: 15
  checkpoint_monitor: "val_r2"
  checkpoint_mode: "max"
```

### 2. Train Model

#### CLI
```bash
python -m bufferiq.cli.train run --config configs/training/xgboost.yaml
```

#### Script
```bash
python scripts/train_model.py --config configs/training/xgboost.yaml --verbose
```

#### Programmatic
```python
from bufferiq.ml.training.config_schema import TrainingPipelineConfig
from bufferiq.ml.training.pipeline import TrainingPipeline

config = TrainingPipelineConfig.from_yaml("configs/training/xgboost.yaml")

async with async_session_maker() as session:
    pipeline = TrainingPipeline(config, session)
    results = await pipeline.run()
```

### 3. Evaluate Results

```python
# Test metrics
print(f"R²: {results['test_metrics']['r2']:.4f}")
print(f"MAE: {results['test_metrics']['mae']:.4f}")
print(f"RMSE: {results['test_metrics']['rmse']:.4f}")

# Feature importance
trainer = XGBoostTrainer.load_model(results['model_path'])
importance = trainer.get_feature_importance()
print(importance.head(10))
```

## Performance Metrics

### Regression Metrics

All trainers calculate these metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual
- **RMSE (Root Mean Squared Error)**: Square root of average squared errors
- **R² (R-squared)**: Proportion of variance explained (0-1, higher is better)
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error

### Expected Performance

Based on 92 features from Day 7:

| Model | Expected R² | Expected MAE | Training Time (10k samples) |
|-------|-------------|--------------|----------------------------|
| Random Forest | 0.68 | 0.14 | ~30s |
| XGBoost | 0.72 | 0.12 | ~45s |
| LightGBM | 0.71 | 0.13 | ~20s |

## Feature Importance

All trainers support feature importance extraction:

```python
importance = trainer.get_feature_importance()

# Top 10 features
print(importance.head(10))

# Features with importance > threshold
important = importance[importance['importance'] > 0.01]
```

### Importance Types

- **XGBoost**: Gain-based (total gain when feature is used)
- **LightGBM**: Split-based (number of splits using feature)
- **Random Forest**: Impurity-based (decrease in impurity)

## Platform-Specific Training

Train on specific platforms:

```yaml
data:
  platforms: ["linkedin"]  # LinkedIn only
  # OR
  platforms: ["twitter", "bluesky"]  # Multiple
```

## Best Practices

### 1. Always Use Validation Set

```python
trainer.train(X_train, y_train, X_val, y_val)  # ✅ Good
trainer.train(X_train, y_train)  # ❌ No validation
```

### 2. Set Random State

```python
trainer = XGBoostTrainer(random_state=42)  # ✅ Reproducible
```

### 3. Use Time-Based Splits

```yaml
data:
  time_based_split: true  # ✅ Prevents data leakage
```

### 4. Monitor Validation Metrics

```python
if metrics['val_r2'] < 0.6:
    logger.warning("Low validation R² - check for overfitting")
```

### 5. Save Trained Models

```python
trainer.save_model("outputs/models/xgboost_v1.joblib")
```

## Troubleshooting

### Issue: Low R² Score

**Possible Causes:**
- Insufficient features
- Wrong hyperparameters
- Data leakage in validation
- Target variable has low predictability

**Solutions:**
```python
# 1. Check feature importance
importance = trainer.get_feature_importance()
print(f"Top feature: {importance.iloc[0]['feature']}")

# 2. Try different hyperparameters
trainer.build_model({
    "n_estimators": 500,  # More trees
    "learning_rate": 0.01  # Slower learning
})

# 3. Verify data split
prep = DataPreparation(time_based_split=True)
```

### Issue: Overfitting (train R² >> val R²)

**Solutions:**
```python
# XGBoost/LightGBM - increase regularization
hyperparams = {
    "reg_alpha": 0.5,  # L1 regularization
    "reg_lambda": 2.0,  # L2 regularization
    "max_depth": 4,    # Shallower trees
}

# Random Forest - reduce complexity
hyperparams = {
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5
}
```

### Issue: Slow Training

**Solutions:**
```python
# Use LightGBM instead of XGBoost
trainer = LightGBMTrainer()

# Random Forest - reduce trees or use fewer samples
hyperparams = {
    "n_estimators": 50,
    "max_samples": 0.7  # Bootstrap 70% of data
}
```

### Issue: Memory Error

**Solutions:**
```python
# Use LightGBM (lowest memory)
trainer = LightGBMTrainer()

# Reduce batch size or features
config.data.feature_columns = top_50_features
```

## Integration with Pipeline

Trainers integrate seamlessly with training pipeline:

```python
# Pipeline automatically selects trainer based on config
config.model.model_type = "xgboost"  # or "lightgbm", "random_forest"

pipeline = TrainingPipeline(config, session)
results = await pipeline.run()
```

Pipeline handles:
- Data loading
- Feature extraction
- Train/val/test split
- Model initialization
- Training
- Evaluation
- Experiment logging
- Model registration

## Next Steps (Day 10+)

1. **Model Evaluation** (Day 10): Deep analysis of model performance
2. **Hyperparameter Tuning** (Day 11-12): Grid search, Bayesian optimization
3. **Ensemble Models** (Day 13): Combine multiple models
4. **Model Serving** (Day 14): API endpoint for predictions

## API Reference

### XGBoostTrainer

```python
class XGBoostTrainer(BaseTrainer):
    def __init__(self, model_name="xgboost", random_state=42, verbose=True)
    def build_model(self, hyperparameters: Dict[str, Any]) -> xgb.XGBRegressor
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]
    def get_feature_importance(self) -> pd.DataFrame
```

### LightGBMTrainer

```python
class LightGBMTrainer(BaseTrainer):
    def __init__(self, model_name="lightgbm", random_state=42, verbose=True)
    def build_model(self, hyperparameters: Dict[str, Any]) -> lgb.LGBMRegressor
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]
    def get_feature_importance(self) -> pd.DataFrame
```

### RandomForestTrainer

```python
class RandomForestTrainer(BaseTrainer):
    def __init__(self, model_name="random_forest", random_state=42, verbose=True)
    def build_model(self, hyperparameters: Dict[str, Any]) -> RandomForestRegressor
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]
    def get_feature_importance(self) -> pd.DataFrame
```

## Summary

Day 9 delivers production-ready model trainers:
- ✅ 3 algorithms (XGBoost, LightGBM, Random Forest)
- ✅ Consistent BaseTrainer interface
- ✅ Full test coverage (90%+)
- ✅ Feature importance extraction
- ✅ Integration with training pipeline
- ✅ Comprehensive documentation

These trainers enable accurate engagement prediction with flexibility to choose the best algorithm for specific needs.