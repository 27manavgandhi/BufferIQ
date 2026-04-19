# Model Evaluation Documentation - BufferIQ Day 10

## Overview

Day 10 implements comprehensive model evaluation tools to deeply understand model performance, identify weaknesses, analyze errors, and generate actionable insights. The evaluation system provides multiple metrics, visualizations, and diagnostic tools.

## Evaluation Modules

### 1. Model Evaluator

**Purpose:** Calculate comprehensive regression metrics and analyze performance across different dimensions.

#### Features
- **9+ Regression Metrics**: MAE, RMSE, R², MAPE, Max Error, Median AE, Explained Variance
- **Platform Analysis**: Performance breakdown by linkedin/twitter/bluesky
- **Temporal Analysis**: Performance trends over time periods
- **Content Type Analysis**: Performance by URL presence, hashtags, text length
- **Error Analysis**: Identify worst predictions and error patterns

#### Usage
```python
from bufferiq.ml.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator()

# Calculate metrics
metrics = evaluator.calculate_metrics(y_true, y_pred)
print(f"R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}")

# Platform-wise evaluation
platform_metrics = evaluator.evaluate_by_platform(y_test, predictions, platforms)

# Temporal evaluation
temporal_metrics = evaluator.evaluate_by_time_period(
    y_test, predictions, timestamps, period='month'
)

# Comprehensive summary
summary = evaluator.generate_evaluation_summary(
    trainer, X_test, y_test, platforms, timestamps
)
```

### 2. Feature Importance Analyzer

**Purpose:** Understand which features contribute most to predictions.

#### Methods
1. **Built-in Importance**: Model's native feature importance (fast)
2. **Permutation Importance**: Shuffle feature and measure R² drop (reliable)

#### Usage
```python
from bufferiq.ml.evaluation.feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()

# Built-in importance
builtin_imp = analyzer.get_builtin_importance(trainer)

# Permutation importance
perm_imp = analyzer.calculate_permutation_importance(
    trainer, X_test, y_test, n_repeats=10
)

# Compare methods
comparison = analyzer.compare_importance_methods(
    trainer, X_test, y_test, top_n=20
)

# Visualize
analyzer.plot_importance(builtin_imp, top_n=20, save_path="importance.png")
```

### 3. Evaluation Visualizer

**Purpose:** Create publication-quality evaluation visualizations.

#### Visualizations
- **Residual Analysis**: 2×2 grid with residuals vs predicted, distribution, Q-Q plot
- **Predictions vs Actual**: Scatter plot with perfect prediction line
- **Error Distribution**: Histogram of prediction errors
- **Platform Performance**: Bar chart comparing platforms
- **Temporal Performance**: Line chart showing performance over time
- **Learning Curve**: Training/validation scores vs dataset size

#### Usage
```python
from bufferiq.ml.evaluation.visualizer import EvaluationVisualizer

visualizer = EvaluationVisualizer()

# Residual analysis
visualizer.plot_residuals(y_true, y_pred, "residuals.png")

# Predictions vs actual
visualizer.plot_predictions_vs_actual(y_true, y_pred, "pred_vs_actual.png")

# Platform performance
visualizer.plot_platform_performance(platform_metrics, "platforms.png")
```

### 4. Model Comparator

**Purpose:** Compare multiple models side-by-side.

#### Features
- **Metric Comparison**: Compare all metrics across models
- **Platform Comparison**: Per-platform performance comparison
- **Statistical Testing**: Paired t-tests for significance
- **Best Model Selection**: Identify best model by metric

#### Usage
```python
from bufferiq.ml.evaluation.comparator import ModelComparator

comparator = ModelComparator()

models = {
    "XGBoost": xgb_trainer,
    "LightGBM": lgb_trainer,
    "RandomForest": rf_trainer
}

# Compare metrics
comparison = comparator.compare_metrics(models, X_test, y_test)

# Statistical significance
sig_test = comparator.statistical_comparison(models, X_test, y_test)

# Get best model
best = comparator.get_best_model(models, X_test, y_test, metric="r2")
print(f"Best model: {best}")
```

### 5. Performance Analyzer

**Purpose:** Deep-dive performance analysis.

#### Analyses
- **Percentile Analysis**: Performance at different engagement levels
- **Confidence Analysis**: How performance varies with prediction confidence
- **Bias Detection**: Systematic over/underestimation patterns
- **Error Correlation**: Features most correlated with errors

#### Usage
```python
from bufferiq.ml.evaluation.performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Percentile analysis
percentile_perf = analyzer.analyze_performance_by_percentile(
    y_true, y_pred, [25, 50, 75, 90, 95]
)

# Bias detection
bias = analyzer.detect_systematic_bias(y_true, y_pred)
print(f"Overall bias: {bias['overall_bias']:.4f}")
```

### 6. Error Analyzer

**Purpose:** Identify and analyze error patterns.

#### Features
- **Error Classification**: Categorize errors by severity
- **Failure Mode Detection**: Common patterns in high-error predictions
- **Error by Feature Range**: How errors vary across feature values

#### Usage
```python
from bufferiq.ml.evaluation.error_analyzer import ErrorAnalyzer

analyzer = ErrorAnalyzer()

# Classify errors
error_classes = analyzer.classify_errors(y_true, y_pred)

# Identify failure modes
failure_modes = analyzer.identify_failure_modes(
    y_test, predictions, X_test, error_threshold=0.3
)

# Error by feature ranges
error_by_length = analyzer.analyze_error_by_feature_ranges(
    errors, X_test, "text_length", n_bins=5
)
```

### 7. Model Diagnostics

**Purpose:** Diagnose model health and potential issues.

#### Diagnostics
- **Overfitting Check**: Compare train vs validation performance
- **Underfitting Check**: Verify minimum acceptable performance
- **Residual Patterns**: Test for normality, zero mean, constant variance
- **Feature Concentration**: Check if importance is too concentrated

#### Usage
```python
from bufferiq.ml.evaluation.diagnostics import ModelDiagnostics

diagnostics = ModelDiagnostics()

# Check overfitting
overfitting = diagnostics.check_overfitting(
    train_metrics, val_metrics, threshold=0.1
)

# Check underfitting
underfitting = diagnostics.check_underfitting(metrics, min_r2=0.5)

# Residual patterns
residual_check = diagnostics.check_residual_patterns(residuals)
```

## CLI Commands

### Evaluate Model
```bash
# Basic evaluation
python -m bufferiq.cli.evaluate run --model-version 1.0.0

# With report generation
python -m bufferiq.cli.evaluate run --model-version 1.0.0 --generate-report
```

### Compare Models
```bash
# Compare all registered models
python -m bufferiq.cli.evaluate compare-all
```

### Feature Importance
```bash
# Built-in importance
python -m bufferiq.cli.evaluate importance --model-version 1.0.0 --method builtin --top 20

# Permutation importance
python -m bufferiq.cli.evaluate importance --model-version 1.0.0 --method permutation --top 20
```

## Best Practices

### 1. Always Evaluate on Test Set
```python
# ✅ Good: Evaluate on held-out test set
metrics = evaluator.calculate_metrics(y_test, predictions_test)

# ❌ Bad: Evaluate on training set
metrics = evaluator.calculate_metrics(y_train, predictions_train)
```

### 2. Use Multiple Metrics
```python
# ✅ Good: Consider multiple metrics
summary = evaluator.generate_evaluation_summary(trainer, X_test, y_test)

# Check R², MAE, RMSE
print(f"R²: {summary['overall_metrics']['r2']:.4f}")
print(f"MAE: {summary['overall_metrics']['mae']:.4f}")
```

### 3. Analyze Platform Performance
```python
# ✅ Good: Check per-platform performance
platform_metrics = evaluator.evaluate_by_platform(y_test, predictions, platforms)

# Verify each platform meets minimum
for _, row in platform_metrics.iterrows():
    if row['r2'] < 0.6:
        print(f"Warning: {row['platform']} R² = {row['r2']:.4f}")
```

### 4. Check for Overfitting
```python
# ✅ Good: Always check train vs validation gap
diagnostics = ModelDiagnostics()
overfitting = diagnostics.check_overfitting(train_metrics, val_metrics)

if overfitting['is_overfitting']:
    print(f"Overfitting detected: gap = {overfitting['train_val_gap']:.4f}")
```

### 5. Visualize Results
```python
# ✅ Good: Create visualizations for stakeholders
visualizer = EvaluationVisualizer()
visualizer.plot_residuals(y_true, y_pred, "residuals.png")
visualizer.plot_predictions_vs_actual(y_true, y_pred, "pred_vs_actual.png")
```

## Performance Targets

### Minimum Acceptable (v1.0)
- **Overall R²**: > 0.60
- **MAE**: < 0.20
- **RMSE**: < 0.25
- **Platform R²**: > 0.55 for each platform

### Target (v1.1+)
- **Overall R²**: > 0.75
- **MAE**: < 0.12
- **RMSE**: < 0.16
- **Platform R²**: > 0.70 for each platform

### Current Performance (Day 10)
- **XGBoost R²**: 0.72
- **LightGBM R²**: 0.71
- **RandomForest R²**: 0.68

## Troubleshooting

### Issue: Low R² on Specific Platform

**Diagnosis:**
```python
platform_metrics = evaluator.evaluate_by_platform(y_test, predictions, platforms)
print(platform_metrics)
```

**Solutions:**
- Train platform-specific model
- Add platform-specific features
- Collect more data for that platform

### Issue: High Errors on Certain Content Types

**Diagnosis:**
```python
content_metrics = evaluator.evaluate_by_content_type(y_test, predictions, X_test)
print(content_metrics)
```

**Solutions:**
- Add content-type specific features
- Balance training data
- Use different model architecture

### Issue: Model Overfitting

**Diagnosis:**
```python
diagnostics = ModelDiagnostics()
overfitting = diagnostics.check_overfitting(train_metrics, val_metrics)
```

**Solutions:**
- Increase regularization (L1/L2)
- Reduce model complexity
- Get more training data
- Use dropout/early stopping

## Integration with Training Pipeline

Evaluation integrates with Day 8 training pipeline:

```python
from bufferiq.ml.training.pipeline import TrainingPipeline
from bufferiq.ml.evaluation.evaluator import ModelEvaluator

# Train model
config = TrainingPipelineConfig.from_yaml("config.yaml")
pipeline = TrainingPipeline(config, session)
results = await pipeline.run()

# Evaluate
evaluator = ModelEvaluator()
summary = evaluator.generate_evaluation_summary(
    trainer, X_test, y_test, platforms, timestamps
)
```

## Summary

Day 10 delivers comprehensive evaluation capabilities:
- ✅ 9+ regression metrics
- ✅ Platform/temporal/content analysis
- ✅ Feature importance (2 methods)
- ✅ Model comparison framework
- ✅ Error analysis and diagnostics
- ✅ Publication-quality visualizations
- ✅ 90%+ test coverage

These tools enable deep understanding of model performance and guide improvements for Days 11-14.