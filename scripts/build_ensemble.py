"""Standalone script to build ensemble models."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bufferiq.core.logging import get_logger
from bufferiq.ml.ensemble.ensemble_builder import EnsembleBuilder

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build ensemble models from configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ensemble configuration YAML",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data (NPZ file with X_train, y_train)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Path to validation data (NPZ file with X_val, y_val)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data (NPZ file with X_test, y_test) - optional",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/models/ensembles",
        help="Output directory for ensemble",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded: {config.get('model_name', 'Unknown')}")
    return config


def load_data(data_path: str):
    """Load data from NPZ file."""
    logger.info(f"Loading data from {data_path}")
    
    data = np.load(data_path)
    
    if "X_train" in data and "y_train" in data:
        X = data["X_train"]
        y = data["y_train"]
    elif "X_val" in data and "y_val" in data:
        X = data["X_val"]
        y = data["y_val"]
    elif "X_test" in data and "y_test" in data:
        X = data["X_test"]
        y = data["y_test"]
    else:
        raise ValueError(f"Unknown data format in {data_path}")
    
    logger.info(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    return X, y


def evaluate_ensemble(ensemble, X, y, dataset_name="Dataset"):
    """Evaluate ensemble on given data."""
    logger.info(f"Evaluating ensemble on {dataset_name}")
    
    predictions = ensemble.predict(X)
    
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    logger.info(f"{dataset_name} - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }


def main():
    """Main execution function."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("BufferIQ Ensemble Builder")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    X_train, y_train = load_data(args.train_data)
    X_val, y_val = load_data(args.val_data)
    
    # Build ensemble
    logger.info("Building ensemble...")
    
    builder = EnsembleBuilder(
        model_paths=config["base_models"],
        ensemble_type=config["ensemble_type"],
        min_performance=config.get("selection", {}).get("min_performance", 0.70),
        min_diversity=config.get("selection", {}).get("min_diversity", 0.10),
        max_models=config.get("selection", {}).get("max_models", 5),
        weight_optimization=config.get("weight_optimization", "performance"),
        output_dir=Path(args.output_dir),
    )
    
    ensemble = builder.build(X_train, y_train, X_val, y_val)
    
    logger.info("Ensemble built successfully!")
    
    # Evaluate on validation set
    val_metrics = evaluate_ensemble(ensemble, X_val, y_val, "Validation")
    
    # Evaluate on test set if provided
    test_metrics = None
    if args.test_data:
        X_test, y_test = load_data(args.test_data)
        test_metrics = evaluate_ensemble(ensemble, X_test, y_test, "Test")
    
    # Save ensemble
    output_path = Path(args.output_dir) / f"{config['model_name']}.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ensemble.save(output_path)
    logger.info(f"Ensemble saved to {output_path}")
    
    # Save metadata
    metadata = {
        "model_name": config["model_name"],
        "ensemble_type": config["ensemble_type"],
        "version": config.get("version", "1.0.0"),
        "description": config.get("description", ""),
        "base_models": config["base_models"],
        "validation_metrics": val_metrics,
    }
    
    if test_metrics:
        metadata["test_metrics"] = test_metrics
    
    metadata_path = output_path.parent / f"{config['model_name']}_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ENSEMBLE BUILD SUMMARY")
    print("=" * 80)
    print(f"Model Name: {config['model_name']}")
    print(f"Ensemble Type: {config['ensemble_type']}")
    print(f"Number of Base Models: {len(config['base_models'])}")
    print(f"\nValidation Metrics:")
    print(f"  R²: {val_metrics['r2']:.4f}")
    print(f"  MAE: {val_metrics['mae']:.4f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    
    if test_metrics:
        print(f"\nTest Metrics:")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
    
    print(f"\nModel saved to: {output_path}")
    print("=" * 80)
    
    logger.info("Ensemble building completed successfully!")


if __name__ == "__main__":
    main()