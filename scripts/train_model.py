"""Script to train engagement prediction model."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from bufferiq.core.database import async_session_maker
from bufferiq.core.logging import get_logger
from bufferiq.ml.training.config_schema import TrainingPipelineConfig
from bufferiq.ml.training.pipeline import TrainingPipeline

logger = get_logger(__name__)


async def train_model(config_path: str, verbose: bool = False) -> None:
    """
    Train engagement prediction model.

    Args:
        config_path: Path to training config YAML
        verbose: Whether to print verbose output
    """
    logger.info(f"Loading config from {config_path}")

    # Load config
    config = TrainingPipelineConfig.from_yaml(config_path)

    if verbose:
        print(f"\nTraining Configuration:")
        print(f"  Experiment: {config.experiment.experiment_name}")
        print(f"  Model: {config.model.model_type}")
        print(f"  Platforms: {config.data.platforms}")
        print(f"  Test size: {config.data.test_size}")
        print()

    # Create training pipeline
    async with async_session_maker() as session:
        pipeline = TrainingPipeline(config, session)

        # Run training
        if config.experiment.use_cross_validation:
            logger.info("Running with cross-validation")
            results = await pipeline.run_with_cross_validation()
        else:
            logger.info("Running single training")
            results = await pipeline.run()

    # Print results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nExperiment: {results['experiment_name']}")
    print(f"Experiment Dir: {results['experiment_dir']}")

    if "model_id" in results:
        print(f"Model ID: {results['model_id']}")

    if "test_metrics" in results:
        print(f"\nTest Metrics:")
        for metric, value in results["test_metrics"].items():
            print(f"  {metric}: {value:.4f}")

    if "data_stats" in results:
        print(f"\nData Statistics:")
        for key, value in results["data_stats"].items():
            print(f"  {key}: {value}")

    print(f"\nDuration: {results.get('training_duration', 0):.2f}s")
    print("\n✅ Training successful!")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train engagement prediction model")
    parser.add_argument(
        "--config", required=True, help="Path to training config YAML"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    try:
        asyncio.run(train_model(args.config, args.verbose))
    except Exception as e:
        print(f"\n❌ Training failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()