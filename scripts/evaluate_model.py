"""Script to evaluate trained models."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from bufferiq.core.logging import get_logger
from bufferiq.ml.training.model_registry import ModelRegistry

logger = get_logger(__name__)


async def evaluate_model(model_version: str, verbose: bool = False) -> None:
    """
    Evaluate a trained model.

    Args:
        model_version: Model version to evaluate
        verbose: Whether to print verbose output
    """
    logger.info(f"Evaluating model version: {model_version}")

    # Load model from registry
    registry = ModelRegistry()

    try:
        model_info = registry.get_model(version=model_version)

        if verbose:
            print(f"\nModel Information:")
            print(f"  Version: {model_info['version']}")
            print(f"  Model ID: {model_info['model_id']}")
            print(f"  Registered: {model_info['registered_at']}")
            print(f"\nMetrics:")
            for metric, value in model_info['metrics'].items():
                print(f"  {metric}: {value:.4f}")

        print("\n✅ Model evaluation complete!")

    except ValueError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model-version", required=True, help="Model version to evaluate"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--compare-all", action="store_true", help="Compare all models")

    args = parser.parse_args()

    try:
        if args.compare_all:
            registry = ModelRegistry()
            models = registry.list_models()
            
            print(f"\nFound {len(models)} registered models:")
            for model in models:
                print(f"\n  {model['version']}")
                print(f"    R²: {model['metrics'].get('r2', 0):.4f}")
                print(f"    MAE: {model['metrics'].get('mae', 0):.4f}")
        else:
            asyncio.run(evaluate_model(args.model_version, args.verbose))
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()