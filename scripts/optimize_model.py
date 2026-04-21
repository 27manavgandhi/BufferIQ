"""Standalone script for hyperparameter optimization."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bufferiq.core.logging import get_logger
from bufferiq.ml.optimization.config_schema import OptimizationConfig
from bufferiq.ml.optimization.pipeline import OptimizationPipeline

logger = get_logger(__name__)


async def main():
    """Run optimization pipeline."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for BufferIQ models"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to optimization config YAML file",
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = OptimizationConfig.from_yaml(args.config)
        
        logger.info("=" * 80)
        logger.info("HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Config: {args.config}")
        logger.info(f"Model: {config.model_type}")
        logger.info(f"Strategy: {config.strategy}")
        logger.info(f"CV folds: {config.cv_folds}")
        if config.n_iter:
            logger.info(f"Iterations: {config.n_iter}")
        
        # Create and run pipeline
        pipeline = OptimizationPipeline(config)
        results = await pipeline.run()
        
        # Print summary
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Best score: {results['best_score']:.4f}")
        print(f"Best params: {results['best_params']}")
        print(f"Total trials: {results['total_trials']}")
        print(f"Time taken: {results['optimization_time']:.2f}s")
        print(f"Results saved to: {results['report_path']}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())