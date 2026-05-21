"""
Analyze experiment script.

Example usage:
    python scripts/analyze_experiment.py --experiment-id exp_20240120_120000
"""

import argparse
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.experiments.intelligence.service import ExperimentIntelligenceService


async def main():
    """Analyze experiment."""
    parser = argparse.ArgumentParser(description="Analyze experiment")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    
    args = parser.parse_args()
    
    # Setup database
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create service
    service = ExperimentIntelligenceService(session)
    
    # Analyze
    results = await service.analyze_experiment(experiment_id=args.experiment_id)
    
    if results["status"] == "insufficient_data":
        print(f"✗ Insufficient data")
        print(f"  Control: {results['n_control']} samples")
        print(f"  Treatment: {results['n_treatment']} samples")
        print(f"  Required: {results['required']} samples")
        return
    
    print(f"✓ Analysis complete!")
    print(f"  Winner: {results['winner_variant'] or 'No clear winner'}")
    print(f"  Confidence: {results['confidence']:.1%}")
    print(f"  P-value: {results['statistical_result']['p_value']:.4f}")
    print(f"  Effect size: {results['statistical_result']['effect_size']:.3f}")
    print(f"  Relative improvement: {results['statistical_result']['relative_diff']:.2%}")
    print(f"  Should launch: {'✓ Yes' if results['should_launch'] else '✗ No'}")
    print(f"\nRecommendation: {results['recommendation']}")


if __name__ == "__main__":
    asyncio.run(main())