"""
Create experiment script.

Example usage:
    python scripts/create_experiment.py \
        --name "Headline Test" \
        --platform linkedin \
        --baseline-rate 0.05 \
        --mde 0.10
"""

import argparse
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.experiments.intelligence.service import ExperimentIntelligenceService
from bufferiq.ml.experiments.design.designer import Variant, MetricType


async def main():
    """Create experiment."""
    parser = argparse.ArgumentParser(description="Create experiment")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--platform", required=True, choices=["linkedin", "twitter", "bluesky"])
    parser.add_argument("--baseline-rate", type=float, required=True, help="Baseline rate")
    parser.add_argument("--mde", type=float, default=0.10, help="Minimum detectable effect")
    parser.add_argument("--traffic-split", type=float, default=0.5, help="Traffic to treatment")
    
    args = parser.parse_args()
    
    # Setup database
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create service
    service = ExperimentIntelligenceService(session)
    
    # Define variants
    variants = [
        Variant(
            id="control",
            name="Control",
            description="Original version",
            traffic_allocation=1 - args.traffic_split,
            changes={},
            is_control=True
        ),
        Variant(
            id="treatment",
            name="Treatment",
            description="New version",
            traffic_allocation=args.traffic_split,
            changes={"version": "new"}
        )
    ]
    
    # Create experiment
    experiment = await service.create_experiment(
        name=args.name,
        description=f"A/B test on {args.platform}",
        variants=variants,
        platform=args.platform,
        primary_metric=MetricType.ENGAGEMENT_RATE,
        baseline_rate=args.baseline_rate,
        mde=args.mde
    )
    
    print(f"✓ Experiment created!")
    print(f"  ID: {experiment.experiment_id}")
    print(f"  Name: {experiment.name}")
    print(f"  Platform: {experiment.platform}")
    print(f"  Required sample size: {experiment.required_sample_size:,} per variant")
    print(f"  Estimated duration: {experiment.estimated_duration_days} days")


if __name__ == "__main__":
    asyncio.run(main())