"""
Monitor experiments script.

Example usage:
    python scripts/monitor_experiments.py --active-only
"""

import argparse
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.experiments.intelligence.service import ExperimentIntelligenceService


async def main():
    """Monitor experiments."""
    parser = argparse.ArgumentParser(description="Monitor experiments")
    parser.add_argument("--active-only", action="store_true", help="Only active experiments")
    
    args = parser.parse_args()
    
    # Setup database
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create service
    service = ExperimentIntelligenceService(session)
    
    # List experiments
    experiments = service.list_experiments()
    
    if not experiments:
        print("No experiments found")
        return
    
    print(f"Found {len(experiments)} experiments\n")
    
    for exp in experiments:
        print(f"Experiment: {exp.name}")
        print(f"  ID: {exp.experiment_id}")
        print(f"  Platform: {exp.platform}")
        print(f"  Created: {exp.created_at.strftime('%Y-%m-%d %H:%M')}")
        
        # Check health
        health = await service.monitor_experiment(exp.experiment_id)
        
        if health["is_healthy"]:
            print(f"  Status: ✓ Healthy")
        else:
            print(f"  Status: ✗ Issues detected")
            for issue in health["issues"]:
                print(f"    - {issue}")
        
        print()


if __name__ == "__main__":
    asyncio.run(main())