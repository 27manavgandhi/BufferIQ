"""
Generate experiment report script.

Example usage:
    python scripts/generate_experiment_report.py \
        --experiment-id exp_20240120_120000 \
        --output reports/
"""

import argparse
import asyncio
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.experiments.intelligence.service import ExperimentIntelligenceService


async def main():
    """Generate experiment report."""
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID")
    parser.add_argument("--output", default="reports/", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup database
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create service
    service = ExperimentIntelligenceService(session)
    
    # Analyze
    results = await service.analyze_experiment(experiment_id=args.experiment_id)
    
    if results["status"] != "complete":
        print(f"✗ Cannot generate report: {results['status']}")
        return
    
    # Get report
    report = results["report"]
    
    # Export to markdown
    markdown = service.report_generator.export_to_markdown(report)
    
    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{args.experiment_id}_report.md"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        f.write(markdown)
    
    print(f"✓ Report saved to: {filepath}")


if __name__ == "__main__":
    asyncio.run(main())