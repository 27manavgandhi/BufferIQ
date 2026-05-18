"""
Batch gap analysis script.

Analyzes content gaps for multiple users from CSV file.
"""

import asyncio
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.gaps.intelligence.service import GapIntelligenceService


async def analyze_user(
    service: GapIntelligenceService,
    user_id: str,
    platform: str,
    competitor_ids: List[str] = None,
    industry: str = None,
) -> Dict[str, Any]:
    """
    Analyze single user.

    Args:
        service: Gap intelligence service
        user_id: User identifier
        platform: Platform
        competitor_ids: Optional competitors
        industry: Optional industry

    Returns:
        Analysis results
    """
    try:
        print(f"Analyzing {user_id}...")

        report = await service.analyze_gaps(
            user_id=user_id,
            platform=platform,
            competitor_ids=competitor_ids,
            industry=industry,
            include_recommendations=True,
        )

        return {
            "user_id": user_id,
            "status": "success",
            "coverage_score": report["coverage_score"],
            "total_gaps": report["total_gaps"],
            "critical_gaps": len(report["critical_gaps"]),
            "recommendations": len(report["recommendations"]),
        }

    except Exception as e:
        print(f"Error analyzing {user_id}: {e}")
        return {
            "user_id": user_id,
            "status": "error",
            "error": str(e),
        }


async def main(input_file: str, output_dir: str):
    """
    Main batch analysis function.

    Args:
        input_file: Path to input CSV file
        output_dir: Output directory for reports
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize service
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    service = GapIntelligenceService(db_session=session)

    # Read input CSV
    users = []
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.append({
                "user_id": row["user_id"],
                "platform": row.get("platform", "linkedin"),
                "competitor_ids": row.get("competitors", "").split(",") if row.get("competitors") else None,
                "industry": row.get("industry"),
            })

    print(f"Found {len(users)} users to analyze")

    # Analyze all users
    results = []
    for user_data in users:
        result = await analyze_user(
            service=service,
            user_id=user_data["user_id"],
            platform=user_data["platform"],
            competitor_ids=user_data["competitor_ids"],
            industry=user_data["industry"],
        )
        results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"gap_analysis_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print("Batch Analysis Complete")
    print(f"{'='*60}")
    print(f"Total users: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_file}")

    # Print top insights
    if successful > 0:
        avg_coverage = sum(
            r.get("coverage_score", 0)
            for r in results
            if r["status"] == "success"
        ) / successful

        print(f"\nAverage coverage score: {avg_coverage:.1f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python analyze_gaps.py <input_csv> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    asyncio.run(main(input_file, output_dir))