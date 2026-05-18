"""
Competitor benchmarking script.

Benchmarks multiple users against their competitors.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.gaps.intelligence.service import GapIntelligenceService


async def benchmark_user(
    service: GapIntelligenceService,
    user_id: str,
    competitor_ids: List[str],
    platform: str,
) -> Dict[str, Any]:
    """
    Benchmark single user against competitors.

    Args:
        service: Gap intelligence service
        user_id: User identifier
        competitor_ids: Competitor IDs
        platform: Platform

    Returns:
        Benchmark results
    """
    try:
        print(f"Benchmarking {user_id} vs {len(competitor_ids)} competitors...")

        analysis = await service.benchmark_competitors(
            user_id=user_id,
            competitor_ids=competitor_ids,
            platform=platform,
        )

        comp_analysis = analysis["competitive_analysis"]

        # Extract key metrics
        result = {
            "user_id": user_id,
            "status": "success",
            "rank": comp_analysis["user_rank"],
            "total_competitors": len(competitor_ids),
            "share_of_voice": comp_analysis["share_of_voice"],
            "engagement_vs_avg": comp_analysis["engagement_vs_avg"],
            "unique_topics": len(comp_analysis["unique_topics"]),
            "missed_topics": len(comp_analysis["missed_topics"]),
            "competitive_position": "leader" if comp_analysis["user_rank"] == 1 else "challenger",
        }

        print(f"  Rank: {result['rank']}/{len(competitor_ids)+1}")
        print(f"  Share of Voice: {result['share_of_voice']:.1f}%")

        return result

    except Exception as e:
        print(f"  Error: {e}")
        return {
            "user_id": user_id,
            "status": "error",
            "error": str(e),
        }


async def main(benchmarks: List[Dict[str, Any]], platform: str = "linkedin"):
    """
    Main benchmarking function.

    Args:
        benchmarks: List of benchmark configs
        platform: Platform
    """
    # Initialize service
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    service = GapIntelligenceService(db_session=session)

    print(f"Running {len(benchmarks)} benchmarks on {platform}")
    print(f"{'='*60}\n")

    # Run benchmarks
    results = []
    for config in benchmarks:
        result = await benchmark_user(
            service=service,
            user_id=config["user_id"],
            competitor_ids=config["competitors"],
            platform=platform,
        )
        results.append(result)
        print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"outputs/benchmarks/benchmark_{timestamp}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    leaders = sum(1 for r in results if r.get("rank") == 1)

    print(f"{'='*60}")
    print("Benchmarking Complete")
    print(f"{'='*60}")
    print(f"Total benchmarks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Market leaders: {leaders}")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    # Example benchmark configuration
    benchmarks = [
        {
            "user_id": "brand1",
            "competitors": ["comp1", "comp2", "comp3"],
        },
        {
            "user_id": "brand2",
            "competitors": ["comp4", "comp5"],
        },
    ]

    asyncio.run(main(
        benchmarks=benchmarks,
        platform="linkedin",
    ))