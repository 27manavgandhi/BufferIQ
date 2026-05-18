"""
Content calendar generation script.

Generates content calendars for multiple brands.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.gaps.intelligence.service import GapIntelligenceService


async def generate_calendar_for_user(
    service: GapIntelligenceService,
    user_id: str,
    platform: str,
    weeks: int = 4,
    posts_per_week: int = 3,
) -> Dict[str, Any]:
    """
    Generate calendar for single user.

    Args:
        service: Gap intelligence service
        user_id: User identifier
        platform: Platform
        weeks: Number of weeks
        posts_per_week: Posts per week

    Returns:
        Calendar data
    """
    try:
        print(f"Generating calendar for {user_id}...")

        calendar = await service.generate_calendar(
            user_id=user_id,
            platform=platform,
            weeks=weeks,
            posts_per_week=posts_per_week,
        )

        # Save individual calendar
        output_file = Path(f"outputs/calendars/{user_id}_{platform}_calendar.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(calendar, f, indent=2)

        print(f"  ✓ Calendar saved: {output_file}")

        return {
            "user_id": user_id,
            "status": "success",
            "total_pieces": calendar["total_pieces"],
            "file": str(output_file),
        }

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            "user_id": user_id,
            "status": "error",
            "error": str(e),
        }


async def main(
    user_ids: list,
    platform: str = "linkedin",
    weeks: int = 4,
    posts_per_week: int = 3,
):
    """
    Main calendar generation function.

    Args:
        user_ids: List of user IDs
        platform: Target platform
        weeks: Number of weeks
        posts_per_week: Posts per week
    """
    # Initialize service
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    service = GapIntelligenceService(db_session=session)

    print(f"Generating calendars for {len(user_ids)} users")
    print(f"Platform: {platform}")
    print(f"Duration: {weeks} weeks")
    print(f"Frequency: {posts_per_week} posts/week")
    print(f"{'='*60}\n")

    # Generate calendars
    results = []
    for user_id in user_ids:
        result = await generate_calendar_for_user(
            service=service,
            user_id=user_id,
            platform=platform,
            weeks=weeks,
            posts_per_week=posts_per_week,
        )
        results.append(result)

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print("Calendar Generation Complete")
    print(f"{'='*60}")
    print(f"Total users: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_calendar.py <user_id1> [user_id2] ...")
        sys.exit(1)

    user_ids = sys.argv[1:]

    asyncio.run(main(
        user_ids=user_ids,
        platform="linkedin",
        weeks=4,
        posts_per_week=3,
    ))