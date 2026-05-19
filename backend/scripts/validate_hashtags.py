#!/usr/bin/env python3
"""
Validate hashtags script.

Checks hashtags for safety and brand compliance.
"""

import asyncio
import argparse
from typing import List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.hashtags.intelligence.service import HashtagIntelligenceService


async def validate_hashtags(
    hashtags: List[str],
    platform: str,
) -> None:
    """
    Validate hashtags.

    Args:
        hashtags: List of hashtags to validate
        platform: Platform name
    """
    # Setup database
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Initialize service
    service = HashtagIntelligenceService(db_session=session)

    print(f"Validating {len(hashtags)} hashtags on {platform}...\n")

    # Validate
    validation = await service.validate_hashtags(
        hashtags=hashtags,
        platform=platform,
    )

    # Display results
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    safe_count = 0
    unsafe_count = 0

    for hashtag, risk in validation.items():
        is_safe = risk.risk_level in ["none", "low"]

        if is_safe:
            safe_count += 1
            icon = "✓"
        else:
            unsafe_count += 1
            icon = "✗"

        print(f"\n{icon} #{hashtag}")
        print(f"   Risk Level: {risk.risk_level}")
        print(f"   Recommendation: {risk.recommendation}")

        if risk.risk_reasons:
            print(f"   Reasons:")
            for reason in risk.risk_reasons:
                print(f"     - {reason}")

        if risk.alternatives:
            print(f"   Alternatives: {', '.join('#' + a for a in risk.alternatives)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal: {len(hashtags)}")
    print(f"Safe: {safe_count} ({safe_count/len(hashtags)*100:.1f}%)")
    print(f"Unsafe: {unsafe_count} ({unsafe_count/len(hashtags)*100:.1f}%)")

    if unsafe_count > 0:
        print(f"\n⚠ Warning: {unsafe_count} hashtag(s) flagged as unsafe")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate hashtags for safety"
    )
    parser.add_argument(
        "--hashtags",
        "-t",
        required=True,
        help="Comma-separated hashtags (without #)",
    )
    parser.add_argument(
        "--platform",
        "-p",
        required=True,
        choices=["linkedin", "twitter", "bluesky"],
        help="Platform name",
    )

    args = parser.parse_args()

    # Parse hashtags
    hashtags = [h.strip() for h in args.hashtags.split(",")]

    asyncio.run(
        validate_hashtags(
            hashtags=hashtags,
            platform=args.platform,
        )
    )


if __name__ == "__main__":
    main()