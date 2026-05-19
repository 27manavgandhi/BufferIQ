#!/usr/bin/env python3
"""
Hashtag analysis script.

Analyzes hashtags from CSV or JSON input.
"""

import asyncio
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bufferiq.ml.hashtags.intelligence.service import HashtagIntelligenceService


async def analyze_from_file(
    input_file: str,
    platform: str,
    output_file: str | None = None,
) -> None:
    """
    Analyze hashtags from input file.

    Args:
        input_file: Path to input file (CSV or JSON)
        platform: Platform name
        output_file: Optional output file path
    """
    # Setup database
    engine = create_engine("sqlite:///./bufferiq.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Initialize service
    service = HashtagIntelligenceService(db_session=session)

    # Read input
    hashtags = read_input_file(input_file)

    print(f"Analyzing {len(hashtags)} hashtags on {platform}...")

    # Analyze each hashtag
    results: List[Dict[str, Any]] = []

    for i, hashtag in enumerate(hashtags, 1):
        print(f"  [{i}/{len(hashtags)}] Analyzing #{hashtag}...")

        try:
            analysis = await service.analyze_hashtag(
                hashtag=hashtag,
                platform=platform,
            )
            results.append(analysis)
        except Exception as e:
            print(f"    Error: {e}")
            results.append({"hashtag": hashtag, "error": str(e)})

    print(f"\n✓ Analysis complete!")

    # Output results
    if output_file:
        write_output_file(results, output_file)
        print(f"✓ Results written to {output_file}")
    else:
        # Print summary
        print_summary(results)


def read_input_file(file_path: str) -> List[str]:
    """Read hashtags from input file."""
    path = Path(file_path)

    if path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "hashtags" in data:
                return data["hashtags"]
            else:
                raise ValueError("JSON must be array or object with 'hashtags' key")

    elif path.suffix == ".csv":
        hashtags = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "hashtag" in row:
                    hashtags.append(row["hashtag"])
        return hashtags

    else:
        # Plain text, one per line
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]


def write_output_file(results: List[Dict[str, Any]], file_path: str) -> None:
    """Write results to output file."""
    path = Path(file_path)

    if path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    elif path.suffix == ".csv":
        if not results:
            return

        # Flatten results for CSV
        fieldnames = ["hashtag", "platform", "avg_engagement", "risk_level", "trend_direction"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                if "error" not in result:
                    writer.writerow({
                        "hashtag": result["hashtag"],
                        "platform": result["platform"],
                        "avg_engagement": result["performance"]["avg_engagement"],
                        "risk_level": result["risk"]["risk_level"],
                        "trend_direction": result["performance"]["trend_direction"],
                    })


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\nTotal analyzed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nTop Performers:")
        sorted_results = sorted(
            successful,
            key=lambda x: x["performance"]["avg_engagement"],
            reverse=True,
        )

        for i, result in enumerate(sorted_results[:5], 1):
            print(f"\n{i}. #{result['hashtag']}")
            print(f"   Avg Engagement: {result['performance']['avg_engagement']:.1f}")
            print(f"   Engagement Lift: {result['performance']['engagement_lift']:.1%}")
            print(f"   Risk Level: {result['risk']['risk_level']}")
            print(f"   Trend: {result['performance']['trend_direction']}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze hashtags from input file"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input file (CSV, JSON, or text)",
    )
    parser.add_argument(
        "--platform",
        "-p",
        required=True,
        choices=["linkedin", "twitter", "bluesky"],
        help="Platform name",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (optional)",
    )

    args = parser.parse_args()

    asyncio.run(
        analyze_from_file(
            input_file=args.input,
            platform=args.platform,
            output_file=args.output,
        )
    )


if __name__ == "__main__":
    main()