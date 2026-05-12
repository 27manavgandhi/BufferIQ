"""
Content analysis script.

Analyzes content from CSV file and outputs results.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from bufferiq.ml.content.intelligence.service import ContentIntelligenceService


def load_posts_from_csv(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load posts from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        List of posts

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    posts = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "text" not in row:
                raise ValueError("CSV must have 'text' column")
            posts.append(row)

    return posts


def analyze_posts(
    posts: List[Dict[str, Any]], platform: str
) -> List[Dict[str, Any]]:
    """
    Analyze posts.

    Args:
        posts: List of posts
        platform: Platform type

    Returns:
        List of analysis results
    """
    service = ContentIntelligenceService()
    results = []

    for i, post in enumerate(posts, 1):
        print(f"Analyzing post {i}/{len(posts)}...", file=sys.stderr)
        text = post.get("text", "")
        if text:
            try:
                result = service.analyze_content(text, platform)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing post {i}: {e}", file=sys.stderr)
                results.append({"text": text, "error": str(e)})

    return results


def save_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save results to JSON file.

    Args:
        results: Analysis results
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze content from CSV file"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input CSV file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSON file"
    )
    parser.add_argument(
        "--platform",
        choices=["linkedin", "twitter", "bluesky"],
        default="linkedin",
        help="Platform type",
    )

    args = parser.parse_args()

    try:
        # Load posts
        print(f"Loading posts from {args.input}...", file=sys.stderr)
        posts = load_posts_from_csv(args.input)
        print(f"Loaded {len(posts)} posts", file=sys.stderr)

        # Analyze posts
        print(f"Analyzing for platform: {args.platform}", file=sys.stderr)
        results = analyze_posts(posts, args.platform)

        # Save results
        print(f"Saving results to {args.output}...", file=sys.stderr)
        save_results(results, args.output)

        print("Analysis complete!", file=sys.stderr)
        print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()