"""Analyze segmentation quality and insights."""

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np


def analyze_segments(segmentation_result: dict) -> dict:
    """
    Analyze segmentation quality.

    Args:
        segmentation_result: Segmentation result dict

    Returns:
        Analysis results
    """
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "clustering_quality": {},
        "segment_analysis": {},
        "persona_diversity": {},
        "recommendations": [],
    }

    # Clustering quality analysis
    quality = segmentation_result.get("clustering_quality", {})
    analysis["clustering_quality"] = {
        "silhouette_score": quality.get("silhouette_score"),
        "quality_assessment": (
            "Excellent"
            if quality.get("silhouette_score", 0) > 0.8
            else "Good"
            if quality.get("silhouette_score", 0) > 0.6
            else "Fair"
            if quality.get("silhouette_score", 0) > 0.4
            else "Poor"
        ),
    }

    # Segment analysis
    personas = segmentation_result.get("personas", [])
    sizes = [p["size"] for p in personas]
    engagement_rates = [p["avg_engagement_rate"] for p in personas]

    if sizes:
        analysis["segment_analysis"] = {
            "n_segments": len(personas),
            "avg_segment_size": np.mean(sizes),
            "segment_size_std": np.std(sizes),
            "largest_segment": max(sizes),
            "smallest_segment": min(sizes),
            "avg_engagement": np.mean(engagement_rates),
            "engagement_std": np.std(engagement_rates),
            "high_engagement_segments": sum(1 for e in engagement_rates if e > 0.5),
            "low_engagement_segments": sum(1 for e in engagement_rates if e < 0.2),
        }

    # Persona diversity
    all_topics = []
    for persona in personas:
        all_topics.extend(persona.get("primary_topics", []))

    unique_topics = set(all_topics)
    analysis["persona_diversity"] = {
        "unique_topics": len(unique_topics),
        "topics": list(unique_topics),
        "topic_distribution": {
            topic: all_topics.count(topic) for topic in unique_topics
        },
    }

    # Recommendations
    recommendations = []

    if quality.get("silhouette_score", 0) < 0.65:
        recommendations.append(
            "Silhouette score below 0.65. Consider adjusting preprocessing or clustering parameters."
        )

    if analysis["segment_analysis"].get("segment_size_std", 0) > np.mean(sizes):
        recommendations.append(
            "High variance in segment sizes. Consider re-clustering or filtering."
        )

    high_eng = analysis["segment_analysis"].get("high_engagement_segments", 0)
    if high_eng == 0:
        recommendations.append(
            "No high-engagement segments found. Review audience data or engagement metrics."
        )

    analysis["recommendations"] = recommendations

    return analysis


def main(input_file: str, output_file: str) -> None:
    """
    Analyze segmentation results.

    Args:
        input_file: Path to segmentation result
        output_file: Path to output analysis
    """
    print(f"[{datetime.now()}] Analyzing segmentation...")

    # Load segmentation result
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_path) as f:
        segmentation_result = json.load(f)

    # Analyze
    analysis = analyze_segments(segmentation_result)

    # Save analysis
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Print summary
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Clustering Quality: {analysis['clustering_quality']['quality_assessment']}")
    print(f"  Silhouette Score: {analysis['clustering_quality']['silhouette_score']:.3f}")

    if analysis["segment_analysis"]:
        print(f"\nSegment Analysis:")
        print(f"  N Segments: {analysis['segment_analysis']['n_segments']}")
        print(f"  Avg Size: {analysis['segment_analysis']['avg_segment_size']:.0f}")
        print(f"  Avg Engagement: {analysis['segment_analysis']['avg_engagement']:.1%}")

    if analysis["recommendations"]:
        print(f"\nRecommendations:")
        for rec in analysis["recommendations"]:
            print(f"  • {rec}")

    print(f"\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze segmentation results")
    parser.add_argument("input_file", help="Segmentation result file")
    parser.add_argument(
        "--output",
        default="outputs/analysis.json",
        help="Output file",
    )

    args = parser.parse_args()
    main(args.input_file, args.output)