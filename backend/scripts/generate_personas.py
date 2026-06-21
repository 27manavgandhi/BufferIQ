"""Generate personas from audience data."""

import asyncio
import json
from pathlib import Path
from datetime import datetime

from bufferiq.ml.segmentation.intelligence.service import SegmentationIntelligenceService
from bufferiq.ml.segmentation.types import AudienceDataPoint


async def main(input_file: str, output_file: str) -> None:
    """
    Generate personas from segmentation.

    Args:
        input_file: Path to segmentation result
        output_file: Path to output personas
    """
    print(f"[{datetime.now()}] Generating personas...")

    # Load segmentation result
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_path) as f:
        segmentation_result = json.load(f)

    personas = segmentation_result.get("personas", [])
    print(f"  Found {len(personas)} personas")

    # Generate detailed persona profiles
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "platform": segmentation_result.get("platform"),
        "n_personas": len(personas),
        "personas": [],
    }

    for persona in personas:
        persona_profile = {
            "id": persona["segment_id"],
            "name": persona["persona_name"],
            "description": persona["persona_description"],
            "size": persona["size"],
            "size_percentage": persona["size_percentage"],
            "demographics": {
                "age_range": persona["estimated_age_range"],
                "location": persona["estimated_location"],
            },
            "engagement": {
                "avg_rate": persona["avg_engagement_rate"],
                "potential_score": persona["engagement_potential_score"],
            },
            "interests": {
                "primary": persona["primary_topics"],
                "secondary": persona["secondary_topics"],
            },
            "recommendations": {
                "content_types": persona["content_type_preferences"],
                "tone": persona["recommended_tone"],
                "posting_times": persona["optimal_posting_times"],
            },
        }

        output_data["personas"].append(persona_profile)

    # Save personas
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  ✓ Saved {len(personas)} personas to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate personas")
    parser.add_argument("input_file", help="Segmentation result file")
    parser.add_argument(
        "--output",
        default="outputs/personas.json",
        help="Output file",
    )

    args = parser.parse_args()
    asyncio.run(main(args.input_file, args.output))