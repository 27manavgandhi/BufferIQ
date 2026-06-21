#!/usr/bin/env python3
"""
Script to analyze images using multi-modal system.

Usage:
    python scripts/analyze_image.py --image-url <URL> --platform linkedin
"""

import argparse
import asyncio
import json
from pathlib import Path

from bufferiq.ml.multimodal.images.analyzer import ImageAnalyzer


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze image")
    parser.add_argument("--image-url", required=True, help="Image URL or path")
    parser.add_argument(
        "--platform",
        required=True,
        choices=["linkedin", "twitter", "bluesky"],
        help="Platform type"
    )
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    print(f"🖼️  Analyzing image: {args.image_url}")
    print(f"📱 Platform: {args.platform}")
    print()
    
    # Initialize analyzer
    analyzer = ImageAnalyzer()
    
    # Analyze image
    try:
        result = await analyzer.analyze(
            args.image_url,
            args.platform  # type: ignore
        )
        
        print("✅ Analysis complete!")
        print()
        print(f"📊 Results:")
        print(f"   Objects detected: {len(result.objects)}")
        print(f"   Text elements: {len(result.text)}")
        print(f"   Faces detected: {len(result.faces)}")
        print(f"   Aesthetic score: {result.aesthetic_score:.1f}/100")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        print()
        
        # Composition scores
        print("🎨 Composition:")
        print(f"   Rule of thirds: {result.composition.rule_of_thirds:.2f}")
        print(f"   Golden ratio: {result.composition.golden_ratio:.2f}")
        print(f"   Symmetry: {result.composition.symmetry:.2f}")
        print(f"   Balance: {result.composition.balance:.2f}")
        print()
        
        # Colors
        print("🎨 Dominant colors:")
        for i, (color, pct) in enumerate(zip(
            result.colors.dominant_colors,
            result.colors.color_percentages
        )):
            print(f"   {i+1}. RGB{tuple(color)} ({pct*100:.1f}%)")
        print()
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(result.to_dict(), indent=2))
            print(f"💾 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))