#!/usr/bin/env python3
"""
Script to optimize link previews.

Usage:
    python scripts/optimize_links.py --url <URL> --platform linkedin
"""

import argparse
import asyncio
import json
from pathlib import Path

from bufferiq.ml.multimodal.links.analyzer import LinkPreviewAnalyzer


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Optimize link preview")
    parser.add_argument("--url", required=True, help="URL to analyze")
    parser.add_argument(
        "--platform",
        required=True,
        choices=["linkedin", "twitter", "bluesky"],
        help="Platform type"
    )
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    print(f"🔗 Analyzing URL: {args.url}")
    print(f"📱 Platform: {args.platform}")
    print()
    
    # Initialize analyzer
    analyzer = LinkPreviewAnalyzer()
    
    # Analyze link
    try:
        result = await analyzer.analyze(
            args.url,
            args.platform  # type: ignore
        )
        
        print("✅ Analysis complete!")
        print()
        print(f"📊 Results:")
        print(f"   Title: {result.metadata.title or 'N/A'}")
        print(f"   Description: {result.metadata.description[:100] if result.metadata.description else 'N/A'}...")
        print(f"   Image: {result.metadata.image_url or 'N/A'}")
        print()
        
        print(f"📈 Quality Scores:")
        print(f"   Title: {result.quality_scores.title_quality:.1f}/100")
        print(f"   Description: {result.quality_scores.description_quality:.1f}/100")
        print(f"   Image: {result.quality_scores.image_quality:.1f}/100")
        print(f"   Overall: {result.quality_scores.overall_quality:.1f}/100")
        print()
        
        print(f"🎯 CTR Prediction: {result.ctr_prediction:.2%}")
        print()
        
        if result.optimization_suggestions:
            print("💡 Optimization Suggestions:")
            for i, suggestion in enumerate(result.optimization_suggestions, 1):
                print(f"   {i}. {suggestion}")
            print()
        
        print(f"⏱️  Processing time: {result.processing_time_ms:.1f}ms")
        
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