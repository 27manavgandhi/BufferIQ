#!/usr/bin/env python3
"""
Script to analyze videos using multi-modal system.

Usage:
    python scripts/analyze_video.py --video-url <URL> --platform linkedin
"""

import argparse
import asyncio
import json
from pathlib import Path

from bufferiq.ml.multimodal.videos.analyzer import VideoAnalyzer


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze video")
    parser.add_argument("--video-url", required=True, help="Video URL or path")
    parser.add_argument(
        "--platform",
        required=True,
        choices=["linkedin", "twitter", "bluesky"],
        help="Platform type"
    )
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    print(f"🎥 Analyzing video: {args.video_url}")
    print(f"📱 Platform: {args.platform}")
    print()
    
    # Initialize analyzer
    analyzer = VideoAnalyzer()
    
    # Analyze video
    try:
        result = await analyzer.analyze(
            args.video_url,
            args.platform  # type: ignore
        )
        
        print("✅ Analysis complete!")
        print()
        print(f"📊 Results:")
        print(f"   Duration: {result.metadata.duration_seconds:.1f}s")
        print(f"   Resolution: {result.metadata.resolution[0]}x{result.metadata.resolution[1]}")
        print(f"   FPS: {result.metadata.fps:.1f}")
        print(f"   Codec: {result.metadata.codec}")
        print(f"   Has audio: {result.metadata.has_audio}")
        print(f"   Keyframes: {len(result.keyframes)}")
        print(f"   Scenes: {len(result.scenes)}")
        print(f"   Engagement prediction: {result.engagement_prediction:.2%}")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
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