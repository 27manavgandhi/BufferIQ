"""
Voice extraction script.

Extract voice profiles from historical content via CLI.
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from bufferiq.core.database import SessionLocal
from bufferiq.ml.voice.intelligence.service import VoiceIntelligenceService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def extract_voice_profile(
    brand_id: str,
    platform: str,
    lookback_days: int,
    output_file: str = None,
):
    """
    Extract voice profile for a brand.
    
    Args:
        brand_id: Brand identifier
        platform: Platform (linkedin/twitter/bluesky)
        lookback_days: Days of history to analyze
        output_file: Optional output file path
    """
    db: Session = SessionLocal()
    
    try:
        logger.info(f"Extracting voice profile for {brand_id} on {platform}")
        
        # Initialize service
        service = VoiceIntelligenceService(db_session=db)
        
        # Extract profile
        profile = await service.build_voice_profile(
            brand_id=brand_id,
            platform=platform,
            lookback_days=lookback_days,
        )
        
        # Print results
        print("\n" + "="*60)
        print(f"Voice Profile Extracted Successfully")
        print("="*60)
        print(f"Profile ID: {profile.profile_id}")
        print(f"Brand ID: {profile.brand_id}")
        print(f"Platform: {platform}")
        print(f"Version: {profile.version}")
        print(f"Confidence: {profile.confidence:.2f}")
        print(f"Sample Size: {profile.sample_size}")
        print(f"Signature: {profile.signature[:16]}...")
        print("="*60)
        
        # Save to file if specified
        if output_file:
            import json
            output_data = {
                "profile_id": profile.profile_id,
                "brand_id": profile.brand_id,
                "platform": platform,
                "version": profile.version,
                "confidence": profile.confidence,
                "sample_size": profile.sample_size,
                "signature": profile.signature,
                "lexical_fingerprint": profile.lexical_fingerprint,
                "syntactic_fingerprint": profile.syntactic_fingerprint,
                "stylistic_fingerprint": profile.stylistic_fingerprint,
                "created_at": profile.created_at.isoformat(),
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nProfile saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)
    
    finally:
        db.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract voice profile from historical content"
    )
    parser.add_argument(
        "--brand-id",
        required=True,
        help="Brand identifier"
    )
    parser.add_argument(
        "--platform",
        required=True,
        choices=["linkedin", "twitter", "bluesky"],
        help="Platform to analyze"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of history to analyze (default: 90)"
    )
    parser.add_argument(
        "--output",
        help="Output file path (JSON)"
    )
    
    args = parser.parse_args()
    
    # Run extraction
    asyncio.run(extract_voice_profile(
        brand_id=args.brand_id,
        platform=args.platform,
        lookback_days=args.days,
        output_file=args.output,
    ))


if __name__ == "__main__":
    main()