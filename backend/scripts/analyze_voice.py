"""
Voice analysis script.

Analyze content voice alignment via CLI.
"""

import asyncio
import argparse
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from bufferiq.core.database import SessionLocal
from bufferiq.ml.voice.intelligence.service import VoiceIntelligenceService

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def analyze_content(
    brand_id: str,
    platform: str,
    content: str,
    show_recommendations: bool = True,
):
    """
    Analyze single piece of content.
    
    Args:
        brand_id: Brand identifier
        platform: Platform
        content: Content to analyze
        show_recommendations: Show recommendations
    """
    db: Session = SessionLocal()
    
    try:
        logger.info(f"Analyzing content for {brand_id} on {platform}")
        
        # Initialize service
        service = VoiceIntelligenceService(db_session=db)
        
        # Analyze content
        analysis = await service.analyze_content(
            text=content,
            brand_id=brand_id,
            platform=platform,
            return_recommendations=show_recommendations,
        )
        
        # Print results
        print("\n" + "="*60)
        print(f"Voice Analysis Results")
        print("="*60)
        print(f"Content: {content[:50]}...")
        print(f"Brand: {brand_id}")
        print(f"Platform: {platform}")
        print("\nConsistency Scores:")
        print(f"  Overall: {analysis['consistency_score']['overall']:.1f}/100")
        print(f"  Lexical: {analysis['consistency_score']['lexical']:.1f}/100")
        print(f"  Syntactic: {analysis['consistency_score']['syntactic']:.1f}/100")
        print(f"  Stylistic: {analysis['consistency_score']['stylistic']:.1f}/100")
        print(f"\nConsistent: {analysis['consistency_score']['is_consistent']}")
        print(f"Severity: {analysis['consistency_score']['severity']}")
        
        if show_recommendations and 'recommendations' in analysis:
            print(f"\nRecommendations ({len(analysis['recommendations'])}):")
            for i, rec in enumerate(analysis['recommendations'][:5], 1):
                print(f"\n  {i}. [{rec['priority'].upper()}] {rec['type']}")
                print(f"     {rec['reason']}")
                print(f"     Impact: {rec['impact_score']:.1f} points")
        
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)
    
    finally:
        db.close()


async def analyze_batch_from_csv(
    brand_id: str,
    platform: str,
    input_file: str,
    output_file: str,
):
    """
    Analyze batch of content from CSV file.
    
    Args:
        brand_id: Brand identifier
        platform: Platform
        input_file: Input CSV file path
        output_file: Output JSON file path
    """
    db: Session = SessionLocal()
    
    try:
        logger.info(f"Analyzing batch from {input_file}")
        
        # Read CSV
        contents = []
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'text' in row:
                    contents.append(row['text'])
                elif 'content' in row:
                    contents.append(row['content'])
        
        logger.info(f"Loaded {len(contents)} items from CSV")
        
        # Initialize service
        service = VoiceIntelligenceService(db_session=db)
        
        # Analyze batch
        results = await service.analyze_batch(
            contents=contents,
            brand_id=brand_id,
            platform=platform,
        )
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnalyzed {len(results)} items")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        if successful > 0:
            avg_score = sum(
                r['consistency_score']['overall']
                for r in results if 'consistency_score' in r
            ) / successful
            
            consistent_count = sum(
                1 for r in results
                if 'consistency_score' in r and r['consistency_score']['is_consistent']
            )
            
            print(f"\nSummary:")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Average Score: {avg_score:.1f}/100")
            print(f"  Consistent: {consistent_count}/{successful}")
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)
    
    finally:
        db.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze content voice alignment"
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
        help="Platform"
    )
    
    # Single content or batch
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        help="Content text to analyze"
    )
    group.add_argument(
        "--input",
        help="Input CSV file for batch analysis"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path (JSON, required for batch)"
    )
    parser.add_argument(
        "--no-recommendations",
        action="store_true",
        help="Don't show recommendations"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    if args.text:
        # Single content
        asyncio.run(analyze_content(
            brand_id=args.brand_id,
            platform=args.platform,
            content=args.text,
            show_recommendations=not args.no_recommendations,
        ))
    else:
        # Batch
        if not args.output:
            print("Error: --output is required for batch analysis")
            sys.exit(1)
        
        asyncio.run(analyze_batch_from_csv(
            brand_id=args.brand_id,
            platform=args.platform,
            input_file=args.input,
            output_file=args.output,
        ))


if __name__ == "__main__":
    main()