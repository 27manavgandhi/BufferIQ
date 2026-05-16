"""
Voice CLI tool.

Command-line interface for voice analysis operations.
"""

import asyncio
import click
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from bufferiq.core.database import SessionLocal
from bufferiq.ml.voice.intelligence.service import VoiceIntelligenceService


@click.group()
def cli():
    """BufferIQ Voice Analysis CLI"""
    pass


@cli.command()
@click.option("--brand-id", required=True, help="Brand identifier")
@click.option("--platform", required=True, type=click.Choice(["linkedin", "twitter", "bluesky"]))
@click.option("--days", default=90, help="Days of history to analyze")
def extract(brand_id: str, platform: str, days: int):
    """Extract voice profile from historical content."""
    
    async def _extract():
        db: Session = SessionLocal()
        try:
            service = VoiceIntelligenceService(db_session=db)
            
            click.echo(f"Extracting voice profile for {brand_id} on {platform}...")
            
            profile = await service.build_voice_profile(
                brand_id=brand_id,
                platform=platform,
                lookback_days=days,
            )
            
            click.echo("\n✓ Profile extracted successfully!")
            click.echo(f"  Profile ID: {profile.profile_id}")
            click.echo(f"  Confidence: {profile.confidence:.2f}")
            click.echo(f"  Sample Size: {profile.sample_size}")
            
        except Exception as e:
            click.echo(f"\n✗ Error: {e}", err=True)
            sys.exit(1)
        finally:
            db.close()
    
    asyncio.run(_extract())


@cli.command()
@click.option("--brand-id", required=True, help="Brand identifier")
@click.option("--platform", required=True, type=click.Choice(["linkedin", "twitter", "bluesky"]))
@click.option("--text", required=True, help="Content to analyze")
def analyze(brand_id: str, platform: str, text: str):
    """Analyze content voice alignment."""
    
    async def _analyze():
        db: Session = SessionLocal()
        try:
            service = VoiceIntelligenceService(db_session=db)
            
            click.echo(f"Analyzing content for {brand_id} on {platform}...")
            
            analysis = await service.analyze_content(
                text=text,
                brand_id=brand_id,
                platform=platform,
                return_recommendations=True,
            )
            
            score = analysis['consistency_score']['overall']
            is_consistent = analysis['consistency_score']['is_consistent']
            
            click.echo(f"\n{'✓' if is_consistent else '✗'} Consistency Score: {score:.1f}/100")
            click.echo(f"  Lexical: {analysis['consistency_score']['lexical']:.1f}")
            click.echo(f"  Syntactic: {analysis['consistency_score']['syntactic']:.1f}")
            click.echo(f"  Stylistic: {analysis['consistency_score']['stylistic']:.1f}")
            
            if 'recommendations' in analysis and analysis['recommendations']:
                click.echo(f"\nTop Recommendations:")
                for rec in analysis['recommendations'][:3]:
                    click.echo(f"  • [{rec['priority'].upper()}] {rec['reason']}")
            
        except Exception as e:
            click.echo(f"\n✗ Error: {e}", err=True)
            sys.exit(1)
        finally:
            db.close()
    
    asyncio.run(_analyze())


@cli.command()
@click.option("--brand-id", required=True, help="Brand identifier")
@click.option("--platform", required=True, type=click.Choice(["linkedin", "twitter", "bluesky"]))
@click.option("--window", default=30, help="Recent window size in days")
def drift(brand_id: str, platform: str, window: int):
    """Detect voice drift."""
    
    async def _drift():
        db: Session = SessionLocal()
        try:
            service = VoiceIntelligenceService(db_session=db)
            
            click.echo(f"Detecting drift for {brand_id} on {platform}...")
            
            drift_result = await service.detect_drift(
                brand_id=brand_id,
                platform=platform,
                window_days=window,
            )
            
            detected = drift_result['drift_detected']
            score = drift_result['drift_score']
            
            click.echo(f"\n{'⚠' if detected else '✓'} Drift Score: {score:.1f}/100")
            click.echo(f"  Type: {drift_result['drift_type']}")
            click.echo(f"  Severity: {drift_result['severity']}")
            
            if drift_result['affected_dimensions']:
                click.echo(f"  Affected: {', '.join(drift_result['affected_dimensions'])}")
            
            if drift_result['likely_causes']:
                click.echo(f"\nLikely Causes:")
                for cause in drift_result['likely_causes']:
                    click.echo(f"  • {cause}")
            
        except Exception as e:
            click.echo(f"\n✗ Error: {e}", err=True)
            sys.exit(1)
        finally:
            db.close()
    
    asyncio.run(_drift())


if __name__ == "__main__":
    cli()