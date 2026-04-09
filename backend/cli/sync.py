"""
Sync CLI commands.

Provides command-line interface for data synchronization.
"""

import asyncio
import sys

import click
from redis.asyncio import Redis

from bufferiq.core.cache import ResponseCache
from bufferiq.core.config import Settings
from bufferiq.core.database import get_async_engine, get_sessionmaker
from bufferiq.infrastructure.buffer.buffer_client import BufferClient
from bufferiq.infrastructure.buffer.rate_limiter import RateLimiter
from bufferiq.infrastructure.sync.progress_tracker import ProgressTracker
from bufferiq.infrastructure.sync.sync_service import SyncService
from bufferiq.infrastructure.sync.transformers import BufferTransformer


@click.group()
def sync() -> None:
    """Data synchronization commands."""
    pass


@sync.command()
@click.option("--user-id", required=True, type=int, help="User ID to sync")
def initial(user_id: int) -> None:
    """Perform initial sync for user."""
    click.echo(f"Starting initial sync for user {user_id}...")

    async def run_sync() -> None:
        settings = Settings()
        engine = get_async_engine(settings)
        sessionmaker = get_sessionmaker(engine)

        async with sessionmaker() as session:
            # Setup dependencies
            redis = Redis.from_url(settings.redis_url, decode_responses=True)
            rate_limiter = RateLimiter(redis)
            cache = ResponseCache(redis)
            client = BufferClient(settings, rate_limiter, cache)
            transformer = BufferTransformer()
            tracker = ProgressTracker(session)

            # Create sync service
            service = SyncService(session, client, transformer, tracker)

            # Run sync
            try:
                job_id = await service.initial_sync(user_id)
                click.echo(f"✓ Initial sync completed successfully (job {job_id})")
            except Exception as e:
                click.echo(f"✗ Initial sync failed: {e}", err=True)
                sys.exit(1)
            finally:
                await redis.close()
                await engine.dispose()

    asyncio.run(run_sync())


@sync.command()
@click.option("--user-id", required=True, type=int, help="User ID to sync")
def incremental(user_id: int) -> None:
    """Perform incremental sync for user."""
    click.echo(f"Starting incremental sync for user {user_id}...")

    async def run_sync() -> None:
        settings = Settings()
        engine = get_async_engine(settings)
        sessionmaker = get_sessionmaker(engine)

        async with sessionmaker() as session:
            # Setup dependencies
            redis = Redis.from_url(settings.redis_url, decode_responses=True)
            rate_limiter = RateLimiter(redis)
            cache = ResponseCache(redis)
            client = BufferClient(settings, rate_limiter, cache)
            transformer = BufferTransformer()
            tracker = ProgressTracker(session)

            # Create sync service
            service = SyncService(session, client, transformer, tracker)

            # Run sync
            try:
                job_id = await service.incremental_sync(user_id)
                click.echo(f"✓ Incremental sync completed successfully (job {job_id})")
            except Exception as e:
                click.echo(f"✗ Incremental sync failed: {e}", err=True)
                sys.exit(1)
            finally:
                await redis.close()
                await engine.dispose()

    asyncio.run(run_sync())


@sync.command()
@click.option("--user-id", required=True, type=int, help="User ID")
def status(user_id: int) -> None:
    """Show sync status for user."""
    click.echo(f"Sync status for user {user_id}:")

    async def get_status() -> None:
        settings = Settings()
        engine = get_async_engine(settings)
        sessionmaker = get_sessionmaker(engine)

        async with sessionmaker() as session:
            tracker = ProgressTracker(session)

            # Get recent jobs
            jobs = await tracker.get_recent_jobs(user_id, limit=5)

            if not jobs:
                click.echo("No sync jobs found")
                return

            for job in jobs:
                eta = tracker.calculate_eta(job)
                eta_str = f" (ETA: {eta}s)" if eta else ""

                click.echo(
                    f"  Job {job.id}: {job.sync_type} - {job.status} "
                    f"({job.processed_items}/{job.total_items}){eta_str}"
                )

        await engine.dispose()

    asyncio.run(get_status())


@sync.command()
@click.option("--user-id", required=True, type=int, help="User ID")
@click.option("--limit", default=10, help="Number of jobs to show")
def history(user_id: int, limit: int) -> None:
    """Show sync history for user."""
    click.echo(f"Sync history for user {user_id} (last {limit} jobs):")

    async def get_history() -> None:
        settings = Settings()
        engine = get_async_engine(settings)
        sessionmaker = get_sessionmaker(engine)

        async with sessionmaker() as session:
            tracker = ProgressTracker(session)
            jobs = await tracker.get_recent_jobs(user_id, limit=limit)

            if not jobs:
                click.echo("No sync jobs found")
                return

            for job in jobs:
                duration = ""
                if job.started_at and job.completed_at:
                    duration = (
                        f" ({(job.completed_at - job.started_at).total_seconds():.1f}s)"
                    )

                click.echo(
                    f"  {job.created_at.strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"Job {job.id}: {job.sync_type} - {job.status} "
                    f"({job.processed_items} items){duration}"
                )

        await engine.dispose()

    asyncio.run(get_history())


if __name__ == "__main__":
    sync()
