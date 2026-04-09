"""
Manual sync execution script.

Run data synchronization from command line.
"""

import argparse
import asyncio
import logging
import sys

from redis.asyncio import Redis

from bufferiq.core.cache import ResponseCache
from bufferiq.core.config import Settings
from bufferiq.core.database import get_async_engine, get_sessionmaker
from bufferiq.infrastructure.buffer.buffer_client import BufferClient
from bufferiq.infrastructure.buffer.rate_limiter import RateLimiter
from bufferiq.infrastructure.sync.progress_tracker import ProgressTracker
from bufferiq.infrastructure.sync.sync_service import SyncService
from bufferiq.infrastructure.sync.transformers import BufferTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_sync(
    user_id: int, mode: str, dry_run: bool = False, verbose: bool = False
) -> None:
    """
    Run sync for user.

    Args:
        user_id: User ID to sync
        mode: Sync mode ('initial' or 'incremental')
        dry_run: If True, don't write to database
        verbose: If True, show detailed logging
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting {mode} sync for user {user_id} (dry_run={dry_run})")

    # Setup
    settings = Settings()
    engine = get_async_engine(settings)
    sessionmaker = get_sessionmaker(engine)

    async with sessionmaker() as session:
        # Dependencies
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        rate_limiter = RateLimiter(redis)
        cache = ResponseCache(redis)
        client = BufferClient(settings, rate_limiter, cache)
        transformer = BufferTransformer()
        tracker = ProgressTracker(session)

        # Service
        service = SyncService(session, client, transformer, tracker)

        try:
            # Run sync
            if mode == "initial":
                job_id = await service.initial_sync(user_id)
            elif mode == "incremental":
                job_id = await service.incremental_sync(user_id)
            else:
                raise ValueError(f"Invalid mode: {mode}")

            if dry_run:
                logger.info("Dry run - rolling back changes")
                await session.rollback()
            else:
                await session.commit()

            logger.info(f"Sync completed successfully (job {job_id})")

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            await session.rollback()
            sys.exit(1)

        finally:
            await redis.close()
            await engine.dispose()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run BufferIQ sync")
    parser.add_argument("--user-id", type=int, required=True, help="User ID")
    parser.add_argument(
        "--mode",
        choices=["initial", "incremental"],
        required=True,
        help="Sync mode",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    asyncio.run(
        run_sync(args.user_id, args.mode, args.dry_run, args.verbose)
    )


if __name__ == "__main__":
    main()
