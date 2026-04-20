"""Sync scheduler for periodic data synchronization."""

import asyncio
from typing import Optional

from bufferiq.core.logging import get_logger
from bufferiq.infrastructure.sync.sync_service import SyncService

logger = get_logger(__name__)


class SyncScheduler:
    """Schedule periodic sync operations."""

    def __init__(
        self,
        sync_service: SyncService,
        interval_hours: int = 24,
    ) -> None:
        """
        Initialize sync scheduler.

        Args:
            sync_service: Sync service instance
            interval_hours: Hours between syncs
        """
        self.sync_service = sync_service
        self.interval_hours = interval_hours
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_scheduler())
        logger.info(f"Scheduler started (interval: {self.interval_hours}h)")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Scheduler stopped")

    async def _run_scheduler(self) -> None:
        """Run the scheduler loop."""
        while self._running:
            try:
                # Run sync
                logger.info("Running scheduled sync...")
                await self.sync_service.sync_all_data()

                # Wait for next interval
                await asyncio.sleep(self.interval_hours * 3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(300)  # 5 minutes

    async def run_now(self) -> None:
        """Run sync immediately."""
        logger.info("Running immediate sync...")
        await self.sync_service.sync_all_data()
