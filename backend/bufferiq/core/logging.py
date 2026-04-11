"""
Structured logging configuration.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any

from bufferiq.core.config import Settings


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class StructuredLogger(logging.LoggerAdapter):
    """
    Logger adapter that supports structured keyword arguments.
    Example:
        logger.info("Loaded posts", count=500)
    """

    def process(self, msg: str, kwargs: dict[str, Any]):
        # Extract structured fields
        structured_fields = {
            k: v for k, v in kwargs.items() if k not in ("exc_info", "stack_info")
        }

        if structured_fields:
            fields = " | ".join(f"{k}={v}" for k, v in structured_fields.items())
            msg = f"{msg} | {fields}"

        # Clear kwargs so logging doesn't crash
        return msg, {}


def setup_logging(settings: Settings) -> None:
    """
    Configure application logging.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))

    if settings.is_production:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Reduce noise from dependencies
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> StructuredLogger:
    """
    Get structured logger instance.
    """

    logger = logging.getLogger(name)

    # If logging isn't configured yet, attach default handler
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return StructuredLogger(logger, {})
