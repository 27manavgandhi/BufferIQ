"""Script to start the BufferIQ API server."""

import argparse
import sys
from pathlib import Path

import uvicorn
import yaml

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start BufferIQ API server")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/api/development.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host to bind to (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (overrides config)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes (overrides config)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (overrides config)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Extract app config
    app_config = config.get("app", {})

    # Override with command line args
    host = args.host or app_config.get("host", "127.0.0.1")
    port = args.port or app_config.get("port", 8000)
    workers = args.workers or app_config.get("workers", 1)
    reload = args.reload or app_config.get("reload", False)

    logger.info(f"Starting BufferIQ API server")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Reload: {reload}")

    # Start server
    uvicorn.run(
        "bufferiq.api.app:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()