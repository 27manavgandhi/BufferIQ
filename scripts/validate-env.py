"""
Environment validation script.

Validates that all required environment variables are set correctly
and that the application can start successfully.
"""

import sys
from pathlib import Path

try:
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

    from bufferiq.core.config import Settings, get_settings
    from bufferiq.core.database import get_async_engine, check_database_health
    import asyncio
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and dependencies are installed")
    sys.exit(1)


async def validate_database() -> bool:
    """Validate database connectivity."""
    try:
        settings = get_settings()
        engine = get_async_engine(settings)

        health = await check_database_health(engine)

        await engine.dispose()

        return health
    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        return False


def validate_environment() -> bool:
    """Validate environment configuration."""
    try:
        settings = get_settings()

        print("✅ Configuration loaded successfully")
        print(f"   Environment: {settings.environment.value}")
        print(f"   Debug: {settings.debug}")
        print(
            f"   Database: {settings.database_url.split('@')[-1] if '@' in settings.database_url else settings.database_url}"
        )
        print(f"   Redis: {settings.redis_url}")
        print(f"   Model path: {settings.model_path}")

        if not settings.model_path.exists():
            print(f"⚠️  Model path does not exist: {settings.model_path}")
            return False

        if settings.is_production and not settings.buffer_api_key:
            print("❌ BUFFER_API_KEY required in production")
            return False

        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


async def main() -> None:
    """Run all validations."""
    print("BufferIQ Environment Validation")
    print("=" * 50)

    # Validate environment
    env_ok = validate_environment()

    if not env_ok:
        print("\n❌ Environment validation failed")
        sys.exit(1)

    # Validate database
    print("\nValidating database connection...")
    db_ok = await validate_database()

    if db_ok:
        print("✅ Database connection OK")
    else:
        print("❌ Database connection failed")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ All validations passed!")
    print("BufferIQ is ready to run")


if __name__ == "__main__":
    asyncio.run(main())
