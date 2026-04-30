
"""

Uses Pydantic Settings for environment variable validation,
type safety, caching, API serving, and production configuration.
"""

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes are loaded from environment variables
    with the same name (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # =========================================================
    # Application
    # =========================================================
    app_name: str = "BufferIQ"
    version: str = "1.0.0"

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True

    # =========================================================
    # Paths
    # =========================================================
    base_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )

    data_path: Path = Field(default_factory=lambda: Path("data"))

    model_path: Path = Field(default_factory=lambda: Path("outputs/models"))

    output_path: Path = Field(default_factory=lambda: Path("outputs"))

    # =========================================================
    # Database
    # =========================================================
    database_url: str = "sqlite:///./bufferiq.db"
    database_echo: bool = False

    # =========================================================
    # Buffer API
    # =========================================================
    buffer_api_url: str = "https://graph.buffer.com/graphql"
    buffer_api_key: str = ""

    # =========================================================
    # Redis (Day 14)
    # =========================================================
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0

    # =========================================================
    # Cache (Day 14)
    # =========================================================
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000

    # =========================================================
    # API Server (Day 14)
    # =========================================================
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = False

    # =========================================================
    # CORS (Day 14)
    # =========================================================
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    cors_credentials: bool = True

    cors_methods: List[str] = [
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "OPTIONS",
    ]

    cors_headers: List[str] = ["*"]

    # =========================================================
    # Rate Limiting (Day 14)
    # =========================================================
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60

    # Legacy Rate Limits (Per User)
    max_requests_per_15min: int = 100
    max_requests_per_24hours: int = 500
    max_requests_per_30days: int = 999

    # =========================================================
    # ML Models (Day 14)
    # =========================================================
    warmup_models: bool = True
    model_cache_size: int = 3

    # =========================================================
    # Supported Platforms
    # =========================================================
    supported_platforms: List[str] = [
        "linkedin",
        "twitter",
        "bluesky",
    ]

    # =========================================================
    # Logging
    # =========================================================
    log_level: str = "INFO"
    log_format: str = "json"

    # =========================================================
    # Validators
    # =========================================================
    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: Any) -> Environment:
        """Validate and convert environment string to enum."""
        if isinstance(v, Environment):
            return v

        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError as err:
                raise ValueError(
                    f"Invalid environment: {v}. "
                    "Must be one of: development, testing, production"
                ) from err

        raise ValueError(f"Invalid environment type: {type(v)}")

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: Any) -> str:
        """Validate log level."""
        valid_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if isinstance(v, str):
            v_upper = v.upper()

            if v_upper not in valid_levels:
                raise ValueError(
                    f"Invalid log level: {v}. "
                    f"Must be one of: {', '.join(valid_levels)}"
                )

            return v_upper

        raise ValueError(f"Invalid log level type: {type(v)}")

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        valid_formats = {"json", "text"}

        if v.lower() not in valid_formats:
            raise ValueError(
                f"Invalid log format: {v}. "
                f"Must be one of: {', '.join(valid_formats)}"
            )

        return v.lower()

    @field_validator("buffer_api_key")
    @classmethod
    def validate_buffer_api_key(cls, v: str, info: Any) -> str:
        """Validate Buffer API key is set in production."""
        values = info.data
        environment = values.get("environment", Environment.DEVELOPMENT)

        if environment == Environment.PRODUCTION and not v:
            raise ValueError(
                "BUFFER_API_KEY is required in production environment"
            )

        return v

    @field_validator(
        "max_requests_per_15min",
        "max_requests_per_24hours",
        "max_requests_per_30days",
        "rate_limit_per_minute",
    )
    @classmethod
    def validate_rate_limits(cls, v: int) -> int:
        """Validate rate limits are within acceptable range."""
        if v < 1:
            raise ValueError(
                "Rate limits must be greater than or equal to 1"
            )

        if v > 100000:
            raise ValueError(
                "Rate limits must be less than or equal to 100000"
            )

        return v

    @field_validator(
        "model_path",
        "data_path",
        "output_path",
        mode="before",
    )
    @classmethod
    def create_paths(cls, v: Any) -> Path:
        """Ensure required paths exist."""
        path = Path(v) if isinstance(v, str) else v

        if not isinstance(path, Path):
            raise ValueError(f"Invalid path type: {type(v)}")

        path.mkdir(parents=True, exist_ok=True)

        return path

    @field_validator("api_port")
    @classmethod
    def validate_api_port(cls, v: int) -> int:
        """Validate API port."""
        if not 1 <= v <= 65535:
            raise ValueError("API port must be between 1 and 65535")

        return v

    @field_validator("api_workers")
    @classmethod
    def validate_api_workers(cls, v: int) -> int:
        """Validate worker count."""
        if v < 1:
            raise ValueError("API workers must be >= 1")

        return v

    @field_validator("cache_ttl")
    @classmethod
    def validate_cache_ttl(cls, v: int) -> int:
        """Validate cache TTL."""
        if v < 0:
            raise ValueError("Cache TTL cannot be negative")

        return v

    @field_validator("cache_max_size")
    @classmethod
    def validate_cache_max_size(cls, v: int) -> int:
        """Validate cache max size."""
        if v < 1:
            raise ValueError("Cache max size must be >= 1")

        return v

    # =========================================================
    # Environment Helpers
    # =========================================================
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == Environment.TESTING

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION

    # =========================================================
    # Database Helpers
    # =========================================================
    @property
    def computed_database_echo(self) -> bool:
        """
        Enable SQLAlchemy query logging in development.

        Uses explicit database_echo config OR development debug mode.
        """
        return self.database_echo or (
            self.is_development and self.debug
        )

    @property
    def database_is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.database_url.startswith("sqlite://")

    @property
    def database_is_postgresql(self) -> bool:
        """Check if using PostgreSQL database."""
        return self.database_url.startswith("postgresql://")

    # =========================================================
    # Redis Helpers
    # =========================================================
    @property
    def redis_connection_url(self) -> str:
        """Get Redis connection URL with DB index."""
        return f"{self.redis_url}/{self.redis_db}"

    # =========================================================
    # Methods
    # =========================================================
    def get_database_echo(self) -> bool:
        """Get database echo setting (method form for tests)."""
        return self.computed_database_echo

    def model_post_init(self, __context: Any) -> None:
        """Run after model initialization."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)


# =============================================================
# Global Settings Singleton
# =============================================================
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings singleton.

    Returns:
        Settings instance
    """
    global _settings

    if _settings is None:
        _settings = Settings()

    return _settings

