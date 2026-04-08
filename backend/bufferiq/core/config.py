"""
Application configuration management.

Uses Pydantic Settings for environment variable validation and type safety.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes are loaded from environment variables with the same name (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True

    # Database
    database_url: str = "sqlite:///./bufferiq.db"

    # Buffer API
    buffer_api_url: str = "https://graph.buffer.com/graphql"
    buffer_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ML Models
    model_path: Path = Path("./models")

    # Logging
    log_level: str = "INFO"

    # Rate Limits (per user)
    max_requests_per_15min: int = 100
    max_requests_per_24hours: int = 500
    max_requests_per_30days: int = 10000

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: Any) -> Environment:
        """Validate and convert environment string to enum."""
        if isinstance(v, Environment):
            return v
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid environment: {v}. Must be one of: development, testing, production"
                )
        raise ValueError(f"Invalid environment type: {type(v)}")

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: Any) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if isinstance(v, str):
            v_upper = v.upper()
            if v_upper not in valid_levels:
                raise ValueError(
                    f"Invalid log level: {v}. Must be one of: {', '.join(valid_levels)}"
                )
            return v_upper
        raise ValueError(f"Invalid log level type: {type(v)}")

    @field_validator("buffer_api_key")
    @classmethod
    def validate_buffer_api_key(cls, v: str, info: Any) -> str:
        """Validate Buffer API key is set in production."""
        values = info.data
        environment = values.get("environment", Environment.DEVELOPMENT)

        if environment == Environment.PRODUCTION and not v:
            raise ValueError("BUFFER_API_KEY is required in production environment")
        return v

    @field_validator(
        "max_requests_per_15min", "max_requests_per_24hours", "max_requests_per_30days"
    )
    @classmethod
    def validate_rate_limits(cls, v: int) -> int:
        """Validate rate limits are within acceptable range."""
        if v < 1:
            raise ValueError("Rate limits must be greater than or equal to 1")
        if v > 1000:
            raise ValueError("Rate limits must be less than or equal to 1000")
        return v

    @field_validator("model_path", mode="before")
    @classmethod
    def create_model_path(cls, v: Any) -> Path:
        """Ensure model path exists."""
        path = Path(v) if isinstance(v, str) else v
        if not isinstance(path, Path):
            raise ValueError(f"Invalid model_path type: {type(v)}")
        path.mkdir(parents=True, exist_ok=True)
        return path

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

    @property
    def database_echo(self) -> bool:
        """Enable SQLAlchemy query logging in development with debug enabled."""
        return self.is_development and self.debug

    @property
    def database_is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.database_url.startswith("sqlite://")

    @property
    def database_is_postgresql(self) -> bool:
        """Check if using PostgreSQL database."""
        return self.database_url.startswith("postgresql://")

    def get_database_echo(self) -> bool:
        """Get database echo setting (method form for tests)."""
        return self.database_echo

    def model_post_init(self, __context: Any) -> None:
        """Run after model initialization."""
        self.model_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Settings | None = None


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
