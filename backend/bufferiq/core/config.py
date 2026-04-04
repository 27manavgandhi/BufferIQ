"""
Type-safe configuration management using Pydantic settings.

This module provides a centralized configuration system with:
- Environment variable loading with validation
- Type-safe access to all settings
- Clear defaults for development
- Validation of required fields
- Support for multiple environments (dev, test, prod)
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Valid environment names."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings are type-safe and validated. Missing required settings
    will raise a ValidationError at application startup.

    Environment variables are loaded from .env file in development.
    In production, set environment variables directly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
    )

    debug: bool = Field(
        default=True,
        description="Enable debug mode (verbose logging, auto-reload)",
    )

    database_url: str = Field(
        default="sqlite:///./bufferiq.db",
        description="Database connection URL (SQLite or PostgreSQL)",
    )

    buffer_api_url: str = Field(
        default="https://graph.buffer.com/graphql",
        description="Buffer GraphQL API endpoint",
    )

    buffer_api_key: Optional[str] = Field(
        default=None,
        description="Buffer API key (required for production)",
    )

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for caching",
    )

    model_path: Path = Field(
        default=Path("./models"),
        description="Directory to store ML models",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    max_requests_per_15min: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Buffer API rate limit: requests per 15 minutes",
    )

    max_requests_per_24hours: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Buffer API rate limit: requests per 24 hours",
    )

    max_requests_per_30days: int = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Buffer API rate limit: requests per 30 days",
    )

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str) -> Environment:
        """Validate and convert environment string to enum."""
        if isinstance(v, Environment):
            return v
        try:
            return Environment(v.lower())
        except ValueError as e:
            raise ValueError(
                f"Invalid environment: {v}. Must be one of: "
                f"{', '.join(e.value for e in Environment)}"
            ) from e

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a recognized level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid log level: {v}. Must be one of: {', '.join(valid_levels)}"
            )
        return v_upper

    @field_validator("model_path", mode="before")
    @classmethod
    def validate_model_path(cls, v: str | Path) -> Path:
        """Convert model path to Path object and create if doesn't exist."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("buffer_api_key", mode="after")
    @classmethod
    def validate_api_key_in_production(
        cls, v: Optional[str], info
    ) -> Optional[str]:
        """Ensure API key is set in production environment."""
        environment = info.data.get("environment")
        if environment == Environment.PRODUCTION and not v:
            raise ValueError("BUFFER_API_KEY is required in production environment")
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def database_is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.database_url.startswith("sqlite")

    @property
    def database_is_postgresql(self) -> bool:
        """Check if using PostgreSQL database."""
        return self.database_url.startswith("postgresql")

    def get_database_echo(self) -> bool:
        """
        Get SQLAlchemy echo setting based on environment.

        Returns:
            True in development (verbose SQL logging), False otherwise
        """
        return self.is_development and self.debug


def get_settings() -> Settings:
    """
    Factory function to create Settings instance.

    This function is the recommended way to get settings in the application.
    It ensures settings are loaded once and can be easily mocked in tests.

    Returns:
        Settings instance with validated configuration

    Raises:
        ValidationError: If configuration is invalid or missing required fields

    Example:
        >>> settings = get_settings()
        >>> print(settings.environment)
        Environment.DEVELOPMENT
    """
    return Settings()