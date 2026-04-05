"""
Tests for configuration management.

This module ensures the Settings class correctly loads and validates
configuration from environment variables with proper type safety.
"""

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bufferiq.core.config import Environment, Settings, get_settings

class TestSettingsDefaults:
    """Test default values when no environment variables are set."""

    def test_default_environment_is_development(self) -> None:
        """Default environment should be development."""
        settings = Settings()
        assert settings.environment == Environment.DEVELOPMENT
        assert settings.is_development is True
        assert settings.is_production is False
        assert settings.is_testing is False

    def test_default_debug_is_true(self) -> None:
        """Debug should be enabled by default."""
        settings = Settings()
        assert settings.debug is True

    def test_default_database_is_sqlite(self) -> None:
        """Default database should be SQLite."""
        settings = Settings()
        assert settings.database_url == "sqlite:///./bufferiq.db"
        assert settings.database_is_sqlite is True
        assert settings.database_is_postgresql is False

    def test_default_buffer_api_url(self) -> None:
        """Default Buffer API URL should be production endpoint."""
        settings = Settings()
        assert settings.buffer_api_url == "https://graph.buffer.com/graphql"

    def test_default_redis_url(self) -> None:
        """Default Redis URL should be localhost."""
        settings = Settings()
        assert settings.redis_url == "redis://localhost:6379/0"

    def test_default_model_path(self) -> None:
        """Default model path should be ./models directory."""
        settings = Settings()
        assert settings.model_path == Path("./models")
        assert settings.model_path.exists()

    def test_default_log_level(self) -> None:
        """Default log level should be INFO."""
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_default_rate_limits(self) -> None:
        """Default rate limits should match Buffer API limits."""
        settings = Settings()
        assert settings.max_requests_per_15min == 100
        assert settings.max_requests_per_24hours == 500
        assert settings.max_requests_per_30days == 10000


class TestSettingsFromEnvironment:
    """Test settings loaded from environment variables."""

    def test_environment_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment should load from ENVIRONMENT variable."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        settings = Settings()
        assert settings.environment == Environment.PRODUCTION
        assert settings.is_production is True

    def test_debug_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Debug flag should load from DEBUG variable."""
        monkeypatch.setenv("DEBUG", "false")
        settings = Settings()
        assert settings.debug is False

    def test_database_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Database URL should load from DATABASE_URL variable."""
        db_url = "postgresql://user:pass@localhost/bufferiq"
        monkeypatch.setenv("DATABASE_URL", db_url)
        settings = Settings()
        assert settings.database_url == db_url
        assert settings.database_is_postgresql is True
        assert settings.database_is_sqlite is False

    def test_buffer_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Buffer API key should load from BUFFER_API_KEY variable."""
        api_key = "test_api_key_12345"
        monkeypatch.setenv("BUFFER_API_KEY", api_key)
        settings = Settings()
        assert settings.buffer_api_key == api_key

    def test_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Log level should load from LOG_LEVEL variable."""
        monkeypatch.setenv("LOG_LEVEL", "debug")
        settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_rate_limits_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rate limits should load from environment variables."""
        monkeypatch.setenv("MAX_REQUESTS_PER_15MIN", "50")
        monkeypatch.setenv("MAX_REQUESTS_PER_24HOURS", "250")
        monkeypatch.setenv("MAX_REQUESTS_PER_30DAYS", "5000")
        settings = Settings()
        assert settings.max_requests_per_15min == 50
        assert settings.max_requests_per_24hours == 250
        assert settings.max_requests_per_30days == 5000


class TestSettingsValidation:
    """Test validation of configuration values."""

    def test_invalid_environment_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid environment value should raise ValidationError."""
        monkeypatch.setenv("ENVIRONMENT", "invalid_env")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "Invalid environment" in str(exc_info.value)

    def test_invalid_log_level_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid log level should raise ValidationError."""
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "Invalid log level" in str(exc_info.value)

    def test_production_without_api_key_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production environment without API key should raise ValidationError."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("BUFFER_API_KEY", raising=False)
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "BUFFER_API_KEY is required in production" in str(exc_info.value)

    def test_production_with_api_key_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production environment with API key should succeed."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("BUFFER_API_KEY", "test_key")
        settings = Settings()
        assert settings.environment == Environment.PRODUCTION
        assert settings.buffer_api_key == "test_key"

    def test_rate_limit_too_low_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rate limit below minimum should raise ValidationError."""
        monkeypatch.setenv("MAX_REQUESTS_PER_15MIN", "0")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_rate_limit_too_high_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rate limit above maximum should raise ValidationError."""
        monkeypatch.setenv("MAX_REQUESTS_PER_15MIN", "1001")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "less than or equal to 1000" in str(exc_info.value)


class TestSettingsProperties:
    """Test computed properties on Settings."""

    def test_database_echo_true_in_development(self) -> None:
        """Database echo should be True in development with debug."""
        settings = Settings(environment=Environment.DEVELOPMENT, debug=True)
        assert settings.get_database_echo() is True

    def test_database_echo_false_in_production(self) -> None:
        """Database echo should be False in production."""
        settings = Settings(
            environment=Environment.PRODUCTION, buffer_api_key="test_key"
        )
        assert settings.get_database_echo() is False

    def test_database_echo_false_when_debug_disabled(self) -> None:
        """Database echo should be False when debug is disabled."""
        settings = Settings(environment=Environment.DEVELOPMENT, debug=False)
        assert settings.get_database_echo() is False


class TestGetSettingsFactory:
    """Test get_settings factory function."""

    def test_get_settings_returns_settings_instance(self) -> None:
        """get_settings should return a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_loads_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_settings should load configuration from environment."""
        monkeypatch.setenv("ENVIRONMENT", "testing")
        settings = get_settings()
        assert settings.environment == Environment.TESTING


class TestModelPathCreation:
    """Test automatic model directory creation."""

    def test_model_path_created_if_not_exists(self, tmp_path: Path) -> None:
        """Model directory should be created if it doesn't exist."""
        model_dir = tmp_path / "models"
        assert not model_dir.exists()

        settings = Settings(model_path=str(model_dir))
        assert settings.model_path.exists()
        assert settings.model_path.is_dir()

    def test_model_path_accepts_existing_directory(self, tmp_path: Path) -> None:
        """Model path should accept an existing directory."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        assert model_dir.exists()

        settings = Settings(model_path=str(model_dir))
        assert settings.model_path == model_dir


class TestCaseInsensitiveEnvironmentVariables:
    """Test that environment variable names are case-insensitive."""

    def test_lowercase_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lowercase environment variables should work."""
        monkeypatch.setenv("environment", "testing")
        monkeypatch.setenv("debug", "false")
        settings = Settings()
        assert settings.environment == Environment.TESTING
        assert settings.debug is False

    def test_uppercase_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uppercase environment variables should work."""
        monkeypatch.setenv("ENVIRONMENT", "testing")
        monkeypatch.setenv("DEBUG", "false")
        settings = Settings()
        assert settings.environment == Environment.TESTING
        assert settings.debug is False

    def test_mixed_case_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mixed case environment variables should work."""
        monkeypatch.setenv("EnViRoNmEnT", "testing")
        monkeypatch.setenv("DeBuG", "false")
        settings = Settings()
        assert settings.environment == Environment.TESTING
        assert settings.debug is False