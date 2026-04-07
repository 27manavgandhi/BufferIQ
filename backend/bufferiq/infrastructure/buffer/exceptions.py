"""
Custom exceptions for Buffer API client.
"""


class BufferAPIError(Exception):
    """Base exception for all Buffer API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class BufferAuthenticationError(BufferAPIError):
    """Authentication failed (401)."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, status_code=401)


class BufferRateLimitError(BufferAPIError):
    """Rate limit exceeded (429)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class BufferValidationError(BufferAPIError):
    """Request validation failed (400)."""

    def __init__(
        self, message: str, errors: dict[str, list[str]] | None = None
    ) -> None:
        super().__init__(message, status_code=400)
        self.errors = errors or {}


class BufferNotFoundError(BufferAPIError):
    """Resource not found (404)."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status_code=404)


class BufferNetworkError(BufferAPIError):
    """Network/connection error."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message, status_code=None)
        self.original_error = original_error
