"""Custom exceptions for segmentation module."""


class SegmentationError(Exception):
    """Base exception for segmentation errors."""

    pass


class UnsupportedPlatformError(SegmentationError):
    """Raised when unsupported platform is provided."""

    def __init__(
        self, platform: str, supported_platforms: list[str]
    ) -> None:
        """Initialize exception."""
        self.platform = platform
        self.supported_platforms = supported_platforms
        super().__init__(
            f"Platform '{platform}' is not supported. "
            f"Supported platforms: {', '.join(supported_platforms)}"
        )


class InsufficientDataError(SegmentationError):
    """Raised when insufficient data for segmentation."""

    def __init__(self, message: str) -> None:
        """Initialize exception."""
        super().__init__(message)


class ClusteringError(SegmentationError):
    """Raised when clustering fails."""

    def __init__(self, message: str) -> None:
        """Initialize exception."""
        super().__init__(message)


class PersonaGenerationError(SegmentationError):
    """Raised when persona generation fails."""

    def __init__(self, message: str) -> None:
        """Initialize exception."""
        super().__init__(message)


class ValidationError(SegmentationError):
    """Raised when data validation fails."""

    def __init__(self, message: str) -> None:
        """Initialize exception."""
        super().__init__(message)