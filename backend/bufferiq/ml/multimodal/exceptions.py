"""Custom exceptions for multi-modal analysis."""


class MultiModalError(Exception):
    """Base exception for multi-modal analysis errors."""
    
    pass


class UnsupportedPlatformError(MultiModalError):
    """Raised when an unsupported platform is used."""
    
    def __init__(self, platform: str, supported_platforms: list[str]):
        """Initialize with platform info."""
        self.platform = platform
        self.supported_platforms = supported_platforms
        message = (
            f"Platform '{platform}' is not supported. "
            f"Supported platforms: {', '.join(supported_platforms)}"
        )
        super().__init__(message)


class MediaProcessingError(MultiModalError):
    """Raised when media processing fails."""
    
    pass


class AnalysisError(MultiModalError):
    """Raised when analysis fails."""
    
    pass


class InvalidMediaError(MultiModalError):
    """Raised when media file is invalid or corrupted."""
    
    pass


class FeatureExtractionError(MultiModalError):
    """Raised when feature extraction fails."""
    
    pass


class PredictionError(MultiModalError):
    """Raised when prediction fails."""
    
    pass