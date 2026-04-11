"""Base feature extractor class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)

# Supported platforms (ONLY these three)
SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


def validate_platform(platform: str) -> None:
    """
    Validate platform is supported.

    Args:
        platform: Platform name to validate

    Raises:
        ValueError: If platform is not supported

    Example:
        >>> validate_platform("linkedin")  # OK
        >>> validate_platform("facebook")  # Raises ValueError
    """
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Platform '{platform}' is not supported. "
            f"Supported platforms: {SUPPORTED_PLATFORMS}"
        )


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors."""

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """
        Return list of feature names this extractor produces.

        Returns:
            List of feature names
        """
        pass

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from DataFrame (batch extraction).

        Args:
            df: Input DataFrame with post data

        Returns:
            DataFrame with extracted features

        Raises:
            ValueError: If required columns are missing
        """
        pass

    @abstractmethod
    def extract_single(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from single post dictionary.

        Args:
            post_data: Dictionary with post data

        Returns:
            Dictionary with extracted features

        Example:
            >>> extractor = SomeFeatureExtractor()
            >>> features = extractor.extract_single({
            ...     "content": "Hello world!",
            ...     "published_at": "2024-01-01T10:00:00Z"
            ... })
        """
        pass

    def validate_input(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Validate input DataFrame has required columns.

        Args:
            df: Input DataFrame to validate
            required_columns: List of required column names

        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {df.columns.tolist()}"
            )

        logger.info(
            f"Input validation passed for {self.__class__.__name__}",
        )