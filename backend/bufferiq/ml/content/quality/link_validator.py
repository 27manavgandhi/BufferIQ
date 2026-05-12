"""
Link validation.

Validates URLs in content.
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse


@dataclass
class LinkValidation:
    """Link validation result."""

    url: str
    is_valid: bool
    is_https: bool
    has_tracking: bool
    issues: List[str]


class LinkValidator:
    """
        Validate links in content.

        Checks URL format, protocol, and potential issues.

        Example:
    ```python
            validator = LinkValidator()
            results = validator.validate_links(
                "Check https://example.com and http://bad.com"
            )
            for result in results:
                print(f"{result.url}: {result.is_valid}")
    ```
    """

    def __init__(self) -> None:
        """Initialize link validator."""
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

        # Common tracking parameters
        self.tracking_params = {
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "fbclid",
            "gclid",
        }

    def validate_links(self, text: str) -> List[LinkValidation]:
        """
        Validate all links in text.

        Args:
            text: Text containing links

        Returns:
            List of link validation results

        Raises:
            ValueError: If text is empty
        """
        if not text:
            raise ValueError("Text cannot be empty")

        urls = self.url_pattern.findall(text)
        results = []

        for url in urls:
            results.append(self.validate_link(url))

        return results

    def validate_link(self, url: str) -> LinkValidation:
        """
        Validate a single link.

        Args:
            url: URL to validate

        Returns:
            Link validation result

        Raises:
            ValueError: If URL is empty
        """
        if not url:
            raise ValueError("URL cannot be empty")

        issues = []
        is_valid = True

        try:
            parsed = urlparse(url)

            # Check for valid scheme
            if parsed.scheme not in ["http", "https"]:
                is_valid = False
                issues.append(f"Invalid scheme: {parsed.scheme}")

            # Check for HTTPS
            is_https = parsed.scheme == "https"
            if not is_https:
                issues.append("Not using HTTPS")

            # Check for netloc
            if not parsed.netloc:
                is_valid = False
                issues.append("Missing domain")

            # Check for tracking parameters
            has_tracking = any(param in url for param in self.tracking_params)
            if has_tracking:
                issues.append("Contains tracking parameters")

        except Exception as e:
            is_valid = False
            issues.append(f"Parse error: {str(e)}")
            is_https = False
            has_tracking = False

        return LinkValidation(
            url=url,
            is_valid=is_valid,
            is_https=is_https,
            has_tracking=has_tracking,
            issues=issues,
        )
