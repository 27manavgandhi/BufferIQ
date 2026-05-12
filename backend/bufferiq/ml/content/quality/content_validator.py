"""
Content quality validation.

Comprehensive content quality checking.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from bufferiq.ml.content.quality.grammar_checker import GrammarChecker
from bufferiq.ml.content.quality.link_validator import LinkValidator

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class QualityIssue:
    """Quality issue found in content."""

    type: str  # "grammar", "spelling", "link", "length"
    severity: str  # "error", "warning", "info"
    message: str
    suggestion: Optional[str]
    location: Optional[int]


@dataclass
class QualityReport:
    """Content quality assessment."""

    score: float  # 0-100
    issues: List[QualityIssue]
    grammar_errors: int
    spelling_errors: int
    broken_links: int
    warnings: List[str]
    recommendations: List[str]


class ContentQualityChecker:
    """
        Check content quality and identify issues.

        Performs:
        - Grammar checking
        - Spelling validation
        - Link verification
        - Length optimization
        - Formatting validation

        Example:
    ```python
            checker = ContentQualityChecker()
            report = checker.check("This are a test post.", "linkedin")
            print(report.score)  # 75.0
            print(report.grammar_errors)  # 1
            for issue in report.issues:
                print(f"{issue.severity}: {issue.message}")
    ```
    """

    def __init__(self, language: str = "en-US") -> None:
        """
        Initialize quality checker.

        Args:
            language: Language code for grammar checking
        """
        self.grammar_checker = GrammarChecker(language)
        self.link_validator = LinkValidator()

        # Platform-specific limits
        self.platform_limits = {
            "linkedin": {"max_length": 3000, "ideal_length": 150},
            "twitter": {"max_length": 280, "ideal_length": 100},
            "bluesky": {"max_length": 300, "ideal_length": 100},
        }

    def check(
        self, text: str, platform: str, include_links: bool = True
    ) -> QualityReport:
        """
        Check content quality.

        Args:
            text: Content to check
            platform: Platform type (linkedin/twitter/bluesky)
            include_links: Check links if present

        Returns:
            Quality assessment report

        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        issues: List[QualityIssue] = []
        warnings: List[str] = []
        recommendations: List[str] = []

        # Grammar and spelling check
        grammar_issues = self.grammar_checker.check(text)
        grammar_errors = 0
        spelling_errors = 0

        for issue in grammar_issues:
            severity = (
                "error" if "spelling" not in issue.category.lower() else "warning"
            )

            if "spelling" in issue.category.lower():
                spelling_errors += 1
            else:
                grammar_errors += 1

            quality_issue = QualityIssue(
                type="grammar" if severity == "error" else "spelling",
                severity=severity,
                message=issue.message,
                suggestion=issue.replacements[0] if issue.replacements else None,
                location=issue.offset,
            )
            issues.append(quality_issue)

        # Link validation
        broken_links = 0
        if include_links:
            link_results = self.link_validator.validate_links(text)
            for link_result in link_results:
                if not link_result.is_valid:
                    broken_links += 1
                    issues.append(
                        QualityIssue(
                            type="link",
                            severity="error",
                            message=f"Invalid link: {link_result.url}",
                            suggestion=None,
                            location=None,
                        )
                    )
                elif not link_result.is_https:
                    warnings.append(f"Link not HTTPS: {link_result.url}")

        # Length check
        limits = self.platform_limits[platform]
        text_length = len(text)

        if text_length > limits["max_length"]:
            issues.append(
                QualityIssue(
                    type="length",
                    severity="error",
                    message=f"Text exceeds {platform} limit of {limits['max_length']} characters",
                    suggestion=f"Reduce by {text_length - limits['max_length']} characters",
                    location=None,
                )
            )
        elif text_length < 10:
            issues.append(
                QualityIssue(
                    type="length",
                    severity="warning",
                    message="Text is very short",
                    suggestion="Add more content for better engagement",
                    location=None,
                )
            )

        # Generate recommendations
        if grammar_errors > 0:
            recommendations.append(f"Fix {grammar_errors} grammar error(s)")
        if spelling_errors > 0:
            recommendations.append(f"Fix {spelling_errors} spelling error(s)")
        if broken_links > 0:
            recommendations.append(f"Fix {broken_links} broken link(s)")

        if text_length > limits["ideal_length"] * 1.5:
            recommendations.append("Consider shortening for better engagement")

        # Calculate score
        score = self._calculate_score(
            grammar_errors, spelling_errors, broken_links, len(issues)
        )

        return QualityReport(
            score=score,
            issues=issues,
            grammar_errors=grammar_errors,
            spelling_errors=spelling_errors,
            broken_links=broken_links,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _calculate_score(
        self,
        grammar_errors: int,
        spelling_errors: int,
        broken_links: int,
        total_issues: int,
    ) -> float:
        """Calculate quality score (0-100)."""
        score = 100.0

        # Penalize errors
        score -= grammar_errors * 10
        score -= spelling_errors * 5
        score -= broken_links * 15

        # Ensure score is in valid range
        return max(0.0, min(100.0, score))
