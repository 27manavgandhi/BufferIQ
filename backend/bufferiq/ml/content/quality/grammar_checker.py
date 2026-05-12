"""
Grammar checking.

Uses language_tool_python for grammar and spelling checks.
"""

from dataclasses import dataclass
from typing import List, Optional

import language_tool_python


@dataclass
class GrammarIssue:
    """Grammar or spelling issue."""

    message: str
    replacements: List[str]
    offset: int
    length: int
    rule_id: str
    category: str


class GrammarChecker:
    """
        Check grammar and spelling.

        Uses LanguageTool for comprehensive grammar checking.

        Example:
    ```python
            checker = GrammarChecker()
            issues = checker.check("This are wrong.")
            for issue in issues:
                print(f"{issue.message}: {issue.replacements}")
    ```
    """

    def __init__(self, language: str = "en-US") -> None:
        """
        Initialize grammar checker.

        Args:
            language: Language code for checking

        Raises:
            ValueError: If language not supported
        """
        try:
            self.tool = language_tool_python.LanguageTool(language)
        except Exception as e:
            raise ValueError(f"Failed to initialize LanguageTool: {e}")

    def check(self, text: str) -> List[GrammarIssue]:
        """
        Check text for grammar and spelling issues.

        Args:
            text: Text to check

        Returns:
            List of grammar issues

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        matches = self.tool.check(text)
        issues = []

        for match in matches:
            issue = GrammarIssue(
                message=match.message,
                replacements=match.replacements[:3],  # Top 3 suggestions
                offset=match.offset,
                length=match.errorLength,
                rule_id=match.ruleId,
                category=match.category,
            )
            issues.append(issue)

        return issues

    def correct(self, text: str) -> str:
        """
        Auto-correct text.

        Args:
            text: Text to correct

        Returns:
            Corrected text

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        return self.tool.correct(text)

    def __del__(self) -> None:
        """Cleanup LanguageTool resources."""
        if hasattr(self, "tool"):
            self.tool.close()
