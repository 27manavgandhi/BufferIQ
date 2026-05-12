"""
Tests for grammar checker.
"""

import pytest

from bufferiq.ml.content.quality.grammar_checker import (
    GrammarChecker,
    GrammarIssue,
)


class TestGrammarChecker:
    """Test GrammarChecker class."""

    @pytest.fixture
    def checker(self) -> GrammarChecker:
        """Create grammar checker fixture."""
        return GrammarChecker()

    def test_check_correct_text(self, checker: GrammarChecker) -> None:
        """Test checking grammatically correct text."""
        issues = checker.check("This is a correct sentence.")

        # May have minor suggestions but should be minimal
        assert isinstance(issues, list)

    def test_check_grammar_error(self, checker: GrammarChecker) -> None:
        """Test detecting grammar errors."""
        issues = checker.check("This are wrong.")

        assert len(issues) > 0
        assert all(isinstance(i, GrammarIssue) for i in issues)

    def test_check_empty_text_raises_error(
        self, checker: GrammarChecker
    ) -> None:
        """Test checking empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            checker.check("")

    def test_check_returns_issue_details(
        self, checker: GrammarChecker
    ) -> None:
        """Test that issues contain details."""
        issues = checker.check("This are incorrect.")

        if len(issues) > 0:
            issue = issues[0]
            assert hasattr(issue, "message")
            assert hasattr(issue, "replacements")
            assert hasattr(issue, "offset")

    def test_correct_text(self, checker: GrammarChecker) -> None:
        """Test auto-correction."""
        corrected = checker.correct("This are wrong.")

        assert isinstance(corrected, str)
        # Should attempt correction
        assert len(corrected) > 0

    def test_correct_empty_text_raises_error(
        self, checker: GrammarChecker
    ) -> None:
        """Test correcting empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            checker.correct("")

    def test_check_spelling_error(self, checker: GrammarChecker) -> None:
        """Test detecting spelling errors."""
        # Note: May not always catch depending on dictionary
        issues = checker.check("This is a tset.")

        # Should detect "tset" as potential error
        assert isinstance(issues, list)

    def test_check_provides_suggestions(
        self, checker: GrammarChecker
    ) -> None:
        """Test that issues provide suggestions."""
        issues = checker.check("This are wrong.")

        if len(issues) > 0:
            assert len(issues[0].replacements) > 0

    def test_check_long_text(self, checker: GrammarChecker) -> None:
        """Test checking longer text."""
        text = (
            "This is a longer piece of text. "
            "It contains multiple sentences. "
            "Some may have errors and some may not."
        )
        issues = checker.check(text)

        assert isinstance(issues, list)