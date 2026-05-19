"""
Hashtag normalizer and variant detector.

Handles normalization, abbreviations, and common misspellings.
"""

from typing import Dict, List, Set


class HashtagNormalizer:
    """
    Normalize and clean hashtags.

    Handles abbreviations, common misspellings,
    and platform-specific conventions.

    Example:
```python
        normalizer = HashtagNormalizer()
        canonical = normalizer.normalize("#AI_Tech")
        # Returns: "aitech"

        variants = normalizer.get_variants("ai")
        # Returns: ["artificialintelligence", "aitech", "ai"]
```
    """

    def __init__(self) -> None:
        """Initialize normalizer."""
        # Canonical forms and their variants
        self.canonical_map: Dict[str, List[str]] = {
            "ai": ["artificialintelligence", "aitech", "artificialint"],
            "ml": ["machinelearning", "mlai", "machinelearn"],
            "seo": ["searchengineoptimization", "searchoptimization"],
            "socialmedia": ["sm", "smm", "socialmediamarketing"],
            "digitalmarketing": ["dm", "digimarketing", "digitalmarket"],
            "contentmarketing": ["cm", "contentmarket"],
            "b2b": ["btob", "businesstobusiness"],
            "b2c": ["btoc", "businesstoconsumer"],
            "saas": ["softwareasaservice"],
            "startup": ["startups", "startuplife"],
            "tech": ["technology", "techlife"],
            "innovation": ["innovate", "innovative"],
            "leadership": ["leader", "leaders"],
            "marketing": ["mktg", "marketingtips"],
            "business": ["biz", "bizdev"],
        }

        # Build reverse map (variant -> canonical)
        self.variant_to_canonical: Dict[str, str] = {}
        for canonical, variants in self.canonical_map.items():
            for variant in variants:
                self.variant_to_canonical[variant] = canonical
            self.variant_to_canonical[canonical] = canonical

        # Common misspellings
        self.misspellings: Dict[str, str] = {
            "artifical": "artificial",
            "machinelearninng": "machinelearning",
            "entrepreneuer": "entrepreneur",
            "buisness": "business",
            "succes": "success",
        }

    def normalize(self, hashtag: str) -> str:
        """
        Normalize hashtag to canonical form.

        Args:
            hashtag: Raw hashtag (with or without #)

        Returns:
            Normalized hashtag (lowercase, no #, no underscores)
        """
        # Remove # and lowercase
        normalized = hashtag.lstrip("#").lower()

        # Remove underscores and spaces
        normalized = normalized.replace("_", "").replace(" ", "")

        # Fix common misspellings
        for misspelling, correction in self.misspellings.items():
            if misspelling in normalized:
                normalized = normalized.replace(misspelling, correction)

        # Map to canonical if it's a known variant
        if normalized in self.variant_to_canonical:
            return self.variant_to_canonical[normalized]

        return normalized

    def get_variants(self, canonical: str) -> List[str]:
        """
        Get known variants of canonical form.

        Args:
            canonical: Canonical hashtag form

        Returns:
            List of variants including canonical
        """
        canonical_normalized = canonical.lower()

        if canonical_normalized in self.canonical_map:
            variants = self.canonical_map[canonical_normalized].copy()
            if canonical_normalized not in variants:
                variants.insert(0, canonical_normalized)
            return variants

        return [canonical_normalized]

    def get_canonical(self, hashtag: str) -> str:
        """
        Get canonical form of hashtag.

        Args:
            hashtag: Hashtag (with or without #)

        Returns:
            Canonical form
        """
        normalized = self.normalize(hashtag)
        return self.variant_to_canonical.get(normalized, normalized)

    def are_variants(self, hashtag1: str, hashtag2: str) -> bool:
        """
        Check if two hashtags are variants of each other.

        Args:
            hashtag1: First hashtag
            hashtag2: Second hashtag

        Returns:
            True if they are variants
        """
        canonical1 = self.get_canonical(hashtag1)
        canonical2 = self.get_canonical(hashtag2)
        return canonical1 == canonical2