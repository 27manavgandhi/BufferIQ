"""
Experiment validators.

Validation functions for experiment data.
"""

from typing import List

from bufferiq.ml.experiments.design.designer import Variant, SUPPORTED_PLATFORMS


def validate_variants(variants: List[Variant]) -> None:
    """
    Validate variants.

    Args:
        variants: List of variants

    Raises:
        ValueError: If validation fails
    """
    if len(variants) < 2:
        raise ValueError("At least 2 variants required")

    # Check traffic allocation
    total = sum(v.traffic_allocation for v in variants)
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"Traffic allocation must sum to 1.0, got {total}")

    # Check control count
    control_count = sum(1 for v in variants if v.is_control)
    if control_count != 1:
        raise ValueError("Exactly one variant must be control")


def validate_platform(platform: str) -> None:
    """
    Validate platform.

    Args:
        platform: Platform name

    Raises:
        ValueError: If platform not supported
    """
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"Platform '{platform}' not supported. Supported: {SUPPORTED_PLATFORMS}"
        )