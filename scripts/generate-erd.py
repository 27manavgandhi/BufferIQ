"""
Generate Entity Relationship Diagram from SQLAlchemy models.

Creates a visual ERD using eralchemy library.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from eralchemy import render_er
    from bufferiq.domain.base import Base
    import bufferiq.domain.models  # noqa: F401 - Import to register models
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install eralchemy: pip install eralchemy")
    sys.exit(1)


def generate_erd() -> None:
    """Generate ERD diagram from models."""
    output_path = Path(__file__).parent.parent / "docs" / "erd.png"

    print("Generating ERD diagram...")
    print(f"Output: {output_path}")

    try:
        # Generate from SQLAlchemy metadata
        render_er(Base.metadata, str(output_path))
        print(f"✅ ERD generated successfully: {output_path}")
    except Exception as e:
        print(f"❌ Failed to generate ERD: {e}")
        sys.exit(1)


if __name__ == "__main__":
    generate_erd()
