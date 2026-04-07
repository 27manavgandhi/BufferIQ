"""
Base model with common fields for all database models.

Provides timestamp tracking and primary key for all models.
"""

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    Base class for all database models.

    Provides common fields:
    - id: Auto-incrementing primary key
    - created_at: Timestamp when record was created
    - updated_at: Timestamp when record was last updated
    """

    pass


class TimestampMixin:
    """Mixin to add timestamp fields to models."""

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )
