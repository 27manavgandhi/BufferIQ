"""Pydantic models for error responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: Optional[str] = None
    message: str
    type: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    errors: Optional[List[ErrorDetail]] = None


class ValidationErrorResponse(BaseModel):
    """Validation error response."""

    detail: str = "Validation error"
    errors: List[Dict[str, Any]]