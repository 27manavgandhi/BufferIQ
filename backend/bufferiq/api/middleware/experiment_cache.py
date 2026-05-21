"""
Experiment cache middleware.

Caches experiment configurations and assignments.
"""

from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class ExperimentCacheMiddleware(BaseHTTPMiddleware):
    """
    Cache experiment data.

    Example:
```python
        app.add_middleware(ExperimentCacheMiddleware)
```
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process request.

        Args:
            request: Request
            call_next: Next middleware

        Returns:
            Response
        """
        # Add caching logic here
        response = await call_next(request)
        return response