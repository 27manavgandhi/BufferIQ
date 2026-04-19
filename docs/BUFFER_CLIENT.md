# Buffer API Client Documentation

## Overview

The Buffer API client provides a type-safe interface to interact with the Buffer API, handling authentication, rate limiting, and error handling.

## Components

### BufferClient

Main client for interacting with Buffer API.

```python
from bufferiq.infrastructure.buffer.client import BufferClient

client = BufferClient(access_token="your_token")

# Get user profile
profile = await client.get_user_profile()

# Get profiles
profiles = await client.get_profiles()

# Get posts
posts = await client.get_posts(profile_id="123")
```

### Rate Limiter

Automatically handles Buffer API rate limits.

```python
from bufferiq.infrastructure.buffer.rate_limiter import RateLimiter

limiter = RateLimiter(max_requests=60, window_seconds=60)
```

### Types

Type definitions for Buffer API responses.

```python
from bufferiq.infrastructure.buffer.types import (
    BufferProfile,
    BufferPost,
    BufferStats
)
```

## Usage

See `scripts/test-buffer-client.py` for examples.

## API Documentation

Official Buffer API docs: https://buffer.com/developers/api