"""Tests for custom middleware."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from bufferiq.api.middleware import TimingMiddleware, LoggingMiddleware


@pytest.fixture
def app_with_timing():
    """Create app with timing middleware."""
    app = FastAPI()
    app.add_middleware(TimingMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    return app


@pytest.fixture
def app_with_logging():
    """Create app with logging middleware."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    return app


def test_timing_middleware_adds_headers(app_with_timing):
    """Test timing middleware adds headers."""
    client = TestClient(app_with_timing)
    response = client.get("/test")

    assert "x-request-id" in response.headers
    assert "x-process-time" in response.headers


def test_timing_middleware_request_id_unique(app_with_timing):
    """Test request IDs are unique."""
    client = TestClient(app_with_timing)

    response1 = client.get("/test")
    response2 = client.get("/test")

    id1 = response1.headers["x-request-id"]
    id2 = response2.headers["x-request-id"]

    assert id1 != id2


def test_timing_middleware_timing_format(app_with_timing):
    """Test timing format."""
    client = TestClient(app_with_timing)
    response = client.get("/test")

    timing = response.headers["x-process-time"]
    assert timing.endswith("ms")

    # Should be parseable as float
    timing_value = float(timing.replace("ms", ""))
    assert timing_value >= 0


def test_logging_middleware_logs_requests(app_with_logging, caplog):
    """Test logging middleware logs requests."""
    client = TestClient(app_with_logging)

    with caplog.at_level("INFO"):
        client.get("/test")

    # Check logs contain request info
    assert any("Request started" in record.message for record in caplog.records)
    assert any("Request completed" in record.message for record in caplog.records)


def test_logging_middleware_logs_errors(app_with_logging, caplog):
    """Test logging middleware logs errors."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")

    client = TestClient(app)

    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            client.get("/error")

    assert any("Request failed" in record.message for record in caplog.records)