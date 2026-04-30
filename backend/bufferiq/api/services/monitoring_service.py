"""Monitoring and metrics collection service."""

from prometheus_client import CollectorRegistry, Counter, Histogram

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class MonitoringService:
    """
    Service for collecting Prometheus metrics.

    Tracks:
    - Request counts
    - Latency
    - Error rates
    - Cache hit rates
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize monitoring service."""
        if self._initialized:
            return

        self.registry = CollectorRegistry()

        # Request counter
        self.request_counter = Counter(
            "bufferiq_requests_total",
            "Total number of requests",
            ["endpoint", "platform"],
            registry=self.registry,
        )

        # Latency histogram
        self.latency_histogram = Histogram(
            "bufferiq_request_duration_seconds",
            "Request duration in seconds",
            ["endpoint"],
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry,
        )

        # Error counter
        self.error_counter = Counter(
            "bufferiq_errors_total",
            "Total number of errors",
            ["endpoint", "error_type"],
            registry=self.registry,
        )

        # Cache metrics
        self.cache_hits = Counter(
            "bufferiq_cache_hits_total",
            "Total cache hits",
            registry=self.registry,
        )

        self.cache_misses = Counter(
            "bufferiq_cache_misses_total",
            "Total cache misses",
            registry=self.registry,
        )

        self._initialized = True
        logger.info("MonitoringService initialized")

    def increment_request_count(self, endpoint: str, platform: str) -> None:
        """
        Increment request counter.

        Args:
            endpoint: API endpoint
            platform: Social media platform
        """
        self.request_counter.labels(endpoint=endpoint, platform=platform).inc()

    def record_latency(self, endpoint: str, duration_ms: float) -> None:
        """
        Record request latency.

        Args:
            endpoint: API endpoint
            duration_ms: Duration in milliseconds
        """
        self.latency_histogram.labels(endpoint=endpoint).observe(
            duration_ms / 1000.0
        )

    def increment_error_count(self, endpoint: str, error_type: str) -> None:
        """
        Increment error counter.

        Args:
            endpoint: API endpoint
            error_type: Type of error
        """
        self.error_counter.labels(
            endpoint=endpoint, error_type=error_type
        ).inc()

    def increment_cache_hits(self) -> None:
        """Increment cache hit counter."""
        self.cache_hits.inc()

    def increment_cache_misses(self) -> None:
        """Increment cache miss counter."""
        self.cache_misses.inc()