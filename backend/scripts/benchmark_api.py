"""Benchmark API performance."""

import argparse
import statistics
import time
from typing import List

import requests

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark BufferIQ API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests to make",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup requests",
    )
    return parser.parse_args()


def make_prediction_request(base_url: str) -> float:
    """
    Make a prediction request and return latency.

    Args:
        base_url: API base URL

    Returns:
        Latency in milliseconds
    """
    payload = {
        "content": "Benchmark test post for API performance testing",
        "platform": "linkedin",
        "has_media": False,
        "has_link": True,
    }

    start_time = time.time()

    try:
        response = requests.post(
            f"{base_url}/api/v1/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()

        latency_ms = (time.time() - start_time) * 1000
        return latency_ms

    except Exception as e:
        logger.error(f"Request failed: {e}")
        return -1


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate percentile.

    Args:
        values: List of values
        percentile: Percentile (0-100)

    Returns:
        Percentile value
    """
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("BufferIQ API Performance Benchmark")
    print("=" * 80)
    print(f"URL: {args.url}")
    print(f"Requests: {args.requests}")
    print(f"Warmup: {args.warmup}")
    print()

    # Check API is available
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        response.raise_for_status()
        print("✓ API is healthy")
    except Exception as e:
        print(f"✗ API not available: {e}")
        return

    # Warmup
    print(f"\nWarming up ({args.warmup} requests)...")
    for _ in range(args.warmup):
        make_prediction_request(args.url)

    # Benchmark
    print(f"\nRunning benchmark ({args.requests} requests)...")
    latencies = []
    errors = 0

    for i in range(args.requests):
        latency = make_prediction_request(args.url)

        if latency > 0:
            latencies.append(latency)
        else:
            errors += 1

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{args.requests}")

    # Calculate statistics
    if not latencies:
        print("\n✗ No successful requests")
        return

    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    stddev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

    p50 = calculate_percentile(latencies, 50)
    p90 = calculate_percentile(latencies, 90)
    p95 = calculate_percentile(latencies, 95)
    p99 = calculate_percentile(latencies, 99)

    # Calculate throughput
    total_time = sum(latencies) / 1000  # Convert to seconds
    throughput = len(latencies) / total_time if total_time > 0 else 0

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nRequests:")
    print(f"  Total:      {args.requests}")
    print(f"  Successful: {len(latencies)}")
    print(f"  Failed:     {errors}")
    print(f"  Success %:  {len(latencies) / args.requests * 100:.1f}%")

    print(f"\nLatency (ms):")
    print(f"  Mean:       {avg_latency:.2f}")
    print(f"  Median:     {median_latency:.2f}")
    print(f"  Std Dev:    {stddev_latency:.2f}")
    print(f"  Min:        {min_latency:.2f}")
    print(f"  Max:        {max_latency:.2f}")

    print(f"\nPercentiles (ms):")
    print(f"  P50:        {p50:.2f}")
    print(f"  P90:        {p90:.2f}")
    print(f"  P95:        {p95:.2f}")
    print(f"  P99:        {p99:.2f}")

    print(f"\nThroughput:")
    print(f"  Requests/s: {throughput:.2f}")

    # Performance assessment
    print("\nPerformance Assessment:")
    if p95 < 100:
        print("  ✓ Excellent (p95 < 100ms)")
    elif p95 < 200:
        print("  ✓ Good (p95 < 200ms)")
    elif p95 < 500:
        print("  ⚠ Acceptable (p95 < 500ms)")
    else:
        print("  ✗ Needs improvement (p95 > 500ms)")

    print("=" * 80)


if __name__ == "__main__":
    main()