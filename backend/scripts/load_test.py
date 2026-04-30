"""Load test the BufferIQ API."""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load test BufferIQ API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="Number of concurrent users",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--ramp-up",
        type=int,
        default=10,
        help="Ramp-up time in seconds",
    )
    return parser.parse_args()


# Sample content for testing
SAMPLE_CONTENT = [
    "Just shipped a new feature! 🚀",
    "Excited to announce our latest product update",
    "Check out this amazing insight from our team",
    "Thoughts on the future of AI and ML",
    "Great discussion at today's conference",
    "New blog post about software engineering best practices",
    "Celebrating our team's achievements this quarter",
    "Looking forward to the next challenge",
    "Innovation starts with curiosity",
    "Building the future, one line of code at a time",
]

PLATFORMS = ["linkedin", "twitter", "bluesky"]


def generate_request() -> dict:
    """Generate random prediction request."""
    return {
        "content": random.choice(SAMPLE_CONTENT),
        "platform": random.choice(PLATFORMS),
        "has_media": random.choice([True, False]),
        "has_link": random.choice([True, False]),
    }


def make_request(base_url: str) -> Dict[str, any]:
    """
    Make a single request.

    Args:
        base_url: API base URL

    Returns:
        Result dictionary
    """
    payload = generate_request()

    start_time = time.time()

    try:
        response = requests.post(
            f"{base_url}/api/v1/predict",
            json=payload,
            timeout=10,
        )

        latency_ms = (time.time() - start_time) * 1000

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "success": False,
            "status_code": 0,
            "latency_ms": latency_ms,
            "error": str(e),
        }


def user_simulation(base_url: str, duration: int) -> List[Dict]:
    """
    Simulate a single user making requests.

    Args:
        base_url: API base URL
        duration: Duration in seconds

    Returns:
        List of request results
    """
    results = []
    end_time = time.time() + duration

    while time.time() < end_time:
        result = make_request(base_url)
        results.append(result)

        # Think time (1-3 seconds)
        time.sleep(random.uniform(1, 3))

    return results


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("BufferIQ API Load Test")
    print("=" * 80)
    print(f"URL:          {args.url}")
    print(f"Users:        {args.users}")
    print(f"Duration:     {args.duration}s")
    print(f"Ramp-up:      {args.ramp_up}s")
    print()

    # Check API
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        response.raise_for_status()
        print("✓ API is healthy")
    except Exception as e:
        print(f"✗ API not available: {e}")
        return

    # Start load test
    print(f"\nStarting load test...")
    start_time = time.time()

    all_results = []

    with ThreadPoolExecutor(max_workers=args.users) as executor:
        # Submit user simulations with ramp-up
        futures = []
        for i in range(args.users):
            # Stagger user starts during ramp-up period
            delay = (args.ramp_up / args.users) * i
            time.sleep(delay)

            future = executor.submit(
                user_simulation, args.url, args.duration
            )
            futures.append(future)

            print(f"  Started user {i + 1}/{args.users}")

        # Collect results
        print("\nWaiting for test to complete...")
        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)

    total_time = time.time() - start_time

    # Analyze results
    total_requests = len(all_results)
    successful = sum(1 for r in all_results if r["success"])
    failed = total_requests - successful

    latencies = [r["latency_ms"] for r in all_results if r["success"]]

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    else:
        avg_latency = min_latency = max_latency = p95_latency = 0

    throughput = total_requests / total_time

    # Print results
    print("\n" + "=" * 80)
    print("LOAD TEST RESULTS")
    print("=" * 80)

    print(f"\nTest Summary:")
    print(f"  Duration:       {total_time:.1f}s")
    print(f"  Total Requests: {total_requests}")
    print(f"  Successful:     {successful}")
    print(f"  Failed:         {failed}")
    print(f"  Success Rate:   {successful / total_requests * 100:.1f}%")

    print(f"\nPerformance:")
    print(f"  Throughput:     {throughput:.2f} req/s")
    print(f"  Avg Latency:    {avg_latency:.2f}ms")
    print(f"  Min Latency:    {min_latency:.2f}ms")
    print(f"  Max Latency:    {max_latency:.2f}ms")
    print(f"  P95 Latency:    {p95_latency:.2f}ms")

    # Status code distribution
    status_codes = {}
    for result in all_results:
        code = result["status_code"]
        status_codes[code] = status_codes.get(code, 0) + 1

    print(f"\nStatus Codes:")
    for code, count in sorted(status_codes.items()):
        print(f"  {code}: {count}")

    # Error summary
    if failed > 0:
        print(f"\nErrors:")
        error_types = {}
        for result in all_results:
            if not result["success"] and "error" in result:
                error = result["error"]
                error_types[error] = error_types.get(error, 0) + 1

        for error, count in sorted(
            error_types.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {error}: {count}")

    print("=" * 80)


if __name__ == "__main__":
    main()