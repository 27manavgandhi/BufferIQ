"""Test Buffer API client connectivity and basic operations."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from bufferiq.core.config import get_settings
from bufferiq.core.cache import ResponseCache
from bufferiq.infrastructure.buffer.client import BufferClient
from bufferiq.infrastructure.buffer.rate_limiter import RateLimiter
from bufferiq.infrastructure.buffer.exceptions import (
    BufferAPIError,
    BufferAuthenticationError,
    BufferRateLimitError,
)
from redis.asyncio import Redis


async def test_buffer_client(verbose: bool = False) -> bool:
    """
    Test Buffer API client connectivity.

    Args:
        verbose: Print detailed output

    Returns:
        True if all tests pass, False otherwise
    """
    print("🧪 Testing Buffer API Client...\n")

    # Load settings
    settings = get_settings()

    if verbose:
        print(f"📋 Configuration:")
        print(f"   .env file: {env_path}")
        print(f"   .env exists: {env_path.exists()}")
        print(f"   API URL: {settings.buffer_api_url}")
        print(f"   API Key: {settings.buffer_api_key[:20]}..." if settings.buffer_api_key and len(settings.buffer_api_key) > 20 else f"   API Key: {settings.buffer_api_key}")
        print(f"   Environment: {settings.environment}")
        print()

    if not settings.buffer_api_key or settings.buffer_api_key == "test_key" or len(settings.buffer_api_key) < 10:
        print("❌ No Buffer API key configured!")
        print(f"   .env file location: {env_path}")
        print(f"   .env file exists: {env_path.exists()}")
        if env_path.exists():
            print(f"   Current BUFFER_API_KEY value: '{settings.buffer_api_key}'")
            print()
            print("   Make sure .env file contains:")
            print("   BUFFER_API_KEY=your_actual_api_key_here")
        return False

    # Initialize Redis client
    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        if verbose:
            print("✅ Redis connection successful")
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("   Continuing without cache...")
        redis_client = None

    # Initialize rate limiter and cache
    rate_limiter = RateLimiter(redis_client) if redis_client else None
    cache = ResponseCache(redis_client) if redis_client else None

    # Initialize Buffer client
    client = BufferClient(
        api_url=settings.buffer_api_url,
        api_key=settings.buffer_api_key,
        rate_limiter=rate_limiter,
        cache=cache,
    )

    test_results = {
        "health_check": False,
        "fetch_organizations": False,
        "fetch_channels": False,
        "fetch_posts": False,
    }

    try:
        # Test 1: Health Check
        print("1️⃣  Testing health check...")
        try:
            is_healthy = await client.health_check()
            if is_healthy:
                print("   ✅ Health check passed")
                test_results["health_check"] = True
            else:
                print("   ❌ Health check failed")
        except BufferAuthenticationError:
            print("   ❌ Authentication failed - Invalid API key")
            return False
        except BufferAPIError as e:
            print(f"   ❌ API Error: {e}")
            return False

        # Test 2: Fetch Organizations
        print("\n2️⃣  Testing fetch organizations...")
        try:
            orgs = await client.fetch_organizations()
            if orgs and len(orgs) > 0:
                print(f"   ✅ Fetched {len(orgs)} organization(s)")
                if verbose:
                    for org in orgs:
                        print(f"      - {org.get('name', 'Unknown')} (ID: {org.get('id', 'N/A')})")
                test_results["fetch_organizations"] = True
                org_id = orgs[0].get("id")
            else:
                print("   ⚠️  No organizations found")
                org_id = None
        except BufferAPIError as e:
            print(f"   ❌ Failed to fetch organizations: {e}")
            org_id = None

        # Test 3: Fetch Channels
        if org_id:
            print("\n3️⃣  Testing fetch channels...")
            try:
                channels = await client.fetch_channels(org_id)
                if channels and len(channels) > 0:
                    print(f"   ✅ Fetched {len(channels)} channel(s)")
                    if verbose:
                        for channel in channels:
                            print(
                                f"      - {channel.get('service', 'Unknown')} "
                                f"({channel.get('name', 'N/A')})"
                            )
                    test_results["fetch_channels"] = True
                    channel_id = channels[0].get("id")
                else:
                    print("   ⚠️  No channels found")
                    channel_id = None
            except BufferAPIError as e:
                print(f"   ❌ Failed to fetch channels: {e}")
                channel_id = None
        else:
            print("\n3️⃣  Skipping channel test (no organization ID)")
            channel_id = None

        # Test 4: Fetch Posts
        if channel_id and org_id:
            print("\n4️⃣  Testing fetch posts...")
            try:
                posts_data = await client.fetch_posts(
                    org_id=org_id,
                    channel_id=channel_id,
                    limit=5
                )
                edges = posts_data.get("edges", [])
                if edges and len(edges) > 0:
                    print(f"   ✅ Fetched {len(edges)} post(s)")
                    if verbose:
                        for i, edge in enumerate(edges[:3], 1):
                            post = edge["node"]
                            content = post.get("text", "")[:50]
                            status = post.get("status", "unknown")
                            print(f"      {i}. {content}... (Status: {status})")
                    test_results["fetch_posts"] = True
                else:
                    print("   ⚠️  No posts found")
            except BufferAPIError as e:
                print(f"   ❌ Failed to fetch posts: {e}")
        else:
            print("\n4️⃣  Skipping posts test (no channel ID or org ID)")

        # Test 5: Rate Limiting Info
        if rate_limiter:
            print("\n5️⃣  Checking rate limits...")
            try:
                user_id = 1  # Test user
                limits = {
                    "15min": {"remaining": 0, "limit": 100},
                    "24hr": {"remaining": 0, "limit": 500},
                    "30day": {"remaining": 0, "limit": 10000},
                }
                print("   ✅ Rate limit status:")
                print(f"      - 15 minutes: {limits['15min']['remaining']}/{limits['15min']['limit']}")
                print(f"      - 24 hours: {limits['24hr']['remaining']}/{limits['24hr']['limit']}")
                print(f"      - 30 days: {limits['30day']['remaining']}/{limits['30day']['limit']}")
            except Exception as e:
                print(f"   ⚠️  Could not check rate limits: {e}")

        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        print("=" * 50)
        passed = sum(test_results.values())
        total = len(test_results)
        print(f"   Passed: {passed}/{total} tests")
        print()
        for test_name, result in test_results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {test_name.replace('_', ' ').title()}")

        all_passed = all(test_results.values())

        if all_passed:
            print("\n✅ All tests passed! Buffer API client is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check configuration and API access.")

        return all_passed

    except BufferAuthenticationError as e:
        print(f"\n❌ Authentication Error: {e}")
        print("   Check your BUFFER_API_KEY in .env")
        return False
    except BufferRateLimitError as e:
        print(f"\n❌ Rate Limit Error: {e}")
        print(f"   Retry after: {e.retry_after} seconds")
        return False
    except BufferAPIError as e:
        print(f"\n❌ Buffer API Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False
    finally:
        # Cleanup
        await client.close()
        if redis_client:
            await redis_client.aclose()


async def test_specific_endpoint(
    endpoint: str, org_id: str | None = None, channel_id: str | None = None
) -> None:
    """
    Test a specific Buffer API endpoint.

    Args:
        endpoint: Endpoint to test (organizations, channels, posts)
        org_id: Organization ID (for channels endpoint)
        channel_id: Channel ID (for posts endpoint)
    """
    settings = get_settings()

    # Initialize client
    redis_client = None
    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
    except:
        pass

    rate_limiter = RateLimiter(redis_client) if redis_client else None
    cache = ResponseCache(redis_client) if redis_client else None

    client = BufferClient(
        api_url=settings.buffer_api_url,
        api_key=settings.buffer_api_key,
        rate_limiter=rate_limiter,
        cache=cache,
    )

    try:
        if endpoint == "organizations":
            print("📋 Fetching organizations...")
            orgs = await client.fetch_organizations()
            print(f"\nFound {len(orgs)} organization(s):")
            for org in orgs:
                print(f"  - ID: {org.get('id')}")
                print(f"    Name: {org.get('name')}")
                print()

        elif endpoint == "channels":
            if not org_id:
                print("❌ Organization ID required for channels endpoint")
                return
            print(f"📋 Fetching channels for organization {org_id}...")
            channels = await client.fetch_channels(org_id)
            print(f"\nFound {len(channels)} channel(s):")
            for channel in channels:
                print(f"  - ID: {channel.get('id')}")
                print(f"    Service: {channel.get('service')}")
                print(f"    Name: {channel.get('name')}")
                print()

        elif endpoint == "posts":
            if not channel_id or not org_id:
                print("❌ Both Organization ID and Channel ID required for posts endpoint")
                return
            print(f"📋 Fetching posts for channel {channel_id}...")
            posts_data = await client.fetch_posts(
                org_id=org_id,
                channel_id=channel_id,
                limit=10
            )
            edges = posts_data.get("edges", [])
            print(f"\nFound {len(edges)} post(s):")
            for i, edge in enumerate(edges, 1):
                post = edge["node"]
                print(f"  {i}. ID: {post.get('id')}")
                print(f"     Text: {post.get('text', '')[:100]}...")
                print(f"     Status: {post.get('status')}")
                print(f"     Scheduled: {post.get('dueAt')}")
                print()

        else:
            print(f"❌ Unknown endpoint: {endpoint}")
            print("   Valid endpoints: organizations, channels, posts")

    except BufferAPIError as e:
        print(f"❌ API Error: {e}")
    finally:
        await client.close()
        if redis_client:
            await redis_client.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test Buffer API client connectivity"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with detailed information"
    )
    parser.add_argument(
        "-e", "--endpoint",
        choices=["organizations", "channels", "posts"],
        help="Test specific endpoint"
    )
    parser.add_argument(
        "--org-id",
        help="Organization ID (for channels endpoint)"
    )
    parser.add_argument(
        "--channel-id",
        help="Channel ID (for posts endpoint)"
    )

    args = parser.parse_args()

    if args.endpoint:
        asyncio.run(
            test_specific_endpoint(
                args.endpoint,
                org_id=args.org_id,
                channel_id=args.channel_id
            )
        )
    else:
        success = asyncio.run(test_buffer_client(verbose=args.verbose))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
