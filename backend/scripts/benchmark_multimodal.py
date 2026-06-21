#!/usr/bin/env python3
"""
Benchmark multi-modal analysis performance.

Usage:
    python scripts/benchmark_multimodal.py --iterations 100
"""

import argparse
import asyncio
import time
import statistics
from typing import List

from bufferiq.ml.multimodal.images.analyzer import ImageAnalyzer
from bufferiq.ml.multimodal.videos.analyzer import VideoAnalyzer
from bufferiq.ml.multimodal.links.analyzer import LinkPreviewAnalyzer


async def benchmark_image_analysis(iterations: int) -> List[float]:
    """Benchmark image analysis."""
    analyzer = ImageAnalyzer()
    times = []
    
    # Use a sample image (in production, use real test images)
    test_image_url = "https://via.placeholder.com/1920x1080"
    
    print(f"🖼️  Benchmarking image analysis ({iterations} iterations)...")
    
    for i in range(iterations):
        start = time.time()
        try:
            await analyzer.analyze(test_image_url, "linkedin")
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{iterations}")
        except Exception as e:
            print(f"   Error in iteration {i + 1}: {str(e)}")
    
    return times


async def benchmark_link_analysis(iterations: int) -> List[float]:
    """Benchmark link analysis."""
    analyzer = LinkPreviewAnalyzer()
    times = []
    
    test_url = "https://example.com"
    
    print(f"🔗 Benchmarking link analysis ({iterations} iterations)...")
    
    for i in range(iterations):
        start = time.time()
        try:
            await analyzer.analyze(test_url, "linkedin")
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{iterations}")
        except Exception as e:
            print(f"   Error in iteration {i + 1}: {str(e)}")
    
    return times


def print_statistics(name: str, times: List[float], target_p95: float):
    """Print performance statistics."""
    if not times:
        print(f"\n❌ {name}: No valid measurements")
        return
    
    avg = statistics.mean(times)
    median = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 {name}:")
    print(f"   Average: {avg:.2f}ms")
    print(f"   Median: {median:.2f}ms")
    print(f"   P95: {p95:.2f}ms")
    print(f"   Min: {min_time:.2f}ms")
    print(f"   Max: {max_time:.2f}ms")
    
    # Check against target
    if p95 <= target_p95:
        print(f"   ✅ Meets target (P95 ≤ {target_p95}ms)")
    else:
        print(f"   ⚠️  Exceeds target (P95 > {target_p95}ms)")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark multi-modal analysis")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    
    args = parser.parse_args()
    
    print("🚀 Multi-Modal Analysis Performance Benchmark")
    print(f"   Iterations: {args.iterations}")
    print()
    
    # Benchmark image analysis
    image_times = await benchmark_image_analysis(args.iterations)
    print_statistics("Image Analysis", image_times, target_p95=200.0)
    
    # Benchmark link analysis
    link_times = await benchmark_link_analysis(args.iterations)
    print_statistics("Link Analysis", link_times, target_p95=300.0)
    
    # Overall summary
    all_times = image_times + link_times
    if all_times:
        avg_overall = statistics.mean(all_times)
        print(f"\n📈 Overall Average: {avg_overall:.2f}ms")
        
        if avg_overall < 100:
            print("🎉 Excellent performance across all components!")
        elif avg_overall < 200:
            print("✅ Good performance overall")
        else:
            print("⚠️  Performance optimization recommended")
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))