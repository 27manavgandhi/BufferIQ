"""Benchmark segmentation performance."""

import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from bufferiq.ml.segmentation.preprocessing.preprocessor import AudienceDataPreprocessor
from bufferiq.ml.segmentation.clustering.optimizer import ClusteringOptimizer
from bufferiq.ml.segmentation.clustering.kmeans import KMeansClusterer
from bufferiq.ml.segmentation.personas.persona_builder import PersonaBuilder
from bufferiq.ml.segmentation.types import AudienceDataPoint


def generate_test_data(n_samples: int = 1000) -> list:
    """Generate test audience data."""
    data = []
    for i in range(n_samples):
        data.append(
            AudienceDataPoint(
                user_id=f"user_{i}",
                platform="linkedin",
                follower_count=np.random.randint(100, 10000),
                following_count=np.random.randint(100, 5000),
                post_count=np.random.randint(1, 500),
                avg_engagement_rate=np.random.random(),
                engagement_history=[],
                interaction_types={
                    "likes": np.random.randint(0, 100),
                    "comments": np.random.randint(0, 20),
                    "shares": np.random.randint(0, 5),
                },
                active_hours=[9, 12, 14, 18],
                active_days=[0, 1, 2, 3, 4],
                topics_engaged=["tech", "business"],
                content_types_engaged=["text", "image"],
                account_age_days=np.random.randint(30, 3650),
                verified=np.random.random() < 0.1,
                bio_keywords=["engineer"],
                location="San Francisco",
                language="en",
            )
        )
    return data


def benchmark_preprocessing(data: list) -> dict:
    """Benchmark preprocessing."""
    preprocessor = AudienceDataPreprocessor()

    start = time.time()
    features = preprocessor.process(data, "linkedin")
    elapsed = (time.time() - start) * 1000

    return {
        "n_samples": len(data),
        "time_ms": elapsed,
        "time_per_sample_ms": elapsed / len(data),
        "within_target": elapsed < 500,
    }


def benchmark_clustering(feature_matrix: np.ndarray) -> dict:
    """Benchmark clustering."""
    clusterer = KMeansClusterer()

    start = time.time()
    result = clusterer.fit(feature_matrix, 5, "linkedin")
    elapsed = (time.time() - start) * 1000

    return {
        "n_samples": feature_matrix.shape[0],
        "n_features": feature_matrix.shape[1],
        "time_ms": elapsed,
        "silhouette_score": result.silhouette_score,
        "within_target": elapsed < 300,
    }


def benchmark_optimizer(feature_matrix: np.ndarray) -> dict:
    """Benchmark clustering optimizer."""
    optimizer = ClusteringOptimizer()

    start = time.time()
    config = optimizer.find_optimal(feature_matrix, "linkedin")
    elapsed = (time.time() - start) * 1000

    return {
        "n_samples": feature_matrix.shape[0],
        "time_ms": elapsed,
        "optimal_clusters": config.n_clusters,
        "within_target": elapsed < 1000,
    }


def main() -> None:
    """Run benchmarks."""
    print("=== SEGMENTATION BENCHMARKS ===\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {},
    }

    # Test different data sizes
    for n in [100, 500, 1000]:
        print(f"Benchmarking with {n} samples...")

        data = generate_test_data(n)
        preprocessor = AudienceDataPreprocessor()
        features = preprocessor.process(data, "linkedin")
        feature_matrix = np.array([f.feature_vector for f in features])

        # Preprocessing
        print(f"  Preprocessing...", end=" ", flush=True)
        prep_result = benchmark_preprocessing(data)
        results["benchmarks"][f"preprocessing_{n}"] = prep_result
        print(f"✓ {prep_result['time_ms']:.0f}ms")

        # Clustering
        print(f"  Clustering...", end=" ", flush=True)
        clust_result = benchmark_clustering(feature_matrix)
        results["benchmarks"][f"clustering_{n}"] = clust_result
        print(f"✓ {clust_result['time_ms']:.0f}ms")

        # Optimizer
        if n <= 500:
            print(f"  Optimizer...", end=" ", flush=True)
            opt_result = benchmark_optimizer(feature_matrix)
            results["benchmarks"][f"optimizer_{n}"] = opt_result
            print(f"✓ {opt_result['time_ms']:.0f}ms")

    # Save results
    output_path = Path("outputs/benchmarks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nBenchmark results saved to: {output_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    all_passed = all(
        b.get("within_target", False) for b in results["benchmarks"].values()
    )
    print(f"All performance targets met: {'✓' if all_passed else '✗'}")


if __name__ == "__main__":
    main()