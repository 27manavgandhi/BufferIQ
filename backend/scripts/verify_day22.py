"""Verify Day 22 implementation completeness."""

import sys
from pathlib import Path


def check_file_exists(path: str, description: str) -> bool:
    """Check if file exists."""
    if Path(path).exists():
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} - MISSING")
        return False


def verify_implementation() -> None:
    """Verify all Day 22 files are present."""
    print("=== DAY 22 IMPLEMENTATION VERIFICATION ===\n")

    results = {
        "preprocessing": [],
        "clustering": [],
        "personas": [],
        "tracking": [],
        "recommendations": [],
        "prediction": [],
        "visualization": [],
        "intelligence": [],
        "domain_models": [],
        "api": [],
        "tests": [],
        "config": [],
    }

    # Check preprocessing files
    print("Preprocessing:")
    results["preprocessing"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/preprocessing/__init__.py",
            "Preprocessor init",
        )
    )
    results["preprocessing"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/preprocessing/normalizer.py",
            "Data normalizer",
        )
    )
    results["preprocessing"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/preprocessing/aggregator.py",
            "Engagement aggregator",
        )
    )
    results["preprocessing"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/preprocessing/feature_extractor.py",
            "Feature extractor",
        )
    )
    results["preprocessing"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/preprocessing/temporal_features.py",
            "Temporal features",
        )
    )
    results["preprocessing"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/preprocessing/validator.py",
            "Data validator",
        )
    )
    results["preprocessing"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/preprocessing/preprocessor.py",
            "Main preprocessor",
        )
    )

    # Check clustering files
    print("\nClustering:")
    results["clustering"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/clustering/__init__.py",
            "Clustering init",
        )
    )
    results["clustering"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/clustering/kmeans.py", "K-Means"
        )
    )
    results["clustering"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/clustering/dbscan.py", "DBSCAN"
        )
    )
    results["clustering"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/clustering/hierarchical.py",
            "Hierarchical",
        )
    )
    results["clustering"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/clustering/gmm.py", "Gaussian Mixture"
        )
    )
    results["clustering"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/clustering/optimizer.py", "Optimizer"
        )
    )
    results["clustering"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/clustering/validator.py",
            "Clustering validator",
        )
    )
    results["clustering"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/clustering/ensemble.py", "Ensemble"
        )
    )

    # Check personas files
    print("\nPersonas:")
    results["personas"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/personas/__init__.py",
            "Personas init",
        )
    )
    results["personas"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/personas/demographic_inferrer.py",
            "Demographic inferrer",
        )
    )
    results["personas"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/personas/behavioral_profiler.py",
            "Behavioral profiler",
        )
    )
    results["personas"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/personas/interest_mapper.py",
            "Interest mapper",
        )
    )
    results["personas"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/personas/content_preference_modeler.py",
            "Content preference modeler",
        )
    )
    results["personas"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/personas/persona_builder.py",
            "Persona builder",
        )
    )
    results["personas"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/personas/namer.py",
            "Persona namer",
        )
    )

    # Check tracking files
    print("\nTracking:")
    results["tracking"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/tracking/__init__.py",
            "Tracking init",
        )
    )
    results["tracking"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/tracking/evolution_tracker.py",
            "Evolution tracker",
        )
    )
    results["tracking"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/tracking/migration_tracker.py",
            "Migration tracker",
        )
    )
    results["tracking"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/tracking/drift_detector.py",
            "Drift detector",
        )
    )
    results["tracking"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/tracking/health_scorer.py",
            "Health scorer",
        )
    )

    # Check recommendations files
    print("\nRecommendations:")
    results["recommendations"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/recommendations/__init__.py",
            "Recommendations init",
        )
    )
    results["recommendations"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/recommendations/content_recommender.py",
            "Content recommender",
        )
    )
    results["recommendations"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/recommendations/timing_recommender.py",
            "Timing recommender",
        )
    )
    results["recommendations"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/recommendations/style_recommender.py",
            "Style recommender",
        )
    )
    results["recommendations"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/recommendations/hashtag_recommender.py",
            "Hashtag recommender",
        )
    )
    results["recommendations"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/recommendations/engine.py",
            "Recommendation engine",
        )
    )

    # Check prediction files
    print("\nPrediction:")
    results["prediction"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/prediction/__init__.py",
            "Prediction init",
        )
    )
    results["prediction"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/prediction/segment_model.py",
            "Segment model",
        )
    )
    results["prediction"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/prediction/cross_segment.py",
            "Cross-segment analyzer",
        )
    )
    results["prediction"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/prediction/calibrator.py",
            "Model calibrator",
        )
    )
    results["prediction"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/prediction/predictor.py",
            "Engagement predictor",
        )
    )

    # Check visualization files
    print("\nVisualization:")
    results["visualization"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/visualization/__init__.py",
            "Visualization init",
        )
    )
    results["visualization"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/visualization/cluster_plotter.py",
            "Cluster plotter",
        )
    )
    results["visualization"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/visualization/persona_renderer.py",
            "Persona renderer",
        )
    )
    results["visualization"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/visualization/evolution_plotter.py",
            "Evolution plotter",
        )
    )

    # Check intelligence files
    print("\nIntelligence:")
    results["intelligence"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/intelligence/__init__.py",
            "Intelligence init",
        )
    )
    results["intelligence"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/intelligence/service.py",
            "Intelligence service",
        )
    )
    results["intelligence"].append(
        check_file_exists(
            "backend/bufferiq/ml/segmentation/intelligence/analyzer.py",
            "Segmentation analyzer",
        )
    )

    # Check API files
    print("\nAPI Layer:")
    results["api"].append(
        check_file_exists(
            "backend/bufferiq/api/models/segmentation.py", "API models"
        )
    )
    results["api"].append(
        check_file_exists(
            "backend/bufferiq/api/routers/segmentation.py", "API router"
        )
    )
    results["api"].append(
        check_file_exists(
            "backend/bufferiq/api/services/segmentation_service.py",
            "API service",
        )
    )
    results["api"].append(
        check_file_exists(
            "backend/bufferiq/api/dependencies/segmentation.py",
            "API dependencies",
        )
    )

    # Print summary
    print("\n=== SUMMARY ===")
    total_checks = sum(len(v) for v in results.values())
    passed_checks = sum(sum(v) for v in results.values())

    for category, checks in results.items():
        status = "✓" if all(checks) else "✗"
        count = sum(checks)
        print(f"{status} {category}: {count}/{len(checks)}")

    print(f"\nTotal: {passed_checks}/{total_checks}")

    if passed_checks == total_checks:
        print("\n✓ Day 22 implementation is complete!")
        sys.exit(0)
    else:
        print(f"\n✗ Missing {total_checks - passed_checks} files")
        sys.exit(1)


if __name__ == "__main__":
    verify_implementation()