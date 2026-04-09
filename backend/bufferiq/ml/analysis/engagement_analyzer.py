"""Engagement pattern analysis."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class EngagementAnalyzer:
    """Analyze engagement patterns and distributions."""

    def calculate_engagement_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate engagement rate for posts.

        Engagement rate = (likes + comments + shares) / impressions

        Args:
            df: DataFrame with engagement metrics

        Returns:
            DataFrame with calculated engagement_rate column

        Example:
            >>> analyzer = EngagementAnalyzer()
            >>> df = analyzer.calculate_engagement_rate(df)
            >>> print(df['engagement_rate'].mean())
        """
        df = df.copy()

        # Calculate total engagement
        df["total_engagement"] = df["likes"] + df["comments"] + df["shares"]

        # Calculate engagement rate (avoid division by zero)
        df["engagement_rate"] = np.where(
            df["impressions"] > 0,
            df["total_engagement"] / df["impressions"],
            0.0,
        )

        logger.info(
            "Calculated engagement rates",
            posts=len(df),
            mean_rate=float(df["engagement_rate"].mean()),
        )

        return df

    def analyze_distribution(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> dict[str, Any]:
        """
        Analyze distribution of a metric.

        Args:
            df: DataFrame with metric column
            metric: Column name to analyze

        Returns:
            Dictionary with distribution statistics

        Example:
            >>> stats = analyzer.analyze_distribution(df, "engagement_rate")
            >>> print(f"Mean: {stats['mean']:.4f}")
            >>> print(f"Median: {stats['median']:.4f}")
        """
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        data = df[metric].dropna()

        if len(data) == 0:
            logger.warning("No data available for distribution analysis")
            return {}

        # Calculate statistics
        distribution_stats = {
            "count": int(len(data)),
            "mean": float(data.mean()),
            "median": float(data.median()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "q25": float(data.quantile(0.25)),
            "q50": float(data.quantile(0.50)),
            "q75": float(data.quantile(0.75)),
            "q90": float(data.quantile(0.90)),
            "q95": float(data.quantile(0.95)),
            "q99": float(data.quantile(0.99)),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data)),
        }

        # Test for normality (if enough data)
        if len(data) >= 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                distribution_stats["shapiro_statistic"] = float(shapiro_stat)
                distribution_stats["shapiro_p_value"] = float(shapiro_p)
                distribution_stats["is_normal"] = shapiro_p > 0.05
            except Exception as e:
                logger.warning(f"Could not perform Shapiro-Wilk test: {e}")

        logger.info(
            "Analyzed distribution",
            metric=metric,
            mean=distribution_stats["mean"],
            median=distribution_stats["median"],
        )

        return distribution_stats

    def identify_outliers(
        self, df: pd.DataFrame, metric: str, method: str = "iqr"
    ) -> pd.DataFrame:
        """
        Identify outliers in a metric.

        Args:
            df: DataFrame with metric column
            metric: Column name to analyze
            method: Outlier detection method (iqr or zscore)

        Returns:
            DataFrame with outlier posts

        Raises:
            ValueError: If invalid method specified

        Example:
            >>> outliers = analyzer.identify_outliers(df, "engagement_rate", "iqr")
            >>> print(f"Found {len(outliers)} outliers")
        """
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        if method not in ["iqr", "zscore"]:
            raise ValueError(f"Invalid method: {method}. Use 'iqr' or 'zscore'")

        data = df[metric].dropna()

        if len(data) == 0:
            logger.warning("No data available for outlier detection")
            return pd.DataFrame()

        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (df[metric] < lower_bound) | (df[metric] > upper_bound)

        else:  # zscore
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > 3

        outliers = df[outlier_mask].copy()

        logger.info(
            "Identified outliers",
            metric=metric,
            method=method,
            count=len(outliers),
            percentage=float(len(outliers) / len(df) * 100),
        )

        return outliers

    def calculate_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric features.

        Args:
            df: DataFrame with numeric columns

        Returns:
            Correlation matrix as DataFrame

        Example:
            >>> corr = analyzer.calculate_correlations(df)
            >>> print(corr.loc['likes', 'engagement_rate'])
        """
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return pd.DataFrame()

        # Calculate Pearson correlation
        correlation_matrix = df[numeric_cols].corr(method="pearson")

        logger.info(
            "Calculated correlation matrix",
            features=len(numeric_cols),
            method="pearson",
        )

        return correlation_matrix

    def find_strong_correlations(
        self, correlation_matrix: pd.DataFrame, threshold: float = 0.5
    ) -> list[tuple[str, str, float]]:
        """
        Find strong correlations in correlation matrix.

        Args:
            correlation_matrix: Correlation matrix DataFrame
            threshold: Minimum absolute correlation value

        Returns:
            List of (feature1, feature2, correlation) tuples

        Example:
            >>> strong = analyzer.find_strong_correlations(corr, threshold=0.7)
            >>> for feat1, feat2, corr_val in strong:
            ...     print(f"{feat1} <-> {feat2}: {corr_val:.3f}")
        """
        strong_correlations = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_correlations.append(
                        (
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            float(corr_value),
                        )
                    )

        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        logger.info(
            "Found strong correlations",
            threshold=threshold,
            count=len(strong_correlations),
        )

        return strong_correlations

    def platform_comparison(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> dict[str, Any]:
        """
        Compare metrics across platforms.

        Args:
            df: DataFrame with platform and metric columns
            metric: Column name to compare

        Returns:
            Dictionary with comparison statistics and test results

        Example:
            >>> comparison = analyzer.platform_comparison(df, "engagement_rate")
            >>> print(comparison['means'])
            >>> print(f"Significant difference: {comparison['anova_significant']}")
        """
        if "platform" not in df.columns:
            raise ValueError("DataFrame must have 'platform' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        # Group by platform
        grouped = df.groupby("platform")[metric]

        # Calculate statistics per platform
        platform_stats = {
            "means": grouped.mean().to_dict(),
            "medians": grouped.median().to_dict(),
            "stds": grouped.std().to_dict(),
            "counts": grouped.count().to_dict(),
        }

        # Perform ANOVA if we have multiple platforms
        platforms = df["platform"].unique()
        if len(platforms) >= 2:
            groups = [df[df["platform"] == p][metric].dropna() for p in platforms]
            # Filter out empty groups
            groups = [g for g in groups if len(g) > 0]

            if len(groups) >= 2:
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    platform_stats["anova_f_statistic"] = float(f_stat)
                    platform_stats["anova_p_value"] = float(p_value)
                    platform_stats["anova_significant"] = p_value < 0.05
                except Exception as e:
                    logger.warning(f"Could not perform ANOVA: {e}")

        logger.info(
            "Performed platform comparison",
            metric=metric,
            platforms=len(platforms),
            significant=platform_stats.get("anova_significant", False),
        )

        return platform_stats

    def segment_by_performance(
        self, df: pd.DataFrame, metric: str = "engagement_rate", n_segments: int = 3
    ) -> pd.DataFrame:
        """
        Segment posts by performance quartiles.

        Args:
            df: DataFrame with metric column
            metric: Column name to segment by
            n_segments: Number of segments (default: 3 for low/medium/high)

        Returns:
            DataFrame with performance_segment column added

        Example:
            >>> df = analyzer.segment_by_performance(df, "engagement_rate", 3)
            >>> print(df['performance_segment'].value_counts())
        """
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        df = df.copy()

        # Create segments using quantiles
        df["performance_segment"] = pd.qcut(
            df[metric], q=n_segments, labels=False, duplicates="drop"
        )

        # Map to labels
        if n_segments == 3:
            segment_labels = {0: "low", 1: "medium", 2: "high"}
        elif n_segments == 4:
            segment_labels = {0: "low", 1: "medium_low", 2: "medium_high", 3: "high"}
        else:
            segment_labels = {i: f"segment_{i}" for i in range(n_segments)}

        df["performance_label"] = df["performance_segment"].map(segment_labels)

        logger.info(
            "Segmented posts by performance",
            metric=metric,
            segments=n_segments,
            posts=len(df),
        )

        return df
