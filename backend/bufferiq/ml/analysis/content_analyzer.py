"""Content characteristic analysis."""

import re
from typing import Any

import pandas as pd
from scipy import stats

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class ContentAnalyzer:
    """Analyze content characteristics and their impact on engagement."""

    def __init__(self) -> None:
        """Initialize content analyzer with regex patterns."""
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.hashtag_pattern = re.compile(r"#\w+")
        # Emoji regex pattern (basic ranges)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

    def analyze_length_impact(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> dict[str, Any]:
        """
        Analyze impact of content length on engagement.

        Args:
            df: DataFrame with 'content' and metric columns
            metric: Metric to analyze

        Returns:
            Dictionary with length analysis results

        Example:
            >>> analyzer = ContentAnalyzer()
            >>> length_analysis = analyzer.analyze_length_impact(df)
            >>> print(length_analysis['correlation'])
        """
        if "content" not in df.columns:
            raise ValueError("DataFrame must have 'content' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        df = df.copy()

        # Calculate length metrics if not present
        if "content_length" not in df.columns:
            df["content_length"] = df["content"].str.len()
        if "word_count" not in df.columns:
            df["word_count"] = df["content"].str.split().str.len()

        # Calculate statistics
        analysis = {
            "mean_length": float(df["content_length"].mean()),
            "median_length": float(df["content_length"].median()),
            "mean_word_count": float(df["word_count"].mean()),
            "median_word_count": float(df["word_count"].median()),
        }

        # Correlation with engagement
        if len(df) >= 3:
            try:
                corr_length, p_length = stats.pearsonr(df["content_length"], df[metric])
                corr_words, p_words = stats.pearsonr(df["word_count"], df[metric])

                analysis["length_correlation"] = float(corr_length)
                analysis["length_p_value"] = float(p_length)
                analysis["word_count_correlation"] = float(corr_words)
                analysis["word_count_p_value"] = float(p_words)
            except Exception as e:
                logger.warning(f"Could not calculate correlation: {e}")

        # Find optimal ranges (split into quartiles)
        df["length_quartile"] = pd.qcut(
            df["content_length"], q=4, labels=False, duplicates="drop"
        )
        quartile_performance = df.groupby("length_quartile")[metric].mean().to_dict()
        analysis["quartile_performance"] = {
            str(k): float(v) for k, v in quartile_performance.items()
        }

        logger.info(
            "Analyzed length impact",
            metric=metric,
            mean_length=analysis["mean_length"],
            correlation=analysis.get("length_correlation", 0.0),
        )

        return analysis

    def analyze_hashtag_impact(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> dict[str, Any]:
        """
        Analyze impact of hashtags on engagement.

        Args:
            df: DataFrame with 'content' and metric columns
            metric: Metric to analyze

        Returns:
            Dictionary with hashtag analysis results

        Example:
            >>> hashtag_analysis = analyzer.analyze_hashtag_impact(df)
            >>> print(f"Optimal count: {hashtag_analysis['optimal_count']}")
        """
        if "content" not in df.columns:
            raise ValueError("DataFrame must have 'content' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        df = df.copy()

        # Extract hashtags
        df["hashtag_count"] = df["content"].apply(
            lambda x: len(self.hashtag_pattern.findall(str(x)))
        )
        df["has_hashtag"] = df["hashtag_count"] > 0

        # Calculate statistics
        with_hashtags = df[df["has_hashtag"]][metric]
        without_hashtags = df[~df["has_hashtag"]][metric]

        analysis = {
            "posts_with_hashtags": int(df["has_hashtag"].sum()),
            "posts_without_hashtags": int((~df["has_hashtag"]).sum()),
            "mean_with_hashtags": float(with_hashtags.mean())
            if len(with_hashtags) > 0
            else 0.0,
            "mean_without_hashtags": float(without_hashtags.mean())
            if len(without_hashtags) > 0
            else 0.0,
            "avg_hashtag_count": float(df["hashtag_count"].mean()),
        }

        # Statistical test
        if len(with_hashtags) >= 2 and len(without_hashtags) >= 2:
            try:
                t_stat, p_value = stats.ttest_ind(with_hashtags, without_hashtags)
                analysis["t_statistic"] = float(t_stat)
                analysis["p_value"] = float(p_value)
                analysis["significant"] = p_value < 0.05
            except Exception as e:
                logger.warning(f"Could not perform t-test: {e}")

        # Optimal hashtag count
        hashtag_counts = df[df["hashtag_count"] > 0].groupby("hashtag_count")[metric]
        if len(hashtag_counts) > 0:
            optimal_performance = hashtag_counts.mean()
            if len(optimal_performance) > 0:
                optimal_count = optimal_performance.idxmax()
                analysis["optimal_count"] = int(optimal_count)
                analysis["count_performance"] = {
                    int(k): float(v) for k, v in optimal_performance.items()
                }

        # Most common hashtags
        all_hashtags: list[str] = []
        for content in df["content"]:
            all_hashtags.extend(self.hashtag_pattern.findall(str(content)))

        if all_hashtags:
            hashtag_counts_series = pd.Series(all_hashtags).value_counts()
            analysis["most_common_hashtags"] = hashtag_counts_series.head(10).to_dict()

        logger.info(
            "Analyzed hashtag impact",
            metric=metric,
            posts_with_hashtags=analysis["posts_with_hashtags"],
            significant=analysis.get("significant", False),
        )

        return analysis

    def analyze_url_impact(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> dict[str, Any]:
        """
        Analyze impact of URLs on engagement.

        Args:
            df: DataFrame with 'content' and metric columns
            metric: Metric to analyze

        Returns:
            Dictionary with URL analysis results

        Example:
            >>> url_analysis = analyzer.analyze_url_impact(df)
            >>> print(f"Mean with URLs: {url_analysis['mean_with_url']:.4f}")
        """
        if "content" not in df.columns:
            raise ValueError("DataFrame must have 'content' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        df = df.copy()

        # Detect URLs
        df["has_url"] = df["content"].apply(
            lambda x: bool(self.url_pattern.search(str(x)))
        )

        # Calculate statistics
        with_url = df[df["has_url"]][metric]
        without_url = df[~df["has_url"]][metric]

        analysis = {
            "posts_with_url": int(df["has_url"].sum()),
            "posts_without_url": int((~df["has_url"]).sum()),
            "mean_with_url": float(with_url.mean()) if len(with_url) > 0 else 0.0,
            "mean_without_url": float(without_url.mean())
            if len(without_url) > 0
            else 0.0,
        }

        # Statistical test
        if len(with_url) >= 2 and len(without_url) >= 2:
            try:
                t_stat, p_value = stats.ttest_ind(with_url, without_url)
                analysis["t_statistic"] = float(t_stat)
                analysis["p_value"] = float(p_value)
                analysis["significant"] = p_value < 0.05
            except Exception as e:
                logger.warning(f"Could not perform t-test: {e}")

        logger.info(
            "Analyzed URL impact",
            metric=metric,
            posts_with_url=analysis["posts_with_url"],
            significant=analysis.get("significant", False),
        )

        return analysis

    def analyze_emoji_impact(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> dict[str, Any]:
        """
        Analyze impact of emojis on engagement.

        Args:
            df: DataFrame with 'content' and metric columns
            metric: Metric to analyze

        Returns:
            Dictionary with emoji analysis results

        Example:
            >>> emoji_analysis = analyzer.analyze_emoji_impact(df)
            >>> print(f"Mean with emojis: {emoji_analysis['mean_with_emoji']:.4f}")
        """
        if "content" not in df.columns:
            raise ValueError("DataFrame must have 'content' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        df = df.copy()

        # Detect emojis
        df["emoji_count"] = df["content"].apply(
            lambda x: len(self.emoji_pattern.findall(str(x)))
        )
        df["has_emoji"] = df["emoji_count"] > 0

        # Calculate statistics
        with_emoji = df[df["has_emoji"]][metric]
        without_emoji = df[~df["has_emoji"]][metric]

        analysis = {
            "posts_with_emoji": int(df["has_emoji"].sum()),
            "posts_without_emoji": int((~df["has_emoji"]).sum()),
            "mean_with_emoji": float(with_emoji.mean()) if len(with_emoji) > 0 else 0.0,
            "mean_without_emoji": float(without_emoji.mean())
            if len(without_emoji) > 0
            else 0.0,
            "avg_emoji_count": float(df["emoji_count"].mean()),
        }

        # Statistical test
        if len(with_emoji) >= 2 and len(without_emoji) >= 2:
            try:
                t_stat, p_value = stats.ttest_ind(with_emoji, without_emoji)
                analysis["t_statistic"] = float(t_stat)
                analysis["p_value"] = float(p_value)
                analysis["significant"] = p_value < 0.05
            except Exception as e:
                logger.warning(f"Could not perform t-test: {e}")

        logger.info(
            "Analyzed emoji impact",
            metric=metric,
            posts_with_emoji=analysis["posts_with_emoji"],
            significant=analysis.get("significant", False),
        )

        return analysis

    def extract_common_patterns(
        self, df: pd.DataFrame, top_n: int = 10, min_engagement: float = 0.05
    ) -> dict[str, Any]:
        """
        Extract common patterns from high-performing content.

        Args:
            df: DataFrame with 'content' and 'engagement_rate' columns
            top_n: Number of top patterns to return
            min_engagement: Minimum engagement rate threshold

        Returns:
            Dictionary with common patterns and characteristics

        Example:
            >>> patterns = analyzer.extract_common_patterns(df, top_n=10)
            >>> print(patterns['common_bigrams'])
        """
        if "content" not in df.columns:
            raise ValueError("DataFrame must have 'content' column")

        if "engagement_rate" not in df.columns:
            raise ValueError("DataFrame must have 'engagement_rate' column")

        # Filter high-performing posts
        high_performing = df[df["engagement_rate"] >= min_engagement]

        if len(high_performing) == 0:
            logger.warning("No high-performing posts found")
            return {}

        patterns: dict[str, Any] = {
            "high_performing_count": int(len(high_performing)),
            "threshold": min_engagement,
        }

        # Extract word patterns (simple tokenization)
        all_words: list[str] = []
        for content in high_performing["content"]:
            words = str(content).lower().split()
            all_words.extend(words)

        if all_words:
            word_counts = pd.Series(all_words).value_counts()
            patterns["common_words"] = word_counts.head(top_n).to_dict()

        # Extract bigrams
        bigrams: list[str] = []
        for content in high_performing["content"]:
            words = str(content).lower().split()
            for i in range(len(words) - 1):
                bigrams.append(f"{words[i]} {words[i+1]}")

        if bigrams:
            bigram_counts = pd.Series(bigrams).value_counts()
            patterns["common_bigrams"] = bigram_counts.head(top_n).to_dict()

        logger.info(
            "Extracted common patterns",
            high_performing_posts=len(high_performing),
            unique_words=len(set(all_words)),
        )

        return patterns

    def analyze_question_impact(
        self, df: pd.DataFrame, metric: str = "engagement_rate"
    ) -> dict[str, Any]:
        """
        Analyze impact of questions on engagement.

        Args:
            df: DataFrame with 'content' and metric columns
            metric: Metric to analyze

        Returns:
            Dictionary with question analysis results

        Example:
            >>> question_analysis = analyzer.analyze_question_impact(df)
            >>> print(f"Mean with questions: {question_analysis['mean_with_question']:.4f}")
        """
        if "content" not in df.columns:
            raise ValueError("DataFrame must have 'content' column")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        df = df.copy()

        # Detect questions
        df["has_question"] = df["content"].str.contains(r"\?", regex=True, na=False)

        # Calculate statistics
        with_question = df[df["has_question"]][metric]
        without_question = df[~df["has_question"]][metric]

        analysis = {
            "posts_with_question": int(df["has_question"].sum()),
            "posts_without_question": int((~df["has_question"]).sum()),
            "mean_with_question": float(with_question.mean())
            if len(with_question) > 0
            else 0.0,
            "mean_without_question": float(without_question.mean())
            if len(without_question) > 0
            else 0.0,
        }

        # Statistical test
        if len(with_question) >= 2 and len(without_question) >= 2:
            try:
                t_stat, p_value = stats.ttest_ind(with_question, without_question)
                analysis["t_statistic"] = float(t_stat)
                analysis["p_value"] = float(p_value)
                analysis["significant"] = p_value < 0.05
            except Exception as e:
                logger.warning(f"Could not perform t-test: {e}")

        logger.info(
            "Analyzed question impact",
            metric=metric,
            posts_with_question=analysis["posts_with_question"],
            significant=analysis.get("significant", False),
        )

        return analysis
