"""Run full data analysis programmatically."""

import argparse
import asyncio
import json
from pathlib import Path

from bufferiq.core.config import get_settings
from bufferiq.core.database import DatabaseManager
from bufferiq.ml.analysis.content_analyzer import ContentAnalyzer
from bufferiq.ml.analysis.data_loader import DataLoader
from bufferiq.ml.analysis.engagement_analyzer import EngagementAnalyzer
from bufferiq.ml.analysis.temporal_analyzer import TemporalAnalyzer
from bufferiq.ml.analysis.visualizer import Visualizer


async def run_full_analysis(
    output_dir: str = "outputs/figures",
    save_insights: bool = True,
    platform: str | None = None,
    min_posts: int = 10,
    dry_run: bool = False,
    insights_only: bool = False,
) -> dict[str, any]:
    """
    Run complete data analysis pipeline.

    Args:
        output_dir: Directory to save figures
        save_insights: Whether to save insights to markdown
        platform: Filter by specific platform
        min_posts: Minimum number of posts required
        dry_run: Don't save files, just run analysis
        insights_only: Skip visualizations

    Returns:
        Dictionary with analysis summary
    """
    print("🚀 Starting BufferIQ Data Analysis Pipeline...")

    # Initialize components
    settings = get_settings()
    db_manager = DatabaseManager(settings)
    await db_manager.connect()

    summary = {}

    try:
        async with db_manager.session() as session:
            # Load data
            print("\n📊 Loading data from database...")
            loader = DataLoader(session)
            df = await loader.load_posts(platform=platform, status="sent")

            if len(df) == 0:
                print("❌ No posts found in database!")
                print("💡 Try running: python scripts/generate_sample_data.py")
                return {}

            if len(df) < min_posts:
                print(
                    f"⚠️  Only {len(df)} posts found, minimum {min_posts} required for reliable analysis"
                )
                return {}

            print(f"✅ Loaded {len(df)} posts")
            summary["total_posts"] = len(df)
            summary["platforms"] = df["platform"].unique().tolist()
            summary["date_range"] = {
                "start": str(df["published_at"].min()),
                "end": str(df["published_at"].max()),
            }

            # Initialize analyzers
            engagement_analyzer = EngagementAnalyzer()
            temporal_analyzer = TemporalAnalyzer()
            content_analyzer = ContentAnalyzer()

            if not dry_run and not insights_only:
                visualizer = Visualizer(output_dir=output_dir)
                print(f"\n📁 Output directory: {output_dir}")

            # Engagement Analysis
            print("\n📈 Running engagement analysis...")
            df = engagement_analyzer.calculate_engagement_rate(df)

            dist_stats = engagement_analyzer.analyze_distribution(df, "engagement_rate")
            summary["engagement_distribution"] = dist_stats

            print(
                f"   Mean engagement rate: {dist_stats['mean']:.4f}"
            )
            print(f"   Median engagement rate: {dist_stats['median']:.4f}")

            # Correlation analysis
            corr_matrix = engagement_analyzer.calculate_correlations(df)
            strong_corr = engagement_analyzer.find_strong_correlations(
                corr_matrix, threshold=0.5
            )
            summary["strong_correlations"] = strong_corr

            if strong_corr:
                print(f"   Found {len(strong_corr)} strong correlations")

            # Platform comparison
            if len(df["platform"].unique()) > 1:
                platform_comp = engagement_analyzer.platform_comparison(
                    df, "engagement_rate"
                )
                summary["platform_comparison"] = platform_comp
                print("   Compared platforms")

            # Visualizations
            if not dry_run and not insights_only:
                print("\n🎨 Creating visualizations...")

                visualizer.plot_distribution(
                    df["engagement_rate"],
                    "Engagement Rate Distribution",
                    xlabel="Engagement Rate",
                    save_path="engagement_distribution.png",
                )
                print("   ✓ Engagement distribution")

                visualizer.plot_correlation_matrix(
                    df.select_dtypes(include=["number"]),
                    save_path="correlation_matrix.png",
                )
                print("   ✓ Correlation matrix")

                if len(df["platform"].unique()) > 1:
                    visualizer.plot_platform_comparison(
                        df, "engagement_rate", save_path="platform_comparison.png"
                    )
                    print("   ✓ Platform comparison")

            # Temporal Analysis
            print("\n⏰ Running temporal analysis...")

            hourly_stats = temporal_analyzer.hourly_patterns(df, "engagement_rate")
            summary["hourly_patterns"] = {
                "peak_hour": int(hourly_stats.loc[hourly_stats["mean"].idxmax(), "hour"]),
                "peak_engagement": float(hourly_stats["mean"].max()),
            }
            print(
                f"   Peak hour: {summary['hourly_patterns']['peak_hour']}:00"
            )

            daily_stats = temporal_analyzer.daily_patterns(df, "engagement_rate")
            summary["daily_patterns"] = {
                "best_day": daily_stats.loc[daily_stats["mean"].idxmax(), "day_name"],
                "best_engagement": float(daily_stats["mean"].max()),
            }
            print(f"   Best day: {summary['daily_patterns']['best_day']}")

            optimal_windows = temporal_analyzer.optimal_posting_windows(
                df, platform=platform, top_n=5
            )
            summary["optimal_windows"] = optimal_windows

            if optimal_windows:
                print(f"   Found {len(optimal_windows)} optimal posting windows")

            # Visualizations
            if not dry_run and not insights_only:
                visualizer.plot_hourly_heatmap(
                    df, "engagement_rate", save_path="hourly_heatmap.png"
                )
                print("   ✓ Hourly heatmap")

                weekly_trends = temporal_analyzer.weekly_trends(df, "engagement_rate")
                if len(weekly_trends) > 0:
                    visualizer.plot_time_series(
                        weekly_trends,
                        "week_start",
                        "mean",
                        "Weekly Engagement Trends",
                        save_path="weekly_trends.png",
                    )
                    print("   ✓ Weekly trends")

            # Content Analysis
            print("\n📝 Running content analysis...")

            length_analysis = content_analyzer.analyze_length_impact(df, "engagement_rate")
            summary["length_analysis"] = length_analysis
            print(
                f"   Mean content length: {length_analysis['mean_length']:.0f} characters"
            )

            hashtag_analysis = content_analyzer.analyze_hashtag_impact(df, "engagement_rate")
            summary["hashtag_analysis"] = hashtag_analysis
            print(
                f"   Posts with hashtags: {hashtag_analysis['posts_with_hashtags']}"
            )

            url_analysis = content_analyzer.analyze_url_impact(df, "engagement_rate")
            summary["url_analysis"] = url_analysis

            emoji_analysis = content_analyzer.analyze_emoji_impact(df, "engagement_rate")
            summary["emoji_analysis"] = emoji_analysis

            # Visualizations
            if not dry_run and not insights_only:
                visualizer.plot_scatter_with_regression(
                    df,
                    "content_length",
                    "engagement_rate",
                    "Content Length vs Engagement",
                    save_path="content_length_scatter.png",
                )
                print("   ✓ Content length scatter")

            # Save insights
            if save_insights and not dry_run:
                print("\n💾 Saving insights to DATA_INSIGHTS.md...")
                insights_path = Path("docs/DATA_INSIGHTS.md")
                insights_path.parent.mkdir(parents=True, exist_ok=True)

                with open(insights_path, "w") as f:
                    f.write("# BufferIQ Data Analysis Insights\n\n")
                    f.write(f"**Analysis Date**: {summary.get('date_range', {}).get('end', 'N/A')}\n")
                    f.write(f"**Total Posts Analyzed**: {summary['total_posts']}\n")
                    f.write(
                        f"**Platforms**: {', '.join(summary['platforms'])}\n\n"
                    )

                    f.write("## 1. Engagement Distribution\n\n")
                    dist = summary.get("engagement_distribution", {})
                    f.write(f"- **Mean Engagement Rate**: {dist.get('mean', 0):.4f}\n")
                    f.write(f"- **Median Engagement Rate**: {dist.get('median', 0):.4f}\n")
                    f.write(f"- **Standard Deviation**: {dist.get('std', 0):.4f}\n\n")

                    f.write("## 2. Temporal Patterns\n\n")
                    f.write(
                        f"- **Peak Posting Hour**: {summary['hourly_patterns']['peak_hour']}:00\n"
                    )
                    f.write(
                        f"- **Best Day of Week**: {summary['daily_patterns']['best_day']}\n\n"
                    )

                    f.write("### Top 5 Optimal Posting Windows\n\n")
                    for i, window in enumerate(optimal_windows[:5], 1):
                        f.write(
                            f"{i}. {window['day_name']} at {window['hour']}:00 "
                            f"(Avg: {window['mean_engagement']:.4f}, Posts: {window['post_count']})\n"
                        )

                    f.write("\n## 3. Content Characteristics\n\n")
                    f.write(
                        f"- **Mean Content Length**: {length_analysis['mean_length']:.0f} characters\n"
                    )
                    f.write(
                        f"- **Posts with Hashtags**: {hashtag_analysis['posts_with_hashtags']}\n"
                    )
                    f.write(f"- **Posts with URLs**: {url_analysis['posts_with_url']}\n")
                    f.write(
                        f"- **Posts with Emojis**: {emoji_analysis['posts_with_emoji']}\n\n"
                    )

                    if "platform_comparison" in summary:
                        f.write("## 4. Platform Comparison\n\n")
                        plat_comp = summary["platform_comparison"]
                        for platform, mean_eng in plat_comp.get("means", {}).items():
                            f.write(f"- **{platform}**: {mean_eng:.4f}\n")

                print(f"✅ Insights saved to {insights_path}")

            print("\n✨ Analysis complete!")
            print(f"\n📊 Summary:")
            print(json.dumps(summary, indent=2, default=str))

            return summary

    finally:
        await db_manager.disconnect()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run BufferIQ data analysis")
    parser.add_argument(
        "--output-dir", default="outputs/figures", help="Output directory for figures"
    )
    parser.add_argument(
        "--platform", help="Filter by specific platform"
    )
    parser.add_argument(
        "--min-posts", type=int, default=10, help="Minimum posts required"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't save files"
    )
    parser.add_argument(
        "--insights-only", action="store_true", help="Skip visualizations"
    )
    parser.add_argument(
        "--no-insights", action="store_true", help="Don't save insights markdown"
    )

    args = parser.parse_args()

    asyncio.run(
        run_full_analysis(
            output_dir=args.output_dir,
            save_insights=not args.no_insights,
            platform=args.platform,
            min_posts=args.min_posts,
            dry_run=args.dry_run,
            insights_only=args.insights_only,
        )
    )


if __name__ == "__main__":
    main()