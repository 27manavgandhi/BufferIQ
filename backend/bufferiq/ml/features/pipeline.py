"""Feature engineering pipeline."""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.core.logging import get_logger
from bufferiq.ml.features.base import BaseFeatureExtractor
from bufferiq.ml.features.content import ContentFeatureExtractor
from bufferiq.ml.features.engagement import EngagementFeatureExtractor
from bufferiq.ml.features.nlp import NLPFeatureExtractor
from bufferiq.ml.features.platform_specific import PlatformSpecificFeatureExtractor
from bufferiq.ml.features.scaler import FeatureScaler
from bufferiq.ml.features.selector import FeatureSelector
from bufferiq.ml.features.temporal import TemporalFeatureExtractor

logger = get_logger(__name__)


class FeatureEngineeringPipeline:
    """Orchestrate all feature extraction."""

    def __init__(
        self,
        extractors: Optional[list[BaseFeatureExtractor]] = None,
        scaler: Optional[FeatureScaler] = None,
        selector: Optional[FeatureSelector] = None,
    ) -> None:
        """
        Initialize feature engineering pipeline.

        Args:
            extractors: List of feature extractors (None = use all)
            scaler: Optional feature scaler
            selector: Optional feature selector

        Example:
            >>> pipeline = FeatureEngineeringPipeline()
            >>> features = await pipeline.extract_features(df)
        """
        if extractors is None:
            # Use all extractors by default
            self.extractors: list[BaseFeatureExtractor] = [
                TemporalFeatureExtractor(),
                ContentFeatureExtractor(),
                NLPFeatureExtractor(),
                EngagementFeatureExtractor(),
                PlatformSpecificFeatureExtractor(),
            ]
        else:
            self.extractors = extractors

        self.scaler = scaler
        self.selector = selector

    async def extract_features(
        self,
        df: pd.DataFrame,
        session: Optional[AsyncSession] = None,
        fit_scaler: bool = False,
        fit_selector: bool = False,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run full feature extraction pipeline.

        Args:
            df: Input DataFrame with post data
            session: Optional AsyncSession for engagement features
            fit_scaler: Whether to fit scaler on this data
            fit_selector: Whether to fit selector on this data
            target_column: Target column name (required if fit_selector=True)

        Returns:
            DataFrame with all extracted features

        Raises:
            ValueError: If required parameters missing

        Example:
            >>> pipeline = FeatureEngineeringPipeline()
            >>> features = await pipeline.extract_features(
            ...     df,
            ...     fit_scaler=True,
            ...     fit_selector=True,
            ...     target_column="engagement_rate"
            ... )
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, returning empty features")
            return pd.DataFrame()

        logger.info(f"Starting feature extraction for {len(df)} posts")

        # Step 1: Extract features from all extractors
        feature_dfs: list[pd.DataFrame] = []

        for extractor in self.extractors:
            try:
                logger.info(f"Running {extractor.__class__.__name__}...")

                # Special handling for engagement extractor
                if isinstance(extractor, EngagementFeatureExtractor) and session:
                    features = await extractor.extract_async(df, session)
                else:
                    features = extractor.extract(df)

                feature_dfs.append(features)

                logger.info(
                    f"Extracted {len(features.columns)} features from "
                    f"{extractor.__class__.__name__}"
                )

            except Exception as e:
                logger.error(
                    f"Error in {extractor.__class__.__name__}: {e}", exc_info=True
                )
                # Continue with other extractors
                continue

        if not feature_dfs:
            raise ValueError("No features extracted from any extractor")

        # Step 2: Combine all features
        result = pd.concat(feature_dfs, axis=1)

        logger.info(f"Combined {len(result.columns)} total features")

        # Step 3: Handle missing values
        result = result.fillna(0)

        # Step 4: Scale features (if scaler provided)
        if self.scaler is not None:
            feature_columns = result.columns.tolist()

            if fit_scaler:
                logger.info("Fitting scaler...")
                self.scaler.fit(result, feature_columns)

            logger.info("Scaling features...")
            result = self.scaler.transform(result, feature_columns)

        # Step 5: Select features (if selector provided)
        if self.selector is not None:
            if fit_selector:
                if target_column is None:
                    raise ValueError("target_column required when fit_selector=True")

                if target_column not in df.columns:
                    raise ValueError(
                        f"Target column '{target_column}' not found in DataFrame"
                    )

                logger.info("Fitting feature selector...")
                self.selector.fit(result, df[target_column])

            logger.info("Selecting features...")
            result = self.selector.transform(result)

            logger.info(f"Selected {len(result.columns)} features after selection")

        logger.info(f"Feature extraction complete: {len(result.columns)} features")

        return result

    def get_all_feature_names(self) -> list[str]:
        """
        Return all feature names from all extractors.

        Returns:
            List of all feature names
        """
        all_features: list[str] = []

        for extractor in self.extractors:
            all_features.extend(extractor.feature_names)

        return all_features

    def save_pipeline(self, path: str) -> None:
        """
        Save entire pipeline to disk.

        Args:
            path: Directory path to save pipeline components

        Example:
            >>> pipeline.save_pipeline("outputs/features/pipeline")
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save extractors metadata
        extractor_metadata = {
            "extractors": [
                extractor.__class__.__name__ for extractor in self.extractors
            ]
        }
        joblib.dump(extractor_metadata, path_obj / "extractors.joblib")

        # Save scaler if exists
        if self.scaler is not None:
            self.scaler.save(str(path_obj / "scaler.joblib"))

        # Save selector if exists
        if self.selector is not None:
            self.selector.save(str(path_obj / "selector.joblib"))

        logger.info(f"Saved pipeline to {path}")

    @classmethod
    def load_pipeline(cls, path: str) -> "FeatureEngineeringPipeline":
        """
        Load pipeline from disk.

        Args:
            path: Directory path to load pipeline from

        Returns:
            Loaded FeatureEngineeringPipeline instance

        Raises:
            FileNotFoundError: If pipeline files not found
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Pipeline directory not found: {path}")

        # Load extractor metadata
        extractor_metadata = joblib.load(path_obj / "extractors.joblib")

        # Recreate extractors
        extractor_map = {
            "TemporalFeatureExtractor": TemporalFeatureExtractor,
            "ContentFeatureExtractor": ContentFeatureExtractor,
            "NLPFeatureExtractor": NLPFeatureExtractor,
            "EngagementFeatureExtractor": EngagementFeatureExtractor,
            "PlatformSpecificFeatureExtractor": PlatformSpecificFeatureExtractor,
        }

        extractors = [
            extractor_map[name]()
            for name in extractor_metadata["extractors"]
            if name in extractor_map
        ]

        # Load scaler if exists
        scaler = None
        scaler_path = path_obj / "scaler.joblib"
        if scaler_path.exists():
            scaler = FeatureScaler.load(str(scaler_path))

        # Load selector if exists
        selector = None
        selector_path = path_obj / "selector.joblib"
        if selector_path.exists():
            selector = FeatureSelector.load(str(selector_path))

        pipeline = cls(extractors=extractors, scaler=scaler, selector=selector)

        logger.info(f"Loaded pipeline from {path}")

        return pipeline

    def get_feature_stats(self) -> dict[str, int]:
        """
        Return statistics about features.

        Returns:
            Dictionary with feature statistics

        Example:
            >>> stats = pipeline.get_feature_stats()
            >>> print(f"Total features: {stats['total_features']}")
        """
        all_features = self.get_all_feature_names()

        stats = {
            "total_features": len(all_features),
            "num_extractors": len(self.extractors),
            "temporal_features": len(
                [e for e in self.extractors if isinstance(e, TemporalFeatureExtractor)]
            )
            * 21,
            "content_features": len(
                [e for e in self.extractors if isinstance(e, ContentFeatureExtractor)]
            )
            * 25,
            "nlp_features": len(
                [e for e in self.extractors if isinstance(e, NLPFeatureExtractor)]
            )
            * 15,
            "engagement_features": len(
                [
                    e
                    for e in self.extractors
                    if isinstance(e, EngagementFeatureExtractor)
                ]
            )
            * 15,
            "platform_features": len(
                [
                    e
                    for e in self.extractors
                    if isinstance(e, PlatformSpecificFeatureExtractor)
                ]
            )
            * 16,
            "has_scaler": self.scaler is not None,
            "has_selector": self.selector is not None,
        }

        return stats
