"""Training pipeline orchestrator."""

import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.core.logging import get_logger
from bufferiq.domain.models import Post
from bufferiq.ml.features.pipeline import FeatureEngineeringPipeline
from bufferiq.ml.training.config_schema import TrainingPipelineConfig
from bufferiq.ml.training.cross_validator import CrossValidator
from bufferiq.ml.training.data_preparation import DataPreparation
from bufferiq.ml.training.experiment_tracker import ExperimentTracker
from bufferiq.ml.training.model_registry import ModelRegistry
from bufferiq.ml.training.trainer_base import BaseTrainer

logger = get_logger(__name__)


class TrainingPipeline:
    """Orchestrate entire training workflow."""

    def __init__(
        self, config: TrainingPipelineConfig, session: AsyncSession
    ) -> None:
        self.config = config
        self.session = session

        self.experiment_tracker = ExperimentTracker(
            experiment_name=config.experiment.experiment_name
        )
        self.model_registry = ModelRegistry()

    async def run(self) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info(
            f"Starting training pipeline: {self.config.experiment.experiment_name}"
        )
        logger.info("=" * 80)

        start_time = time.time()

        try:
            # Step 1: Load data
            df = await self._load_data()

            # Step 2: Feature engineering
            features_df = await self._extract_features(df)

            # Step 3: Split
            X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(
                features_df
            )

            # Step 4: Validate
            quality_report = self._validate_data_quality(features_df)

            # Step 5: Trainer
            trainer = self._initialize_trainer()

            # Step 6: Train
            training_metrics = trainer.train(X_train, y_train, X_val, y_val)

            # Step 7: Evaluate
            test_metrics = trainer.evaluate(X_test, y_test)

            # Step 8: Log
            training_duration = time.time() - start_time
            experiment_dir = self._log_experiment(
                trainer, test_metrics, training_duration, quality_report
            )

            # Step 9: Register
            model_id = self._register_model(trainer, test_metrics)

            # Step 10: Results
            results = {
                "experiment_name": self.config.experiment.experiment_name,
                "experiment_dir": experiment_dir,
                "model_id": model_id,
                "training_metrics": training_metrics,
                "test_metrics": test_metrics,
                "training_duration": training_duration,
                "data_stats": {
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "test_size": len(X_test),
                    "num_features": len(X_train.columns),
                },
                "quality_report": quality_report,
            }

            logger.info("Training completed successfully")
            return results

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            raise

    async def run_with_cross_validation(self) -> Dict[str, Any]:
        logger.info(
            f"Starting {self.config.experiment.cv_folds}-fold cross-validation"
        )

        start_time = time.time()

        df = await self._load_data()
        features_df = await self._extract_features(df)

        target_column = self.config.data.target_column
        feature_columns = (
            self.config.data.feature_columns
            if self.config.data.feature_columns
            else [col for col in features_df.columns if col != target_column]
        )

        X = features_df[feature_columns]
        y = features_df[target_column]

        trainer = self._initialize_trainer()

        cv = CrossValidator(
            n_splits=self.config.experiment.cv_folds,
            strategy="timeseries" if self.config.data.time_based_split else "kfold",
        )

        cv_results = cv.cross_validate(trainer, X, y)

        training_duration = time.time() - start_time

        self.experiment_tracker.log_params(self.config.dict())
        self.experiment_tracker.log_metrics(cv_results["mean_metrics"])

        cv_summary = cv.get_cv_summary()
        self.experiment_tracker.log_dataframe(cv_summary, "cv_summary")

        experiment_dir = self.experiment_tracker.save_experiment()

        return {
            "experiment_name": self.config.experiment.experiment_name,
            "experiment_dir": experiment_dir,
            "cv_results": cv_results,
            "training_duration": training_duration,
        }

    async def _load_data(self) -> pd.DataFrame:
        stmt = select(Post).where(Post.status == "sent")

        if self.config.data.platforms:
            stmt = stmt.where(Post.platform.in_(self.config.data.platforms))

        result = await self.session.execute(stmt)
        posts = result.scalars().all()

        if not posts:
            raise ValueError("No posts found")

        df = pd.DataFrame(
            [
                {
                    "id": p.id,
                    "user_id": p.user_id,
                    "channel_id": p.channel_id,
                    "platform": p.platform,
                    "content": p.content,
                    "published_at": p.sent_at or p.scheduled_at,
                    "likes": p.likes or 0,
                    "comments": p.comments or 0,
                    "shares": p.shares or 0,
                    "impressions": p.impressions or 1,
                    "clicks": p.clicks or 0,
                }
                for p in posts
            ]
        )

        df["engagement_rate"] = (
            (df["likes"] + df["comments"] + df["shares"])
            / df["impressions"].replace(0, 1)
            * 100
        )

        logger.info(f"Loaded {len(df)} rows")
        return df

    async def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pipeline = FeatureEngineeringPipeline()
        features = await pipeline.extract_features(df, session=self.session)

        features[self.config.data.target_column] = df[
            self.config.data.target_column
        ]

        return features

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        prep = DataPreparation(
            test_size=self.config.data.test_size,
            validation_size=self.config.data.validation_size,
            time_based_split=self.config.data.time_based_split,
        )

        target_column = self.config.data.target_column
        feature_columns = (
            self.config.data.feature_columns
            if self.config.data.feature_columns
            else [col for col in df.columns if col != target_column]
        )

        return prep.split_data(df, target_column, feature_columns)

    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        prep = DataPreparation()
        report = prep.validate_data_quality(df)

        for w in report.get("warnings", []):
            logger.warning(w)

        return report

    def _initialize_trainer(self) -> BaseTrainer:
        from bufferiq.ml.trainers import (
            LightGBMTrainer,
            RandomForestTrainer,
            XGBoostTrainer,
        )

        model_type = self.config.model.model_type

        if model_type == "xgboost":
            trainer = XGBoostTrainer(
                random_state=self.config.model.random_state, verbose=True
            )
        elif model_type == "lightgbm":
            trainer = LightGBMTrainer(
                random_state=self.config.model.random_state, verbose=True
            )
        elif model_type == "random_forest":
            trainer = RandomForestTrainer(
                random_state=self.config.model.random_state, verbose=True
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        trainer.build_model(self.config.model.hyperparameters)
        return trainer

    def _log_experiment(
        self,
        trainer: BaseTrainer,
        metrics: Dict[str, float],
        duration: float,
        quality_report: Dict[str, Any],
    ) -> str:
        self.experiment_tracker.log_params(self.config.dict())
        self.experiment_tracker.log_metrics(metrics)
        self.experiment_tracker.log_metric("training_duration", duration)
        self.experiment_tracker.log_params({"data_quality": quality_report})

        try:
            importance = trainer.get_feature_importance()
            self.experiment_tracker.log_dataframe(importance, "feature_importance")
        except Exception as e:
            logger.warning(f"Feature importance logging failed: {e}")

        return self.experiment_tracker.save_experiment()

    def _register_model(
        self, trainer: BaseTrainer, metrics: Dict[str, float]
    ) -> str:
        model_path = self.experiment_tracker.experiment_dir / "model.joblib"
        trainer.save_model(str(model_path))

        version = f"1.0.{len(self.model_registry.list_models())}"

        return self.model_registry.register_model(
            model_path=str(model_path),
            version=version,
            metrics=metrics,
            metadata={
                "experiment_name": self.config.experiment.experiment_name,
                "model_type": self.config.model.model_type,
                "hyperparameters": self.config.model.hyperparameters,
                "platforms": self.config.data.platforms,
            },
        )