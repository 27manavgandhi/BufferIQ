"""Training pipeline orchestrator."""

import time
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.core.logging import get_logger
from bufferiq.domain.models import Channel, Post
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

    def __init__(self, config: TrainingPipelineConfig, session: AsyncSession) -> None:
        """
        Initialize training pipeline with config.

        Args:
            config: Training pipeline configuration
            session: Async database session

        Example:
            >>> config = TrainingPipelineConfig.from_yaml('config.yaml')
            >>> pipeline = TrainingPipeline(config, session)
            >>> results = await pipeline.run()
        """
        self.config = config
        self.session = session

        # Initialize components
        self.experiment_tracker = ExperimentTracker(
            experiment_name=config.experiment.experiment_name
        )
        self.model_registry = ModelRegistry()

    async def run(self) -> dict[str, Any]:
        """
        Run complete training pipeline.

        Returns:
            Dict with training results and paths

        Steps:
            1. Load data from database
            2. Extract features
            3. Prepare data (train/val/test split)
            4. Validate data quality
            5. Initialize trainer
            6. Train model (with checkpointing)
            7. Evaluate on test set
            8. Log experiment
            9. Register model
            10. Return results
        """
        logger.info("=" * 80)
        logger.info(
            f"Starting training pipeline: {self.config.experiment.experiment_name}"
        )
        logger.info("=" * 80)

        start_time = time.time()

        try:
            # Step 1: Load data
            logger.info("Step 1/10: Loading data from database")
            df = await self._load_data()

            # Step 2: Extract features
            logger.info("Step 2/10: Extracting features")
            features_df = await self._extract_features(df)

            # Step 3: Prepare data
            logger.info("Step 3/10: Preparing train/val/test splits")
            X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(
                features_df
            )

            # Step 4: Validate data quality
            logger.info("Step 4/10: Validating data quality")
            quality_report = self._validate_data_quality(features_df)

            # Step 5: Initialize trainer
            logger.info("Step 5/10: Initializing model trainer")
            trainer = self._initialize_trainer()

            # Step 6: Train model
            logger.info("Step 6/10: Training model")
            training_metrics = self._train_model(X_train, y_train, X_val, y_val)

            # Step 7: Evaluate on test set
            logger.info("Step 7/10: Evaluating on test set")
            test_metrics = trainer.evaluate(X_test, y_test)

            # Step 8: Log experiment
            logger.info("Step 8/10: Logging experiment")
            training_duration = time.time() - start_time
            experiment_dir = self._log_experiment(
                trainer, test_metrics, training_duration, quality_report
            )

            # Step 9: Register model
            logger.info("Step 9/10: Registering model in registry")
            model_id = self._register_model(trainer, test_metrics)

            # Step 10: Return results
            logger.info("Step 10/10: Preparing results")
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

            logger.info("=" * 80)
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Duration: {training_duration:.2f}s")
            logger.info(f"Test R²: {test_metrics.get('r2', 0):.4f}")
            logger.info(f"Test MAE: {test_metrics.get('mae', 0):.4f}")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            raise

    async def run_with_cross_validation(self) -> dict[str, Any]:
        """
        Run training with cross-validation.

        Returns:
            Dict with CV results
        """
        logger.info(f"Starting {self.config.experiment.cv_folds}-fold cross-validation")

        start_time = time.time()

        # Load and prepare data
        df = await self._load_data()
        features_df = await self._extract_features(df)

        # Get all features and target
        target_column = self.config.data.target_column
        feature_columns = (
            self.config.data.feature_columns
            if self.config.data.feature_columns
            else [col for col in features_df.columns if col != target_column]
        )

        X = features_df[feature_columns]
        y = features_df[target_column]

        # Initialize trainer and cross-validator
        trainer = self._initialize_trainer()
        cv = CrossValidator(
            n_splits=self.config.experiment.cv_folds,
            strategy="timeseries" if self.config.data.time_based_split else "kfold",
        )

        # Run cross-validation
        cv_results = cv.cross_validate(trainer, X, y)

        # Log results
        training_duration = time.time() - start_time
        self.experiment_tracker.log_params(self.config.dict())
        self.experiment_tracker.log_metrics(cv_results["mean_metrics"])

        # Save CV summary
        cv_summary = cv.get_cv_summary()
        self.experiment_tracker.log_dataframe(cv_summary, "cv_summary")
        experiment_dir = self.experiment_tracker.save_experiment()

        results = {
            "experiment_name": self.config.experiment.experiment_name,
            "experiment_dir": experiment_dir,
            "cv_results": cv_results,
            "training_duration": training_duration,
        }

        logger.info(f"Cross-validation complete: {cv_results['mean_metrics']}")

        return results

    async def _load_data(self) -> pd.DataFrame:
        """Load data from database safely for ML training."""
        stmt = (
            select(Post)
            .join(Channel, Post.channel_id == Channel.id)
            .where(Post.status == "sent")
        )

        if self.config.data.platforms:
            stmt = stmt.where(Channel.platform.in_(self.config.data.platforms))

        result = await self.session.execute(stmt)
        posts = result.scalars().all()

        if not posts:
            raise ValueError("No posts found matching criteria")

        df = pd.DataFrame(
            [
                {
                    "id": p.id,
                    "user_id": p.channel.organization.user_id
                    if p.channel and p.channel.organization
                    else None,
                    "channel_id": p.channel_id,
                    "platform": p.channel.platform if p.channel else None,
                    "content": p.content,
                    "published_at": p.published_at or p.sent_at or p.scheduled_at,
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

        df["total_engagement"] = df["likes"] + df["comments"] + df["shares"]

        df["content_length"] = df["content"].astype(str).str.len()

        logger.info(f"Loaded {len(df)} posts from database")

        return df

    async def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract ML features."""
        pipeline = FeatureEngineeringPipeline()
        features = await pipeline.extract_features(df, session=self.session)

        # Add target column
        features[self.config.data.target_column] = df[self.config.data.target_column]

        logger.info(f"Extracted {len(features.columns)} features")

        return features

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """Prepare train/val/test sets."""
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

    def _validate_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate data quality."""
        prep = DataPreparation()
        quality_report = prep.validate_data_quality(df)

        if quality_report["warnings"]:
            logger.warning("Data quality issues found:")
            for warning in quality_report["warnings"]:
                logger.warning(f"  - {warning}")

        return quality_report

    def _initialize_trainer(self) -> BaseTrainer:
        """Initialize model trainer."""
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

        # Build model with hyperparameters
        trainer.build_model(self.config.model.hyperparameters)

        return trainer

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Train model with checkpointing."""
        # Get trainer
        trainer = self._initialize_trainer()

        # Train
        training_metrics = trainer.train(X_train, y_train, X_val, y_val)

        return training_metrics

    def _log_experiment(
        self,
        trainer: BaseTrainer,
        metrics: dict[str, float],
        training_duration: float,
        quality_report: dict[str, Any],
    ) -> str:
        """Log experiment with tracker."""
        # Log parameters
        self.experiment_tracker.log_params(self.config.dict())

        # Log metrics
        self.experiment_tracker.log_metrics(metrics)
        self.experiment_tracker.log_metric("training_duration", training_duration)

        # Log quality report
        self.experiment_tracker.log_params({"data_quality": quality_report})

        # Log feature importance
        try:
            importance = trainer.get_feature_importance()
            self.experiment_tracker.log_dataframe(importance, "feature_importance")
        except Exception as e:
            logger.warning(f"Could not log feature importance: {e}")

        # Save experiment
        experiment_dir = self.experiment_tracker.save_experiment()

        return experiment_dir

    def _register_model(self, trainer: BaseTrainer, metrics: dict[str, float]) -> str:
        """Register model in registry."""
        # Save model
        model_path = self.experiment_tracker.experiment_dir / "model.joblib"
        trainer.save_model(str(model_path))

        # Generate version
        version = f"1.0.{len(self.model_registry.list_models())}"

        # Register
        model_id = self.model_registry.register_model(
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

        return model_id
