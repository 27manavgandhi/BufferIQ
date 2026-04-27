"""Automated ensemble construction pipeline."""

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from bufferiq.core.logging import get_logger
from bufferiq.ml.ensemble.blending import BlendingEnsemble
from bufferiq.ml.ensemble.diversity_analyzer import DiversityAnalyzer
from bufferiq.ml.ensemble.model_selector import ModelSelector
from bufferiq.ml.ensemble.stacking import StackingEnsemble
from bufferiq.ml.ensemble.voting import VotingEnsemble
from bufferiq.ml.ensemble.weight_optimizer import WeightOptimizer
from bufferiq.ml.ensemble.weighted_average import WeightedAverageEnsemble

logger = get_logger(__name__)


class EnsembleBuilder:
    """
    Automated ensemble construction pipeline.

    Orchestrates the complete ensemble building process:
    1. Load trained base models
    2. Analyze model diversity
    3. Select complementary models
    4. Choose ensemble type
    5. Optimize weights (if applicable)
    6. Train final ensemble
    7. Evaluate and save

    Example:
        >>> builder = EnsembleBuilder(
        ...     model_paths=['model1.joblib', 'model2.joblib'],
        ...     ensemble_type='stacking'
        ... )
        >>> ensemble = builder.build(X_train, y_train, X_val, y_val)
        >>> ensemble.save('outputs/models/ensembles/ensemble.joblib')
    """

    def __init__(
        self,
        model_paths: list[Path],
        ensemble_type: str = "auto",
        min_performance: float = 0.70,
        min_diversity: float = 0.10,
        max_models: int = 5,
        weight_optimization: str = "optuna",
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize ensemble builder.

        Args:
            model_paths: Paths to trained model files
            ensemble_type: Type of ensemble ('voting', 'stacking', 'blending',
                          'weighted_average', 'auto')
            min_performance: Minimum R² for model selection
            min_diversity: Minimum diversity for model selection
            max_models: Maximum models in ensemble
            weight_optimization: Method for weight optimization
            output_dir: Directory for saving outputs

        Raises:
            ValueError: If model_paths is empty or ensemble_type invalid

        Example:
            >>> builder = EnsembleBuilder(
            ...     model_paths=[Path('model1.joblib'), Path('model2.joblib')],
            ...     ensemble_type='stacking'
            ... )
        """
        if not model_paths:
            raise ValueError("model_paths cannot be empty")

        valid_types = ["voting", "stacking", "blending", "weighted_average", "auto"]
        if ensemble_type not in valid_types:
            raise ValueError(
                f"ensemble_type must be one of {valid_types}, got {ensemble_type}"
            )

        self.model_paths = [Path(p) for p in model_paths]
        self.ensemble_type = ensemble_type
        self.min_performance = min_performance
        self.min_diversity = min_diversity
        self.max_models = max_models
        self.weight_optimization = weight_optimization
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/ensembles")

        logger.info(
            f"EnsembleBuilder initialized: ensemble_type={ensemble_type}, "
            f"n_models={len(model_paths)}"
        )

    def load_models(self) -> list[BaseEstimator]:
        """
        Load base models from disk.

        Returns:
            List of loaded models

        Raises:
            FileNotFoundError: If model file not found

        Example:
            >>> models = builder.load_models()
            >>> print(f"Loaded {len(models)} models")
        """
        logger.info(f"Loading {len(self.model_paths)} base models")

        models = []
        for path in self.model_paths:
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")

            model = joblib.load(path)
            models.append(model)
            logger.debug(f"Loaded model from {path}")

        logger.info(f"Successfully loaded {len(models)} models")
        return models

    def analyze_diversity(
        self,
        models: list[BaseEstimator],
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """
        Analyze diversity of base models.

        Args:
            models: List of models
            X: Features for diversity analysis
            y: Targets for diversity analysis

        Returns:
            Dictionary with diversity metrics

        Example:
            >>> diversity = builder.analyze_diversity(models, X_val, y_val)
            >>> print(f"Diversity: {diversity['correlation_diversity']:.4f}")
        """
        logger.info("Analyzing model diversity")

        # Get predictions
        predictions = np.column_stack([model.predict(X) for model in models])

        # Calculate metrics
        model_names = [f"Model_{i+1}" for i in range(len(models))]

        diversity_metrics = DiversityAnalyzer.analyze_all(
            predictions, y, model_names, self.output_dir / "diversity_analysis"
        )

        return diversity_metrics

    def select_models(
        self,
        models: list[BaseEstimator],
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[BaseEstimator]:
        """
        Select diverse, high-performing models.

        Args:
            models: Candidate models
            X: Validation features
            y: Validation targets

        Returns:
            Selected models

        Example:
            >>> selected = builder.select_models(models, X_val, y_val)
            >>> print(f"Selected {len(selected)} models")
        """
        logger.info("Selecting models for ensemble")

        selector = ModelSelector(
            min_performance=self.min_performance,
            min_diversity=self.min_diversity,
            max_models=self.max_models,
        )

        selected_indices = selector.select(models, X, y)
        selected_models = [models[i] for i in selected_indices]

        logger.info(f"Selected {len(selected_models)} models: {selected_indices}")

        # Save selection details
        selection_info = {
            "selected_indices": selected_indices,
            "total_candidates": len(models),
            "min_performance": self.min_performance,
            "min_diversity": self.min_diversity,
        }

        output_path = self.output_dir / "model_selection" / "selected_models.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(selection_info, f, indent=2)

        return selected_models

    def build_voting(
        self,
        models: list[BaseEstimator],
        X: np.ndarray,
        y: np.ndarray,
    ) -> VotingEnsemble:
        """
        Build voting ensemble.

        Args:
            models: Base models
            X: Training features
            y: Training targets

        Returns:
            Fitted voting ensemble
        """
        logger.info("Building voting ensemble")

        # Optimize weights
        optimizer = WeightOptimizer(
            base_models=models,
            method=self.weight_optimization,
        )
        weights = optimizer.optimize(X, y)

        # Create ensemble
        ensemble = VotingEnsemble(base_models=models, weights=weights)
        ensemble.fit(X, y)

        logger.info(f"Voting ensemble created with weights: {weights}")

        return ensemble

    def build_stacking(
        self,
        models: list[BaseEstimator],
        X: np.ndarray,
        y: np.ndarray,
    ) -> StackingEnsemble:
        """
        Build stacking ensemble.

        Args:
            models: Base models
            X: Training features
            y: Training targets

        Returns:
            Fitted stacking ensemble
        """
        logger.info("Building stacking ensemble")

        # Use Ridge as meta-learner
        meta_learner = Ridge(alpha=1.0)

        # Create ensemble
        ensemble = StackingEnsemble(
            base_models=models,
            meta_learner=meta_learner,
            cv=5,
            passthrough=False,
        )
        ensemble.fit(X, y)

        logger.info("Stacking ensemble created")

        return ensemble

    def build_blending(
        self,
        models: list[BaseEstimator],
        X: np.ndarray,
        y: np.ndarray,
    ) -> BlendingEnsemble:
        """
        Build blending ensemble.

        Args:
            models: Base models
            X: Training features
            y: Training targets

        Returns:
            Fitted blending ensemble
        """
        logger.info("Building blending ensemble")

        # Use Ridge as meta-learner
        meta_learner = Ridge(alpha=0.5)

        # Create ensemble
        ensemble = BlendingEnsemble(
            base_models=models,
            meta_learner=meta_learner,
            blend_split=0.3,
        )
        ensemble.fit(X, y)

        logger.info("Blending ensemble created")

        return ensemble

    def build_weighted_average(
        self,
        models: list[BaseEstimator],
        X: np.ndarray,
        y: np.ndarray,
    ) -> WeightedAverageEnsemble:
        """
        Build weighted average ensemble.

        Args:
            models: Base models
            X: Training features
            y: Training targets

        Returns:
            Fitted weighted average ensemble
        """
        logger.info("Building weighted average ensemble")

        # Create ensemble
        ensemble = WeightedAverageEnsemble(
            base_models=models,
            weight_method=self.weight_optimization,
        )
        ensemble.fit(X, y)

        logger.info("Weighted average ensemble created")

        return ensemble

    def build(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> BaseEstimator:
        """
        Build complete ensemble.

        Full pipeline:
        1. Load models
        2. Analyze diversity
        3. Select models
        4. Build ensemble
        5. Save results

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Fitted ensemble

        Example:
            >>> ensemble = builder.build(X_train, y_train, X_val, y_val)
            >>> predictions = ensemble.predict(X_test)
        """
        logger.info("Starting automated ensemble building")

        # Load models
        models = self.load_models()

        # Analyze diversity
        diversity_metrics = self.analyze_diversity(models, X_val, y_val)
        logger.info(f"Diversity metrics: {diversity_metrics}")

        # Select models
        selected_models = self.select_models(models, X_val, y_val)

        # Build ensemble
        if self.ensemble_type == "voting":
            ensemble = self.build_voting(selected_models, X_train, y_train)
        elif self.ensemble_type == "stacking":
            ensemble = self.build_stacking(selected_models, X_train, y_train)
        elif self.ensemble_type == "blending":
            ensemble = self.build_blending(selected_models, X_train, y_train)
        elif self.ensemble_type == "weighted_average":
            ensemble = self.build_weighted_average(selected_models, X_train, y_train)
        elif self.ensemble_type == "auto":
            # Try all and pick best
            logger.info("Auto mode: trying all ensemble types")
            ensemble = self._auto_select(
                selected_models, X_train, y_train, X_val, y_val
            )
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")

        logger.info("Ensemble building complete")

        return ensemble

    def _auto_select(
        self,
        models: list[BaseEstimator],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> BaseEstimator:
        """
        Automatically select best ensemble type.

        Args:
            models: Selected models
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Best performing ensemble
        """
        from sklearn.metrics import r2_score

        ensembles = {
            "voting": self.build_voting(models, X_train, y_train),
            "stacking": self.build_stacking(models, X_train, y_train),
            "blending": self.build_blending(models, X_train, y_train),
            "weighted_average": self.build_weighted_average(models, X_train, y_train),
        }

        # Evaluate on validation set
        scores = {}
        for name, ensemble in ensembles.items():
            pred = ensemble.predict(X_val)
            score = r2_score(y_val, pred)
            scores[name] = score
            logger.info(f"{name} ensemble R²: {score:.4f}")

        # Select best
        best_type = max(scores, key=scores.get)
        best_ensemble = ensembles[best_type]

        logger.info(f"Auto-selected {best_type} ensemble (R²={scores[best_type]:.4f})")

        return best_ensemble
