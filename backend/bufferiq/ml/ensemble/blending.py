"""Blending ensemble implementation."""


import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split

from bufferiq.core.logging import get_logger
from bufferiq.ml.ensemble.base import BaseEnsemble

logger = get_logger(__name__)


class BlendingEnsemble(BaseEnsemble):
    """
    Blending ensemble with holdout validation set.

    Simpler alternative to stacking that uses a single holdout set
    instead of cross-validation for generating meta-features.

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> from xgboost import XGBRegressor
        >>> from lightgbm import LGBMRegressor
        >>>
        >>> base_models = [
        ...     XGBRegressor(n_estimators=100),
        ...     LGBMRegressor(n_estimators=100)
        ... ]
        >>> meta_learner = Ridge(alpha=0.5)
        >>>
        >>> ensemble = BlendingEnsemble(base_models, meta_learner, blend_split=0.3)
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        base_models: list[BaseEstimator],
        meta_learner: BaseEstimator,
        blend_split: float = 0.3,
        random_state: int = 42,
    ) -> None:
        """
        Initialize blending ensemble.

        Args:
            base_models: List of base models
            meta_learner: Meta-model
            blend_split: Fraction of training data for blending (0.2-0.4 recommended)
            random_state: Random seed for reproducibility

        Raises:
            ValueError: If base_models is empty or blend_split invalid

        Example:
            >>> ensemble = BlendingEnsemble(
            ...     base_models=[model1, model2],
            ...     meta_learner=Ridge(),
            ...     blend_split=0.3
            ... )
        """
        super().__init__()

        if not base_models:
            raise ValueError("base_models cannot be empty")

        if not 0.0 < blend_split < 1.0:
            raise ValueError(f"blend_split must be in (0, 1), got {blend_split}")

        self.base_models = [clone(model) for model in base_models]
        self.meta_learner = clone(meta_learner)
        self.blend_split = blend_split
        self.random_state = random_state

        logger.info(
            f"BlendingEnsemble initialized with {len(base_models)} base models, "
            f"blend_split={blend_split}"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlendingEnsemble":
        """
        Fit blending ensemble.

        1. Split data into train and blend sets
        2. Train base models on train set
        3. Get predictions on blend set
        4. Train meta-learner on blend predictions
        5. Refit base models on full training data

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)

        Returns:
            self: Fitted ensemble

        Example:
            >>> ensemble.fit(X_train, y_train)
        """
        self.validate_inputs(X, y)

        # Split into train and blend sets
        logger.info(f"Splitting data: blend_split={self.blend_split}")
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=self.blend_split, random_state=self.random_state
        )

        logger.info(
            f"Train set: {len(X_train)} samples, Blend set: {len(X_blend)} samples"
        )

        # Train base models on train set
        logger.info("Training base models on train set")
        blend_predictions = []

        for i, model in enumerate(self.base_models):
            logger.debug(f"Training base model {i+1}")
            model.fit(X_train, y_train)

            # Get predictions on blend set
            pred = model.predict(X_blend)
            blend_predictions.append(pred)

        blend_features = np.column_stack(blend_predictions)

        # Train meta-learner on blend predictions
        logger.info("Training meta-learner on blend predictions")
        self.meta_learner.fit(blend_features, y_blend)

        # Refit base models on full training data
        logger.info("Refitting base models on full training data")
        for i, model in enumerate(self.base_models):
            logger.debug(f"Refitting base model {i+1}")
            model.fit(X, y)

        self._is_fitted = True
        logger.info("BlendingEnsemble training complete")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using blending.

        1. Get predictions from base models
        2. Feed to meta-learner

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Meta-learner predictions, shape (n_samples,)

        Raises:
            ValueError: If ensemble is not fitted

        Example:
            >>> predictions = ensemble.predict(X_test)
        """
        self.check_is_fitted()
        self.validate_inputs(X)

        # Get predictions from base models
        base_predictions = np.column_stack(
            [model.predict(X) for model in self.base_models]
        )

        # Get meta-learner prediction
        predictions = self.meta_learner.predict(base_predictions)

        logger.debug(f"Generated predictions for {len(X)} samples")

        return predictions

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BlendingEnsemble(n_base_models={len(self.base_models)}, "
            f"meta_learner={self.meta_learner.__class__.__name__}, "
            f"blend_split={self.blend_split})"
        )
