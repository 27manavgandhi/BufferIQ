"""Stacking ensemble implementation."""


import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict

from bufferiq.core.logging import get_logger
from bufferiq.ml.ensemble.base import BaseEnsemble

logger = get_logger(__name__)


class StackingEnsemble(BaseEnsemble):
    """
    Stacking ensemble with meta-learner.

    Uses out-of-fold predictions to prevent overfitting.
    Level 0: Base models make predictions
    Level 1: Meta-learner learns from base model predictions

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> from xgboost import XGBRegressor
        >>> from lightgbm import LGBMRegressor
        >>>
        >>> base_models = [
        ...     XGBRegressor(n_estimators=100),
        ...     LGBMRegressor(n_estimators=100)
        ... ]
        >>> meta_learner = Ridge(alpha=1.0)
        >>>
        >>> ensemble = StackingEnsemble(base_models, meta_learner, cv=5)
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        base_models: list[BaseEstimator],
        meta_learner: BaseEstimator,
        cv: int = 5,
        passthrough: bool = False,
    ) -> None:
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of base models (level 0)
            meta_learner: Meta-model (level 1)
            cv: Number of cross-validation folds for out-of-fold predictions
            passthrough: Whether to include original features in meta-learner

        Raises:
            ValueError: If base_models is empty or cv < 2

        Example:
            >>> ensemble = StackingEnsemble(
            ...     base_models=[model1, model2],
            ...     meta_learner=Ridge(),
            ...     cv=5,
            ...     passthrough=False
            ... )
        """
        super().__init__()

        if not base_models:
            raise ValueError("base_models cannot be empty")

        if cv < 2:
            raise ValueError(f"cv must be >= 2, got {cv}")

        self.base_models = [clone(model) for model in base_models]
        self.meta_learner = clone(meta_learner)
        self.cv = cv
        self.passthrough = passthrough

        logger.info(
            f"StackingEnsemble initialized with {len(base_models)} base models, "
            f"cv={cv}, passthrough={passthrough}"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """
        Fit stacking ensemble.

        1. Generate out-of-fold predictions from base models
        2. Train meta-learner on out-of-fold predictions
        3. Refit base models on full training data

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)

        Returns:
            self: Fitted ensemble

        Example:
            >>> ensemble.fit(X_train, y_train)
        """
        self.validate_inputs(X, y)

        logger.info("Generating out-of-fold predictions from base models")

        # Generate out-of-fold predictions
        meta_features = []
        for i, model in enumerate(self.base_models):
            logger.debug(f"Generating OOF predictions for base model {i+1}")

            oof_predictions = cross_val_predict(
                model, X, y, cv=self.cv, method="predict"
            )
            meta_features.append(oof_predictions)

        meta_features_array = np.column_stack(meta_features)

        # Optionally include original features
        if self.passthrough:
            logger.info("Including original features (passthrough=True)")
            meta_features_array = np.hstack([meta_features_array, X])

        # Train meta-learner
        logger.info("Training meta-learner")
        self.meta_learner.fit(meta_features_array, y)

        # Refit base models on full training data
        logger.info("Refitting base models on full training data")
        for i, model in enumerate(self.base_models):
            logger.debug(f"Refitting base model {i+1}")
            model.fit(X, y)

        self._is_fitted = True
        logger.info("StackingEnsemble training complete")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacking.

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

        # Optionally include original features
        if self.passthrough:
            meta_input = np.hstack([base_predictions, X])
        else:
            meta_input = base_predictions

        # Get meta-learner prediction
        predictions = self.meta_learner.predict(meta_input)

        logger.debug(f"Generated predictions for {len(X)} samples")

        return predictions

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StackingEnsemble(n_base_models={len(self.base_models)}, "
            f"meta_learner={self.meta_learner.__class__.__name__}, "
            f"cv={self.cv}, passthrough={self.passthrough})"
        )
