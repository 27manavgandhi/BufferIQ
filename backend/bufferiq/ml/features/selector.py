"""Feature selection utilities."""

from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)

SelectionMethod = Literal["variance", "correlation", "mutual_info", "k_best"]


class FeatureSelector:
    """Select most important features."""

    def __init__(
        self,
        method: SelectionMethod = "mutual_info",
        threshold: float = 0.8,
        k: int = 20,
    ) -> None:
        """
        Initialize feature selector.

        Args:
            method: Selection method
            threshold: Threshold for variance/correlation methods
            k: Number of features to select for k_best method

        Example:
            >>> selector = FeatureSelector(method="mutual_info", k=20)
            >>> selector.fit(X_train, y_train)
            >>> X_selected = selector.transform(X_test)
        """
        self.method = method
        self.threshold = threshold
        self.k = k
        self._selector: Optional[VarianceThreshold | SelectKBest] = None
        self._selected_features: list[str] = []
        self._feature_importance: pd.DataFrame = pd.DataFrame()
        self._is_fitted = False
        self._all_features: list[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        """
        Fit selector on training data.

        Args:
            X: Feature DataFrame
            y: Target series (required for mutual_info and k_best)

        Returns:
            Self (for method chaining)

        Raises:
            ValueError: If y is None for methods that require it
        """
        self._all_features = X.columns.tolist()

        if self.method == "variance":
            self._selector = VarianceThreshold(threshold=self.threshold)
            self._selector.fit(X)
            mask = self._selector.get_support()
            self._selected_features = X.columns[mask].tolist()

            # Feature importance (variance)
            variances = X.var()
            self._feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": variances}
            ).sort_values("importance", ascending=False)

        elif self.method == "correlation":
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > self.threshold)
            ]
            self._selected_features = [col for col in X.columns if col not in to_drop]

            # Feature importance (mean absolute correlation)
            mean_corr = corr_matrix.mean().abs()
            self._feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": mean_corr}
            ).sort_values("importance", ascending=False)

        elif self.method == "mutual_info":
            if y is None:
                raise ValueError("Target y required for mutual_info method")

            self._selector = SelectKBest(
                score_func=mutual_info_regression, k=min(self.k, len(X.columns))
            )
            self._selector.fit(X, y)
            mask = self._selector.get_support()
            self._selected_features = X.columns[mask].tolist()

            # Feature importance (mutual info scores)
            scores = self._selector.scores_
            self._feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": scores}
            ).sort_values("importance", ascending=False)

        elif self.method == "k_best":
            if y is None:
                raise ValueError("Target y required for k_best method")

            self._selector = SelectKBest(
                score_func=f_regression, k=min(self.k, len(X.columns))
            )
            self._selector.fit(X, y)
            mask = self._selector.get_support()
            self._selected_features = X.columns[mask].tolist()

            # Feature importance (F-scores)
            scores = self._selector.scores_
            self._feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": scores}
            ).sort_values("importance", ascending=False)

        else:
            raise ValueError(
                f"Invalid selection method: {self.method}. "
                f"Choose from: 'variance', 'correlation', 'mutual_info', 'k_best'"
            )

        self._is_fitted = True

        logger.info(
            f"Selected {len(self._selected_features)}/{len(X.columns)} features "
            f"using {self.method} method"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform by selecting features.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with selected features only

        Raises:
            ValueError: If selector not fitted
        """
        if not self._is_fitted:
            raise ValueError("Selector must be fitted before transform")

        return X[self._selected_features].copy()

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Feature DataFrame
            y: Target series (required for some methods)

        Returns:
            DataFrame with selected features
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance scores.

        Returns:
            DataFrame with features and importance scores

        Raises:
            ValueError: If selector not fitted
        """
        if not self._is_fitted:
            raise ValueError("Selector must be fitted before getting importance")

        return self._feature_importance.copy()

    def get_selected_features(self) -> list[str]:
        """
        Return list of selected feature names.

        Returns:
            List of selected feature names

        Raises:
            ValueError: If selector not fitted
        """
        if not self._is_fitted:
            raise ValueError("Selector must be fitted before getting features")

        return self._selected_features.copy()

    def save(self, path: str) -> None:
        """
        Save fitted selector to disk.

        Args:
            path: File path to save selector

        Raises:
            ValueError: If selector not fitted
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted selector")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "selector": self._selector,
                "method": self.method,
                "threshold": self.threshold,
                "k": self.k,
                "selected_features": self._selected_features,
                "feature_importance": self._feature_importance,
                "all_features": self._all_features,
            },
            path,
        )

        logger.info(f"Saved selector to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureSelector":
        """
        Load fitted selector from disk.

        Args:
            path: File path to load selector from

        Returns:
            Loaded FeatureSelector instance

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Selector file not found: {path}")

        data = joblib.load(path)

        selector = cls(method=data["method"], threshold=data["threshold"], k=data["k"])
        selector._selector = data["selector"]
        selector._selected_features = data["selected_features"]
        selector._feature_importance = data["feature_importance"]
        selector._all_features = data["all_features"]
        selector._is_fitted = True

        logger.info(f"Loaded selector from {path}")

        return selector
