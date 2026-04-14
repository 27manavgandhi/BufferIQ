"""Data preparation utilities for ML training."""

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class DataPreparation:
    """Prepare data for ML training."""

    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42,
        stratify_column: Optional[str] = None,
        time_based_split: bool = True,
    ) -> None:
        """
        Initialize data preparation.

        Args:
            test_size: Proportion of data for test set (0.0-1.0)
            validation_size: Proportion of training data for validation (0.0-1.0)
            random_state: Random seed for reproducibility
            stratify_column: Column to stratify split (e.g., 'platform')
            time_based_split: If True, use temporal split (NO data leakage)

        Example:
            >>> prep = DataPreparation(test_size=0.2, validation_size=0.1)
            >>> X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(df, 'target', features)
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0")
        if not 0.0 <= validation_size < 1.0:
            raise ValueError("validation_size must be between 0.0 and 1.0")

        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.stratify_column = stratify_column
        self.time_based_split = time_based_split

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        time_column: Optional[str] = "published_at",
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split data into train/val/test sets.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            feature_columns: List of feature column names
            time_column: Column to use for temporal split

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test

        Raises:
            ValueError: If required columns missing or data too small
        """
        # Validate columns
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        # Check minimum sample size
        if len(df) < 100:
            raise ValueError(
                f"Insufficient data: {len(df)} samples. Minimum required: 100"
            )

        logger.info(f"Splitting {len(df)} samples into train/val/test")

        # Extract features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()

        if self.time_based_split and time_column and time_column in df.columns:
            # Time-based split (no data leakage)
            df_sorted = df.copy()
            df_sorted = df_sorted.sort_values(time_column)

            # Calculate split indices
            n = len(df_sorted)
            test_idx = int(n * (1 - self.test_size))
            train_val_idx = int(test_idx * (1 - self.validation_size))

            # Split data
            train_data = df_sorted.iloc[:train_val_idx]
            val_data = df_sorted.iloc[train_val_idx:test_idx]
            test_data = df_sorted.iloc[test_idx:]

            X_train = train_data[feature_columns]
            y_train = train_data[target_column]

            X_val = val_data[feature_columns]
            y_val = val_data[target_column]

            X_test = test_data[feature_columns]
            y_test = test_data[target_column]

            logger.info("Used time-based split (prevents data leakage)")

        else:
            # Random split
            stratify = df[self.stratify_column] if self.stratify_column else None

            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify,
            )

            # Second split: train vs val
            if self.validation_size > 0:
                val_size_adjusted = self.validation_size / (1 - self.test_size)

                stratify_temp = None
                if self.stratify_column and stratify is not None:
                    stratify_temp = y_temp.index.map(
                        lambda idx: df.loc[idx, self.stratify_column]
                    )

                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=val_size_adjusted,
                    random_state=self.random_state,
                    stratify=stratify_temp,
                )
            else:
                X_train = X_temp
                y_train = y_temp
                X_val = pd.DataFrame(columns=X_train.columns)
                y_val = pd.Series(dtype=y_train.dtype)

            logger.info("Used random split")

        logger.info(
            f"Split complete: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def validate_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Check data quality before training.

        Args:
            df: DataFrame to validate

        Returns:
            Dict with quality metrics and warnings

        Example:
            >>> quality = prep.validate_data_quality(df)
            >>> if quality['warnings']:
            ...     print(quality['warnings'])
        """
        quality_report: dict[str, Any] = {
            "total_samples": len(df),
            "missing_values": {},
            "duplicate_rows": 0,
            "constant_features": [],
            "warnings": [],
        }

        # Check missing values
        missing = df.isnull().sum()
        quality_report["missing_values"] = missing[missing > 0].to_dict()

        if quality_report["missing_values"]:
            quality_report["warnings"].append(
                f"Missing values found in {len(quality_report['missing_values'])} columns"
            )

        # Check duplicates
        duplicates = df.duplicated().sum()
        quality_report["duplicate_rows"] = int(duplicates)

        if duplicates > 0:
            quality_report["warnings"].append(f"Found {duplicates} duplicate rows")

        # Check constant features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() == 1:
                quality_report["constant_features"].append(col)

        if quality_report["constant_features"]:
            quality_report["warnings"].append(
                f"Constant features: {quality_report['constant_features']}"
            )

        # Check feature variance
        low_variance_features = []
        for col in numeric_cols:
            if df[col].var() < 1e-6:
                low_variance_features.append(col)

        if low_variance_features:
            quality_report["warnings"].append(
                f"Low variance features: {low_variance_features}"
            )

        # Platform distribution check
        if "platform" in df.columns:
            platform_counts = df["platform"].value_counts()
            quality_report["platform_distribution"] = platform_counts.to_dict()

            for platform, count in platform_counts.items():
                if count < 10:
                    quality_report["warnings"].append(
                        f"Platform '{platform}' has only {count} samples"
                    )

        logger.info(f"Data quality check: {len(quality_report['warnings'])} warnings")

        return quality_report

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Literal["drop", "mean", "median", "mode", "zero"] = "median",
    ) -> pd.DataFrame:
        """
        Handle missing values in dataset.

        Args:
            df: Input DataFrame
            strategy: How to handle missing values

        Returns:
            DataFrame with missing values handled

        Example:
            >>> df_clean = prep.handle_missing_values(df, strategy='median')
        """
        df_clean = df.copy()

        if strategy == "drop":
            df_clean = df_clean.dropna()
            logger.info(f"Dropped rows with missing values: {len(df) - len(df_clean)}")

        elif strategy in ["mean", "median", "mode"]:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    if strategy == "mean":
                        fill_value = df_clean[col].mean()
                    elif strategy == "median":
                        fill_value = df_clean[col].median()
                    else:  # mode
                        fill_value = df_clean[col].mode()[0]

                    df_clean[col] = df_clean[col].fillna(fill_value)

            logger.info(f"Filled missing values using {strategy}")

        elif strategy == "zero":
            df_clean = df_clean.fillna(0)
            logger.info("Filled missing values with zero")

        return df_clean

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: list[str],
        method: Literal["iqr", "zscore"] = "iqr",
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Remove outliers from dataset.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: Outlier detection method
            threshold: IQR multiplier or z-score threshold

        Returns:
            DataFrame with outliers removed

        Example:
            >>> df_clean = prep.remove_outliers(df, ['engagement_rate'], method='iqr')
        """
        df_clean = df.copy()
        initial_size = len(df_clean)

        for col in columns:
            if col not in df_clean.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                ]

            elif method == "zscore":
                z_scores = np.abs(
                    (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
                )
                df_clean = df_clean[z_scores < threshold]

        removed = initial_size - len(df_clean)
        logger.info(f"Removed {removed} outliers using {method} method")

        return df_clean
