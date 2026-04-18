"""Tests for LightGBM trainer."""

import pandas as pd
import pytest

from bufferiq.ml.trainers.lightgbm_trainer import LightGBMTrainer


class TestLightGBMTrainer:
    """Test LightGBM trainer."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create sample data."""
        X = pd.DataFrame(
            {
                "feature1": list(range(100)),
                "feature2": list(range(100, 200)),
                "feature3": list(range(200, 300)),
            }
        )
        y = pd.Series([10 + i * 0.1 for i in range(100)])
        return X, y

    def test_init(self) -> None:
        """Test initialization."""
        trainer = LightGBMTrainer(random_state=42)

        assert trainer.model_name == "lightgbm"
        assert trainer.random_state == 42
        assert trainer.model is None

    def test_build_model(self) -> None:
        """Test model building."""
        trainer = LightGBMTrainer()
        hyperparams = {"n_estimators": 50, "max_depth": 5}

        model = trainer.build_model(hyperparams)

        assert model is not None
        assert trainer.model is not None
        assert trainer.hyperparameters["n_estimators"] == 50

    def test_train(self, sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test model training."""
        X, y = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        trainer = LightGBMTrainer(verbose=False)
        trainer.build_model({"n_estimators": 10})

        metrics = trainer.train(X_train, y_train, X_val, y_val)

        assert "train_mae" in metrics
        assert "train_rmse" in metrics
        assert "train_r2" in metrics
        assert "val_mae" in metrics
        assert "val_r2" in metrics

    def test_train_without_validation(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test training without validation set."""
        X, y = sample_data

        trainer = LightGBMTrainer(verbose=False)
        trainer.build_model({"n_estimators": 10})

        metrics = trainer.train(X, y)

        assert "train_mae" in metrics
        assert "val_mae" not in metrics

    def test_predict(self, sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test predictions."""
        X, y = sample_data

        trainer = LightGBMTrainer(verbose=False)
        trainer.build_model({"n_estimators": 10})
        trainer.train(X, y)

        predictions = trainer.predict(X)

        assert len(predictions) == len(X)
        assert predictions.shape == (len(X),)

    def test_predict_untrained_raises_error(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test predict before training raises error."""
        X, _ = sample_data
        trainer = LightGBMTrainer()

        with pytest.raises(ValueError, match="Model not trained"):
            trainer.predict(X)

    def test_evaluate(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test model evaluation."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        trainer = LightGBMTrainer(verbose=False)
        trainer.build_model({"n_estimators": 10})
        trainer.train(X_train, y_train)

        metrics = trainer.evaluate(X_test, y_test)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert isinstance(metrics["mae"], float)

    def test_get_feature_importance(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Test feature importance extraction."""
        X, y = sample_data

        trainer = LightGBMTrainer(verbose=False)
        trainer.build_model({"n_estimators": 10})
        trainer.train(X, y)

        importance = trainer.get_feature_importance()

        assert len(importance) == 3  # 3 features
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert importance["importance"].sum() > 0

    def test_save_and_load_model(
        self, sample_data: tuple[pd.DataFrame, pd.Series], tmp_path: str
    ) -> None:
        """Test model save and load."""
        X, y = sample_data

        # Train and save
        trainer = LightGBMTrainer(verbose=False)
        trainer.build_model({"n_estimators": 10})
        trainer.train(X, y)

        model_path = tmp_path / "model.joblib"
        trainer.save_model(str(model_path))

        # Load
        loaded_trainer = LightGBMTrainer.load_model(str(model_path))

        # Compare predictions
        original_pred = trainer.predict(X)
        loaded_pred = loaded_trainer.predict(X)

        assert (original_pred == loaded_pred).all()