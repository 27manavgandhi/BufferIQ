# =========================
# PHONY
# =========================
.PHONY: help setup install setup-db migrate migration db-upgrade db-downgrade db-reset db-init \
test test-cov test-features test-training test-trainers test-evaluation test-ml test-optimization \
lint format type-check validate quality clean \
run docker-build docker-up docker-down docker-logs docker-test \
extract-features list-features \
train-baseline train-xgboost train-lightgbm train-model \
evaluate-model compare-models \
run-sync run-analysis \
optimize-grid optimize-random optimize-bayesian optimize-model \
optimize-optuna optimize-optuna-pruned optimize-multi-objective optimize-parallel \
resume-study analyze-importance optuna-dashboard list-studies advanced-optimize \
sync-initial sync-incremental sync-status sync-history \
dev-setup ml-pipeline

# =========================
# HELP
# =========================
help:
	@echo "BufferIQ Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make setup            Full project setup"
	@echo "  make setup-db         Initialize database"
	@echo "  make migrate          Run migrations"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run tests"
	@echo "  make test-cov         Coverage tests"
	@echo "  make test-training    Training tests"
	@echo "  make test-trainers    Trainer tests"
	@echo "  make test-evaluation  Evaluation tests"
	@echo "  make test-ml          ML tests"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make run-sync         Sync data"
	@echo "  make run-analysis     EDA"
	@echo "  make extract-features Feature extraction"
	@echo "  make train-model      Train model"
	@echo "  make evaluate-model   Evaluate model"
	@echo ""
	@echo "Optimization:"
	@echo "  make optimize-optuna  Optuna optimization"
	@echo "  make optimize-grid    Grid search"
	@echo "  make optimize-random  Random search"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Cleanup"

# =========================
# SETUP
# =========================
setup:
	python -m venv venv
	.\venv\Scripts\Activate.ps1 && pip install --upgrade pip
	.\venv\Scripts\Activate.ps1 && pip install -r backend/requirements.txt
	.\venv\Scripts\Activate.ps1 && pip install -e backend
	.\venv\Scripts\Activate.ps1 && pre-commit install
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
	mkdir -p outputs/models/checkpoints outputs/models/registry outputs/experiments outputs/features
	mkdir -p outputs/evaluations/reports outputs/evaluations/residual_plots outputs/evaluations/feature_importance
	@echo "Setup complete!"

install:
	cd backend && pip install -r requirements.txt
	cd backend && pip install -e .

setup-db:
	cd backend && bash scripts/init-db.sh

# =========================
# DATABASE
# =========================
migration:
	cd backend && alembic revision --autogenerate -m "$(msg)"

migrate:
	cd backend && alembic upgrade head

db-upgrade:
	cd backend && alembic upgrade head

db-downgrade:
	cd backend && alembic downgrade -1

db-reset:
	cd backend && alembic downgrade base && alembic upgrade head

db-init:
	docker-compose exec backend bash /scripts/init-db.sh

# =========================
# TESTING
# =========================
test:
	cd backend && pytest tests/ -v

test-cov:
	cd backend && pytest tests/ -v --cov=bufferiq --cov-report=term-missing --cov-report=html

test-features:
	cd backend && pytest tests/test_feature_pipeline.py -v

test-training:
	cd backend && pytest tests/test_training_pipeline.py -v

test-trainers:
	cd backend && pytest tests/test_xgboost_trainer.py tests/test_lightgbm_trainer.py -v

test-evaluation:
	cd backend && pytest tests/test_evaluator.py tests/test_comparator.py -v

test-ml:
	cd backend && pytest tests/ml/ -v

test-optimization:
	cd backend && pytest tests/ml/optimization/ -v

# =========================
# CODE QUALITY
# =========================
lint:
	cd backend && ruff bufferiq/
	cd backend && ruff tests/

format:
	cd backend && black bufferiq/ tests/
	cd backend && ruff bufferiq/ --fix
	cd backend && ruff tests/ --fix

type-check:
	cd backend && mypy bufferiq/ --strict

validate: lint test
quality: lint type-check test

# =========================
# RUN / DOCKER
# =========================
run:
	cd backend && uvicorn bufferiq.main:app --reload

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-test:
	docker-compose exec backend pytest tests/

# =========================
# FEATURE ENGINEERING
# =========================
extract-features:
	cd backend && python -m bufferiq.cli.features extract --user-id=1 --stats

list-features:
	cd backend && python -m bufferiq.cli.features list-features

# =========================
# ML PIPELINE
# =========================
run-sync:
	cd backend && python -m bufferiq.cli.sync run --full

run-analysis:
	cd backend && python scripts/run_analysis.py

train-model:
	cd backend && python scripts/train_model.py --config configs/training/xgboost.yaml

train-baseline:
	cd backend && python -m bufferiq.cli.train run --config ../configs/training/baseline.yaml

train-xgboost:
	cd backend && python -m bufferiq.cli.train run --config ../configs/training/xgboost.yaml

train-lightgbm:
	cd backend && python -m bufferiq.cli.train run --config ../configs/training/lightgbm.yaml

evaluate-model:
	cd backend && python scripts/evaluate_model.py --model-path outputs/models/registry/xgboost_v1.0.0.joblib

compare-models:
	cd backend && python -m bufferiq.cli.evaluate compare-all

# =========================
# OPTIMIZATION
# =========================
optimize-grid:
	cd backend && python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_grid.yaml

optimize-random:
	cd backend && python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_random.yaml

optimize-bayesian:
	cd backend && python -m bufferiq.cli.optimize run --config configs/optimization/xgboost_bayesian.yaml

optimize-model:
	cd backend && python scripts/optimize_model.py --config configs/optimization/xgboost_grid.yaml

optimize-optuna:
	cd backend && python -m bufferiq.cli.optimize optuna --config configs/optimization/xgboost_optuna.yaml

optimize-optuna-pruned:
	cd backend && python -m bufferiq.cli.optimize optuna --config configs/optimization/xgboost_optuna_pruned.yaml

optimize-multi-objective:
	cd backend && python -m bufferiq.cli.optimize multi-objective --config configs/optimization/xgboost_multi_objective.yaml

optimize-parallel:
	cd backend && python -m bufferiq.cli.optimize parallel --config configs/optimization/parallel_optimization.yaml

resume-study:
	cd backend && python -m bufferiq.cli.optimize resume --study-name $(STUDY_NAME) --n-trials 50

analyze-importance:
	cd backend && python -m bufferiq.cli.optimize importance --study-name $(STUDY_NAME)

optuna-dashboard:
	optuna-dashboard sqlite:///outputs/optimizations/optuna_studies/xgboost_001.db

list-studies:
	cd backend && python -m bufferiq.cli.optimize list-studies

advanced-optimize:
	cd backend && python scripts/advanced_optimize.py --config configs/optimization/xgboost_optuna.yaml --mode optuna

# =========================
# SYNC COMMANDS (LEGACY)
# =========================
sync-initial:
	cd backend && python -m bufferiq.cli.sync initial --user-id=$(USER_ID)

sync-incremental:
	cd backend && python -m bufferiq.cli.sync incremental --user-id=$(USER_ID)

sync-status:
	cd backend && python -m bufferiq.cli.sync status --user-id=$(USER_ID)

sync-history:
	cd backend && python -m bufferiq.cli.sync history --user-id=$(USER_ID)

# =========================
# WORKFLOWS
# =========================
dev-setup: install setup-db migrate
	@echo "Development ready!"

ml-pipeline: run-sync run-analysis extract-features train-model evaluate-model optimize-optuna
	@echo "ML pipeline complete!"

# =========================
# CLEANUP
# =========================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build dist htmlcov backend/htmlcov backend/.coverage