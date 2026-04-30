# =========================
# PHONY
# =========================
.PHONY: help setup install setup-db migrate migration db-upgrade db-downgrade db-reset db-init \
test test-cov test-features test-training test-trainers test-evaluation test-ml test-optimization test-ensemble \
lint format type-check validate quality clean \
run docker-build docker-up docker-down docker-logs docker-test \
extract-features list-features \
train-baseline train-xgboost train-lightgbm train-model \
evaluate-model compare-models \
run-sync run-analysis \
optimize-grid optimize-random optimize-bayesian optimize-model \
optimize-optuna optimize-optuna-pruned optimize-multi-objective optimize-parallel \
resume-study analyze-importance optuna-dashboard list-studies advanced-optimize \
ensemble-voting ensemble-stacking ensemble-auto ensemble-analyze-diversity ensemble-compare ensemble-build-production ensemble-all \
api-dev api-prod api-test api-benchmark api-load-test api-docker-build api-docker-run api-docker-stop test-api-all \
sync-initial sync-incremental sync-status sync-history \
dev-setup ml-pipeline

# =========================
# HELP
# =========================
help:
	@echo "BufferIQ Development Commands"

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
	mkdir -p outputs/models/ensembles outputs/ensembles
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

test-ensemble:
	cd backend && pytest tests/test_ensemble*.py -v --tb=short

# =========================
# CODE QUALITY
# =========================
lint:
	cd backend && ruff bufferiq/ tests/

format:
	cd backend && black bufferiq/ tests/
	cd backend && ruff bufferiq/ tests/ --fix

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
# ENSEMBLE
# =========================
ensemble-voting:
	cd backend && python -m bufferiq.cli.ensemble voting --models outputs/models/xgboost_best.joblib outputs/models/lightgbm_best.joblib outputs/models/random_forest_best.joblib --train-data data/processed/train.npz --output outputs/models/ensembles/voting_ensemble.joblib

ensemble-stacking:
	cd backend && python -m bufferiq.cli.ensemble stacking --models outputs/models/xgboost_best.joblib outputs/models/lightgbm_best.joblib outputs/models/random_forest_best.joblib --train-data data/processed/train.npz --cv 5 --output outputs/models/ensembles/stacking_ensemble.joblib

ensemble-auto:
	cd backend && python -m bufferiq.cli.ensemble auto --models outputs/models/xgboost_best.joblib outputs/models/lightgbm_best.joblib outputs/models/random_forest_best.joblib --train-data data/processed/train.npz --val-data data/processed/val.npz --output outputs/models/ensembles/auto_ensemble.joblib

ensemble-analyze-diversity:
	cd backend && python -m bufferiq.cli.ensemble analyze-diversity --models outputs/models/xgboost_best.joblib outputs/models/lightgbm_best.joblib outputs/models/random_forest_best.joblib --val-data data/processed/val.npz --output-dir outputs/ensembles/diversity

ensemble-compare:
	cd backend && python -m bufferiq.cli.ensemble compare --ensemble outputs/models/ensembles/stacking_ensemble.joblib --models outputs/models/xgboost_best.joblib outputs/models/lightgbm_best.joblib outputs/models/random_forest_best.joblib --test-data data/processed/test.npz --output-dir outputs/ensembles/comparison

ensemble-build-production:
	cd backend && python scripts/build_ensemble.py --config configs/ensemble/production_ensemble.yaml --train-data data/processed/train.npz --val-data data/processed/val.npz --test-data data/processed/test.npz --output-dir outputs/models/ensembles/production

ensemble-all: ensemble-analyze-diversity ensemble-auto ensemble-compare

# =========================
# API (DAY 14)
# =========================
api-dev:
	cd backend && python scripts/start_api.py --config configs/api/development.yaml --reload

api-prod:
	cd backend && python scripts/start_api.py --config configs/api/production.yaml

api-test:
	pytest backend/tests/api/ -v --cov=bufferiq/api --cov-report=term-missing --cov-fail-under=90

api-benchmark:
	cd backend && python scripts/benchmark_api.py --url http://localhost:8000 --requests 100

api-load-test:
	cd backend && python scripts/load_test.py --url http://localhost:8000 --users 10 --duration 60

api-docker-build:
	docker build -t bufferiq-api:latest -f backend/Dockerfile backend/

api-docker-run:
	docker-compose -f docker-compose.yml up -d

api-docker-stop:
	docker-compose -f docker-compose.yml down

test-api-all: api-test

# =========================
# WORKFLOWS
# =========================
dev-setup: install setup-db migrate
ml-pipeline: run-sync run-analysis extract-features train-model evaluate-model optimize-optuna ensemble-auto

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