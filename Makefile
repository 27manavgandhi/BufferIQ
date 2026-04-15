# =========================
# PHONY
# =========================
.PHONY: help setup install test test-cov test-features test-training lint format type-check clean run docker-build docker-up docker-down docker-test docker-logs db-migrate db-upgrade db-downgrade db-reset db-init validate migration migrate extract-features list-features train-baseline train-xgboost list-experiments list-models sync-initial sync-incremental sync-status sync-history

# =========================
# HELP
# =========================
help:
	@echo "BufferIQ Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup              - Initial project setup"
	@echo "  install            - Install dependencies"
	@echo ""
	@echo "Development:"
	@echo "  run                - Start development server"
	@echo "  docker-build       - Build Docker images"
	@echo "  docker-up          - Start all services"
	@echo "  docker-down        - Stop all services"
	@echo "  docker-logs        - View service logs"
	@echo ""
	@echo "Testing:"
	@echo "  test               - Run tests"
	@echo "  test-cov           - Run tests with coverage"
	@echo "  test-features      - Feature engineering tests"
	@echo "  test-training      - Training pipeline tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint               - Run linters"
	@echo "  format             - Format code"
	@echo "  type-check         - Run mypy"
	@echo "  validate           - Run all checks"
	@echo ""
	@echo "Database:"
	@echo "  migration          - Create migration"
	@echo "  migrate            - Apply migrations"
	@echo "  db-upgrade         - Upgrade DB"
	@echo "  db-downgrade       - Downgrade DB"
	@echo "  db-reset           - Reset DB"
	@echo "  db-init            - Init DB"
	@echo ""
	@echo "Feature Engineering:"
	@echo "  extract-features   - Extract features"
	@echo "  list-features      - List features"
	@echo ""
	@echo "Model Training:"
	@echo "  train-baseline     - Train baseline model"
	@echo "  train-xgboost      - Train XGBoost model"
	@echo "  list-experiments   - List experiments"
	@echo "  list-models        - List models"
	@echo ""
	@echo "Utilities:"
	@echo "  clean              - Cleanup"

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
	mkdir -p outputs/models/checkpoints
	mkdir -p outputs/models/registry
	mkdir -p outputs/experiments
	mkdir -p outputs/features
	@echo "Setup complete! Activate venv: .\venv\Scripts\Activate.ps1"

install:
	cd backend && pip install -r requirements.txt
	cd backend && pip install -e .

# =========================
# TESTING
# =========================
test:
	cd backend && pytest tests/ -v

test-cov:
	cd backend && pytest tests/ -v --cov=bufferiq --cov-report=term-missing --cov-report=html

test-features:
	cd backend && pytest tests/test_temporal_features.py tests/test_content_features.py tests/test_nlp_features.py tests/test_engagement_features.py tests/test_platform_features.py tests/test_feature_scaler.py tests/test_feature_selector.py tests/test_feature_pipeline.py -v --cov=bufferiq/ml/features --cov-report=term-missing --cov-fail-under=90

test-training:
	cd backend && pytest tests/test_data_preparation.py tests/test_experiment_tracker.py tests/test_model_registry.py tests/test_checkpoint.py tests/test_cross_validator.py tests/test_training_pipeline.py -v --cov=bufferiq/ml/training --cov-report=term-missing --cov-fail-under=90

# =========================
# CODE QUALITY
# =========================
lint:
	cd backend && python -m ruff .
	cd backend && python -m mypy bufferiq/ --strict

format:
	cd backend && python -m black .
	cd backend && python -m ruff . --fix

type-check:
	cd backend && python -m mypy bufferiq/ --strict

validate: lint test
	@echo "✅ All validation checks passed!"

# =========================
# RUN
# =========================
run:
	cd backend && uvicorn bufferiq.main:app --reload

# =========================
# DOCKER
# =========================
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-test:
	docker-compose exec backend python -m pytest tests/ -v --cov=bufferiq --cov-report=term-missing

# =========================
# DATABASE
# =========================
migration:
	cd backend && alembic revision --autogenerate -m "$(msg)"

migrate:
	cd backend && alembic upgrade head

db-migrate: migration

db-upgrade:
	cd backend && alembic upgrade head

db-downgrade:
	cd backend && alembic downgrade -1

db-reset:
	cd backend && alembic downgrade base
	cd backend && alembic upgrade head

db-init:
	docker-compose exec backend bash /scripts/init-db.sh

# =========================
# FEATURE ENGINEERING
# =========================
extract-features:
	cd backend && python -m bufferiq.cli.features extract --user-id=1 --stats

list-features:
	cd backend && python -m bufferiq.cli.features list-features

# =========================
# MODEL TRAINING
# =========================
train-baseline:
	cd backend && python -m bufferiq.cli.train run --config ../configs/training/baseline.yaml

train-xgboost:
	cd backend && python -m bufferiq.cli.train run --config ../configs/training/xgboost.yaml

list-experiments:
	cd backend && python -m bufferiq.cli.train list-experiments

list-models:
	cd backend && python -m bufferiq.cli.train list-models

# =========================
# SYNC COMMANDS
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
# CLEANUP
# =========================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -f bufferiq.db
	rm -rf backend/htmlcov
	rm -rf backend/.coverage