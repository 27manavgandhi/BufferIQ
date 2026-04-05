.PHONY: help setup install test lint format type-check clean run docker-build docker-up docker-down docker-test docker-logs db-migrate db-upgrade db-downgrade db-init validate

help:
	@echo "BufferIQ Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup          - Initial project setup"
	@echo "  install        - Install dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  test           - Run tests with coverage"
	@echo "  lint           - Run all linters"
	@echo "  format         - Format code with black"
	@echo "  type-check     - Run mypy type checker"
	@echo "  validate       - Run all checks (lint + test)"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build   - Build Docker images"
	@echo "  docker-up      - Start all services"
	@echo "  docker-down    - Stop all services"
	@echo "  docker-test    - Run tests in Docker"
	@echo "  docker-logs    - View service logs"
	@echo ""
	@echo "Database:"
	@echo "  db-migrate     - Create new migration"
	@echo "  db-upgrade     - Apply migrations"
	@echo "  db-downgrade   - Rollback last migration"
	@echo "  db-init        - Initialize database"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          - Remove build artifacts"
	@echo "  run            - Start development server"

setup:
	python -m venv venv
	.\venv\Scripts\Activate.ps1 && pip install --upgrade pip
	.\venv\Scripts\Activate.ps1 && pip install -r backend/requirements.txt
	.\venv\Scripts\Activate.ps1 && pre-commit install
	@echo "Setup complete! Activate venv: .\venv\Scripts\Activate.ps1"

install:
	pip install -r backend/requirements.txt

test:
	cd backend && pytest tests/ -v --cov=bufferiq --cov-report=term-missing --cov-report=html

lint:
	black backend/ --check
	ruff backend/
	mypy backend/bufferiq/ --strict

format:
	black backend/
	ruff backend/ --fix

type-check:
	mypy backend/bufferiq/ --strict

validate: lint test
	@echo "✅ All validation checks passed!"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -f bufferiq.db

run:
	cd backend && uvicorn bufferiq.main:app --reload

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	docker-compose ps

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-test:
	docker-compose exec backend pytest tests/ -v --cov=bufferiq --cov-report=term-missing

db-migrate:
	cd backend && alembic revision --autogenerate -m "$(message)"

db-upgrade:
	cd backend && alembic upgrade head

db-downgrade:
	cd backend && alembic downgrade -1

db-init:
	docker-compose exec backend bash /scripts/init-db.sh