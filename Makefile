.PHONY: help setup install test lint format type-check clean run

help:
	@echo "BufferIQ Development Commands"
	@echo "=============================="
	@echo "setup         - Initial project setup"
	@echo "install       - Install dependencies"
	@echo "test          - Run tests with coverage"
	@echo "lint          - Run all linters"
	@echo "format        - Format code with black"
	@echo "type-check    - Run mypy type checker"
	@echo "clean         - Remove build artifacts"
	@echo "run           - Start development server"

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

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -f bufferiq.db

run:
	cd backend && uvicorn bufferiq.main:app --reload