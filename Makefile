.PHONY: install train evaluate serve test lint format clean docker-build

## Install all dependencies with uv and set up pre-commit hooks
install:
	uv sync --extra dev
	uv run pre-commit install

## Fine-tune the DeBERTa classifier (reads configs/training.yaml)
train:
	uv run python -m firewall.classifier.train --config configs/training.yaml

## Evaluate the fine-tuned model on the test split (reads configs/training.yaml)
evaluate:
	uv run python -m firewall.classifier.evaluate --config configs/training.yaml

## Start the FastAPI inference server (available after Phase 5)
serve:
	uv run uvicorn firewall.api.app:app --host 0.0.0.0 --port 8000 --reload

## Run the full test suite with coverage
test:
	uv run pytest

## Run ruff linter across source, tests, and data scripts
lint:
	uv run ruff check src/ tests/ data/

## Run mypy strict type checking on source files
typecheck:
	uv run mypy --strict src/

## Apply ruff formatter across source, tests, and data scripts
format:
	uv run ruff format src/ tests/ data/

## Build Docker image
docker-build:
	docker build -t llm-firewall .

## Remove build artifacts, caches, and compiled files
clean:
	rm -rf dist/ build/ .eggs/ *.egg-info/ .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache  -exec rm -rf {} +
	find . -type d -name .ruff_cache  -exec rm -rf {} +
	find . -name "*.pyc" -delete
