.PHONY: install train evaluate serve test lint format clean

## Install all dependencies in editable mode + dev extras, then install pre-commit hooks
install:
	pip install -e ".[dev]"
	pre-commit install

## Fine-tune the DeBERTa classifier (reads configs/training.yaml)
train:
	python -m firewall.classifier.train --config configs/training.yaml

## Evaluate the fine-tuned model on the test split (reads configs/training.yaml)
evaluate:
	python -m firewall.classifier.evaluate --config configs/training.yaml

## Start the FastAPI inference server (available after Phase 5)
serve:
	uvicorn firewall.api.app:app --host 0.0.0.0 --port 8000 --reload

## Run the full test suite with coverage
test:
	pytest

## Run ruff linter across source, tests, and data scripts
lint:
	ruff check src/ tests/ data/

## Apply ruff formatter across source, tests, and data scripts
format:
	ruff format src/ tests/ data/

## Remove build artifacts, caches, and compiled files
clean:
	rm -rf dist/ build/ .eggs/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache  -exec rm -rf {} +
	find . -type d -name .ruff_cache  -exec rm -rf {} +
	find . -name "*.pyc" -delete
