.PHONY: setup train evaluate distill-data distill-eval serve test lint format clean docker-build

## Install all dependencies with uv and set up pre-commit hooks
setup:
	uv sync --extra dev
	uv run pre-commit install

## Fine-tune the DeBERTa classifier (reads configs/training.yaml)
train:
	uv run python -m firewall.classifier.train --config configs/training.yaml

## Evaluate the fine-tuned model on the test split (reads configs/training.yaml)
evaluate:
	uv run python -m firewall.classifier.evaluate --config configs/training.yaml

## Build the distillation corpus (reads configs/distill.yaml; needs ANTHROPIC_API_KEY + classifier)
distill-data:
	uv run python -m firewall.judge.distill.data --config configs/distill.yaml

## Eval baselines on data/distill/test.jsonl (Claude + base MLX models; writes reports/)
distill-eval:
	uv run python -m firewall.judge.distill.eval --config configs/distill.yaml

## Start the FastAPI inference server (reads host/port/log_level from configs/serving.yaml)
serve:
	uv run firewall-serve

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
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage coverage.json htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
