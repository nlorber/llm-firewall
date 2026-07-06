.PHONY: setup train evaluate distill-data distill-topup distill-train-1.7b distill-train-4b distill-eval distill-staleness serve test lint typecheck format clean docker-build

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

## Add a benign-gray PASS slice to the corpus and re-split (reads configs/distill.yaml)
distill-topup:
	uv run python -m firewall.judge.distill.data --config configs/distill.yaml --topup

## QLoRA-fine-tune Qwen3-1.7B on the corpus (prepares data/distill/mlx, then mlx_lm.lora)
distill-train-1.7b:
	uv run python -m firewall.judge.distill.train --config configs/distill_train_1.7b.yaml

## QLoRA-fine-tune Qwen3-4B-Instruct-2507 on the corpus
distill-train-4b:
	uv run python -m firewall.judge.distill.train --config configs/distill_train_4b.yaml

## Eval baselines on data/distill/test.jsonl (Claude + base MLX models; writes reports/)
distill-eval:
	uv run python -m firewall.judge.distill.eval --config configs/distill.yaml

## Staleness probe: ground-truth BLOCK recall on the GRAY-band adversarial subset
distill-staleness:
	uv run python -m firewall.judge.distill.staleness --config configs/distill.yaml

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
