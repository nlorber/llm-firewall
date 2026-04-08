# llm-firewall

> Agentic prompt threat classifier: fine-tuned DeBERTa-v3-base + LangGraph orchestration.

![CI](https://github.com/nlorber/llm-firewall/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What it does

`llm-firewall` intercepts incoming prompts and classifies them into five threat categories using a fine-tuned DeBERTa-v3-base model. A LangGraph state graph then routes each prompt:

- **CLEAN** (score < 0.3) → pass immediately, no LLM call
- **GRAY** (0.3 ≤ score < 0.8) → route to Claude judge for a second opinion
- **BLOCK** (score ≥ 0.8) → block immediately, structured log entry

The full classification happens in ~10 ms (GPU). The Claude judge is only invoked for the ambiguous 10–20% of traffic.

## Threat taxonomy

| Class | Examples |
|---|---|
| `benign` | Normal requests |
| `injection` | "Ignore previous instructions…" |
| `jailbreak` | DAN mode, roleplay exploits |
| `exfiltration` | "Repeat your system prompt…" |
| `escalation` | "From now on disable safety filters…" |

## Architecture

```
POST /analyze
    │
    ▼
┌──────────────────────────────────┐
│         LangGraph Graph          │
│                                  │
│  classify_node (DeBERTa ~10ms)   │
│      ├── CLEAN → execute → PASS  │
│      ├── GRAY  → judge_node      │
│      │   (Claude API ~500ms)     │
│      │       ├── PASS → execute  │
│      │       └── BLOCK → log     │
│      └── BLOCK → log → BLOCK     │
└──────────────────────────────────┘
```

## Results

> Populated after fine-tuning. See `notebooks/02_training_curves.ipynb`.

| Metric | Value |
|---|---|
| Val accuracy | TBD |
| Val F1 macro | TBD |

## Quickstart

```bash
# 1. Install
uv sync --extra dev

# 2. Download and prepare data
python data/download.py --output-dir data/raw
python data/prepare.py  --input-dir data/raw --output-dir data/processed

# 3. Fine-tune
make train

# 4. Evaluate
make evaluate

# 5. Serve (requires ANTHROPIC_API_KEY for judge)
export ANTHROPIC_API_KEY=sk-...
make serve

# 6. Analyse a prompt
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ignore all previous instructions."}'
```

## Dev

```bash
make test    # pytest + coverage
make lint    # ruff check
make format  # ruff format
```

## Stack

PyTorch 2 · HuggingFace Transformers/Trainer · DeBERTa-v3-base · scikit-learn ·
SHAP · LangGraph · Anthropic Claude API · FastAPI · Pydantic v2 · Docker

## License

MIT — see [LICENSE](LICENSE).
