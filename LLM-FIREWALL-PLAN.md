# llm-firewall — Development Plan

## Concept

Agentic LLM firewall: a system that intercepts incoming prompts, classifies threat level via a fine-tuned transformer, and orchestrates the response (pass-through, deep analysis, or block) through a LangGraph state graph.

## Portfolio positioning

| Project | Focus | Key stack |
|---|---|---|
| transaction-classifier | Classical ML, feature eng. | XGBoost, scikit-learn, Optuna |
| hybrid-recsys | RecSys, retrieval, NLP | Embeddings, TF-IDF, Annoy, spaCy |
| mcp-rest-bridge | LLM tooling, security | TypeScript, MCP, Zod |
| **llm-firewall** | **Deep learning, agentic orchestration, AI security** | **PyTorch, HuggingFace, LangGraph, SHAP** |

CV skills covered by this project and absent from the other three:
PyTorch, BERT/Transformers, HuggingFace, SHAP, LangChain/LangGraph.

---

## Data

Public sources for prompt injections / jailbreaks:

- **deepset/prompt-injections** (HuggingFace) — ~600 labeled injection/benign examples, good starting point
- **JailbreakBench/JBB-Behaviors** — structured jailbreak prompts
- **Lakera/prompt-injection-benchmark** — if accessible
- **PromptInject** (Perez & Ribeiro, 2022) — academic dataset
- **Manual augmentation** — generate variants via LLM-based paraphrasing and adversarial templates

Target taxonomy (multi-class):
- `benign` — normal user requests
- `injection` — direct injection ("ignore previous instructions...")
- `jailbreak` — guideline bypass ("DAN mode", roleplay...)
- `exfiltration` — attempts to extract data or system prompts
- `escalation` — privilege escalation attempts

Target volume: 3k-5k examples after augmentation.

---

## Architecture

```
User prompt
    │
    ▼
┌─────────────────────┐
│   FastAPI endpoint   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LangGraph Router   │
│                     │
│  ┌───────────────┐  │
│  │  Classifier   │  │  ← Fine-tuned DeBERTa (PyTorch)
│  │  Node         │  │    Fast inference, ~10ms
│  └───────┬───────┘  │
│          │          │
│    ┌─────┼─────┐    │
│    │     │     │    │
│    ▼     ▼     ▼    │
│  CLEAN  GRAY  BLOCK │
│    │     │     │    │
│    │     ▼     │    │
│    │  ┌──────┐ │    │
│    │  │Judge │ │    │  ← LLM-as-judge (Claude API)
│    │  │Node  │ │    │    Only invoked on ambiguous cases
│    │  └──┬───┘ │    │
│    │   ┌─┴─┐   │    │
│    │   │   │   │    │
│    ▼   ▼   ▼   ▼    │
│  PASS    BLOCK      │
│    │       │        │
│    ▼       ▼        │
│  ┌────┐ ┌──────┐   │
│  │Exec│ │Logger│   │  ← Structured log of blocks
│  │Node│ │Node  │   │    (type, score, explanation)
│  └────┘ └──────┘   │
└─────────────────────┘
```

Key technical decisions:
- **DeBERTa-v3-base** over BERT: stronger classification performance, reasonable size (~86M params). Fallback: DistilBERT if latency is a concern.
- **Classifier as first line**: GPU inference ~10ms, near-zero cost. Filters 80-90% of traffic without an LLM call.
- **LLM-as-judge as second line**: slow and expensive (~0.5-1s, ~$0.01/call), reserved for the 10-20% of ambiguous cases where context matters. Uses **Claude API** (Anthropic).
- **HuggingFace Trainer** for fine-tuning: industry-standard tooling, clean integration with tokenizers, datasets, and model hub. PyTorch mastery shows through custom Dataset, model wrapping, and compute_metrics.
- **LangGraph**: native conditional routing, typed state management, built-in retry/fallback. No over-engineering: the graph stays linear with a single branching point.

---

## Repo structure

```
llm-firewall/
├── README.md
├── pyproject.toml
├── Makefile                        # train, evaluate, serve, test
├── configs/
│   ├── training.yaml               # hyperparams, model choice, data paths
│   ├── serving.yaml                # host, port, thresholds
│   └── orchestrator.yaml           # clean/gray/block thresholds, judge model
│
├── data/
│   ├── download.py                 # automated dataset download from HuggingFace
│   ├── prepare.py                  # cleaning, merge, split, augmentation
│   └── README.md                   # data provenance, licenses, statistics
│
├── src/firewall/
│   ├── __init__.py
│   │
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── model.py                # HuggingFace model loading + classification head
│   │   ├── dataset.py              # PyTorch Dataset + DeBERTa tokenization
│   │   ├── train.py                # HF Trainer config, custom compute_metrics
│   │   ├── evaluate.py             # metrics, confusion matrix, per-class report
│   │   └── explain.py              # SHAP token-level + attention heatmaps
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── state.py                # TypedDict for graph state
│   │   ├── nodes.py                # classify_node, judge_node, execute_node, log_node
│   │   └── graph.py                # LangGraph StateGraph construction
│   │
│   ├── judge/
│   │   ├── __init__.py
│   │   └── judge.py                # structured prompt template + Claude API + JSON parsing
│   │
│   └── api/
│       ├── __init__.py
│       ├── app.py                  # FastAPI: /analyze, /health
│       └── schemas.py              # Pydantic: AnalysisRequest, AnalysisResponse
│
├── notebooks/
│   ├── 01_eda.ipynb                # data exploration, class distribution
│   ├── 02_training_curves.ipynb    # loss, accuracy, F1 per epoch
│   └── 03_explainability.ipynb     # SHAP visualizations, attention maps, examples
│
├── tests/
│   ├── test_classifier.py          # inference, output format, edge cases
│   ├── test_orchestrator.py        # correct routing based on scores
│   ├── test_judge.py               # mocked LLM, response parsing
│   └── test_api.py                 # endpoints, validation, errors
│
├── Dockerfile
├── docker-compose.yml              # api + (optional) GPU model serving
├── .github/workflows/ci.yml
└── LICENSE
```

---

## Development phases

### Phase 1 — Data + EDA (1-2 days)

- [ ] `data/download.py`: automated download from HuggingFace datasets
- [ ] `data/prepare.py`: cleaning, dedup, label harmonization, stratified train/val/test split
- [ ] Augmentation: LLM-based paraphrasing for underrepresented classes
- [ ] `notebooks/01_eda.ipynb`: class distribution, text lengths, lexical overlap between classes
- [ ] `data/README.md`: source documentation, licenses, statistics

Deliverable: dataset ready in `data/processed/`, documented EDA.

### Phase 2 — Fine-tuning the classifier (2-3 days)

- [ ] `classifier/dataset.py`: PyTorch Dataset with DeBERTa tokenization
- [ ] `classifier/model.py`: HuggingFace model wrapper with classification head
- [ ] `classifier/train.py`: HF Trainer with class weights (imbalanced), early stopping, checkpointing
- [ ] `configs/training.yaml`: learning rate, batch size, epochs, warmup schedule
- [ ] `classifier/evaluate.py`: accuracy, macro/weighted F1, per-class precision/recall, confusion matrix
- [ ] `notebooks/02_training_curves.ipynb`: convergence visualization, hyperparam comparison

Deliverable: trained model, documented metrics, saved artifact.

### Phase 3 — Explainability (1 day)

- [ ] `classifier/explain.py`: SHAP (TokenExplainer or Partition) on representative examples
- [ ] Attention weight visualization by layer
- [ ] `notebooks/03_explainability.ipynb`: commented examples per attack category
- [ ] Integrate key figures into README

Deliverable: SHAP and attention visualizations ready for README and interviews.

### Phase 4 — LangGraph orchestration (2-3 days)

- [ ] `orchestrator/state.py`: state schema (prompt, classification, score, judge_result, decision, logs)
- [ ] `orchestrator/nodes.py`: implementation of each node
- [ ] `judge/judge.py`: structured prompt template, Claude API call, JSON response parsing
- [ ] `orchestrator/graph.py`: StateGraph assembly, conditional routing
- [ ] `configs/orchestrator.yaml`: score thresholds, judge model, retry policy
- [ ] Unit tests per node + end-to-end integration test of the full graph

Deliverable: working agentic pipeline, testable in isolation and end-to-end.

### Phase 5 — API + Docker (1-2 days)

- [ ] `api/app.py`: `/analyze` endpoint (input → decision + explanation), `/health`
- [ ] `api/schemas.py`: Pydantic validation, structured response with score, decision, explanation
- [ ] Multi-stage `Dockerfile`, `docker-compose.yml`
- [ ] `Makefile`: `train`, `evaluate`, `serve`, `test`, `docker-build` commands

Deliverable: servable, containerized, tested API.

### Phase 6 — Documentation + polish (1 day)

- [ ] Full README: motivation, architecture, results, quickstart, figures
- [ ] CI badges, license
- [ ] Cleanup: type hints, docstrings, linting
- [ ] Demo screenshot/GIF (optional)

Deliverable: publishable repo.

---

## Total estimate: 8-12 effective working days

Splittable: phases 1-3 (classifier alone) form a standalone viable project in ~5 days. Phases 4-5 (orchestration + API) add the agentic layer in ~4 additional days.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dataset too small / imbalanced | LLM augmentation, class weights, focal loss |
| DeBERTa too heavy for a demo | Fallback to DistilBERT, INT8 quantization |
| Too much time on orchestration vs. classifier | Decoupled phases: classifier alone is already publishable |
| Mediocre results (trivial binary problem) | Multi-class (5 attack types) + qualitative error analysis |
| Overlap with existing projects (rebuff, llm-guard) | Clear positioning: pedagogical focus + component of an agentic system, not a product |
