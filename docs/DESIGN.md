# Design Document

Technical deep-dive into the llm-firewall architecture. For setup and usage, see [README.md](../README.md).

---

## 1. Why a Hybrid Classifier + LLM Judge?

The core design question is: how do you classify prompt threats at scale without calling an LLM for every request?

### Pure LLM approach

Route every incoming prompt to Claude (or equivalent) for threat assessment.

- **Latency:** ~500ms per prompt (API round trip)
- **Cost:** ~$0.003 per prompt (assuming ~300 input tokens + ~100 output tokens on Sonnet)
- **At 10K prompts/day:** ~$30/day, 5M ms of LLM wait time

### Hybrid approach (this project)

A fine-tuned DeBERTa classifier handles the first pass. The LLM is only invoked for ambiguous cases.

- **Classifier latency:** ~10ms (GPU), ~50ms (CPU)
- **Classifier cost:** ~$0 marginal (model runs locally)
- **Gray zone rate:** 10–20% of traffic (configurable via thresholds)
- **At 10K prompts/day:** 8,000–9,000 prompts resolved by classifier alone; 1,000–2,000 sent to LLM → ~$3–6/day

The hybrid approach cuts cost by **80–90%** and reduces p50 latency from ~500ms to ~10ms, while maintaining LLM-grade accuracy on the ambiguous prompts that matter most.

### Why not fine-tune to avoid the LLM entirely?

Adversarial prompt engineering evolves faster than any fixed classifier can adapt. The LLM judge acts as a safety net for novel attack patterns that fall outside the classifier's training distribution. This is the same pattern used in production content moderation: fast model for clear-cut cases, expensive model for edge cases.

---

## 2. Why DeBERTa-v3-base?

| Model | Parameters | Key advantage | Concern |
|---|---|---|---|
| BERT-base | 110M | Well-understood baseline | Attention is position-unaware |
| RoBERTa-base | 125M | Better pretraining | Same attention limitation |
| **DeBERTa-v3-base** | **86M** | **Disentangled attention + ELECTRA pretraining** | Slightly more complex architecture |
| DeBERTa-v3-large | 304M | Higher capacity | Overkill for 5-class, 1K-example task |

DeBERTa-v3's disentangled attention mechanism decomposes attention into content-to-content and position-to-content components. This matters for adversarial text where attackers deliberately manipulate word order ("instructions previous all ignore") — the model can attend to content meaning independently of positional expectations.

The v3 variant uses ELECTRA-style replaced token detection during pretraining, which is more sample-efficient than masked language modeling. On a dataset of ~1,270 examples, this efficiency translates to better representations with less data.

At 86M parameters, DeBERTa-v3-base is actually smaller than BERT-base (110M), making inference faster while being more accurate on NLU benchmarks.

---

## 3. Three-Zone Routing

Binary classification forces a hard decision on every prompt. For a security system, this creates two failure modes:

1. **False positive:** benign prompt blocked → user frustration
2. **False negative:** malicious prompt passed → security breach

The three-zone design adds a deliberate uncertainty band:

```
score < 0.3  →  CLEAN  →  pass immediately
0.3 ≤ score < 0.8  →  GRAY  →  defer to LLM judge
score ≥ 0.8  →  BLOCK  →  block immediately
```

### Threshold selection

The thresholds (0.3 / 0.8) were chosen as conservative defaults informed by typical score distributions in similar binary/multiclass threat classifiers:

- Below 0.3: the model's threat-class probability is low enough to pass without review.
- Above 0.8: the model's threat-class probability is high enough to block without review.
- Between 0.3 and 0.8: the classifier is uncertain — this is exactly where the LLM judge adds value.

These defaults should be calibrated per-deployment by examining the classifier's PR curve on a held-out validation set.

Both thresholds are configurable in `configs/orchestrator.yaml` and can be tuned per-deployment depending on the cost of false positives vs. false negatives.

---

## 4. Why LangGraph?

The routing logic is simple today — three zones, two possible paths through the graph. A plain `if/elif/else` would work.

LangGraph was chosen for two reasons:

1. **Extensibility.** The graph structure makes it straightforward to add nodes without refactoring control flow:
   - Rate limiting node between classify and execute
   - Logging/metrics node for observability
   - A/B testing between classifier versions
   - Multi-model ensemble (run two classifiers, merge scores)

2. **Portfolio demonstration.** LangGraph is a production-relevant agentic framework. Showing fluency with state graphs, conditional routing, and node composition demonstrates skills that transfer to more complex orchestration tasks.

### State design

`FirewallState` is a `TypedDict` with progressive population — fields start as `None` and are set by the node that produces them. This avoids optional chaining and makes it clear which node is responsible for each field.

---

## 5. Data Pipeline

### Sources

| Source | Class | Count | Method |
|---|---|---|---|
| deepset/prompt-injections | benign, injection | ~546 | HuggingFace hub download |
| JailbreakBench/JBB-Behaviors | jailbreak | ~200 | HuggingFace hub download |
| Claude API synthetic | exfiltration | ~119 | LLM-generated examples |
| Claude API synthetic | escalation | ~106 | LLM-generated examples |

### Label harmonisation

Source datasets use different label schemes (`"0"`, `"1"`, `"jailbreak"`, etc.). `data/prepare.py:harmonise_labels()` maps all source labels to the canonical 5-class taxonomy. Unknown labels raise `ValueError` to catch schema changes in upstream datasets.

### Deduplication

Case-insensitive exact-match deduplication. The first occurrence is kept. This handles overlapping examples between datasets (e.g., a prompt appearing in both deepset and JailbreakBench with different labels).

### Augmentation

LLM-based paraphrasing via Claude API for underrepresented classes (`exfiltration`, `escalation`). This produces semantically equivalent variants rather than simple token-level perturbations. Can be skipped with `--skip-augment` when no API key is available.

### Split strategy

Stratified 70/15/15 train/val/test split using scikit-learn's `StratifiedShuffleSplit` with seed=42 for reproducibility. Stratification ensures each class is proportionally represented in all splits, which matters when the smallest class (escalation) has only ~106 examples.

---

## 6. Training Regime

### Class-weighted loss

The dataset is imbalanced (benign: 343, escalation: 106). Rather than oversampling, which risks memorisation on a small dataset, we use inverse-frequency class weights in the cross-entropy loss:

```python
weights = compute_class_weight("balanced", classes=np.arange(num_labels), y=train_labels)
loss = F.cross_entropy(logits, labels, weight=class_weights)
```

This increases the gradient contribution of underrepresented classes without duplicating data.

### Early stopping

Patience of 3 epochs on validation F1 macro. This metric was chosen over accuracy because accuracy is dominated by the majority class (benign), while F1 macro weights all classes equally — a better signal for a security classifier where detecting rare attacks matters more than getting benign prompts right.

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 2e-5 | Standard for fine-tuning pretrained transformers |
| Batch size | 8 | Small dataset, small batches avoid gradient noise averaging |
| Max epochs | 10 | Upper bound; early stopping typically triggers at epoch 5–7 |
| Warmup ratio | 0.1 | 10% linear warmup prevents early training instability |
| Weight decay | 0.01 | Standard AdamW regularisation |
| Max length | 128 | Prompts are 1–3 sentences; 128 tokens is sufficient |

---

## 7. Explainability

### Why SHAP over other methods

| Method | Advantage | Limitation |
|---|---|---|
| Attention weights | Built into the model | Attention ≠ attribution (well-documented) |
| Integrated Gradients | Theoretically grounded | Requires gradient access, complex to interpret |
| LIME | Model-agnostic | Unstable on short texts |
| **SHAP (Partition)** | **Theoretically grounded, model-agnostic, token-level** | **Slow (100+ forward passes per explanation)** |

SHAP's partition explainer splits text by tokens and measures each token's marginal contribution to the prediction. For a security classifier, this answers the critical question: "which words made the model flag this prompt?" — essential for debugging false positives.

The slowness (~2–5 seconds per explanation with `max_evals=200`) is acceptable because explanations are generated for audit/debugging, not on the inference hot path.

---

## 8. Evaluation Methodology

### Metrics

- **Accuracy** — overall correctness, dominated by majority class
- **F1 macro** — primary metric; averages F1 across all classes equally, ensuring rare attack types are weighted fairly
- **F1 weighted** — secondary metric; weights by class frequency for real-world distribution relevance
- **Confusion matrix** — per-class error analysis; critical for identifying which threat types are confused with each other
- **Classification report** — per-class precision/recall/F1 for granular analysis

### Why F1 macro over weighted

For a security classifier, a model that perfectly detects benign and injection but completely misses escalation (the rarest class) would score well on F1 weighted but poorly on F1 macro. F1 macro treats all classes as equally important, which aligns with the security goal: every attack type must be caught.
