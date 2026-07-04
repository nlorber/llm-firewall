# Decisions Log — Distilling the Gray-Zone Judge

Every non-obvious choice in distilling the Claude gray-zone judge into a local SLM and
composing it as a tiered judge, with the rationale and the alternative rejected. Numbers
are from the eval reports (`reports/distill_eval_baselines_*`); run-to-run MLX/Claude
non-determinism means exact figures wobble — the *directions* are what hold.

| Decision | Choice | Rationale | Alternative rejected |
|---|---|---|---|
| Training framework | **Apple MLX (`mlx-lm`)** | Runs natively on the M3 Max; QLoRA fine-tune + in-process serving; €0 and instant iteration. The concepts (LoRA, SFT, distillation, agreement eval) are framework-agnostic. | Unsloth / bitsandbytes / vLLM — all CUDA-only, don't run on Apple Silicon |
| Base models | **Qwen3-4B-Instruct-2507 + Qwen3-1.7B** (non-thinking) | Non-thinking output keeps schema-validity on the single greedy generation and makes the decision token locatable for the escalation signal. | Thinking/hybrid checkpoints — a `<think>` block buries the JSON behind an unbounded prefix |
| Distillation target | **Claude teacher labels, not ground-truth** | The goal is to *replicate the existing Claude judge* locally; GRAY-zone "ground truth" is scarce and ambiguous by definition (that is *why* a judge is needed there). Claude is the best scalable proxy. | Human-labeled ground truth — a different, far costlier project; and if we had it we'd retrain the classifier, not add a judge |
| Corpus balance | **Benign-gray top-up → 71% BLOCK / 29% PASS** | Naive SFT on the natural 88/12 GRAY distribution collapsed the 1.7B to always-BLOCK (0% specificity). Adding benign-but-suspicious prompts (only ~3% land in GRAY) rebalanced training *and* gave the test set enough PASS to measure specificity. | Train on the skewed corpus / downsample BLOCK only (leaves test specificity unmeasurable) |
| Escalation location | **`TieredJudge` composite** (implements `Judge`) | No control-flow refactor — the orchestrator already holds a `Judge`; swappable and unit-testable with fakes. | Branching inside `nodes.py` |
| Escalation signal | **Decision-token logprob uncertainty**, selected on val | Emitted confidence is teacher-mimicry — it predicts *Claude's* confidence, not the student's own uncertainty (**val AUC 0.481**, useless). The decision-token margin/entropy predicts disagreement-with-teacher (**val AUC 0.681**). | Emitted-confidence threshold |
| Threshold τ | **Fit on val (τ=0.5), reported on test** | Unbiased reporting; val→test generalized cleanly (44% → 42% escalation). At τ=0.5 the signal escalates 42% to recover specificity 31%→85%. | Fit τ on the report set (optimistic bias) |
| Empty `<think>` handling | **Strip an empty `<think></think>`, reject only non-empty** | The Qwen3 template emits an empty think block as scaffolding; the 4B fine-tune reproduces it in the output. It's a bounded artifact, not reasoning leakage, so rejecting it would needlessly fail every 4B verdict. | Hard-reject any `<think>` (the spec's original rule; breaks the 4B) |
| Inference determinism | **Greedy / temperature 0** | The "predictability" property; a temp-0 retry on invalid output is a no-op. | Sampled inference |
| Short-circuit (abort local decode on escalation) | **Deferred to backlog** | It only speeds the *escalated* minority; the ft-4B's verbose reasoning makes the *kept* path the real latency cost, which the short-circuit doesn't touch. | Build it now (premature; doesn't fix the dominant cost) |
| Default `judge_backend` | **`claude`** | Backward-compatible; no clone-time dependency on a trained adapter or MLX. `tiered` is the showcase once an adapter exists. | Default to `tiered`/`local` |
| `local`-mode failure | **Fail-closed → BLOCK** | Appropriate for a firewall; never an implicit PASS on model failure. (`claude`/`tiered` escalate or surface a 500.) | Fail to 500/error |

## The honest bottom line

Distilling the judge is worth it **as a tiered system**: near-teacher quality (94% decision-match, 85% specificity, 98% BLOCK recall) at **~40% of Claude's cost** with **58% of verdicts staying on-device**, at a **latency cost** (2.4s vs Claude's 1.7s — the local reasoning isn't short-circuited). Standalone, the **4B distills** (recall 81→98%, specificity held) but the **1.7B is too small and collapses** to always-BLOCK. Agreement with the teacher is *not* a safety guarantee, and the independent ground-truth staleness probe is currently underpowered (only 1 of 20 adversarial attacks reaches the GRAY band the judge sees).
