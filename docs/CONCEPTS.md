# Concepts — Distillation, LoRA/QLoRA, and Agreement-Based Evaluation

A didactic companion to the distilled local judge. It explains the techniques behind
`src/firewall/judge/distill/` and `tiered.py` from first principles, and is deliberate
about where the honest limits are.

## Knowledge distillation

**Distillation** trains a small *student* model to reproduce the behavior of a large
*teacher*. Here the teacher is the Claude gray-zone judge; the student is a ~2 GB local
Qwen3. The teacher labels every gray-zone prompt with a `{decision, reasoning, confidence}`
verdict, and the student learns to emit the same JSON for the same input.

The payoff: you bottle a frontier model's judgment into something that runs on a laptop for
~$0. The catch, stated plainly: **the student's ceiling is the teacher, including the
teacher's mistakes.** Our headline metric is therefore *agreement with the teacher*, not
accuracy — a Claude error is invisible to it, and the student faithfully reproduces it. That
is why agreement is paired with a separate, ground-truth **staleness probe** (below).

Why distill Claude rather than train on ground-truth labels? Because the gray zone is
*defined* as the prompts the ground-truth-trained classifier is unsure about — there often
is no crisp answer, and labeling ambiguous cases needs expensive human adjudication. If we
had abundant reliable gray-zone ground truth we wouldn't need a judge at all; we'd retrain
the classifier. Claude is the best *scalable* proxy for that missing ground truth.

## SFT (supervised fine-tuning)

The student is trained by **SFT**: given the (system + user) judge prompt, maximize the
likelihood of the teacher's exact completion (the JSON verdict). Loss is masked to the
completion only (`mask_prompt`), so the model learns the *verdict mapping*, not to parrot
the near-identical prompt. We early-stop on validation loss — the small corpus (~290 train)
overfits within a few hundred iterations.

## LoRA and QLoRA

Full fine-tuning updates all of a model's billions of weights — memory-heavy and overkill
for teaching a narrow output format.

- **LoRA (Low-Rank Adaptation)** freezes the base model and inserts small trainable low-rank
  matrices (rank `r`) into the attention/MLP projections. Only those adapters train — here
  **~0.1–0.5%** of parameters (2.5M of 1.7B; 3.7M of 4B). The result is a tiny (~10 MB)
  adapter you load *on top of* the frozen base.
- **QLoRA** takes it further: the frozen base is **quantized to 4-bit**, so the whole thing
  fits in a few GB of memory, while the LoRA adapter stays full precision. MLX does this
  natively — a 4-bit base plus an FP adapter.

**Honesty note.** MLX realizes the *QLoRA concept* (frozen 4-bit base + full-precision
adapter) using **affine group quantization**, *not* the QLoRA paper's NF4 + double
quantization (which is bitsandbytes/CUDA). Same idea — memory-efficient adaptation of a
quantized base — different quantizer.

`rank` sets the adapter's capacity; `scale` (α) scales its contribution. Capacity turned out
to be the binding constraint here — rank 8 under-fit the benign/malicious boundary and
collapsed to ~35% specificity, so this project needed **rank 32** across all 36 layers.

## Chat templates and the empty `<think>` artifact

Instruction models expect input formatted by a **chat template** — special tokens wrapping
each role (`<|im_start|>system … <|im_end|>`, etc.). The corpus is stored as role-tagged
*messages*, and each model's own template is applied at train and inference time, so one
corpus serves both models.

A subtlety we hit: Qwen3's template emits an **empty** `<think>\n\n</think>` block as
structural scaffolding, even in non-thinking mode. For the 1.7B this sits in both the
training target and the inference prompt (harmless). For the 4B-Instruct the template puts
it in the training *target* but not the generation *prompt*, so the fine-tuned model learns
to emit it — a bounded, empty artifact, not reasoning. The judge therefore **strips a
leading empty think block and rejects only a non-empty one** (real reasoning leakage, which
would bury the JSON and break the decision-token signal).

## Agreement-based evaluation, and its blind spot

Because the reference labels are Claude's, every eval number is *teacher-agreement*:

- **Decision-match** — fraction where the judge's PASS/BLOCK equals Claude's.
- **BLOCK recall** — of Claude's BLOCKs, fraction the judge also BLOCKs (a missed BLOCK is
  the costly firewall error).
- **Benign-PASS (specificity)** — of Claude's PASSes, fraction the judge also PASSes (the
  false-positive rate we care about; the always-BLOCK failure mode shows up here as ~0%).

Small test sets and extreme rates make point estimates noisy, so proportions are reported
with **Wilson 95% confidence intervals** (better than the normal approximation at small
n / extreme p). Same-family generation (Claude judging Claude-written prompts) inflates
agreement, so results are broken down **by provenance**.

The blind spot is structural: **agreement ≠ safety.** A judge that perfectly mimics a
mistaken teacher scores 100% and is still wrong. The independent read is the **staleness
probe** — run the judge on the held-out `data/adversarial/` set, which has *true* attack
labels, and measure ground-truth BLOCK recall. (Ours is currently underpowered: the
classifier confidently blocks 19 of 20 attacks before the judge sees them, leaving N=1 —
an honest limitation, not a pass.)

## Escalation signal (why the model's own uncertainty, not its confidence)

The tiered judge escalates *uncertain* local verdicts to Claude. The obvious signal —
the model's emitted `confidence` field — is useless (**val AUC 0.481**): the student was
trained to mimic *Claude's* confidence, so it carries no information about the student's own
likelihood of being wrong. The signal that works is the **decision-token logprob margin**:
read the probabilities of the `PASS` and `BLOCK` tokens at the JSON decision position and
measure how close they are (**val AUC 0.681**). A model that's genuinely torn between PASS
and BLOCK is the one worth escalating.
