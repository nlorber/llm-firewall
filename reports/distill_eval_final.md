# Distillation eval — canonical run (2026-07-05)

Test split N=90 (64 BLOCK / 26 PASS). Teacher/judge: claude-haiku-4-5-20251001; students: Qwen3-1.7B / Qwen3-4B-Instruct-2507 (QLoRA, MLX); tiered = fine-tuned 4B with escalation threshold τ=0.5. Brackets are 95% Wilson CIs. Raw data: [distill_eval_final.json](distill_eval_final.json). Regenerate with `make distill-eval` (see README — exact figures wobble run-to-run on borderline cases; directions hold).

| Judge | N | Decision-match | BLOCK recall | Benign-PASS | Schema-valid | Latency p50/p95 (s) | Cost/call |
|---|---|---|---|---|---|---|---|
| claude (claude-haiku-4-5-20251001) | 90 | 95.6% [89–98] (n=90) | 93.8% [85–98] (n=64) | 100.0% [87–100] (n=26) | 100.0% | 1.989 / 5.342 | $0.00086 |
| base mlx-community/Qwen3-1.7B-4bit | 90 | 70.0% [60–78] (n=90) | 96.9% [89–99] (n=64) | 3.8% [1–19] (n=26) | 97.8% | 0.609 / 0.838 | $0.00000 |
| base mlx-community/Qwen3-4B-Instruct-2507-4bit | 90 | 75.6% [66–83] (n=90) | 82.8% [72–90] (n=64) | 57.7% [39–74] (n=26) | 100.0% | 0.976 / 1.226 | $0.00000 |
| finetuned Qwen3-1.7B | 90 | 70.0% [60–78] (n=90) | 98.4% [92–100] (n=64) | 0.0% [0–13] (n=26) | 98.9% | 0.807 / 1.005 | $0.00000 |
| finetuned Qwen3-4B-Instruct-2507 | 90 | 90.0% [82–95] (n=90) | 92.2% [83–97] (n=64) | 84.6% [66–94] (n=26) | 100.0% | 3.080 / 4.108 | $0.00000 |
| tiered (4B, τ=0.5) | 90 | 94.4% [88–98] (n=90) | 96.9% [89–99] (n=64) | 88.5% [71–96] (n=26) | 100.0% | 2.819 / 4.970 | $0.00000 |

### Tiered system (escalation accounting)

**tiered (4B, τ=0.5)**
- escalation rate: 18.9% (17/90) by reason: {'uncertainty': 17}
- blended cost/call: $0.00016
- latency p50/p95: kept 2.676/3.267s · escalated 4.667/5.998s

### Agreement by provenance (same-family generation inflates agreement)

**claude (claude-haiku-4-5-20251001)**
| Provenance | N | Decision-match | BLOCK recall |
|---|---|---|---|
| benign_gray | 24 | 91.7% [74–98] (n=24) | 60.0% [23–88] (n=5) |
| coercion | 24 | 91.7% [74–98] (n=24) | 91.3% [73–98] (n=23) |
| gen | 42 | 100.0% [92–100] (n=42) | 100.0% [90–100] (n=36) |

**base mlx-community/Qwen3-1.7B-4bit**
| Provenance | N | Decision-match | BLOCK recall |
|---|---|---|---|
| benign_gray | 24 | 25.0% [12–45] (n=24) | 100.0% [57–100] (n=5) |
| coercion | 24 | 95.8% [80–99] (n=24) | 100.0% [86–100] (n=23) |
| gen | 42 | 81.0% [67–90] (n=42) | 94.4% [82–98] (n=36) |

**base mlx-community/Qwen3-4B-Instruct-2507-4bit**
| Provenance | N | Decision-match | BLOCK recall |
|---|---|---|---|
| benign_gray | 24 | 54.2% [35–72] (n=24) | 20.0% [4–62] (n=5) |
| coercion | 24 | 91.7% [74–98] (n=24) | 91.3% [73–98] (n=23) |
| gen | 42 | 78.6% [64–88] (n=42) | 86.1% [71–94] (n=36) |

**finetuned Qwen3-1.7B**
| Provenance | N | Decision-match | BLOCK recall |
|---|---|---|---|
| benign_gray | 24 | 20.8% [9–40] (n=24) | 100.0% [57–100] (n=5) |
| coercion | 24 | 91.7% [74–98] (n=24) | 95.7% [79–99] (n=23) |
| gen | 42 | 85.7% [72–93] (n=42) | 100.0% [90–100] (n=36) |

**finetuned Qwen3-4B-Instruct-2507**
| Provenance | N | Decision-match | BLOCK recall |
|---|---|---|---|
| benign_gray | 24 | 91.7% [74–98] (n=24) | 60.0% [23–88] (n=5) |
| coercion | 24 | 95.8% [80–99] (n=24) | 100.0% [86–100] (n=23) |
| gen | 42 | 85.7% [72–93] (n=42) | 91.7% [78–97] (n=36) |

**tiered (4B, τ=0.5)**
| Provenance | N | Decision-match | BLOCK recall |
|---|---|---|---|
| benign_gray | 24 | 95.8% [80–99] (n=24) | 80.0% [38–96] (n=5) |
| coercion | 24 | 95.8% [80–99] (n=24) | 100.0% [86–100] (n=23) |
| gen | 42 | 92.9% [81–98] (n=42) | 97.2% [86–100] (n=36) |
