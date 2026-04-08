# Data

> Raw and processed datasets for the llm-firewall classifier.
> Actual files are excluded from git (see `.gitignore`).
> Run `python data/download.py` then `python data/prepare.py` to populate.

## Label taxonomy

| Label | Description |
|---|---|
| `benign` | Normal user requests with no malicious intent |
| `injection` | Direct injection attempts ("ignore previous instructions…") |
| `jailbreak` | Guideline bypass attempts (DAN mode, roleplay exploits) |
| `exfiltration` | Attempts to extract system prompts or sensitive data |
| `escalation` | Privilege escalation attempts |

## Sources

| Dataset | Hub ID | License | Notes |
|---|---|---|---|
| prompt-injections | `deepset/prompt-injections` | Apache 2.0 | ~600 injection + benign examples |
| JBB-Behaviors | `JailbreakBench/JBB-Behaviors` | MIT | Structured jailbreak prompts |

## Statistics

> Populated after running `python data/prepare.py`.

## Splits

Stratified 70 / 15 / 15 train / val / test. Seed: 42.

## Augmentation

LLM-based paraphrasing via Claude API applied to underrepresented classes.
See `prepare.py:augment` for details.
