# Data

> Raw and processed datasets for the llm-firewall classifier.
> Actual files are excluded from git (see `.gitignore`).
> Run `python data/download.py` then `python data/prepare.py` to populate.

## Label taxonomy

The classes are defined by *what the prompt does to the assistant*, not by how harmful the
subject matter is. A request for dangerous content phrased as a plain question is not an
attack on the assistant and belongs in `benign`; the classes below all try to change how the
assistant behaves.

| Label | Description | Boundary |
|---|---|---|
| `benign` | Normal user requests, including roleplay and persona prompts | Persona framing alone is benign — only a bypass attempt makes it `jailbreak` |
| `injection` | Instructions aimed at overriding the current context ("ignore previous instructions…") | Targets the *instructions*; `jailbreak` targets the *guidelines* |
| `jailbreak` | Talking the model out of its safety guidelines: DAN personas, "developer mode", "your filters are disabled" | The technique, not the requested content |
| `exfiltration` | Attempts to extract the system prompt, internal instructions, or other users' data | Wants *data out*; `escalation` wants *rights in* |
| `escalation` | Claims of authority, privilege, or an internal role to obtain actions the user is not entitled to | Claims a right rather than arguing the guidelines away |

## Sources

| Dataset | Hub ID | License | Notes |
|---|---|---|---|
| prompt-injections | `deepset/prompt-injections` | Apache 2.0 | ~546 injection + benign examples |
| jailbreak-classification | `jackhhao/jailbreak-classification` | Apache 2.0 | Jailbreak techniques + roleplay benign rows; deduplicated, unfilled templates dropped, sampled to ~300 per label |
| Synthetic (exfiltration) | Claude API generated | — | ~119 examples via `download.py` |
| Synthetic (escalation) | Claude API generated | — | ~106 examples via `download.py` |

`JailbreakBench/JBB-Behaviors` was previously used for the `jailbreak` class. It was dropped:
its rows are harmful *content* requests rather than jailbreak techniques, and both its
`harmful` and `benign` splits were ingested under the `jailbreak` label. Delete any
`data/raw/jailbreak_bench.jsonl` left over from an earlier run — `prepare.py` merges every
`*.jsonl` in the directory.

## Statistics

The counts below describe the corpus behind the **currently committed classifier
checkpoint**, which predates the source change above. Re-running `download.py` and
`prepare.py` produces a different mix, so these numbers are regenerated with the model.

| Class | Count |
|---|---|
| benign | 343 |
| injection | 318 |
| exfiltration | 303 |
| jailbreak | 200 |
| escalation | 106 |
| **Total** | **1,270** |

## Splits

Stratified 70 / 15 / 15 train / val / test. Seed: 42.

## Augmentation

LLM-based paraphrasing via Claude API applied to underrepresented classes of the **training
split only**, after the split, so paraphrases of a training row cannot land in val or test.
See `prepare.py:augment()` for details. Skip with `--skip-augment`.
