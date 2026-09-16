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
| jailbreak-classification | `jackhhao/jailbreak-classification` | Apache 2.0 | Jailbreak techniques + roleplay benign rows; deduplicated, unfilled templates dropped, short prompts preferred, sampled to 300 jailbreak / 150 benign |
| Synthetic (exfiltration) | Claude API generated | — | ~116 examples via `download.py` |
| Synthetic (escalation) | Claude API generated | — | ~106 examples via `download.py` |
| Synthetic (benign requests) | Claude API generated | — | ~125 examples via `download.py`; ordinary assistant requests (draft/rewrite/summarise/translate), a shape the public benign data does not cover |

Counts are approximate: the synthetic sources are regenerated per run and the model returns
roughly, not exactly, the number asked for.

`JailbreakBench/JBB-Behaviors` was previously used for the `jailbreak` class. It was dropped:
its rows are harmful *content* requests rather than jailbreak techniques, and both its
`harmful` and `benign` splits were ingested under the `jailbreak` label. Delete any
`data/raw/jailbreak_bench.jsonl` left over from an earlier run — `prepare.py` merges every
`*.jsonl` in the directory.

## Statistics

Counts from the run behind the committed checkpoint. The synthetic sources and the
paraphrase step vary per run, so a rebuild shifts these by a few tens of rows.

| Class | Train | Val | Test |
|---|---|---|---|
| benign | 410 | 132 | 132 |
| jailbreak | 306 | 61 | 56 |
| exfiltration | 301 | 19 | 27 |
| injection | 208 | 41 | 39 |
| escalation | 201 | 21 | 21 |
| **Total** | **1,426** | **274** | **275** |

## Long-document examples

Serving scores an over-length prompt as overlapping windows and keeps the most threatening
one, so a window is usually a lot of benign text containing at most one attack sentence.
Trained only on standalone prompts, the model reads such a window as benign and the attack is
passed as CLEAN — below the gray band, so the judge never sees it either.

`prepare.py:make_context_examples()` therefore adds documents built by concatenating benign
prompts, half of them with an attack spliced in at a random position and labelled with that
attack's class: **+320 train, +80 val, +80 test**. The attack-free half is load-bearing —
without it the model learns "long document = attack". Each split builds its documents from
its own rows, so no text crosses the split boundary, and they are added after capping so the
balancing step cannot discard them.

## Splits

Stratified 70 / 15 / 15 train / val / test. Seed: 42.

## Augmentation

LLM-based paraphrasing via Claude API applied to underrepresented classes of the **training
split only**, after the split, so paraphrases of a training row cannot land in val or test.
See `prepare.py:augment()` for details. Skip with `--skip-augment`.
