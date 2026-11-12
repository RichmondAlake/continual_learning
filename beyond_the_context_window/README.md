# Continual Learning Beyond the Context Window

Companion notebooks for the [O'Reilly live course](https://learning.oreilly.com/live-events/continual-learning-beyond-the-context-window/0642572457105/).

Production agents already record traces: every request, tool call, answer and correction. Each trace is
a possible training example, and because traces hold failures as well as successes, they can also give
preference pairs. This course turns Calderwood's support traces into training data and uses it in two
places: an embedding model that decides what the agent retrieves, and a small model that replaces a
large one for a scoped task.

**Status: planned.** The notebooks below follow the course outline and will be built on
[`../data/calderwood/`](../data/calderwood/) and MemoRizz's fine-tuning module (`memorizz.finetune`:
trainers, frozen test sets, paired significance tests, swap in and swap back).

## Planned notebooks

| # | Notebook | Course segment | What it does |
|---|---|---|---|
| 01 | Traces to datasets | 1, 2 and 3 | Select high-signal traces, then build three datasets from them: SFT rows (question, answer), contrastive pairs (question, the memory that answers it, hard negatives from neighbouring memories) and preference pairs (the assistant's "I can't find that" against the answer a senior handler gave) |
| 02 | Fine-tune an embedding model | 2 | Fine-tune an open embedding model with sentence-transformers and LoRA on the contrastive pairs. Measure retrieval lift on Calderwood's held-out questions and regression on a general retrieval set. Deploy it safely: version, re-index, roll back |
| 03 | A small model against a large one | 3 | Train a small language model with LoRA on the SFT rows, then compare three set-ups on the held-out questions: a large model, a large model with token-space memory, and the fine-tuned small model, on quality, cost per answer and latency |
| 04 | The weight-space lifecycle | 3 | Retrain triggers, promotion gates (a paired test on a frozen set), regression tests (general knowledge, general retrieval) and rollback, when Calderwood's January 2027 changes make the trained weights stale |

## Course segments

| Segment | Time | Content |
|---|---|---|
| 1. When token space is not enough | 45 min | Presentation: context and memory against weights, training data from traces, LoRA and adapters, forgetting |
| 2. Fine-tuning embedding models from memory traces | 35 min | Notebooks 01 and 02 |
| 3. Training small language models to replace larger models | 40 min | Notebooks 03 and 04 |

## Hardware

Notebooks 02 and 03 train models. They are being written to run on an Apple Silicon Mac (PyTorch MPS,
bf16 through sentence-transformers and PEFT) and on a single NVIDIA GPU. Unsloth is NVIDIA-only.

## Open items

- **Rounds 2 and 3 of the data.** `data/calderwood/` holds round 1 (September 2026). Notebook 04 needs
  the January 2027 changes, which the Calderwood report puts in a `kb_rounds.py` beside `kb.py`; that
  file is not in this repository yet.
