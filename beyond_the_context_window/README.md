# Continual Learning Beyond the Context Window

Companion notebooks for the [O'Reilly live course](https://learning.oreilly.com/live-events/continual-learning-beyond-the-context-window/0642572457105/).

Production agents already record traces: every request, tool call, answer and correction. Each trace is
a possible training example, and because traces hold failures as well as successes, they can also give
preference pairs. This course turns Calderwood's support traces into training data and uses it in two
places: an embedding model that decides what the agent retrieves, and a small model that replaces a
large one for a scoped task.

**Status: built.** Notebooks 01 to 03 run end to end and are saved with their outputs. Notebook 04 is
saved without outputs: its text and the results below come from its last full run (12 November 2026),
and its training loop has since been changed to use less memory. The notebooks read the
Calderwood data in [`../data/calderwood/`](../data/calderwood/) and the shared world in
[`calderwood_world.py`](calderwood_world.py), and import nothing else from the repository.

## Notebooks

| # | Notebook | Course segment | What it does | Result in the saved run |
|---|---|---|---|---|
| 01 | [Traces to datasets](01_traces_to_datasets.ipynb) | 1, 2 and 3 | Splits 37 sessions into exchanges, keeps the ones a person confirmed, strips claim references and customer names, and writes three datasets to [`datasets/`](datasets/) | 94 of 96 exchanges selected; 250 SFT rows, 240 contrastive pairs with hard negatives, 47 preference pairs; no held-out question in any dataset |
| 02 | [Fine-tune an embedding model](02_finetune_embeddings.ipynb) | 2 | Fine-tunes `BAAI/bge-small-en-v1.5` (full fine-tuning, MultipleNegativesRankingLoss with a hard negative), measures lift on Calderwood and regression on BEIR SciFact, then deploys it: versioned directories, a re-index, a promotion check and a rollback drill. Optional LoRA run on `Qwen/Qwen3-Embedding-0.6B` | Calderwood nDCG@10 0.857 → 0.955, Recall@5 0.946 → 1.000, MRR 0.826 → 0.943; SciFact nDCG@10 0.713 → 0.716; promoted to `v1`. Qwen3-Embedding + LoRA: 0.897 → 0.961 |
| 03 | [A small model against a large one](03_small_model_vs_large.ipynb) | 3 | Trains `Qwen/Qwen3-0.6B` with LoRA on the SFT rows and compares it, on the 61 held-out questions, with `gpt-6.1-sol` alone and with the top three memory units retrieved by notebook 02's model; quality per category, cost per answer, p50/p95 latency; preference pairs and a routing simulation | Out of 61 (56 Calderwood + 5 general): `gpt-6.1-sol` alone 4 (0 of 56 Calderwood), $0.00165 per answer, p50 5.49 s; with three memory units 58 (53 of 56), $0.00077, p50 2.27 s; Qwen3-0.6B untrained 3, fine-tuned 30 (27 of 56), $0 per call, p50 0.56 s. Costs at list prices ($2.00 input, $0.10 cached, $10.00 output per million tokens); API spend $0.15 per run. Prior knowledge 4/5, 5/5, 2/5, 3/5; the trained model answers general TPA and IBNR questions with Calderwood's meanings. Similarity routing lowers quality at every threshold |
| 04 | [The weight-space lifecycle](04_weight_space_lifecycle.ipynb) | 3 | Reads Calderwood's January 2027 changes from `calderwood_world.py`, generates round 2 rows and frozen test sets by rule, fires the retrain triggers, retrains with and without replay (and from the base model when both are rejected), gates the candidates (exact McNemar test plus regression tests), then promotes or keeps the version in service and rehearses rollback | 4 policy changes; `v1` passes 0 of 13 January questions, 3 answers stale. Without replay: 12 of 13 learned, 3 of 52 unchanged facts kept. With replay: 8 of 13, 24 of 52. Fresh adapter on the corrected set: 7 of 13, 30 of 52, but 2 of 5 general questions against `v1`'s 3. All three beat `v1` on the new facts (exact McNemar p ≤ 0.016) and each failed a regression test, so the gate kept `v1` in service; the rollback drill ran on a staging pointer |

## Course segments

| Segment | Time | Content |
|---|---|---|
| 1. When token space is not enough | 45 min | Presentation: context and memory against weights, training data from traces, LoRA and adapters, forgetting |
| 2. Fine-tuning embedding models from memory traces | 35 min | Notebooks 01 and 02 |
| 3. Training small language models to replace larger models | 40 min | Notebooks 03 and 04 |

Each notebook opens with its key takeaways, a table of its Parts and a **live path**: the ⭐ sections to
show on screen, with minutes for each.

## Running them

From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-weight-space.txt
cp .env.example .env            # add OPENAI_API_KEY: only notebook 03 needs it
.venv/bin/jupyter lab beyond_the_context_window/
```

Run the notebooks in order, one at a time. Each one also runs on its own: 02 reads the datasets 01
wrote (they are committed), 03 falls back to the base embedder if 02 hasn't run, and 04 trains `v1`
itself if 03 hasn't run.

What they write, all ignored by git except `datasets/`:

| Path | Written by | What it is |
|---|---|---|
| `datasets/` | 01, 04 | SFT rows, contrastive pairs, preference pairs, a manifest with hashes; round 2 rows and frozen test sets |
| `models/embedder/` | 02 | `v0` (base) and `v1` cards, `v1` weights, one index per version, `ACTIVE`, `history.jsonl` |
| `models/slm/` | 03, 04 | LoRA adapters `v1`, `v2-no-replay`, `v2` and `v3` with cards, `ACTIVE`, `STAGING`, `history.jsonl` |
| `runs/` | 02, 03 | Trainer scratch, and notebook 03's answers (`03_answers.jsonl`) |

The models (`BAAI/bge-small-en-v1.5`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-Embedding-0.6B`) and BEIR SciFact
(pinned revisions, read with pandas) download from the Hugging Face Hub on first use. The diagrams'
Mermaid sources are in [`images/diagrams/`](images/diagrams/); `python scripts/diagrams/render.py
beyond_the_context_window/<notebook>.ipynb` re-renders them into the notebook.

## Hardware

The saved runs are from an Apple Silicon Mac with 16 GB of memory, shared with other work, using PyTorch's
MPS backend:

- **Memory.** Each notebook caps PyTorch's share of the Mac's memory (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5`),
  so a run that needs too much fails with an error instead of pushing the machine into swap.
- **The small model** trains in bf16 with gradient checkpointing, left-padded batches of 8 and logits
  computed only for the answer tokens. At the end of notebook 03's training the MPS driver held 5.3 GB,
  including the embedder and PyTorch's cache; an epoch over 250 rows takes about a minute.
- **Embedders.** `bge-small-en-v1.5` (33M parameters) is fully fine-tuned in fp32 in about 10 seconds;
  the optional LoRA run on `Qwen3-Embedding-0.6B` takes about 100 seconds in bf16 with gradient
  checkpointing. SciFact (5,183 abstracts) takes under a minute per model.
- **Run times** in the saved runs: 01 about 2 s, 02 235 s, 03 954 s (including 122 API calls), and 04
  about 15 minutes in its last full run (three trainings).

The notebooks choose CUDA when an NVIDIA GPU is present, and CPU as a last resort; the saved runs
used MPS only. Unsloth is NVIDIA-only and isn't used.
