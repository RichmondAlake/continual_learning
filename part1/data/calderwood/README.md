# Calderwood support data

Simulated support sessions between Calderwood's handlers and Ada, and the training data built from them.
Everything is invented: the company, its people, systems, codes and policies.

The pipeline has the same three stages as a real one:

| Stage | File | Rows | What it is |
|---|---|---|---|
| 1. Simulate | `traces.jsonl` | 37 | Support sessions, 1–16 September 2026, each a list of events: messages, tool calls, memory searches and writes. When Ada can't answer, a senior handler does, and Ada stores the answer |
| 2. Extract | `memory_units.jsonl` | 48 | The memory those sessions wrote, each with its type, the traces it came from and how often it was recalled |
| 3. Format | `train.jsonl` | 250 | Chat rows for supervised fine-tuning, written from the memory units |
| | `eval.jsonl` | 61 | Held-out questions, never seen in stages 1 or 3, each with the terms a right answer must include |

In a production pipeline, stage 1 is your agent's real logs and stage 3 uses a model to write the
question and answer paraphrases. Here the paraphrase bank in `generator/kb.py` stands in for that step,
so every answer is known to be right.

The held-out questions cover eight categories: policy, abbreviation, multi-hop, system, collision
(Calderwood terms that clash with common meanings), error code, status code and prior knowledge
(general questions, to check that training hasn't made the model forget what it knew).

## Score an answer

```python
from generator.scoring import score
score(answer, row["must_include"])   # True when every required term (or an accepted variant) appears
```

## Rebuild

```bash
cd generator && python build_data.py      # writes to generator/data/
```

The generator is seeded, so a rebuild gives the same files.
