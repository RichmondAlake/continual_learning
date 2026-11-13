# Continual Learning for AI Agents

Companion notebooks, data and code for two O'Reilly live courses by Richmond Alake:

| Course | Space | Folder |
|---|---|---|
| [Build Self-Improving Agents with Memory, Traces, and Skills](https://learning.oreilly.com/live-events/build-self-improving-agents-with-memory-traces-and-skills/0642572457082/) | token and representation space | [`self_improving_agents/`](self_improving_agents/) |
| [Continual Learning Beyond the Context Window](https://learning.oreilly.com/live-events/continual-learning-beyond-the-context-window/0642572457105/) | weight space | [`beyond_the_context_window/`](beyond_the_context_window/) |

A deployed model is frozen the moment its training ends. An agent built on it can still get better
at its job, and there are three places the improvement can live:

| Space | What changes | Examples here | Undo |
|---|---|---|---|
| **Token space** | The tokens the harness puts in front of the model | skills, persona, entity facts, tool descriptions, the refine cycle | edit or retire a record |
| **Representation space** | The models that decide what reaches the context | an embedding model, a reranker, a router trained on the agent's own traces | swap the previous model back |
| **Weight space** | The model that writes the answer | a small model trained on traces to replace a large one for a scoped task | roll back the checkpoint |

Each step down that table costs more and is harder to inspect, so the courses teach when to take it:
token space first, and the others when the traces show token space is not enough.

## One world: Calderwood Insurance

Every notebook runs on the same fictional company, so one fact, one handler and one system recur from
the first skill to the last fine-tune. **Calderwood Insurance** is a UK insurer, and **Ada** is its
internal claims support assistant. Handlers ask her about status codes, error codes and payment limits,
and she has tools into the claims systems. Everything about Calderwood is invented, so no model knows
any of it before it is told.

| | |
|---|---|
| **People** | Priya Shah (Grade 2 handler), Tom Okafor (Grade 1), Senior Handlers Gwen and Ruth, Desk Leads Dev and Callum |
| **Systems** | Beacon (claims, `CW-` references, `Q` status codes), Atlas (policies), Tally (payments), the Ringfence (sanctions screening), Drop then Compass (customer uploads) |
| **Teams** | Counter Fraud Unit (CFU), Financial Crime, Large Loss Unit (LLU), Settlement Review Team (SRT) |

Its knowledge changes in rounds, which is what the agent has to keep up with:

| Round | When | What changes |
|---|---|---|
| 1 | September 2026 | The baseline: Drop in use, Grade 1 payout authority £2,500, an Amber Pack due within 2 working days |
| 2 | January 2027 | Compass replaces Drop; Grade 1 authority rises to £3,000; the Amber Pack deadline becomes 3 working days; payments over £25,000 need an SRT clearance |
| 3 | April 2027 | The motor total loss deadline falls from 10 to 7 working days; PCL customers must get a response within 1 working hour instead of 2 |

The people, systems, codes and limits live in one module,
[`self_improving_agents/calderwood_world.py`](self_improving_agents/calderwood_world.py), and the
support traces and training data in [`data/calderwood/`](data/calderwood/).

## What's here

```
self_improving_agents/       Course 1: five notebooks on continual learning in token space (MemoRizz)
beyond_the_context_window/   Course 2: four notebooks on continual learning in weight space (PyTorch, PEFT)
part1/                       Course 2: two guided Colab notebooks on Qwen3.5-27B (Unsloth, A100), with their data
data/calderwood/             Support traces, memory units, SFT rows and held-out questions, with their generator
scripts/                     sync_token_space.py: refresh course 1's notebooks from the MemoRizz repository;
                             diagrams/render.py: render course 2's Mermaid diagrams into its notebooks
slides/                      Agent Memory Engineering: Towards Continual Learning in Token Space (deck)
```

## Setup

Python 3.11 or later.

```bash
git clone https://github.com/RichmondAlake/continual_learning.git
cd continual_learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                    # course 1
pip install -r requirements-weight-space.txt       # course 2 (PyTorch, Hugging Face, PEFT)
cp .env.example .env                               # then add your OpenAI key
```

With [uv](https://docs.astral.sh/uv/), course 2's environment is
`uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-weight-space.txt`.
Course 2 was tested on Python 3.12. Its models (`BAAI/bge-small-en-v1.5`, `Qwen/Qwen3-0.6B`,
optionally `Qwen/Qwen3-Embedding-0.6B`) and the BEIR SciFact benchmark download from the Hugging Face
Hub on first use, and only its notebook 03 needs the OpenAI key.

Course 1 uses features from the **MemoRizz release after 0.17.24** (the refine cycle, tool learning,
dated entity history). Until that release is on PyPI, point the notebooks at a MemoRizz checkout:

```bash
git clone https://github.com/RichmondAlake/memorizz.git ~/memorizz
export MEMORIZZ_SRC=~/memorizz/src
```

The notebooks call OpenAI's cheapest current model (`gpt-6-luna`) and `text-embedding-3-small`, and
store memory in local files, so no database is needed to follow along. MemoRizz also runs on Oracle AI
Database; see the [MemoRizz docs](https://github.com/RichmondAlake/memorizz).

## Status

| Part | State |
|---|---|
| Course 1 notebooks 01–05 | Written and run with saved outputs, in the MemoRizz repository (`examples/continual_learning`); copied here by `scripts/sync_token_space.py` |
| Calderwood data | Traces, memory units, 250 SFT rows and 61 held-out questions, with the generator |
| Course 2 notebooks 01–04 | Built on an Apple Silicon Mac (16 GB); 01–03 saved with their outputs, 04 without; see [`beyond_the_context_window/`](beyond_the_context_window/) |
| Part 1: Qwen3.5-27B notebooks | Two guided Colab notebooks (traces to weights; three rounds with replay) and the round 1 data; notebook 02's round 1–3 data (`calderwood_data_v2.zip`) is still to add; see [`part1/`](part1/) |
