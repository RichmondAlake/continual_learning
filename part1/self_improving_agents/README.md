# Build Self-Improving Agents with Memory, Traces, and Skills

Companion notebooks for the [O'Reilly live course](https://learning.oreilly.com/live-events/build-self-improving-agents-with-memory-traces-and-skills/0642572457082/).

Five notebooks on how an agent gets better at its job without anyone touching the model's weights.
What changes is the *tokens* the harness puts in front of the model (skills, persona, facts, tool
descriptions), so every lesson is something you can read, review, measure and undo. The agent is Ada,
Calderwood Insurance's claims support assistant (see the [repository README](../../README.md)), built with
[MemoRizz](https://github.com/RichmondAlake/memorizz) as the intelligence plane: the layer that decides
what an agent knows, remembers and learns.

## The notebooks

| # | Notebook | Calderwood scene | What evolves | From what evidence |
|---|---|---|---|---|
| 01 | [Workflows to skills](01_workflows_to_skills.ipynb) | Paying a new payee: look up the claim, check the handler's payout authority, screen the payee, then release or stop | A procedure that keeps succeeding becomes a `SKILL.md`; it is demoted when the January rule change breaks it | Workflow memory |
| 02 | [Persona evolution](02_persona_evolution.ipynb) | Handlers keep asking for plain-English customer updates; a Desk Lead keeps correcting drafts that mention a fraud referral | One evidenced goals sentence; the rule goes to a person as a guardrail | The episodic log |
| 03 | [Entity evolution](03_entity_evolution.ipynb) | Grade 1 payout authority: a rumour, then the real January change; Drop retired; the CFU filed under two names | Values superseded with their history kept; duplicates merged | Conversations, under a source rule |
| 04 | [Toolbox augmentation and evolution](04_toolbox_evolution.ipynb) | Handlers say "Amber Pack it" and "raise an IBNR"; Ada passes claim numbers without `CW-` | Each tool's description: enriched at registration, then revised from real usage | The tool log |
| 05 | [The refine cycle](05_refine_cycle.ipynb) | One January 2027 day, tidied overnight | All of the above, as one ordered offline pass with one report | Transcripts and existing memory |

Each notebook stands on its own. Read them in order, or start with 05 for the whole picture.

## Where they fit in the course

| Segment | Time | Notebooks |
|---|---|---|
| 1. Foundations of continual learning for agents | 35 min | Presentation: token, representation and weight space, and which lever fits which problem |
| 2. High-signal traces and the workflows-to-skills pipeline | 45 min | **01**: capture traces from live runs, run a promotion cycle, activate a distilled skill and its paired guardrail |
| 3. Continual learning across the memory system | 40 min | **02** and **03**: persona and entity update loops, a staleness conflict; **05**: the full loop. **04** is the take-home on tools |

In notebook 01, the sections marked ⭐ are the ones shown live.

## Run them

```bash
source ../../.venv/bin/activate
export MEMORIZZ_SRC=~/memorizz/src     # until the MemoRizz release after 0.17.24 is on PyPI
jupyter lab
```

The key comes from the environment or the `.env` file at the repository root. Each notebook writes its
memory to `calderwood_store/` beside it, so you can open its runs in the MemoRizz UI; a cell near the
end of each notebook prints the command.

These copies come from the MemoRizz repository's `examples/continual_learning`. To refresh them after
a MemoRizz change, run `python scripts/sync_token_space.py <path to memorizz>` from the repository root.
