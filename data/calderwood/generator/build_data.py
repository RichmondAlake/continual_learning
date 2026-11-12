"""
Builds the Calderwood Insurance dataset in three stages, the same shape as a real pipeline:

  1. simulate   -> traces.jsonl        raw support sessions with tool calls (the agent's experience)
  2. extract    -> memory_units.jsonl  memory pulled out of the traces (memory_write + memory_search events)
  3. format     -> train.jsonl         memory units turned into chat rows for SFT
                -> eval.jsonl          held-out questions with key terms, never seen in 1 or 3

In a production pipeline stage 1 is your real agent logs and stage 3 uses an LLM to write the
question/answer paraphrases. Here the paraphrase bank in kb.py stands in for that LLM step,
so every answer is known to be correct.
"""
import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

from kb import FACTS, MULTI_HOP_TRAIN, MULTI_HOP_EVAL, PRIOR_KNOWLEDGE_EVAL, SYSTEM_PROMPT
from scoring import score, normalise

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
rng = random.Random(7)

REPS = ["Priya Shah", "Tom Okafor", "Ellie Burns", "Marcus Webb", "Hannah Doyle", "Sam Ferreira", "Aisha Karim", "Josh Lund"]
SENIORS = ["Gwen (Senior Handler)", "Dev (Desk Lead)", "Ruth (Senior Handler)", "Callum (Desk Lead)"]
CUSTOMERS = ["Mr D. Hollis", "Mrs P. Achebe", "Brightwater Bakeries Ltd", "Ms J. Kowalski", "Fenmoor Haulage", "Mr R. Iqbal",
             "Linden Dental Group", "Mrs S. O'Neill", "Mr T. Varga", "Oakham Lettings", "Ms C. Duarte", "Penrose Joinery"]
OPENERS = ["Got a call on {t} ({c}).", "Working {t} for {c}.", "Quick one on {t}, {c}.", "Customer on the line, {t}.", "Picking up {t} for {c}."]
MISS_REPLIES = ["I can't find that in the knowledge base. Can anyone on the desk confirm?",
                "Nothing in the KB for this. Can someone senior help?",
                "The knowledge base doesn't cover this. Anyone know?"]
SAVED_REPLIES = ["Thanks {s}. Saved to memory so I can answer this next time.",
                 "Got it, thanks {s}. I've stored that.",
                 "Thanks {s}, that's in memory now."]


# ---------------------------------------------------------------- stage 1: simulate traces
def simulate_traces():
    """Each fact is taught once (KB miss, a senior explains, agent writes memory) and later
    recalled once (agent finds it in memory). Customer names and ticket numbers are episodic
    detail that stays in the traces and never reaches training."""
    by_id = {f["id"]: f for f in FACTS}

    def chunk(ids):
        ids = ids[:]
        rng.shuffle(ids)
        out = []
        while ids:
            n = rng.choice([2, 3]) if len(ids) > 3 else len(ids)
            out.append(ids[:n])
            ids = ids[n:]
        return out

    all_ids = [f["id"] for f in FACTS]
    sessions = [("teach", s) for s in chunk(all_ids)] + [("recall", s) for s in chunk(all_ids)]

    traces, memory_ids, teach_q = [], {}, {}
    clock = datetime(2026, 9, 1, 9, 0)
    mem_counter = 0
    for n, (phase, fact_ids) in enumerate(sessions, start=1):
        clock += timedelta(hours=rng.randint(3, 20))
        t = clock
        rep = rng.choice(REPS)
        senior = rng.choice(SENIORS)
        events = []

        def ev(**kw):
            nonlocal t
            t += timedelta(seconds=rng.randint(20, 240))
            events.append({"ts": t.isoformat(), **kw})

        for fid in fact_ids:
            f = by_id[fid]
            ticket, cust = f"CW-{rng.randint(100000, 999999)}", rng.choice(CUSTOMERS)
            opener = rng.choice(OPENERS).format(t=ticket, c=cust)
            if phase == "teach":
                qi = rng.randrange(len(f["qs"]))
                teach_q[fid] = qi
                q = f["qs"][qi]
                ev(type="message", sender=f"rep:{rep}", content=f"{opener} {q}")
                ev(type="tool_call", tool="kb_search", args={"query": q})
                ev(type="tool_result", tool="kb_search", content={"results": []})
                ev(type="message", sender="agent", content=rng.choice(MISS_REPLIES))
                ev(type="message", sender=f"senior:{senior}", content=f["said"])
                ev(type="tool_call", tool="memory_write",
                   args={"memory_type": f["mtype"], "content": f["fact"], "tags": [f["cat"]]})
                mem_counter += 1
                mid = f"mem_{mem_counter:04d}"
                memory_ids[fid] = mid
                ev(type="tool_result", tool="memory_write", content={"memory_id": mid, "status": "stored"})
                ev(type="message", sender="agent", content=rng.choice(SAVED_REPLIES).format(s=senior.split(" ")[0]))
            else:
                qi = rng.choice([i for i in range(len(f["qs"])) if i != teach_q[fid]])
                q = f["qs"][qi]
                ev(type="message", sender=f"rep:{rep}", content=f"{opener} {q}")
                ev(type="tool_call", tool="memory_search", args={"query": q, "top_k": 3})
                ev(type="tool_result", tool="memory_search",
                   content={"results": [{"memory_id": memory_ids[fid], "content": f["fact"],
                                         "score": round(rng.uniform(0.78, 0.95), 2)}]})
                ev(type="message", sender="agent", content=f["ans"][qi % len(f["ans"])])
                ev(type="message", sender=f"rep:{rep}", content=rng.choice(["Perfect, thanks.", "Great, cheers.", "Thanks, that's sorted it."]))

        traces.append({"trace_id": f"trace_{n:03d}", "started_at": clock.isoformat(),
                       "agent": "calderwood-support-assistant", "channel": "internal-support-chat",
                       "outcome": "resolved", "events": events})
    return traces


# ---------------------------------------------------------------- stage 2: extract memory
def extract_memory(traces):
    """Reads ONLY the traces. memory_write calls create units; memory_search hits reinforce them."""
    units = {}
    pending_write = None
    for tr in traces:
        for e in tr["events"]:
            if e["type"] == "tool_call" and e["tool"] == "memory_write":
                pending_write = e["args"]
            elif e["type"] == "tool_result" and e["tool"] == "memory_write" and pending_write:
                mid = e["content"]["memory_id"]
                units[mid] = {"memory_id": mid, "memory_type": pending_write["memory_type"],
                              "content": pending_write["content"], "tags": pending_write["tags"],
                              "first_seen": e["ts"], "source_trace_ids": [tr["trace_id"]], "recall_count": 0}
                pending_write = None
            elif e["type"] == "tool_result" and e["tool"] == "memory_search":
                for hit in e["content"]["results"]:
                    u = units.get(hit["memory_id"])
                    if u:
                        u["recall_count"] += 1
                        if tr["trace_id"] not in u["source_trace_ids"]:
                            u["source_trace_ids"].append(tr["trace_id"])
    episodic = set()
    for tr in traces:
        for e in tr["events"]:
            if e["type"] == "message":
                episodic.update(re.findall(r"CW-\d{6}", e["content"]))
    return list(units.values()), len(episodic)


# ---------------------------------------------------------------- stage 3: format for SFT
def row(user, assistant, **meta):
    return {"conversations": [{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user},
                              {"role": "assistant", "content": assistant}], **meta}


def build_train(units):
    by_content = {f["fact"]: f for f in FACTS}  # stand-in for the LLM paraphrase step
    rows = []
    for u in units:
        f = by_content[u["content"]]
        u["kb_ref"] = f["id"]
        for i, q in enumerate(f["qs"]):
            rows.append(row(q, f["ans"][i % len(f["ans"])], source="memory_unit",
                            memory_id=u["memory_id"], memory_type=u["memory_type"], category=f["cat"]))
    for q, a in MULTI_HOP_TRAIN:
        rows.append(row(q, a, source="composed", memory_id="", memory_type="procedural", category="multi_hop"))
    rng.shuffle(rows)
    return rows


def build_eval():
    items = []
    for f in FACTS:
        items.append({"id": f"eval_{f['id']}", "category": f["cat"], "question": f["eval_q"], "must_include": f["keys"]})
    for i, m in enumerate(MULTI_HOP_EVAL, 1):
        items.append({"id": f"eval_multi_hop_{i}", "category": "multi_hop", "question": m["q"], "must_include": m["keys"]})
    for i, m in enumerate(PRIOR_KNOWLEDGE_EVAL, 1):
        items.append({"id": f"eval_prior_{i}", "category": "prior_knowledge", "question": m["q"], "must_include": m["keys"]})
    for it in items:
        it["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it["question"]}]
    return items


# ---------------------------------------------------------------- checks
def check(traces, units, train, evals):
    train_qs = {normalise(r["conversations"][1]["content"]) for r in train}
    train_answers = " ".join(r["conversations"][2]["content"] for r in train)
    trace_text = normalise(json.dumps(traces))
    problems = []
    for it in evals:
        q = normalise(it["question"])
        if q in train_qs:
            problems.append(f"{it['id']}: eval question appears in train")
        if q in trace_text:
            problems.append(f"{it['id']}: eval question appears in traces")
        if score(it["question"], it["must_include"]):
            problems.append(f"{it['id']}: question alone passes (answer could just echo it)")
        if it["category"] != "prior_knowledge":
            for group in it["must_include"]:
                if not score(train_answers, [group]):
                    problems.append(f"{it['id']}: key group {group} never taught in train answers")
    for f in FACTS:
        for a in f["ans"]:
            if not score(a, f["keys"]):
                problems.append(f"{f['id']}: answer variant misses eval keys: {a}")
    if len(units) != len(FACTS):
        problems.append(f"extracted {len(units)} units, expected {len(FACTS)}")
    for r in train:
        if "—" in json.dumps(r, ensure_ascii=False):
            problems.append("em dash in training data")
    return problems


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    traces = simulate_traces()
    units, n_episodic = extract_memory(traces)
    train = build_train(units)
    evals = build_eval()
    problems = check(traces, units, train, evals)

    write_jsonl(OUT / "traces.jsonl", traces)
    write_jsonl(OUT / "memory_units.jsonl", units)
    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "eval.jsonl", evals)

    n_events = sum(len(t["events"]) for t in traces)
    print(f"traces: {len(traces)} sessions, {n_events} events")
    print(f"memory units extracted: {len(units)} "
          f"({sum(u['memory_type'] == 'semantic' for u in units)} semantic, "
          f"{sum(u['memory_type'] == 'procedural' for u in units)} procedural); "
          f"{n_episodic} ticket-level episodic details left out")
    print(f"train rows: {len(train)}")
    cats = {}
    for e in evals:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    print(f"eval items: {len(evals)} {cats}")
    longest = max(len(json.dumps(r["conversations"])) for r in train)
    print(f"longest train row: {longest} chars (~{longest // 4} tokens)")
    print("CHECKS:", "all passed" if not problems else "\n  " + "\n  ".join(problems))
