"""T2 review (data-validity lens): two CPU-only artifacts for reporting, no project file touched.

  CUDA_VISIBLE_DEVICES=-1 python docs/t2_review/make_xframe_and_substrata.py

t2_eval_xframe.jsonl  the 600 t2_eval rows re-rendered in the OTHER chain frame (P -> one-step
                      verbatim frame, L/R -> multi-step rule with one event), same before/event/target;
                      measures frame transfer, which is what chain_one-P and chain_m1-L/R gains are.
t2_eval_substrata.jsonl  per t2_eval row (same order): a sub-stratum label that separates rows on which
                      the targeted rule is actually tested from rows a copy / rule-ignoring answer passes.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import env, runtime  # noqa: E402

OUT = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (ROOT / "data" / "t2_eval.jsonl").read_text(encoding="utf-8").splitlines()]
tok = runtime.tokenizer()


def sub(r):
    fam, b, e, s = r["family"], r["before"], r["event"], r["stratum"]
    gold = json.loads(r["target"])
    if fam == "prerequisite_plan":
        applied = (b & ~e["delete"]) | e["add"]
        if s == "applicable":
            return "app-changed" if gold != b else "app-noop(copy passes)"
        return "inapp-disc" if applied != b else "inapp-nondisc(apply passes)"
    if fam == "ledger_replay":
        if s.endswith("/void"):
            return "void-disc" if env.step(fam, b, dict(e, status="posted")) != b else "void-nondisc(apply passes)"
        return s + ("(copy passes)" if gold == b else "")
    if e["kind"] == "conditional":
        return s + ("/v0(branch-free)" if e["value"] % 97 == 0 else "")
    if e["kind"] == "mix":
        return f"mix/wraps={(b[e['register']] + 2 * b[e['other']] + e['value']) // 97}"
    return s


with (OUT / "t2_eval_substrata.jsonl").open("x", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        f.write(json.dumps({"idx": i, "family": r["family"], "stratum": r["stratum"], "sub": sub(r)}) + "\n")
with (OUT / "t2_eval_xframe.jsonl").open("x", encoding="utf-8") as f:
    for r in rows:
        fam = r["family"]
        if r["frame"] == "m1":
            msgs, fr = env.one_step_messages(fam, r["before"], r["event"]), "one"
        else:
            msgs, fr = env.multi_step_messages(fam, r["before"], [r["event"]]), "m1"
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        f.write(json.dumps(dict(r, prompt=p, frame=fr)) + "\n")
from collections import Counter  # noqa: E402
print(sorted(Counter((r["family"][:1].upper(), sub(r)) for r in rows).items()))
