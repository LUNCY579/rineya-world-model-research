"""RCWM-T2 data: stratified, oracle-labelled one-step rows targeting Titan's weak operations (R8).

Frames (the ones the chains use):
  L, R  Titan's verbatim one-step frame (env.one_step_messages)            -> chain_one
  P     multi-step rule with a single event (explicit inapplicable=no-op)   -> chain_m1
Strata (train / eval per stratum below):
  L  {add, scale, transfer} x {posted, void}            void ~25%
  R  swap, add (wrap / no-wrap), mix (wrap / no-wrap), conditional (gt / le)
  P  applicable / inapplicable (50/50)
Freshness: every (before, event) payload is checked against the 6,464 historical transition
payloads and against every transition of bank_pilot and bank_confirm (incl. order twins); train
and eval rows are disjoint. Output: data/t2_train.jsonl, data/t2_eval.jsonl (create-only).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcwmt import bank as bankmod, env, runtime  # noqa: E402

TRAIN = {"L": {f"{k}/{s}": n for k in ("add", "scale", "transfer") for s, n in (("posted", 125), ("void", 42))},
         "R": {"swap": 50, "add/wrap": 60, "add/nowrap": 65, "mix/wrap": 90, "mix/nowrap": 85, "conditional/gt": 75, "conditional/le": 75},
         "P": {"applicable": 250, "inapplicable": 250}}
EVAL_PER_STRATUM = 40
FAM = {"L": "ledger_replay", "R": "register_machine", "P": "prerequisite_plan"}


def stratum(code, before, ev):
    if code == "L":
        return f"{ev['kind']}/{'posted' if ev['status'] == 'posted' else 'void'}"
    if code == "R":
        k = ev["kind"]
        if k == "swap":
            return "swap"
        if k == "conditional":
            return "conditional/" + ("gt" if before[ev["other"]] > ev["threshold"] else "le")
        raw = before[ev["register"]] + (ev["value"] if k == "add" else 2 * before[ev["other"]] + ev["value"])
        return f"{k}/" + ("wrap" if not 0 <= raw < env.MOD else "nowrap")
    return "applicable" if env.applicable(before, ev) else "inapplicable"


def bank_payloads(path):
    b = json.loads(path.read_text(encoding="utf-8"))
    out = set()
    for p in b["parents"]:
        cands = p["candidates"] + ([p["twin"]] if "twin" in p else [])
        for c in cands:
            st = env.rollout(p["family"], p["s0"], c["events"])
            for t, ev in enumerate(c["events"]):
                out.add(env.dumps({"before": st[t], "event": ev}))
    return out


def main():
    tr_path, ev_path = ROOT / "data" / "t2_train.jsonl", ROOT / "data" / "t2_eval.jsonl"
    if tr_path.exists() or ev_path.exists():
        raise SystemExit("t2 data exists")
    excl = bankmod.load_exclusions()
    for b in ("bank_pilot.json", "bank_confirm.json"):
        excl |= bank_payloads(ROOT / "data" / b)
    print("excluded payloads:", len(excl), flush=True)
    m = env.load_marginals()
    rng = random.Random(20261007)
    tok = runtime.tokenizer()
    seen = set()
    rows = {"train": [], "eval": []}
    for code, quota in TRAIN.items():
        fam = FAM[code]
        need = {s: {"train": n, "eval": EVAL_PER_STRATUM} for s, n in quota.items()}
        guard = 0
        while any(v["train"] or v["eval"] for v in need.values()):
            guard += 1
            assert guard < 2_000_000, f"could not fill {code}: {need}"
            before = env.gen_start(fam, rng, m)
            ev = env.gen_action_library(rng, m, 1)[0] if code == "P" else env.gen_event(fam, rng, m, tick=rng.randint(1, 99))
            s = stratum(code, before, ev)
            if s not in need:
                continue
            split = "eval" if need[s]["eval"] and (not need[s]["train"] or rng.random() < 0.25) else ("train" if need[s]["train"] else None)
            if split is None:
                continue
            payload = env.dumps({"before": before, "event": ev})
            if payload in excl or payload in seen:
                continue
            seen.add(payload)
            after = env.step(fam, before, ev)
            env.cross_check(fam, before, ev)
            msgs = env.multi_step_messages(fam, before, [ev]) if code == "P" else env.one_step_messages(fam, before, ev)
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            rows[split].append({"prompt": prompt, "target": env.dumps(after), "family": fam, "stratum": s,
                                "frame": "m1" if code == "P" else "one", "before": before, "event": ev})
            need[s][split] -= 1
    for split, path in (("train", tr_path), ("eval", ev_path)):
        random.Random(7).shuffle(rows[split])
        with path.open("x", encoding="utf-8") as f:
            for r in rows[split]:
                f.write(json.dumps(r) + "\n")
        print(split, len(rows[split]), hashlib.sha256(path.read_bytes()).hexdigest()[:16],
              dict(Counter((r["family"][:6], r["stratum"]) for r in rows[split])))


if __name__ == "__main__":
    main()
