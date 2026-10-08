"""RCWM-T2 supplementary eval set (pre-flight review M4b / S6; DECISIONS D-19). Eval only, never trained on.

Cells the original t2_eval cannot decide (docs/t2_review/):
  P  app-del        applicable, and delete clears a set bit that add does not restore     (m1 frame)
  P  pre_off-only   pre_on satisfied, a pre_off bit set, and applying would change the state
  R  mix-2wrap      mix whose raw sum >= 2*97 (double wrap)                                 (one frame)
Legality-flip minimal pairs (same payload except the rule-relevant field; both answers differ):
  L  pair-posted / pair-void             same event, status posted vs void
  P  pair-applicable / pair-inapp1bit    same before, one extra pre_on bit that is clear in `before`
  R  pair-gt-boundary / pair-le-boundary conditional, threshold = before[other]-1 vs = before[other]
Freshness: payloads excluded against the historical set, every transition of bank_pilot and
bank_confirm (incl. twins), t2_train and t2_eval. Output: data/t2_eval_supp.jsonl (create-only).
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
sys.path.insert(0, str(ROOT / "scripts"))
from make_t2_data import bank_payloads  # noqa: E402
from rcwmt import bank as bankmod, env, runtime  # noqa: E402

N_CELL, N_PAIRS = 40, 40


def _fresh(payload, excl, seen):
    return payload not in excl and payload not in seen


def main():
    out_path = ROOT / "data" / "t2_eval_supp.jsonl"
    if out_path.exists():
        raise SystemExit("supp exists")
    excl = bankmod.load_exclusions()
    for b in ("bank_pilot.json", "bank_confirm.json"):
        excl |= bank_payloads(ROOT / "data" / b)
    for f in ("t2_train.jsonl", "t2_eval.jsonl"):
        for l in (ROOT / "data" / f).read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            excl.add(env.dumps({"before": r["before"], "event": r["event"]}))
    print("excluded payloads:", len(excl), flush=True)
    m = env.load_marginals()
    rng = random.Random(20261008)
    tok = runtime.tokenizer()
    seen, rows = set(), []

    def row(fam, before, ev, cell, **extra):
        p = fam == "prerequisite_plan"
        msgs = env.multi_step_messages(fam, before, [ev]) if p else env.one_step_messages(fam, before, ev)
        env.cross_check(fam, before, ev)
        seen.add(env.dumps({"before": before, "event": ev}))
        return {"prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                "target": env.dumps(env.step(fam, before, ev)), "family": fam, "stratum": cell,
                "frame": "m1" if p else "one", "before": before, "event": ev, **extra}

    def fill(n, draw):
        got, guard = [], 0
        while len(got) < n:
            guard += 1
            assert guard < 2_000_000, f"could not fill: {len(got)}/{n}"
            r = draw()
            if r:
                got += r
        return got

    P, L, R = "prerequisite_plan", "ledger_replay", "register_machine"

    def p_app_del():
        b, ev = env.gen_start(P, rng, m), env.gen_action_library(rng, m, 1)[0]
        if env.applicable(b, ev) and (b & ev["delete"] & ~ev["add"]) and _fresh(env.dumps({"before": b, "event": ev}), excl, seen):
            return [row(P, b, ev, "app-del")]

    def p_pre_off_only():
        b, ev = env.gen_start(P, rng, m), env.gen_action_library(rng, m, 1)[0]
        if ((b & ev["pre_on"]) == ev["pre_on"] and (b & ev["pre_off"]) and ((b & ~ev["delete"]) | ev["add"]) != b
                and _fresh(env.dumps({"before": b, "event": ev}), excl, seen)):
            return [row(P, b, ev, "pre_off-only")]

    def r_mix_2wrap():
        b, ev = env.gen_start(R, rng, m), env.gen_event(R, rng, m, tick=rng.randint(1, 99))
        if ev["kind"] == "mix" and b[ev["register"]] + 2 * b[ev["other"]] + ev["value"] >= 2 * env.MOD \
                and _fresh(env.dumps({"before": b, "event": ev}), excl, seen):
            return [row(R, b, ev, "mix-2wrap")]

    pair_id = [0]

    def pair(fam, b, ev_a, ev_b, cell_a, cell_b):
        pa, pb = env.dumps({"before": b, "event": ev_a}), env.dumps({"before": b, "event": ev_b})
        if pa == pb or not (_fresh(pa, excl, seen) and _fresh(pb, excl, seen)):
            return None
        if env.dumps(env.step(fam, b, ev_a)) == env.dumps(env.step(fam, b, ev_b)):
            return None
        pair_id[0] += 1
        pid = f"{fam[:1].upper()}{pair_id[0]:03d}"
        return [row(fam, b, ev_a, cell_a, pair_id=pid, pair_role="a"), row(fam, b, ev_b, cell_b, pair_id=pid, pair_role="b")]

    def l_pair():
        b, ev = env.gen_start(L, rng, m), env.gen_event(L, rng, m, tick=rng.randint(1, 99))
        return pair(L, b, {**ev, "status": "posted"}, {**ev, "status": "void"}, "pair-posted", "pair-void")

    def p_pair():
        b, ev = env.gen_start(P, rng, m), env.gen_action_library(rng, m, 1)[0]
        if not env.applicable(b, ev) or ((b & ~ev["delete"]) | ev["add"]) == b:
            return None
        free = [i for i in range(env.NBITS) if not (b >> i) & 1 and not (ev["add"] >> i) & 1 and not (ev["pre_off"] >> i) & 1]
        if not free:
            return None
        return pair(P, b, ev, {**ev, "pre_on": ev["pre_on"] | (1 << rng.choice(free))}, "pair-applicable", "pair-inapp1bit")

    def r_pair():
        b, ev = env.gen_start(R, rng, m), env.gen_event(R, rng, m, tick=rng.randint(1, 99))
        o = ev["other"]
        if ev["kind"] != "conditional" or b[o] < 1:
            return None
        return pair(R, b, {**ev, "threshold": b[o] - 1}, {**ev, "threshold": b[o]}, "pair-gt-boundary", "pair-le-boundary")

    for draw, n in ((p_app_del, N_CELL), (p_pre_off_only, N_CELL), (r_mix_2wrap, N_CELL)):
        rows += fill(n, draw)
    for draw in (l_pair, p_pair, r_pair):
        rows += fill(2 * N_PAIRS, draw)
    with out_path.open("x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("supp", len(rows), hashlib.sha256(out_path.read_bytes()).hexdigest()[:16],
          dict(Counter((r["family"][:6], r["stratum"]) for r in rows)), flush=True)


if __name__ == "__main__":
    main()
