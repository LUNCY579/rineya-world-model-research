"""Teacher qualification for the RCL interpreter (bundle 06_templates/TEACHER_QUALIFICATION.md).

Checks:
  Q1  posted-rule subset: world.transition(S+{posted}) == RCWM rcwmt.env.step('ledger_replay')
      == RCWM scripts/p3_common.rule_gold, for kinds add/scale/transfer, random statuses incl. near-misses.
  Q2  every record of a battery and of every training regime: world == oracle_indep (independent
      implementation from SPEC_RCL.md only), on parsed input, gold after, ok, listed and target strings.
  Q3  round trip: world.parse_input(record.input) reproduces the record's AST, state, event and orders.
RCWM modules are imported read-only from this worktree's own copy of rcwm/ (sha pinned; identical to
the live RCWM bundle at the time of pinning). Writes no bytecode.
"""
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from . import gen, vocab, world as W  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RCWM = REPO / "rcwm" / "experiments" / "RCWM-T1"
PINS = {
    "rcwmt/env.py": "92abb6787588bfb0a38d2d73618f0aeb9e70b9dc4a2d86f428712cd4466db1e2",
    "scripts/p3_common.py": "737d7984f118c248f8169ad55806a2f5852f5e0b06bf046868a15fa8e764c648",
}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rcwm():
    for rel, want in PINS.items():
        got = _sha(RCWM / rel)
        if got != want:
            raise RuntimeError(f"RCWM pin mismatch for {rel}: {got}")
    if str(RCWM) not in sys.path:
        sys.path.insert(0, str(RCWM))
    from rcwmt import env  # noqa: E402
    spec = importlib.util.spec_from_file_location("rcx_p3_common", RCWM / "scripts" / "p3_common.py")
    p3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p3)
    return env, p3


def q1_posted_subset(n=20000, seed=20261202):
    env, p3 = load_rcwm()
    rng = random.Random(seed)
    words = list(vocab.REAL_TRAIN + vocab.REAL_DEV + vocab.NEARMISS_VOID + vocab.POSTEDLIKE) + ["posted", "void"]
    rule = {"op": "S+", "set": ["posted"]}
    n_apply = 0
    for _ in range(n):
        accts = ["A", "B", "C", "D"]
        st = {a: rng.randint(-99, 999) for a in accts}
        kind = rng.choice(("add", "scale", "transfer"))
        acc, oth = rng.sample(accts, 2)
        ev = {"account": acc, "amount": rng.randint(-20, 99), "kind": kind, "other": oth,
              "status": "posted" if rng.random() < 0.5 else rng.choice(words), "tick": 1}
        ok, mine = W.transition(rule, st, {k: v for k, v in ev.items() if k != "tick"})
        theirs = env.step("ledger_replay", st, ev)
        gold3 = p3.rule_gold(st, ev)
        if not (mine == theirs == gold3):
            raise AssertionError(f"Q1 mismatch: {st} {ev} mine={mine} env={theirs} rule_gold={gold3}")
        n_apply += ok
    return {"n": n, "n_apply": n_apply, "pins": PINS, "result": "PASS"}


def q2_q3_records(records, oracle):
    n = 0
    for r in records:
        ast, tid, st, ev = W.parse_input(r["input"])
        assert ast == r["rule_ast"] and st == r["state"] and list(st) == r["state_order"] and ev == r["event"], r["input"]
        assert list(ev) == list(r["event"]), r["input"]
        o = oracle.parse_input(r["input"])
        assert o["rule_ast"] == r["rule_ast"], (o["rule_ast"], r["rule_ast"])
        assert list(o["state"].items()) == list(r["state"].items())
        assert list(o["event"].items()) == list(r["event"].items())
        g = oracle.gold_targets(r["input"])
        assert g["ok"] == r["gold"]["ok"] and g["listed"] == r["gold"]["listed"], r["input"]
        assert list(g["after"].items()) == list(r["gold"]["after"].items()), r["input"]
        assert g["ans"] == r["targets"]["ans"] and g["trace"] == r["targets"]["trace"], (g, r["targets"])
        for fmt in ("ans", "trace"):
            assert oracle.parse_target(r["targets"][fmt], fmt) is not None
            p = W.parse_output(r["targets"][fmt], fmt, r["state_order"])
            assert p is not None and p["after"] == r["gold"]["after"]
        n += 1
    return n


def run(out_path=None, n_train=3000, n_parents=60):
    from . import oracle_indep as O
    report = {"q1": q1_posted_subset()}
    counts = {}
    for world in sorted(gen.WORLDS):
        gen.use_world(world)
        for regime in gen.REGIMES:
            recs = gen.make_train(regime, n_train, 20261203)
            counts[f"{world}:train:{regime}"] = q2_q3_records(recs, O)
        for split, seed in (("dev", 20261204), ("sealed", 20261205)):
            bat = gen.make_battery(split, n_parents, seed)
            counts[f"{world}:battery:{split}"] = q2_q3_records(bat, O)
    gen.use_world("arith")
    report["q2_q3"] = {"records_checked": counts, "result": "PASS"}
    report["vocab"] = vocab.audit()
    report["oracle_indep_sha256"] = _sha(Path(O.__file__))
    report["world_sha256"] = _sha(Path(W.__file__))
    report["gen_sha256"] = _sha(Path(gen.__file__))
    report["templates_sha256"] = _sha(Path(W.T.__file__))
    report["status"] = "QUALIFIED for: S+/S-/K+/AGE/ALT/BGE atoms, AND/OR of two atoms, kinds add/transfer/scale/set/swap, " \
                       "all template ids in templates.py, 3-6 accounts, letters/letters2/words names, all key orders. " \
                       "NOT qualified for: deeper nesting, other templates, spaces in status words."
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1] if len(sys.argv) > 1 else None), indent=1))
