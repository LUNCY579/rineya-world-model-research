"""Golden-value canary for the B4 hook (RCX co-sign edit 8; COSIGN_REVIEW_B4_SCREEN.md).

    python -B experiments/RCX-B4/make_canary.py      (1 CPU thread; writes experiments/RCX-B4/canary/B4_HOOK_CANARY.json)

Probes come from RCX *test* instances only (never seeds_fresh): 10 from RCX-E07 test (n <= 12) and 10 from RCX-E07b
TEST-B with n in {14, 15} (so the 13-15-block sizes of seeds_fresh are exercised). 20 consistent states are reached by
seeded random legal walks from each instance's init (BWM blocks.py via the pinned importer), plus 5 inconsistent states made by corrupting a
consistent one in a way `b4_hook.is_consistent` flags (holding + handempty, a block on two blocks, a block both on a
block and on the table, two blocks on one block, a held block that is also on another). Expected values are computed
with exactly the pinned call LearnedUnmet.from_file(gnn_s20261276.pt, invalid_policy="fallback", hybrid_lambda=1.0),
one fresh hook, probes in file order; the expected counters after all probes are recorded too. BWM asserts
|value - expected| <= 1e-5 per probe at each shard start (and may compare counters on a fresh hook).
"""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from rcx.bw import b4_hook as H  # noqa: E402
from rcx.bw import bwm_import as BI  # noqa: E402

SEED = 20261241
MODEL = ROOT / "runs/RCX-E07/main/models/gnn_s20261276.pt"
OUT = Path(__file__).resolve().parent / "canary" / "B4_HOOK_CANARY.json"
E07_TEST = "runs/RCX-E07/main/data/instances.json"
TESTB = "runs/RCX-E07/e07b/data/testb_instances.json"
PINNED = ("rcx/bw/b4_hook.py", "rcx/bw/models.py", "rcx/bw/encode.py", "rcx/bw/space.py", "rcx/bw/bwm_import.py",
          "rcx/bw/__init__.py", "rcx/__init__.py", "runs/RCX-E07/main/models/gnn_s20261276.pt",
          "runs/RCX-E07/main/models/gnn_s20261276.json", "runs/RCX-E07/main/data/instances.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fs(x):
    return frozenset(tuple(f) for f in x)


def as_list(s):
    return sorted(list(f) for f in s)


def walk(B, s, objs, rng, steps):
    for _ in range(steps):
        acts = [a for a in candidate_actions(objs) if B.applicable(s, a)]
        s = B.step(s, rng.choice(acts))[0]
    return s


def candidate_actions(objs):
    out = []
    for x in objs:
        out += [("pick-up", x), ("put-down", x)]
        for y in objs:
            if x != y:
                out += [("stack", x, y), ("unstack", x, y)]
    return out


def corruptions(s, objs):
    """Candidate inconsistent variants of consistent state s (each tagged); only flagged ones are kept."""
    on = {f[1]: f[2] for f in s if f[0] == "on"}
    table = sorted(f[1] for f in s if f[0] == "ontable")
    clear = sorted(f[1] for f in s if f[0] == "clear")
    out = []
    if ("handempty",) in s and clear:
        out.append(("holding_and_handempty", s | {("holding", clear[0])}))
    if on:
        x, y = sorted(on.items())[0]
        z = next((b for b in objs if b not in (x, y)), None)
        if z:
            out.append(("on_two_blocks", s | {("on", x, z)}))
        out.append(("on_and_ontable", s | {("ontable", x)}))
    if len(clear) >= 2:
        a, b = clear[0], clear[1]
        c = next((t for t in clear[2:]), None)
        if c:
            out.append(("two_on_one", (s - {("clear", c)}) | {("on", a, c), ("on", b, c)}))
    if table and clear:
        x = clear[0]
        y = next((b for b in objs if b != x), None)
        out.append(("held_and_on", (s - {("handempty",)}) | {("holding", x), ("on", x, y)}))
    return out


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists (create-only)")
    B, _ = BI.load()
    e07 = [x for x in json.loads((ROOT / E07_TEST).read_text(encoding="utf-8")) if x["split"] == "test"]
    tb = [x for x in json.loads((ROOT / TESTB).read_text(encoding="utf-8")) if x["n"] in (14, 15)]
    rng = random.Random(SEED)
    picks = rng.sample(e07, 10) + rng.sample(tb, 10)
    probes = []
    for k, it in enumerate(picks):
        objs, goal = list(it["objs"]), fs(it["goal"])
        s = walk(B, fs(it["init"]), objs, rng, rng.randint(0, 12))
        ok, why = H.is_consistent(s, goal, objs)
        assert ok, why
        probes.append({"id": f"c{k:02d}", "instance_id": it["id"], "n": it["n"], "objs": objs, "state": as_list(s),
                       "goal": as_list(goal), "consistent": True})
    bad = 0
    for it in picks:
        if bad == 5:
            break
        objs, goal = list(it["objs"]), fs(it["goal"])
        s = walk(B, fs(it["init"]), objs, rng, rng.randint(1, 8))
        for tag, t in corruptions(s, objs):
            ok, why = H.is_consistent(t, goal, objs)
            if not ok and tag not in {p.get("corruption") for p in probes}:
                probes.append({"id": f"i{bad:02d}", "instance_id": it["id"], "n": it["n"], "objs": objs,
                               "state": as_list(t), "goal": as_list(goal), "consistent": False, "corruption": tag,
                               "reasons": why})
                bad += 1
                break
    if bad < 5:
        raise SystemExit(f"only {bad} flagged inconsistent probes")
    hook = H.LearnedUnmet.from_file(str(MODEL), invalid_policy="fallback", hybrid_lambda=1.0)
    for p in probes:
        p["expected"] = float(hook(fs(p["state"]), fs(p["goal"])))
        assert (p["expected"] >= H.PENALTY) == (not p["consistent"]), p
    rec = {"what": "B4 hook golden-value canary (RCX co-sign edit 8)", "seed": SEED, "tolerance_abs": 1e-5,
           "hook_call": {"path": "runs/RCX-E07/main/models/gnn_s20261276.pt", "invalid_policy": "fallback",
                         "hybrid_lambda": 1.0},
           "evaluation": "fresh hook, probes in file order, state and goal as frozensets of tuples",
           "expected_counters_after_all": dict(hook.counters),
           "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
           "pins_sha256": {p: sha(ROOT / p) for p in PINNED},
           "probe_sources_sha256": {E07_TEST: sha(ROOT / E07_TEST), TESTB: sha(ROOT / TESTB)},
           "probe_n": sorted({p["n"] for p in probes}), "probes": probes}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1) + "\n")
    print(f"{len(probes)} probes ({sum(p['consistent'] for p in probes)} consistent); counters {dict(hook.counters)}; "
          f"sha256 {sha(OUT)}")


if __name__ == "__main__":
    main()
