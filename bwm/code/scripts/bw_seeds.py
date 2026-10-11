"""Item 7 seed replication (PREREG_ITEM7_SEEDS.md, D-65).

  fresh    (CPU) runs/BW_val_large/instances_seeds_fresh.json, n = 40, seed 20261014; refuses to overwrite.
  freeze   (CPU) runs/FREEZE_I7_{R0,R1,R2}_{beam,direct}.json from the frozen B2 args with only the adapter paths
           swapped, plus adapter, prereg and fresh-set SHA-256s; refuses to overwrite; needs all adapters.
  run      (GPU) per system R0, R1, R2: S-direct (one process), then S-beam in 4 shards (index % 4); blocks_eval.py
           --set seeds_fresh --val with the system's freeze record (test-set guard); resumable per shard.
  analyze  (CPU) registered tests and report-only spread; writes runs/BW_SEEDS/SEEDS_REPORT.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bw_run_tests import argv_for, merge  # noqa: E402

FRESH = ROOT / "runs/BW_val_large/instances_seeds_fresh.json"
FRESH_SEED, N = 20261014, 40
SYSTEMS = {"R0": ("runs/BW_Fdelta_s0/adapter", "runs/BW_Pa_v2_s1/adapter"),
           "R1": ("runs/BW_Fdelta_s1/adapter", "runs/BW_Pa_v2_s2/adapter"),
           "R2": ("runs/BW_Fdelta_s2/adapter", "runs/BW_Pa_v2_s3/adapter")}
OUT = ROOT / "runs/BW_SEEDS"
BEAM_SHARDS = 4


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fresh(a):
    from collections import Counter
    from bwmn import blocks as B
    import blocks_val_large as VL
    import bw_distill as BD
    if FRESH.exists():
        sys.exit(f"{FRESH} exists (written once)")
    hard = [B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")]
    inv = {VL.invariant(*h) for h in hard}
    for fn in ("instances_val_large.json", "instances_mirror_hard.json", "instances_b3_fresh.json"):
        for r in json.loads((ROOT / "runs/BW_val_large" / fn).read_text()):
            inv.add(VL.invariant(r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))))
    for it in BD.problems(240):
        inv.add(VL.invariant(it["objs"], it["init"], it["goal"]))
    sizes = sorted(len(h[0]) for h in hard)
    pick = [sizes[int(i * len(sizes) / N)] for i in range(N)]
    out = VL.make(random.Random(FRESH_SEED), pick, B.hard_signatures(), inv)
    FRESH.write_text(json.dumps(out))
    print("seeds_fresh", len(out), sorted(Counter(r["n"] for r in out).items()), "excluded invariants", len(inv))


def freeze(a):
    recs = {}
    for name, (f, pa) in SYSTEMS.items():
        for cfg, src in (("beam", "runs/FREEZE_B2_beam.json"), ("direct", "runs/FREEZE_B2_direct.json")):
            dst = ROOT / f"runs/FREEZE_I7_{name}_{cfg}.json"
            if dst.exists():
                sys.exit(f"{dst} exists")
            args = dict(json.loads((ROOT / src).read_text())["args"], pa=pa)
            if cfg == "beam":
                args["fwd_adapter"] = f
            recs[dst] = {"args": args, "derived_from": src, "system": name,
                         "pa_sha256": sha(ROOT / pa / "adapter_model.safetensors"),
                         "fwd_sha256": sha(ROOT / f / "adapter_model.safetensors"),
                         "prereg_sha256": sha(ROOT / "PREREG_ITEM7_SEEDS.md"), "fresh_sha256": sha(FRESH)}
    for dst, rec in recs.items():
        dst.write_text(json.dumps(rec, indent=1))
    print("wrote", len(recs), "freeze records")


def run(a):
    for name in SYSTEMS:
        for cfg, k in (("direct", 1), ("beam", BEAM_SHARDS)):
            frz = f"runs/FREEZE_I7_{name}_{cfg}.json"
            rec = json.loads((ROOT / frz).read_text())
            for key, path in (("pa_sha256", rec["args"]["pa"]), ("fwd_sha256", SYSTEMS[name][0])):
                assert sha(ROOT / path / "adapter_model.safetensors") == rec[key], f"{frz}: {key} changed"
            out = OUT / f"{name}_{cfg}"
            if (out / "summary.json").exists():
                continue
            for i in range(k):
                dst = out if k == 1 else out / f"shard_{i}of{k}"
                if (dst / "summary.json").exists():
                    continue
                cmd = [sys.executable, "scripts/blocks_eval.py", "--set", "seeds_fresh", "--val", "--out",
                       str(dst.relative_to(ROOT)), "--freeze", frz] + argv_for(rec["args"]) + \
                    (["--shard", f"{i}/{k}"] if k > 1 else [])
                print(f"[seeds] run {name} {cfg}" + (f" shard {i}/{k}" if k > 1 else ""), flush=True)
                rc = subprocess.call(cmd, cwd=str(ROOT))
                if rc != 0:
                    print(f"[seeds] {name}/{cfg} exited {rc}", flush=True)
                    return rc
            if k > 1:
                merge(out, k)
            print(f"[seeds] done {name} {cfg}", (out / "summary.json").read_text()[:300], flush=True)
    (OUT / ".run_done").write_text("all systems evaluated")
    return 0


def analyze(a):
    from scipy.stats import beta, binomtest

    def load(d):
        return {r["id"]: int(bool(r.get("val", r["success"]))) for r in map(json.loads, open(d / "plans.jsonl", encoding="utf-8"))}

    def cp(k, n):
        return [0.0 if k == 0 else round(float(beta.ppf(0.025, k, n - k + 1)), 4),
                1.0 if k == n else round(float(beta.ppf(0.975, k + 1, n - k)), 4)]

    def mc(x, y):
        b = sum(x[i] and not y[i] for i in x)
        c = sum(y[i] and not x[i] for i in x)
        return b, c, (1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue)

    res, sysres = {}, {}
    for name in SYSTEMS:
        beam, direct = load(OUT / f"{name}_beam"), load(OUT / f"{name}_direct")
        assert set(beam) == set(direct) and len(beam) == N
        b, c, p = mc(beam, direct)
        d = (b - c) / N
        sysres[name] = (beam, direct)
        res[name] = {"s_beam": f"{sum(beam.values())}/{N}", "s_beam_ci95": cp(sum(beam.values()), N),
                     "s_direct": f"{sum(direct.values())}/{N}", "s_direct_ci95": cp(sum(direct.values()), N),
                     "beam_only": b, "direct_only": c, "delta": round(d, 4), "p_mcnemar": p,
                     "replicates": bool(p < 0.01 and b > c and d >= 0.30)}
    spread = {}
    for name in ("R1", "R2"):
        for i, cfg in enumerate(("beam", "direct")):
            b, c, p = mc(sysres[name][i], sysres["R0"][i])
            spread[f"{name}_vs_R0_{cfg}"] = {"only_" + name: b, "only_R0": c, "p_mcnemar": p}
    rep = {"systems": res, "spread_report_only": spread,
           "release_gate_reading": "the B2 effect replicates across seeds" if res["R1"]["replicates"] and
           res["R2"]["replicates"] else "not replicated in at least one replicate"}
    (OUT / "SEEDS_REPORT.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fresh", "freeze", "run", "analyze"])
    a = ap.parse_args()
    sys.exit({"fresh": fresh, "freeze": freeze, "run": run, "analyze": analyze}[a.stage](a) or 0)


if __name__ == "__main__":
    main()
