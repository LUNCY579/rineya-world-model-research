"""Run the frozen Track B/B2 configurations (runs/FREEZE_B2_{beam,direct}.json, written by bw_freeze.py) on the test
sets, each exactly once: Blocksworld-hard (110), its generated mirror (110), PlanBench-600. Every run is a separate
blocks_eval.py process with VAL checking; runs whose summary.json exists are skipped, so a turn cap or crash resumes
with the next run (a run interrupted mid-way restarts from scratch with identical seeds and a warm F cache).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = [("hard", "direct"), ("hard", "beam"), ("mirror", "direct"), ("mirror", "beam"),
         ("planbench", "direct"), ("planbench", "beam")]  # cheap direct runs first, so a turn cap never strands them


def argv_for(args: dict) -> list[str]:
    out = []
    flags = {"proposer": "--proposer", "pa": "--pa", "mode": "--mode", "fwd": "--fwd", "fwd_adapter": "--fwd-adapter",
             "fwd_fmt": "--fwd-fmt", "K": "--K", "eval_seed": "--eval-seed", "planner": "--planner", "W": "--W",
             "E": "--E", "max_depth": "--max-depth", "step_cap": "--step-cap", "R": "--R", "M": "--M", "P": "--P"}
    for k, f in flags.items():
        if args.get(k) is not None:
            out += [f, str(args[k])]
    return out


SHARDS = {("hard", "beam"): 8, ("mirror", "beam"): 8, ("planbench", "beam"): 4}  # D-50: each shard ~1 h of GPU


def merge(out: Path, k: int) -> None:
    """Combine shard outputs into out/summary.json + out/plans.jsonl (each instance appears in exactly one shard)."""
    rows, summ = [], {}
    for i in range(k):
        d = out / f"shard_{i}of{k}"
        rows += [json.loads(l) for l in open(d / "plans.jsonl", encoding="utf-8")]
        s = json.loads((d / "summary.json").read_text())["summary"]
        for key in ("f_model_queries", "wall_s"):
            summ[key] = summ.get(key, 0) + (s.get(key) or 0)
    n = len(rows)
    ok = [bool(r.get("val", r["success"])) for r in rows]
    by = {}
    for r in rows:
        by.setdefault(r.get("n_blocks"), []).append(bool(r.get("val", r["success"])))
    summ.update({"n": n, "success": sum(r["success"] for r in rows) / n, "val_success": sum(ok) / n,
                 "val_disagreements": sum(bool(r.get("val", r["success"])) != bool(r["success"]) for r in rows),
                 "verified_rate": sum(r["verified"] for r in rows) / n,
                 "f_queries_per_instance": round(summ["f_model_queries"] / n, 1),
                 "proposer_tokens_per_instance": round(sum(r["gen_tokens"] for r in rows) / n, 1),
                 "shards": k,
                 "success_by_blocks": {b: round(sum(v) / len(v), 4) for b, v in sorted(by.items())}})
    with open(out / "plans.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps({"summary": summ}, indent=1))


def main():
    for tset, sysname in ORDER:
        frz = f"runs/FREEZE_B2_{sysname}.json"
        out = ROOT / f"runs/BW_TEST_B2/{tset}_{sysname}"
        if (out / "summary.json").exists():
            continue
        args = json.loads((ROOT / frz).read_text())["args"]
        k = SHARDS.get((tset, sysname), 1)
        for i in range(k):
            dst = out if k == 1 else out / f"shard_{i}of{k}"
            if (dst / "summary.json").exists():
                continue
            cmd = [sys.executable, "scripts/blocks_eval.py", "--set", tset, "--val", "--out",
                   str(dst.relative_to(ROOT)), "--freeze", frz] + argv_for(args) + (["--shard", f"{i}/{k}"] if k > 1 else [])
            print("[tests] run", tset, sysname, f"shard {i}/{k}" if k > 1 else "", flush=True)
            rc = subprocess.call(cmd, cwd=str(ROOT))
            if rc != 0:
                print(f"[tests] {tset}/{sysname} exited {rc}", flush=True)
                return rc
        if k > 1:
            merge(out, k)
        print("[tests] done", tset, sysname, (out / "summary.json").read_text()[:400], flush=True)
    (ROOT / "runs/BW_TEST_B2/.done").write_text("all test runs complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
