"""Addendum B3 primary and secondary evaluation (PREREG_B3_DISTILL.md), GPU, run once after the freeze.

  freeze   (CPU) write runs/FREEZE_B3.json: SHA-256 of the three proposer adapters (Pa_v3, Pa_v2c, Pa_v2), the B3 prereg,
           the fresh test set, MIX_META and DISTILL_META. Refuses to overwrite.
  run      (GPU) S-direct (frozen B2 direct settings: greedy candidate 0, K 1, M 256, eval_seed 0) for each arm on
           b3fresh (primary) and mirror (secondary). Verifies every SHA against the freeze first. Resumable per
           (arm, set); a finished pair is never rerun. Writes runs/BW_B3_EVAL/results.jsonl + plans_<arm>_<set>.jsonl.
  analyze  (CPU) paired exact McNemar for the registered contrasts, Clopper-Pearson CIs, the pre-set reading.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FREEZE = ROOT / "runs/FREEZE_B3.json"
OUT = ROOT / "runs/BW_B3_EVAL"
ARMS = {"v3": "runs/BW_Pa_v3_s1/adapter", "v2c": "runs/BW_Pa_v2c_s1/adapter", "v2": "runs/BW_Pa_v2_s1/adapter"}
SETS = {"b3fresh": "runs/BW_val_large/instances_b3_fresh.json", "mirror": "runs/BW_val_large/instances_mirror_hard.json"}
FILES = {"prereg": "PREREG_B3_DISTILL.md", "b3fresh": SETS["b3fresh"], "mix_meta": "runs/BW_DISTILL_r1/MIX_META.json",
         "distill_meta": "runs/BW_DISTILL_r1/DISTILL_META.json"}
M, K, EVAL_SEED = 256, 1, 0  # = runs/FREEZE_B2_direct.json


def _sha(p: str) -> str:
    from bwmn import lm
    return lm.file_sha256(ROOT / p)


def freeze(a):
    if FREEZE.exists():
        sys.exit(f"{FREEZE} exists")
    rec = {"arms": {k: {"path": v, "sha256": _sha(v + "/adapter_model.safetensors")} for k, v in ARMS.items()},
           "files": {k: {"path": v, "sha256": _sha(v)} for k, v in FILES.items()},
           "direct": {"K": K, "M": M, "eval_seed": EVAL_SEED}, "frozen_unix": time.time()}
    FREEZE.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def run(a):
    from blocks_eval import _json_set
    from bwmn import blocks_planner as BP
    from bwmn import lm
    from peft import PeftModel

    frz = json.loads(FREEZE.read_text())
    for k, v in frz["arms"].items():
        if _sha(v["path"] + "/adapter_model.safetensors") != v["sha256"]:
            sys.exit(f"adapter {k} differs from the freeze")
    for k, v in frz["files"].items():
        if _sha(v["path"]) != v["sha256"]:
            sys.exit(f"{v['path']} differs from the freeze")
    OUT.mkdir(parents=True, exist_ok=True)
    res = OUT / "results.jsonl"
    done = {(r["arm"], r["set"]) for r in map(json.loads, res.read_text().splitlines())} if res.exists() else set()
    base, tok = lm.load_core("titan-sparse")
    model = None
    for k, v in frz["arms"].items():
        if model is None:
            model = PeftModel.from_pretrained(base, str(ROOT / v["path"]), adapter_name=k, is_trainable=False)
        else:
            model.load_adapter(str(ROOT / v["path"]), adapter_name=k, is_trainable=False)
    model.eval()
    for sname, path in SETS.items():
        inst = _json_set(ROOT / path, sname)
        for arm in frz["arms"]:
            if (arm, sname) in done:
                continue
            t0 = time.time()
            prop = BP.LMProposerBW(model, tok, arm, "actions", K=K, max_new=M, max_batch_tokens=a.prop_batch_tokens)
            eps = BP.plan_direct(inst, prop, eval_seed=EVAL_SEED)
            summ, rows = BP.score(inst, eps)
            with open(OUT / f"plans_{arm}_{sname}.jsonl", "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            rec = {"arm": arm, "set": sname, "n": len(inst), "success": summ["success"],
                   "truncated": sum(e["gen_tokens"] >= M for e in eps), "wall_s": round(time.time() - t0, 1)}
            with open(res, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


def _cp(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return round(lo, 3), round(hi, 3)


def _mcnemar(b, c):
    from scipy.stats import binomtest
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue


def analyze(a):
    rows = {}
    for p in OUT.glob("plans_*_*.jsonl"):
        arm, sname = p.stem.split("_", 2)[1:]
        rows[(arm, sname)] = {r["id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    rep = {"cells": {}, "contrasts": {}}
    for (arm, sname), d in sorted(rows.items()):
        k, n = sum(r["success"] for r in d.values()), len(d)
        inr = [r["success"] for r in d.values() if r["n_blocks"] <= 10]
        out = [r["success"] for r in d.values() if r["n_blocks"] > 10]
        rep["cells"][f"{arm}/{sname}"] = {"success": f"{k}/{n}", "rate": round(k / n, 4), "ci95": _cp(k, n),
                                          "in_range_6_10": f"{sum(inr)}/{len(inr)}",
                                          "out_of_range_11_15": f"{sum(out)}/{len(out)}"}
    for x, y in (("v3", "v2c"), ("v3", "v2"), ("v2c", "v2")):
        for sname in SETS:
            if (x, sname) in rows and (y, sname) in rows:
                dx, dy = rows[(x, sname)], rows[(y, sname)]
                b = sum(dx[i]["success"] and not dy[i]["success"] for i in dx)
                c = sum(dy[i]["success"] and not dx[i]["success"] for i in dx)
                n = len(dx)
                rep["contrasts"][f"{x}-{y}/{sname}"] = {"diff_pp": round(100 * (b - c) / n, 1), "x_only": b,
                                                        "y_only": c, "p_mcnemar": _mcnemar(b, c)}
    pc = rep["contrasts"].get("v3-v2c/b3fresh")
    if pc:
        rep["primary_reading"] = ("met" if pc["diff_pp"] >= 10 and pc["p_mcnemar"] < 0.05 else
                                  "partial" if pc["diff_pp"] > 0 else "no")
    (OUT / "B3_REPORT.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "run", "analyze"])
    ap.add_argument("--prop-batch-tokens", type=int, default=6000)
    a = ap.parse_args()
    {"freeze": freeze, "run": run, "analyze": analyze}[a.stage](a)


if __name__ == "__main__":
    main()
