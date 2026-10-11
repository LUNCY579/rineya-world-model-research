"""B3 conditional hard-set runs H1/H2 (PREREG_B3_DISTILL.md, "Hard-set runs"; D-65: they run before item 7 if met).

  gate     (CPU; a bundleE sub-job right after b3-analyze) Reads runs/BW_B3_EVAL/B3_REPORT.json. If the registered
           primary reading is "met": writes runs/FREEZE_B3_H1.json / FREEZE_B3_H2.json (the frozen B2 direct / beam
           args with ONLY the proposer adapter swapped to Pa_v3, plus the Pa_v3, FREEZE_B3 and prereg SHA-256s) and
           runs/BW_B3_EVAL/H_GO.json, which makes the queued bundleH runnable. Otherwise writes H_NO.json. Exit 0.
  run      (GPU; bundleH) H1 = hard S-direct (one process), H2 = hard S-beam in 8 shards (index % 8, as D-50), each
           blocks_eval.py with --val and the H freeze record (test-set guard: args must match, run once). Resumable
           per shard; merges H2 with bw_run_tests.merge.
  analyze  (CPU) Paired exact McNemar of H1 vs the frozen B2 S-direct and H2 vs the frozen B2 S-beam on the same 110
           instances; Clopper-Pearson CIs. Writes runs/BW_B3_HARD/B3_HARD_REPORT.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bw_run_tests import argv_for, merge  # noqa: E402

PA_V3 = "runs/BW_Pa_v3_s1/adapter"
OUT = ROOT / "runs/BW_B3_HARD"
GO, NO = ROOT / "runs/BW_B3_EVAL/H_GO.json", ROOT / "runs/BW_B3_EVAL/H_NO.json"
H2_SHARDS = 8


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def gate(a):
    if GO.exists() or NO.exists():
        print("gate already decided:", (GO if GO.exists() else NO).name)
        return 0
    rep = json.loads((ROOT / "runs/BW_B3_EVAL/B3_REPORT.json").read_text())
    reading = rep.get("primary_reading")
    if reading != "met":
        NO.write_text(json.dumps({"primary_reading": reading, "note": "registered: no hard-set runs"}, indent=1))
        print("B3 primary reading", reading, "-> no H1/H2")
        return 0
    frz_b3 = json.loads((ROOT / "runs/FREEZE_B3.json").read_text())
    pa_sha = sha(ROOT / PA_V3 / "adapter_model.safetensors")
    assert pa_sha == frz_b3["arms"]["v3"]["sha256"], "Pa_v3 differs from FREEZE_B3"
    for name, src in (("H1", "runs/FREEZE_B2_direct.json"), ("H2", "runs/FREEZE_B2_beam.json")):
        b2 = json.loads((ROOT / src).read_text())
        args = dict(b2["args"], pa=PA_V3)
        rec = {"args": args, "derived_from": src, "only_change": "pa -> " + PA_V3, "pa_sha256": pa_sha,
               "freeze_b3_sha256": sha(ROOT / "runs/FREEZE_B3.json"), "prereg_b3_sha256": sha(ROOT / "PREREG_B3_DISTILL.md"),
               "b3_report_sha256": sha(ROOT / "runs/BW_B3_EVAL/B3_REPORT.json")}
        dst = ROOT / f"runs/FREEZE_B3_{name}.json"
        if dst.exists():
            sys.exit(f"{dst} exists")
        dst.write_text(json.dumps(rec, indent=1))
    GO.write_text(json.dumps({"primary_reading": reading, "freeze": ["runs/FREEZE_B3_H1.json", "runs/FREEZE_B3_H2.json"]},
                             indent=1))
    print("B3 primary reading met -> FREEZE_B3_H1/H2 written; bundleH runnable")
    return 0


def run(a):
    if not GO.exists():
        sys.exit("H_GO.json missing: H1/H2 are registered only if B3's primary reading is met")
    for name, out, k in (("H1", OUT / "h1_direct", 1), ("H2", OUT / "h2_beam", H2_SHARDS)):
        frz = f"runs/FREEZE_B3_{name}.json"
        rec = json.loads((ROOT / frz).read_text())
        assert sha(ROOT / PA_V3 / "adapter_model.safetensors") == rec["pa_sha256"], "Pa_v3 changed since the H freeze"
        if (out / "summary.json").exists():
            continue
        for i in range(k):
            dst = out if k == 1 else out / f"shard_{i}of{k}"
            if (dst / "summary.json").exists():
                continue
            cmd = [sys.executable, "scripts/blocks_eval.py", "--set", "hard", "--val", "--out", str(dst.relative_to(ROOT)),
                   "--freeze", frz] + argv_for(rec["args"]) + (["--shard", f"{i}/{k}"] if k > 1 else [])
            print(f"[b3-hard] run {name}" + (f" shard {i}/{k}" if k > 1 else ""), flush=True)
            rc = subprocess.call(cmd, cwd=str(ROOT))
            if rc != 0:
                print(f"[b3-hard] {name} exited {rc}", flush=True)
                return rc
        if k > 1:
            merge(out, k)
        print(f"[b3-hard] done {name}", (out / "summary.json").read_text()[:300], flush=True)
    (OUT / ".run_done").write_text("H1 and H2 complete")
    return 0


def analyze(a):
    from scipy.stats import beta, binomtest

    def load(d):
        return {r["id"]: int(bool(r.get("val", r["success"]))) for r in map(json.loads, open(d / "plans.jsonl", encoding="utf-8"))}

    def cp(k, n):
        return [0.0 if k == 0 else round(float(beta.ppf(0.025, k, n - k + 1)), 4),
                1.0 if k == n else round(float(beta.ppf(0.975, k + 1, n - k)), 4)]

    rep = {}
    for name, new, old in (("H1", OUT / "h1_direct", ROOT / "runs/BW_TEST_B2/hard_direct"),
                           ("H2", OUT / "h2_beam", ROOT / "runs/BW_TEST_B2/hard_beam")):
        x, y = load(new), load(old)
        assert set(x) == set(y) and len(x) == 110
        b = sum(x[i] and not y[i] for i in x)
        c = sum(y[i] and not x[i] for i in x)
        rep[name] = {"pa_v3": f"{sum(x.values())}/110", "pa_v3_ci95": cp(sum(x.values()), 110),
                     "b2_frozen": f"{sum(y.values())}/110", "v3_only": b, "b2_only": c,
                     "p_mcnemar": 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue}
    (OUT / "B3_HARD_REPORT.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gate", "run", "analyze"])
    a = ap.parse_args()
    sys.exit({"gate": gate, "run": run, "analyze": analyze}[a.stage](a))


if __name__ == "__main__":
    main()
