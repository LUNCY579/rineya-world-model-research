"""Re-score every published PlanBench Logistics plan (bench/planbench/results/logistics, read-only) with BWM's exact
executor (bwmn/logistics) and with the local VAL on the vendored domain (item 9, CPU-1/CPU-2; CPU only, no model, VAL
calls sequential).

Per plan (raw = the published extracted_llm_plan text as stored; list items joined by newlines):
  val_raw   VAL on raw: the faithful, authoritative re-score of the published flag.
  exec_raw  the executor's re-implementation of that verdict: logistics.execute_text reads raw as VAL does
            (read_val_plan: skipped characters, stop at prose, bad-plan errors, timestamps) and executes it.
  exec      execute() on the lenient extraction of raw (plan_from_pddl: every '(...)' group; an unknown operator or a
            wrong arity is an inapplicable action).
  val_norm  VAL on that same extracted sequence re-rendered one action per line: exec vs val_norm is the semantic
            check on identical action sequences; exec_raw vs val_raw adds the text reading.
  val_read  the steps VAL itself read from raw (`validate -v` listing) vs read_val_plan(raw), where VAL prints a
            listing (it prints none when it stops on an unknown operator or crashes on a wrong arity).
VAL exits 0 on a valid plan and 1 on an invalid one; it segfaults on a wrong-arity action and exits -1 on an unknown
operator. Any exit other than 0 counts as not valid; exits other than 0/1 are counted as val_abnormal_exits. VAL runs
on raw only when its whitespace-normalised text differs from the rendering (otherwise VAL reads the same plan).
The instance file is generated_basic/instance-<id>.pddl; where a file carries ground_truth_plan, the GT is executed on
that file as a mapping check.
Writes runs/LG_published/PUBLISHED_RESCORE.json and prints a summary.

  python -B scripts/lg_published.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import logistics as L  # noqa: E402

OUT = ROOT / "runs" / "LG_published" / "PUBLISHED_RESCORE.json"
BRIEF = ("id", "published", "exec", "exec_fail_step", "exec_raw", "exec_raw_fail_step", "read", "n_read", "val_norm",
         "val_raw", "val_read_agree", "raw_differs", "n_actions", "raw", "val_raw_stdout")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ws(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def plan_text(x) -> str:
    p = x.get("extracted_llm_plan")
    if isinstance(p, list):
        return "\n".join(p)
    return p or ""


def flag(x):
    v = x.get("llm_correct", x.get("correct"))
    return v if isinstance(v, bool) else None


def score(path: Path, init, goal, raw: str) -> tuple[dict, int]:
    """One published plan: the row and the number of VAL calls."""
    acts = L.plan_from_pddl(raw)
    ex = L.execute(init, goal, acts)
    exr = L.execute_text(init, goal, raw)
    steps = L.read_val_plan(raw)
    norm = L.plan_to_pddl(acts) if acts else ""
    vn = L.val_run(path, norm, verbose=True)
    calls = 1
    differs = ws(raw) != ws(norm)
    if differs:
        vr = L.val_run(path, raw, verbose=True)
        calls += 1
    else:
        vr = vn
    listing = L.val_listing(vr["stdout"])
    agree = None if listing == "n/a" else listing == (None if steps is None else sorted(steps))
    tail = vr["stdout"].split("Plan Validation details")[-1][-400:]
    return {"published": None, "exec": ex["success"], "exec_fail_step": ex["fail_step"], "exec_raw": exr["success"],
            "exec_raw_fail_step": exr["fail_step"], "read": exr["read"], "n_read": None if steps is None else len(steps),
            "val_norm": vn["valid"], "val_raw": vr["valid"], "val_read_agree": agree,
            "val_abnormal": vn["returncode"] not in (0, 1), "raw_differs": differs, "n_actions": len(acts), "raw": raw,
            "val_raw_stdout": tail}, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0, help="debug: plans per file (0 = all)")
    args = ap.parse_args()
    assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
    t0 = time.time()
    probs = {}
    files = sorted(L.RESULTS_DIR.rglob("*.json"))
    models, n_val_calls = {}, 0
    for f in files:
        key = f"{f.parent.name}/{f.stem}"
        d = json.loads(f.read_text(encoding="utf-8"))
        insts = d["instances"][:args.limit] if args.limit else d["instances"]
        rows = []
        for x in insts:
            iid = int(x["instance_id"])
            if iid not in probs:
                probs[iid] = L.parse_problem(L.planbench_path(iid))
            objs, init, goal = probs[iid]
            r, calls = score(L.planbench_path(iid), init, goal, plan_text(x))
            n_val_calls += calls
            gt = x.get("ground_truth_plan")
            r.update(id=iid, published=flag(x),
                     gt_ok=None if gt is None else L.execute(init, goal, L.plan_from_pddl(gt))["success"])
            rows.append(r)
        n = len(rows)
        cnt = lambda k: sum(1 for r in rows if r[k] is True)
        differ = lambda a, b: [r for r in rows if r[a] != r[b]]
        ex_vn, exr_vr = differ("exec", "val_norm"), differ("exec_raw", "val_raw")
        pub_vr, pub_exr = differ("published", "val_raw"), differ("published", "exec_raw")
        rd = [r for r in rows if r["val_read_agree"] is False]
        brief = lambda rs: [{k: r[k] for k in BRIEF} for r in rs]
        models[key] = {
            "file": str(f.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(f), "engine": d.get("engine"),
            "prompt_type": d.get("prompt_type"), "n": n, "ids": [min(r["id"] for r in rows), max(r["id"] for r in rows)],
            "published_correct": cnt("published"), "val_raw_valid": cnt("val_raw"), "exec_raw_valid": cnt("exec_raw"),
            "val_norm_valid": cnt("val_norm"), "exec_valid": cnt("exec"),
            "published_val_raw_agree": n - len(pub_vr), "published_exec_raw_agree": n - len(pub_exr),
            "exec_raw_val_raw_agree": n - len(exr_vr), "exec_val_norm_agree": n - len(ex_vn),
            "val_read_checked": sum(r["val_read_agree"] is not None for r in rows), "val_read_agree": cnt("val_read_agree"),
            "read_bad": sum(r["read"] == "bad" for r in rows),
            "val_abnormal_exits": sum(r["val_abnormal"] for r in rows), "raw_text_differs": sum(r["raw_differs"] for r in rows),
            "reading_changes_verdict_ids": sorted(r["id"] for r in rows if r["exec"] != r["exec_raw"]),
            "gt_mapping_checked": sum(r["gt_ok"] is not None for r in rows),
            "gt_mapping_ok": sum(r["gt_ok"] is True for r in rows),
            "disagreements_published_vs_val_raw": brief(pub_vr),
            "disagreements_published_vs_exec_raw": brief(pub_exr),
            "disagreements_exec_raw_vs_val_raw": brief(exr_vr),
            "disagreements_exec_vs_val_norm": brief(ex_vn),
            "disagreements_val_read": brief(rd),
            "val_raw_valid_ids": sorted(r["id"] for r in rows if r["val_raw"]),
            "published_correct_ids": sorted(r["id"] for r in rows if r["published"]),
        }
        m = models[key]
        print(f"{key:55s} n {n:3d} pub {m['published_correct']:3d} VALraw {m['val_raw_valid']:3d} "
              f"execraw {m['exec_raw_valid']:3d} VALnorm {m['val_norm_valid']:3d} exec {m['exec_valid']:3d} | "
              f"pub!=VALraw {len(pub_vr)} pub!=execraw {len(pub_exr)} execraw!=VALraw {len(exr_vr)} "
              f"exec!=VALnorm {len(ex_vn)} read {m['val_read_agree']}/{m['val_read_checked']} | bad-read "
              f"{m['read_bad']} abnormal {m['val_abnormal_exits']} GT-map {m['gt_mapping_ok']}/{m['gt_mapping_checked']}",
              flush=True)
    tot = lambda k: sum(m[k] for m in models.values())
    summary = {"n_files": len(models), "n_plans": tot("n"), "published_correct": tot("published_correct"),
               "val_raw_valid": tot("val_raw_valid"), "exec_raw_valid": tot("exec_raw_valid"),
               "val_norm_valid": tot("val_norm_valid"), "exec_valid": tot("exec_valid"),
               "published_vs_val_raw_disagreements": tot("n") - tot("published_val_raw_agree"),
               "published_vs_exec_raw_disagreements": tot("n") - tot("published_exec_raw_agree"),
               "exec_raw_vs_val_raw_disagreements": tot("n") - tot("exec_raw_val_raw_agree"),
               "exec_vs_val_norm_disagreements": tot("n") - tot("exec_val_norm_agree"),
               "val_read_checked": tot("val_read_checked"), "val_read_agree": tot("val_read_agree"),
               "read_bad": tot("read_bad"), "reading_changes_verdict": sum(len(m["reading_changes_verdict_ids"])
                                                                           for m in models.values()),
               "val_abnormal_exits": tot("val_abnormal_exits"), "raw_text_differs": tot("raw_text_differs"),
               "gt_mapping_ok": tot("gt_mapping_ok"), "gt_mapping_checked": tot("gt_mapping_checked"),
               "val_calls": n_val_calls, "seconds": round(time.time() - t0, 1)}
    out = {"meta": {"script": "scripts/lg_published.py", "logistics_py_sha256": sha(ROOT / "bwmn" / "logistics.py"),
                    "domain": str(L.DOMAIN_PDDL.relative_to(ROOT)).replace("\\", "/"),
                    "domain_sha256": sha(L.DOMAIN_PDDL), "val_exe_sha256": sha(L.VAL_EXE),
                    "instances": "bench/planbench/logistics/generated_basic/instance-<id>.pddl",
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"), "limit": args.limit},
           "summary": summary, "models": models}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("SUMMARY", json.dumps(summary))
    for key, m in models.items():
        for kind in ("disagreements_exec_vs_val_norm", "disagreements_exec_raw_vs_val_raw",
                     "disagreements_published_vs_val_raw", "disagreements_published_vs_exec_raw", "disagreements_val_read"):
            for r in m[kind]:
                print(f"  {kind} {key} id {r['id']}: published {r['published']} exec {r['exec']} exec_raw "
                      f"{r['exec_raw']} VALnorm {r['val_norm']} VALraw {r['val_raw']} read {r['read']}/{r['n_read']} "
                      f"val_read_agree {r['val_read_agree']}")
        if m["reading_changes_verdict_ids"]:
            print(f"  reading changes the executor verdict {key}: ids {m['reading_changes_verdict_ids']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
