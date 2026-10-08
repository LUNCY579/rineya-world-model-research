"""RCWM-T2 review (lens: retention / confounds). CPU only, read-only.

Reproduces the answer-free baselines a T2 result must beat, per stratum and per P sub-stratum:
  copy      output == input state
  ign       Titan's observed heuristic: ledger ignores status; prerequisite = s0 + add (integer
            addition, ignores preconditions and delete); register: n/a
plus Titan-Sparse's pilot step-1 accuracy (unique items, pilot TEST, chain_one) and copy-chain
accuracy on the pilot TEST chains that run_t2 evaluates. Also first-token peakedness of the
off-task capture.

  CUDA_VISIBLE_DEVICES=-1 python docs/t2_review/retention_confounds_baselines.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import env  # noqa: E402


def sub(fam, s, ev):
    if fam == "prerequisite_plan":
        if not env.applicable(s, ev):
            return "inapp"
        a = (s & ~ev["delete"]) | ev["add"]
        return "app_same" if a == s else ("app_del" if s & ev["delete"] else "app_addonly")
    if fam == "ledger_replay":
        return f"{ev['kind']}/{ev['status']}"
    k = ev["kind"]
    if k == "conditional":
        return "cond/" + ("gt" if s[ev["other"]] > ev["threshold"] else "le")
    if k in ("add", "mix"):
        raw = s[ev["register"]] + (ev["value"] if k == "add" else 2 * s[ev["other"]] + ev["value"])
        return f"{k}/" + ("wrap" if not 0 <= raw < env.MOD else "nowrap")
    return k


def titan_heur(fam, s, ev):
    if fam == "prerequisite_plan":
        return s + ev["add"]
    if fam == "ledger_replay":
        return env.step(fam, s, {**ev, "status": "posted"})
    return None


def main():
    print("== T2 data: copy-equal targets and heuristic baselines per sub-stratum")
    for split in ("train", "eval"):
        rows = [json.loads(l) for l in (ROOT / "data" / f"t2_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        c = collections.defaultdict(lambda: [0, 0, 0])
        for r in rows:
            k = (r["family"][:1], sub(r["family"], r["before"], r["event"]))
            c[k][0] += 1
            c[k][1] += env.dumps(r["before"]) == r["target"]
            h = titan_heur(r["family"], r["before"], r["event"])
            c[k][2] += h is not None and env.dumps(h) == r["target"]
        n = len(rows)
        print(f" {split}: n={n} copy-equal={sum(v[1] for v in c.values()) / n:.3f}")
        for k, (m, cp, hh) in sorted(c.items()):
            print(f"   {k[0]}|{k[1]:<16} n={m:4d} copy={cp / m:.3f} titan_heur={hh / m:.3f}")

    b = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
    cand = {c["cand_id"]: (p, c) for p in b["parents"] for c in p["candidates"]}
    print("== pilot TEST chains: copy-chain baseline vs Titan-Sparse (lenient)")
    for mode in ("chain_one", "chain_m1"):
        gen = [json.loads(l) for l in (ROOT / "runs/bank_pilot/sparse" / f"gen_{mode}.jsonl").read_text(encoding="utf-8").splitlines()]
        acc, cp = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
        step1, seen = collections.defaultdict(lambda: [0, 0]), set()
        for r in gen:
            p, c = cand[r["cand_id"]]
            if p["split"] != "test":
                continue
            fam = p["family"]
            g = env.rollout(fam, p["s0"], c["events"])
            for k in (1, 2, 4):
                acc[(fam[:1], k)][0] += 1
                acc[(fam[:1], k)][1] += len(r["states"]) > k and r["states"][k] == env.dumps(g[k])
                cp[(fam[:1], k)][1] += g[k] == g[0]
            key = env.dumps({"b": p["s0"], "e": c["events"][0]})
            if key not in seen:
                seen.add(key)
                kk = (fam[:1], sub(fam, p["s0"], c["events"][0]))
                step1[kk][0] += 1
                step1[kk][1] += r["states"][1] == env.dumps(g[1])
        print(f" {mode} chain acc  " + " ".join(f"{f}k{k}={v[1] / v[0]:.3f}(copy {cp[(f, k)][1] / v[0]:.3f})" for (f, k), v in sorted(acc.items())))
        if mode == "chain_one":
            print(" step-1 unique items, Titan-Sparse acc: " + ", ".join(f"{f}|{s} {v[1] / v[0]:.2f} (n={v[0]})" for (f, s), v in sorted(step1.items())))

    LP = np.load(ROOT / "runs/bank_pilot/sparse/offtask_logprobs.npy", mmap_mode="r")
    top = np.array([np.exp(LP[i].astype(np.float64)).max() for i in range(LP.shape[0])])
    print(f"== off-task first token (Sparse): top-1 p quartiles {np.round(np.quantile(top, [.25, .5, .75]), 3)}, frac>0.9 {(top > .9).mean():.3f}")


if __name__ == "__main__":
    main()
