"""Report likelihood ranking of candidate consequences (scripts/run_lik.py outputs).

  python scripts/lik_report.py --bank bank_pilot --arms sparse,supa [--split test]

Per arm x family x frame (one / k1 / k2 / k4):
  top1        gold has the highest log p among {gold} U foils
  win_<foil>  P(log p(gold) > log p(foil)) over items where that foil exists (foil != gold)
Cross-arm: paired parent bootstrap on top1 (first arm minus each other arm).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
from rcwmt import decide, env  # noqa: E402


def load(bank, arm):
    p = ROOT / "runs" / bank / arm / "lik_rank.jsonl"
    return {r["item_id"]: r["logp"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())} if p.exists() else None


def frame_of(iid):
    return "one" if iid.endswith("-one") else iid.rsplit("-", 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    bank = json.loads((ROOT / "data" / f"{a.bank}.json").read_text(encoding="utf-8"))
    pinfo = {p["parent_id"]: (p["family"], p["split"]) for p in bank["parents"]}
    arms = {x: load(a.bank, x) for x in a.arms.split(",")}
    arms = {k: v for k, v in arms.items() if v}
    report = {}
    for fam in env.FAMILIES:
        for frame in ("one", "k1", "k2", "k4"):
            res = {}
            per_parent = {}
            for arm, L in arms.items():
                ids = [i for i in L if frame_of(i) == frame and pinfo[i.split("-c")[0]][0] == fam
                       and pinfo[i.split("-c")[0]][1] in a.split.split(",")]
                top1, wins = [], {}
                for i in ids:
                    d = L[i]
                    g = d["gold"]
                    foils = {k: v for k, v in d.items() if k != "gold"}
                    top1.append(float(all(g > v for v in foils.values())))
                    for k, v in foils.items():
                        base = k[:-1] if k[-1].isdigit() else k
                        wins.setdefault(base, []).append(float(g > v))
                pids = [i.split("-c")[0] for i in ids]
                up = sorted(set(pids)); ix = {p: n for n, p in enumerate(up)}
                s_, c_ = np.zeros(len(up)), np.zeros(len(up))
                for p, t in zip(pids, top1):
                    s_[ix[p]] += t; c_[ix[p]] += 1
                per_parent[arm] = (up, s_ / np.maximum(c_, 1))
                res[arm] = {"n": len(ids), "top1": float(np.mean(top1)) if top1 else float("nan"),
                            **{f"win_{k}": float(np.mean(v)) for k, v in sorted(wins.items())}}
            names = list(arms)
            if len(names) > 1:
                up0, v0 = per_parent[names[0]]
                for other in names[1:]:
                    up1, v1 = per_parent[other]
                    assert up0 == up1
                    res[f"top1 {names[0]} - {other}"] = decide.paired_bootstrap(v0, v1)
            report[f"{fam}|{frame}"] = res
            line = f"{fam[:6]} {frame:<3} " + " | ".join(
                f"{arm}: top1 {res[arm]['top1']:.3f} " + " ".join(f"{k[4:]}={v:.2f}" for k, v in res[arm].items() if k.startswith("win_"))
                for arm in arms)
            for k, v in res.items():
                if k.startswith("top1 "):
                    line += f" || {k} {v['mean_diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
            print(line, flush=True)
    out = ROOT / "runs" / a.bank / "analysis" / f"lik_report_{a.split.replace(',', '_')}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
