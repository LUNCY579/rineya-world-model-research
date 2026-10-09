"""Track B/B2 planner selection on GENERATED validation sets only (PREREG_PLANBENCH.md addendum B2).

Loads the frozen core once with the proposer and one or two forward adapters, then runs every configuration in a
small grid on val_v2 (3-6 blocks) and val_large (6-15 blocks, hard-style goals). Each forward adapter's predictions
are memoised across configurations (F is a frozen function, so sharing its cache changes no result, only cost; the
reported F-query counts are per configuration, recomputed from the unique keys each configuration touched).
Results are appended per (config, set) to <out>/select.jsonl; finished pairs are skipped on restart.

Usage: python scripts/bw_select.py --pa runs/BW_Pa_v2_s1/adapter --fwd-state runs/BW_F_s0/adapter \
           --fwd-delta runs/BW_Fdelta_s0/adapter --out runs/BW_SELECT_v1
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
from bwmn import blocks_planner as BP  # noqa: E402
from bwmn import lm  # noqa: E402
from blocks_eval import load_set  # noqa: E402


class CountingF:
    """View of a shared memoised F that records which keys this configuration needed."""

    def __init__(self, f):
        self.f, self.keys = f, set()

    def predict(self, keys):
        self.keys.update(keys)
        return self.f.predict(keys)


GRID = [
    {"planner": "direct", "K": 1},
    {"planner": "beam", "K": 4, "W": 4, "E": 24, "R": 12},
    # D-48: direct with room for a complete 40-step plan; beam generating only ~R actions (prefix parse)
    {"planner": "direct", "K": 1, "M": 256},
    {"planner": "beam", "K": 4, "W": 4, "E": 24, "R": 12, "M": 80, "P": 1},
    {"planner": "beam", "K": 8, "W": 4, "E": 24, "R": 12, "M": 80, "P": 1},
]  # R = rollout cap (actions simulated per candidate; D-47); M = proposer token cap; P = accept truncated prefixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa", required=True)
    ap.add_argument("--fwd-state")
    ap.add_argument("--fwd-delta")
    ap.add_argument("--sets", default="val_large,val_v2")
    ap.add_argument("--n-val-v2", type=int, default=50)
    ap.add_argument("--n-val-large", type=int, default=30, help="stratified by block count (every k-th instance)")
    ap.add_argument("--max-batch-tokens", type=int, default=5000, help="F decode budget on the GPU")
    ap.add_argument("--prop-batch-tokens", type=int, default=6000, help="proposer sampling budget on the GPU")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--configs", default="", help="comma-separated GRID indices (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: first N instances of every set")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "select.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add((r["config_id"], r["set"]))
            except Exception:
                pass
    from peft import PeftModel
    base, tok = lm.load_core("titan-sparse")
    model = PeftModel.from_pretrained(base, str(ROOT / a.pa), adapter_name="pa", is_trainable=False)
    fwds = {}
    for fmt, d in (("state", a.fwd_state), ("delta", a.fwd_delta)):
        if d:
            model.load_adapter(str(ROOT / d), adapter_name=f"fwd_{fmt}", is_trainable=False)
            fwds[fmt] = BP.LMForwardBW(model, tok, f"fwd_{fmt}", fmt=fmt, cache=out / f"fcache_{fmt}.jsonl",
                                       max_batch_tokens=a.max_batch_tokens)
    model.eval()
    sets = {s: load_set(s, a.limit or (a.n_val_v2 if s == "val_v2" else 0)) for s in a.sets.split(",")}
    if "val_large" in sets and not a.limit and a.n_val_large < len(sets["val_large"]):
        vl = sorted(sets["val_large"], key=lambda it: (len(it["objs"]), it["id"]))  # every k-th: keeps the size mix
        step = len(vl) / a.n_val_large
        sets["val_large"] = [vl[int(i * step)] for i in range(a.n_val_large)]
    grid = [GRID[int(i)] for i in a.configs.split(",")] if a.configs else GRID
    prog = open(out / "progress.txt", "a", encoding="utf-8")
    for cfg in grid:
        for fmt in (fwds if cfg["planner"] != "direct" else ["none"]):
            cid = json.dumps({**cfg, "fwd": fmt}, sort_keys=True)
            for sname, inst in sets.items():
                if (cid, sname) in done:
                    continue
                t0 = time.time()
                prop = BP.LMProposerBW(model, tok, "pa", "actions", K=cfg["K"], max_new=cfg.get("M"),
                                       prefix_ok=bool(cfg.get("P")), max_batch_tokens=a.prop_batch_tokens)
                if cfg["planner"] == "direct":
                    eps = BP.plan_direct(inst, prop, eval_seed=a.eval_seed)
                    nq = 0
                else:
                    f = CountingF(fwds[fmt])
                    eps = BP.plan_beam(inst, prop, f, eval_seed=a.eval_seed, width=cfg["W"], max_expansions=cfg["E"],
                                       max_rollout=cfg.get("R", 0),
                                       log=lambda m: (prog.write(f"{time.strftime('%H:%M:%S')} {sname} {cid} {m}\n"),
                                                      prog.flush()))
                    nq = len(f.keys)
                summ, rows = BP.score(inst, eps)
                by = {}
                for it, r in zip(inst, rows):
                    by.setdefault(len(it["objs"]), []).append(r["success"])
                rec = {"config_id": cid, "set": sname, "n": len(inst), "success": summ["success"],
                       "verified_rate": summ["verified_rate"], "verification_precision": summ["verification_precision"],
                       "f_queries_per_instance": round(nq / len(inst), 1),
                       "proposer_tokens_per_instance": round(sum(e["gen_tokens"] for e in eps) / len(inst), 1),
                       "expansions_per_instance": round(sum(e.get("expansions", 0) for e in eps) / len(inst), 2),
                       "success_by_blocks": {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())},
                       "wall_s": round(time.time() - t0, 1)}
                with open(res_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
                with open(out / f"plans_{len(done)}_{sname}.jsonl", "w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps({"config_id": cid, **r}) + "\n")
                done.add((cid, sname))


if __name__ == "__main__":
    main()
