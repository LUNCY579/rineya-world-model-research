"""Early sanity probe of a Blocksworld proposer from a training checkpoint (ckpt.pt written by lm.train_sft).
Rebuilds the LoRA (same config as train_generic) on the frozen core, loads the checkpoint's trainable tensors, and runs
the no-F direct planner on a few generated validation instances (never PlanBench). Diagnostic only: nothing here
is used for selection.

Usage: python scripts/bw_ckpt_probe.py --ckpt runs/BW_Pa_v2_s1/ckpt/ckpt.pt --n 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import blocks as B  # noqa: E402
from bwmn import blocks_planner as BP  # noqa: E402
from bwmn import lm  # noqa: E402
from blocks_eval import load_set  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--beam", type=int, default=0, help="also run plan_beam with BW_F on this many val_large instances")
    ap.add_argument("--fwd", default="runs/BW_F_s0/adapter")
    a = ap.parse_args()
    import torch
    st = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    model, tok = lm.load_core("titan-sparse")
    pm = lm.attach_lora(model, seed=1)
    named = dict(pm.named_parameters())
    with torch.no_grad():
        for n, t in st["params"].items():
            named[n].copy_(t)
    pm.eval()
    print(f"checkpoint step {st['step']}/{st['steps']}", flush=True)
    if a.beam:
        import time
        pm.load_adapter(str(ROOT / a.fwd), adapter_name="fwd", is_trainable=False)
        pm.eval()
        inst = load_set("val_large", 0)
        inst = sorted(inst, key=lambda it: (len(it["objs"]), it["id"]))[:: max(1, len(inst) // a.beam)][:a.beam]
        f = BP.LMForwardBW(pm, tok, "fwd")
        prop = BP.LMProposerBW(pm, tok, "default", "actions", K=4)
        t0 = time.time()
        eps = BP.plan_beam(inst, prop, f, width=4, max_expansions=24,
                           log=lambda m: print("   ", time.strftime("%H:%M:%S"), m, flush=True))
        summ, rows = BP.score(inst, eps)
        print("BEAM val_large", json.dumps({"success": summ["success"], "verified": summ["verified_rate"],
                                            "precision": summ["verification_precision"], "f_queries": f.model_queries,
                                            "by_n": [(len(it["objs"]), r["success"], r["len"]) for it, r in zip(inst, rows)],
                                            "wall_s": round(time.time() - t0)}), flush=True)
        return
    for sname, n in (("val_v2", a.n), ("val_large", a.n // 2)):
        inst = load_set(sname, 0)
        inst = sorted(inst, key=lambda it: (len(it["objs"]), it["id"]))[:: max(1, len(inst) // n)][:n]
        prop = BP.LMProposerBW(pm, tok, "default", "actions", K=1)
        eps = BP.plan_direct(inst, prop)
        summ, rows = BP.score(inst, eps)
        parsed = sum(e["fail"] != "unparseable" for e in eps) / len(eps)
        by = {}
        for it, r in zip(inst, rows):
            by.setdefault(len(it["objs"]), []).append(r["success"])
        print(sname, json.dumps({"success": summ["success"], "parsed": parsed,
                                 "by_blocks": {k: round(sum(v) / len(v), 2) for k, v in sorted(by.items())}}), flush=True)
        print("  example:", rows[0]["plan"][:12], flush=True)


if __name__ == "__main__":
    main()
