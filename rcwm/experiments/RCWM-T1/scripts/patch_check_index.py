"""Check for run_patch_cpu.py: the hook on model.model.layers[L-1] captures hidden_states[L].

Loads Supa on CPU (same recipe as run_patch_cpu.py), takes the first 2 smoke items per family as
one left-padded batch per family, and compares (i) run_patch_cpu.capture()'s block-(L-1) output
during generate's prefill with (ii) a plain forward's output_hidden_states[L] (generate's
position_ids), at real positions, for L in 8, 16, 24. Also checks the answer-slot state against the
captured arm data used by R5/run_steer offsets (data.Arm X_last) when the item is in that capture.

  python scripts/patch_check_index.py
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RCWM_THREADS"] = "3"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "3"
sys.argv = [sys.argv[0], "--threads", "3"]

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_patch_cpu as P  # noqa: E402
from rcwmt import bank as bankmod, data, env, runtime  # noqa: E402


def main():
    torch.set_num_threads(3)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    ROOT = P.ROOT
    bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
    pool = [it for it in bankmod.items(bank) if it["split"] == "test" and it["format"] == "one"]

    class A:
        seed, per_family = 17, 2
        items_from = str(ROOT / "runs" / "bank_pilot" / "steer_cpu" / "steer_cpu_results.json")
    items = P.select_items(A, pool)
    tok = runtime.tokenizer()
    model, _ = P.load("supa")
    layers = [8, 16, 24]
    try:
        arm = data.Arm("bank_pilot", "supa")
        arm_ids = {it["item_id"] for it in arm.items}
    except Exception as e:  # noqa: BLE001
        print("arm data unavailable:", e)
        arm, arm_ids = None, set()
    worst = 0.0
    for b in P.batches_of(items, 16):
        cap = P.capture(model, tok, b, layers)
        ids, att = cap["ids"], cap["att"]
        pos = att.long().cumsum(-1) - 1
        pos = pos.masked_fill(att == 0, 0)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=att, position_ids=pos, output_hidden_states=True, use_cache=False)
        m = att.bool()
        for L in layers:
            d = float((cap["H"][L][m] - out.hidden_states[L][m]).abs().max())
            dn = float((cap["H"][L][m] - out.hidden_states[L + 1][m]).abs().max())
            worst = max(worst, d)
            print(f"{b[0]['family'][:8]} L{L}: max|hook(block {L - 1}) - hidden_states[{L}]| = {d:.3g}   "
                  f"(vs hidden_states[{L + 1}]: {dn:.3g})")
            if arm is not None:
                for r, it in enumerate(b):
                    if it["item_id"] in arm_ids and L in arm.layers:
                        x = arm.X_last([it["item_id"]], L)[0].astype(np.float32)
                        e = float(np.abs(cap["H"][L][r, -1].numpy() - x).max())
                        print(f"    {it['item_id']} answer slot vs captured arm data L{L}: max|d| {e:.3g} "
                              f"(rel {e / np.abs(x).max():.2e})")
                        break
    print("PASS" if worst < 1e-3 else "FAIL", "worst", worst)


if __name__ == "__main__":
    main()
