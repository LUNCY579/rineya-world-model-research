"""C1: P first-answer-digit read through the tied LM head from the captured L28 answer-slot state.
Parent-level bootstrap CIs for Sparse vs Supa (Supa read restricted to digit tokens), on items where
the true first digit differs from the start state's first digit (copying cannot be right there)."""
from __future__ import annotations

import json

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

import vh
from vh import env

SP = "C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze"
with safe_open(SP + "/model.safetensors", "pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").float()
tok = AutoTokenizer.from_pretrained(SP)
digit_ids = [tok(str(d), add_special_tokens=False)["input_ids"][0] for d in range(10)]
Ed = E[digit_ids]                                  # [10, 1024]
res = {}
fam = "prerequisite_plan"
for frame in ("one", 1):
    for split in ("test", "dev"):
        rows = vh.split_rows(fam, frame, split)
        ids = [r["item_id"] for r in rows]
        truth = np.array([int(str(r["states"][-1])[0]) for r in rows])
        copy = np.array([int(str(r["s0"])[0]) for r in rows])
        changed = np.array([r["states"][-1] != r["s0"] for r in rows])
        differs = truth != copy
        out = {"n": len(rows), "n_state_changed": int(changed.sum()), "n_first_digit_differs": int(differs.sum())}
        corr = {}
        for a in ("sparse", "supa"):
            H = torch.tensor(vh.arm(a).X_last(ids, 28))
            pred = (H @ Ed.T).argmax(-1).numpy()
            c = (pred == truth).astype(float)
            corr[a] = c
            out[a] = {"acc_all": float(c.mean()), "acc_state_changed": float(c[changed].mean()),
                      "acc_first_digit_differs": float(c[differs].mean()),
                      "copy_rate_first_digit_differs": float((pred == copy)[differs].mean())}
        rd = [r for r, d in zip(rows, differs) if d]
        rc = [r for r, d in zip(rows, changed) if d]
        out["diff_sparse_minus_supa|first_digit_differs"] = vh.paired_boot(rd, corr["sparse"][differs], corr["supa"][differs])
        out["diff_sparse_minus_supa|state_changed"] = vh.paired_boot(rc, corr["sparse"][changed], corr["supa"][changed])
        # Sparse vs copy baseline on differing items (copy is 0 there by construction)
        out["sparse_vs_copy|first_digit_differs"] = vh.paired_boot(rd, corr["sparse"][differs], np.zeros(differs.sum()))
        res[f"{frame}|{split}"] = out
        print(frame, split, json.dumps(out), flush=True)
(vh.ROOT / "docs" / "verification" / "c1_p_first_digit.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
