"""C4: does Sparse's own LM head read a COMPOSED prerequisite state from the k-event answer slot?
First answer digit (top digit token from the captured L28 state) vs truth and vs heuristic states
(copy s0, first event only, last event only, ignore preconditions), at k = 1, 2, 4."""
from __future__ import annotations

import json

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

import vh
from vh import fidelity

SP = "C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze"
with safe_open(SP + "/model.safetensors", "pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").float()
tok = AutoTokenizer.from_pretrained(SP)
digit_ids = [tok(str(d), add_special_tokens=False)["input_ids"][0] for d in range(10)]
Ed = E[digit_ids]
fam = "prerequisite_plan"
fd = lambda s: int(str(s)[0])
res = {}
for k in (1, 2, 4):
    rows = vh.split_rows(fam, k, "test") + vh.split_rows(fam, k, "dev")
    ids = [r["item_id"] for r in rows]
    truth = np.array([fd(r["states"][-1]) for r in rows])
    heur = {n: np.array([fd(fidelity.heuristic_states(fam, r["s0"], r["events"])[n]) for r in rows])
            for n in ("copy_s0", "first_only", "last_only", "ignore_pre")}
    contested = np.all([truth != h for h in heur.values()], axis=0)
    out = {"n": len(rows), "n_contested_first_digit": int(contested.sum())}
    for a in ("sparse", "supa"):
        H = torch.tensor(vh.arm(a).X_last(ids, 28))
        pred = (H @ Ed.T).argmax(-1).numpy()
        o = {"acc_truth_all": float((pred == truth).mean()),
             **{f"match_{n}_all": float((pred == h).mean()) for n, h in heur.items()}}
        if contested.any():
            o["acc_truth_contested"] = float((pred == truth)[contested].mean())
            o.update({f"match_{n}_contested": float((pred == h)[contested].mean()) for n, h in heur.items()})
        out[a] = o
    # chance reference: frequency of the most common true first digit
    vals, cnt = np.unique(truth[contested], return_counts=True) if contested.any() else (np.array([]), np.array([1]))
    out["majority_digit_rate_contested"] = float(cnt.max() / max(cnt.sum(), 1))
    res[f"k{k}"] = out
    print(k, json.dumps(out), flush=True)
(vh.ROOT / "docs" / "verification" / "c4_p_first_digit.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
