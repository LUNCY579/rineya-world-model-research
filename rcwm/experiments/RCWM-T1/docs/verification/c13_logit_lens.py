"""C1/C3: behavioural read-out of the captured layer-28 (post final-norm) answer-slot states through
the tied LM head (embed_tokens; blocks 8-27 are the only parameters the delta touches). CPU only.

  - first-token top-1 per arm, agreement and KL(Sparse || Supa)
  - prerequisite: the first answer token is the first digit of the final integer, i.e. CONTENT.
    First-digit accuracy per arm vs truth and vs copying s0, by applicability.
  - counterfactual alignment: Supa state + mean shift (input-independent), per-dim affine, full
    linear map -> does the first-token behaviour become Sparse-like?
"""
from __future__ import annotations

import json

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

import vh
from vh import env

SP = "C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze"
SU = "C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze"
OUT = vh.ROOT / "docs" / "verification" / "c13_logit_lens.json"
torch.set_num_threads(8)

with safe_open(SP + "/model.safetensors", "pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").float()
with safe_open(SU + "/model.safetensors", "pt") as f:
    Eu = f.get_tensor("model.embed_tokens.weight").float()
emb_maxdiff = float((E - Eu).abs().max())
del Eu
tok = AutoTokenizer.from_pretrained(SP)
print("embed max|diff| sparse vs supa:", emb_maxdiff, flush=True)
print("tokenize '146' ->", tok("146", add_special_tokens=False)["input_ids"], [tok.decode([t]) for t in tok("146", add_special_tokens=False)["input_ids"]])

A, B = vh.arm("supa"), vh.arm("sparse")
li = A.layers.index(28)


def logp(H):
    out = []
    for s in range(0, len(H), 64):
        z = torch.tensor(H[s:s + 64], dtype=torch.float32) @ E.T
        out.append(torch.log_softmax(z, -1))
    return torch.cat(out)


def ridge_map(Xtr, Ytr, lam=1e3):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = torch.tensor(Xtr - mx, dtype=torch.float64), torch.tensor(Ytr - my, dtype=torch.float64)
    W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(Xc.shape[1], dtype=torch.float64), Xc.T @ Yc).numpy()
    return lambda X: (X - mx) @ W + my


res = {"embed_maxdiff": emb_maxdiff}
for fam in vh.FAMS:
    for frame in ("one", 1):
        sp = vh.sp3(fam, frame)
        te, tr = sp["test"], sp["train"]
        ids_te, ids_tr = [r["item_id"] for r in te], [r["item_id"] for r in tr]
        Ha, Hb = A.X_last(ids_te, 28).astype(np.float64), B.X_last(ids_te, 28).astype(np.float64)
        Hatr, Hbtr = A.X_last(ids_tr, 28).astype(np.float64), B.X_last(ids_tr, 28).astype(np.float64)
        variants = {
            "supa": Ha, "sparse": Hb,
            "supa+mean_shift": Ha + (Hbtr - Hatr).mean(0),
            "supa_per_dim_affine": Hbtr.mean(0) + (Ha - Hatr.mean(0)) * (Hbtr.std(0) + 1e-6) / (Hatr.std(0) + 1e-6),
            "supa_linear_map": ridge_map(Hatr, Hbtr)(Ha),
            "sparse-mean_shift": Hb - (Hbtr - Hatr).mean(0),
        }
        LP = {k: logp(v) for k, v in variants.items()}
        top = {k: v.argmax(-1).numpy() for k, v in LP.items()}
        e = {}
        for k in variants:
            vals, cnt = np.unique(top[k], return_counts=True)
            o = np.argsort(-cnt)[:4]
            e[k] = {"top1_tokens": {repr(tok.decode([int(vals[i])])): int(cnt[i]) for i in o},
                    "top1_agree_with_sparse": float((top[k] == top["sparse"]).mean()),
                    "top1_agree_with_supa": float((top[k] == top["supa"]).mean())}
            pb = LP["sparse"].exp()
            e[k]["KL_sparse_to_this_mean"] = float((pb * (LP["sparse"] - LP[k])).sum(-1).mean())
        if fam == "prerequisite_plan":
            digit_ids = [tok(str(d), add_special_tokens=False)["input_ids"][0] for d in range(10)]
            truth = np.array([digit_ids[int(str(r["states"][-1])[0])] for r in te])
            copy = np.array([digit_ids[int(str(r["s0"])[0])] for r in te])
            appl = np.array([env.applicable(r["s0"], r["events"][0]) for r in te])
            diff_first = truth != copy
            for k in variants:
                dl = LP[k][:, digit_ids].argmax(-1).numpy()
                pred = np.array(digit_ids)[dl]
                e[k]["P_first_digit"] = {
                    "acc_truth_all": float((pred == truth).mean()), "acc_copy_all": float((pred == copy).mean()),
                    "acc_truth_applicable": float((pred == truth)[appl].mean()), "n_applicable": int(appl.sum()),
                    "acc_truth_inapplicable": float((pred == truth)[~appl].mean()),
                    "acc_truth_where_truth_first_digit!=s0_first_digit": float((pred == truth)[diff_first].mean()),
                    "acc_copy_where_differs": float((pred == copy)[diff_first].mean()),
                    "n_differs": int(diff_first.sum()),
                    "top1_is_digit": float(np.isin(top[k], digit_ids).mean())}
        res[f"{fam}|{frame}"] = e
        print(fam, frame, json.dumps(e), flush=True)
OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
