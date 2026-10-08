"""Audit 2: recompute hidden states on CPU (FP32) for a handful of items and compare with stored rows.
Confirms row<->prompt alignment, LAYERS<->hidden_states indexing, event-boundary positions."""
import json, sys, os
from pathlib import Path
import numpy as np, torch
assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import bank as bankmod, capture, data, runtime
torch.set_num_threads(8)
arm = sys.argv[1]
bank = data.load_bank("bank_pilot")
items = bankmod.items(bank)
tok = runtime.tokenizer()
A = data.Arm("bank_pilot", arm)
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
fams = {}
for i, it in enumerate(items):
    fams.setdefault((it["family"], it["format"], it["k"]), []).append(i)
pick = [int(rng.choice(v)) for kk, v in sorted(fams.items())]
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(str(runtime.PATHS[arm]), dtype=torch.float32, local_files_only=True,
                                         low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
print("fingerprint", runtime.param_fingerprint(m)[:12])
worst = {}
for i in pick:
    it = items[i]
    e = capture.encode_item(tok, it)
    with torch.no_grad():
        hs = m(input_ids=torch.tensor([e["ids"]]), output_hidden_states=True, use_cache=False).hidden_states
    assert len(hs) == 29, len(hs)
    stored = A.X_last([it["item_id"]], 28)  # sanity path through Arm
    row = A.pos[it["item_id"]]
    for li, L in enumerate(A.layers):
        ref = hs[L][0, e["last"]].numpy()
        got = np.asarray(A.last[row, li]).astype(np.float32)
        rel = np.abs(ref - got).max() / (np.abs(ref).max() + 1e-9)
        # also: is the stored row closer to ANY other layer index? (layer off-by-one check)
        others = [np.abs(hs[L2][0, e["last"]].numpy() - got).max() / (np.abs(ref).max() + 1e-9) for L2 in range(29)]
        worst.setdefault("last", []).append(rel)
        if int(np.argmin(others)) != L:
            print("  LAYER MISMATCH", it["item_id"], L, "best", int(np.argmin(others)))
    if e["events"] and it["item_id"] in A.ev_pos:
        r = A.ev_pos[it["item_id"]]
        for j, p in enumerate(e["events"]):
            for li, L in enumerate(A.layers):
                ref = hs[L][0, p].numpy()
                got = np.asarray(A.events[r, j, li]).astype(np.float32)
                worst.setdefault("event", []).append(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-9))
            # token at boundary
        print("  event tokens:", [tok.decode([e["ids"][p]]) for p in e["events"]])
    print(it["item_id"], it["format"], it["k"], "max rel err last:", f"{max(worst['last'][-8:]):.2e}",
          "| answer-slot token:", repr(tok.decode(e["ids"][-6:])))
print({k: f"{max(v):.2e}" for k, v in worst.items()})
# Alignment discrimination: compare stored row of item i vs recomputed state of a DIFFERENT item of same parent
it = items[pick[0]]; sib = [j for j, x in enumerate(items) if x["parent_id"] == it["parent_id"] and x["item_id"] != it["item_id"] and x["format"] == it["format"] and x["k"] == it["k"]][0]
e = capture.encode_item(tok, items[sib])
with torch.no_grad():
    hs = m(input_ids=torch.tensor([e["ids"]]), output_hidden_states=True, use_cache=False).hidden_states
ref = hs[24][0, e["last"]].numpy(); got = np.asarray(A.last[A.pos[it["item_id"]], 6]).astype(np.float32)
print("sibling-item rel err at L24 (should be >> fp16 error):", f"{np.abs(ref-got).max()/np.abs(ref).max():.2e}")
