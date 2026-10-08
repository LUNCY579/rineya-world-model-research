"""Audit 10: (a) share of answer-slot variance that is between-family (drives pooled CKA / pooled SD);
CKA after removing family means. (b) which weight tensors differ between Sparse and Supa, and the delta norm."""
import sys, re
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import data, runtime
sys.path.insert(0, str(ROOT / "scripts"))
from mechanism_report import cka
sp, su = data.Arm("bank_pilot", "sparse"), data.Arm("bank_pilot", "supa")
items = [it for it in su.items if it["split"] == "test" and it["format"] == "one"]
ids = [it["item_id"] for it in items]; fam = np.array([it["family"] for it in items])
for L in (12, 16, 20, 24, 28):
    A = su.X_last(ids, L).astype(np.float64); B = sp.X_last(ids, L).astype(np.float64)
    Ac, Bc = A.copy(), B.copy()
    for f in np.unique(fam):
        Ac[fam == f] -= A[fam == f].mean(0); Bc[fam == f] -= B[fam == f].mean(0)
    tot = ((A - A.mean(0)) ** 2).sum(); within = (Ac ** 2).sum()
    print(f"L{L}: between-family share of Supa variance {1 - within / tot:.2f} | pooled CKA {cka(A, B):.3f} | CKA after removing family means {cka(Ac, Bc):.3f}")
from safetensors import safe_open
fa = safe_open(str(runtime.PATHS["supa"] / "model.safetensors"), framework="pt", device="cpu")
fb = safe_open(str(runtime.PATHS["sparse"] / "model.safetensors"), framework="pt", device="cpu")
changed, dn2, bn2, blocks = 0, 0.0, 0.0, set()
for k in fa.keys():
    a = fa.get_tensor(k).double(); b = fb.get_tensor(k).double()
    d = b - a
    bn2 += float((a ** 2).sum())
    if torch.count_nonzero(d):
        changed += 1; dn2 += float((d ** 2).sum())
        m = re.search(r"layers\.(\d+)\.", k)
        blocks.add(int(m.group(1)) if m else k)
print("dtypes", fa.get_tensor("model.norm.weight").dtype, fb.get_tensor("model.norm.weight").dtype)
print("changed tensors", changed, "blocks touched", sorted(x for x in blocks if isinstance(x, int)), [x for x in blocks if not isinstance(x, int)])
print(f"delta rel norm {np.sqrt(dn2 / bn2) * 100:.4f}%")
