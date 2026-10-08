import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["RCWM_THREADS"] = "2"
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1"
sys.path.insert(0, ROOT)
import numpy as np
from rcwmt import data, bank as bankmod, capture

def fake_arm(bank_stem, tag):
    b = data.load_bank(bank_stem)
    its = bankmod.items(b)
    a = object.__new__(data.Arm)
    a.dir = tag; a.layers = list(capture.LAYERS)
    a.items = [{"item_id": it["item_id"]} for it in its]
    # encode (arm tag, row index, layer) into the vector
    n = len(its); L = len(a.layers)
    last = np.zeros((n, L, 4), np.float32)
    last[:, :, 0] = 1 if tag == "pilot" else 2
    last[:, :, 1] = np.arange(n)[:, None]
    last[:, :, 2] = np.array(a.layers)[None, :]
    a.last = last; a.events = None
    a.pos = {it["item_id"]: i for i, it in enumerate(a.items)}
    a.ev_pos = {}
    return a

P, C = fake_arm("bank_pilot", "pilot"), fake_arm("bank_confirm", "confirm")
U = data.ArmUnion([P, C])
rows_p = [r for r in data.candidate_table(data.load_bank("bank_pilot")) if r["split"] in ("train", "dev")]
rows_c = data.candidate_table(data.load_bank("bank_confirm"))
rng = np.random.default_rng(0)
ids = [r["item_id"] for r in rows_p[:300]] + [r["item_id"] for r in rows_c[:300]]
ids = [ids[i] for i in rng.permutation(len(ids))]
X = U.X_last(ids, 16)
ok = all((X[n, 0] == (1 if i in P.pos else 2)) and X[n, 1] == (P.pos[i] if i in P.pos else C.pos[i]) and X[n, 2] == 16 for n, i in enumerate(ids))
print("ArmUnion mixed lookup correct:", ok)
X = U.X_last([r["item_id"] for r in rows_c[:50]], 8)
print("confirm-only lookup from confirm part:", bool((X[:, 0] == 2).all()))
# pilot TEST ids must not be in rows used by confirm_analysis
pt = {r["item_id"] for r in data.candidate_table(data.load_bank("bank_pilot")) if r["split"] == "test"}
print("pilot test ids in pilot_rows:", len(pt & {r["item_id"] for r in rows_p}))
