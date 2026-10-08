import sys, itertools
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import data
sys.path.insert(0, str(ROOT / "scripts"))
from mechanism_report import cka, shift
sp, su = data.Arm("bank_pilot", "sparse"), data.Arm("bank_pilot", "supa")
it = su.items
def ids(splits, fmts, fams=None):
    return [x["item_id"] for x in it if x["split"] in splits and (("one" if x["format"] == "one" else f"k{x['k']}") in fmts) and (fams is None or x["family"] in fams)]
target12, target24 = 0.16451812032975255, 0.7456655686933179
best = []
for splits in (("test",), ("dev",), ("train",), ("dev", "test"), ("train", "dev", "test")):
    for r in range(1, 5):
        for fmts in itertools.combinations(("one", "k1", "k2", "k4"), r):
            I = ids(splits, fmts)
            if len(I) > 6000: continue
            A = su.X_last(I, 12).astype(np.float64); B = sp.X_last(I, 12).astype(np.float64)
            v12 = np.linalg.norm(B - A) / np.linalg.norm(A)
            if abs(v12 - target12) < 0.002:
                A = su.X_last(I, 24).astype(np.float64); B = sp.X_last(I, 24).astype(np.float64)
                v24 = np.linalg.norm(B - A) / np.linalg.norm(A)
                print(splits, fmts, len(I), round(v12, 5), round(v24, 5))
