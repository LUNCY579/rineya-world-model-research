import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["RCWM_THREADS"] = "2"
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1"
sys.path.insert(0, ROOT)
import numpy as np, psutil
from rcwmt import heads
D = int(sys.argv[1]); kind = sys.argv[2]
p = psutil.Process()
Xte = np.random.default_rng(0).standard_normal((1200, D)).astype(np.float32)
Yte = np.random.default_rng(1).standard_normal((1200, 12)).astype(np.float32)
G = np.random.default_rng(2).standard_normal((256, 12)).astype(np.float32)
b2 = heads.ScalarHead(kind)
b2.std = heads.Standardizer(Xte[:200]); b2.m = b2._model(D, 12); b2.m.eval()
base = p.memory_info().rss
expand = lambda X, Y, G: (np.repeat(X, len(G), 0), np.tile(G, (len(X), 1)), (Y @ G.T).reshape(-1))
Hte, Wte, _ = expand(Xte, Yte, G)
out = b2.predict(Hte, Wte)
mi = p.memory_info()
print(D, kind, "Hte MB", Hte.nbytes / 2**20, "peak_wset-base MB", (mi.peak_wset - base) / 2**20)
