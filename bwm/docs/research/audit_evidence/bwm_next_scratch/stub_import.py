# Feasibility only: import the pinned planner without torch by stubbing bwmn.lm (no file is modified).
import sys, types, hashlib
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, R)
import bwmn
stub = types.ModuleType("bwmn.lm"); sys.modules["bwmn.lm"] = stub; bwmn.lm = stub
from bwmn import blocks_planner as BP, blocks as B
print("planner sha", hashlib.sha256(open(R + "/bwmn/blocks_planner.py", "rb").read()).hexdigest()[:8])
print("torch loaded:", "torch" in sys.modules)
# tiny smoke: OracleProposer + ExactF on a 5-block generated problem
import random
it = B.generate(random.Random(1), 5); print(type(it), (list(it.keys()) if isinstance(it, dict) else len(it)))
