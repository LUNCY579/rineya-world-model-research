import sys, subprocess, tempfile
from pathlib import Path
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
gt = L.planbench_gt()
p = L.planbench_path(2)
G = L.plan_to_pddl(gt[2])
t = G.replace("\n", " and\n", 1)
with tempfile.TemporaryDirectory() as d:
    pp = Path(d) / "plan.txt"; pp.write_text("0: (load-airplane p0 a1 l1-0)\n0: (fly-airplane a1 l1-0 l0-0)\n5: (unload-airplane p0 a1 l0-0)\n")
    r = subprocess.run([str(L.VAL_EXE), "-v", str(L.DOMAIN_PDDL), str(p), str(pp)], capture_output=True, text=True)
    print(r.returncode); print(r.stdout)
