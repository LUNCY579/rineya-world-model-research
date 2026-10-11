import sys
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
gt = L.planbench_gt()
p = L.planbench_path(2)
G = L.plan_to_pddl(gt[2])
print(repr(G))
for t in (G, "Here is my plan:\n" + G, G + "(fly-airplane a0 l0-0 l1-0\n", G.replace("\n", " and\n", 1)):
    r = L.val_run(p, t)
    print("=====", repr(t[:60]), r["returncode"], r["valid"])
    print(r["stdout"])
