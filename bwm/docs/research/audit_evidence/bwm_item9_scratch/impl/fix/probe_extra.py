import sys
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
p = L.planbench_path(2)
G = L.plan_to_pddl(L.planbench_gt()[2])
a, b, c = G.strip().split("\n")
cases = {
    "extra_declared": G.replace("a1 l1-0)", "a1 l1-0 c0)", 1),
    "extra_undeclared": G.replace("a1 l1-0)", "a1 l1-0 zzz)", 1),
    "extra_two": G.replace("a1 l1-0)", "a1 l1-0 c0 c1)", 1),
    "extra_on_fly": G.replace("(fly-airplane a1 l1-0 l0-0)", "(fly-airplane a1 l1-0 l0-0 c0)"),
    "extra_on_all": "".join(ln[:-1] + " p0)\n" for ln in G.strip().split("\n")),
    "extra_drive": "(drive-truck t1 l1-0 l1-0 c1 t0)\n" + G,
    "fewer_args": G.replace("a1 l1-0)", "a1)", 1),
}
for k, t in cases.items():
    v = L.val_run(p, t, verbose=True)
    print(f"{k:18s} VAL valid={v['valid']} rc={v['returncode']} listing={L.val_listing(v['stdout'])}")
    if k == "extra_undeclared": print(v["stdout"][:600])
