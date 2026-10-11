import sys
sys.path.insert(0, ".")
from fuzz_reader import val_listing, L
p = L.planbench_path(2)
G = L.plan_to_pddl(L.planbench_gt()[2])
for frag in ["(load-truc k p0 t1 l1-0)", "(unl oad-truck p0 t1 l1-0)", "(unl%oad-truck p0 t1 l1-0)", "(x oad p0)", "(oad p0)",
             "(x k)", "(k)", "(unl x)", "(x unl)", "(load-truc'k p0 t1 l1-0)", "(foo-truck p0)", "(oad-truck p0)", "(x oad-truck)",
             "(x y)", "(x y z)", "(load-truck p0 t1 l1-0 k)"]:
    for pos in ("lead", "trail"):
        t = frag + "\n" + G if pos == "lead" else G + frag + "\n"
        v = L.val_run(p, t, verbose=True)
        vl = val_listing(v["stdout"])
        ours = L.read_val_plan(t)
        print(f"{frag:32s} {pos:5s} VAL {'bad' if vl is None else ('other' if vl == 'other' else len(vl))} rc {v['returncode']} ours {None if ours is None else len(ours)}")
