import sys
sys.path.insert(0, ".")
from fuzz_reader import val_listing, L
p = L.planbench_path(218)
text = "(drive-truck t0 l0-2 l0-0 c0)\n(load-truck p2 t0 l0-0)\n(drive-truck t0 l0-0 l0-1 c0)\n(unload-truck p2 t0 l0-1)\n(drive-truck t2 l2-0 l2-2 c2)\n(load-truck p4 #t2 l2-2)\n(load-truc'k p3 t2 \rl2-2)\n(load-truck p1 t2 l2-2)\n(drive-truck t2 l2-2 l2-0 c2)\n(unload-truck p4 t2 l2-0)\n(unl%oad-truck p1 t2 l2-0)\n(load-airplane p4 a1 l2-0)"
def kind(t):
    v = L.val_run(p, t, verbose=True); vl = val_listing(v["stdout"])
    return "bad" if vl is None else ("other" if vl == "other" else len(vl)), v["stdout"][:200].replace("\r", "")
print(kind(text))
lines = text.split("\n")
for k in range(len(lines)):
    print(k, ascii(lines[k]), kind("\n".join(lines[:k] + lines[k + 1:]))[0])
for t in ["(load-truck p0 t1 \rl1-0)\n", "(load-truck p0 t1 \r l1-0)\n", "(x\ry)\n", "(load-truck p0 t1\rl1-0)\n"]:
    print(ascii(t), kind(L.plan_to_pddl(L.planbench_gt()[218]) + t))
