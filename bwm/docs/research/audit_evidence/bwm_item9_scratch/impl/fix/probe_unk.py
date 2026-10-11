import sys
sys.path.insert(0, ".")
from fuzz_reader import val_listing, L
p = L.planbench_path(2)
G = L.plan_to_pddl(L.planbench_gt()[2])
a, b, c = G.strip().split("\n")
cases = {"two_diff_unknown": G + "(foo p0)\n(bar p0)\n", "two_same_unknown": G + "(foo p0)\n(foo p0)\n",
         "one_unknown": G + "(foo p0)\n", "unknown_then_wrong_arity": G + "(foo p0)\n(load-truck p0)\n",
         "wrong_arity": G + "(load-truck p0)\n", "two_diff_unknown_arity_same": G + "(foo p0 t1)\n(bar p0 t1)\n",
         "unknown_mid": f"{a}\n(foo p0)\n{b}\n{c}\n", "two_unknown_mid": f"{a}\n(foo p0)\n{b}\n(bar x)\n{c}\n",
         "same_op_two_arities": G + "(foo p0)\n(foo p0 t1)\n"}
for k, t in cases.items():
    v = L.val_run(p, t, verbose=True)
    vl = val_listing(v["stdout"])
    print(f"{k:30s} VAL {'bad' if vl is None else ('other' if vl == 'other' else len(vl))} rc {v['returncode']} | {v['stdout'][:160]!r}")
