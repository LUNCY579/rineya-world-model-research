import sys
sys.path.insert(0, ".")
from fuzz_reader import val_listing, L
p = L.planbench_path(2)
G = L.plan_to_pddl(L.planbench_gt()[2])
lines = G.strip().split("\n")
def kind(t):
    v = L.val_run(p, t, verbose=True); vl = val_listing(v["stdout"])
    return ("bad" if vl is None else ("other" if vl == "other" else len(vl))), v["valid"]
SK = ',"\'\{}|!#$%&@^`~'
for ch in SK:
    for nxt in ("t", "T", "a", "x", "1", "=", "-", "("):
        frag = ch + nxt
        inside = G.replace("p0 a1", "p0 " + frag + " a1", 1) if nxt not in "1=-(" else G.replace(" a1 ", " " + ch + "a1 ", 1)
        between = lines[0] + "\n" + frag + "\n" + "\n".join(lines[1:]) + "\n"
        ki, kb = kind(inside), kind(between)
        oi, ob = L.read_val_plan(inside), L.read_val_plan(between)
        tag = lambda o: "bad" if o is None else len(o)
        flag = "" if (ki[0] == tag(oi) or (ki[0] == "other")) and kb[0] == tag(ob) else "  <-- DIFF"
        if flag or ch == "#":
            print(f"{frag!r:6s} inside VAL {ki} ours {tag(oi)} | between VAL {kb} ours {tag(ob)}{flag}")
print("done")
