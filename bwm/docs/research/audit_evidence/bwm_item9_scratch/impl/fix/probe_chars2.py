import sys, json
sys.path.insert(0, ".")
from valread import val_read, L
gt = L.planbench_gt()
p = L.planbench_path(2)
G = L.plan_to_pddl(gt[2])
lines = G.strip().split("\n")
for ch in [chr(c) for c in range(ord("G"), ord("a"))] + ["→", "é", "\f", "\v", "\x00", "\x01", "\x7f"]:
    if ch.isalpha() and ch not in "GZ": continue
    inside = G.replace("p0 a1", "p0 " + ch + " a1", 1)
    between = lines[0] + "\n" + ch + "\n" + "\n".join(lines[1:]) + "\n"
    trailing = G + ch + "\n"
    leading = ch + "\n" + G
    r = {}
    for k, t in (("inside", inside), ("between", between), ("trailing", trailing), ("leading", leading)):
        kind, acts, valid, rc = val_read(p, t)
        r[k] = (kind, None if acts is None else len(acts), valid, rc)
    print(ascii(ch), r, flush=True)
# leading junk: which steps does VAL read?
for lead in ["*", "* *", "* * *", "Here is my plan:", "plan", "plan plan", "plan plan plan", "-", "=", ":", "?"]:
    kind, acts, valid, rc = val_read(p, lead + "\n" + G)
    print("LEAD", ascii(lead), kind, acts, valid)
