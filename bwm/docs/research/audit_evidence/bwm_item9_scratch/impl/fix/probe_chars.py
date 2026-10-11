import sys, json
sys.path.insert(0, ".")
from valread import val_read, L
gt = L.planbench_gt()
p = L.planbench_path(2)
G = L.plan_to_pddl(gt[2])
lines = G.strip().split("\n")
res = {}
chars = [chr(c) for c in range(33, 127) if chr(c) not in "()"] + ["\r", "\t", "\u00a0", "\u2192", "\u00e9", "\f", "\v"]
for ch in chars:
    inside = G.replace("p0 a1", "p0 " + ch + " a1", 1)
    between = lines[0] + "\n" + ch + "\n" + "\n".join(lines[1:]) + "\n"
    trailing = G + ch + "\n"
    leading = ch + "\n" + G
    r = {}
    for k, t in (("inside", inside), ("between", between), ("trailing", trailing), ("leading", leading)):
        kind, acts, valid, rc = val_read(p, t)
        r[k] = (kind, None if acts is None else len(acts), valid, rc)
    res[ch] = r
    print(repr(ch), r, flush=True)
json.dump({repr(k): v for k, v in res.items()}, open("probe_chars.json", "w"), indent=0)
