import sys, random
sys.path.insert(0, ".")
from fuzz_reader import cases, val_listing, L
rng = random.Random(2)
for iid, init, goal, text, tags in cases(rng, 1000):
    if iid == 218 and tags == ['skip'] * 4:
        v = L.val_run(L.planbench_path(iid), text, verbose=True)
        print(ascii(text))
        vl = val_listing(v["stdout"]); ours = L.read_val_plan(text)
        print("VAL rc", v["returncode"], "n", None if vl is None else len(vl))
        print("VAL ", vl)
        print("OURS", sorted(ours) if ours else ours)
        print(v["stdout"][:1500])
        break
