import json, random, sys
from pathlib import Path
REPO = Path(sys.argv[1]); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).parent))
from bwmn import blocks as B
import blocks_val_large as VL
from canon import canon
assert "torch" not in sys.modules
FS = lambda xs: frozenset(map(tuple, xs))
hard = {p.stem: B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")}
hc = {canon(h[1], h[2], h[0]): k for k, h in hard.items()}
hinv = {}
for k, h in hard.items(): hinv.setdefault(VL.invariant(*h), []).append(k)
other = set()
for fn in ("instances_val_large.json", "instances_mirror_hard.json"):
    for r in json.loads((REPO / "runs/BW_val_large" / fn).read_text()):
        other.add(VL.invariant(r["objs"], FS(r["init"]), FS(r["goal"])))
rng = random.Random(20261010); sizes = [6, 7, 8, 8, 9, 9, 10, 10]; sigs = B.hard_signatures()
for i in range(240):
    n = sizes[i % 8]
    while True:
        g = B.generate(rng, n, full_goal_frac=1.0)
        if g is None: continue
        o, s, gl = g
        if n <= 7 and B.signature(o, s, gl) in sigs: print("sig-reject", i); continue
        v = VL.invariant(o, s, gl)
        if v in hinv:
            print(f"slot {i}: rejected by hard invariant {v} (hard {hinv[v]}); exact isomorph of a hard instance: {canon(s, gl, o) in hc}")
            continue
        if v in other: continue
        break
