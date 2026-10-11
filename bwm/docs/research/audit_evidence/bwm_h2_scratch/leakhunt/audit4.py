import json, sys
from pathlib import Path
from collections import Counter
REPO = Path(sys.argv[1]); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(REPO / "scripts"))
from bwmn import blocks as B
from canon import goal_canon_chain, goal_canon_n
import blocks_val_large as VL
FS = lambda xs: frozenset(map(tuple, xs))
hard = {p.stem: B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")}
sols = [json.loads(l) for l in (REPO / "runs/BW_DISTILL_r1/solutions.jsonl").open()]
dc = {goal_canon_chain(FS(r["goal"])) for r in sols}
dn = Counter(goal_canon_n(FS(r["goal"]), r["objs"]) for r in sols)
hits = sorted((len(h[0]), k, VL.goal_towers(h[2], h[0])) for k, h in hard.items() if goal_canon_chain(h[2]) in dc)
print("chain-only hits by n:", Counter(n for n, _, _ in hits))
print("n>10 chain-only hits:", [(k, n, t) for n, k, t in hits if n > 10])
nh = sorted((len(h[0]), k, VL.goal_towers(h[2], h[0]), dn[goal_canon_n(h[2], h[0])]) for k, h in hard.items() if goal_canon_n(h[2], h[0]) in dn)
print("with-n hits (n, id, goal towers, #distill problems sharing):", nh)
print("hard n<=10 without with-n hit:", sorted((len(h[0]), k, VL.goal_towers(h[2], h[0])) for k, h in hard.items() if len(h[0]) <= 10 and goal_canon_n(h[2], h[0]) not in dn))
