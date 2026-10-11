"""Does the hard set change WHICH 240 distillation problems were generated? Replay VL.make's RNG stream for
bw_distill.problems(240) and attribute every rejected candidate to its exclusion source. Also: F training data range,
H2/B2 shard membership."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from bwmn import blocks as B  # noqa: E402
import blocks_val_large as VL  # noqa: E402

assert "torch" not in sys.modules
FS = lambda xs: frozenset(map(tuple, xs))  # noqa: E731
hard = [B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")]
hard_sigs = B.hard_signatures()
hard_inv = {VL.invariant(*h) for h in hard}
other_inv = set()
for fn in ("instances_val_large.json", "instances_mirror_hard.json"):
    for r in json.loads((REPO / "runs/BW_val_large" / fn).read_text()):
        other_inv.add(VL.invariant(r["objs"], FS(r["init"]), FS(r["goal"])))
rng = random.Random(20261010)
sizes = [6, 7, 8, 8, 9, 9, 10, 10]
rej = Counter()
accepted = []
for i in range(240):
    n = sizes[i % len(sizes)]
    while True:
        g = B.generate(rng, n, full_goal_frac=1.0)
        if g is None:
            rej["none"] += 1
            continue
        objs, init, goal = g
        if n <= 7 and B.signature(objs, init, goal) in hard_sigs:
            rej["hard_signature"] += 1
            continue
        v = VL.invariant(objs, init, goal)
        if v in hard_inv or v in other_inv:
            rej["hard_invariant" if v in hard_inv else "val_large_or_mirror_invariant"] += 1
            if v in hard_inv and v in other_inv:
                rej["(both)"] += 1
            continue
        accepted.append((objs, init, goal))
        break
sols = [json.loads(l) for l in (REPO / "runs/BW_DISTILL_r1/solutions.jsonl").read_text(encoding="utf-8").splitlines()]
same = all(a[0] == s["objs"] and a[1] == FS(s["init"]) and a[2] == FS(s["goal"]) for a, s in zip(accepted, sols))
print("rejections by reason:", dict(rej), "| replay equals solutions.jsonl:", same, len(accepted))

# F training data (BW_Fdelta_s0 rows) block range
fr = Counter()
for l in (REPO / "runs/BW_data/forward_delta_train.jsonl").open(encoding="utf-8"):
    fr[json.loads(l).get("n_blocks")] += 1
print("F (BW_Fdelta_s0) training rows by n_blocks:", dict(sorted(fr.items())))
print("BW_data v1 META:", (REPO / "runs/BW_data/META.json").read_text().replace("\n", " "))

# H2 vs B2 shard membership
for i in range(8):
    a = [json.loads(l)["id"] for l in (REPO / f"runs/BW_B3_HARD/h2_beam/shard_{i}of8/plans.jsonl").open()]
    b = [json.loads(l)["id"] for l in (REPO / f"runs/BW_TEST_B2/hard_beam/shard_{i}of8/plans.jsonl").open()]
    assert a == b, i
print("H2 and B2 hard_beam shards contain identical instance ids in identical order: ok")
