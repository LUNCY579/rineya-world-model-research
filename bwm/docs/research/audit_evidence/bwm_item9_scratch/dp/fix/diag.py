import sys, hashlib, random, tempfile
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT / "tests"))
import test_domain_planner as T
T.BP.lm.greedy_shared_prefix = T.fake_greedy_shared_prefix
T.BP.lm.sample_crn_shared = T.fake_sample_crn_shared
L, DP = T.L, T.DP
rng = random.Random(T.LG_SEED)
d = Path(tempfile.mkdtemp())
inst = []
for n in (1, 2, 2, 3, 3):
    while True:
        g = L.generate(rng, n)
        if g is None: continue
        p = L.astar_plan(g[1], g[2], max_expansions=20_000)
        if p is not None and len(p) >= 4: break
    inst.append({"id": f"lg{len(inst)}", "objs": g[0], "init": g[1], "goal": g[2], "n_packages": n, "opt": len(p)})
exact = T.lg_forward_handler()
def slippy(prompt, seed):
    text = exact(prompt, seed)
    r = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) % 25
    return (T.DEGENERATE[0], T.DEGENERATE[2])[r] if r < 2 else (T.alias_slip(text) if r < 7 else text)
tok = T.FakeTok()
model = T.FakeLM({"fwd": slippy, "pa": T.lg_proposer_handler(rate=0.5)})
fwd = DP.LMForwardD(model, tok, "fwd", dom=L)
prop = DP.LMProposerD(model, tok, "pa", K=4, max_new=184, prefix_ok=True, dom=L)
eps = DP.plan_beam(inst, prop, fwd, eval_seed=0, width=4, max_expansions=24, max_depth=100, max_rollout=12)
summ, rows = DP.score_d(inst, eps, L)
slips = {k: slippy(L.forward_prompt_delta(*k), None) for k in fwd.memo}
print(summ, [(e["fail"], e["verified"]) for e in eps])
print("memo", len(fwd.memo), "None", sum(v is None for v in fwd.memo.values()),
      "degenerate", sum(x in T.DEGENERATE for x in slips.values()),
      "alias", sum('"add":["in p' in x for x in slips.values()))
