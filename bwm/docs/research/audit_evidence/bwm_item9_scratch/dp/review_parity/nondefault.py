import sys, json, random, types, hashlib
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
src = Path(__file__).with_name("parity_scale.py").read_text(encoding="utf-8")
# reuse the stub + handlers only (everything before the Blocksworld snapshot section)
exec(src.split("for f in SCR.glob(\"wl_*.jsonl\"):")[0])
from bwmn import domain_planner as DP
def rand_handler(mode):
    """Cheap deterministic outputs depending on (prompt, seed): random plans over the state's blocks."""
    def h(prompt, seed):
        user = prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        d = json.loads(user)
        s = B.state_from_json(d["state"])
        objs = B.blocks_of(s)
        r = random.Random(hashlib.sha256(f"{prompt}|{seed}".encode()).hexdigest())
        acts = [r.choice(B.ground_actions(objs)) for _ in range(r.randint(1, 70 if mode == "actions" else 6))]
        if mode == "actions":
            t = B.target_actions(acts)
        else:
            steps, st = [], s
            for a in acts:
                st = B.step(st, a)[0]
                steps.append((a, st))
            t = B.target_states(steps)
        k = r.randrange(9)
        if k == 0:
            return t[:r.randrange(1, len(t))]
        if k == 1:
            return t.replace(acts[0][0], "teleport", 1)
        if k == 2:
            return B.target_actions(acts) if mode == "states" else "Sorry."
        return t
    return h


RH = rand_handler
tok = Tok()
items = []
rng = random.Random(3)
while len(items) < 300:
    g = B.generate(rng, rng.randint(3, 9), full_goal_frac=0.3)
    if g:
        items.append({"id": f"i{len(items)}", "objs": g[0], "goal": g[2], "state": g[1], "round": f"{len(items)%3}.0"})
res = []
for mode, pok in (("actions", True), ("states", True), ("states", False)):
    mb, md = Model({"pa_lg": RH(mode)}), Model({"pa_lg": RH(mode)})
    pb = BP.LMProposerBW(mb, tok, "pa_lg", mode, K=3, temperature=0.7, top_p=0.8, max_new=150, prefix_ok=pok)
    pd = DP.LMProposerD(md, tok, "pa_lg", mode, K=3, temperature=0.7, top_p=0.8, max_new=150, prefix_ok=pok, dom=B)
    ob, od = pb.propose(items, 4), pd.propose(items, 4)
    res.append((mode, pok, ob == od, mb.log == md.log, mb.alog == md.alog, {x[5:7] for x in md.log} if False else sorted({(x[5], x[6]) for x in md.log}),
                sum(c["plan"] is not None for cs in od for c in cs)))
mb, md = Model({"fwd_delta": fwd_handler()}), Model({"fwd_delta": fwd_handler()})
keys = [(it["state"], a) for it in items[:100] for a in B.ground_actions(it["objs"])[:10]]
fb = BP.LMForwardBW(mb, tok, "fwd_delta", fmt="delta")
fd = DP.LMForwardD(md, tok, "fwd_delta", dom=B)
res.append(("fwd", fb.predict(keys) == fd.predict(keys), mb.log == md.log, mb.alog, md.alog))
print(json.dumps(res, default=str))
