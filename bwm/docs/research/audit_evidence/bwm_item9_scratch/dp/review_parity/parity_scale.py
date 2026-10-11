"""Adversarial parity review of bwmn/domain_planner.py vs the pinned Blocksworld classes (scratch, read-only inputs).

Runs with a pure-python stand-in for bwmn.lm (no torch). Prints a JSON report.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import types
from pathlib import Path

ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
SCR = Path(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_item9_scratch/dp/review_parity")
sys.path.insert(0, str(ROOT))
PIN = "18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960"
REPORT = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


REPORT["bp_sha_before"] = sha(ROOT / "bwmn" / "blocks_planner.py")

# ---------------------------------------------------------------- stub lm (recording, deterministic)
LM = types.ModuleType("bwmn.lm")


class Tok:
    CH = 4

    def n(self, t):
        return -(-len(t) // self.CH)


class Model:
    def __init__(self, handlers):
        self.h, self.adapter, self.alog, self.log, self.outs = handlers, None, [], [], []

    def set_adapter(self, a):
        self.adapter = a
        self.alog.append(a)

    def gen(self, tok, prompt, max_new, seed):
        text = self.h[self.adapter](prompt, seed)
        n = tok.n(text) + 1
        out = ({"text": text[:max_new * tok.CH], "n_tokens": max_new} if n > max_new else {"text": text, "n_tokens": n})
        self.outs.append(out["text"])
        return out


def greedy_shared_prefix(model, tok, prompts, *, max_new_tokens=64, max_batch_tokens=2000, min_prefix=16):
    model.log.append(("greedy", model.adapter, list(prompts), max_new_tokens, max_batch_tokens))
    return [model.gen(tok, p, max_new_tokens, None) for p in prompts]


def sample_crn_shared(model, tok, prompts, seeds, *, max_new_tokens=64, temperature=1.0, top_p=0.95,
                      max_batch_tokens=2000):
    model.log.append(("sample", model.adapter, list(prompts), [list(s) for s in seeds], max_new_tokens, temperature,
                      top_p, max_batch_tokens))
    return [[model.gen(tok, p, max_new_tokens, sd) for sd in row] for p, row in zip(prompts, seeds)]


LM.greedy_shared_prefix, LM.sample_crn_shared = greedy_shared_prefix, sample_crn_shared
assert "bwmn.lm" not in sys.modules
sys.modules["bwmn.lm"] = LM

from bwmn import blocks as B  # noqa: E402
from bwmn import blocks_planner as BP  # noqa: E402


def snapshot(mod):
    out = {}
    for k, v in vars(mod).items():
        if k.startswith("__") and k not in ("__file__",):
            continue
        code = getattr(v, "__code__", None)
        if isinstance(v, type):
            cd = {kk: (id(vv), hashlib.sha256(vv.__code__.co_code).hexdigest() if hasattr(vv, "__code__") else repr(vv))
                  for kk, vv in vars(v).items() if not kk.startswith("__")}
            out[k] = ("class", id(v), json.dumps(cd, sort_keys=True))
        elif code is not None:
            out[k] = ("func", id(v), hashlib.sha256(code.co_code).hexdigest(), repr(v.__defaults__),
                      repr(v.__kwdefaults__))
        elif isinstance(v, types.ModuleType):
            out[k] = ("mod", id(v))
        else:
            try:
                out[k] = ("val", id(v), repr(v)[:2000])
            except Exception:
                out[k] = ("val", id(v))
    return out


# ---------------------------------------------------------------- Blocksworld reference workload
def bw_instances(n, seed):
    rng = random.Random(seed)
    inst = []
    while len(inst) < n:
        g = B.generate(rng, rng.randint(3, 6), full_goal_frac=0.3)
        if g:
            inst.append({"id": f"w{len(inst)}", "objs": g[0], "init": g[1], "goal": g[2]})
    return inst


def pa_handler(mode, noise=0.5):
    def h(prompt, seed):
        user = prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        d = json.loads(user)
        s = B.state_from_json(d["state"])
        goal = frozenset(("on", x[0], x[1]) for x in d["goal"])
        objs = B.blocks_of(s)
        r = random.Random(hashlib.sha256(f"{prompt}|{seed}".encode()).hexdigest())
        plan = B.bfs_plan(s, goal, objs, max_states=20000) if goal is not None else None
        plan = list(plan or [])
        if seed is not None and plan and r.random() < noise:
            plan[r.randrange(len(plan))] = r.choice(B.ground_actions(objs))
        if not plan:
            plan = [r.choice(B.ground_actions(objs))]
        if mode == "actions":
            t = B.target_actions(plan)
        else:
            steps, st = [], s
            for a in plan:
                st = B.step(st, a)[0]
                steps.append((a, st))
            t = B.target_states(steps)
        k = r.randrange(10)
        if k == 0:
            return t[:r.randrange(1, len(t))]
        if k == 1:
            return "I cannot do that."
        return t
    return h


def fwd_handler(noise=0.15):
    def h(prompt, seed):
        user = prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        d = json.loads(user)
        s = B.state_from_json(d["state"])
        a = tuple(d["action"].split())
        ns, ok = B.step(s, a)
        t = B.forward_target_delta(s, ns, ok)
        r = random.Random(hashlib.sha256(prompt.encode()).hexdigest())
        k = r.random()
        if k < noise / 3:
            return t.replace("true", "false") if ok else t.replace("false", "true")
        if k < 2 * noise / 3:
            return t[:r.randrange(1, len(t))]
        if k < noise:
            return '{"add":["holding zz"],"del":[],"legal":true}'
        return t
    return h


def bw_workload(tag):
    tok = Tok()
    inst = bw_instances(8, 41)
    res = {}
    m = Model({"fwd": fwd_handler(), "pa": pa_handler("actions")})
    fwd = BP.LMForwardBW(m, tok, "fwd", fmt="delta", cache=SCR / f"wl_{tag}.jsonl")
    prop = BP.LMProposerBW(m, tok, "pa", "actions", K=4, max_new=80, prefix_ok=True)
    eps = BP.plan_beam(inst, prop, fwd, eval_seed=0, width=4, max_expansions=12, max_depth=80, max_rollout=12)
    res["beam_lm"] = BP.score(inst, eps)
    res["beam_lm_log"] = hashlib.sha256(json.dumps(m.log, default=str).encode()).hexdigest()
    eps = BP.plan_beam(inst, BP.OracleProposer(K=2), BP.ExactF(), width=2, max_expansions=8)
    res["beam_exact"] = BP.score(inst, eps)
    m2 = Model({"fwd": fwd_handler(), "pa": pa_handler("states")})
    eps = BP.plan_instances(inst, BP.LMProposerBW(m2, tok, "pa", "states", K=3),
                            BP.LMForwardBW(m2, tok, "fwd", fmt="delta"), step_cap=12)
    res["imagine_states"] = BP.score(inst, eps)
    eps = BP.plan_direct(inst, BP.LMProposerBW(Model({"pa": pa_handler("actions")}), tok, "pa", K=1, max_new=256))
    res["direct"] = BP.score(inst, eps)
    (SCR / f"wl_{tag}.jsonl").unlink()
    return json.loads(json.dumps(res, default=str))


for f in SCR.glob("wl_*.jsonl"):
    f.unlink()
snap_B0, snap_BP0 = snapshot(B), snapshot(BP)
cls0 = {c: dict(vars(getattr(BP, c))) for c in ("LMForwardBW", "LMProposerBW", "ExactF", "OracleProposer")}
wl0 = bw_workload("before")

from bwmn import domain_planner as DP  # noqa: E402
from bwmn import logistics as L  # noqa: E402

snap_B1, snap_BP1 = snapshot(B), snapshot(BP)
cls1 = {c: dict(vars(getattr(BP, c))) for c in ("LMForwardBW", "LMProposerBW", "ExactF", "OracleProposer")}
wl1 = bw_workload("after")


def diff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


REPORT["import_effect"] = {"blocks_diff": diff(snap_B0, snap_B1), "bp_diff": diff(snap_BP0, snap_BP1),
                           "class_dict_diff": {c: diff(cls0[c], cls1[c]) for c in cls0},
                           "workload_equal": wl0 == wl1, "bp_lm_is_stub": BP.lm is LM, "dp_bp_is_bp": DP.BP is BP,
                           "dp_B_is_B": DP.B is B, "L_B_is_B": L.B is B}

# ---------------------------------------------------------------- forward parity on ALL non-test cache keys
lines = []
for p in (ROOT / "runs/BW_SELECT_v1/fcache_delta.jsonl", ROOT / "runs/BW_DISTILL_r1/fcache.jsonl"):
    lines += [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
rows = [json.loads(ln) for ln in lines]


def key(r):
    return frozenset(tuple(f) for f in r["s"]), tuple(r["a"])


keys_all = [key(r) for r in rows]
text_by_key = {}
for r, k in zip(rows, keys_all):
    text_by_key[k] = r["text"]
uniq = list(dict.fromkeys(keys_all))
REPORT["fwd_n_lines"], REPORT["fwd_n_unique_keys"] = len(lines), len(uniq)

# (1) cache loader parity on the full concatenated caches (plus a torn line)
full = "\n".join(lines) + "\n" + '{"s": [["clear", "a"]], "a": ["pick' + "\n"
(SCR / "c_bw.jsonl").write_text(full, encoding="utf-8")
(SCR / "c_d.jsonl").write_text(full, encoding="utf-8")
tok = Tok()
fb = BP.LMForwardBW(Model({}), tok, "fwd", fmt="delta", cache=SCR / "c_bw.jsonl")
fd = DP.LMForwardD(Model({}), tok, "fwd", cache=SCR / "c_d.jsonl", dom=B)
REPORT["loader_memo_equal"] = list(fb.memo.items()) == list(fd.memo.items())
REPORT["loader_memo_size"] = len(fb.memo)
REPORT["loader_n_none"] = sum(v is None for v in fb.memo.values())
REPORT["loader_n_correct"] = sum(v == B.step(*k) for k, v in fb.memo.items())
# memo predict on all keys (pure hits)
REPORT["loader_predict_equal"] = fb.predict(keys_all) == fd.predict(keys_all) and fb.model_queries == fd.model_queries == 0
for f in ("c_bw.jsonl", "c_d.jsonl"):
    (SCR / f).unlink()

# (2) fresh predict parity: every unique key generated; answers = real raw texts, a fraction corrupted
rng = random.Random(123)


def corrupt(t, r):
    k = r.randrange(12)
    if k == 0:
        return t[:r.randrange(0, len(t) + 1)]
    if k == 1:
        return t.replace('"legal":true', '"legal":1')
    if k == 2:
        return "  " + t + "\n\n"
    if k == 3:
        return t.replace('"del":[', '"del":["on q q",', 1)
    if k == 4:
        return '{"legal":true,"add":[],"del":[]}'
    if k == 5:
        return t + " trailing"
    return t


answers = {}
for k in uniq:
    r = random.Random(hashlib.sha256(repr(sorted(k[0])).encode() + repr(k[1]).encode()).hexdigest())
    answers[B.forward_prompt_delta(*k)] = corrupt(text_by_key[k], r) if r.random() < 0.3 else text_by_key[k]
h = lambda p, s: answers[p]
mb, md = Model({"fwd": h}), Model({"fwd": h})
fb = BP.LMForwardBW(mb, tok, "fwd", fmt="delta", cache=SCR / "p_bw.jsonl", max_batch_tokens=5000)
fd = DP.LMForwardD(md, tok, "fwd", cache=SCR / "p_d.jsonl", max_batch_tokens=5000, dom=B)
order = list(keys_all)
rng.shuffle(order)
ob, od = [], []
i = 0
while i < len(order):
    n = rng.randint(1, 900)
    batch = order[i:i + n] + rng.sample(order, min(20, len(order)))
    ob += fb.predict(batch)
    od += fd.predict(batch)
    i += n
REPORT["predict_out_equal"] = ob == od
REPORT["predict_n"] = len(ob)
REPORT["predict_log_equal"] = mb.log == md.log
REPORT["predict_n_batches"] = len(mb.log)
REPORT["predict_caps"] = sorted({x[3] for x in md.log}), sorted({x[4] for x in md.log})
REPORT["predict_counts"] = (fb.model_queries, fd.model_queries, fb.model_tokens, fd.model_tokens)
REPORT["predict_cache_bytes_equal"] = (SCR / "p_bw.jsonl").read_bytes() == (SCR / "p_d.jsonl").read_bytes()
REPORT["predict_memo_equal"] = list(fb.memo.items()) == list(fd.memo.items())
REPORT["predict_n_none"] = sum(o is None for o in od)
# prompts byte-identical and equal to B.forward_prompt_delta
allp_b = [p for x in mb.log for p in x[2]]
allp_d = [p for x in md.log for p in x[2]]
REPORT["prompts_equal_B"] = sorted(allp_d) == sorted(B.forward_prompt_delta(*k) for k in uniq)
REPORT["prompts_bytes_equal"] = "\n".join(allp_b).encode() == "\n".join(allp_d).encode() and len(allp_d) == len(uniq)
# reload both caches
rb = BP.LMForwardBW(Model({}), tok, "fwd", fmt="delta", cache=SCR / "p_bw.jsonl")
rd = DP.LMForwardD(Model({}), tok, "fwd", cache=SCR / "p_d.jsonl", dom=B)
REPORT["reload_equal"] = list(rb.memo.items()) == list(rd.memo.items()) == list(fb.memo.items())
for f in ("p_bw.jsonl", "p_d.jsonl"):
    (SCR / f).unlink()

# ExactD(B) vs ExactF on all keys
REPORT["exact_equal"] = DP.ExactD(B).predict(keys_all) == BP.ExactF().predict(keys_all)

# ---------------------------------------------------------------- proposer parity on many items from real states
states = list(dict.fromkeys(k[0] for k in uniq))
REPORT["n_states"] = len(states)
rng = random.Random(77)
items = []
for i, s in enumerate(states):
    objs = B.blocks_of(s)
    ons = sorted(f for f in B.towers_to_state(B.random_towers(objs, rng, p_stack=0.85)) if f[0] == "on")
    goal = frozenset(ons if rng.random() < 0.3 else rng.sample(ons, rng.randint(1, min(4, len(ons))))) if ons \
        else frozenset({("on", objs[0], objs[-1])})
    rid = rng.choice([f"{i % 7}.{i % 4}", i % 5, f"x{i}"])
    items.append({"id": rng.choice([f"s{i}", i, f"inst-{i // 3}"]), "objs": objs, "goal": goal, "state": s,
                  "round": rid})


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


configs = [("actions", False, None, 8, 0), ("actions", True, 80, 8, 0), ("actions", True, 120, 4, 7),
           ("actions", False, 256, 1, 0), ("states", False, None, 4, 0), ("states", False, 300, 2, 3),
           ("states", True, 200, 3, 1), ("actions", True, None, 2, 12345678901234567890)]
prop_res = []
for mode, pok, mx, K, es in configs:
    sub = items if K <= 4 else items[::2]
    mb, md = Model({"pa": rand_handler(mode)}), Model({"pa": rand_handler(mode)})
    pb = BP.LMProposerBW(mb, tok, "pa", mode, K=K, max_new=mx, prefix_ok=pok, max_batch_tokens=6000)
    pd = DP.LMProposerD(md, tok, "pa", mode, K=K, max_new=mx, prefix_ok=pok, max_batch_tokens=6000, dom=B)
    ob, od = [], []
    for j in range(0, len(sub), 500):
        ob += pb.propose(sub[j:j + 500], es)
        od += pd.propose(sub[j:j + 500], es)
    # candidate order/seed check: out[i][j] must be the parse of handler(prompt_i, seed_ij) (within the cap)
    hh = rand_handler(mode)
    bad_order = 0
    for it, cands in zip(sub, od):
        p = B.proposer_prompt(it["state"], it["goal"])
        for j, c in enumerate(cands):
            sd = None if j == 0 else BP._seed(es, it["id"], it["round"], j)
            t = hh(p, sd)
            n = tok.n(t) + 1
            mxx = pd.max_new
            if n > mxx:
                t = t[:mxx * tok.CH]
            pr = (B.parse_plan_prefix(t, it["objs"]) if pok and mode == "actions" else B.parse_plan(t, it["objs"], mode))
            if c["plan"] != (pr[0] if pr else None):
                bad_order += 1
    prop_res.append({"cfg": [mode, pok, mx, K, str(es)], "n_items": len(sub), "out_equal": ob == od,
                     "log_equal": mb.log == md.log, "outs_equal": mb.outs == md.outs,
                     "alog_equal": mb.alog == md.alog, "max_new": (pb.max_new, pd.max_new),
                     "n_parsed": sum(c["plan"] is not None for cs in od for c in cs),
                     "n_cands": sum(len(cs) for cs in od), "seed_order_mismatch": bad_order})
REPORT["proposer"] = prop_res

# shuffled item order: per-item candidates identical (CRN) for both
sub = items[:400]
perm = list(range(len(sub)))
random.Random(9).shuffle(perm)
pd = DP.LMProposerD(Model({"pa": rand_handler("actions")}), tok, "pa", K=4, max_new=80, prefix_ok=True, dom=B)
a1 = pd.propose(sub, 0)
a2 = pd.propose([sub[i] for i in perm], 0)
REPORT["proposer_crn_order_invariant"] = all(a2[k] == a1[i] for k, i in enumerate(perm))

# ---------------------------------------------------------------- score_d vs BP.score on many eps
inst = bw_instances(60, 5)
rng = random.Random(8)
eps = []
for it in inst:
    p = B.bfs_plan(it["init"], it["goal"], it["objs"]) or []
    k = rng.randrange(4)
    plan = None if k == 0 else (p if k == 1 else (p[:-1] if k == 2 else p + [rng.choice(B.ground_actions(it["objs"]))]))
    eps.append({"id": it["id"], "plan": plan, "verified": rng.random() < 0.5, "rounds": rng.randint(0, 9),
                "gen_tokens": rng.randint(0, 999), "fail": rng.choice([None, "budget", "dead_end"])})
REPORT["score_equal"] = json.dumps(DP.score_d(inst, eps, B)) == json.dumps(BP.score(inst, eps))

REPORT["bp_sha_after"] = sha(ROOT / "bwmn" / "blocks_planner.py")
REPORT["bp_sha_pinned"] = REPORT["bp_sha_after"] == PIN == REPORT["bp_sha_before"]
REPORT["torch_loaded"] = "torch" in sys.modules
print(json.dumps(REPORT, indent=1, default=str))
