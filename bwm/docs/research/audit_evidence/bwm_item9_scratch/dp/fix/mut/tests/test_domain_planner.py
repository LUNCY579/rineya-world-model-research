"""Tests for bwmn/domain_planner.py (item 9 CPU-3). CPU only, pure python: no torch/transformers/peft and no model.

The pinned bwmn/blocks_planner.py imports bwmn.lm, which imports torch at module level. Unless bwmn.lm is already
imported, this module registers a pure-python stand-in for bwmn.lm only for the import of the planner modules, then
removes the stand-in and the planner modules bound to it from sys.modules (and the bwmn package) right away: this
file keeps its own references (B, BP, DP), and any other module imported later in the same session (collection
included) gets the real bwmn.lm. Generation is always routed to FakeLM through the two lm functions the planners call
(greedy_shared_prefix, sample_crn_shared), patched per test, so no real decode can run even if the real lm is loaded.
FakeTok counts 1 token per 4 characters; outputs over max_new_tokens are cut.

  python -B -m pytest tests/test_domain_planner.py -q -p no:cacheprovider

Read-only inputs: runs/BW_SELECT_v1/fcache_delta.jsonl and runs/BW_DISTILL_r1/fcache.jsonl (sampled keys and raw F
outputs; copies go to tmp_path), the local VAL binary with the vendored Logistics domain. Logistics instances are
generated with a test-only seed (not a registered data or evaluation seed); no PlanBench instance is planned on.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import random
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_STUB_LM = None
_STUB_BOUND = ("domain_planner", "blocks_planner", "lm")  # modules that would hold the stand-in
if "bwmn.lm" not in sys.modules:
    _STUB_LM = types.ModuleType("bwmn.lm")
    _STUB_LM.__doc__ = "pure-python stand-in for bwmn.lm (tests/test_domain_planner.py); generation is patched per test"

    def _no_model(*a, **k):
        raise RuntimeError("stub bwmn.lm: no generation outside the FakeLM fixture")

    _STUB_LM.greedy_shared_prefix = _STUB_LM.sample_crn_shared = _no_model
    for _m in _STUB_BOUND:
        assert f"bwmn.{_m}" not in sys.modules, _m
    sys.modules["bwmn.lm"] = _STUB_LM

try:
    from bwmn import blocks as B  # noqa: E402
    from bwmn import blocks_planner as BP  # noqa: E402
    from bwmn import domain_planner as DP  # noqa: E402
    from bwmn import logistics as L  # noqa: E402
finally:
    if _STUB_LM is not None:  # unregister at once: later imports (other test modules) get the real modules
        import bwmn
        for _m in _STUB_BOUND:  # none existed before the stand-in (asserted above): all are bound to it
            sys.modules.pop(f"bwmn.{_m}", None)
            if _m in vars(bwmn):
                delattr(bwmn, _m)

PIN = "18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960"
FCACHE_SELECT = ROOT / "runs" / "BW_SELECT_v1" / "fcache_delta.jsonl"
FCACHE_DISTILL = ROOT / "runs" / "BW_DISTILL_r1" / "fcache.jsonl"
LG_SEED = 7311  # test-only
NEEDS_VAL = pytest.mark.skipif(not (L.VAL_EXE.exists() and L.DOMAIN_PDDL.exists()), reason="VAL or domain missing")


# ----------------------------------------------------------------------------- fake model / tokenizer

class FakeTok:
    """1 token per CH characters (ceil); +1 for EOS."""
    CH = 4

    def n_tokens(self, text: str) -> int:
        return -(-len(text) // self.CH)


class FakeLM:
    """Stand-in for the core with two LoRA adapters: handlers[adapter](prompt, seed) -> text. Records every
    set_adapter and every generation call (via the patched lm functions)."""

    def __init__(self, handlers: dict):
        self.handlers = handlers
        self.adapter = None
        self.adapter_log = []
        self.log = []
        self.outputs = []  # every returned text, in generation order
        self.n_generations = 0

    def set_adapter(self, name):
        self.adapter = name
        self.adapter_log.append(name)

    def complete(self, tok, prompt, max_new, seed):
        self.n_generations += 1
        text = self.handlers[self.adapter](prompt, seed)
        n = tok.n_tokens(text) + 1
        out = ({"text": text[:max_new * tok.CH], "n_tokens": max_new, "hit_eos": False} if n > max_new
               else {"text": text, "n_tokens": n, "hit_eos": True})
        self.outputs.append(out["text"])
        return out


def fake_greedy_shared_prefix(model, tok, prompts, *, max_new_tokens=64, max_batch_tokens=2000, min_prefix=16):
    model.log.append(("greedy", model.adapter, list(prompts), max_new_tokens, max_batch_tokens))
    return [model.complete(tok, p, max_new_tokens, None) for p in prompts]


def fake_sample_crn_shared(model, tok, prompts, seeds, *, max_new_tokens=64, temperature=1.0, top_p=0.95,
                           max_batch_tokens=2000):
    assert all(len(s) == len(seeds[0]) for s in seeds)
    model.log.append(("sample", model.adapter, list(prompts), [list(s) for s in seeds], max_new_tokens, temperature,
                      top_p, max_batch_tokens))
    return [[model.complete(tok, p, max_new_tokens, sd) for sd in row] for p, row in zip(prompts, seeds)]


@pytest.fixture(autouse=True)
def fake_lm_functions(monkeypatch):
    monkeypatch.setattr(BP.lm, "greedy_shared_prefix", fake_greedy_shared_prefix, raising=False)
    monkeypatch.setattr(BP.lm, "sample_crn_shared", fake_sample_crn_shared, raising=False)


def split_chat(prompt: str):
    head, rest = prompt.split("<|im_start|>user\n", 1)
    system = head[len("<|im_start|>system\n"):].split("<|im_end|>", 1)[0]
    user = rest.split("<|im_end|>", 1)[0]
    assert B.render_chat(system, user) == prompt
    return system, user


# ----------------------------------------------------------------------------- module contract

def test_pin_and_reexports():
    assert DP.BLOCKS_PLANNER_SHA256 == PIN
    assert DP.file_sha256(ROOT / "bwmn" / "blocks_planner.py") == PIN
    assert DP.source_shas()["blocks_planner.py"] == PIN
    assert Path(BP.__file__).resolve() == (ROOT / "bwmn" / "blocks_planner.py").resolve()
    assert DP.plan_beam is BP.plan_beam and DP.plan_direct is BP.plan_direct and DP.rollout_many is BP.rollout_many
    assert issubclass(DP.LMForwardD, BP.LMForwardBW) and issubclass(DP.LMProposerD, BP.LMProposerBW)


def test_no_model_stack_imported():
    if _STUB_LM is None:
        pytest.skip("bwmn.lm was already imported by another test module")
    assert BP.lm is _STUB_LM and DP.BP is BP
    for m in ("torch", "transformers", "peft"):
        assert m not in sys.modules, m
    # the stand-in is not visible to anything imported after this module
    assert all(sys.modules.get(f"bwmn.{m}") not in (_STUB_LM, BP, DP) for m in _STUB_BOUND)


def test_import_refuses_modified_planner(tmp_path):
    """Copies of the package: the pristine copy imports; one appended byte in blocks_planner.py -> ImportError,
    raised before the modified file is executed."""
    for tag, extra in (("ok", b""), ("bad", b"\n# modified\n")):
        pkg = tmp_path / tag / f"bwmn_{tag}"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        for f in ("blocks.py", "domain_planner.py"):
            shutil.copyfile(ROOT / "bwmn" / f, pkg / f)
        (pkg / "blocks_planner.py").write_bytes((ROOT / "bwmn" / "blocks_planner.py").read_bytes() + extra)
        (pkg / "lm.py").write_text("raise_if_called = True\n")  # the planner's 'from . import lm' needs a module
        sys.path.insert(0, str(tmp_path / tag))
        try:
            if tag == "ok":
                m = importlib.import_module(f"bwmn_{tag}.domain_planner")
                assert m.BLOCKS_PLANNER_SHA256 == PIN
            else:
                with pytest.raises(ImportError, match="pinned"):
                    importlib.import_module(f"bwmn_{tag}.domain_planner")
                assert f"bwmn_{tag}.blocks_planner" not in sys.modules
        finally:
            sys.path.remove(str(tmp_path / tag))
            for k in [k for k in sys.modules if k.startswith(f"bwmn_{tag}")]:
                del sys.modules[k]


def test_constructor_contracts():
    m, tok = FakeLM({}), FakeTok()
    with pytest.raises(ValueError):
        DP.LMForwardD(m, tok, "fwd", fmt="state", dom=B)
    with pytest.raises(ValueError):
        DP.LMProposerD(m, tok, "pa", dom=L)  # no M for a non-Blocksworld domain
    with pytest.raises(TypeError):
        DP.LMForwardD(m, tok, "fwd", dom=types.SimpleNamespace())
    with pytest.raises(TypeError):
        DP.LMProposerD(m, tok, "pa", dom=types.SimpleNamespace(proposer_prompt=len, parse_plan=len), max_new=9,
                       prefix_ok=True)
    with pytest.raises(TypeError):
        DP.ExactD(types.SimpleNamespace(step=len))
    assert DP.LMProposerD(m, tok, "pa", dom=B).max_new == 120  # B2 defaults kept for Blocksworld
    assert DP.LMProposerD(m, tok, "pa", "states", dom=B).max_new == 700
    p = DP.LMProposerD(m, tok, "pa", K=4, max_new=184, prefix_ok=True, dom=L)
    assert (p.K, p.max_new, p.prefix_ok, p.mode, p.temperature, p.top_p) == (4, 184, True, "actions", 1.0, 0.95)
    f = DP.LMForwardD(m, tok, dom=L)
    assert (f.adapter, f.fmt, f.max_new, f.max_batch_tokens, f.cache) == ("fwd", "delta", 40, 2000, None)
    assert f.roundtrip and not DP.LMForwardD(m, tok, dom=B).roundtrip  # Blocksworld keeps B2's parse rule
    with pytest.raises(TypeError, match="state_json"):  # the render check needs the domain's state codec
        DP.LMForwardD(m, tok, dom=types.SimpleNamespace(forward_prompt_delta=len, parse_forward_delta=len))


# ----------------------------------------------------------------------------- PARITY with dom = bwmn.blocks

def _sample_lines(path: Path, step: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines[::step] if ln.strip()]


def _key(r):
    return frozenset(tuple(f) for f in r["s"]), tuple(r["a"])


@pytest.fixture(scope="module")
def bw_real_rows():
    if not FCACHE_SELECT.exists():
        pytest.skip("runs/BW_SELECT_v1/fcache_delta.jsonl missing")
    lines = _sample_lines(FCACHE_SELECT, 24)
    if FCACHE_DISTILL.exists():
        lines += _sample_lines(FCACHE_DISTILL, 75)
    rows = [json.loads(ln) for ln in lines]
    assert len(rows) >= 400
    return lines, rows


def test_forward_parity_blocks_real_keys(bw_real_rows, tmp_path):
    lines, rows = bw_real_rows
    cached, fresh = lines[0::2], rows[1::2]
    torn = '{"s": [["clear", "a"]], "a": ["pick'  # crash-torn last line: skipped by both loaders
    for name in ("bw", "d"):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(cached) + "\n" + torn + "\n", encoding="utf-8")
    # model answers for the fresh keys: the real outputs, with some corrupted (refusal, bad del, non-bool legal)
    answers = {}
    for i, r in enumerate(fresh):
        s, a = _key(r)
        text = r["text"]
        if i % 7 == 3:
            text = "I'm sorry, but I can't help with that."
        elif i % 7 == 5:
            text = '{"add":[],"del":["on zz yy"],"legal":true}'
        elif i % 11 == 4:
            text = text.replace('"legal":true', '"legal":"yes"').replace('"legal":false', '"legal":"no"')
        elif i % 13 == 6:  # degenerate fact: B2's rule accepts it (state with ()); dom=B must keep that rule
            text = '{"add":[""],"del":[],"legal":true}'
        answers[B.forward_prompt_delta(s, a)] = text
    handler = lambda prompt, seed: answers.get(prompt, "<missing>")
    ad = "fwd_delta"  # bw_select.py's F adapter name (f"fwd_{fmt}"), not the default
    m_bw, m_d, tok = FakeLM({ad: handler}), FakeLM({ad: handler}), FakeTok()
    f_bw = BP.LMForwardBW(m_bw, tok, ad, fmt="delta", cache=tmp_path / "bw.jsonl", max_batch_tokens=3000)
    f_d = DP.LMForwardD(m_d, tok, ad, cache=tmp_path / "d.jsonl", max_batch_tokens=3000, dom=B)
    assert len(f_bw.memo) > 200 and f_d.memo == f_bw.memo
    rng = random.Random(5)
    keys = [_key(r) for r in rows]
    keys += rng.sample(keys, 60)  # duplicates within a batch
    rng.shuffle(keys)
    out_bw = f_bw.predict(keys)
    out_d = f_d.predict(keys)
    assert out_d == out_bw
    assert m_d.log == m_bw.log and len(m_d.log) == 1
    assert m_d.adapter_log == m_bw.adapter_log == [ad]
    kind, adapter, prompts, max_new, mbt = m_d.log[0]
    assert (kind, adapter, max_new, mbt) == ("greedy", ad, 40, 3000)
    assert "<missing>" not in [answers.get(p, "<missing>") for p in prompts]
    assert (f_d.model_queries, f_d.model_tokens) == (f_bw.model_queries, f_bw.model_tokens)
    assert f_d.model_queries == len(prompts) == len({k for k in keys if k not in dict.fromkeys(_key(json.loads(x))
                                                                                                 for x in cached)})
    assert (tmp_path / "d.jsonl").read_bytes() == (tmp_path / "bw.jsonl").read_bytes()
    n_none = sum(o is None for o in out_d)
    assert 0 < n_none < len(out_d) // 2
    assert any(o is not None and () in o[0] for o in out_d)  # the B2 rule, unchanged for dom=B
    # the real outputs are mostly correct: the sample is a meaningful parse workload
    ok = [o == B.step(*k) for k, o in zip(keys, out_d) if o is not None]
    assert sum(ok) / len(ok) > 0.9
    # a reload of either cache gives the same memo; the parse is the domain's
    r_bw = BP.LMForwardBW(FakeLM({}), tok, ad, fmt="delta", cache=tmp_path / "bw.jsonl")
    r_d = DP.LMForwardD(FakeLM({}), tok, ad, cache=tmp_path / "d.jsonl", dom=B)
    assert r_d.memo == r_bw.memo == f_bw.memo
    assert r_d.predict(keys) == out_bw and r_d.model_queries == 0


def _bw_items(n_items=24, seed=11):
    rng = random.Random(seed)
    items = []
    while len(items) < n_items:
        g = B.generate(rng, rng.randint(3, 9), full_goal_frac=0.3)
        if g is None:
            continue
        objs, init, goal = g
        i = len(items)
        items.append({"id": f"bw{i}", "objs": objs, "goal": goal, "state": init, "round": f"{i % 3}.{i % 4}"})
    return items


def _bw_handler(mode):
    """Deterministic varied proposer outputs in the format of `mode` (most kinds), plus the other format, torn JSON,
    unknown actions or blocks and refusals. State plans are 1-6 steps so that some fit even a 90-token cap."""
    def target(acts, s, fmt):
        if fmt == "actions":
            return B.target_actions(acts)
        steps, st = [], s
        for a in acts:
            st = B.step(st, a)[0]
            steps.append((a, st))
        return B.target_states(steps)

    other = "states" if mode == "actions" else "actions"

    def h(prompt, seed):
        system, user = split_chat(prompt)
        assert system == B.SYS_PROPOSER
        d = json.loads(user)
        s = B.state_from_json(d["state"])
        objs = B.blocks_of(s)
        r = random.Random(hashlib.sha256(f"{prompt}|{seed}".encode()).hexdigest())
        acts = [r.choice(B.ground_actions(objs)) for _ in range(r.randint(1, 40 if mode == "actions" else 6))]
        kind = r.randrange(7)
        if kind in (0, 1, 2):
            return target(acts, s, mode)
        if kind == 3:
            t = target(acts, s, mode)
            return t[:r.randrange(1, len(t))]  # torn (a prefix may still be accepted with prefix_ok)
        if kind == 4:
            return target(acts[:3], s, mode).replace(acts[0][0], "teleport", 1)
        if kind == 5:
            return r.choice((target(acts[:2] + [("pick-up", "zz")] + acts[2:], s, mode), target(acts, s, other)))
        return "Sorry, I cannot plan that."
    return h


@pytest.mark.parametrize("mode,prefix_ok,max_new,K,eval_seed,min_parsed", [
    ("actions", False, None, 4, 0, 0.25), ("actions", True, 60, 4, 0, 0.25), ("actions", True, 184, 8, 7, 0.25),
    ("states", False, None, 3, 0, 0.25),
    ("states", True, None, 3, 1, 0.25),  # prefix_ok is an actions-mode rule: state plans keep the full parse
    ("states", False, 90, 2, 3, 0.05)])  # 90-token cap: most state plans are cut and unparseable
def test_proposer_parity_blocks_generated_items(mode, prefix_ok, max_new, K, eval_seed, min_parsed):
    _proposer_parity(_bw_items(), mode, prefix_ok, max_new, K, eval_seed, min_parsed)


@pytest.mark.parametrize("mode,prefix_ok,max_new", [("actions", True, 184), ("states", True, None),
                                                   ("states", False, None)])
def test_proposer_parity_blocks_nondefault_sampling(mode, prefix_ok, max_new):
    """Non-default temperature, top-p and adapter name reach the sampler exactly as LMProposerBW passes them."""
    _proposer_parity(_bw_items(12, seed=13), mode, prefix_ok, max_new, 3, 5, 0.25, adapter="pa_lg",
                     temperature=0.7, top_p=0.8)


def test_proposer_parity_blocks_real_states(bw_real_rows):
    """Proposer items on the imagined states of the real F caches (holding states included), goals drawn as
    B.generate draws them (1-4 on-pairs of a random target, or every on-pair) over each state's blocks."""
    _, rows = bw_real_rows
    rng = random.Random(29)
    items = []
    for i, r in enumerate(rows[::7]):
        s = _key(r)[0]
        objs = B.blocks_of(s)
        ons = sorted(f for f in B.towers_to_state(B.random_towers(objs, rng, p_stack=0.85)) if f[0] == "on")
        goal = frozenset(ons if rng.random() < 0.3 else rng.sample(ons, rng.randint(1, min(4, len(ons))))) if ons \
            else frozenset({("on", objs[0], objs[-1])})
        items.append({"id": f"real{i}", "objs": objs, "goal": goal, "state": s, "round": f"{i % 5}.{i % 4}"})
    assert len(items) >= 50 and any(("handempty",) not in it["state"] for it in items)
    _proposer_parity(items, "actions", True, 120, 4, 0, 0.25)
    _proposer_parity(items, "actions", False, None, 2, 11, 0.25)


def _proposer_parity(items, mode, prefix_ok, max_new, K, eval_seed, min_parsed, adapter="pa", temperature=1.0,
                     top_p=0.95):
    """LMProposerD(dom=B) vs LMProposerBW on the same items and the same fake Pa: identical prompts, seeds, caps,
    sampling parameters, adapter switches, raw outputs and parsed candidates."""
    tok = FakeTok()
    m_bw, m_d = FakeLM({adapter: _bw_handler(mode)}), FakeLM({adapter: _bw_handler(mode)})
    kw = dict(K=K, temperature=temperature, top_p=top_p, max_new=max_new, prefix_ok=prefix_ok, max_batch_tokens=2500)
    p_bw = BP.LMProposerBW(m_bw, tok, adapter, mode, **kw)
    p_d = DP.LMProposerD(m_d, tok, adapter, mode, dom=B, **kw)
    out_bw = p_bw.propose(items, eval_seed)
    out_d = p_d.propose(items, eval_seed)
    assert out_d == out_bw
    assert m_d.log == m_bw.log and m_d.adapter_log == m_bw.adapter_log == [adapter]
    _, ad, prompts, seeds, mx, temp, tp, mbt = m_d.log[0]
    assert prompts == [B.proposer_prompt(it["state"], it["goal"]) for it in items]
    assert seeds == [[None] + [BP._seed(eval_seed, it["id"], it["round"], j) for j in range(1, K)] for it in items]
    assert (ad, mx, temp, tp, mbt) == (adapter, p_bw.max_new, temperature, top_p, 2500)
    assert m_d.outputs == m_bw.outputs and len(m_d.outputs) == K * len(items)
    flat = [(it, c) for it, cands in zip(items, out_d) for c in cands]
    plans = [c["plan"] for _, c in flat]
    assert any(p is None for p in plans) and sum(p is not None for p in plans) > min_parsed * len(plans)
    # each candidate is the domain parse of its own raw text, with the item's objs passed through
    for (it, c), text in zip(flat, m_d.outputs):
        full = B.parse_plan(text, it["objs"], mode)
        if prefix_ok and mode == "actions":  # LMProposerBW's rule: the prefix parse applies in actions mode only
            assert c["plan"] == (B.parse_plan_prefix(text, it["objs"]) or (None,))[0]
        else:
            assert c["plan"] == (full[0] if full else None) and c["claimed"] == (full[1] if full else None)
    if prefix_ok and mode == "actions":  # some truncated outputs are accepted as prefixes, others are not
        assert any(c["plan"] is not None and B.parse_plan(t, it["objs"], "actions") is None
                   for (it, c), t in zip(flat, m_d.outputs))
    if mode == "states":
        assert any(c["claimed"] is not None for _, c in flat)


def test_exact_and_score_parity_blocks():
    rng = random.Random(3)
    inst = []
    while len(inst) < 6:
        g = B.generate(rng, rng.randint(3, 5))
        if g:
            inst.append({"id": f"b{len(inst)}", "objs": g[0], "init": g[1], "goal": g[2]})
    keys = [(it["init"], a) for it in inst for a in B.ground_actions(it["objs"])]
    assert DP.ExactD(B).predict(keys) == BP.ExactF().predict(keys)
    e1 = BP.plan_beam(inst, BP.OracleProposer(K=2), BP.ExactF(), width=2, max_expansions=8)
    e2 = DP.plan_beam(inst, BP.OracleProposer(K=2), DP.ExactD(B), width=2, max_expansions=8)
    assert e1 == e2
    e2[0]["plan"] = e2[0]["plan"][:-1]  # one failing plan
    assert DP.score_d(inst, e2, B) == BP.score(inst, e2)
    s, rows = DP.score_d(inst, e2, B)
    assert s["success"] == 5 / 6 and "n_packages" not in rows[0]
    assert json.dumps(rows) == json.dumps(BP.score(inst, e2)[1])  # same key order


# ----------------------------------------------------------------------------- Logistics end to end

_ASTAR = {}


def astar(s, goal):
    k = (s, goal)
    if k not in _ASTAR:
        _ASTAR[k] = L.astar_plan(s, goal, max_expansions=50_000)
    return _ASTAR[k]


def lg_candidate(s, goal, seed, rate):
    """A* plan from s; with probability `rate` (seeded) one action replaced by a random typed ground action."""
    plan = list(astar(s, goal) or [])
    if seed is not None and plan:
        r = random.Random(seed)
        if r.random() < rate:
            plan[r.randrange(len(plan))] = r.choice(L.ground_actions(L.objs_of(s), s))
    return plan


class ScriptedProposer:
    """Text-free proposer: K candidates per item, candidate j seeded as LMProposerD seeds it (CRN key)."""

    def __init__(self, K=4, rate=0.5, rate0=0.0):
        self.K, self.rate, self.rate0 = K, rate, rate0

    def propose(self, items, eval_seed):
        out = []
        for it in items:
            cands = []
            for j in range(self.K):
                p = lg_candidate(it["state"], it["goal"], BP._seed(eval_seed, it["id"], it["round"], j),
                                 self.rate if j else self.rate0)
                cands.append({"plan": p or None, "claimed": None, "n_tokens": 13 * len(p)})
            out.append(cands)
        return out


def lg_forward_handler(refuse=(), garbage=()):
    """Fake F adapter: reads state and action from the prompt and answers the exact delta (or a refusal)."""
    def h(prompt, seed):
        system, user = split_chat(prompt)
        assert system == L.SYS_FORWARD_DELTA and seed is None
        d = json.loads(user)
        if d["action"] in refuse:
            return "I'm sorry, I can't help with that request."
        if d["action"] in garbage:
            return '{"add":["at p0 l'
        s = L.state_from_json(d["state"])
        ns, ok = L.step(s, tuple(d["action"].split()))
        return L.forward_target_delta(s, ns, ok)
    return h


def lg_proposer_handler(rate=0.5):
    """Fake Pa adapter: greedy (seed None) = the A* plan from the prompt's state; sampled = lg_candidate."""
    def h(prompt, seed):
        system, user = split_chat(prompt)
        assert system == L.SYS_PROPOSER
        d = json.loads(user)
        assert d["outcome"] == "success"
        s = L.state_from_json(d["state"])
        goal = frozenset(("at", p, l) for p, l in d["goal"].items())
        return L.target_actions(lg_candidate(s, goal, seed, rate))
    return h


@pytest.fixture(scope="module")
def lg_instances(tmp_path_factory):
    rng = random.Random(LG_SEED)
    d = tmp_path_factory.mktemp("lg_problems")
    out = []
    for n in (1, 2, 2, 3, 3):
        while True:
            g = L.generate(rng, n)
            if g is None:
                continue
            p = L.astar_plan(g[1], g[2], max_expansions=20_000)
            if p is not None and len(p) >= 4:
                break
        objs, init, goal = g
        i = len(out)
        path = d / f"lg{i}.pddl"
        path.write_text(L.write_problem(objs, init, goal, name=f"lg{i}"))
        out.append({"id": f"lg{i}", "objs": objs, "init": init, "goal": goal, "n_packages": n, "path": path,
                    "opt": len(p)})
    assert L.state_from_json(L.state_json(out[-1]["init"])) == out[-1]["init"]
    assert max(it["opt"] for it in out) > 12  # some plans exceed the C1 rollout window R = 12
    return out


def _check_scored(inst, eps, val=True):
    """score_d vs logistics.execute and VAL, per plan; returns (summary, rows)."""
    summ, rows = DP.score_d(inst, eps, L)
    by = {it["id"]: it for it in inst}
    for e, r in zip(eps, rows):
        it = by[e["id"]]
        ex = L.execute(it["init"], it["goal"], e["plan"] or [])
        assert (r["success"], r["valid"]) == (ex["success"], ex["valid"])
        assert r["n_blocks"] == r["n_packages"] == it["n_packages"] == len(it["objs"]["obj"])
        assert [tuple(x.split()) for x in r["plan"]] == list(e["plan"] or [])
        if val:
            assert L.val_check(it["path"], [tuple(x.split()) for x in r["plan"]]) == r["success"]
        if e["verified"]:
            assert r["success"]  # exact F: a verified plan is a valid plan
    return summ, rows


@NEEDS_VAL
@pytest.mark.parametrize("W,E,R", [(4, 24, 12), (4, 12, 24), (2, 6, 48), (4, 24, 0)])
def test_plan_beam_logistics_exactd(lg_instances, W, E, R):
    fwd = DP.ExactD(L)
    eps = DP.plan_beam(lg_instances, ScriptedProposer(K=4, rate=0.5), fwd, eval_seed=0, width=W, max_expansions=E,
                       max_depth=100, max_rollout=R)
    summ, rows = _check_scored(lg_instances, eps)
    assert summ["success"] == 1.0 and summ["verified_rate"] == 1.0 and summ["verification_precision"] == 1.0
    assert all(r["len"] == it["opt"] for r, it in zip(rows, lg_instances))  # A* candidates: optimal plans
    if R:
        long_ = [e for e, it in zip(eps, lg_instances) if it["opt"] > R]
        assert all(e["expansions"] > 1 for e in long_)  # built across expansions
    assert fwd.calls > 0


@NEEDS_VAL
def test_plan_beam_logistics_noisy_greedy(lg_instances):
    """Candidate 0 is perturbed too (rate 0.5): the beam still reaches every goal under exact F."""
    eps = DP.plan_beam(lg_instances, ScriptedProposer(K=4, rate=0.5, rate0=0.5), DP.ExactD(L), eval_seed=1, width=4,
                       max_expansions=24, max_depth=100, max_rollout=12)
    summ, _ = _check_scored(lg_instances, eps)
    assert summ["success"] == 1.0


@NEEDS_VAL
def test_plan_direct_logistics(lg_instances):
    eps = DP.plan_direct(lg_instances, ScriptedProposer(K=1))
    summ, rows = _check_scored(lg_instances, eps)
    assert summ["success"] == 1.0 and summ["verified_rate"] == 0.0
    eps = DP.plan_direct(lg_instances, ScriptedProposer(K=1, rate0=0.6), eval_seed=2)
    summ, rows = _check_scored(lg_instances, eps)
    assert 0 < summ["success"] < 1  # some perturbed greedy plans fail: execute and VAL agree on both outcomes
    assert any(not r["valid"] for r in rows)


def test_lmforwardd_stub_lm_logistics_cache(lg_instances, tmp_path):
    it = lg_instances[-1]
    plan = astar(it["init"], it["goal"])
    states = [it["init"]]
    for a in plan:
        states.append(L.step(states[-1], a)[0])
    keys = []
    for s in states[:6]:
        app = L.applicable_actions(s, self_loops=True)
        keys += [(s, a) for a in app[:4]]
        keys += [(s, a) for a in L.ground_actions(L.objs_of(s), s) if not L.applicable(s, a)][:3]
    objs = it["objs"]
    keys.append((states[0], ("load-truck", objs["obj"][0], objs["airplane"][0], objs["location"][0])))  # type-wrong
    keys.append((states[0], ("drive-truck", objs["truck"][0], objs["location"][0], objs["location"][0],
                             objs["city"][0])))  # self-loop (legal no-op if t0 is there, else illegal)
    refuse = {L.action_str(keys[2][1])}
    garbage = {L.action_str(keys[9][1])}
    keys += keys[:5]  # duplicates
    uniq = list(dict.fromkeys(keys))
    model, tok = FakeLM({"fwd": lg_forward_handler(refuse, garbage)}), FakeTok()
    cache = tmp_path / "lg_fcache.jsonl"
    fwd = DP.LMForwardD(model, tok, "fwd", cache=cache, dom=L)
    out = fwd.predict(keys)
    bad = lambda k: L.action_str(k[1]) in refuse | garbage
    for k, o in zip(keys, out):
        assert o == (None if bad(k) else L.step(*k)), k
    assert any(o is not None and not o[1] for o in out) and any(o is not None and o[1] for o in out)
    assert sum(o is None for o in out) >= 2
    assert model.n_generations == fwd.model_queries == len(uniq) == len(cache.read_text().splitlines())
    assert model.adapter_log == ["fwd"] and model.log[0][3] == 40
    assert fwd.model_tokens == sum(tok.n_tokens(lg_forward_handler(refuse, garbage)(L.forward_prompt_delta(*k), None))
                                   + 1 for k in uniq)
    # second predict: all memo hits, no model call
    assert fwd.predict(list(reversed(keys))) == list(reversed(out))
    assert model.n_generations == fwd.model_queries == len(uniq) and len(model.log) == 1
    # partial overlap: only the new key is generated
    extra = (states[6], L.applicable_actions(states[6])[0])
    assert fwd.predict(keys[:3] + [extra]) == out[:3] + [L.step(*extra)]
    assert model.n_generations == fwd.model_queries == len(uniq) + 1 and len(model.log[-1][2]) == 1
    # reload from the cache (crash-torn last line skipped): parses with the domain set before super().__init__
    with open(cache, "a", encoding="utf-8") as fh:
        fh.write('{"s": [["at", "p0"')
    m2 = FakeLM({"fwd": lg_forward_handler()})
    fwd2 = DP.LMForwardD(m2, tok, "fwd", cache=cache, dom=L)
    assert fwd2.memo == fwd.memo and len(fwd2.memo) == len(uniq) + 1
    for line in cache.read_text().splitlines()[:-1]:
        r = json.loads(line)
        k = _key(r)
        assert fwd2.memo[k] == L.parse_forward_delta(r["text"], k[0])
    assert fwd2.predict(keys) == out and m2.n_generations == 0 and fwd2.model_queries == 0
    # truncated outputs (decode cap too small for the delta) are unparseable -> None
    m3 = FakeLM({"fwd": lg_forward_handler()})
    fwd3 = DP.LMForwardD(m3, tok, "fwd", dom=L, max_new=6)
    assert fwd3.predict(uniq[:5]) == [None] * 5 and m3.log[0][3] == 6
    # explicit malformed answers
    s, a = uniq[0]
    for text in ('{"add":[],"del":["at p9 l9-9"],"legal":true}', '{"add":[],"del":[],"legal":"true"}',
                 '{"add":[],"del":[]}', "", "legal: true"):
        m4 = FakeLM({"fwd": lambda p, sd, t=text: t})
        assert DP.LMForwardD(m4, tok, "fwd", dom=L).predict([(s, a)]) == [None]


def test_lmproposerd_stub_lm_logistics_truncation(lg_instances):
    it = lg_instances[-1]
    plan = astar(it["init"], it["goal"])
    item = {"id": it["id"], "objs": it["objs"], "goal": it["goal"], "state": it["init"], "round": "0.0"}
    tok = FakeTok()
    full = len(L.target_actions(plan))
    M = (full // 2) // tok.CH  # cuts the plan in the middle
    m = FakeLM({"pa": lg_proposer_handler(rate=0.0)})
    out = DP.LMProposerD(m, tok, "pa", K=2, max_new=M, prefix_ok=True, dom=L).propose([item], 0)[0]
    pre = out[0]["plan"]
    assert pre and 0 < len(pre) < len(plan) and pre == plan[:len(pre)] and out[0]["n_tokens"] == M
    assert m.log[0][4] == M and m.log[0][3] == [[None, BP._seed(0, it["id"], "0.0", 1)]]
    out = DP.LMProposerD(FakeLM({"pa": lg_proposer_handler(rate=0.0)}), tok, "pa", K=2, max_new=M, dom=L).propose(
        [item], 0)[0]
    assert out[0]["plan"] is None  # no prefix acceptance: a truncated plan is unparseable
    out = DP.LMProposerD(FakeLM({"pa": lg_proposer_handler(rate=0.0)}), tok, "pa", K=2, max_new=704, dom=L).propose(
        [item], 0)[0]
    assert out[0]["plan"] == plan and out[0]["claimed"] is None


# ----------------------------------------------------------------------------- unrenderable F outputs, missing objs

# accepted by the shared B2 delta parser, but the next state holds the empty fact (): both renderers raise IndexError
DEGENERATE = ('{"add":[""],"del":[],"legal":true}', '{"add":["   "],"del":[],"legal":false}',
              '{"add":"at p0 l0-1","del":[],"legal":true}')


def alias_slip(delta_text):
    """F slip on an unload: 'in p l' written for 'at p l'. Rendered exactly like the true next state, but not goal_met
    (the review's LG-DP-2 case); deltas that add no package 'at' fact are unchanged."""
    return delta_text.replace('"add":["at p', '"add":["in p', 1)


def test_lmforwardd_rejects_unrenderable_states(lg_instances, tmp_path):
    """Logistics: an F output whose next state does not round-trip through state_json is unsupported (None), live
    and on cache reload; correct outputs are unaffected; a non-renderable key state is a harness error."""
    it = lg_instances[2]
    s, goal, objs = it["init"], it["goal"], it["objs"]
    plan = astar(s, goal)
    states = [s]
    for a in plan:
        states.append(L.step(states[-1], a)[0])
    t = next(i for i, a in enumerate(plan) if a[0].startswith("unload"))
    st, un = states[t], plan[t]
    true_ns = L.step(st, un)[0]
    p, tr, loc = objs["obj"][0], objs["truck"][0], objs["location"][0]
    acts = [a for a in L.ground_actions(objs, s) if not L.applicable(s, a)]
    # each bad answer on its own key: (key, raw F text)
    bad = [((s, acts[i]), text) for i, text in enumerate(DEGENERATE)]
    bad += [((st, un), alias_slip(L.forward_target_delta(st, true_ns, True))),
            ((s, acts[3]), '{"add":["at %s"],"del":[],"legal":false}' % p),  # 2-ary 'at': not rendered
            ((s, acts[4]), '{"add":["location l9-9"],"del":[],"legal":false}'),  # a location in no city
            ((s, acts[5]), '{"add":["at %s %s"],"del":[],"legal":false}' % (p, tr))]  # 'at' a truck: shown as 'in'
    assert len({k for k, _ in bad}) == len(bad)
    for (ks, ka), text in bad:  # all accepted by the raw (B2) delta parse
        raw = L.parse_forward_delta(text, ks)
        assert raw is not None and raw[0] != L.step(ks, ka)[0]
        try:
            assert L.state_from_json(L.state_json(raw[0])) != raw[0]
        except IndexError:
            assert () in raw[0]
    for text in DEGENERATE:  # the crash the render check prevents
        with pytest.raises(IndexError):
            L.forward_prompt_delta(L.parse_forward_delta(text, s)[0], plan[0])
        with pytest.raises(IndexError):
            L.proposer_prompt(L.parse_forward_delta(text, s)[0], goal)
    alias = L.parse_forward_delta(bad[3][1], st)[0]  # the alias: same prompts as the true state, goal fact missing
    assert alias != true_ns and ("at",) + un[1:2] + un[3:] in true_ns - alias
    assert L.forward_prompt_delta(alias, plan[0]) == L.forward_prompt_delta(true_ns, plan[0])
    assert L.proposer_prompt(alias, goal) == L.proposer_prompt(true_ns, goal)
    # legal, illegal and goal-reaching keys: exact answers still parse
    good = [(s, plan[0]), (s, acts[6])] + [k for k in [(states[-2], plan[-1])] if k != (st, un)]
    answers = {L.forward_prompt_delta(*k): x for k, x in bad}
    exact = lg_forward_handler()
    model, tok = FakeLM({"fwd": lambda pr, sd: answers.get(pr) or exact(pr, sd)}), FakeTok()
    cache = tmp_path / "fc.jsonl"
    fwd = DP.LMForwardD(model, tok, "fwd", cache=cache, dom=L)
    keys = [k for k, _ in bad] + good
    assert fwd.predict(keys) == [None] * len(bad) + [L.step(*k) for k in good]
    assert fwd.model_queries == len(keys) == len(cache.read_text().splitlines())
    # reload: the raw texts are re-parsed under the same rule (a degenerate line no longer replays a crash)
    fwd2 = DP.LMForwardD(FakeLM({}), tok, "fwd", cache=cache, dom=L)
    assert fwd2.memo == fwd.memo and fwd2.predict(keys) == fwd.predict(keys) and fwd2.model_queries == 0
    # a key state the renderer cannot show exactly (e.g. an instance init with a location in no city) is refused
    m3 = FakeLM({"fwd": exact})
    f3 = DP.LMForwardD(m3, tok, "fwd", dom=L)
    for ks in (s | {("location", "l9-9")}, s | {("in", p, loc)}, s | {()}):
        with pytest.raises(ValueError, match="not renderable"):
            f3.predict([(s, plan[0]), (ks, plan[0])])
    assert m3.n_generations == f3.model_queries == 0 and not f3.memo


def test_plan_beam_survives_unrenderable_f_outputs(lg_instances, tmp_path):
    """End to end: a fake F that emits degenerate deltas and 'in'-for-'at' slips on a fixed subset of keys. The
    beam completes (the unguarded B2 rule crashes on the same run), every accepted state renders exactly, planner
    keys and F prompts correspond one to one, verified plans are valid, and the cached rerun is identical."""
    exact = lg_forward_handler()

    def slippy(prompt, seed):
        text = exact(prompt, seed)
        r = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) % 25
        return (DEGENERATE[0], DEGENERATE[2])[r] if r < 2 else (alias_slip(text) if r < 7 else text)

    tok = FakeTok()
    kw = dict(eval_seed=0, width=4, max_expansions=24, max_depth=100, max_rollout=12)

    def run(cache, roundtrip=True):
        model = FakeLM({"fwd": slippy, "pa": lg_proposer_handler(rate=0.5)})
        fwd = DP.LMForwardD(model, tok, "fwd", cache=cache, dom=L)
        fwd.roundtrip = roundtrip
        prop = DP.LMProposerD(model, tok, "pa", K=4, max_new=14 * 12 + 16, prefix_ok=True, dom=L)
        return DP.plan_beam(lg_instances, prop, fwd, **kw), fwd

    with pytest.raises(IndexError):  # without the render check (B2's rule) a degenerate state reaches a renderer
        run(None, roundtrip=False)
    eps, fwd = run(tmp_path / "fc.jsonl")
    summ, rows = _check_scored(lg_instances, eps, val=False)  # verified => valid (asserted inside)
    slips = {k: slippy(L.forward_prompt_delta(*k), None) for k in fwd.memo}
    wrong = [k for k, x in slips.items() if x != exact(L.forward_prompt_delta(*k), None)]
    assert any(slips[k] in DEGENERATE for k in wrong) and any(slips[k] not in DEGENERATE for k in wrong)  # both kinds
    assert all(fwd.memo[k] is None for k in wrong)
    assert all(v == L.step(*k) for k, v in fwd.memo.items() if k not in wrong)
    assert all(fwd.renders(v[0]) for v in fwd.memo.values() if v is not None)
    prompts = {L.forward_prompt_delta(*k) for k in fwd.memo}
    assert len(prompts) == len(fwd.memo) == fwd.model_queries  # one F query per distinct prompt
    assert summ["n"] == len(lg_instances) and summ["verified_rate"] > 0
    eps2, fwd2 = run(tmp_path / "fc.jsonl")
    assert eps2 == eps and fwd2.model_queries == 0


def test_lmproposerd_requires_objs(lg_instances):
    """Logistics items without objs fail loudly before any generation (the parsers would reject every candidate,
    every instance ending dead_end); score_d has no silent size fallback; Blocksworld keeps the parent's path."""
    noobjs = [{k: v for k, v in it.items() if k != "objs"} for it in lg_instances]
    tok = FakeTok()
    m = FakeLM({"pa": lg_proposer_handler(rate=0.0)})
    prop = DP.LMProposerD(m, tok, "pa", K=4, max_new=184, prefix_ok=True, dom=L)
    with pytest.raises(ValueError, match="objs missing"):
        DP.plan_beam(noobjs, prop, DP.ExactD(L), width=4, max_expansions=24, max_depth=100, max_rollout=12)
    item = {"id": "x", "objs": None, "goal": noobjs[0]["goal"], "state": noobjs[0]["init"], "round": "0.0"}
    with pytest.raises(ValueError, match="objs missing"):
        prop.propose([item], 0)
    assert m.n_generations == 0 and m.log == [] and m.adapter_log == []
    eps = DP.plan_direct(lg_instances, ScriptedProposer(K=1))
    assert DP.score_d(lg_instances, eps, L)[0]["success"] == 1.0
    with pytest.raises(ValueError, match="no 'objs'"):
        DP.score_d(noobjs, eps, L)
    # Blocksworld (parity): no objs check is added; both classes give the same all-unparseable candidates
    bw = [dict(x, objs=None) for x in _bw_items(4)]
    outs = []
    for cls, extra in ((BP.LMProposerBW, {}), (DP.LMProposerD, {"dom": B})):
        outs.append(cls(FakeLM({"pa": _bw_handler("actions")}), tok, "pa", K=3, **extra).propose(bw, 0))
    assert outs[0] == outs[1] and all(c["plan"] is None for cands in outs[1] for c in cands)


@NEEDS_VAL
def test_stub_lm_end_to_end_logistics(lg_instances, tmp_path):
    """LMProposerD + LMForwardD over a perfect fake LM == ExactD + the scripted proposer (same candidates, same
    seeds); F cache reuse changes no result and costs no model call; plan_direct at the S-direct M."""
    R = 12
    M = 14 * R + 16
    tok = FakeTok()
    model = FakeLM({"fwd": lg_forward_handler(), "pa": lg_proposer_handler(rate=0.5)})
    fwd = DP.LMForwardD(model, tok, "fwd", cache=tmp_path / "fc.jsonl", dom=L)
    prop = DP.LMProposerD(model, tok, "pa", K=4, max_new=M, prefix_ok=True, dom=L)
    kw = dict(eval_seed=0, width=4, max_expansions=24, max_depth=100, max_rollout=R)
    eps = DP.plan_beam(lg_instances, prop, fwd, **kw)
    ref = DP.plan_beam(lg_instances, ScriptedProposer(K=4, rate=0.5), DP.ExactD(L), **kw)
    strip = lambda es: [{k: v for k, v in e.items() if k != "gen_tokens"} for e in es]
    assert strip(eps) == strip(ref)
    summ, rows = _check_scored(lg_instances, eps)
    assert summ["success"] == 1.0 and summ["verified_rate"] == 1.0
    assert all(e["gen_tokens"] > 0 for e in eps)
    assert {x[0] for x in model.log} == {"greedy", "sample"}
    assert {x[4] for x in model.log if x[0] == "sample"} == {M}  # proposer cap M = 14R + 16
    assert {x[3] for x in model.log if x[0] == "greedy"} == {40}  # F cap
    assert set(model.adapter_log) == {"fwd", "pa"} and fwd.model_queries > 0
    n_lines = len((tmp_path / "fc.jsonl").read_text().splitlines())
    assert n_lines == fwd.model_queries
    # frozen F: a second run on the cache reproduces everything with zero new F model queries
    m2 = FakeLM({"fwd": lg_forward_handler(), "pa": lg_proposer_handler(rate=0.5)})
    fwd2 = DP.LMForwardD(m2, tok, "fwd", cache=tmp_path / "fc.jsonl", dom=L)
    eps2 = DP.plan_beam(lg_instances, DP.LMProposerD(m2, tok, "pa", K=4, max_new=M, prefix_ok=True, dom=L), fwd2,
                        **kw)
    assert eps2 == eps and fwd2.model_queries == 0 and all(x[0] == "sample" for x in m2.log)
    assert len((tmp_path / "fc.jsonl").read_text().splitlines()) == n_lines
    # S-direct: same Pa, greedy, K = 1, M = 704
    m3 = FakeLM({"pa": lg_proposer_handler(rate=0.5)})
    eps3 = DP.plan_direct(lg_instances, DP.LMProposerD(m3, tok, "pa", K=1, max_new=704, dom=L))
    summ3, rows3 = _check_scored(lg_instances, eps3)
    assert summ3["success"] == 1.0 and m3.log[0][3] == [[None]] * len(lg_instances) and m3.log[0][4] == 704
    assert [r["len"] for r in rows3] == [it["opt"] for it in lg_instances]
