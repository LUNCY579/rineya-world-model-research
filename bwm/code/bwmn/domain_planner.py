"""Domain-generic wrappers around the pinned Blocksworld planner (item 9 CPU-3; PREREG_LOGISTICS_DRAFT 'Systems').

bwmn/blocks_planner.py is imported read-only and never patched: its file SHA-256 must equal BLOCKS_PLANNER_SHA256
(D-63) or this module refuses to import. plan_beam, plan_direct and rollout_many are reused byte-identical with their
default hooks (goal_met = goal <= state, unmet = |goal - state|), so the domain rules stay inside F.

A domain module `dom` (bwmn.blocks or bwmn.logistics) supplies:
  ExactD:      step(s, a) -> (next_state, legal) and applicable(s, a)
  LMForwardD:  forward_prompt_delta(s, a), parse_forward_delta(text, s)
  LMProposerD: proposer_prompt(s, goal), parse_plan(text, objs, mode), parse_plan_prefix(text, objs)
  score_d:     execute(init, goal, plan), action_str(a)

LMForwardD / LMProposerD subclass LMForwardBW / LMProposerBW. The domain is set BEFORE super().__init__ because the
inherited cache loader calls self._parse (blocks_planner.py line 61). predict / propose are copies of the parent
methods (lines 66-88 and 115-131) with the domain's prompt and parser functions; everything else (memo, JSONL cache
format and fsync, model_queries / model_tokens accounting, CRN seeds, candidate-0 greedy, adapter switching, the
item 'objs' handed to the parser unchanged) is the parent's. With dom = bwmn.blocks they reproduce LMForwardBW
(fmt="delta") and LMProposerBW string for string (tests/test_domain_planner.py).

Two rules apply to every domain other than bwmn.blocks (Blocksworld keeps B2's behaviour exactly, for parity):
  * F next states must be renderable keys. An F output is accepted only if its next state s' satisfies
    dom.state_from_json(dom.state_json(s')) == s', i.e. s' is exactly the state the models are shown next; otherwise
    the output is unsupported (None), like any unparseable output. Exact states always pass, so no correct output is
    affected. Rejected: degenerate facts such as "" (the empty tuple, which crashes the renderer), and facts the
    rendering hides or rewrites (e.g. 'in p0 l0-0', rendered as if p0 were at l0-0 but failing goal_met). Planner keys
    and model prompts therefore correspond one to one. A key whose state is not renderable (an instance's init) is a
    harness error: predict raises ValueError.
  * Proposer items must carry 'objs' (the domain parsers need them): propose raises ValueError before generating.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

BLOCKS_PLANNER_SHA256 = "18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960"  # D-63 pin
_HERE = Path(__file__).resolve().parent


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# check the pinned file BEFORE executing it, and again on the module object actually imported
if file_sha256(_HERE / "blocks_planner.py") != BLOCKS_PLANNER_SHA256:
    raise ImportError(f"bwmn/blocks_planner.py sha256 {file_sha256(_HERE / 'blocks_planner.py')} != pinned "
                      f"{BLOCKS_PLANNER_SHA256} (D-63); domain_planner refuses to run on a modified planner")

from . import blocks as B  # noqa: E402
from . import blocks_planner as BP  # noqa: E402

if Path(BP.__file__).resolve() != (_HERE / "blocks_planner.py").resolve() or file_sha256(BP.__file__) != \
        BLOCKS_PLANNER_SHA256:
    raise ImportError(f"imported blocks_planner ({BP.__file__}) is not the pinned file {BLOCKS_PLANNER_SHA256}")

# the planner itself, reused unchanged (re-exported so callers need one import)
plan_beam = BP.plan_beam
plan_direct = BP.plan_direct
rollout_many = BP.rollout_many

F_MAX_NEW = 40  # B2's F decode cap (blocks_planner.py line 73); Logistics delta targets are <= 26 tokens with EOS


def _require(dom, names, who):
    miss = [n for n in names if not callable(getattr(dom, n, None))]
    if miss:
        raise TypeError(f"{who}: domain module {getattr(dom, '__name__', dom)!r} lacks {miss}")


def source_shas() -> dict:
    """SHA-256 of the planner stack's source files (for freeze records); lm.py is hashed, never imported here."""
    names = ("blocks_planner.py", "blocks.py", "lm.py", "logistics.py", "domain_planner.py")
    return {n: file_sha256(_HERE / n) for n in names if (_HERE / n).exists()}


class ExactD:
    """Exact F for any domain module exposing step/applicable: (next_state, legal) per key, the oracle ceiling and
    harness instrument (never used by a learned system while planning)."""
    name = "exact"

    def __init__(self, dom):
        _require(dom, ("step", "applicable"), "ExactD")
        self.dom = dom
        self.calls = 0

    def predict(self, keys):
        self.calls += len(keys)
        return [self.dom.step(s, a) for s, a in keys]


class LMForwardD(BP.LMForwardBW):
    """LMForwardBW (delta format) with the domain's F prompt and parser. Goal-blind, memoised, optional JSONL cache
    (format unchanged: {"s": sorted facts, "a": action, "text": raw output}, re-parsed on load); unparseable or refused
    output -> None, and for domains other than bwmn.blocks also an output whose next state is not renderable (module
    docstring). max_new: F decode cap (default 40, as B2)."""

    def __init__(self, model, tok, adapter="fwd", fmt="delta", cache=None, max_batch_tokens=2000, *, dom,
                 max_new=F_MAX_NEW):
        if fmt != "delta":
            raise ValueError("LMForwardD supports only the delta format")
        self.roundtrip = dom is not B  # B2 parity: Blocksworld keeps blocks_planner's parse rule unchanged
        _require(dom, ("forward_prompt_delta", "parse_forward_delta")
                 + (("state_json", "state_from_json") if self.roundtrip else ()), "LMForwardD")
        self.dom = dom  # before super().__init__: its cache loader calls self._parse
        self.max_new = max_new
        super().__init__(model, tok, adapter=adapter, fmt=fmt, cache=cache, max_batch_tokens=max_batch_tokens)

    def renders(self, s) -> bool:
        """s is exactly the state the models are shown: dom.state_from_json(dom.state_json(s)) == s, no exception."""
        try:
            return self.dom.state_from_json(self.dom.state_json(s)) == s
        except Exception:
            return False

    def _parse(self, text, s):
        r = self.dom.parse_forward_delta(text, s)
        if r is None or not self.roundtrip:
            return r
        return r if self.renders(r[0]) else None

    def predict(self, keys):
        need = list(dict.fromkeys(k for k in keys if k not in self.memo))
        if need and self.roundtrip:  # harness check (instance inits); F's own next states always pass
            bad = [k for k in need if not self.renders(k[0])]
            if bad:
                raise ValueError(f"LMForwardD: {len(bad)} key state(s) not renderable by "
                                 f"{getattr(self.dom, '__name__', self.dom)}.state_json (the models would be shown a "
                                 f"different state), e.g. {sorted(bad[0][0])}")
        if need:
            self.model_queries += len(need)
            self.model.set_adapter(self.adapter)
            prompts = [self.dom.forward_prompt_delta(s, a) for s, a in need]
            gens = BP.lm.greedy_shared_prefix(self.model, self.tok, prompts, max_new_tokens=self.max_new,
                                              max_batch_tokens=self.max_batch_tokens)
            for k, g in zip(need, gens):
                self.model_tokens += g["n_tokens"]
                self.memo[k] = self._parse(g["text"], k[0])
            if self.cache is not None:
                with open(self.cache, "a", encoding="utf-8") as fh:
                    for k, g in zip(need, gens):
                        fh.write(json.dumps({"s": sorted(map(list, k[0])), "a": list(k[1]), "text": g["text"]}) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        return [self.memo[k] for k in keys]


class LMProposerD(BP.LMProposerBW):
    """LMProposerBW with the domain's proposer prompt and plan parsers. max_new is the generation cap M (required for
    any domain other than bwmn.blocks, whose B2 defaults 120 / 700 would silently truncate longer plans).
    prefix_ok: accept the leading complete actions of a truncated output (actions mode). For domains other than
    bwmn.blocks every item must carry 'objs' (else ValueError before any generation: the domain parsers would reject
    every candidate silently)."""

    def __init__(self, model, tok, adapter, mode="actions", K=8, temperature=1.0, top_p=0.95, max_new=None,
                 prefix_ok=False, max_batch_tokens=2000, *, dom):
        _require(dom, ("proposer_prompt", "parse_plan") + (("parse_plan_prefix",) if prefix_ok else ()),
                 "LMProposerD")
        if max_new is None and dom is not B:
            raise ValueError(f"LMProposerD: max_new (M) is required for domain {getattr(dom, '__name__', dom)!r}")
        self.dom = dom
        super().__init__(model, tok, adapter, mode=mode, K=K, temperature=temperature, top_p=top_p, max_new=max_new,
                         prefix_ok=prefix_ok, max_batch_tokens=max_batch_tokens)

    def propose(self, items, eval_seed):
        if self.dom is not B:  # B2 parity: Blocksworld items are passed to the parser as the parent passes them
            miss = [it.get("id") for it in items if it.get("objs") is None]
            if miss:
                raise ValueError(f"LMProposerD: item objs missing for {len(miss)} item(s), e.g. id {miss[0]!r} "
                                 f"(needed by {getattr(self.dom, '__name__', self.dom)}.parse_plan)")
        prompts = [self.dom.proposer_prompt(it["state"], it["goal"]) for it in items]
        seeds = [[None if j == 0 else BP._seed(eval_seed, it["id"], it["round"], j) for j in range(self.K)]
                 for it in items]
        self.model.set_adapter(self.adapter)
        rows = BP.lm.sample_crn_shared(self.model, self.tok, prompts, seeds, max_new_tokens=self.max_new,
                                       temperature=self.temperature, top_p=self.top_p,
                                       max_batch_tokens=self.max_batch_tokens)
        out = [[] for _ in items]
        owner = [i for i in range(len(items)) for _ in range(self.K)]
        gens = [g for row in rows for g in row]
        for i, g in zip(owner, gens):
            objs = items[i]["objs"]  # passed through unchanged (a block list, or Logistics objects by type)
            parsed = (self.dom.parse_plan_prefix(g["text"], objs) if self.prefix_ok and self.mode == "actions"
                      else self.dom.parse_plan(g["text"], objs, self.mode))
            out[i].append({"plan": parsed[0] if parsed else None, "claimed": parsed[1] if parsed else None,
                           "n_tokens": g["n_tokens"]})
        return out


def _size_cols(it, dom) -> dict:
    """'n_blocks' is the size stratum bw_run_tests.merge groups on (line 45): the block count for an untyped object
    list, the package count for typed domains (objs = {type: names}, packages under 'obj'), which also get
    'n_packages'. Like BP.score, every instance must carry 'objs' (no fallback that could mask a loader bug)."""
    objs = it.get("objs")
    if objs is None:
        raise ValueError(f"score_d: instance {it.get('id')!r} has no 'objs'")
    if isinstance(objs, dict):
        n = len(objs["obj"])
        return {"n_blocks": n, "n_packages": n}
    return {"n_blocks": len(objs)}


def score_d(instances, eps, dom) -> tuple[dict, list]:
    """blocks_planner.score for any domain: every submitted plan is executed by dom.execute (exact VAL semantics).
    With dom = bwmn.blocks the output equals BP.score's exactly."""
    _require(dom, ("execute", "action_str"), "score_d")
    by = {it["id"]: it for it in instances}
    res = Counter()
    rows = []
    for e in eps:
        it = by[e["id"]]
        ex = dom.execute(it["init"], it["goal"], e["plan"] or [])
        res["n"] += 1
        res["success"] += ex["success"]
        res["verified"] += e["verified"]
        res["verified_true"] += e["verified"] and ex["success"]
        rows.append({"id": e["id"], **_size_cols(it, dom), "success": ex["success"], "valid": ex["valid"],
                     "len": len(e["plan"] or []),
                     "verified": e["verified"], "rounds": e["rounds"], "gen_tokens": e["gen_tokens"], "fail": e["fail"],
                     "plan": [dom.action_str(a) for a in (e["plan"] or [])]})
    return {"success": res["success"] / max(1, res["n"]), "n": res["n"],
            "verified_rate": res["verified"] / max(1, res["n"]),
            "verification_precision": res["verified_true"] / max(1, res["verified"])}, rows
