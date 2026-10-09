"""Shared trace-target schema (joint roadmap item 5b, DECISIONS D-52; agreed with RCWM-T1).

One JSON object with keys in FIXED order: {"cmp": <predicate inputs + "ok">, "calc": <chosen-branch arithmetic>,
"after": <the domain's existing answer format>}. Serialized with explicit order (never sorted); `parse_trace` rejects
any other key order, so the "compute before answering" order stays literal. Arms built from the same keys:
  plain   the existing answer only (= `after`)
  trace   cmp -> calc -> after
  filler  cmp/calc replaced by a neutral token run matched per row to the trace's token count (extra-compute control)
Decision pipelines read only `after`; `cmp_ok` / `calc_ok` are scored separately as mechanism.
"""
from __future__ import annotations

import json

from . import blocks as B
from . import rw2

FILL = " ."  # neutral filler unit (one token in the Qwen3 tokenizer)


def _dump(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


# ----------------------------------------------------------------------------- RegisterWorld-v2 legality

def rw2_parts(x: int, action: str, L: int):
    nx, legal, term = rw2.exact_step(x, action, L)
    after = json.loads(rw2.forward_target(nx, legal, term))
    if action in ("add1", "sub1", "double"):
        c = {"add1": x + 1, "sub1": x - 1, "double": 2 * x}[action]
        slack = L - abs(c)
        cmp = {"cand": c, "slack": slack, "ok": slack >= 0}
        calc = c if slack >= 0 else x
    elif action == "halve":
        cmp = {"x": x, "rem": x % 2, "ok": x % 2 == 0}
        calc = x // 2 if x % 2 == 0 else x
    else:  # finish
        cmp, calc = None, x
    return cmp, calc, after


def rw2_target(x: int, action: str, L: int, arm: str, tok=None) -> str:
    cmp, calc, after = rw2_parts(x, action, L)
    return _target(cmp, calc, after, arm, tok)


# ----------------------------------------------------------------------------- Blocksworld preconditions

def bw_preconditions(a: tuple) -> list[tuple]:
    op = a[0]
    if op == "pick-up":
        return [("clear", a[1]), ("ontable", a[1]), ("handempty",)]
    if op == "put-down":
        return [("holding", a[1])]
    if op == "stack":
        return [("holding", a[1]), ("clear", a[2])]
    return [("on", a[1], a[2]), ("clear", a[1]), ("handempty",)]  # unstack


def bw_parts(s: frozenset, a: tuple):
    ns, legal = B.step(s, a)
    cmp = {B.fact_str(f): f in s for f in bw_preconditions(a)}
    cmp["ok"] = legal
    calc = {"add": sorted(B.fact_str(f) for f in ns - s), "del": sorted(B.fact_str(f) for f in s - ns)}
    after = json.loads(B.forward_target_delta(s, ns, legal))
    return cmp, calc, after


def bw_target(s: frozenset, a: tuple, arm: str, tok=None) -> str:
    cmp, calc, after = bw_parts(s, a)
    return _target(cmp, calc, after, arm, tok)


# ----------------------------------------------------------------------------- arms and parsing

def _target(cmp, calc, after, arm: str, tok=None) -> str:
    if arm == "plain":
        return _dump(after)
    trace = '{"cmp":' + _dump(cmp) + ',"calc":' + _dump(calc) + ',"after":' + _dump(after) + "}"
    if arm == "trace":
        return trace
    if arm == "filler":
        if tok is None:
            raise ValueError("filler arm needs the tokenizer to match token counts")
        head = '{"cmp":' + _dump(cmp) + ',"calc":' + _dump(calc)
        n = len(tok(head, add_special_tokens=False).input_ids)
        base = '{"fill":"'
        k = max(1, n - len(tok(base + '"', add_special_tokens=False).input_ids))
        return base + FILL * k + '","after":' + _dump(after) + "}"
    raise ValueError(arm)


def parse_trace(text: str):
    """(cmp, calc, after) from a trace answer, or None. Keys must be exactly cmp, calc, after in that order."""
    try:
        d = json.loads(text.strip())
    except Exception:
        return None
    if not isinstance(d, dict) or list(d) != ["cmp", "calc", "after"]:
        return None
    return d["cmp"], d["calc"], d["after"]


def after_text(text: str, arm: str) -> str | None:
    """The `after` part re-serialized for the domain's existing parser (plain: the text itself)."""
    if arm == "plain":
        return text
    try:
        d = json.loads(text.strip())
    except Exception:
        return None
    if arm == "trace" and (not isinstance(d, dict) or list(d) != ["cmp", "calc", "after"]):
        return None
    if arm == "filler" and (not isinstance(d, dict) or list(d) != ["fill", "after"]):
        return None
    return json.dumps(d["after"], separators=(",", ":"), sort_keys=True)
