"""RCL world: exact interpreter, rendering, targets and strict parsing (SPEC_RCL.md).

This is the primary interpreter. It is qualified against an independently written
oracle (oracle_indep.py) and, on the posted-rule subset, against RCWM's
rcwmt.env.step (qualify.py).
"""
import hashlib
import json
import re

from . import templates as T

STATUS_OPS = ("S+", "S-")
EVENT_KEYS = ("account", "amount", "kind", "other", "status")


# ----------------------------------------------------------------------------- semantics

def eval_atom(atom, state, event):
    op = atom["op"]
    if op == "S+":
        return event["status"] in atom["set"]
    if op == "S-":
        return event["status"] not in atom["set"]
    if op == "K+":
        return event["kind"] in atom["kinds"]
    if op == "AGE":
        return event["amount"] >= atom["t"]
    if op == "ALT":
        return event["amount"] < atom["t"]
    if op == "BGE":
        return state[event["account"]] >= atom["t"]
    raise ValueError(f"unknown atom {op}")


def eval_rule(ast, state, event):
    op = ast["op"]
    if op == "AND":
        return all(eval_atom(a, state, event) for a in ast["args"])
    if op == "OR":
        return any(eval_atom(a, state, event) for a in ast["args"])
    return eval_atom(ast, state, event)


def status_atom(ast):
    """First status atom left to right, or None."""
    if ast["op"] in STATUS_OPS:
        return ast
    if ast["op"] in ("AND", "OR"):
        for a in ast["args"]:
            if a["op"] in STATUS_OPS:
                return a
    return None


def listed(ast, event):
    a = status_atom(ast)
    return None if a is None else (event["status"] in a["set"])


def touched(event):
    return [event["account"]] + ([event["other"]] if event["kind"] == "transfer" else [])


def applied(state, event):
    """State after applying the event unconditionally (key order preserved)."""
    after = dict(state)
    acc, amt, kind = event["account"], event["amount"], event["kind"]
    if kind == "add":
        after[acc] = state[acc] + amt
    elif kind == "transfer":
        after[acc] = state[acc] - amt
        after[event["other"]] = state[event["other"]] + amt
    elif kind == "scale":
        after[acc] = state[acc] * amt
    else:
        raise ValueError(f"unknown kind {kind}")
    return after


def transition(ast, state, event):
    ok = eval_rule(ast, state, event)
    return ok, (applied(state, event) if ok else dict(state))


# ----------------------------------------------------------------------------- rendering

def dumps_ordered(d, order=None):
    """Compact JSON object in the given key order (default: dict order)."""
    keys = list(order) if order is not None else list(d.keys())
    assert sorted(keys) == sorted(d.keys()), (keys, list(d.keys()))
    return "{" + ",".join(json.dumps(k) + ":" + json.dumps(d[k]) for k in keys) + "}"


def render_input(ast, state, event, tid="t0", state_order=None, event_order=None):
    rule = T.render_rule(ast, tid)
    st = dumps_ordered(state, state_order)
    ev = dumps_ordered(event, event_order or EVENT_KEYS)
    return f"RULE: {rule}\nSTATE: {st}\nEVENT: {ev}"


def target_ans(after, state_order):
    return '{"after":' + dumps_ordered(after, state_order) + "}"


def target_trace(ast, state, event, state_order):
    ok, after = transition(ast, state, event)
    lst = listed(ast, event)
    cmp_ = ('{"status":' + json.dumps(event["status"]) + ',"listed":' + json.dumps(lst)
            + ',"ok":' + json.dumps(ok) + "}")
    calc = "{" + ",".join(json.dumps(k) + ":" + json.dumps(after[k]) for k in touched(event)) + "}"
    return '{"cmp":' + cmp_ + ',"calc":' + calc + ',"after":' + dumps_ordered(after, state_order) + "}"


def target_fill(after, state_order, n_units):
    return '{"fill":' + json.dumps(" ." * n_units) + ',"after":' + dumps_ordered(after, state_order) + "}"


# ----------------------------------------------------------------------------- strict parsing

_INT = r"-?(?:0|[1-9][0-9]*)"
_KEY = r'"[^"\\]+"'


def _parse_flat_int_obj(s):
    """Parse a compact flat JSON object of string->int exactly; return ordered dict or None."""
    if not (s.startswith("{") and s.endswith("}")):
        return None
    body = s[1:-1]
    if body == "":
        return {}
    out = {}
    for part in body.split(","):
        m = re.fullmatch(f"({_KEY}):({_INT})", part)
        if not m:
            return None
        k = json.loads(m.group(1))
        if k in out:
            return None
        out[k] = int(m.group(2))
    return out


def parse_output(text, fmt, state_order):
    """Strict parser. Returns dict(after=..., [ok, listed, status, calc]) or None if malformed.

    The 'after' keys must equal state_order exactly, in that order (SPEC 4)."""
    s = text.strip()
    if fmt == "ans":
        m = re.fullmatch(r'\{"after":(\{[^{}]*\})\}', s)
        if not m:
            return None
        after = _parse_flat_int_obj(m.group(1))
        res = {"after": after}
    elif fmt == "fill":
        m = re.fullmatch(r'\{"fill":"((?: \.)*)","after":(\{[^{}]*\})\}', s)
        if not m:
            return None
        after = _parse_flat_int_obj(m.group(2))
        res = {"after": after}
    elif fmt == "trace":
        m = re.fullmatch(
            r'\{"cmp":\{"status":("(?:[^"\\]|\\.)*"),"listed":(true|false|null),"ok":(true|false)\},'
            r'"calc":(\{[^{}]*\}),"after":(\{[^{}]*\})\}', s)
        if not m:
            return None
        after = _parse_flat_int_obj(m.group(5))
        calc = _parse_flat_int_obj(m.group(4))
        if calc is None:
            return None
        res = {"after": after, "status": json.loads(m.group(1)), "listed": json.loads(m.group(2)),
               "ok": json.loads(m.group(3)), "calc": calc}
    else:
        raise ValueError(fmt)
    if res["after"] is None or list(res["after"].keys()) != list(state_order):
        return None
    return res


# ----------------------------------------------------------------------------- rule-text parser

_WORD = r"[A-Za-z0-9_\-]+"
_LIST_RE = rf"(?P<L>{_WORD}(?:, {_WORD})*(?: or {_WORD})?)"


def _split_list(s):
    if " or " in s:
        head, last = s.rsplit(" or ", 1)
        items = head.split(", ") + [last]
    else:
        items = [s]
    if not 1 <= len(items) <= 3 or len(set(items)) != len(items):
        raise ValueError(f"bad status list: {s!r}")
    if len(items) == 3 and ", " not in s:
        raise ValueError(s)
    return items


def _tpl_regex(tpl, slots):
    pat = re.escape(tpl)
    for name, rx in slots.items():
        pat = pat.replace(re.escape("{" + name + "}"), rx)
    return re.compile("^" + pat + "$")


_CLAUSE_RE = {
    "S+": re.compile(r"^its status is in \[(?P<B>" + _WORD + r"(?:, " + _WORD + r")*)\]$"),
    "S-": re.compile(r"^its status is not in \[(?P<B>" + _WORD + r"(?:, " + _WORD + r")*)\]$"),
    "K+": re.compile(r"^its kind is (?P<K>(?:add|transfer)(?: or (?:add|transfer))?)$"),
    "AGE": re.compile(r"^its amount is at least (?P<T>-?[0-9]+)$"),
    "ALT": re.compile(r"^its amount is below (?P<T>-?[0-9]+)$"),
    "BGE": re.compile(r"^the account balance is at least (?P<T>-?[0-9]+)$"),
}


def _parse_clause(c):
    hits = []
    for op, rx in _CLAUSE_RE.items():
        m = rx.match(c)
        if m:
            if op in STATUS_OPS:
                hits.append({"op": op, "set": m.group("B").split(", ")})
            elif op == "K+":
                hits.append({"op": op, "kinds": m.group("K").split(" or ")})
            else:
                hits.append({"op": op, "t": int(m.group("T"))})
    if len(hits) != 1:
        raise ValueError(f"clause not uniquely parseable: {c!r}")
    return hits[0]


_SINGLE = [(op, tid, _tpl_regex(tpl, {"L": _LIST_RE}))
           for op, pool in (("S+", T.SPOS), ("S-", T.SNEG)) for tid, tpl in pool.items()]
_CL = r"(?P<C1>.+?)"
_CL2 = r"(?P<C2>.+?)"
_FRAMES = [(op, fid, _tpl_regex(tpl, {"C1": _CL, "C2": _CL2}))
           for op, pool in (("AND", T.AND_FRAMES), ("OR", T.OR_FRAMES)) for fid, tpl in pool.items()]
_SOLO = _tpl_regex("An event takes effect only if {C}; otherwise nothing changes.", {"C": r"(?P<C>.+?)"})


def parse_rule(sentence):
    """Invert the closed template pools. Returns (ast, template_id). Fails closed."""
    found = []
    for op, tid, rx in _SINGLE:
        m = rx.match(sentence)
        if m:
            try:
                found.append(({"op": op, "set": _split_list(m.group("L"))}, f"{op}:{tid}"))
            except ValueError:
                pass
    for op, fid, rx in _FRAMES:
        m = rx.match(sentence)
        if m:
            try:
                found.append(({"op": op, "args": [_parse_clause(m.group("C1")), _parse_clause(m.group("C2"))]},
                              f"{op}:{fid}"))
            except ValueError:
                pass
    m = _SOLO.match(sentence)
    if m:
        try:
            found.append((_parse_clause(m.group("C")), "SOLO"))
        except ValueError:
            pass
    asts = {json.dumps(a, sort_keys=True) for a, _ in found}
    if len(asts) != 1:
        raise ValueError(f"rule not uniquely parseable ({len(asts)} readings): {sentence!r}")
    return found[0]


def parse_input(text):
    """Parse a rendered input into (ast, template_id, state, event) with key orders preserved."""
    lines = text.split("\n")
    if len(lines) != 3 or not lines[0].startswith("RULE: ") or not lines[1].startswith("STATE: ") \
            or not lines[2].startswith("EVENT: "):
        raise ValueError("bad input layout")
    ast, tid = parse_rule(lines[0][len("RULE: "):])
    state = json.loads(lines[1][len("STATE: "):])
    event = json.loads(lines[2][len("EVENT: "):])
    if sorted(event.keys()) != sorted(EVENT_KEYS):
        raise ValueError("bad event keys")
    return ast, tid, state, event


# ----------------------------------------------------------------------------- hashing

def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def semantic_hash(ast, state, event):
    return hashlib.sha256(_canon({"rule": ast, "state": state, "event": event}).encode()).hexdigest()


def skeleton_hash(state, event):
    """(state, event-without-status) skeleton, for cross-split structural overlap audits."""
    ev = {k: v for k, v in event.items() if k != "status"}
    return hashlib.sha256(_canon({"state": state, "event": ev}).encode()).hexdigest()


# ----------------------------------------------------------------------------- canonicalizer (tool tier B)

CANON_NAMES = ("A", "B", "C", "D", "E", "F", "G", "H")


def canonicalize(text):
    """Map any rendering to the canonical surface: t0 wording, canonical event key order, accounts
    renamed to A.. in their rendered order. Returns (canon_text, canon_state_order, back) where
    back(canonical_after) -> after dict in the original names and original key order (None if the
    canonical prediction is malformed). Interface tool (information tier B): uses the exact parser."""
    ast, _tid, state, event = parse_input(text)
    orig = list(state.keys())
    fwd = {k: CANON_NAMES[i] for i, k in enumerate(orig)}
    inv = {v: k for k, v in fwd.items()}
    cstate = {fwd[k]: state[k] for k in orig}
    cevent = dict(event)
    cevent["account"] = fwd[event["account"]]
    cevent["other"] = fwd[event["other"]]
    ctext = render_input(ast, cstate, cevent, "t0", list(cstate.keys()), EVENT_KEYS)

    def back(cafter):
        if cafter is None or sorted(cafter) != sorted(cstate):
            return None
        return {inv[k]: cafter[k] for k in cstate}

    return ctext, list(cstate.keys()), back
