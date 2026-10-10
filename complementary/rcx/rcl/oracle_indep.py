"""Independent exact oracle for the RCL world, written from SPEC_RCL.md alone.

It parses rendered inputs (inverting the closed template pools, whose strings are
copied verbatim from templates.py), evaluates the rule predicate, applies the ledger
transition and builds/parses the `ans` and `trace` targets. Parsing is exhaustive:
every template/split that could produce the sentence is tried, and the sentence is
rejected unless exactly one reading exists (fail closed). The `fill` format is not
implemented because its width is a TRAIN-derived constant.

Event kinds: the arith world (`add`, `transfer`, plus the native-kit `scale`) and the
copy-world variant of SPEC section 1.1 (`set`, `swap`). `K+` clauses may name any
non-empty set of distinct kinds from {add, transfer, set, swap}; `scale` is never
nameable in a rule.
"""
import json
import re

__all__ = [
    "SPOS", "SNEG", "AND_FRAMES", "OR_FRAMES", "ATOM_FRAME", "CLAUSE", "TEMPLATE_IDS",
    "KINDS", "EVENT_KINDS", "TWO_ACCOUNT_KINDS",
    "parse_input", "parse_rule", "validate_ast", "evaluate", "transition", "listed_of",
    "gold_targets", "parse_target", "render_list", "render_clause", "render_rule", "render_input",
]

SPOS = {
    "t0": "Only events with status {L} take effect; other events change nothing.",
    "t1": "An event is applied only if its status is {L}.",
    "t2": "Apply the event when its status is {L}; otherwise leave the balances unchanged.",
    "t3": "Events whose status is {L} update the balances. All other events are ignored.",
    "h0": "If the status is {L}, carry out the event; for any other status, do nothing.",
    "h1": "Balances change only for {L} events.",
}
SNEG = {
    "t0": "Events with status {L} are ignored; all other events take effect.",
    "t1": "An event is applied unless its status is {L}.",
    "t2": "Skip the event when its status is {L}; otherwise apply it.",
    "t3": "Events whose status is {L} change nothing. All other events update the balances.",
    "h0": "If the status is {L}, do nothing; for any other status, carry out the event.",
    "h1": "Every event takes effect except {L} events.",
}
AND_FRAMES = {
    "t0": "An event takes effect only if {C1} and {C2}; otherwise nothing changes.",
    "t1": "Apply the event when {C1} and {C2}. Otherwise leave the balances unchanged.",
    "h0": "Both conditions must hold for an event to take effect: {C1}; {C2}.",
}
OR_FRAMES = {
    "t0": "An event takes effect if {C1} or if {C2}; otherwise nothing changes.",
    "t1": "Apply the event when {C1} or when {C2}. Otherwise leave the balances unchanged.",
    "h0": "An event takes effect when either {C1} or {C2}.",
}
ATOM_FRAME = "An event takes effect only if {C}; otherwise nothing changes."
CLAUSE = {
    "S+": "its status is in [{B}]",
    "S-": "its status is not in [{B}]",
    "K+": "its kind is {K}",
    "AGE": "its amount is at least {T}",
    "ALT": "its amount is below {T}",
    "BGE": "the account balance is at least {T}",
}

_SINGLE = {"SPOS": ("S+", SPOS), "SNEG": ("S-", SNEG)}
_FRAMES = {"AND": AND_FRAMES, "OR": OR_FRAMES}
TEMPLATE_IDS = tuple(
    [f"{fam}:{tid}" for fam, (_, pool) in _SINGLE.items() for tid in pool]
    + [f"{op}:{fid}" for op, pool in _FRAMES.items() for fid in pool]
    + ["ATOM:t0"]
)

STATUS_OPS = ("S+", "S-")
THRESH_OPS = ("AGE", "ALT", "BGE")
# Kinds a K+ atom may name (SPEC section 2, extended by section 1.1); scale is not nameable.
KINDS = ("add", "transfer", "set", "swap")
# Kinds an event may carry (SPEC section 1 and 1.1).
EVENT_KINDS = ("add", "transfer", "scale", "set", "swap")
# Kinds that touch `other` as well as `account` (calc keys: account, then other).
TWO_ACCOUNT_KINDS = ("transfer", "swap")
EVENT_KEYS = frozenset({"account", "amount", "kind", "other", "status"})

_WORD = r"[A-Za-z0-9_-]+"
_WORD_RE = re.compile(_WORD, re.ASCII)
_LIST_RES = (
    re.compile(rf"({_WORD})", re.ASCII),
    re.compile(rf"({_WORD}) or ({_WORD})", re.ASCII),
    re.compile(rf"({_WORD}), ({_WORD}) or ({_WORD})", re.ASCII),
)
_INT = r"-?(?:0|[1-9][0-9]*)"
_JSON_WS = " \t\n\r"
_CLAUSE_RES = (
    ("S+", re.compile(r"its status is in \[(.*)\]", re.ASCII | re.DOTALL)),
    ("S-", re.compile(r"its status is not in \[(.*)\]", re.ASCII | re.DOTALL)),
    ("K+", re.compile(r"its kind is (.*)", re.ASCII | re.DOTALL)),
    ("AGE", re.compile(rf"its amount is at least ({_INT})", re.ASCII)),
    ("ALT", re.compile(rf"its amount is below ({_INT})", re.ASCII)),
    ("BGE", re.compile(rf"the account balance is at least ({_INT})", re.ASCII)),
)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _distinct(xs):
    return len(set(xs)) == len(xs)


def _parse_list(text):
    """'x' | 'x or y' | 'x, y or z' -> list of distinct words, else None."""
    for rx in _LIST_RES:
        m = rx.fullmatch(text)
        if m:
            words = list(m.groups())
            return words if _distinct(words) else None
    return None


def _parse_bracket(text):
    words = text.split(", ")
    if not 1 <= len(words) <= 3 or not _distinct(words):
        return None
    return words if all(_WORD_RE.fullmatch(w) for w in words) else None


def _parse_int(text):
    return None if text == "-0" else int(text)


def _parse_clause(text):
    """Invert one clause fragment into an atom AST, else None."""
    for op, rx in _CLAUSE_RES:
        m = rx.fullmatch(text)
        if not m:
            continue
        arg = m.group(1)
        if op in STATUS_OPS:
            words = _parse_bracket(arg)
            return None if words is None else {"op": op, "set": words}
        if op == "K+":
            kinds = arg.split(" or ")
            ok = 1 <= len(kinds) <= len(KINDS) and _distinct(kinds) and all(k in KINDS for k in kinds)
            return {"op": op, "kinds": kinds} if ok else None
        t = _parse_int(arg)
        return None if t is None else {"op": op, "t": t}
    return None


def _strip_frame(text, prefix, suffix):
    if len(text) < len(prefix) + len(suffix) or not text.startswith(prefix) or not text.endswith(suffix):
        return None
    return text[len(prefix):len(text) - len(suffix)]


def _readings(sentence):
    """All (template_id, ast) readings of a rule sentence across every pool."""
    out = []
    for fam, (op, pool) in _SINGLE.items():
        for tid, tpl in pool.items():
            mid = _strip_frame(sentence, *tpl.split("{L}"))
            words = None if mid is None else _parse_list(mid)
            if words is not None:
                out.append((f"{fam}:{tid}", {"op": op, "set": words}))
    mid = _strip_frame(sentence, *ATOM_FRAME.split("{C}"))
    atom = None if mid is None else _parse_clause(mid)
    if atom is not None and atom["op"] not in STATUS_OPS:
        out.append(("ATOM:t0", atom))
    for op, pool in _FRAMES.items():
        for fid, tpl in pool.items():
            prefix, rest = tpl.split("{C1}")
            sep, suffix = rest.split("{C2}")
            mid = _strip_frame(sentence, prefix, suffix)
            if mid is None:
                continue
            i = mid.find(sep)
            while i >= 0:
                a, b = _parse_clause(mid[:i]), _parse_clause(mid[i + len(sep):])
                if a is not None and b is not None:
                    out.append((f"{op}:{fid}", {"op": op, "args": [a, b]}))
                i = mid.find(sep, i + 1)
    return out


def parse_rule(sentence):
    """Invert a rule sentence. Returns (ast, template_id); ValueError if no or several readings."""
    if not isinstance(sentence, str):
        raise ValueError("rule sentence must be a str")
    found = _readings(sentence)
    if not found:
        raise ValueError(f"rule sentence matches no template: {sentence!r}")
    if len(found) > 1:
        raise ValueError(f"ambiguous rule sentence ({[t for t, _ in found]}): {sentence!r}")
    tid, ast = found[0]
    validate_ast(ast)
    return ast, tid


def _reject_number(x):
    raise ValueError(f"non-integer number {x!r}")


def _pairs_hook(pairs):
    keys = [k for k, _ in pairs]
    if not _distinct(keys):
        raise ValueError("duplicate JSON key")
    return dict(pairs)


_DECODER = json.JSONDecoder(object_pairs_hook=_pairs_hook, parse_float=_reject_number,
                            parse_constant=_reject_number)


def _compact(obj):
    return json.dumps(obj, separators=(",", ":"))


def _load_compact_object(text, what):
    try:
        obj, end = _DECODER.raw_decode(text)
    except RecursionError:
        raise ValueError(f"{what} JSON is nested too deeply") from None
    if end != len(text) or not isinstance(obj, dict) or _compact(obj) != text:
        raise ValueError(f"{what} is not a compact JSON object: {text!r}")
    return obj


def _check_state(state):
    if not isinstance(state, dict) or len(state) < 2:
        raise ValueError("state must be a mapping with at least 2 accounts")
    for k, v in state.items():
        if not isinstance(k, str) or not k or not _is_int(v):
            raise ValueError(f"bad state entry {k!r}: {v!r}")


def _check_event(event, state):
    if not isinstance(event, dict) or set(event) != EVENT_KEYS or len(event) != 5:
        raise ValueError(f"event must have exactly the keys {sorted(EVENT_KEYS)}")
    acc, oth = event["account"], event["other"]
    if not isinstance(acc, str) or acc not in state or not isinstance(oth, str) or oth not in state:
        raise ValueError("event account/other must be state keys")
    if acc == oth:
        raise ValueError("event other must differ from account")
    if not _is_int(event["amount"]):
        raise ValueError("event amount must be an integer")
    if event["kind"] not in EVENT_KINDS:
        raise ValueError(f"bad event kind {event['kind']!r}")
    if not isinstance(event["status"], str) or not _WORD_RE.fullmatch(event["status"]):
        raise ValueError(f"bad event status {event['status']!r}")


def parse_input(text):
    """Parse 'RULE: ...\\nSTATE: ...\\nEVENT: ...' exactly.

    Returns {rule_ast, state, event, state_order, template_id}; state/event keep the
    rendered key order. Raises ValueError on any deviation from SPEC_RCL section 3.
    """
    if not isinstance(text, str):
        raise ValueError("input must be a str")
    lines = text.split("\n")
    heads = ("RULE: ", "STATE: ", "EVENT: ")
    if len(lines) != 3 or not all(ln.startswith(h) for ln, h in zip(lines, heads)):
        raise ValueError("input must be exactly the lines RULE:, STATE:, EVENT:")
    rule, state_s, event_s = (ln[len(h):] for ln, h in zip(lines, heads))
    ast, tid = parse_rule(rule)
    state = _load_compact_object(state_s, "STATE")
    _check_state(state)
    event = _load_compact_object(event_s, "EVENT")
    _check_event(event, state)
    return {"rule_ast": ast, "state": state, "event": event, "state_order": list(state), "template_id": tid}


def _check_words(ws, maxlen, allowed=None):
    if not isinstance(ws, (list, tuple)) or not 1 <= len(ws) <= maxlen:
        return False
    if not all(isinstance(w, str) for w in ws) or not _distinct(list(ws)):
        return False
    if allowed is not None:
        return all(w in allowed for w in ws)
    return all(_WORD_RE.fullmatch(w) for w in ws)


def _op_of(node):
    op = node.get("op") if isinstance(node, dict) else None
    return op if isinstance(op, str) else None


def _validate_atom(node):
    op = _op_of(node)
    if op in STATUS_OPS:
        ok = set(node) == {"op", "set"} and _check_words(node["set"], 3)
    elif op == "K+":
        ok = set(node) == {"op", "kinds"} and _check_words(node["kinds"], len(KINDS), KINDS)
    elif op in THRESH_OPS:
        ok = set(node) == {"op", "t"} and _is_int(node["t"])
    else:
        ok = False
    if not ok:
        raise ValueError(f"malformed atom: {node!r}")


def validate_ast(ast):
    """Raise ValueError unless ast is an atom or AND/OR of exactly two atoms (SPEC section 2)."""
    if _op_of(ast) in _FRAMES:
        args = ast.get("args")
        if set(ast) != {"op", "args"} or not isinstance(args, (list, tuple)) or len(args) != 2:
            raise ValueError(f"malformed composition: {ast!r}")
        for a in args:
            _validate_atom(a)
    else:
        _validate_atom(ast)


def _eval_atom(node, state, event):
    op = node["op"]
    if op == "S+":
        return event["status"] in node["set"]
    if op == "S-":
        return event["status"] not in node["set"]
    if op == "K+":
        return event["kind"] in node["kinds"]
    if op == "AGE":
        return event["amount"] >= node["t"]
    if op == "ALT":
        return event["amount"] < node["t"]
    return state[event["account"]] >= node["t"]


def evaluate(rule_ast, state, event):
    """Applicability predicate of SPEC section 2 (exact, case-sensitive status equality)."""
    validate_ast(rule_ast)
    _check_state(state)
    _check_event(event, state)
    if rule_ast["op"] in _FRAMES:
        a, b = (_eval_atom(x, state, event) for x in rule_ast["args"])
        return (a and b) if rule_ast["op"] == "AND" else (a or b)
    return _eval_atom(rule_ast, state, event)


def transition(state, event, ok):
    """Ledger dynamics of SPEC sections 1 and 1.1; returns a new dict in the input key order."""
    _check_state(state)
    _check_event(event, state)
    if not isinstance(ok, bool):
        raise ValueError("ok must be a bool")
    after = dict(state)
    if not ok:
        return after
    acc, oth, amt, kind = event["account"], event["other"], event["amount"], event["kind"]
    if kind == "add":
        after[acc] = state[acc] + amt
    elif kind == "transfer":
        after[acc] = state[acc] - amt
        after[oth] = state[oth] + amt
    elif kind == "scale":
        after[acc] = state[acc] * amt
    elif kind == "set":
        after[acc] = amt
    elif kind == "swap":
        after[acc] = state[oth]
        after[oth] = state[acc]
    else:  # unreachable: _check_event already restricts kinds
        raise ValueError(f"bad event kind {kind!r}")
    return after


def listed_of(rule_ast, event):
    """Membership of event.status in the first status atom's set (left to right), None if none."""
    validate_ast(rule_ast)
    nodes = rule_ast["args"] if rule_ast["op"] in _FRAMES else [rule_ast]
    for node in nodes:
        if node["op"] in STATUS_OPS:
            return event["status"] in node["set"]
    return None


def _calc_keys(event):
    """Touched accounts: [account], plus other for transfer and swap (account first)."""
    return [event["account"]] + ([event["other"]] if event["kind"] in TWO_ACCOUNT_KINDS else [])


def gold_targets(text):
    """Gold {ok, listed, after, ans, trace} for an input text (SPEC section 4)."""
    p = parse_input(text)
    ast, state, event = p["rule_ast"], p["state"], p["event"]
    ok = evaluate(ast, state, event)
    listed = listed_of(ast, event)
    after = transition(state, event, ok)
    calc = {k: after[k] for k in _calc_keys(event)}
    ans = _compact({"after": after})
    trace = _compact({"cmp": {"status": event["status"], "listed": listed, "ok": ok},
                      "calc": calc, "after": after})
    return {"ok": ok, "listed": listed, "after": after, "ans": ans, "trace": trace}


def _balances(d, nonempty=True):
    if not isinstance(d, dict) or (nonempty and not d):
        return False
    return all(isinstance(k, str) and _is_int(v) for k, v in d.items())


def parse_target(text, fmt, source=None):
    """Strictly parse an 'ans' or 'trace' output; None if malformed.

    Leading/trailing JSON whitespace (space, tab, LF, CR) is allowed; any other surrounding
    text (including other Unicode/control "whitespace"), a different key
    order, extra/missing keys, duplicate keys or non-integer balances make it malformed.
    If `source` (input text or parse_input result) is given, `after` keys must equal the
    STATE key order and `calc` keys must be [account] (+ [other] for transfer and swap).
    """
    if fmt not in ("ans", "trace"):
        raise ValueError(f"unsupported format {fmt!r}")
    if not isinstance(text, str):
        return None
    s = text.lstrip(_JSON_WS)
    try:
        obj, end = _DECODER.raw_decode(s)
    except (ValueError, RecursionError):
        return None
    if s[end:].strip(_JSON_WS) or not isinstance(obj, dict):
        return None
    if fmt == "ans":
        if list(obj) != ["after"] or not _balances(obj["after"]):
            return None
    else:
        if list(obj) != ["cmp", "calc", "after"]:
            return None
        cmp_ = obj["cmp"]
        if not isinstance(cmp_, dict) or list(cmp_) != ["status", "listed", "ok"]:
            return None
        if not isinstance(cmp_["status"], str) or not isinstance(cmp_["ok"], bool):
            return None
        if cmp_["listed"] is not None and not isinstance(cmp_["listed"], bool):
            return None
        if not _balances(obj["calc"]) or len(obj["calc"]) > 2 or not _balances(obj["after"]):
            return None
    if source is not None:
        src = parse_input(source) if isinstance(source, str) else source
        if list(obj["after"]) != list(src["state_order"]):
            return None
        if fmt == "trace" and list(obj["calc"]) != _calc_keys(src["event"]):
            return None
    return obj


def render_list(words):
    """1 -> 'x'; 2 -> 'x or y'; 3 -> 'x, y or z'."""
    words = list(words)
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} or {words[1]}"
    if len(words) == 3:
        return f"{words[0]}, {words[1]} or {words[2]}"
    raise ValueError("status list must have 1..3 items")


def render_clause(atom):
    """Clause fragment for an atom inside a composition (or the single-atom frame)."""
    _validate_atom(atom)
    op = atom["op"]
    if op in STATUS_OPS:
        return CLAUSE[op].replace("{B}", ", ".join(atom["set"]))
    if op == "K+":
        return CLAUSE[op].replace("{K}", " or ".join(atom["kinds"]))
    return CLAUSE[op].replace("{T}", str(atom["t"]))


def render_rule(ast, template_id):
    """Render a rule AST with a full template id such as 'SPOS:t2', 'AND:h0' or 'ATOM:t0'."""
    validate_ast(ast)
    fam, _, tid = template_id.partition(":")
    op = ast["op"]
    if fam in _SINGLE and _SINGLE[fam][0] == op and tid in _SINGLE[fam][1]:
        return _SINGLE[fam][1][tid].replace("{L}", render_list(ast["set"]))
    if fam in _FRAMES and fam == op and tid in _FRAMES[fam]:
        c1, c2 = (render_clause(a) for a in ast["args"])
        return _FRAMES[fam][tid].replace("{C1}", c1).replace("{C2}", c2)
    if fam == "ATOM" and tid == "t0" and op in ("K+",) + THRESH_OPS:
        return ATOM_FRAME.replace("{C}", render_clause(ast))
    raise ValueError(f"template {template_id!r} cannot render op {op!r}")


def render_input(rule_ast, state, event, template_id):
    """Three-line input text with compact JSON in the given key orders."""
    return (f"RULE: {render_rule(rule_ast, template_id)}\nSTATE: {_compact(state)}\n"
            f"EVENT: {_compact(event)}")
