"""Closed template pools for rendering RCL rules (SPEC_RCL.md section 3).

Train templates have ids t0..; held-out templates have ids h0... t0 is canonical.
Every template has exactly one {L} slot (status list) for single status atoms, or
{C1}/{C2} clause slots for compositions. Status words never contain spaces, so
list rendering ("x", "x or y", "x, y or z") is unambiguous.
"""

# Single status atom, positive polarity: applies iff status in set.
SPOS = {
    "t0": "Only events with status {L} take effect; other events change nothing.",
    "t1": "An event is applied only if its status is {L}.",
    "t2": "Apply the event when its status is {L}; otherwise leave the balances unchanged.",
    "t3": "Events whose status is {L} update the balances. All other events are ignored.",
    "h0": "If the status is {L}, carry out the event; for any other status, do nothing.",
    "h1": "Balances change only for {L} events.",
}

# Single status atom, negative polarity: applies iff status NOT in set.
SNEG = {
    "t0": "Events with status {L} are ignored; all other events take effect.",
    "t1": "An event is applied unless its status is {L}.",
    "t2": "Skip the event when its status is {L}; otherwise apply it.",
    "t3": "Events whose status is {L} change nothing. All other events update the balances.",
    "h0": "If the status is {L}, do nothing; for any other status, carry out the event.",
    "h1": "Every event takes effect except {L} events.",
}

# Composition frames. {C1} is always the status clause, {C2} the other atom's clause.
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

# Clause fragments used inside composition frames (fixed wording, not a surface factor).
# Status clauses use a bracketed list so that negation is unambiguous.
CLAUSE = {
    "S+": "its status is in [{B}]",
    "S-": "its status is not in [{B}]",
    "K+": "its kind is {K}",
    "AGE": "its amount is at least {T}",
    "ALT": "its amount is below {T}",
    "BGE": "the account balance is at least {T}",
}

TRAIN_IDS = ("t0", "t1", "t2", "t3")
HELDOUT_IDS = ("h0", "h1")
FRAME_TRAIN_IDS = ("t0", "t1")
FRAME_HELDOUT_IDS = ("h0",)


def render_list(words):
    """1 -> 'x'; 2 -> 'x or y'; 3 -> 'x, y or z'."""
    words = list(words)
    if not 1 <= len(words) <= 3:
        raise ValueError("status list must have 1..3 items")
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} or {words[1]}"
    return f"{words[0]}, {words[1]} or {words[2]}"


def render_bracket(words):
    return ", ".join(words)


def clause(atom):
    op = atom["op"]
    if op in ("S+", "S-"):
        return CLAUSE[op].format(B=render_bracket(atom["set"]))
    if op == "K+":
        return CLAUSE[op].format(K=" or ".join(atom["kinds"]))
    if op in ("AGE", "ALT", "BGE"):
        return CLAUSE[op].format(T=atom["t"])
    raise ValueError(op)


def render_rule(ast, tid):
    """Render a rule AST with template id tid (t*/h*). For compositions the
    template id selects the frame; frames only have t0,t1,h0 so h1 maps to h0."""
    op = ast["op"]
    if op == "S+":
        return SPOS[tid].format(L=render_list(ast["set"]))
    if op == "S-":
        return SNEG[tid].format(L=render_list(ast["set"]))
    if op in ("AND", "OR"):
        frames = AND_FRAMES if op == "AND" else OR_FRAMES
        fid = tid if tid in frames else ("h0" if tid.startswith("h") else "t0")
        a, b = ast["args"]
        return frames[fid].format(C1=clause(a), C2=clause(b))
    if op in ("K+", "AGE", "ALT", "BGE"):
        frame = "An event takes effect only if {C}; otherwise nothing changes."
        return frame.format(C=clause(ast))
    raise ValueError(op)
