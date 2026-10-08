"""RCWM-T1 environment: multi-step action chains over Titan's three transition families.

Semantics are the FDS oracles (validated 384/384 against the historical probe) extended to chains:
a chain is a start state plus k events applied in order. For prerequisite_plan an inapplicable
action is a no-op (the FDS oracle convention); the verbatim one-step Titan prompt presumes
applicability, so the multi-step prompt states the no-op rule explicitly.

Event parameters are drawn from the recorded SELECT-VIEWS/probe input marginals
(CANDIDATE_VIEWS.jsonl, sha 45674251...), the same repair generate_v2 used, except that
prerequisite preconditions are drawn over all bits so applicability varies along a chain.
"""
from __future__ import annotations

import hashlib
import json
import re
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

FDS = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1")
RECORDED = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/process-transition-probe-001/CANDIDATE_VIEWS.jsonl")
RECORDED_SHA = "45674251e38e574229f4ea98d04c30fe623ec6ac58ad308ad491661c84afab7e"
MOD = 97
FAMILIES = ("ledger_replay", "prerequisite_plan", "register_machine")
SHORT = {"ledger_replay": "L", "prerequisite_plan": "P", "register_machine": "R"}
ACCOUNTS = ("A", "B", "C", "D")
REGISTERS = ("X", "Y", "Z")
NBITS = 8


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


# Verbatim one-step system prompts (generate.SYSTEM; the prompts Titan was trained and scored on).
SYSTEM_ONE = {
    "ledger_replay": (
        "Apply one event to the supplied account balances. If status is not posted, leave all "
        "balances unchanged. add: add amount to account. scale: multiply account by amount. "
        "transfer: subtract amount from account and add it to other, simultaneously. Return only "
        "the complete resulting balances as a JSON object."
    ),
    "prerequisite_plan": (
        "The state is an 8-bit flag mask. A supplied applicable action requires all pre_on bits "
        "set and all pre_off bits clear. Apply delete bits first, then add bits. Return only the "
        "resulting state as a JSON integer. Names and costs do not change the state."
    ),
    "register_machine": (
        "Apply one instruction to the supplied registers. Registers are integers modulo 97. swap "
        "exchanges register and other simultaneously. add adds value to register. mix adds (2 "
        "times other plus value) to register. conditional adds value if other is greater than "
        "threshold, otherwise subtracts value. Reduce updated numeric registers modulo 97; "
        "preserve other registers. Return only the complete resulting registers as a JSON object."
    ),
}

# Multi-step rules: the one-step rule with "in order" semantics. The ledger text is the DMRF
# RULES_MULTI precedent verbatim; the other two follow the same minimal-edit pattern.
SYSTEM_MULTI = {
    "ledger_replay": (
        "Apply each event to the supplied account balances, in order. If an event's status is not "
        "posted, leave all balances unchanged for that event. add: add amount to account. scale: "
        "multiply account by amount. transfer: subtract amount from account and add it to other, "
        "simultaneously. Return only the complete resulting balances as a JSON object."
    ),
    "prerequisite_plan": (
        "The state is an 8-bit flag mask. Apply each action to the state, in order. An action is "
        "applicable only if all its pre_on bits are set and all its pre_off bits are clear in the "
        "current state; an inapplicable action leaves the state unchanged. For an applicable "
        "action, apply delete bits first, then add bits. Return only the final resulting state as "
        "a JSON integer. Names and costs do not change the state."
    ),
    "register_machine": (
        "Apply each instruction to the supplied registers, in order. Registers are integers "
        "modulo 97. swap exchanges register and other simultaneously. add adds value to register. "
        "mix adds (2 times other plus value) to register. conditional adds value if other is "
        "greater than threshold, otherwise subtracts value. Reduce updated numeric registers "
        "modulo 97 after each instruction; preserve other registers. Return only the complete "
        "final registers as a JSON object."
    ),
}

# ---------------------------------------------------------------- dynamics (one step)


def applicable(state: int, ev: dict) -> bool:
    return (state & ev["pre_on"]) == ev["pre_on"] and (state & ev["pre_off"]) == 0


def step(fam: str, state, ev: dict):
    """Return (next_state, changed_semantics_flag). Mirrors FDS oracles exactly."""
    if fam == "ledger_replay":
        bal = dict(state)
        if ev.get("status") == "posted":
            k, a, amt = ev["kind"], ev["account"], ev["amount"]
            if k == "add":
                bal[a] = bal[a] + amt
            elif k == "scale":
                bal[a] = bal[a] * amt
            elif k == "transfer":
                bal[a] = bal[a] - amt
                bal[ev["other"]] = bal[ev["other"]] + amt
            else:
                raise ValueError(k)
        return bal
    if fam == "prerequisite_plan":
        if applicable(state, ev):
            return (state & ~ev["delete"]) | ev["add"]
        return state
    if fam == "register_machine":
        reg = dict(state)
        k, r, o, v = ev["kind"], ev["register"], ev.get("other"), ev.get("value", 0)
        if k == "swap":
            reg[r], reg[o] = reg[o], reg[r]
        elif k == "add":
            reg[r] = (reg[r] + v) % MOD
        elif k == "mix":
            reg[r] = (reg[r] + 2 * reg[o] + v) % MOD
        elif k == "conditional":
            reg[r] = (reg[r] + (v if reg[o] > ev["threshold"] else -v)) % MOD
        else:
            raise ValueError(k)
        return reg
    raise ValueError(fam)


def rollout(fam: str, s0, events: list[dict]) -> list:
    """States s0..sk."""
    states = [s0]
    for ev in events:
        states.append(step(fam, states[-1], ev))
    return states


_FDS_ORACLES = None


def fds_oracles():
    global _FDS_ORACLES
    if _FDS_ORACLES is None:
        sys.path.insert(0, str(FDS))
        from oracles import ORACLES  # type: ignore
        _FDS_ORACLES = ORACLES
    return _FDS_ORACLES


def cross_check(fam: str, s, ev) -> None:
    """Independent check: our step must equal the validated FDS oracle on every transition."""
    ref = fds_oracles()[fam]({"before": s, "event": ev})
    got = dumps(step(fam, s, ev))
    if ref != got:
        raise AssertionError(f"oracle mismatch {fam}: {ref} != {got} for {s} {ev}")


# ---------------------------------------------------------------- consequence features


def bits(x: int) -> list[int]:
    return [(x >> b) & 1 for b in range(NBITS)]


def features(fam: str, states: list, events: list[dict]) -> dict:
    """Grounded consequence features of a chain (final-state + cumulative)."""
    s0, sk = states[0], states[-1]
    if fam == "ledger_replay":
        return {
            "final": [sk[a] for a in ACCOUNTS],
            "changed": [int(sk[a] != s0[a]) for a in ACCOUNTS],
            "sign_neg": [int(sk[a] < 0) for a in ACCOUNTS],
            "n_posted": sum(e["status"] == "posted" for e in events),
        }
    if fam == "prerequisite_plan":
        app = [int(applicable(states[i], e)) for i, e in enumerate(events)]
        return {
            "final": bits(sk),
            "changed": [int(a != b) for a, b in zip(bits(sk), bits(s0))],
            "applied": app,
            "n_applied": sum(app),
            "cost_applied": sum(e["cost"] for e, a in zip(events, app) if a),
        }
    if fam == "register_machine":
        return {
            "final": [sk[r] for r in REGISTERS],
            "changed": [int(sk[r] != s0[r]) for r in REGISTERS],
            "hi": [int(sk[r] > 48) for r in REGISTERS],
            "parity": [sk[r] % 2 for r in REGISTERS],
        }
    raise ValueError(fam)


# ---------------------------------------------------------------- generators


@dataclass
class Marginals:
    ledger_kinds: list = field(default_factory=list)
    ledger_status: list = field(default_factory=list)
    ledger_amounts: list = field(default_factory=list)
    ledger_balances: list = field(default_factory=list)
    p_before_bits: list = field(default_factory=list)
    p_delete_bits: list = field(default_factory=list)
    p_pre_on_bits: list = field(default_factory=list)
    p_pre_off_bits: list = field(default_factory=list)
    p_costs: list = field(default_factory=list)
    r_kinds: list = field(default_factory=list)
    r_values: list = field(default_factory=list)
    r_thresholds: list = field(default_factory=list)
    r_befores: list = field(default_factory=list)


def load_marginals() -> Marginals:
    raw = RECORDED.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECORDED_SHA:
        raise RuntimeError("recorded marginal source changed")
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    by = {f: [] for f in FAMILIES}
    for r in rows:
        by[r["family"]].append(json.loads(r["prompt"][1]["content"]))
    bc = lambda x: bin(x).count("1")
    L, P, R = by["ledger_replay"], by["prerequisite_plan"], by["register_machine"]
    return Marginals(
        ledger_kinds=[p["event"]["kind"] for p in L],
        ledger_status=[p["event"]["status"] for p in L],
        ledger_amounts=[p["event"]["amount"] for p in L],
        ledger_balances=[v for p in L for v in p["before"].values()],
        p_before_bits=[bc(p["before"]) for p in P],
        p_delete_bits=[bc(p["event"]["delete"]) for p in P],
        p_pre_on_bits=[bc(p["event"]["pre_on"]) for p in P],
        p_pre_off_bits=[bc(p["event"]["pre_off"]) for p in P],
        p_costs=[p["event"]["cost"] for p in P],
        r_kinds=[p["event"]["kind"] for p in R],
        r_values=[p["event"]["value"] for p in R],
        r_thresholds=[p["event"]["threshold"] for p in R],
        r_befores=[v for p in R for v in p["before"].values()],
    )


def _pick_bits(rng: random.Random, n: int, pool: list[int]) -> int:
    n = min(max(n, 0), len(pool))
    m = 0
    for b in rng.sample(pool, n):
        m |= 1 << b
    return m


def gen_start(fam: str, rng: random.Random, m: Marginals):
    if fam == "ledger_replay":
        # Recorded balances span [-865, 1018] because they come from long replays; keep the
        # recorded marginal but reject the extreme tail so chains stay in a short-number regime.
        pool = [b for b in m.ledger_balances if -99 <= b <= 99]
        return {a: rng.choice(pool) for a in ACCOUNTS}
    if fam == "prerequisite_plan":
        return _pick_bits(rng, rng.choice(m.p_before_bits), list(range(NBITS)))
    if fam == "register_machine":
        return {r: rng.choice(m.r_befores) for r in REGISTERS}
    raise ValueError(fam)


def gen_event(fam: str, rng: random.Random, m: Marginals, tick: int) -> dict:
    if fam == "ledger_replay":
        a = rng.choice(ACCOUNTS)
        return {"account": a, "amount": rng.choice(m.ledger_amounts), "kind": rng.choice(m.ledger_kinds),
                "other": rng.choice([x for x in ACCOUNTS if x != a]),
                "status": rng.choice(m.ledger_status), "tick": tick}
    if fam == "register_machine":
        r = rng.choice(REGISTERS)
        return {"kind": rng.choice(m.r_kinds), "other": rng.choice([x for x in REGISTERS if x != r]),
                "register": r, "threshold": rng.choice(m.r_thresholds), "tick": tick,
                "value": rng.choice(m.r_values)}
    raise ValueError(fam)


def gen_action_library(rng: random.Random, m: Marginals, n_actions: int = 6) -> list[dict]:
    """Prerequisite actions with recorded bit-count marginals; preconditions over ALL bits, so an
    action can be inapplicable now and applicable after another action sets its pre_on bit."""
    lib, names = [], rng.sample(range(1, 41), n_actions)
    for i in range(n_actions):
        add = 1 << rng.randrange(NBITS)
        pre_on = _pick_bits(rng, rng.choice(m.p_pre_on_bits), [b for b in range(NBITS) if not (add >> b) & 1])
        pre_off = _pick_bits(rng, rng.choice(m.p_pre_off_bits), [b for b in range(NBITS) if not (pre_on >> b) & 1])
        delete = _pick_bits(rng, rng.choice(m.p_delete_bits), [b for b in range(NBITS) if not (add >> b) & 1])
        lib.append({"add": add, "cost": rng.choice(m.p_costs), "delete": delete, "kind": "action",
                    "name": f"A{names[i]}", "pre_off": pre_off, "pre_on": pre_on})
    return lib


# ---------------------------------------------------------------- prompts


def one_step_messages(fam: str, before, ev: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_ONE[fam]},
            {"role": "user", "content": dumps({"before": before, "event": ev})}]


def multi_step_messages(fam: str, before, events: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_MULTI[fam]},
            {"role": "user", "content": dumps({"before": before, "events": events})}]


def event_end_char_offsets(user_json: str, n_events: int) -> list[int]:
    """Character index of the closing brace of each event object inside the compact user JSON.

    The user JSON is dumps({"before":..., "events":[{...},{...}]}); events contain no nested
    objects, so each '}' after the '"events":[' marker closes one event.
    """
    start = user_json.index('"events":[') + len('"events":[')
    ends, i = [], start
    for _ in range(n_events):
        j = user_json.index("}", i)
        ends.append(j)
        i = j + 1
    return ends


_EQ_RHS = re.compile(r'"([A-Za-z]+)"\s*:\s*[-\d\s+*/x()]*?=\s*(-?\d+)')


def _last_after_value(t: str):
    """Value of the last complete `"after": ...` (object or integer) in a possibly truncated text."""
    for m in reversed(list(re.finditer(r'"after"\s*:\s*', t))):
        j = m.end()
        if j < len(t) and t[j] == "{":
            depth = 0
            for k in range(j, len(t)):
                if t[k] == "{":
                    depth += 1
                elif t[k] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[j:k + 1])
                        except (json.JSONDecodeError, ValueError):
                            break
        else:
            n = re.match(r"-?\d+", t[j:])
            if n:
                return int(n.group(0))
    return None


def parse_state(fam: str, text: str, lenient: bool = False):
    """Parse a model answer into a state, or None.

    lenient=True (registered in PREREG-1, identical for every arm) additionally:
      * unwraps {"before":..., "after":...} / {"after":...} answers (verification B3);
      * rewrites `"K": a op b = c` arithmetic inside values to `"K": c`;
      * falls back to the LAST complete "after" value in a text that is not valid JSON.
    Compact JSON answers (Titan's trained format) parse identically under strict and lenient."""
    t = text.strip()
    if t.endswith("."):
        t = t[:-1]
    try:
        v = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        if not lenient:
            return None
        t2 = _EQ_RHS.sub(r'"\1": \2', t)
        try:
            v = json.loads(t2)
        except (json.JSONDecodeError, ValueError):
            v = _last_after_value(t2)
            if v is None:
                return None
    if lenient and isinstance(v, dict) and "after" in v:
        v = v["after"]
    if fam == "prerequisite_plan":
        return v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 2 ** NBITS else None
    keys = ACCOUNTS if fam == "ledger_replay" else REGISTERS
    if not isinstance(v, dict) or set(v) != set(keys):
        return None
    if not all(isinstance(v[k], int) and not isinstance(v[k], bool) for k in keys):
        return None
    return {k: v[k] for k in keys}
