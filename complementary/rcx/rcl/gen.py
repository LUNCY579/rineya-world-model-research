"""RCL generators: training regimes, test battery strata, surfaces and splits (SPEC_RCL.md 5-7).

Every record is produced from a semantic parent first; renderings are chosen afterwards.
Train, dev and sealed draw from split-specific vocabulary pools and seeds.
"""
import itertools
import random

from . import templates as T
from . import vocab as V
from . import world as W

LETTERS = ("A", "B", "C", "D")
LETTERS2 = ("P", "Q", "R", "S", "T", "U")
WORDS = ("north", "south", "east", "west", "pine", "oak")

CANON_EVENT_ORDER = W.EVENT_KEYS
HELDOUT_EVENT_ORDERS = (
    ("status", "other", "kind", "amount", "account"),
    ("kind", "status", "account", "other", "amount"),
    ("other", "amount", "status", "account", "kind"),
)
TRAIN_EVENT_ORDERS = tuple(p for p in itertools.permutations(W.EVENT_KEYS) if p not in HELDOUT_EVENT_ORDERS)

# Composition combos (connective, other-atom op). Train vs held-out (V6) are disjoint.
TRAIN_COMBOS = (("AND", "K+"), ("OR", "ALT"), ("AND", "AGE"))
HELDOUT_COMBOS = (("OR", "K+"), ("AND", "ALT"), ("OR", "AGE"))

REGIMES = ("FX", "FX_SD", "VR", "VR_SD", "CF", "CF_SD")

# World variant: "arith" = RCWM-compatible kinds (add, transfer); "copy" = arithmetic-free kinds (set, swap),
# used by the CPU mechanism tier to factor out the calculation A and study applicability C (SPEC_RCL 1.1).
WORLDS = {"arith": ("add", "transfer"), "copy": ("set", "swap")}
KINDS = WORLDS["arith"]


def use_world(name):
    """Select the event-kind variant for all subsequent generation in this process."""
    global KINDS
    KINDS = WORLDS[name]
    return KINDS


def _thr_range():
    return (10, 90) if "set" in KINDS else (2, 10)
STRATA = ("F1_void", "F2_unseen_real", "F3_unseen_nonce", "F4_nearmiss", "F5_postedlike",
          "V1_pos", "V2_neg", "V3_inverted", "V5_posted_override", "V6_compose", "V7_bigset",
          "M1_bal", "M2_amt")
SURFACES = ("in", "ho", "ho_tpl", "ho_key", "ho_name", "ho_many")
POSTED = V.ANCHOR_APPLY
VOID = V.ANCHOR_NOOP


# ----------------------------------------------------------------------------- semantic parents

def _event(rng, accounts, kind=None, amt_range="std"):
    kind = kind or rng.choice(KINDS)
    acc, other = rng.sample(list(accounts), 2)
    if kind == "set":
        amount = rng.randint(100, 999) if amt_range == "big" else rng.randint(0, 98)
    elif kind == "swap":
        amount = rng.randint(13, 99) if amt_range == "big" else rng.randint(1, 12)
    elif amt_range == "big":
        amount = rng.choice([a for a in range(-99, 100) if abs(a) >= 13]) if kind == "add" \
            else rng.randint(13, 99)
    else:
        amount = rng.choice([a for a in range(-9, 13) if a != 0]) if kind == "add" else rng.randint(1, 12)
    return {"account": acc, "amount": amount, "kind": kind, "other": other, "status": None}


def _state(rng, accounts, bal_range="std"):
    lo, hi = (0, 98) if bal_range == "std" else (100, 999)
    return {a: rng.randint(lo, hi) for a in accounts}


class WordSource:
    """Draws status words for a split. mix_nonce in [0,1] = probability a word is nonce."""

    def __init__(self, split, mix_nonce, extra=()):
        self.real, self.nonce = V.pools(split)
        self.mix_nonce = mix_nonce
        self.extra = tuple(extra)

    def draw(self, rng, k=1, exclude=()):
        out = []
        guard = 0
        while len(out) < k:
            guard += 1
            assert guard < 10000
            if self.extra and rng.random() < 0.15:
                w = rng.choice(self.extra)
            elif rng.random() < self.mix_nonce:
                w = rng.choice(self.nonce)
            else:
                w = rng.choice(self.real)
            if w in out or w in exclude:
                continue
            out.append(w)
        return out


def _status_atom(rng, ws, polarity=None, size=None):
    pol = polarity or rng.choice(("S+", "S-"))
    size = size or rng.choice((1, 2))
    return {"op": pol, "set": ws.draw(rng, size)}


def _other_atom(rng, op):
    if op == "K+":
        return {"op": "K+", "kinds": [rng.choice(KINDS)]}
    if op in ("AGE", "ALT"):
        return {"op": op, "t": rng.randint(*_thr_range())}
    if op == "BGE":
        return {"op": op, "t": rng.randint(10, 90)}
    raise ValueError(op)


def _set_status(event, status):
    e = dict(event)
    e["status"] = status
    return e


def _status_for(rng, ws, atom, want_listed):
    if want_listed:
        return rng.choice(atom["set"])
    return ws.draw(rng, 1, exclude=atom["set"])[0]


# ----------------------------------------------------------------------------- surfaces

def surface_spec(rng, mode, n_acc_default=4):
    """Rendering factors for a surface mode. Returns dict(tid, names, n_acc, state_perm, event_order)."""
    tpl_ho = mode in ("ho", "ho_tpl")
    key_ho = mode in ("ho", "ho_key")
    name_ho = mode in ("ho", "ho_name")
    many = mode in ("ho", "ho_many")
    sd = mode == "sd"
    if tpl_ho:
        tid = rng.choice(T.HELDOUT_IDS)
    elif sd:
        tid = rng.choice(T.TRAIN_IDS)
    else:
        tid = "t0"
    n_acc = rng.choice((5, 6)) if many else (rng.choice((3, 4)) if sd else n_acc_default)
    if name_ho:
        names = LETTERS2 if rng.random() < 0.5 else WORDS
    else:
        names = LETTERS + ("E", "F")  # E,F only appear when many accounts are requested
    if key_ho:
        event_order = rng.choice(HELDOUT_EVENT_ORDERS)
        state_perm = "reversed"
    elif sd:
        event_order = rng.choice(TRAIN_EVENT_ORDERS)
        state_perm = "shuffle"
    else:
        event_order = CANON_EVENT_ORDER
        state_perm = "sorted"
    return {"mode": mode, "tid": tid, "names": tuple(names[:n_acc]), "n_acc": n_acc,
            "state_perm": state_perm, "event_order": tuple(event_order)}


def _state_order(rng, keys, perm):
    keys = list(keys)
    if perm == "sorted":
        return keys
    if perm == "reversed":
        return keys[::-1]
    for _ in range(50):
        p = keys[:]
        rng.shuffle(p)
        if p != keys[::-1] or len(keys) < 3:
            return p
    return keys


def realize(parent_core, surf, rng):
    """Map a canonical parent core (accounts A..D) onto a surface: rename accounts, add extra
    untouched accounts for many-account surfaces, fix key orders. Returns (state, event, state_order)."""
    st, ev = parent_core["state"], parent_core["event"]
    base_keys = list(st.keys())
    n_acc = max(surf["n_acc"], len(base_keys)) if surf["mode"] != "sd" else surf["n_acc"]
    if surf["mode"] == "sd" and n_acc < len(base_keys):
        # drop untouched accounts to reach n_acc
        keep = [k for k in base_keys if k in (ev["account"], ev["other"])]
        rest = [k for k in base_keys if k not in keep]
        keep += rest[: n_acc - len(keep)]
        base_keys = [k for k in base_keys if k in keep]
    extra_rng = random.Random(parent_core["extra_seed"])
    vals = [st[k] for k in base_keys]
    while len(vals) < n_acc:
        vals.append(extra_rng.randint(0, 98) if parent_core.get("bal_range", "std") == "std"
                    else extra_rng.randint(100, 999))
    names = surf["names"]
    mapping = {k: names[i] for i, k in enumerate(base_keys)}
    state = {names[i]: vals[i] for i in range(n_acc)}
    event = dict(ev)
    event["account"] = mapping[ev["account"]]
    event["other"] = mapping[ev["other"]]
    order = _state_order(rng, list(state.keys()), surf["state_perm"])
    return state, event, order


def make_record(rule, parent_core, surf, rng, meta):
    state, event, order = realize(parent_core, surf, rng)
    ok, after = W.transition(rule, state, event)
    text = W.render_input(rule, state, event, surf["tid"], order, surf["event_order"])
    rec = {
        "parent_id": meta["parent_id"], "split": meta["split"], "group": meta["group"],
        "variant": meta["variant"], "surface": surf["mode"], "tid": surf["tid"],
        "rule_ast": rule, "state": {k: state[k] for k in order}, "event": {k: event[k] for k in surf["event_order"]},
        "state_order": order, "input": text,
        "gold": {"ok": ok, "listed": W.listed(rule, event), "after": {k: after[k] for k in order},
                 "decision": "apply" if ok else "noop"},
        "targets": {"ans": W.target_ans(after, order), "trace": W.target_trace(rule, state, event, order)},
        "semantic_hash": W.semantic_hash(rule, state, event),
        "skeleton_hash": W.skeleton_hash(parent_core["state"], parent_core["event"]),
    }
    return rec


def _core(rng, bal_range="std", amt_range="std", kind=None):
    st = _state(rng, LETTERS, bal_range)
    ev = _event(rng, LETTERS, kind=kind, amt_range=amt_range)
    return {"state": st, "event": ev, "extra_seed": rng.randrange(2 ** 31), "bal_range": bal_range}


# ----------------------------------------------------------------------------- training regimes

def _varied_rule(rng, ws, combos=TRAIN_COMBOS, p_compose=0.3, size_choices=(1, 2)):
    sa = _status_atom(rng, ws, size=rng.choice(size_choices))
    if rng.random() < p_compose:
        conn, op = rng.choice(combos)
        return {"op": conn, "args": [sa, _other_atom(rng, op)]}
    return sa


def _flip_status_atom(rule):
    def flip(a):
        if a["op"] in W.STATUS_OPS:
            return {"op": "S-" if a["op"] == "S+" else "S+", "set": list(a["set"])}
        return a
    if rule["op"] in ("AND", "OR"):
        return {"op": rule["op"], "args": [flip(a) for a in rule["args"]]}
    return flip(rule)


def _replace_status_set(rule, new_set):
    def rep(a):
        if a["op"] in W.STATUS_OPS:
            return {"op": a["op"], "set": list(new_set)}
        return a
    if rule["op"] in ("AND", "OR"):
        return {"op": rule["op"], "args": [rep(a) for a in rule["args"]]}
    return rep(rule)


def train_parent_variants(regime, rng, ws):
    """Return a list of (rule, event_status, variant_name) for one TRAIN parent, plus the core."""
    core = _core(rng)
    if regime in ("FX", "FX_SD"):
        rule = {"op": "S+", "set": [POSTED]}
        if regime == "FX":
            status = POSTED if rng.random() < 0.75 else VOID
        else:
            u = rng.random()
            status = POSTED if u < 0.5 else (VOID if u < 0.6 else ws.draw(rng, 1, exclude=(POSTED,))[0])
        return core, [(rule, status, "base")]
    rule = _varied_rule(rng, ws)
    sa = W.status_atom(rule)
    want = rng.random() < 0.5
    status = _status_for(rng, ws, sa, want)
    if regime in ("VR", "VR_SD"):
        return core, [(rule, status, "base")]
    # Counterfactual parents: base + rule_flip + status_swap + set_swap
    out = [(rule, status, "base"), (_flip_status_atom(rule), status, "rule_flip")]
    sw = _status_for(rng, ws, sa, not want)
    out.append((rule, sw, "status_swap"))
    excl = set(sa["set"]) | {status}
    new_set = ws.draw(rng, len(sa["set"]), exclude=excl) if want else \
        ([status] + ws.draw(rng, len(sa["set"]) - 1, exclude=excl))
    out.append((_replace_status_set(rule, new_set), status, "set_swap"))
    return core, out


def regime_wordsource(regime):
    sd = regime.endswith("_SD")
    extra = (POSTED, VOID)
    return WordSource("train", mix_nonce=0.5 if sd else 0.0, extra=extra)


def make_train(regime, n_examples, seed):
    """Matched-size training set for a regime. All records are TRAIN split."""
    assert regime in REGIMES
    rng = random.Random(seed)
    ws = regime_wordsource(regime)
    sd = regime.endswith("_SD")
    recs, pid = [], 0
    while len(recs) < n_examples:
        core, variants = train_parent_variants(regime, rng, ws)
        surf_rng = random.Random(rng.randrange(2 ** 31))
        for rule, status, vname in variants:
            if len(recs) >= n_examples:
                break
            core_v = dict(core)
            core_v["event"] = _set_status(core["event"], status)
            surf = surface_spec(surf_rng, "sd" if sd else "in")
            meta = {"parent_id": f"tr-{regime}-{seed}-{pid}", "split": "train", "group": regime, "variant": vname}
            recs.append(make_record(rule, core_v, surf, surf_rng, meta))
        pid += 1
    return recs


def fill_units(train_recs):
    """Constant filler length per event kind, from TRAIN trace targets only (SPEC 4; D-36 4b analog)."""
    mx = {}
    for r in train_recs:
        k = r["event"]["kind"]
        mx[k] = max(mx.get(k, 0), len(r["targets"]["trace"]))
    return {k: (v + 1) // 2 for k, v in mx.items()}


def with_fill(rec, units):
    r = dict(rec)
    t = dict(r["targets"])
    t["fill"] = W.target_fill(r["gold"]["after"], r["state_order"], units[r["event"]["kind"]])
    r["targets"] = t
    return r


# ----------------------------------------------------------------------------- test battery

def _stratum_parent(stratum, rng, ws_real, ws_nonce, ws_mix):
    """Return (core, [(rule, status, variant)]) for one test parent of a stratum."""
    fixed = {"op": "S+", "set": [POSTED]}
    if stratum.startswith("F"):
        core = _core(rng)
        if stratum == "F1_void":
            w = VOID
        elif stratum == "F2_unseen_real":
            w = ws_real.draw(rng)[0]
        elif stratum == "F3_unseen_nonce":
            w = ws_nonce.draw(rng)[0]
        elif stratum == "F4_nearmiss":
            w = rng.choice(V.NEARMISS_VOID)
        elif stratum == "F5_postedlike":
            w = rng.choice(V.POSTEDLIKE)
        else:
            raise ValueError(stratum)
        return core, [(fixed, POSTED, "twin_posted"), (fixed, w, "probe")]
    if stratum in ("V1_pos", "V2_neg", "V7_bigset", "M1_bal", "M2_amt"):
        core = _core(rng, bal_range="big" if stratum == "M1_bal" else "std",
                     amt_range="big" if stratum == "M2_amt" else "std")
        pol = {"V1_pos": "S+", "V2_neg": "S-"}.get(stratum, rng.choice(("S+", "S-")))
        size = 3 if stratum == "V7_bigset" else rng.choice((1, 2))
        atom = {"op": pol, "set": ws_mix.draw(rng, size)}
        inw = rng.choice(atom["set"])
        outw = ws_mix.draw(rng, 1, exclude=atom["set"])[0]
        return core, [(atom, inw, "listed"), (atom, outw, "unlisted")]
    if stratum == "V3_inverted":
        core = _core(rng)
        x = ws_mix.draw(rng)[0]
        return core, [({"op": "S+", "set": [x]}, x, "pos"), ({"op": "S-", "set": [x]}, x, "neg")]
    if stratum == "V5_posted_override":
        core = _core(rng)
        x = ws_mix.draw(rng)[0]
        return core, [({"op": "S+", "set": [x]}, POSTED, "pos_x_status_posted"),
                      ({"op": "S-", "set": [POSTED]}, POSTED, "neg_posted_status_posted"),
                      ({"op": "S-", "set": [x]}, POSTED, "neg_x_status_posted"),
                      ({"op": "S+", "set": [x]}, x, "pos_x_status_x")]
    if stratum == "V6_compose":
        conn, op = rng.choice(HELDOUT_COMBOS)
        kind = rng.choice(KINDS)
        core = _core(rng, kind=kind)
        sa = _status_atom(rng, ws_mix, size=1)
        oa = _other_atom(rng, op)
        rule = {"op": conn, "args": [sa, oa]}
        # choose two statuses (listed / unlisted) and keep the pair only if decisions differ
        inw = sa["set"][0]
        outw = ws_mix.draw(rng, 1, exclude=sa["set"])[0]
        return core, [(rule, inw, "listed"), (rule, outw, "unlisted")]
    raise ValueError(stratum)


def make_battery(split, n_parents, seed, strata=STRATA, surfaces=SURFACES):
    """Test battery for split in {'dev','sealed'}: every stratum x surface, paired by parent.
    Parents whose variants do not differ in gold decision are rejected for twin strata (except
    F-strata, where the posted twin always applies and the probe is always a no-op)."""
    assert split in ("dev", "sealed")
    rng = random.Random(seed)
    ws_real = WordSource(split, 0.0)
    ws_nonce = WordSource(split, 1.0)
    ws_mix = WordSource(split, 0.5)
    recs = []
    for stratum in strata:
        made = 0
        tries = 0
        while made < n_parents:
            tries += 1
            assert tries < n_parents * 50, stratum
            core, variants = _stratum_parent(stratum, rng, ws_real, ws_nonce, ws_mix)
            decisions = set()
            for rule, status, _ in variants:
                ev = _set_status(core["event"], status)
                decisions.add(W.eval_rule(rule, core["state"], ev))
            if len(decisions) < 2:
                continue  # every test parent must contain both an apply and a no-op variant
            pid = f"{split}-{stratum}-{made}"
            surf_seed = rng.randrange(2 ** 31)
            for mode in surfaces:
                srng = random.Random(f"{surf_seed}-{mode}")
                surf = surface_spec(srng, mode)
                for rule, status, vname in variants:
                    core_v = dict(core)
                    core_v["event"] = _set_status(core["event"], status)
                    meta = {"parent_id": pid, "split": split, "group": stratum, "variant": vname}
                    recs.append(make_record(rule, core_v, surf, srng, meta))
            made += 1
    return recs
