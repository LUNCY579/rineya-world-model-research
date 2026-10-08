"""Consequence-fidelity analysis with the design-critique amendments (M2-M9).

Primary target is the CHANGE MASK Δ = φ_bin(s_k) XOR φ_bin(s0): a probe gets no credit for merely
re-encoding the start state, and for any state predictor exact-Δ equals exact-φ_bin. Ridge is the
confirmatory readout (M4); the MLP head is secondary. Every comparator — a ladder of answer-free
heuristics, a no-LLM learner and every generative mode — is scored on the same Δ metrics (M2, M8).
"""
from __future__ import annotations

import json

import numpy as np

from . import data, decide, env, heads

SELECT_LAYERS_EXCLUDE = (0,)        # layer 0 at both capture positions embeds a constant token (C4)


def phi_bin(fam: str, s0, state) -> np.ndarray:
    return data.phi_from_state(fam, s0, [], state)[: data.PHI_SPEC[fam]["bin"]]


def delta_of(fam: str, s0, state) -> np.ndarray:
    """Binary change mask relative to the start state's φ_bin."""
    return (phi_bin(fam, s0, state) != phi_bin(fam, s0, s0)).astype(np.float32)


def cont_of(fam: str, s0, state) -> np.ndarray:
    full = data.phi_from_state(fam, s0, [], state)
    return full[data.PHI_SPEC[fam]["bin"]:]


# ---------------------------------------------------------------- heuristic ladder (answer-free)


def _ignore_pre(state, ev):
    return (state & ~ev["delete"]) | ev["add"]


def heuristic_states(fam: str, s0, events: list[dict]) -> dict:
    out = {"copy_s0": s0, "first_only": env.step(fam, s0, events[0]),
           "last_only": env.step(fam, s0, events[-1])}
    if fam == "prerequisite_plan":
        s = s0
        for e in events:
            s = _ignore_pre(s, e)
        out["ignore_pre"] = s
    if fam == "ledger_replay":
        s = s0
        for e in events:
            s = env.step(fam, s, {**e, "status": "posted"})
        out["ignore_status"] = s
    return out


def touched_delta(fam: str, s0, events: list[dict]) -> np.ndarray | None:
    """'changed' features predicted by 'an entity changes iff an applied event names it'."""
    if fam == "ledger_replay":
        named = set()
        for e in events:
            if e["status"] == "posted":
                named.add(e["account"])
                if e["kind"] == "transfer":
                    named.add(e["other"])
        d = np.zeros(8, np.float32)
        d[4:8] = [float(a in named) for a in env.ACCOUNTS]
        return d
    if fam == "register_machine":
        named = set()
        for e in events:
            named.add(e["register"])
            if e["kind"] == "swap":
                named.add(e["other"])
        d = np.zeros(9, np.float32)
        d[6:9] = [float(r in named) for r in env.REGISTERS]
        return d
    return None


# ---------------------------------------------------------------- metrics


def delta_metrics(Dp: np.ndarray, Dt: np.ndarray) -> dict:
    """Dp: predicted change probabilities/masks, Dt: true masks."""
    hit = (Dp > 0.5) == (Dt > 0.5)
    pos, neg = Dt > 0.5, Dt <= 0.5
    tpr = (hit & pos).sum() / max(pos.sum(), 1)
    tnr = (hit & neg).sum() / max(neg.sum(), 1)
    return {"exact_delta": float(hit.all(1).mean()), "feat_acc": float(hit.mean()),
            "balanced_acc": float((tpr + tnr) / 2), "per_feature_acc": [float(x) for x in hit.mean(0)],
            "per_item_exact": hit.all(1).astype(np.float32)}


def strip(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "per_item_exact"}


# ---------------------------------------------------------------- probes


def select_layer_ridge(arm, ids_tr, Ytr, ids_dv, Ydv, nbin, layers=None):
    best = None
    by_layer = {}
    for L in (layers or arm.layers):
        if L in SELECT_LAYERS_EXCLUDE:
            continue
        rg = heads.Ridge().fit(arm.X_last(ids_tr, L), Ytr, arm.X_last(ids_dv, L), Ydv, nbin)
        by_layer[L] = rg.dev_score
        if best is None or rg.dev_score > best[0]:
            best = (rg.dev_score, L, rg)
    return best[1], best[2], by_layer


def input_targets(fam: str, r: dict) -> np.ndarray:
    """M5: what the start state / first event look like, decoded at the same position."""
    s0, e = r["s0"], r["events"][0]
    if fam == "prerequisite_plan":
        x = env.bits(s0)
        x += env.bits(e["add"])
    elif fam == "ledger_replay":
        x = [int(s0[a] < 0) for a in env.ACCOUNTS]
        x += decide.onehot(decide.KINDS_L.index(e["kind"]), 3) + decide.onehot(env.ACCOUNTS.index(e["account"]), 4)
        x += [int(e["status"] == "posted")]
    else:
        x = [int(s0[q] > 48) for q in env.REGISTERS] + [s0[q] % 2 for q in env.REGISTERS]
        x += decide.onehot(decide.KINDS_R.index(e["kind"]), 4) + decide.onehot(env.REGISTERS.index(e["register"]), 3)
    return np.array(x, dtype=np.float32)


def p_strata(r: dict) -> dict:
    fam = r["family"]
    if fam != "prerequisite_plan":
        return {}
    h = heuristic_states(fam, r["s0"], r["events"])
    truth = r["states"][-1]
    return {"contested": truth not in {h["copy_s0"], h["first_only"], h["last_only"], h["ignore_pre"]},
            "discriminating": h["ignore_pre"] != truth}


def analyse(arm, rows: list[dict], fam: str, k: int, gen: dict | None, mlp: bool = True,
            symbolic: bool = True) -> dict:
    sp = {s: [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == s] for s in ("train", "dev", "test")}
    nb = data.PHI_SPEC[fam]["bin"]
    D = {s: np.stack([delta_of(fam, r["s0"], r["states"][-1]) for r in x]) for s, x in sp.items()}
    C = {s: np.stack([cont_of(fam, r["s0"], r["states"][-1]) for r in x]) for s, x in sp.items()} if nb < len(sp["train"][0]["phi"]) else None
    ids = {s: [r["item_id"] for r in x] for s, x in sp.items()}
    out = {"n_test": len(sp["test"]), "n_test_parents": len({r["parent_id"] for r in sp["test"]})}

    # Latent ridge on Δ (confirmatory) — layer chosen on DEV
    L, rg, by_layer = select_layer_ridge(arm, ids["train"], D["train"], ids["dev"], D["dev"], nb)
    Xte = arm.X_last(ids["test"], L)
    lat = delta_metrics(rg.predict(Xte), D["test"])
    out["latent_ridge"] = {**strip(lat), "layer": L, "lambda": rg.lam, "dev_by_layer": by_layer}
    per_item = {"latent_ridge": lat["per_item_exact"]}
    if C is not None:
        rc = heads.Ridge().fit(arm.X_last(ids["train"], L), C["train"], arm.X_last(ids["dev"], L), C["dev"], 0)
        pc = rc.predict(Xte)
        out["latent_ridge"]["cont_r2"] = float(np.mean(1 - ((C["test"] - pc) ** 2).sum(0) / (((C["test"] - C["test"].mean(0)) ** 2).sum(0) + 1e-9)))
    if mlp:
        mh = heads.MLPHead().fit(arm.X_last(ids["train"], L), D["train"], arm.X_last(ids["dev"], L), D["dev"], nb)
        m = delta_metrics(mh.predict(Xte), D["test"])
        out["latent_mlp"] = {**strip(m), "lr": mh.lr}
        per_item["latent_mlp"] = m["per_item_exact"]

    # M5 input decodability at the same position/layer
    I = {s: np.stack([input_targets(fam, r) for r in sp[s]]) for s in sp}
    ri = heads.Ridge().fit(arm.X_last(ids["train"], L), I["train"], arm.X_last(ids["dev"], L), I["dev"], I["train"].shape[1])
    pi = ri.predict(Xte)
    out["input_decodability"] = {"feat_acc": float(((pi > 0.5) == (I["test"] > 0.5)).mean()),
                                 "all_exact": float(((pi > 0.5) == (I["test"] > 0.5)).all(1).mean())}

    # M2 ladder
    ladder = {}
    for r in sp["test"]:
        for name, st in heuristic_states(fam, r["s0"], r["events"]).items():
            ladder.setdefault(name, []).append(delta_of(fam, r["s0"], st))
        td = touched_delta(fam, r["s0"], r["events"])
        if td is not None:
            ladder.setdefault("touched_rule", []).append(td)
    lad = {}
    for name, v in ladder.items():
        m = delta_metrics(np.stack(v), D["test"])
        lad[name] = strip(m)
        per_item[f"floor_{name}"] = m["per_item_exact"]
    maj = (D["train"].mean(0, keepdims=True) > 0.5).astype(np.float32).repeat(len(sp["test"]), 0)
    lad["majority"] = strip(delta_metrics(maj, D["test"]))
    if symbolic:
        enc = {s: np.stack([decide.symbolic(fam, r["s0"], r["events"]) for r in sp[s]]) for s in sp}
        sr = heads.Ridge().fit(enc["train"], D["train"], enc["dev"], D["dev"], nb)
        m = delta_metrics(sr.predict(enc["test"]), D["test"]); lad["S_ridge"] = strip(m); per_item["S_ridge"] = m["per_item_exact"]
        sm = heads.MLPHead().fit(enc["train"], D["train"], enc["dev"], D["dev"], nb)
        m = delta_metrics(sm.predict(enc["test"]), D["test"]); lad["S_mlp"] = strip(m); per_item["S_mlp"] = m["per_item_exact"]
    out["ladder"] = lad

    # Generative modes (M6, M7)
    gens = {}
    if gen:
        for mode in ("e2e", "chain_one", "chain_m1"):
            if mode not in gen:
                continue
            preds, parsed = [], []
            for r in sp["test"]:
                st = _gen_state(gen, mode, r)
                parsed.append(st is not None)
                preds.append(st)
            wrong = np.full(nb, -1.0, np.float32)
            Dp = np.stack([delta_of(fam, r["s0"], st) if st is not None else wrong for r, st in zip(sp["test"], preds)])
            m = delta_metrics(np.where(Dp < 0, 1 - D["test"], Dp), D["test"])   # unparseable = every feature wrong
            Di = np.stack([delta_of(fam, r["s0"], st if st is not None else r["s0"]) for r, st in zip(sp["test"], preds)])
            mi = delta_metrics(Di, D["test"])
            gens[mode] = {**strip(m), "parse_rate": float(np.mean(parsed)),
                          "exact_delta_impute_s0": mi["exact_delta"]}
            per_item[f"G_{mode}"] = m["per_item_exact"]
            if fam == "prerequisite_plan" and mode == "chain_one":
                gens[mode]["per_step"] = _p_chain_steps(gen, sp["test"])
        out["generative"] = gens

    # M8 comparator: best non-latent exact-Δ on test (conservative for H1). Deviation from the
    # critique, recorded in DECISIONS.md: ignore_pre / ignore_status are full simulators with ONE
    # rule removed (they execute every other transition exactly), so they are reported as
    # rule-ablation diagnostics (H4) rather than counted as answer-free floors for H1.
    # first_only / last_only also execute TRUE transitions (at k=1 they ARE the oracle), so they
    # join the partial-simulator diagnostics rather than the H1 floor.
    RULE_ABLATED = ("ignore_pre", "ignore_status", "first_only", "last_only")
    out["rule_ablated"] = {n: lad[n]["exact_delta"] for n in RULE_ABLATED if n in lad}
    comp = {f"floor_{n}": v["exact_delta"] for n, v in lad.items() if n not in ("S_ridge", "S_mlp") + RULE_ABLATED}
    comp.update({f"S_{n[2:]}": lad[n]["exact_delta"] for n in ("S_ridge", "S_mlp") if n in lad})
    comp.update({f"G_{n}": v["exact_delta"] for n, v in gens.items()})
    best_name = max(comp, key=comp.get)
    out["comparator"] = {"best": best_name, "exact_delta": comp[best_name], "all": comp}
    ckey = best_name if best_name in per_item else None
    if ckey:
        out["H1_contrast"] = decide.paired_bootstrap(_parent_mean(sp["test"], per_item["latent_ridge"]),
                                                     _parent_mean(sp["test"], per_item[ckey]))

    # M9 strata (P)
    if fam == "prerequisite_plan":
        st = [p_strata(r) for r in sp["test"]]
        for sname in ("contested", "discriminating"):
            mask = np.array([s[sname] for s in st])
            out.setdefault("strata", {})[sname] = {
                "n": int(mask.sum()), "n_parents": len({r["parent_id"] for r, m_ in zip(sp["test"], mask) if m_}),
                **{name: float(v[mask].mean()) if mask.any() else None for name, v in per_item.items()}}
    out["_per_item"] = {k_: v.tolist() for k_, v in per_item.items()}
    out["_test_parents"] = [r["parent_id"] for r in sp["test"]]
    return out


def _parent_mean(rows, v):
    pids = sorted({r["parent_id"] for r in rows})
    ix = {p: i for i, p in enumerate(pids)}
    s, n = np.zeros(len(pids)), np.zeros(len(pids))
    for r, x in zip(rows, v):
        s[ix[r["parent_id"]]] += x
        n[ix[r["parent_id"]]] += 1
    return s / n


def _gen_state(gen, mode, r):
    fam = r["family"]
    if mode == "e2e":
        g = gen["e2e"].get(r["item_id"])
        # lenient: unwrap base Supa's {"before","after"} answers (verification B3; conservative for Titan > Supa)
        return None if g is None else env.parse_state(fam, g["text"], lenient=True)
    g = gen[mode].get(r["cand_id"])
    s = None if g is None else g["states"][r["k"]]
    return None if s is None else json.loads(s)


def _p_chain_steps(gen, test_rows):
    """M6: chain_one per-step accuracy for P, split by applicability, against both the true step
    and the ignore-preconditions step, evaluated from the model's OWN previous state."""
    stats = {"applicable": [0, 0, 0], "inapplicable": [0, 0, 0]}   # n, matches truth, matches ignore-pre
    seen = set()
    for r in test_rows:
        if r["cand_id"] in seen:
            continue
        seen.add(r["cand_id"])
        g = gen["chain_one"].get(r["cand_id"])
        if g is None:
            continue
        states = [None if s is None else json.loads(s) for s in g["states"]]
        for t in range(min(len(states) - 1, len(r["events"]))):
            prev, nxt = states[t], states[t + 1]
            if prev is None:
                break
            e = r["events"][t]
            key = "applicable" if env.applicable(prev, e) else "inapplicable"
            stats[key][0] += 1
            stats[key][1] += int(nxt == env.step("prerequisite_plan", prev, e))
            stats[key][2] += int(nxt == _ignore_pre(prev, e))
    return {k: {"n": v[0], "acc_true": v[1] / max(v[0], 1), "acc_ignore_pre": v[2] / max(v[0], 1)} for k, v in stats.items()}


def tracking(arm, rows: list[dict], fam: str, H: int = 4) -> dict:
    """Latent state tracking: residual stream at action boundary j of the k=H item -> change mask
    of the prefix state after j events (relative to s0). Ridge, layer chosen on DEV per j.
    Compared against the answer-slot readout of the k=j item in `analyse`."""
    sp = {s: [r for r in rows if r["family"] == fam and r["k"] == H and r["split"] == s] for s in ("train", "dev", "test")}
    nb = data.PHI_SPEC[fam]["bin"]
    out = {}
    for j in range(1, H + 1):
        D = {s: np.stack([delta_of(fam, r["s0"], r["states"][j]) for r in x]) for s, x in sp.items()}
        ids = {s: [r["item_id"] for r in x] for s, x in sp.items()}
        best = None
        for L in arm.layers:
            if L in SELECT_LAYERS_EXCLUDE:
                continue
            rg = heads.Ridge().fit(arm.X_event(ids["train"], j, L), D["train"], arm.X_event(ids["dev"], j, L), D["dev"], nb)
            if best is None or rg.dev_score > best[0]:
                best = (rg.dev_score, L, rg)
        _, Lb, rg = best
        m = delta_metrics(rg.predict(arm.X_event(ids["test"], j, Lb)), D["test"])
        copy = delta_metrics(np.zeros_like(D["test"]), D["test"])
        out[j] = {"layer": Lb, **strip(m), "copy_floor_exact": copy["exact_delta"]}
    return out
