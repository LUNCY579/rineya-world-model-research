"""H2b controls: is the decoded consequence READ from the model, or COMPUTED by the probe?

control_labels  Hewitt–Liang-style selectivity. Same prompts, same captured states, same probe and
                selection; targets are the change mask under a counterfactual rule of matched
                surface complexity that the prompt does NOT describe. A probe that computes the
                answer from surface features fits both rule sets about equally well; a probe that
                reads a consequence the model computed under the stated rule fits the true rule
                better. selectivity = exact-Δ(true) − exact-Δ(cf).
  ledger   cf: account references rotated A->B->C->D->A (matched base rates)
  register cf: conditional comparison inverted (<=) and mix uses 3*other
  prereq   cf: pre_on / pre_off roles swapped
transfer        cross-arm probe transfer: fit on arm A's states, evaluate on arm B's states for the
                same items. High transfer = the delta did not move the consequence code.
"""
from __future__ import annotations

import numpy as np

from . import env, fidelity, heads


def step_cf(fam: str, state, ev: dict):
    if fam == "ledger_replay":
        # matched-difficulty cf: every account reference rotated A->B->C->D->A (same base rates of
        # change, sign flips and void no-ops; a mapping the prompt never describes)
        rot = {a: env.ACCOUNTS[(i + 1) % 4] for i, a in enumerate(env.ACCOUNTS)}
        return env.step(fam, state, dict(ev, account=rot[ev["account"]], other=rot[ev["other"]]))
    if fam == "register_machine":
        reg = dict(state)
        k, r, o, v = ev["kind"], ev["register"], ev.get("other"), ev.get("value", 0)
        if k == "swap":
            reg[r], reg[o] = reg[o], reg[r]
        elif k == "add":
            reg[r] = (reg[r] + v) % env.MOD
        elif k == "mix":
            reg[r] = (reg[r] + 3 * reg[o] + v) % env.MOD
        elif k == "conditional":
            reg[r] = (reg[r] + (v if reg[o] <= ev["threshold"] else -v)) % env.MOD
        return reg
    if fam == "prerequisite_plan":
        swapped = dict(ev, pre_on=ev["pre_off"], pre_off=ev["pre_on"])
        return env.step(fam, state, swapped)
    raise ValueError(fam)


def rollout_cf(fam, s0, events):
    s = s0
    for e in events:
        s = step_cf(fam, s, e)
    return s


def _fit_eval(arm_tr, arm_te, rows, fam, k, target_fn, layer=None, restandardize=False):
    sp = {s: [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == s] for s in ("train", "dev", "test")}
    D = {s: np.stack([target_fn(r) for r in x]) for s, x in sp.items()}
    ids = {s: [r["item_id"] for r in x] for s, x in sp.items()}
    nb = D["train"].shape[1]
    if layer is None:
        layer, rg, _ = fidelity.select_layer_ridge(arm_tr, ids["train"], D["train"], ids["dev"], D["dev"], nb)
    else:
        rg = heads.Ridge().fit(arm_tr.X_last(ids["train"], layer), D["train"], arm_tr.X_last(ids["dev"], layer), D["dev"], nb)
    Xte = arm_te.X_last(ids["test"], layer)
    if restandardize and arm_te is not arm_tr:
        # map B's states into A's standardized coordinates using B's own TRAIN statistics
        std_b = heads.Standardizer(arm_te.X_last(ids["train"], layer))
        Xte = std_b(Xte) * rg.std.sd + rg.std.mu
    m = fidelity.delta_metrics(rg.predict(Xte), D["test"])
    return layer, m


def control_labels(arm, rows, fam, k) -> dict:
    true_fn = lambda r: fidelity.delta_of(fam, r["s0"], r["states"][-1])
    cf_fn = lambda r: fidelity.delta_of(fam, r["s0"], rollout_cf(fam, r["s0"], r["events"]))
    L, mt = _fit_eval(arm, arm, rows, fam, k, true_fn)
    Lc, mc = _fit_eval(arm, arm, rows, fam, k, cf_fn)
    te = [r for r in rows if r["family"] == fam and r["k"] == k and r["split"] == "test"]
    disc = np.array([not np.array_equal(true_fn(r), cf_fn(r)) for r in te])
    same = float(1 - disc.mean())
    sel_disc = float(mt["per_item_exact"][disc].mean() - mc["per_item_exact"][disc].mean()) if disc.any() else float("nan")
    copy_cf = float(np.mean([not cf_fn(r).any() for r in te]))
    return {"true_exact": mt["exact_delta"], "cf_exact": mc["exact_delta"], "selectivity": mt["exact_delta"] - mc["exact_delta"],
            "n_discriminating": int(disc.sum()), "selectivity_discriminating": sel_disc,
            "true_exact_disc": float(mt["per_item_exact"][disc].mean()) if disc.any() else float("nan"),
            "cf_exact_disc": float(mc["per_item_exact"][disc].mean()) if disc.any() else float("nan"),
            "true_layer": L, "cf_layer": Lc, "share_items_true_equals_cf": same,
            "copy_floor_true": float(np.mean([not true_fn(r).any() for r in te])), "copy_floor_cf": copy_cf,
            "per_item_true": mt["per_item_exact"].tolist(), "per_item_cf": mc["per_item_exact"].tolist()}


def transfer(arm_a, arm_b, rows, fam, k) -> dict:
    """Probe fit on A (layer chosen on A's DEV) evaluated on A and on B; and the reverse."""
    fn = lambda r: fidelity.delta_of(fam, r["s0"], r["states"][-1])
    La, aa = _fit_eval(arm_a, arm_a, rows, fam, k, fn)
    _, ab = _fit_eval(arm_a, arm_b, rows, fam, k, fn, layer=La)
    _, abr = _fit_eval(arm_a, arm_b, rows, fam, k, fn, layer=La, restandardize=True)
    Lb, bb = _fit_eval(arm_b, arm_b, rows, fam, k, fn)
    _, ba = _fit_eval(arm_b, arm_a, rows, fam, k, fn, layer=Lb)
    _, bar = _fit_eval(arm_b, arm_a, rows, fam, k, fn, layer=Lb, restandardize=True)
    return {"A_on_A": aa["exact_delta"], "A_on_B": ab["exact_delta"], "A_on_B_restd": abr["exact_delta"],
            "B_on_B": bb["exact_delta"], "B_on_A": ba["exact_delta"], "B_on_A_restd": bar["exact_delta"],
            "layer_A": La, "layer_B": Lb}


def state_shift(arm_a, arm_b, item_ids, layers=None) -> dict:
    """How far the delta moves the residual stream at matched items: relative L2 of the difference,
    cosine, and the mean-vs-centered decomposition (a pure mean shift breaks fixed-standardizer probes)."""
    out = {}
    for L in (layers or arm_a.layers):
        A, B = arm_a.X_last(item_ids, L).astype(np.float64), arm_b.X_last(item_ids, L).astype(np.float64)
        d = B - A
        mean_shift = d.mean(0)
        resid = d - mean_shift
        out[L] = {"rel_l2": float(np.linalg.norm(d) / np.linalg.norm(A)),
                  "rel_l2_mean_part": float(np.linalg.norm(np.tile(mean_shift, (len(d), 1))) / np.linalg.norm(A)),
                  "rel_l2_centered_part": float(np.linalg.norm(resid) / np.linalg.norm(A - A.mean(0))),
                  "cos_mean": float(np.mean((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))))}
    return out
