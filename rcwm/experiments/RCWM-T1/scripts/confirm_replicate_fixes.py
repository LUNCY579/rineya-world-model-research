"""Post-verification fixes to the EXPLORATORY confirmation-bank replication (scripts/confirm_replicate.py).

Not PREREG-1, not preregistered, nothing tuned, selected or thresholded on the confirmation data. This
script changes no number of confirm_replicate.json; it re-runs (A)/(B) with corrected reporting and adds
pilot-defined checks that the first run left out. It imports confirm_replicate (unchanged, so the stored
(C) part files were produced by the code on disk, which is hash-checked) and writes:
  runs/bank_pilot/analysis/confirm_replicate_fixes.{json,log}
  docs/replication/CONFIRM_REPLICATION.md

Fixes (verifier report on the first run):
  F1 (SHOULD) (A) R add: the "dominant error" is reported by the typology's actual mode (wrong value),
     with missing-mod-97 as a secondary count, and the add/mix wrap strata are split by wrap direction
     (raw = r + v (+2*other) >= 97: upward; < 0: downward). The split is descriptive; it selects nothing.
  F2 (NOTE) (C) every pilot-vs-confirmation contrast is classified (CI excludes 0 or not; sign of the
     point estimates) instead of a hand count.
  F3 (NOTE) (B) L void first steps: "applied as posted" is counted among WRONG outputs; the void steps
     whose posted application is a no-op (so posted == gold) are counted separately.
  F4 (NOTE) (C) P: the pilot's leave-out-overlap check (s_roll.run_family: drop parents having any
     transition whose learner input occurs in pilot TRAIN), on the stored per-parent arrays.
  F5 (NOTE) (B) the share "wrong output == state + add" is given over PARSED wrong outputs, R10's own
     denominator ("50 of the 77"), as well as over all wrong outputs; the case-mix re-weighting is labelled
     post hoc / descriptive, and the within-cell accuracy is shown with a CI.

CPU only (CUDA hidden, DECISIONS I-21), 2 threads, below-normal priority. Runs in seconds.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["RCWM_THREADS"] = "2"
import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.set_num_threads(2)

import collections  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import confirm_replicate as CR  # noqa: E402  (unchanged; import only)

SR, env, decide = CR.SR, CR.env, CR.decide
assert not torch.cuda.is_available() and torch.cuda.device_count() == 0
FAMS, SHORT, KS = CR.FAMS, CR.SHORT, CR.KS
OUT = CR.OUT
DOC = ROOT / "docs" / "replication"
S2F = {v: k for k, v in SHORT.items()}


# ============================================================================ F1: R wrap direction

def r_raw(inp, e):
    k, r, o, v = e["kind"], e["register"], e.get("other"), e.get("value", 0)
    if k == "add":
        return inp[r] + v
    if k == "mix":
        return inp[r] + 2 * inp[o] + v
    if k == "conditional":
        return inp[r] + (v if inp[o] > e["threshold"] else -v)
    return None


def r_wrap_detail(gen, cands, rule):
    """R add / mix / conditional steps (same selection as CR.step_records) split by wrap direction of the
    gold update: up (raw >= 97), down (raw < 0), none. Per cell: n, right, error types (CR.err_type),
    outputs equal to the unreduced raw value. Downward-wrap add steps are listed in full."""
    fam = "register_machine"
    cells, examples = {}, []
    for c in cands:
        g = gen["chain_one"].get(c["cand_id"])
        if g is None:
            continue
        ps = [None if s is None else json.loads(s) for s in g["states"]]
        for t in range(CR.H):
            if rule == "step1" and t > 0:
                break
            if ps[t] is None or ps[t] != c["gold"][t]:
                continue
            inp, want, e = c["gold"][t], c["gold"][t + 1], c["events"][t]
            if e["kind"] == "swap":
                continue
            raw = r_raw(inp, e)
            d = "up" if raw >= env.MOD else ("down" if raw < 0 else "none")
            out = ps[t + 1]
            ok = out is not None and out == want
            cc = cells.setdefault(f"{e['kind']}|wrap_{d}", collections.Counter())
            cc["n"] += 1
            cc["right"] += int(ok)
            if not ok:
                cc["err:" + CR.err_type(fam, inp, e, out, want)] += 1
                cc["out_eq_unreduced_raw"] += int(out is not None and out.get(e["register"]) == raw)
            if e["kind"] == "add" and d == "down":
                examples.append({"cand_id": c["cand_id"], "t": t, "in": inp[e["register"]], "v": e["value"], "raw": raw,
                                 "gold": want[e["register"]], "titan": None if out is None else out.get(e["register"]),
                                 "right": bool(ok)})
    return {"cells": {k: dict(v) for k, v in sorted(cells.items())}, "add_downward_wrap_steps": examples}


# ============================================================================ F3: L void anatomy

def l_void_detail(gen, cands):
    fam = "ledger_replay"
    c = collections.Counter()
    for cd in cands:
        s, e, true = cd["s0"], cd["events"][0], cd["gold"][1]
        if e["status"] == "posted":
            continue
        g = gen["chain_one"].get(cd["cand_id"])
        st = None if g is None else g["states"][1]
        t = None if st is None else json.loads(st)
        posted = env.step(fam, s, {**e, "status": "posted"})
        right = t == true
        c["void_first_steps"] += 1
        c["right"] += int(right)
        c["wrong"] += int(not right)
        c["posted_application_is_noop"] += int(posted == true)
        c["right_and_posted_is_noop"] += int(right and posted == true)
        c["wrong_eq_applied_as_posted"] += int((not right) and t == posted)
        c["any_eq_applied_as_posted (s_roll.titan_error_anatomy)"] += int(t == posted)
    return dict(c)


# ============================================================================ F5: P shares with CIs

def cluster_ratio_ci(pairs, n_boot=2000, seed=17):
    """pairs: per-parent [numerator, denominator]; parent-cluster bootstrap of sum(num)/sum(den)."""
    arr = np.array([p for p in pairs if p[1] > 0], np.float64)
    idx = np.random.default_rng(seed).integers(0, len(arr), (n_boot, len(arr)))
    r = arr[idx, 0].sum(1) / arr[idx, 1].sum(1)
    return [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]


def p_shares(gen, cands):
    pp_parsed, pp_cell = {}, {}
    for c in cands:
        s, e = c["s0"], c["events"][0]
        if not env.applicable(s, e):
            continue
        g = gen["chain_one"].get(c["cand_id"])
        st = None if g is None else g["states"][1]
        out = None if st is None else json.loads(st)
        true = c["gold"][1]
        right = out == true
        if not right and out is not None:
            a = pp_parsed.setdefault(c["parent_id"], [0, 0])
            a[0] += int(out == s + e["add"])
            a[1] += 1
        if not (s & e["add"]) and not (s & e["delete"]):     # the cell where state + add == gold
            a = pp_cell.setdefault(c["parent_id"], [0, 0])
            a[0] += int(right)
            a[1] += 1
    num, den = sum(v[0] for v in pp_parsed.values()), sum(v[1] for v in pp_parsed.values())
    cn, cd = sum(v[0] for v in pp_cell.values()), sum(v[1] for v in pp_cell.values())
    return {"wrong_parsed_eq_state_plus_add": [num, den], "share_of_parsed_wrong": num / den,
            "share_of_parsed_wrong_ci95_parent_cluster": cluster_ratio_ci(list(pp_parsed.values())),
            "cell_add_clear_delete_nothing_right": [cn, cd], "cell_add_clear_delete_nothing_acc": cn / cd,
            "cell_add_clear_delete_nothing_acc_ci95_parent_cluster": cluster_ratio_ci(list(pp_cell.values()))}


# ============================================================================ F4: P leave-out-overlap

def overlap_pids(fam, P, C):
    tr = SR.dedupe(fam, P["T"][fam]["train"])
    keys = {SR.enc(fam, r["s"], r["e"]).tobytes() for r in tr}
    return sorted({r["parent_id"] for r in C["T"][fam] if SR.enc(fam, r["s"], r["e"]).tobytes() in keys})


def excl_overlap(parts, titan_c, P, C, log):
    out = {}
    for fam in FAMS:
        f = SHORT[fam]
        ov = overlap_pids(fam, P, C)
        out[f] = {"n_overlap_parents": len(ov), "contrasts": {}}
        assert len(ov) == parts[f"exact|{f}"]["data"]["conf_parents_with_such_a_transition"]
        if not ov:
            continue
        out[f]["overlap_parents"] = ov
        for crit in ("exact", "regret"):
            part = parts[f"{crit}|{f}"]
            z = np.load(OUT / f"confirm_replicate_part_{crit}_{f}.npz")
            pil = json.loads((OUT / f"s_roll{SR.CRIT_TAG[crit]}_{f}.json").read_text(encoding="utf-8"))["contrasts"]
            for name in part["arms"]:
                for k in KS:
                    pids, a = titan_c[fam]["chain_one"][k]
                    b = np.asarray(z[f"{crit}|{f}|{name}|k{k}|per_parent_confirm"], np.float64)
                    full = decide.paired_bootstrap(a, b)
                    stored = part["arms"][name]["contrasts"][f"Titan chain_one - S-roll|k{k}"]["confirm"]
                    assert full["n"] == stored["n"] and abs(full["mean_diff"] - stored["mean_diff"]) < 1e-12, (crit, name, k)
                    keep = np.array([p not in set(ov) for p in pids])
                    c_ = decide.paired_bootstrap(a[keep], b[keep])
                    out[f]["contrasts"][f"{crit}|{name}|k{k}"] = {
                        "confirm_all": full, "confirm_excl_overlap": {**c_, "n_parents_dropped": len(ov)},
                        "pilot_all": part["arms"][name]["contrasts"][f"Titan chain_one - S-roll|k{k}"]["pilot"],
                        "pilot_excl_overlap": pil.get(f"Titan chain_one - S-roll {name}|k{k}|excl_overlap")}
        log(f"F4 {f}: {len(ov)} confirmation parents with a transition input in pilot TRAIN; contrasts recomputed "
            f"without them (stored full-sample contrasts reproduced exactly from the npz)")
    return out


# ============================================================================ F2: contrast status

def sig(ci):
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def classify(p, c):
    ps, cs = sig(p["ci95"]), sig(c["ci95"])
    same = np.sign(p["mean_diff"]) == np.sign(c["mean_diff"])
    if ps and cs:
        return "both significant, same sign" if same else "both significant, SIGN FLIP"
    if not ps and cs:
        return ("pilot n.s. -> confirm significant, same direction as pilot point estimate" if same
                else "pilot n.s. -> confirm significant, opposite to pilot point estimate")
    if ps and not cs:
        return "pilot significant -> confirm n.s."
    return "both n.s."


def contrast_status(parts):
    rows = []
    for key, part in parts.items():
        crit, f = key.split("|")
        for name, arm in part["arms"].items():
            for k in KS:
                cc = arm["contrasts"][f"Titan chain_one - S-roll|k{k}"]
                if not cc["pilot"]:
                    continue
                rows.append({"family": f, "rule": CR.CRIT_NAME[crit], "arm": name, "k": k,
                             "main": name in CR.MAIN_ARMS, "pilot": cc["pilot"], "confirm": cc["confirm"],
                             "status": classify(cc["pilot"], cc["confirm"])})
    counts = collections.Counter((r["main"], r["k"] == 4, r["status"]) for r in rows)
    summ = {f"{'main' if m else 'supplementary'}|{'k4' if k4 else 'k1/k2'}|{s}": n for (m, k4, s), n in sorted(counts.items())}
    return {"rows": rows, "counts": summ, "n_contrasts": len(rows)}


# ============================================================================ markdown

def f3(x):
    return f"{x:.3f}".replace("-", "−")


def fd(x):
    return f"{x:+.3f}".replace("-", "−")


def fci(ci, signed=True):
    g = fd if signed else (lambda v: f"{v:.3f}")
    return f"[{g(ci[0])}, {g(ci[1])}]"


ERR_LABEL = {"one_balance_wrong": "one balance wrong", "two_balances_wrong": "two balances wrong",
             "void_event_applied_as_posted": "void applied as posted", "eq_state_plus_add": "output = state + add",
             "applied_anyway": "applied anyway", "wrong_value": "wrong value", "missing_mod97": "missing mod 97",
             "wrong_branch": "wrong branch", "swap_wrong": "swap wrong", "other_wrong_integer": "other wrong integer",
             "unparseable": "unparseable"}


def dom(cell):
    d = cell["dominant_error"]
    return "—" if d is None else f"{ERR_LABEL.get(d['type'], d['type'])} {d['count']}/{d['of_wrong']}"


def write_doc(R, F, path):
    A, B = R["A_failure_map"], R["B_prerequisite_anatomy"]
    L = []
    L.append("# PREREG-1 confirmation bank: replication of pilot findings R8 and R10")
    L.append("")
    L.append("**Status: post-hoc exploratory replication on the confirmation bank; not preregistered; no tuning on these data.** "
             "Every definition, hyper-parameter, head, selection rule, seed and checkpoint rule comes from the pilot "
             "(`scripts/s_roll.py`, `s_roll_select*_{L,P,R}.json`, RESULTS_PILOT R8/R10). Confirmation data: 150 parents "
             "per family, 8 non-twin candidates each (twins excluded everywhere, as in the decision set).")
    L.append("")
    L.append("Sanity (both pass): Titan chain_one k=4 regret through this path = "
             + " / ".join(f"{k} {R['sanity']['titan_confirm']['chain_one_k4'][k]:.4f}" for k in ("L", "R", "P"))
             + " (confirmatory 0.539 / 0.356 / 0.899; every per-parent value equal to `PREREG_1_RESULTS_PER_PARENT.npz`); "
               "an oracle one-step model through the same rollout gives regret 0 and exact state 1 for every family and k. "
               "Retrained S-roll models are bit-identical to the pilot models (DEV exact, best step, pilot-TEST per-parent regret).")
    L.append("")
    # ---------------------------------------------------------------- (A)
    L.append("## (A) R8: Titan's one-step accuracy by operation (chain_one)")
    L.append("")
    L.append("R8 rule = chain steps whose input state is correct; 1st step = first step from gold s0. Cells: accuracy (n); "
             "confirmation 95% parent-cluster CI. Error = most common type among wrong outputs under the R8 rule (same "
             "typology on both banks), pilot → confirmation.")
    L.append("")
    L.append("| Fam | Operation | R8 rule: pilot | R8 rule: confirm [CI] | 1st step: pilot | 1st step: confirm | Most common error (R8 rule), pilot → confirm |")
    L.append("|---|---|---|---|---|---|---|")
    order = {"L": ["add|posted", "scale|posted", "transfer|posted", "void|any"],
             "P": ["applicable", "inapplicable"],
             "R": ["swap", "add", "add|nowrap", "add|wrap", "conditional", "mix"]}
    lab_name = {"add|posted": "add, posted", "scale|posted": "scale, posted", "transfer|posted": "transfer, posted",
                "void|any": "any void event", "add|nowrap": "add, no wrap", "add|wrap": "add, wrap"}
    for f in ("L", "P", "R"):
        for lab in order[f]:
            pi, ci = A["pilot"][f]["input_correct"][lab], A["confirm"][f]["input_correct"][lab]
            ps, cs = A["pilot"][f]["step1"][lab], A["confirm"][f]["step1"][lab]
            err = f"{dom(pi)} → {dom(ci)}"
            if lab == "add":
                pe, ce = pi["errors"].get("missing_mod97", 0), ci["errors"].get("missing_mod97", 0)
                err += f"; missing mod 97 {pe}/{pi['n_wrong']} → {ce}/{ci['n_wrong']}"
            if lab == "add|wrap":
                wd = {nm: F["F1_R_wrap_direction"][nm]["input_correct"]["cells"] for nm in ("pilot", "confirm")}
                up = {nm: wd[nm].get("add|wrap_up", {}) for nm in wd}
                dn = {nm: wd[nm].get("add|wrap_down", {}) for nm in wd}
                err = (f"upward wrap (r + v ≥ 97): missing mod {up['pilot'].get('err:missing_mod97', 0)}/{up['pilot'].get('n', 0)} → "
                       f"{up['confirm'].get('err:missing_mod97', 0)}/{up['confirm'].get('n', 0)}, of which exactly r + v "
                       f"{up['pilot'].get('out_eq_unreduced_raw', 0)} → {up['confirm'].get('out_eq_unreduced_raw', 0)}; "
                       f"downward wrap (r + v < 0): wrong value {dn['pilot'].get('err:wrong_value', 0)}/{dn['pilot'].get('n', 0)} → "
                       f"{dn['confirm'].get('err:wrong_value', 0)}/{dn['confirm'].get('n', 0)}")
            if lab == "mix":
                err += (f"; missing mod 97 {pi['errors'].get('missing_mod97', 0)}/{pi['n_wrong']} → "
                        f"{ci['errors'].get('missing_mod97', 0)}/{ci['n_wrong']}")
            L.append(f"| {f} | {lab_name.get(lab, lab)} | {pi['acc']:.3f} ({pi['n']}) | {ci['acc']:.3f} ({ci['n']}) "
                     f"{fci(ci['ci95_parent_cluster'], False)} | {ps['acc']:.3f} ({ps['n']}) | {cs['acc']:.3f} ({cs['n']}) | {err} |")
    v1 = F["F3_L_void_first_steps"]
    acc = B["applicable_first_step_accuracy"]
    L.append("")
    L.append(f"Notes. Void first steps, wrong outputs equal to the posted application: {v1['pilot']['wrong_eq_applied_as_posted']}/"
             f"{v1['pilot']['wrong']} → {v1['confirm']['wrong_eq_applied_as_posted']}/{v1['confirm']['wrong']} "
             f"(posted application is a no-op for {v1['pilot']['posted_application_is_noop']} pilot and "
             f"{v1['confirm']['posted_application_is_noop']} confirmation void steps; those are excluded from this count). "
             f"P applicable first steps are higher on confirmation ({acc['pilot']:.3f} → {acc['confirm']:.3f}). Post hoc and "
             f"descriptive only (not a pilot definition): the confirmation bank has more steps where state + add is already the "
             f"gold answer ({acc['cell_share_pilot']['add_bit_clear|delete_clears_nothing']:.0%} → "
             f"{acc['cell_share_confirm']['add_bit_clear|delete_clears_nothing']:.0%}); re-weighted to the pilot cell mix the "
             f"confirmation accuracy is {acc['confirm_reweighted_to_pilot_cell_mix']:.3f}, so the rise is largely consistent "
             f"with case mix, with a smaller within-cell rise ({F['F5_P_shares']['pilot']['cell_add_clear_delete_nothing_acc']:.3f} → "
             f"{F['F5_P_shares']['confirm']['cell_add_clear_delete_nothing_acc']:.3f}, see (B)).")
    L.append("")
    ra_p, ra_c = A["pilot"]["R"]["input_correct"], A["confirm"]["R"]["input_correct"]
    L.append("**Verdict (A): replicates, except the R add error label, which replicates only partially.** Every operation keeps "
             "its pilot accuracy profile (swap ≈ 1, mix ≈ 0, conditional 0.474 → 0.465 with mostly wrong-branch errors, void events and "
             "inapplicable actions applied anyway, posted ledger operations 0.32–0.64 → 0.33–0.58, add fails at every wrap "
             f"{ra_p['add|wrap']['n'] - ra_p['add|wrap']['n_wrong']}/{ra_p['add|wrap']['n']} → "
             f"{ra_c['add|wrap']['n'] - ra_c['add|wrap']['n_wrong']}/{ra_c['add|wrap']['n']}); but R8's \"missing mod 97\" is "
             "not the most common add error on either bank (wrong value "
             f"{ra_p['add']['errors'].get('wrong_value', 0)}/{ra_p['add']['n_wrong']} → {ra_c['add']['errors'].get('wrong_value', 0)}/"
             f"{ra_c['add']['n_wrong']}): the unreduced sum replicates at upward wraps, but downward wraps (rare on the pilot) "
             "give other wrong values and wraps are a smaller share of confirmation add steps "
             f"({ra_p['add|wrap']['n'] / ra_p['add']['n']:.1%} → {ra_c['add|wrap']['n'] / ra_c['add']['n']:.1%}).")
    L.append("")
    # ---------------------------------------------------------------- (B)
    Bp, Bc = B["pilot"]["P"], B["confirm"]["P"]
    Sp, Sc = F["F5_P_shares"]["pilot"], F["F5_P_shares"]["confirm"]
    L.append("## (B) R10: Titan's prerequisite rule is integer addition (applicable first steps, gold s0)")
    L.append("")
    L.append("| Quantity | Pilot | Confirm |")
    L.append("|---|---|---|")
    L.append(f"| Applicable first steps: right / wrong (unparseable) | {Bp['right']} / {Bp['wrong_total']} ({Bp['wrong_unparseable']}) "
             f"| {Bc['right']} / {Bc['wrong_total']} ({Bc['wrong_unparseable']}) |")
    L.append(f"| Parsed wrong outputs = state + add (R10 denominator) | {Sp['wrong_parsed_eq_state_plus_add'][0]}/"
             f"{Sp['wrong_parsed_eq_state_plus_add'][1]} ({Sp['share_of_parsed_wrong']:.0%}) | "
             f"{Sc['wrong_parsed_eq_state_plus_add'][0]}/{Sc['wrong_parsed_eq_state_plus_add'][1]} "
             f"({Sc['share_of_parsed_wrong']:.0%}; CI {Sc['share_of_parsed_wrong_ci95_parent_cluster'][0]:.0%}–"
             f"{Sc['share_of_parsed_wrong_ci95_parent_cluster'][1]:.0%}) |")
    L.append(f"| … add bit already set (unambiguous carry); of those, nothing deleted | {Bp['unambiguous_carry_add_bit_already_set']}; "
             f"{Bp['unambiguous_carry_and_nothing_deleted_gold_is_unchanged_state']} | {Bc['unambiguous_carry_add_bit_already_set']}; "
             f"{Bc['unambiguous_carry_and_nothing_deleted_gold_is_unchanged_state']} |")
    L.append(f"| … add bit clear (consistent with ignoring delete) | {Bp['add_bit_clear_eq_sum_consistent_with_ignoring_delete']} | "
             f"{Bc['add_bit_clear_eq_sum_consistent_with_ignoring_delete']} |")
    g, ng = "titan_right_where_state_plus_add_is_gold", "titan_right_where_state_plus_add_is_not_gold"
    L.append(f"| Titan right where state + add = gold | {Bp[g][0]}/{Bp[g][1]} ({Sp['cell_add_clear_delete_nothing_acc']:.3f}) | "
             f"{Bc[g][0]}/{Bc[g][1]} ({Sc['cell_add_clear_delete_nothing_acc']:.3f}, CI "
             f"{Sc['cell_add_clear_delete_nothing_acc_ci95_parent_cluster'][0]:.3f}–{Sc['cell_add_clear_delete_nothing_acc_ci95_parent_cluster'][1]:.3f}) |")
    L.append(f"| Titan right where state + add ≠ gold | {Bp[ng][0]}/{Bp[ng][1]} | {Bc[ng][0]}/{Bc[ng][1]} |")
    an_p, an_c = B["pilot"]["anatomy"]["P"], B["confirm"]["anatomy"]["P"]
    ip, ic = A["pilot"]["P"]["step1"]["inapplicable"], A["confirm"]["P"]["step1"]["inapplicable"]
    assert ip["n"] == an_p["inapplicable_n"] and ic["n"] == an_c["inapplicable_n"]
    L.append(f"| Inapplicable first steps right; output = state + add | {ip['n'] - ip['n_wrong']}/{ip['n']}; "
             f"{an_p['inapplicable_titan_eq_s_plus_add']} | {ic['n'] - ic['n_wrong']}/{ic['n']}; "
             f"{an_c['inapplicable_titan_eq_s_plus_add']} |")
    L.append("")
    L.append("**Verdict (B): replicates.** Titan is right almost only where integer addition happens to give the gold state "
             f"({Bc[g][0]}/{Bc[g][1]} vs {Bc[ng][0]}/{Bc[ng][1]}; pilot {Bp[g][0]}/{Bp[g][1]} vs {Bp[ng][0]}/{Bp[ng][1]}), and the "
             f"unambiguous-carry errors recur ({Bp['unambiguous_carry_add_bit_already_set']} → {Bc['unambiguous_carry_add_bit_already_set']}).")
    L.append("")
    # ---------------------------------------------------------------- (C)
    C_ = R["C_sroll"]
    L.append("## (C) R10: S-roll vs Titan chain_one (k=4 normalised regret)")
    L.append("")
    L.append("Cells: Titan − S-roll [95% paired parent bootstrap]; negative = Titan better. Pilot n = 26 parents, confirmation n = 150. "
             "spec / DEV-regret are the two pilot selection rules.")
    L.append("")
    L.append("| Fam | Titan pilot → confirm | S-roll arm | spec: pilot | spec: confirm | DEV-regret: pilot | DEV-regret: confirm |")
    L.append("|---|---|---|---|---|---|---|")
    for f in ("L", "P", "R"):
        for name in ("FULL", "MATCHED128", "TITANROWS", "N512", "N2000"):
            cells = []
            for crit in ("exact", "regret"):
                arm = C_["parts"][f"{crit}|{f}"]["arms"].get(name)
                if arm is None:
                    cells += ["—", "—"]
                    continue
                cc = arm["contrasts"]["Titan chain_one - S-roll|k4"]
                cells += [f"{fd(cc['pilot']['mean_diff'])} {fci(cc['pilot']['ci95'])}",
                          f"{fd(cc['confirm']['mean_diff'])} {fci(cc['confirm']['ci95'])}"]
            if all(x == "—" for x in cells):
                continue
            tt = f"{C_['titan_pilot'][f]['4']:.3f} → {C_['titan_confirm'][f]['4']:.3f}" if name == "FULL" else ""
            L.append(f"| {f} | {tt} | {name}{' (suppl.)' if name not in CR.MAIN_ARMS else ''} | " + " | ".join(cells) + " |")
    st = F["F2_contrast_status"]
    main4 = [r for r in st["rows"] if r["main"] and r["k"] == 4]
    flips = [r for r in st["rows"] if "opposite" in r["status"] or "FLIP" in r["status"]]
    ns2s = [r for r in st["rows"] if r["status"].startswith("pilot n.s. -> confirm significant, same")]
    s2ns = [r for r in st["rows"] if r["status"] == "pilot significant -> confirm n.s."]
    ov = F["F4_P_excl_overlap"]["P"]
    ov_same = sum(bool(sig(v["confirm_all"]["ci95"]) == sig(v["confirm_excl_overlap"]["ci95"])
                       and np.sign(v["confirm_all"]["mean_diff"]) == np.sign(v["confirm_excl_overlap"]["mean_diff"]))
                  for v in ov["contrasts"].values())
    L.append("")
    L.append(f"All {st['n_contrasts']} pilot-vs-confirmation contrasts (k = 1, 2, 4; main and supplementary arms): "
             f"{sum(r['status'] == 'both significant, same sign' for r in st['rows'])} significant on both banks with the same sign; "
             f"{len(ns2s)} go from a pilot CI including 0 to a confirmation CI excluding 0 in the direction of the pilot point "
             "estimate ("
             + "; ".join(f"{r['family']} {r['arm']} {r['rule']} k{r['k']} {fd(r['pilot']['mean_diff'])} → {fd(r['confirm']['mean_diff'])}"
                         for r in ns2s)
             + f"); {len(flips)} sign change(s) with a significant confirmation CI ("
             + "; ".join(f"{r['family']} {r['arm']} {r['rule']} k{r['k']} {fd(r['pilot']['mean_diff'])} {fci(r['pilot']['ci95'])} → "
                         f"{fd(r['confirm']['mean_diff'])} {fci(r['confirm']['ci95'])}" for r in flips)
             + f"); {len(s2ns)} lose significance; "
             f"{sum(r['status'] == 'both n.s.' for r in st['rows'])} are n.s. on both. "
             f"P leave-out-overlap (pilot check; {ov['n_overlap_parents']} confirmation parents have a transition whose learner input occurs "
             "in pilot TRAIN): without them, k=4 FULL "
             + f"{fd(ov['contrasts']['exact|FULL|k4']['confirm_excl_overlap']['mean_diff'])} {fci(ov['contrasts']['exact|FULL|k4']['confirm_excl_overlap']['ci95'])}, "
             + f"MATCHED128 {fd(ov['contrasts']['exact|MATCHED128|k4']['confirm_excl_overlap']['mean_diff'])} {fci(ov['contrasts']['exact|MATCHED128|k4']['confirm_excl_overlap']['ci95'])}, "
             + f"TITANROWS {fd(ov['contrasts']['exact|TITANROWS|k4']['confirm_excl_overlap']['mean_diff'])} {fci(ov['contrasts']['exact|TITANROWS|k4']['confirm_excl_overlap']['ci95'])} "
             + f"(spec; n = {ov['contrasts']['exact|FULL|k4']['confirm_excl_overlap']['n']}); {ov_same} of the "
             + f"{len(ov['contrasts'])} P contrasts (both rules, all arms, k = 1, 2, 4) keep the sign and the CI-excludes-0 "
             + "status they have on all 150 parents.")
    L.append("")
    n_main_same = sum(r["status"] == "both significant, same sign" for r in main4)
    L.append(f"**Verdict (C): replicates.** All {len(main4)} main k=4 contrasts (FULL and MATCHED128 × spec and DEV-regret × L, P, R) "
             f"keep the pilot sign with confirmation CIs excluding 0 ({n_main_same}/{len(main4)} significant on both banks): at k=4 "
             "S-roll beats Titan on P at every training budget and on L with all TRAIN transitions, and Titan beats S-roll on R at "
             "every budget and on L at 128 rows; the pilot's two supplementary \"no difference\" readings (P TITANROWS, L N512 "
             "DEV-regret) become same-direction differences favouring S-roll, a gain in precision rather than a reversal.")
    L.append("")
    L.append("Deviations: none forced by the data. Analysis choices are listed in `confirm_replicate.json` (`deviations`) and "
             "`confirm_replicate_fixes.json` (`deviations_added`): twins excluded from (A)/(B) too; pilot (A) values are the "
             "`s_roll.onestep_stats` recomputation of R8 (R8's own L/P counts differ by 1–5 steps); the error typology and "
             "parent-cluster CIs are new; S-roll models retrained (bit-identical to the pilot's); the wrap-direction split and "
             "the case-mix re-weighting are post hoc and descriptive.")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append("- `scripts/confirm_replicate.py`, `scripts/confirm_replicate_fixes.py`")
    L.append("- `runs/bank_pilot/analysis/confirm_replicate.json` (all numbers; `tables_markdown`), "
             "`confirm_replicate_fixes.json` (fixes F1–F5), `confirm_replicate_part_{exact,regret}_{L,P,R}.{json,npz}`, logs")
    L.append("- Fixes after verification: F1 R add error label and wrap-direction split; F2 full contrast-status scan; "
             "F3 L void \"applied as posted\" counted among wrong outputs; F4 P leave-out-overlap check; F5 P share over parsed "
             "wrong outputs (R10 denominator) with CI, re-weighting labelled post hoc.")
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(L) + "\n"
    path.write_text(txt, encoding="utf-8")
    return txt


# ============================================================================ main

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log = CR.Log(OUT / "confirm_replicate_fixes.log")
    log("=" * 100)
    log(f"confirm_replicate_fixes start {time.strftime('%Y-%m-%d %H:%M:%S')}; cuda_available={torch.cuda.is_available()} "
        f"threads={torch.get_num_threads()}")
    t0, cpu0 = time.time(), CR.cpu_sec()
    R = json.loads((OUT / "confirm_replicate.json").read_text(encoding="utf-8"))
    meta_now = CR.code_meta()["code_sha256"]
    parts = R["C_sroll"]["parts"]
    assert R["meta"]["code_sha256"] == meta_now, "confirm_replicate.json was not produced by the code on disk"
    for key in parts:
        crit, f = key.split("|")
        pj = json.loads((OUT / f"confirm_replicate_part_{crit}_{f}.json").read_text(encoding="utf-8"))
        assert pj["meta"]["code_sha256"] == meta_now and pj["arms"].keys() == parts[key]["arms"].keys(), key
    log("stored confirm_replicate.json and all 6 (C) part files were produced by the code on disk (sha256 match)")

    bank_p, rows_p, T_p, cands_p, _ = SR.load_inputs(log)
    P = {"rows": rows_p, "T": T_p, "cands": cands_p, "gen": SR.load_gen()}
    C = CR.load_confirm(log)
    titan_c, _ = CR.sanity_titan_confirm(C, log)
    CR.guard(log, "inputs loaded")

    # re-run (A)/(B) with the original functions: must equal the stored JSON
    for nm, X in (("pilot", P), ("confirm", C)):
        for fam in FAMS:
            fm = CR.jsonable(CR.failure_map(fam, X["gen"], X["cands"][fam]))
            assert fm == R["A_failure_map"][nm][SHORT[fam]], (nm, fam)
        assert CR.jsonable(CR.p_anatomy(X["gen"], X["cands"]["prerequisite_plan"])) == R["B_prerequisite_anatomy"][nm]["P"], nm
    log("re-run (A) failure maps and (B) P anatomy == stored confirm_replicate.json (both banks)")

    F = {"status": "post-hoc exploratory replication on the confirmation bank; not preregistered; no tuning on these data. "
                   "Post-verification fixes F1-F5 (reporting and pilot-defined checks; no number of confirm_replicate.json changes).",
         "F1_R_wrap_direction": {}, "F3_L_void_first_steps": {}, "F5_P_shares": {}}
    for nm, X in (("pilot", P), ("confirm", C)):
        F["F1_R_wrap_direction"][nm] = {rule: r_wrap_detail(X["gen"], X["cands"]["register_machine"], rule)
                                        for rule in ("input_correct", "step1")}
        F["F3_L_void_first_steps"][nm] = l_void_detail(X["gen"], X["cands"]["ledger_replay"])
        F["F5_P_shares"][nm] = p_shares(X["gen"], X["cands"]["prerequisite_plan"])
        # consistency with the stored tables
        cells = F["F1_R_wrap_direction"][nm]["input_correct"]["cells"]
        for kind in ("add", "mix"):
            st = R["A_failure_map"][nm]["R"]["input_correct"]
            assert sum(v["n"] for k, v in cells.items() if k.startswith(kind + "|wrap_")) == st[kind]["n"]
            assert sum(v["n"] for k, v in cells.items() if k in (f"{kind}|wrap_up", f"{kind}|wrap_down")) == st[f"{kind}|wrap"]["n"]
            mm = sum(v.get("err:missing_mod97", 0) for k, v in cells.items() if k.startswith(kind + "|"))
            assert mm == st[kind]["errors"].get("missing_mod97", 0)
        v = F["F3_L_void_first_steps"][nm]
        an = R["B_prerequisite_anatomy"][nm]["anatomy"]["L"]
        assert v["void_first_steps"] == an["void_n"] and v["right"] == an["void_titan_right"]
        assert v["any_eq_applied_as_posted (s_roll.titan_error_anatomy)"] == an["void_titan_eq_applied_as_posted"]
        assert v["wrong_eq_applied_as_posted"] == R["A_failure_map"][nm]["L"]["step1"]["void|any"]["errors"].get("void_event_applied_as_posted", 0)
        log(f"F1 {nm} R wrap cells (R8 rule): " + json.dumps(cells))
        log(f"F1 {nm} R add downward-wrap steps (R8 rule): " + json.dumps(F["F1_R_wrap_direction"][nm]["input_correct"]["add_downward_wrap_steps"]))
        log(f"F3 {nm} L void first steps: {v}")
        log(f"F5 {nm} P shares: {F['F5_P_shares'][nm]}")
    for nm in ("pilot", "confirm"):
        assert F["F5_P_shares"][nm]["cell_add_clear_delete_nothing_right"] == \
            R["B_prerequisite_anatomy"][nm]["P"]["titan_right_where_state_plus_add_is_gold"], nm
    F["F4_P_excl_overlap"] = excl_overlap(parts, titan_c, P, C, log)
    F["F2_contrast_status"] = contrast_status(parts)
    for k_, n_ in F["F2_contrast_status"]["counts"].items():
        log(f"F2 {k_}: {n_}")
    for r in F["F2_contrast_status"]["rows"]:
        if r["status"] != "both significant, same sign":
            log(f"F2   {r['family']} {r['arm']:<10} {r['rule']:<10} k{r['k']}: pilot {r['pilot']['mean_diff']:+.3f} "
                f"{CR.fmt_ci(r['pilot']['ci95'])} -> confirm {r['confirm']['mean_diff']:+.3f} {CR.fmt_ci(r['confirm']['ci95'])}: {r['status']}")
    for key, v in F["F4_P_excl_overlap"]["P"]["contrasts"].items():
        c_, p_ = v["confirm_excl_overlap"], v["pilot_excl_overlap"]
        log(f"F4 P {key:<22} confirm all {v['confirm_all']['mean_diff']:+.3f} -> excl overlap {c_['mean_diff']:+.3f} "
            f"{CR.fmt_ci(c_['ci95'])} (n {c_['n']}) | pilot excl overlap "
            + ("n/a" if not p_ else f"{p_['mean_diff']:+.3f} {CR.fmt_ci(p_['ci95'])} (n {p_['n']})"))
    F["B_case_mix_reweighting"] = {**R["B_prerequisite_anatomy"]["applicable_first_step_accuracy"],
                                   "label": "post hoc, descriptive: prompted by the confirmation result; not a pilot definition"}
    F["deviations_added"] = [
        "F1: wrap-direction split of R add/mix (descriptive; written after seeing the verifier's note on confirmation data).",
        "F5: re-weighting of P applicable accuracy to the pilot cell mix is post hoc and descriptive.",
    ]
    F["meta"] = {**CR.code_meta(), "this_script_sha256": CR.sha(Path(__file__).resolve()),
                 "wall_sec": round(time.time() - t0, 1), "process_cpu_sec": round(CR.cpu_sec() - cpu0, 1),
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    F = CR.jsonable(F)
    (OUT / "confirm_replicate_fixes.json").write_text(json.dumps(F, indent=1) + "\n", encoding="utf-8")
    txt = write_doc(R, F, DOC / "CONFIRM_REPLICATION.md")
    log(f"wrote confirm_replicate_fixes.json and {DOC / 'CONFIRM_REPLICATION.md'} | wall {F['meta']['wall_sec']:.0f}s, "
        f"CPU {F['meta']['process_cpu_sec']:.0f}s")
    print(txt)


if __name__ == "__main__":
    main()
