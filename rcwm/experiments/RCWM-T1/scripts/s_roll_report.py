"""Markdown rendering for scripts/s_roll.py (kept separate so report wording can change without
changing the hash of the computation code that produced the numbers).

Pure formatting: every number comes from the merged results dict that s_roll.py writes to
runs/bank_pilot/analysis/s_roll.json, plus two optional robustness files
(s_roll_deep.json: depth extension; s_roll_verify_stop.json: the fairness verifier's DEV-loss
checkpoint check). Every directional sentence in "What this means" is checked against the numbers by
_claim(): if a rerun changed a sign or a CI verdict, rendering fails instead of printing a stale
sentence (smoke runs print the sentence with an [UNSUPPORTED IN THIS RUN] tag instead).
"""
from __future__ import annotations

import json
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1] / "runs" / "bank_pilot" / "analysis"
DEEP = ANALYSIS / "s_roll_deep.json"
VSTOP = ANALYSIS / "s_roll_verify_stop.json"
LONGF = ANALYSIS / "s_roll_long.json"

FAMS = ("ledger_replay", "prerequisite_plan", "register_machine")
SHORT = {"ledger_replay": "L", "prerequisite_plan": "P", "register_machine": "R"}
KS = (1, 2, 4)
CS = {"exact": "spec", "regret": "DEV-regret"}
ARM_LABEL = {"FULL": "S-roll FULL (all TRAIN transitions)",
             "MATCHED128": "S-roll TITAN-MATCHED (128 random TRAIN rows)",
             "TITANROWS": "S-roll TITAN-ROWS (Titan's own 128 rows)"}
ARM_SHORT = {"FULL": "FULL", "MATCHED128": "TITAN-MATCHED", "TITANROWS": "TITAN-ROWS",
             "MATCHED128_NOSTOP": "TITAN-MATCHED no-stop", "NOOPFREE128": "NOOP-FREE-128",
             "N512": "N512", "N2000": "N2000"}
MAIN_ARMS = ("FULL", "MATCHED128", "TITANROWS")
STRATA = {"ledger_replay": ("add|posted", "scale|posted", "transfer|posted", "add|void", "scale|void", "transfer|void",
                            "void|any", "ALL"),
          "prerequisite_plan": ("applicable", "inapplicable", "ALL"),
          "register_machine": ("swap", "add", "add|nowrap", "add|wrap", "conditional", "conditional|gt",
                               "conditional|le", "mix", "mix|nowrap", "mix|wrap", "ALL")}


# ============================================================================ helpers

def _g(d, k):
    """dict lookup by k that works before and after a JSON round trip (int or str keys)."""
    if d is None:
        return None
    return d.get(k, d.get(str(k)))


def _lab(lab):
    return lab.replace("|", " / ")


def _ci(c, nd=3):
    return f"{c['mean_diff']:+.{nd}f} [{c['ci95'][0]:+.{nd}f}, {c['ci95'][1]:+.{nd}f}]"


def _verdict(c):
    if c["ci95"][1] < 0:
        return "Titan better"
    if c["ci95"][0] > 0:
        return "S-roll better"
    return "CI includes 0"


def _cv(c):
    return "—" if c is None else f"{_ci(c)} ({_verdict(c)})"


def _reg(r, ci=True):
    if r is None:
        return "—"
    s = f"{r['mean']:.3f}"
    if ci and "ci95" in r:
        s += f" [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
    return s


def _acc(d, key, nkey=None):
    if d is None:
        return "—"
    s = f"{d[key]:.3f}"
    return s if nkey is None else f"{s} ({round(d[nkey])})"


class V:
    """Accessors over the merged results."""

    def __init__(self, R):
        self.R = R
        self.fams = [f for f in FAMS if f in R["fam"]]
        self.crits = [c for c in ("exact", "regret") if all(c in R["fam"][f] for f in self.fams)]

    def part(self, fam, crit="exact"):
        return self.R["fam"][fam][crit]

    def reg(self, fam, name, k, crit="exact"):
        return _g(self.part(fam, crit)["regret"].get(name), k)

    def con(self, fam, key, crit="exact"):
        return self.part(fam, crit)["contrasts"].get(key)

    def tc(self, fam, arm, k, crit="exact", mode="chain_one", suffix=""):
        return self.con(fam, f"Titan {mode} - S-roll {arm}|k{k}{suffix}", crit)

    def titan(self, fam, k, mode="chain_one"):
        return self.reg(fam, f"Titan {mode}", k)["mean"]

    def has(self, fam, arm, crit="exact"):
        return arm in self.part(fam, crit)["regret"]

    def os(self, fam, name, rule, lab, crit="exact"):
        return self.part(fam, crit)["onestep"].get(name, {}).get(rule, {}).get(lab)

    def ex(self, fam, name, k, crit="exact"):
        e = _g(self.part(fam, crit)["exact_state"].get(name), k)
        return None if e is None else e["mean"]


STRICT = {"on": True}


def _claim(cond, text):
    if not cond:
        if STRICT["on"]:
            raise AssertionError("report sentence no longer supported by the numbers: " + text)
        return "[UNSUPPORTED IN THIS RUN] " + text
    return text


# ============================================================================ render

def render(R) -> str:
    v = V(R)
    STRICT["on"] = not any(p.get("meta", {}).get("smoke") for d in R["fam"].values() for p in d.values())
    L = []
    w = L.append
    w("# S-roll on the pilot bank (EXPLORATORY): does Titan's chained rollout beat a generic one-step learner "
      "chained the same way?")
    w("")
    np_ = v.tc("prerequisite_plan", "FULL", 4)["n"] if "prerequisite_plan" in v.fams else "—"
    nd_ = (v.part("prerequisite_plan")["data"]["n_test_parents"] - np_) if "prerequisite_plan" in v.fams else "—"
    w("**Status: exploratory, every number.** PREREG-1 sec 6 registered S-roll but did not implement it, so "
      "nothing here is confirmatory. The pilot TEST split was used heavily while this analysis was built: "
      "pre-grid feasibility fits printed TEST accuracy, three adversarial verifiers recomputed TEST contrasts, "
      "and the alternative learners below were added after their TEST numbers were seen. S-roll was not run on "
      "the sealed confirmation bank. TEST parents: " + ", ".join(
          f"{SHORT[f]} {v.part(f)['data']['n_test_parents']}" for f in v.fams)
      + f" (P paired n = {np_}: the frozen regret rule drops {nd_} P parents with a zero denominator). "
      "Generated by `scripts/s_roll.py` (+ `s_roll_report.py` for formatting); numbers are in "
      "`runs/bank_pilot/analysis/s_roll.json`.")
    w("")
    w("- **Regret**: normalised k-step decision regret (0 = oracle, about 1 = random), computed by "
      "`decisions.g_only` itself: TEST goals seed 1001, `decide.regret_table`, a fresh rng 17 per method, "
      "parent-mean fill for unparseable states. S-roll's per-parent regret is the nanmean over its models.")
    w("- **S-roll**: an MLP on `decide.symbolic(fam, state, [event], kmax=1)` trained on gold one-step "
      "transitions of pilot TRAIN parents and chained 4 times on its own discrete predictions, exactly like "
      "Titan's chain. **FULL** = all TRAIN transitions (5 seeds). **TITAN-MATCHED** = 128 TRAIN transitions per "
      "family (equal row count, in-distribution; 5 resamples x 5 seeds). **TITAN-ROWS** = Titan's own 128 training "
      "rows per family (5 seeds).")
    w("- **Two selection rules**, each a complete set of arms; neither sees TEST. *spec*: head and size by DEV "
      "one-step exact accuracy (the S-roll spec). *DEV-regret*: by DEV k=4 regret under the selection goals "
      "(seed 1002), PREREG-1's rule for S.")
    w("- **Differences** are Titan − S-roll: **negative = Titan better**. 95% CIs: `decide.paired_bootstrap` "
      "over TEST parents.")
    w("")
    meaning_section(v, w)
    headline_section(v, w)
    other_k_section(v, w)
    onestep_section(v, w)
    exact_section(v, w)
    leakage_section(v, w)
    sensitivity_section(v, w)
    anatomy_section(v, w)
    curve_section(v, w)
    design_section(v, w)
    checks_section(v, w)
    review_section(v, w)
    w(CAVEATS)
    return "\n".join(L) + "\n"


# ============================================================================ what this means

def meaning_section(v, w):
    """At most 8 sentences. Every directional statement is checked against the numbers (_claim)."""
    Lf, Pf, Rf = "ledger_replay", "prerequisite_plan", "register_machine"
    k = 4
    T = {f: v.titan(f, k) for f in v.fams}
    c = {(a, cr): {f: v.tc(f, a, k, cr) for f in v.fams} for a in MAIN_ARMS + ("N512", "N2000") for cr in v.crits}
    lo = lambda x: x["ci95"][0] > 0          # S-roll better
    hi = lambda x: x["ci95"][1] < 0          # Titan better
    nul = lambda x: x["ci95"][0] <= 0 <= x["ci95"][1]
    E, G = "exact", "regret"
    s = []
    # Each element of s is exactly ONE sentence (the section is capped at 8 sentences).
    # 1. L
    s.append(_claim(lo(c[("FULL", E)][Lf]) and lo(c[("FULL", G)][Lf])
                    and all(hi(c[(a, cr)][Lf]) for a in ("MATCHED128", "TITANROWS") for cr in v.crits),
             f"**L — Titan wins at matched data and loses at full data:** at 128 rows Titan chain_one ({T[Lf]:.3f}) "
             f"is better, Titan − S-roll = {_ci(c[('MATCHED128', E)][Lf])} with random TRAIN rows and "
             f"{_ci(c[('TITANROWS', E)][Lf])} with Titan's own rows (DEV-regret selection: "
             f"{_ci(c[('MATCHED128', G)][Lf], 2)} and {_ci(c[('TITANROWS', G)][Lf], 2)}), but at full data S-roll is "
             f"better, {_ci(c[('FULL', E)][Lf])} with the spec-selected classifier and {_ci(c[('FULL', G)][Lf])} "
             f"with the DEV-regret-selected regression head."))
    # 2. L crossover
    s.append(_claim(nul(c[("N512", G)][Lf]) and lo(c[("N2000", G)][Lf]),
             f"With that regression head Titan's L lead is already gone at 512 TRAIN transitions "
             f"({_ci(c[('N512', G)][Lf])}) and reversed at 2,000 ({_ci(c[('N2000', G)][Lf])})."))
    # 3. P
    t1 = {cr: v.tc(Pf, "TITANROWS", 1, cr) for cr in v.crits}
    s.append(_claim(all(lo(c[(a, cr)][Pf]) for a in ("FULL", "MATCHED128") for cr in v.crits)
                    and all(nul(c[("TITANROWS", cr)][Pf]) and hi(t1[cr]) for cr in v.crits),
             f"**P — Titan never wins at k=4:** S-roll is better at full data (Titan − S-roll = "
             f"{_ci(c[('FULL', E)][Pf])}) and with "
             f"128 random TRAIN rows ({_ci(c[('MATCHED128', E)][Pf])}), and level with Titan ({T[Pf]:.3f}) when "
             f"trained on Titan's own 128 all-applicable rows ({_ci(c[('TITANROWS', E)][Pf])}; Titan is better "
             f"there only at k=1, {_ci(t1[E])})."))
    # 4. why P
    pa = v.part(Pf)["titan"]["error_anatomy_step1"]
    tp = v.os(Pf, "Titan chain_one", "chain_step1", "inapplicable")
    pr4 = _g(v.part(Pf)["titan"]["parse_rate"]["chain_one"], 4)
    s.append(_claim(pa["applicable_wrong"] > pa["applicable_n"] / 2 and tp["acc"] < 0.05,
             f"Titan's one-step P rule is itself wrong: {pa['applicable_wrong']} of {pa['applicable_n']} first "
             f"steps on applicable actions are wrong ({pa['applicable_wrong_eq_s_plus_add']} of them the integer sum "
             f"s + add), it is right on {round(tp['acc'] * tp['n'])} of {tp['n']} inapplicable ones, and only "
             f"{pr4:.0%} of its P rollouts still parse at k=4."))
    # 5. R
    s.append(_claim(all(hi(c[(a, cr)][Rf]) for a in MAIN_ARMS + ("N2000",) for cr in v.crits),
             f"**R — Titan wins at matched and at full data:** Titan chain_one ({T[Rf]:.3f}) is better than S-roll "
             f"FULL (Titan − S-roll = {_ci(c[('FULL', E)][Rf])}) and than 128-row S-roll "
             f"({_ci(c[('MATCHED128', G)][Rf])} with DEV-regret selection; "
             f"{_ci(c[('MATCHED128', E)][Rf], 2)} under the spec rule, whose 128-row checkpoints stop at step 25 and "
             f"nearly copy the input)."))
    # 6. why R
    sw_t = v.os(Rf, "Titan chain_one", "chain_step1", "swap")
    sw_s = v.os(Rf, "FULL", "chain_step1", "swap")
    ad_t, ad_s = v.os(Rf, "Titan chain_one", "chain_step1", "add"), v.os(Rf, "FULL", "chain_step1", "add")
    co_t, co_s = v.os(Rf, "Titan chain_one", "chain_step1", "conditional"), v.os(Rf, "FULL", "chain_step1", "conditional")
    mx_t, mx_s = v.os(Rf, "Titan chain_one", "chain_step1", "mix"), v.os(Rf, "FULL", "chain_step1", "mix")
    s.append(_claim(sw_t["acc"] > 0.95 and sw_s["acc_mean"] < 0.05 and ad_s["acc_mean"] >= ad_t["acc"] - 0.05
                    and co_s["acc_mean"] >= co_t["acc"] and max(mx_t["acc"], mx_s["acc_mean"]) < 0.05,
             f"The R gap is register `swap` (first-step accuracy Titan {sw_t['acc']:.3f}, S-roll FULL "
             f"{sw_s['acc_mean']:.3f}), which an MLP fed numbers as scaled scalars does not learn, while on `add` "
             f"({ad_t['acc']:.2f} vs {ad_s['acc_mean']:.2f}) and `conditional` ({co_t['acc']:.2f} vs "
             f"{co_s['acc_mean']:.2f}) the full-data learner matches or beats Titan and both fail `mix`."))
    # 7. bottom line
    s.append("So, relative to this generic chained learner, the LM adds something beyond \"having a one-step "
             "simulator\" on R (exact copying) and, at Titan's 128-row budget only, on L arithmetic, and nothing on "
             "P.")
    # 8. caveat
    s.append("All of this is exploratory: 26-28 pilot TEST parents per family, TEST seen during development, one "
             "learner family (no copy mechanism, no digit inputs), and S-roll's FULL and TITAN-MATCHED rows come from "
             "TEST's own generator while Titan's 384 rows do not.")
    assert len(s) <= 8
    w("## What this means")
    w("")
    w(" ".join(s))
    w("")


# ============================================================================ headline

def headline_section(v, w):
    w("## Headline: decision regret at k = 4")
    w("")
    w("Per family: regret [95% CI over parents], then the paired difference Titan − method [95% CI] "
      "(negative = Titan better).")
    w("")
    for f in v.fams:
        d = v.part(f)["data"]
        n = v.tc(f, "FULL", 4)["n"]
        w(f"### {SHORT[f]} ({d['n_test_parents']} TEST parents, paired n = {n})")
        w("")
        w("| Method | k=4 regret | Titan chain_one − method | Titan chain_m1 − method |")
        w("|---|---|---|---|")
        for nm in ("oracle", "random (all candidates tied)"):
            w(f"| {nm} | {_reg(v.reg(f, nm, 4), ci=False)} | | |")
        w(f"| **Titan chain_one** | {_reg(v.reg(f, 'Titan chain_one', 4))} | | |")
        w(f"| **Titan chain_m1** | {_reg(v.reg(f, 'Titan chain_m1', 4))} | | |")
        for arm in MAIN_ARMS:
            for crit in v.crits:
                w(f"| {ARM_LABEL[arm]}, {CS[crit]} | {_reg(v.reg(f, arm, 4, crit))} | "
                  f"{_cv(v.tc(f, arm, 4, crit))} | {_cv(v.tc(f, arm, 4, crit, 'chain_m1'))} |")
        w("")


def other_k_section(v, w):
    w("## Other horizons: Titan chain_one − S-roll at k = 1 and k = 2")
    w("")
    w("| k | Titan chain_one regret / S-roll arm | " + " | ".join(SHORT[f] for f in v.fams) + " |")
    w("|---|---|" + "---|" * len(v.fams))
    for k in (1, 2):
        w(f"| {k} | Titan chain_one regret | " + " | ".join(f"{v.titan(f, k):.3f}" for f in v.fams) + " |")
        for arm in MAIN_ARMS:
            for crit in v.crits:
                w(f"| {k} | − S-roll {ARM_SHORT[arm]}, {CS[crit]} | "
                  + " | ".join(f"{v.reg(f, arm, k, crit)['mean']:.3f}; {_ci(v.tc(f, arm, k, crit))}" for f in v.fams) + " |")
    w("")
    w("Cells: S-roll regret; Titan − S-roll [95% CI]. References at k = 4 (not S-roll): Titan end-to-end "
      "generation " + ", ".join(f"{SHORT[f]} {v.reg(f, 'Titan e2e', 4)['mean']:.3f}" for f in v.fams)
      + "; the registered k-step symbolic learner S (PREREG_1_DRYRUN, same TRAIN parents, final states only) "
      + ", ".join(f"{SHORT[f]} {v.reg(f, 'ref: S (k-step symbolic, equal label budget; PREREG_1_DRYRUN)', 4)['mean']:.3f}"
                  for f in v.fams) + ".")
    w("")


# ============================================================================ one-step

def onestep_section(v, w):
    w("## One-step exact accuracy by operation (TEST)")
    w("")
    w("- **S-roll TF**: teacher-forced on every TEST gold one-step transition (mean over models), shown as "
      "spec / DEV-regret selection.")
    w("- **Titan R8**: steps of Titan's chain_one whose input state was correct, judged against the gold next "
      "state (the RESULTS_PILOT R8 rule; n in brackets). **Titan own**: every step with a parsed input, judged "
      "against env.step(Titan's own input), oracle-free. Unparseable output = wrong.")
    w("- **First step**: the first transition of every TEST chain (input = gold s0 for both models), the one "
      "comparison on identical, unselected transitions.")
    w("")
    rg = "regret" in v.crits
    for f in v.fams:
        w(f"### {SHORT[f]}")
        w("")
        w("| Stratum | n gold | S-roll TF FULL | S-roll TF TITAN-MATCHED | S-roll TF TITAN-ROWS | Titan R8 (n) | "
          "Titan own (n) | First step n | First step Titan | First step S-roll FULL |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for lab in STRATA[f]:
            tf = v.os(f, "FULL", "teacher_forced_test", lab)
            if tf is None:
                continue

            def two(arm, rule="teacher_forced_test"):
                a = _acc(v.os(f, arm, rule, lab), "acc_mean")
                b = _acc(v.os(f, arm, rule, lab, "regret"), "acc_mean") if rg else "—"
                return f"{a} / {b}"
            t1 = v.os(f, "Titan chain_one", "chain_step1", lab)
            w(f"| {_lab(lab)} | {int(tf['n_mean'])} | {two('FULL')} | {two('MATCHED128')} | {two('TITANROWS')} | "
              f"{_acc(v.os(f, 'Titan chain_one', 'chain_input_correct', lab), 'acc', 'n')} | "
              f"{_acc(v.os(f, 'Titan chain_one', 'chain_own_input', lab), 'acc', 'n')} | "
              f"{'—' if t1 is None else t1['n']} | {_acc(t1, 'acc')} | {two('FULL', 'chain_step1')} |")
        w("")
        vc = v.part(f)["data"].get("cls_delta_vocab_ceiling") or {}
        if vc:
            fr = vc["test_frac_all_deltas_in_vocab"]
            w(f"Ceiling of the L classification head (cls_delta): {vc['vocab_size']} distinct per-account deltas occur "
              "in TRAIN; the share of TEST transitions whose deltas are all in that vocabulary is "
              + ", ".join(f"{_lab(lab)} {fr[lab]:.3f}" for lab in ("scale|posted", "ALL") if lab in fr)
              + ", so the binding limit on `scale` is exact multiplication by an MLP, not the vocabulary.")
            w("")
    w("Titan's R8 counts reproduce RESULTS_PILOT R8 exactly for R; for L and P this script finds 1-5 more steps "
      "per stratum, all wrong outputs (e.g. P applicable 166 vs 161). The R8 script is not in the repository, so "
      "the residual is unexplained; accuracies agree within 0.01.")
    w("")


def exact_section(v, w):
    w("## Exact-state accuracy of the rollouts at k = 4")
    w("")
    w("Share of TEST candidates whose chained k = 4 state is exactly right (S-roll: mean over models, spec / "
      "DEV-regret selection; unparseable Titan output = wrong).")
    w("")
    w("| Family | Titan chain_one | Titan chain_m1 | S-roll FULL | S-roll TITAN-MATCHED | S-roll TITAN-ROWS |")
    w("|---|---|---|---|---|---|")
    for f in v.fams:
        cells = [f"{v.ex(f, 'Titan chain_one', 4):.3f}", f"{v.ex(f, 'Titan chain_m1', 4):.3f}"]
        for arm in MAIN_ARMS:
            cells.append(" / ".join(f"{v.ex(f, arm, 4, cr):.3f}" for cr in v.crits))
        w(f"| {SHORT[f]} | " + " | ".join(cells) + " |")
    w("")
    w("At k = 1 / 2: " + "; ".join(
        f"{SHORT[f]} Titan chain_one {v.ex(f, 'Titan chain_one', 1):.3f} / {v.ex(f, 'Titan chain_one', 2):.3f}, "
        f"S-roll FULL (spec) {v.ex(f, 'FULL', 1):.3f} / {v.ex(f, 'FULL', 2):.3f}" for f in v.fams)
      + ". Titan parse rate (chain_one, k = 1/2/4): " + "; ".join(
        f"{SHORT[f]} " + "/".join(f"{_g(v.part(f)['titan']['parse_rate']['chain_one'], k):.3f}" for k in KS)
        for f in v.fams) + ". S-roll always emits a valid state.")
    w("")


# ============================================================================ leakage

def leakage_section(v, w):
    w("## Leakage share")
    w("")
    w("TRAIN, DEV and TEST parents are disjoint, TEST parents are never fitted on, and rollouts see only s0 and the "
      "events. \"TEST transition in TRAIN\" means the exact learner input (`decide.symbolic` of state and event) "
      "also occurs as a TRAIN input.")
    w("")
    w("| Family | TEST transitions | TEST transition in TRAIN | TEST parents affected | TEST input state seen as a TRAIN input | TEST s0 seen as a TRAIN s0 | TEST transition in Titan's rows |")
    w("|---|---|---|---|---|---|---|")
    for f in v.fams:
        d = v.part(f)["data"]
        w(f"| {SHORT[f]} | {d['n_test']} | {d['test_inputs_seen_in_train_frac']:.3f} | {len(d['test_overlap_parents'])} | "
          f"{d['test_input_state_seen_as_train_input_frac']:.3f} | {d['test_s0_seen_as_train_s0_frac']:.3f} | "
          f"{d['titan_rows_test_inputs_overlap_frac']:.3f} |")
    w("")
    if "prerequisite_plan" in v.fams:
        f = "prerequisite_plan"
        d = v.part(f)["data"]
        w(f"P has 256 states, so TRAIN nearly enumerates the input states; that is context for the P result, not "
          f"memorisation. {d['test_inputs_seen_in_train_frac']:.3f} of P TEST transitions occur verbatim in TRAIN. "
          f"Without the {len(d['test_overlap_parents'])} TEST parents containing any of them (Titan chain_one − S-roll):")
        w("")
        w("| S-roll arm | selection | k=4, all parents | k=4, without overlap parents | k=1, without overlap parents |")
        w("|---|---|---|---|---|")
        for arm in ("FULL", "MATCHED128", "TITANROWS", "NOOPFREE128"):
            for crit in v.crits:
                a = v.tc(f, arm, 4, crit)
                b = v.tc(f, arm, 4, crit, suffix="|excl_overlap")
                b1 = v.tc(f, arm, 1, crit, suffix="|excl_overlap")
                if a is None or b is None:
                    continue
                w(f"| {ARM_SHORT[arm]} | {CS[crit]} | {_ci(a)} (n={a['n']}) | {_ci(b)} (n={b['n']}) | {_cv(b1)} |")
        w("")


# ============================================================================ sensitivity

def _deep(D, f):
    d = D.get(f)
    if d is None:
        return None
    if not d.get("improved"):
        return f"no DEV gain from depth {d['base']['depth'] + 1}"
    c = d["contrasts"]["Titan chain_one - S-roll FULL-deep|k4"]
    return f"{_g(d['regret'], 4)['mean']:.3f}; {_ci(c)} (depth {d['chosen']['depth']})"


def _long(LG, f):
    d = LG.get(SHORT[f])
    if d is None:
        return None
    c = d["contrasts"]["Titan chain_one - S-roll FULL-long|k4"]
    bs = [m["best_step"] for m in d["models"]]
    return f"{_g(d['regret'], 4)['mean']:.3f}; {_ci(c)} (best step {min(bs)}-{max(bs)})"


def sensitivity_section(v, w):
    D = json.loads(DEEP.read_text(encoding="utf-8")) if DEEP.exists() else {}
    LG = json.loads(LONGF.read_text(encoding="utf-8")) if LONGF.exists() else {}
    VS = json.loads(VSTOP.read_text(encoding="utf-8")) if VSTOP.exists() else {}
    w("## Alternative-learner and analysis sensitivity (k = 4)")
    w("")
    w("Each cell: S-roll k=4 regret; Titan chain_one − S-roll [95% CI].")
    w("")
    w("| Variant | " + " | ".join(SHORT[f] for f in v.fams) + " |")
    w("|---|" + "---|" * len(v.fams))

    def row(label, getter):
        cells = []
        for f in v.fams:
            try:
                x = getter(f)
            except (KeyError, TypeError):
                x = None
            cells.append(x if isinstance(x, str) else ("—" if x is None else f"{x[0]:.3f}; {_ci(x[1])}"))
        w(f"| {label} | " + " | ".join(cells) + " |")

    def arm(a, crit, suffix=""):
        return lambda f: (v.reg(f, a, 4, crit)["mean"], v.tc(f, a, 4, crit, suffix=suffix)) if v.has(f, a, crit) else None

    row("FULL, spec selection (main)", arm("FULL", "exact"))
    if "regret" in v.crits:
        row("FULL, DEV-regret selection (longer schedule)", arm("FULL", "regret"))
    if LG:
        row("FULL, DEV-regret config, schedule doubled to 16,000 steps (`s_roll_long.py`)", lambda f: _long(LG, f))
    row("FULL, spec config + extra depth chosen on DEV (`s_roll_deep.py`)", lambda f: _deep(D, f))
    row("TITAN-MATCHED, spec selection, best-DEV-exact checkpoint (main)", arm("MATCHED128", "exact"))
    row("TITAN-MATCHED, spec config, no early stopping (5 resamples, seed 17)", arm("MATCHED128_NOSTOP", "exact"))
    if VS:
        row("TITAN-MATCHED, spec config, best-DEV-loss checkpoint (`s_roll_verify_stop.py`; 5 resamples, seed 17)",
            lambda f: (VS[SHORT[f]]["summary"]["loss"]["test_regret_k4"]["mean"],
                       VS[SHORT[f]]["summary"]["loss"]["titan_chain_one_minus"]))
    if "regret" in v.crits:
        row("TITAN-MATCHED, DEV-regret selection", arm("MATCHED128", "regret"))
    row("TITAN-ROWS, spec selection", arm("TITANROWS", "exact"))
    if "regret" in v.crits:
        row("TITAN-ROWS, DEV-regret selection", arm("TITANROWS", "regret"))
    row("NOOP-FREE-128 (P only: 128 rows without inapplicable actions), spec", arm("NOOPFREE128", "exact"))
    w("")
    w("Analysis variants:")
    w("")
    w("| Variant | " + " | ".join(SHORT[f] for f in v.fams) + " |")
    w("|---|" + "---|" * len(v.fams))
    w("| Titan chain_one regret with the registered last-parsed fill | "
      + " | ".join(f"{v.reg(f, 'Titan chain_one [last-parsed fill]', 4)['mean']:.3f}" for f in v.fams) + " |")
    for a in ("FULL", "MATCHED128"):
        for crit in v.crits:
            w(f"| Titan (last-parsed fill) − S-roll {ARM_SHORT[a]}, {CS[crit]} | " + " | ".join(
                _ci(v.con(f, f"Titan chain_one [last-parsed fill] - S-roll {a}|k4", crit)) for f in v.fams) + " |")
    for a in ("FULL", "MATCHED128"):
        cells = []
        for f in v.fams:
            c = v.tc(f, a, 4, suffix="|excl_degenerate")
            cells.append("none degenerate" if c is None else
                         f"{_ci(c)} (n={c['n']}; {c['n_degenerate']} degenerate, {c['n_degenerate_kept_by_frozen_rule']} kept by the frozen rule)")
        w(f"| Titan − S-roll {ARM_SHORT[a]} (spec) without parents whose candidates all reach one true φ | " + " | ".join(cells) + " |")
    if "regret" in v.crits:
        for a in MAIN_ARMS:
            cells = []
            for f in v.fams:
                c = v.R["cross"].get(f, {}).get(f"S-roll {a} [regret-sel] - S-roll {a} [exact-sel]|k4")
                cells.append("—" if c is None else _ci(c))
            w(f"| S-roll {ARM_SHORT[a]}: DEV-regret − spec selection (negative = DEV-regret better) | " + " | ".join(cells) + " |")
    w("")
    w("Selected configurations (head, hidden, depth, LR), spec / DEV-regret: " + "; ".join(
        f"{SHORT[f]} {b}: " + " / ".join(
            (lambda c: f"{c['head']}{'' if c['encoding'] == 'sym' else '+' + c['encoding']} {c['hidden']}x{c['depth']} {c['lr']:g}")(
                v.part(f, cr)["selection"][b]["chosen"]) for cr in v.crits)
        for f in v.fams for b in ("FULL", "N128")) + ".")
    w("")
    w("Best checkpoint step of the 128-row arms (min / median / max over models): " + "; ".join(
        f"{SHORT[f]} " + ", ".join(
            f"{ARM_SHORT[a]} {CS[c]} " + "/".join(f"{x:g}" for x in v.part(f, c)['fits'][a]['best_step_min_median_max'])
            for c in v.crits for a in ("MATCHED128", "TITANROWS") if a in v.part(f, c)["fits"])
        for f in v.fams) + ". With 128 rows DEV exact accuracy peaks within the first evaluations (fast "
      "overfitting); the no-stop and DEV-loss rows show what that checkpoint rule costs.")
    w("")


# ============================================================================ Titan anatomy

def anatomy_section(v, w):
    w("## What Titan's first steps get wrong")
    w("")
    if "prerequisite_plan" in v.fams:
        a = v.part("prerequisite_plan")["titan"]["error_anatomy_step1"]
        w(f"- **P, applicable actions** (gold s0 input): Titan is wrong on {a['applicable_wrong']} of "
          f"{a['applicable_n']}; {a['applicable_wrong_eq_s_plus_add']} of the wrong outputs equal the integer sum "
          f"s + add (delete mask ignored, add bits added arithmetically), {a['applicable_wrong_unparsed']} are "
          "unparseable.")
        w(f"- **P, inapplicable actions where ignoring the precondition would still give the gold state** "
          f"({a['inapplicable_where_ignoring_gives_gold_n']}): Titan is right on "
          f"{a['inapplicable_where_ignoring_gives_gold_titan_right']}; "
          f"{a['inapplicable_where_ignoring_gives_gold_titan_eq_s_plus_add']} outputs equal s + add. Titan's P "
          "failures therefore combine the precondition shortcut with a wrong bit-update rule.")
    if "ledger_replay" in v.fams:
        a = v.part("ledger_replay")["titan"]["error_anatomy_step1"]
        nv = v.part("ledger_replay")["data"]["titan_rows_strata"].get("void|any", 0)
        w(f"- **L, void events**: Titan is right on {a['void_titan_right']} of {a['void_n']}; "
          f"{a['void_titan_eq_applied_as_posted']} outputs apply the event as if posted, although {nv} of Titan's "
          "128 L training rows were void.")
    w("")


# ============================================================================ curve

def curve_section(v, w):
    w("## Learning curve (k = 4)")
    w("")
    w("N512 / N2000 reuse the FULL configuration of the same selection rule (3 models: resample i with seed 17+i); "
      "128 = TITAN-MATCHED. Cells: TEST one-step exact; k=4 regret; Titan chain_one − S-roll [95% CI]. Titan "
      "chain_one: " + ", ".join(f"{SHORT[f]} {v.titan(f, 4):.3f}" for f in v.fams) + ".")
    w("")
    w("| Family | n train | " + " | ".join(f"{CS[c]} selection" for c in v.crits) + " |")
    w("|---|---|" + "---|" * len(v.crits))
    for f in v.fams:
        names = sorted(v.part(f)["curve"].keys(), key=lambda n: v.part(f)["curve"][n]["n_train_mean"])
        for nm in names:
            cells = []
            for c in v.crits:
                x = v.part(f, c)["curve"].get(nm)
                cells.append("—" if x is None else
                             f"{x['test_onestep_exact']:.3f}; {x['regret_k4']:.3f}; {_ci(v.tc(f, nm, 4, c))}")
            w(f"| {SHORT[f]} | {v.part(f)['curve'][nm]['n_train_mean']:.0f} | " + " | ".join(cells) + " |")
    w("")


# ============================================================================ design

def design_section(v, w):
    HEADS = v.R["meta"]["heads"]
    w("## Design choices (made on DEV)")
    w("")
    w("- **Outputs always decode to a valid discrete state.** P: 8 sigmoid bits, threshold 0.5. R: 97-way "
      "classification of the value (cls97) or of the residue (next − current) mod 97 (cls_res), a rounded "
      "regression mod 97 (reg_abs, reg_delta), or a gated delta (per-register change logit + regression). L: "
      "classification over the per-account deltas seen in TRAIN (cls_delta), rounded delta regression (reg_delta, "
      "Huber on delta/10), or the gated delta. R at FULL also screens a one-hot register-value input (sym1h).")
    w("- **Grid.** Stage A screens every head at 256 wide x 2 deep, LR 3e-3 (" + "; ".join(
        f"{SHORT[f]}: {', '.join(HEADS[f])}" for f in v.fams) + "); stage B tries (256, 2, 1e-3), (512, 2, 3e-3) "
      "and (256, 3, 3e-3) for the winner. Spec rule: DEV one-step exact (seed 17, resample 0). DEV-regret rule: "
      "DEV k=4 regret under selection goals 1002 (FULL: seed 17; 128 rows: mean of resamples 0-2), with a longer "
      "schedule for n > 1000 (≤ 8,000 steps, patience 20) because regression heads were still improving at the "
      "4,000-step cap.")
    w("- **Optimiser.** AdamW (wd 1e-4), cosine decay to 5%, checkpoint = best DEV one-step exact under both "
      "rules. n > 1000: batch 256, ≤ 4,000 steps (spec), eval every 100, patience 12; n = 512: ≤ 3,000 steps; "
      "n = 128: full batch, ≤ 2,000 steps, eval every 25, patience 15. The 128-row arms also use pilot DEV "
      "(896-1,312 labelled transitions) for checkpointing, so their label budget is not strictly 128.")
    w("- **Not TEST-blind.** Nine pre-grid feasibility fits (R 7, L 2) printed TEST one-step accuracy; they led "
      "to the cls_res and gated_delta heads, the /10 Huber L regression and the sym1h screen. The DEV-regret "
      "rule and the extra arms were added after the verifiers reported TEST numbers. Within each rule every "
      "choice is made on DEV alone.")
    w("")
    w("Stage-A DEV scores at FULL (spec: DEV exact; DEV-regret: DEV exact / DEV k=4 regret):")
    w("")
    for f in v.fams:
        parts = []
        for crit in v.crits:
            g = v.part(f, crit)["selection"]["FULL"]["grid"]
            items = [r["head"] + ("" if r["encoding"] == "sym" else "+" + r["encoding"]) + f" {r['dev_exact']:.3f}"
                     + (f"/{r['dev_regret_k4']:.3f}" if "dev_regret_k4" in r else "") for r in g if r["stage"] == "A"]
            parts.append(f"{CS[crit]}: " + ", ".join(items))
        w(f"- {SHORT[f]}: " + "; ".join(parts) + ".")
    w("")


def checks_section(v, w):
    w("## Data and checks")
    w("")
    w("| Family | TRAIN transitions raw / dedup (used) | DEV | TEST | Titan's rows by stratum |")
    w("|---|---|---|---|---|")
    for f in v.fams:
        d = v.part(f)["data"]
        keep = [k for k in d["titan_rows_strata"] if "|" in k or k in ("applicable", "inapplicable", "swap")]
        if f == "register_machine":
            keep = [k for k in d["titan_rows_strata"] if "|" not in k]
        w(f"| {SHORT[f]} | {d['n_train_raw']} / {d['n_train_dedup']} | {d['n_dev']} | {d['n_test']} | "
          + ", ".join(f"{_lab(k)} {d['titan_rows_strata'][k]}" for k in keep if k != "void|any") + " |")
    w("")
    mx = max(max(x["absdiff"] for x in v.part(f)["sanity"]["titan_vs_dryrun"].values()) for f in v.fams)
    w("- An oracle one-step model run through the same rollout → φ → regret path gives regret 0.000 and exact "
      "state 1.000 for every family and k (asserted in the run). Titan regrets recomputed here (parent-mean and "
      f"last-parsed fill) match PREREG_1_DRYRUN to max |diff| {mx:.1e}.")
    w("- All 19,200 gold one-step transitions match the independent FDS oracle (`env.cross_check`), and every "
      "chained gold state and its φ equal the bank's k = 1/2/4 gold. Titan's 384 training rows (RRP-015 "
      "`control-prepared/microfit-train.jsonl`, sha `0881c027…`: the rows the RRP-015 control was fine-tuned on; "
      "Titan-Sparse keeps the top 25% per tensor of that control's weight delta) were re-derived with env.step: "
      "every reference matches.")
    m = v.R["meta"]
    stale = {f: {c: x for c, x in d.items() if x} for f, d in m.get("code_changed_since_family_run", {}).items()}
    stale = {f: d for f, d in stale.items() if d}
    w(f"- Provenance: every per-family output carries the sha256 of `s_roll.py` and `rcwmt/*.py`; code changed "
      f"since a family run: {stale if stale else 'none'}. Reruns with the same code reproduce every number "
      "(deterministic CPU training, fixed seeds).")
    w("- Rerun end to end: `python scripts/s_roll_chain.py` (both selection rules, L/P/R, depth and "
      "schedule-doubling checks, then this report; restartable, one stage per process, each < 25 min on 2 "
      "threads, about 2 h in total). The DEV-loss-checkpoint row comes from `python scripts/s_roll_verify_stop.py`. "
      "Re-render this report only: `python scripts/s_roll_render.py` (`s_roll.py --report-only` also re-runs "
      "the deterministic depth check, about 6 min, a quirk left in place so the stored code hashes stay valid).")
    w("")


def review_section(v, w):
    w("## Changes after adversarial review")
    w("")
    w("- **DEV-regret selection** added as a complete second set of arms (fairness verifier, must-fix), with a "
      "longer schedule for n > 1,000. It picks a gated-delta regression head for L at both budgets (L full-data "
      "margin +0.20 → +0.46, so the old \"S-roll beats Titan on L is learner-dependent\" caveat is withdrawn) and "
      "keeps the residue classifier for R, at 128 rows with a higher LR whose checkpoints are no longer all at "
      "step 25 (R 128-row gap −0.61 → −0.39). Doubling the schedule again (`s_roll_long.py`) changes no conclusion.")
    w("- **TITAN-ROWS** added: S-roll trained on Titan's actual 128 rows per family. The verifier proposed "
      "`CANDIDATE_VIEWS.jsonl`; that file is the historical probe, which RRP-015's DATA_CONTRACT lists as a "
      "scored instrument verified disjoint from the training rows, so the sha-pinned training file is used. TITAN-MATCHED is relabelled \"equal row "
      "count, in-distribution\". The earlier statement that Titan's rows had no no-op cases was wrong for L "
      "(17 of 128 are void events); it holds for P (128 of 128 applicable), so NOOP-FREE-128 is run for P only.")
    w("- **TITAN-MATCHED without early stopping** added and the step-25 checkpoints reported (every spec-rule "
      "128-row R model stops at the first evaluation).")
    w("- **Titan parse rates and last-parsed-fill contrasts** added (the old caveat that Titan had \"at most a few "
      "unparseable outputs\" was wrong for P), plus P contrasts without the 10 overlap parents, contrasts "
      "without degenerate parents, the P near-enumeration context, Titan's P error anatomy (the old reading "
      "\"precondition shortcut, not a missing simulator\" is withdrawn) and the L vocabulary ceiling.")
    w("- The DEV-loss-checkpoint row is the fairness verifier's own run (`s_roll_verify_stop.py`); its spec-rule "
      "rows reproduce this run's per-model regrets exactly.")
    w("")


CAVEATS = """## Caveats

- **Exploratory, pilot bank only**, 26-28 TEST parents per family, TEST used repeatedly during development.
  Bootstrap CIs resample parents; they do not include learner-choice variability, which the sensitivity
  table shows is material for L at full data and R at 128 rows.
- **S-roll has a distribution advantage.** FULL and TITAN-MATCHED train on transitions from the pilot
  generator that also produced TEST, including void events and inapplicable actions. Titan's 384 rows came
  from the earlier Forge v2 distribution. TITAN-ROWS removes that advantage for the rows but keeps S-roll's
  access to pilot DEV for selection and checkpointing.
- **The budgets are not exactly matched.** Titan brings pre-training (number formats, copying, arithmetic);
  S-roll has only its rows plus DEV labels for checkpointing. FULL uses every intermediate transition of the
  pilot TRAIN chains, about 15-35 times Titan's 128 rows per family.
- **One generic learner family.** `decide.symbolic` feeds numbers as scaled scalars, so an MLP must reproduce
  products (L scale), copies (R swap) and modular sums (R mix) to integer precision. A learner with a copy
  mechanism or digit inputs would probably close most of the R gap; that was not tested.
- **The R capacity search did not converge.** With the spec configuration, DEV exact kept rising up to the
  depth-6 cap (0.29 → 0.41); that model gets R `add` 0.998 and `conditional` 0.897 teacher-forced but
  `swap` 0.029, and its k=4 regret (0.568) is still behind Titan. A larger MLP might narrow the R gap a little;
  removing it probably needs a learner that can copy values (untested).
- **Frozen float32 regret rule.** Some P parents have candidates that all reach one true φ; float32 rounding
  leaves a denominator of about 1e-6, so the frozen rule keeps some of them (each scores 0 for every method).
  The sensitivity table drops them without changing a conclusion. This also affects the PREREG-1 P analysis;
  the frozen code was not changed.
- **One-step exactness and regret do not map one to one.** A regression head that is approximately right on
  ledger `scale` has lower decision regret than a classifier that is exactly right more often but far off on
  `scale`; that is why the two selection rules differ on L.
"""
