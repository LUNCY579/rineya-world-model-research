"""RCX-E01N analysis: compare native arms (OFF, T2, SUPA from probe1; N-* cells) on the same DEV battery.

Usage (from complementary/):
  python -B experiments/RCX-E01N/analyze_native.py --probe runs/RCX-E01N/probe1 --cells runs/RCX-E01N/cells \
      --out runs/RCX-E01N/ANALYSIS_DEV.md
All arms must have been scored on the identical battery (checked by sha256). Descriptive unless a frozen
preregistration names the comparison. Unit of independence: the semantic parent.

SEALED confirmation (PREREG_E01N; run once, after open-sealed and every SEALED eval):
  python -B experiments/RCX-E01N/analyze_native.py --battery sealed --cells runs/RCX-E01N/cells
writes runs/RCX-E01N/ANALYSIS_SEALED.md and ANALYSIS_SEALED.json (create-only). The battery is rebuilt with
native_e01.sealed_battery (needs SEALED_OPEN.json); every eval_sealed_<tag>.json must name that battery's sha256.
C1/C2, Holm, the protected strata and the decision are computed mechanically by confirmatory(), protected() and
decide() below; everything else in the report is descriptive.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcx import stats as ST  # noqa: E402

COMPOSITE = ("F2_unseen_real", "F3_unseen_nonce", "F5_postedlike", "V1_pos", "V2_neg", "V3_inverted",
             "V5_posted_override")
VARIED = ("V1_pos", "V2_neg", "V3_inverted", "V5_posted_override", "V6_compose", "V7_bigset")
STRATA = ("F1_void", "F2_unseen_real", "F3_unseen_nonce", "F4_nearmiss", "F5_postedlike", "V1_pos", "V2_neg",
          "V3_inverted", "V5_posted_override", "V6_compose", "V7_bigset", "M1_bal", "M2_amt")
CONTRASTS = [("N-CFSD", "N-VR", "PRIMARY: counterfactual/SD vs ordinary rule-varied data"),
             ("N-VR", "N-FX", "rule-varied vs fixed-rule data (matched)"),
             ("N-CFSD", "T2", "RCX treatment vs registered T2"),
             ("N-VR", "OFF", "training effect (C - A), ordinary"),
             ("N-CFSD", "OFF", "training effect (C - A), counterfactual"),
             ("N-FX", "T2", "RCX fixed-rule recipe vs T2 (sanity)")]


def load_preds(path):
    return [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]


def load_arms(probe, cells):
    battery = [json.loads(l) for l in open(Path(probe) / "battery.jsonl", encoding="utf-8")]
    bsha = hashlib.sha256((Path(probe) / "battery.jsonl").read_bytes()).hexdigest()
    arms = {}
    for arm in ("OFF", "T2", "SUPA"):
        p = Path(probe) / f"preds_{arm}.jsonl.gz"
        if p.exists():
            arms[arm] = [x for x in load_preds(p) if x.get("frame") != "verbatim"]
    for p in sorted(glob.glob(os.path.join(cells, "preds_N-*_s*.jsonl.gz"))):
        tag = os.path.basename(p)[len("preds_"):-len(".jsonl.gz")]
        ev = json.loads((Path(cells) / f"eval_{tag}.json").read_text())
        if ev["battery_sha256"] != bsha:
            raise SystemExit(f"{tag}: battery sha mismatch ({ev['battery_sha256'][:12]} vs {bsha[:12]})")
        arms[tag] = load_preds(p)
    for a, P in arms.items():
        assert len(P) == len(battery), (a, len(P), len(battery))
    return battery, bsha, arms


def cell(battery, P, stratum, surface, field="exact"):
    by = defaultdict(list)
    for r, p in zip(battery, P):
        if r["group"] == stratum and r["surface"] == surface:
            by[r["parent_id"]].append(p[field])
    vals = [sum(v) / len(v) for v in by.values()]
    return sum(vals) / len(vals) if vals else None


def parents(battery, P, groups, surface, field="exact"):
    by = defaultdict(list)
    for r, p in zip(battery, P):
        if r["group"] in groups and r["surface"] == surface:
            by[r["parent_id"]].append(p[field])
    return dict(by)


def noop_rate(battery, P, surface):
    k = n = 0
    for r, p in zip(battery, P):
        if r["surface"] == surface and r["group"] in VARIED and r["gold"]["decision"] == "noop":
            n += 1
            k += p["intent"] == "noop"
    return k, n


def rule_sensitivity(battery, P, surface):
    by = defaultdict(dict)
    for r, p in zip(battery, P):
        if r["group"] == "V3_inverted" and r["surface"] == surface:
            by[r["parent_id"]][r["variant"]] = (p["intent"], p["exact"])
    sens = sum(v["pos"][0] != v["neg"][0] for v in by.values()) / len(by)
    both = sum(v["pos"][1] and v["neg"][1] for v in by.values()) / len(by)
    return sens, both


def verdict(battery, P, surface):
    t = defaultdict(lambda: [0, 0])
    for r, p in zip(battery, P):
        if r["surface"] == surface and r["group"].startswith("F"):
            t[(r["group"], r["variant"])][0] += p["intent"] == "noop" if r["variant"] == "probe" else p["exact"]
            t[(r["group"], r["variant"])][1] += 1
    acc = {k: a / n for k, (a, n) in t.items()}
    unseen = [acc.get((g, "probe")) for g in ("F2_unseen_real", "F3_unseen_nonce", "F5_postedlike")]
    void = acc.get(("F1_void", "probe"))
    near = acc.get(("F4_nearmiss", "probe"))
    if None in unseen or void is None:
        return "NA", acc
    if all(u <= 0.3 for u in unseen) and void <= 0.3:
        v = "never no-op (ignores the rule)"
    elif all(u >= 0.8 for u in unseen):
        v = "generalises to unseen statuses (RULE-like)"
    elif all(u <= 0.3 for u in unseen) and void >= 0.8:
        v = "TRIGGER" + (" (trigger widened: near-misses)" if near is not None and near >= 0.8 else "")
    else:
        v = "partial"
    return v, acc


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


# ============================================================================ SEALED confirmation (PREREG_E01N)
# Used only with --battery sealed; the DEV path above is unchanged. The constants are PREREG_E01N's values.

CONFIRMATORY_CELLS = ("N-VR_s0", "N-VR_s1", "N-VR_s2", "N-CFSD_s0", "N-CFSD_s1", "N-CFSD_s2")  # = SEALED_COMMIT
#                                       arms_required (checked); a missing one forces INCONCLUSIVE
REFERENCE_CELLS = ("OFF", "T2")         # descriptive references only; a missing one never changes the decision
SEALED_REQUIRED = CONFIRMATORY_CELLS + REFERENCE_CELLS   # all needed unless --allow-missing
SEALED_OPTIONAL = ("N-FX_s0",)          # descriptive fixed-rule control; never in a test
CONTROL, TREATMENT = "N-VR", "N-CFSD"   # every confirmatory and protected contrast is TREATMENT - CONTROL
ENDPOINTS = (("C1", "in"), ("C2", "ho_tpl"))
SECONDARY_CONTRASTS = ((TREATMENT, "T2"), (CONTROL, "OFF"))   # (b, a): b - a, descriptive
RESAMPLES_SEALED = 10000
BOOT_SEED_SEALED = 20261236
P_FLOOR = 1.0 / 10000
ALPHA = 0.05            # Holm-adjusted p < ALPHA
PRACTICAL = 0.10        # ADVANCE: point estimate >= +0.10 (REVISE_RULE "literal": neither endpoint reaches it)
OTHER_LB = -0.05        # ADVANCE: the other endpoint's 95% CI lower bound > -0.05
STOP_UB = 0.05          # STOP: both endpoints' 95% CI upper bounds < +0.05
F0_MARGIN = -0.05       # protected: F0 posted twins (exact, surface in), N-CFSD - N-VR
FORMAT_RISE_MAX = 0.02  # protected: format-error rate (all records), N-CFSD - N-VR rises by at most this
PROTECTED_BASIS = "ci"  # "ci": F0 95% CI lower bound > F0_MARGIN and format 95% CI upper bound <= FORMAT_RISE_MAX;
#                         "point": the same thresholds applied to the point estimates
DROP_TRAIN_OVERLAP = True  # parents listed in SEALED_OPEN.json train_overlap.parent_ids are dropped from every arm
P_TIES = "both"         # "both": p = 2 min(share <= 0, share >= 0) (replicates equal to 0 count on both sides);
#                         "low": on the <= 0 side only (an identically-0 contrast then gets the floor, i.e. "significant");
#                         "split": half on each side. See rcx.stats.bootstrap_p_two_sided.
REVISE_RULE = "not_advance"  # "not_advance": REVISE whenever an endpoint is Holm-significant and positive and no earlier
#                              outcome holds (incl. an endpoint >= +0.10 whose ADVANCE guards fail);
#                              "literal": the pre-review text (neither endpoint's point estimate reaches +0.10)
DECISION_ORDER = ("ADVANCE", "STOP", "REVISE")  # the first outcome that holds is the decision; else INCONCLUSIVE
ADVANCE_PER_SEED = True  # ADVANCE also needs every matched per-seed contrast (N-CFSD_sk - N-VR_sk) of the qualifying
#                          endpoint > 0
SENSITIVITY_SEEDS = (1, 2)  # descriptive: C1/C2 without the DEV-gated seed 0
T975 = {1: 12.706204736174698, 2: 4.302652729911275, 3: 3.182446305284263}  # Student t 0.975 quantile by df
EPS = 1e-9              # threshold tolerance (all compared values are rationals at least ~1e-4 apart)
F_STRATA = tuple(s for s in STRATA if s.startswith("F"))
SCORE_KEYS = ("exact", "intent", "decision_ok", "err")
ERR_KINDS = ("applied_when_noop", "false_copy", "arith", "format")


def _ge(x, t):
    return x >= t - EPS


def _gt(x, t):
    return x > t + EPS


def _lt(x, t):
    return x < t - EPS


def _le(x, t):
    return x <= t + EPS


def battery_sha256(battery):
    """sha256 of the battery serialised as native_e01.cmd_eval hashes it (= native_probe's battery.jsonl bytes)."""
    return hashlib.sha256("".join(json.dumps(r) + "\n" for r in battery).encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native_e01_module():
    """native_e01, imported lazily (it imports torch; nothing here uses it). CPU only, one torch thread."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("RCX_THREADS", "1")
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import native_e01
    return native_e01


def split_tag(tag):
    """'N-VR_s1' -> ('N-VR', 1); 'OFF' -> ('OFF', None)."""
    if tag.startswith("N-"):
        arm, _, s = tag.rpartition("_s")
        return arm, int(s)
    return tag, None


def tags_of(arms, arm):
    """The loaded cells of an arm: its seed tags for N-* arms (sorted), else [arm] if present."""
    if arm.startswith("N-"):
        return sorted(t for t in arms if t.startswith("N-") and split_tag(t)[0] == arm)
    return [arm] if arm in arms else []


def rescore(ne, r, p):
    return ne.S.classify(r, ne.NP.parse_answer(p["out"], list(r["gold"]["after"].keys())))


def load_sealed(cells, allow_missing=False, ne=None):
    """Rebuild the SEALED battery (native_e01.sealed_battery: needs SEALED_OPEN.json) and load every SEALED cell
    (eval_sealed_<tag>.json + preds_sealed_<tag>.jsonl.gz) of SEALED_REQUIRED + SEALED_OPTIONAL.

    SEALED_COMMIT.json arms_required must equal CONFIRMATORY_CELLS and open-sealed must have recorded each one's
    adapter sha. A present cell must name this battery (battery 'sealed', battery_sha256, battery_seed, n) and its
    tag's arm/seed; an N-* cell's adapter must have the sha open-sealed recorded (and the one its eval recorded); its
    preds must be aligned with the battery (i = 0..n-1) and every stored score must equal a re-score of the raw output
    against the rebuilt gold. Any violation raises SystemExit. A missing required cell raises unless allow_missing
    (PREREG section 6: such a cell is reported as missing and never replaced; a missing CONFIRMATORY cell makes the
    decision INCONCLUSIVE, a missing reference cell only removes its descriptives). Parents in SEALED_OPEN.json
    train_overlap.parent_ids are then dropped from the battery and every arm (DROP_TRAIN_OVERLAP).
    Returns (battery, arms {tag: preds}, meta)."""
    ne = ne or native_e01_module()
    cells = Path(cells)
    commit_p, _, open_p = ne._sealed_paths(cells)
    if not commit_p.exists():
        raise SystemExit(f"no SEALED_COMMIT.json in {cells}")
    commit = json.loads(commit_p.read_text(encoding="utf-8"))
    if sorted(commit.get("arms_required", [])) != sorted(CONFIRMATORY_CELLS):
        raise SystemExit(f"SEALED_COMMIT.json arms_required {commit.get('arms_required')} differ from PREREG_E01N's "
                         f"confirmatory cells {list(CONFIRMATORY_CELLS)}")
    battery, bseed = ne.sealed_battery(cells, commit["surfaces"])
    opened = json.loads(open_p.read_text(encoding="utf-8"))
    recorded = opened.get("adapters_sha256", {})
    bsha = battery_sha256(battery)
    if opened.get("battery_sha256") not in (None, bsha):
        raise SystemExit("rebuilt SEALED battery sha differs from SEALED_OPEN.json battery_sha256")
    warnings = []
    unrecorded = [t for t in CONFIRMATORY_CELLS if t not in recorded]
    if unrecorded:      # an adapter not fixed at opening must never enter C1/C2
        raise SystemExit(f"open-sealed recorded no adapter sha for confirmatory cells {unrecorded}")
    arms, missing, info = {}, [], {}
    for tag in SEALED_REQUIRED + SEALED_OPTIONAL:
        ep, pp = cells / f"eval_sealed_{tag}.json", cells / f"preds_sealed_{tag}.jsonl.gz"
        if not (ep.exists() and pp.exists()):
            if tag in SEALED_REQUIRED:
                missing.append(tag)
            continue
        ev = json.loads(ep.read_text(encoding="utf-8"))
        arm, seed = split_tag(tag)
        bad = []
        if ev.get("battery") != "sealed":
            bad.append(f"battery {ev.get('battery')!r} is not 'sealed'")
        if ev.get("battery_sha256") != bsha:
            bad.append(f"battery sha {str(ev.get('battery_sha256'))[:12]}... != rebuilt SEALED battery {bsha[:12]}...")
        if ev.get("battery_seed") != bseed or ev.get("n") != len(battery):
            bad.append("battery seed / size differ from the rebuilt SEALED battery")
        if ev.get("arm") != arm or (seed is not None and ev.get("seed") != seed):
            bad.append(f"eval is for arm {ev.get('arm')} seed {ev.get('seed')}")
        adapter = None
        if seed is not None:
            ap_ = cells / tag / "adapter" / "adapter_model.safetensors"
            adapter = sha_file(ap_) if ap_.exists() else None
            want = recorded.get(tag)
            if adapter is None:
                bad.append("adapter file missing (cannot verify which adapter was evaluated)")
            elif want is not None and adapter != want:
                bad.append(f"adapter sha {adapter[:12]}... != {want[:12]}... recorded by open-sealed")
            if adapter is not None and ev.get("adapter_sha256") not in (None, adapter):
                bad.append("eval recorded a different adapter sha")
        P = load_preds(pp)
        if len(P) != len(battery) or any(p.get("i") != i for i, p in enumerate(P)):
            bad.append("preds are not aligned with the battery (i = 0..n-1)")
        else:
            n_bad = sum({k: p.get(k) for k in SCORE_KEYS} != rescore(ne, r, p) for r, p in zip(battery, P))
            if n_bad:
                bad.append(f"{n_bad} stored scores differ from a re-score of the raw outputs")
        if bad:
            raise SystemExit(f"SEALED cell {tag}: " + "; ".join(bad))
        arms[tag] = P
        info[tag] = {"eval_sha256": sha_file(ep), "preds_sha256": sha_file(pp), "adapter_sha256": adapter,
                     "adapter_sha256_at_open": recorded.get(tag), "required": tag in SEALED_REQUIRED}
    if missing and not allow_missing:
        raise SystemExit(f"required SEALED cells missing: {missing}. PREREG_E01N analyses once, after every SEALED "
                         "eval; pass --allow-missing only for cells that cannot be completed (reported as missing)")
    n_rebuilt = len(battery)
    dropped = sorted(opened.get("train_overlap", {}).get("parent_ids", [])) if DROP_TRAIN_OVERLAP else []
    if dropped:
        drop = set(dropped)
        keep = [i for i, r in enumerate(battery) if r["parent_id"] not in drop]
        battery = [battery[i] for i in keep]
        arms = {t: [P[i] for i in keep] for t, P in arms.items()}
    meta = {"battery_sha256": bsha, "battery_seed": bseed, "n_records_rebuilt": n_rebuilt,
            "surfaces": commit["surfaces"], "n_parents_per_stratum": commit.get("n_parents"),
            "commit_sha256_salt": commit.get("sha256_salt"), "sealed_commit_file_sha256": sha_file(commit_p),
            "sealed_open_file_sha256": sha_file(open_p), "minted_at": commit.get("minted_at"),
            "opened_at": opened.get("opened_at"), "cells": info, "missing": missing,
            "missing_confirmatory": [t for t in missing if t in CONFIRMATORY_CELLS],
            "dropped_train_overlap_parents": dropped, "warnings": warnings}
    return battery, arms, meta


def parents_where(battery, P, keep, value):
    """parent_id -> [value(record, pred) for the parent's records kept by keep(record)]."""
    by = defaultdict(list)
    for r, p in zip(battery, P):
        if keep(r):
            by[r["parent_id"]].append(value(r, p))
    return dict(by)


def _exact(r, p):
    return p["exact"]


def _is_format(r, p):
    return int(p["intent"] == "format")


def _composite_on(surface):
    return lambda r: r["group"] in COMPOSITE and r["surface"] == surface


def _f0_twin(r):
    return r["group"] in F_STRATA and r["variant"] == "twin_posted" and r["surface"] == "in"


def _every(r):
    return True


def contrast(battery, arms, a_tags, b_tags, keep, value=_exact, resamples=RESAMPLES_SEALED, seed=BOOT_SEED_SEALED):
    """Seed x parent paired cluster bootstrap of mean(b) - mean(a): variants averaged within each parent per seed,
    seeds averaged, parents resampled; with the two-sided bootstrap p floored at P_FLOOR."""
    pa = [parents_where(battery, arms[t], keep, value) for t in a_tags]
    pb = [parents_where(battery, arms[t], keep, value) for t in b_tags]
    return ST.seed_parent_bootstrap_test(pa, pb, resamples=resamples, seed=seed, p_floor=P_FLOOR, ties=P_TIES)


def _pair(arms):
    a, b = tags_of(arms, CONTROL), tags_of(arms, TREATMENT)
    if not a or not b:
        raise SystemExit(f"C1/C2 need at least one {CONTROL} and one {TREATMENT} SEALED cell")
    return a, b


def matched_seed_arms(arms):
    """arms without the CONTROL / TREATMENT cells whose seed the other arm lacks (used only when a confirmatory cell
    is missing: C1/C2 are then descriptive, on the seed indices both arms have)."""
    both = ({split_tag(t)[1] for t in tags_of(arms, CONTROL)} & {split_tag(t)[1] for t in tags_of(arms, TREATMENT)})
    return {t: P for t, P in arms.items() if split_tag(t)[0] not in (CONTROL, TREATMENT) or split_tag(t)[1] in both}


def per_seed_contrasts(battery, arms, keep, value=_exact):
    """{'s<k>': parent macro of TREATMENT_sk - parent macro of CONTROL_sk} over the records kept by keep, for every
    seed both arms have (the C1/C2 estimand for one matched seed pair)."""
    out = {}
    for t in tags_of(arms, TREATMENT):
        s = split_tag(t)[1]
        c = f"{CONTROL}_s{s}"
        if c in arms:
            out[f"s{s}"] = (ST.parent_macro(parents_where(battery, arms[t], keep, value))
                            - ST.parent_macro(parents_where(battery, arms[c], keep, value)))
    return out


def seed_t_interval(values):
    """Descriptive 95% t interval (df = k - 1) over k per-seed contrasts; None if k < 2."""
    xs = list(values)
    if len(xs) < 2 or len(xs) - 1 not in T975:
        return None
    m, sd = _mean(xs), _sd(xs)
    half = T975[len(xs) - 1] * sd / len(xs) ** 0.5
    return {"k": len(xs), "df": len(xs) - 1, "mean": m, "sd": sd, "lo": m - half, "hi": m + half}


def confirmatory(battery, arms, resamples=RESAMPLES_SEALED, seed=BOOT_SEED_SEALED):
    """C1 (surface in) and C2 (surface ho_tpl): N-CFSD - N-VR on the parent-macro exact composite (COMPOSITE strata
    pooled by parent), each with its bootstrap p and its matched per-seed contrasts; Holm over the family {C1, C2}."""
    a, b = _pair(arms)
    res = {ep: dict(contrast(battery, arms, a, b, _composite_on(s), _exact, resamples, seed), surface=s,
                    control=a, treatment=b, per_seed=per_seed_contrasts(battery, arms, _composite_on(s)))
           for ep, s in ENDPOINTS}
    adj = ST.holm({ep: r["p"] for ep, r in res.items()})
    for ep in res:
        res[ep]["p_holm"] = adj[ep]
    return res


def protected(battery, arms, resamples=RESAMPLES_SEALED, seed=BOOT_SEED_SEALED, basis=PROTECTED_BASIS):
    """F0 posted twins (exact on twin_posted variants of the F strata, surface in) and the format-error rate (share
    of malformed outputs, all strata and surfaces, parent-macro), each N-CFSD - N-VR with a 95% CI."""
    a, b = _pair(arms)
    f0 = contrast(battery, arms, a, b, _f0_twin, _exact, resamples, seed)
    fe = contrast(battery, arms, a, b, _every, _is_format, resamples, seed)
    if basis == "ci":
        f0_ok, fe_ok = _gt(f0["lo"], F0_MARGIN), _le(fe["hi"], FORMAT_RISE_MAX)
        rules = (f"95% CI lower bound > {F0_MARGIN:+.2f}", f"95% CI upper bound <= {FORMAT_RISE_MAX:+.2f}")
    elif basis == "point":
        f0_ok, fe_ok = _gt(f0["diff"], F0_MARGIN), _le(fe["diff"], FORMAT_RISE_MAX)
        rules = (f"point estimate > {F0_MARGIN:+.2f}", f"point estimate <= {FORMAT_RISE_MAX:+.2f}")
    else:
        raise ValueError(f"unknown protected basis {basis!r}")
    return {"F0_twin_posted": dict(f0, ok=f0_ok, rule=rules[0]), "format_error_rate": dict(fe, ok=fe_ok, rule=rules[1]),
            "basis": basis, "ok": f0_ok and fe_ok}


def decide(conf, protected_ok, revise_rule=None, complete=True, order=None, per_seed_rule=None):
    """PREREG_E01N section 5, applied mechanically to conf = {endpoint: {diff, lo, hi, p_holm, per_seed}}.
    complete: every confirmatory cell was analysed (else no outcome holds: INCONCLUSIVE). The outcomes are tested in
    DECISION_ORDER (ADVANCE, STOP, REVISE); the first that holds is the decision, else INCONCLUSIVE. ADVANCE_PER_SEED:
    the qualifying endpoint's matched per-seed contrasts must all be > 0. Every clause value is returned
    (also_satisfied lists later outcomes that hold too)."""
    revise_rule = revise_rule or REVISE_RULE
    order = tuple(order or DECISION_ORDER)
    per_seed_rule = ADVANCE_PER_SEED if per_seed_rule is None else bool(per_seed_rule)
    if revise_rule not in ("literal", "not_advance"):
        raise ValueError(f"unknown revise rule {revise_rule!r}")
    if sorted(order) != ["ADVANCE", "REVISE", "STOP"]:
        raise ValueError(f"decision order must be a permutation of ADVANCE, REVISE, STOP, got {order}")
    (e1, _), (e2, _) = ENDPOINTS
    other = {e1: e2, e2: e1}
    per = {}
    for k in (e1, e2):
        c = conf[k]
        ps = c.get("per_seed")
        if per_seed_rule and ps is None:
            raise ValueError(f"{k}: per-seed contrasts are required by the ADVANCE per-seed condition")
        per[k] = {"holm_p_lt_alpha": _lt(c["p_holm"], ALPHA), "point_ge_practical": _ge(c["diff"], PRACTICAL),
                  "ci_lo_gt_0": _gt(c["lo"], 0.0), "point_gt_0": _gt(c["diff"], 0.0),
                  "all_seeds_gt_0": bool(ps) and all(_gt(v, 0.0) for v in ps.values()),
                  "other_ci_lo_gt_other_lb": _gt(conf[other[k]]["lo"], OTHER_LB),
                  "ci_hi_lt_stop_ub": _lt(c["hi"], STOP_UB)}
        per[k]["meets_advance_triple"] = (per[k]["holm_p_lt_alpha"] and per[k]["point_ge_practical"]
                                          and per[k]["ci_lo_gt_0"])
        per[k]["meets_advance_endpoint"] = per[k]["meets_advance_triple"] and (per[k]["all_seeds_gt_0"]
                                                                               or not per_seed_rule)
    advance = (bool(complete) and bool(protected_ok)
               and any(v["meets_advance_endpoint"] and v["other_ci_lo_gt_other_lb"] for v in per.values()))
    positive = any(v["holm_p_lt_alpha"] and v["point_gt_0"] for v in per.values())
    small = not any(v["point_ge_practical"] for v in per.values())
    clauses = {"ADVANCE": advance,
               "REVISE": bool(complete) and positive and (small if revise_rule == "literal" else not advance),
               "STOP": bool(complete) and all(v["ci_hi_lt_stop_ub"] for v in per.values())}
    decision = next((d for d in order if clauses[d]), "INCONCLUSIVE")
    return {"decision": decision, "clauses": clauses, "per_endpoint": per, "protected_ok": bool(protected_ok),
            "complete": bool(complete), "order": list(order), "revise_rule": revise_rule,
            "per_seed_rule": per_seed_rule, "also_satisfied": [d for d in order if clauses[d] and d != decision]}


def composite_score(battery, P, surface, field="exact"):
    return sum(cell(battery, P, s, surface, field) for s in COMPOSITE) / len(COMPOSITE)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _sd(xs):
    if len(xs) < 2:
        return None
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def error_counts(battery, P, surface):
    c = defaultdict(int)
    for r, p in zip(battery, P):
        if r["surface"] == surface and p["err"]:
            c[p["err"]] += 1
    return {k: c[k] for k in ERR_KINDS}


def wording_breakdown(battery, P):
    """Surface ho_tpl, all strata, per held-out wording '<rule op> <template id>' (AND/OR compositions use the h0
    frame for both ids): no-op items decided no-op, apply items decided apply, exact."""
    t = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r, p in zip(battery, P):
        if r["surface"] != "ho_tpl":
            continue
        x = t[f"{r['rule_ast']['op']} {r['tid']}"]
        j = 0 if r["gold"]["decision"] == "noop" else 2
        x[j] += p["decision_ok"]
        x[j + 1] += 1
        x[4] += p["exact"]
    return {k: {"noop_ok": v[0], "noop_n": v[1], "apply_ok": v[2], "apply_n": v[3], "exact": v[4]}
            for k, v in sorted(t.items())}


def describe_cell(battery, P):
    """Per-cell descriptives on each surface (composite, strata, no-op when required + Wilson, V3, verdict, errors)."""
    d = {}
    for _, s in ENDPOINTS:
        k, n = noop_rate(battery, P, s)
        lo, hi = ST.wilson(k, n)
        sens, both = rule_sensitivity(battery, P, s)
        d[s] = {"composite_exact": composite_score(battery, P, s),
                "composite_decision": composite_score(battery, P, s, "decision_ok"),
                "noop_when_required": {"k": k, "n": n, "wilson_lo": lo, "wilson_hi": hi},
                "v3_sensitivity": sens, "v3_both_exact": both, "verdict": verdict(battery, P, s)[0],
                "strata": {st: {"exact": cell(battery, P, st, s), "decision": cell(battery, P, st, s, "decision_ok")}
                           for st in STRATA},
                "errors": error_counts(battery, P, s)}
    d["wording_ho_tpl"] = wording_breakdown(battery, P)
    return d


def secondary(battery, arms, resamples=RESAMPLES_SEALED, seed=BOOT_SEED_SEALED):
    """Descriptive only (PREREG_E01N section 4 'Secondary'): per-cell descriptives, per-arm seed means and spread,
    matched-seed differences with a t interval (df = k - 1), C1/C2 on seeds SENSITIVITY_SEEDS only, N-CFSD - T2 and
    N-VR - OFF (95% CI, no test)."""
    order = [t for arm in ("OFF", "T2", CONTROL, TREATMENT, "N-FX") for t in tags_of(arms, arm)]
    order += [t for t in arms if t not in order]
    cells_d = {t: describe_cell(battery, arms[t]) for t in order}
    groups = {}
    for arm in ("OFF", "T2", CONTROL, TREATMENT, "N-FX"):
        tags = tags_of(arms, arm)
        if not tags:
            continue
        g = {"tags": tags}
        for _, s in ENDPOINTS:
            vals = [cells_d[t][s]["composite_exact"] for t in tags]
            k = sum(cells_d[t][s]["noop_when_required"]["k"] for t in tags)
            n = sum(cells_d[t][s]["noop_when_required"]["n"] for t in tags)
            lo, hi = ST.wilson(k, n)
            g[s] = {"composite_exact_mean": _mean(vals), "composite_exact_per_seed": vals, "sd": _sd(vals),
                    "range": max(vals) - min(vals),
                    "noop_when_required_pooled": {"k": k, "n": n, "wilson_lo": lo, "wilson_hi": hi},
                    "strata_exact_mean": {st: _mean([cells_d[t][s]["strata"][st]["exact"] for t in tags])
                                          for st in STRATA},
                    "strata_decision_mean": {st: _mean([cells_d[t][s]["strata"][st]["decision"] for t in tags])
                                             for st in STRATA}}
        w = defaultdict(lambda: defaultdict(int))
        for t in tags:
            for key, v in cells_d[t]["wording_ho_tpl"].items():
                for f, x in v.items():
                    w[key][f] += x
        g["wording_ho_tpl_pooled"] = {k: dict(v) for k, v in sorted(w.items())}
        groups[arm] = g
    matched = {}                    # the C1/C2 estimand per matched seed pair (parent macro, composite pooled)
    for _, s in ENDPOINTS:
        for k, v in per_seed_contrasts(battery, arms, _composite_on(s)).items():
            matched.setdefault(k, {})[s] = v
    seed_t = {s: seed_t_interval([v[s] for v in matched.values()]) for _, s in ENDPOINTS}
    sens = {}                       # without the DEV-gated seed 0 (same bootstrap; descriptive, p unadjusted)
    sa = [t for t in tags_of(arms, CONTROL) if split_tag(t)[1] in SENSITIVITY_SEEDS]
    sb = [t for t in tags_of(arms, TREATMENT) if split_tag(t)[1] in SENSITIVITY_SEEDS]
    if sa and sb:
        for ep, s in ENDPOINTS:
            x = contrast(battery, arms, sa, sb, _composite_on(s), _exact, resamples, seed)
            sens[ep] = dict(x, surface=s, control=sa, treatment=sb,
                            point_ge_practical_and_ci_lo_gt_0=_ge(x["diff"], PRACTICAL) and _gt(x["lo"], 0.0))
    refs = {}
    for b, a in SECONDARY_CONTRASTS:
        ta, tb = tags_of(arms, a), tags_of(arms, b)
        if ta and tb:
            refs[f"{b} - {a}"] = {s: {k: v for k, v in contrast(battery, arms, ta, tb, _composite_on(s), _exact,
                                                                 resamples, seed).items() if k not in ("p",)}
                                  for _, s in ENDPOINTS}
    return {"cells": cells_d, "arms": groups, "matched_seed_differences": matched, "seed_t_interval": seed_t,
            "seed_sensitivity": {"seeds": list(SENSITIVITY_SEEDS), "endpoints": sens}, "reference_contrasts": refs}


def analyze_sealed(battery, arms, meta, resamples=RESAMPLES_SEALED, seed=BOOT_SEED_SEALED):
    """The whole SEALED analysis: returns (result dict, markdown text). With a confirmatory cell missing, C1/C2 and the
    protected strata are computed on the seed indices both arms have (descriptive) and the decision is INCONCLUSIVE."""
    complete = not [t for t in CONFIRMATORY_CELLS if t not in arms]
    used = arms if complete else matched_seed_arms(arms)
    conf = confirmatory(battery, used, resamples, seed)
    prot = protected(battery, used, resamples, seed)
    dec = decide(conf, prot["ok"], complete=complete)
    here = Path(__file__).resolve()
    code = {p.name: sha_file(p) for p in (here, Path(ST.__file__).resolve(), here.parent / "native_e01.py")
            if p.exists()}
    res = {"experiment": "RCX-E01N", "analysis": "SEALED confirmation (PREREG_E01N)", "decision": dec["decision"],
           "decision_trace": dec, "confirmatory": conf, "protected": prot,
           "parameters": {"resamples": resamples, "prereg_resamples": resamples == RESAMPLES_SEALED,
                          "bootstrap_seed": seed, "p_floor": P_FLOOR, "alpha": ALPHA, "practical": PRACTICAL,
                          "other_lb": OTHER_LB, "stop_ub": STOP_UB, "f0_margin": F0_MARGIN,
                          "format_rise_max": FORMAT_RISE_MAX, "protected_basis": PROTECTED_BASIS, "eps": EPS,
                          "zero_tol": ST.ZERO_TOL, "composite": list(COMPOSITE), "f0_strata": list(F_STRATA),
                          "contrast": f"{TREATMENT} - {CONTROL}", "drop_train_overlap": DROP_TRAIN_OVERLAP,
                          "p_ties": P_TIES, "revise_rule": REVISE_RULE, "decision_order": list(DECISION_ORDER),
                          "advance_per_seed": ADVANCE_PER_SEED, "confirmatory_cells": list(CONFIRMATORY_CELLS),
                          "reference_cells": list(REFERENCE_CELLS), "sensitivity_seeds": list(SENSITIVITY_SEEDS)},
           "inputs": dict(meta, n_records_analysed=len(battery)), "code_sha256": code,
           "secondary": secondary(battery, arms, resamples, seed)}
    return res, render_sealed(res)


def _ci(r):
    return f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"


def _yn(b):
    return "yes" if b else "no"


def _kn(x):
    return f"{x['k']}/{x['n']} [{x['wilson_lo']:.3f}, {x['wilson_hi']:.3f}]"


def render_sealed(res):
    conf, prot, dec = res["confirmatory"], res["protected"], res["decision_trace"]
    inp, par, sec = res["inputs"], res["parameters"], res["secondary"]
    surfs = [s for _, s in ENDPOINTS]
    L = ["# RCX-E01N SEALED confirmation (PREREG_E01N)", "",
         f"**DECISION: {res['decision']}** (applied mechanically by `analyze_native.decide`; PREREG_E01N "
         "section 5)", ""]
    if not par["prereg_resamples"]:
        L += [f"**NOT THE PREREGISTERED ANALYSIS: {par['resamples']} resamples (PREREG_E01N fixes "
              f"{RESAMPLES_SEALED}).**", ""]
    if inp["missing"]:
        L += ["**Missing required cells** (reported as missing, never replaced; PREREG_E01N section 6): "
              + ", ".join(inp["missing"]) + "."
              + (" A confirmatory cell is missing: the decision is INCONCLUSIVE and C1/C2 below are descriptive, on the "
                 "seed indices both arms have." if not dec["complete"] else
                 " Only reference cells are missing: the decision is unaffected."), ""]
    L += [f"**Warning:** {w}" for w in inp["warnings"]] + ([""] if inp["warnings"] else [])
    L += ["## Inputs", "",
          f"- SEALED battery: seed {inp['battery_seed']}, sha256 `{inp['battery_sha256']}`, "
          f"{inp['n_records_rebuilt']} records (surfaces {', '.join(inp['surfaces'])}; "
          f"{inp['n_parents_per_stratum']} parents per stratum). Dropped train-overlap parents: "
          f"{len(inp['dropped_train_overlap_parents'])}; records analysed: {inp['n_records_analysed']}.",
          f"- Commitment sha256(salt) `{inp['commit_sha256_salt']}`, minted {inp['minted_at']}, opened "
          f"{inp['opened_at']}.",
          "- Cells (each `eval_sealed_<tag>.json` names this battery's sha256; N-* adapter shas equal open-sealed's "
          "record; every score re-derived from the raw output): "
          + ", ".join(t + ("" if v["required"] else " (descriptive)") for t, v in inp["cells"].items()) + ".",
          f"- Method: seed × parent paired cluster bootstrap, {par['resamples']} resamples, seed "
          f"{par['bootstrap_seed']} (variants averaged within parent per seed, seeds averaged, parents resampled; "
          "inference conditional on these adapters). Two-sided p = "
          + {"both": "2·min(share ≤ 0, share ≥ 0)", "low": "2·min(share ≤ 0, share > 0)",
             "split": "2·min(share < 0, share > 0) with zeros split"}[par["p_ties"]]
          + f" of the replicate differences, floored at {par['p_floor']:g}, capped at 1; Holm over C1, C2 at "
          f"α = {par['alpha']}. CIs are 95% percentile intervals, not multiplicity-adjusted.", ""]
    L += [f"## Confirmatory: {TREATMENT} − {CONTROL}", "",
          f"| endpoint | surface | {CONTROL} | {TREATMENT} | diff | 95% CI | p | p (Holm) | per-seed diff | parents "
          "| seeds |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ep, c in conf.items():
        L.append(f"| {ep} | {c['surface']} | {c['mean_a']:.3f} | {c['mean_b']:.3f} | {c['diff']:+.3f} | {_ci(c)} | "
                 f"{c['p']:.4f} | {c['p_holm']:.4f} | "
                 f"{', '.join(f'{k} {v:+.3f}' for k, v in c['per_seed'].items()) or '—'} | {c['n_parents']} | "
                 f"{c['n_seeds_a']} / {c['n_seeds_b']} |")
    L += ["", f"## Protected strata: {TREATMENT} − {CONTROL} (basis: {prot['basis']})", "",
          f"| stratum | {CONTROL} | {TREATMENT} | diff | 95% CI | parents | rule | within margin |",
          "|---|---|---|---|---|---|---|---|"]
    for name, label in (("F0_twin_posted", "F0 posted twins, exact (`in`)"),
                        ("format_error_rate", "format-error rate (all records)")):
        x = prot[name]
        L.append(f"| {label} | {x['mean_a']:.3f} | {x['mean_b']:.3f} | {x['diff']:+.3f} | {_ci(x)} | "
                 f"{x['n_parents']} | {x['rule']} | {_yn(x['ok'])} |")
    L += ["", f"## Decision trace (outcomes tested in the order {', '.join(dec['order'])}; the first that holds "
          "decides)", "",
          f"| endpoint | Holm p < {ALPHA} | point ≥ {PRACTICAL:+.2f} | CI lo > 0 | every per-seed diff > 0 | "
          f"other CI lo > {OTHER_LB:+.2f} | point > 0 | CI hi < {STOP_UB:+.2f} |", "|---|---|---|---|---|---|---|---|"]
    for ep, v in dec["per_endpoint"].items():
        L.append(f"| {ep} | {_yn(v['holm_p_lt_alpha'])} | {_yn(v['point_ge_practical'])} | {_yn(v['ci_lo_gt_0'])} | "
                 f"{_yn(v['all_seeds_gt_0'])}{'' if dec['per_seed_rule'] else ' (not required)'} | "
                 f"{_yn(v['other_ci_lo_gt_other_lb'])} | {_yn(v['point_gt_0'])} | {_yn(v['ci_hi_lt_stop_ub'])} |")
    L += ["", f"- All confirmatory cells analysed: {_yn(dec['complete'])}.",
          f"- Protected strata within margin: {_yn(dec['protected_ok'])}."]
    L += [f"- {d}: {_yn(dec['clauses'][d])}." for d in dec["order"]]
    L += [f"- **Decision: {dec['decision']}**"
          + (f" (later clauses that also hold: {', '.join(dec['also_satisfied'])})" if dec["also_satisfied"] else ""),
          ""]
    # ---------------------------------------------------------------- secondary (descriptive)
    arms, cells_d = sec["arms"], sec["cells"]
    L += ["## Secondary (descriptive; no test)", "", "### Arms (seed means; per-seed values; spread)", "",
          "| arm | cells | composite `in` | per seed | composite `ho_tpl` | per seed | SD in / ho_tpl | "
          "no-op when required, `in` (pooled, Wilson) | no-op when required, `ho_tpl` |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm, g in arms.items():
        a_, b_ = g[surfs[0]], g[surfs[1]]
        L.append(f"| {arm} | {len(g['tags'])} | {fmt(a_['composite_exact_mean'])} | "
                 f"{', '.join(f'{x:.3f}' for x in a_['composite_exact_per_seed'])} | "
                 f"{fmt(b_['composite_exact_mean'])} | "
                 f"{', '.join(f'{x:.3f}' for x in b_['composite_exact_per_seed'])} | "
                 f"{fmt(a_['sd'])} / {fmt(b_['sd'])} | {_kn(a_['noop_when_required_pooled'])} | "
                 f"{_kn(b_['noop_when_required_pooled'])} |")
    L += ["", "### Cells", "",
          "| cell | composite in / ho_tpl | decision in / ho_tpl | no-op when required, `in` | V3 sensitivity in / "
          "ho_tpl | V3 both-twins in / ho_tpl | PREREG-3-style verdict `in` | verdict `ho_tpl` |",
          "|---|---|---|---|---|---|---|---|"]
    for t, d in cells_d.items():
        a_, b_ = d[surfs[0]], d[surfs[1]]
        L.append(f"| {t} | {a_['composite_exact']:.3f} / {b_['composite_exact']:.3f} | "
                 f"{a_['composite_decision']:.3f} / {b_['composite_decision']:.3f} | {_kn(a_['noop_when_required'])} | "
                 f"{a_['v3_sensitivity']:.3f} / {b_['v3_sensitivity']:.3f} | {a_['v3_both_exact']:.3f} / "
                 f"{b_['v3_both_exact']:.3f} | {a_['verdict']} | {b_['verdict']} |")
    if sec["matched_seed_differences"]:
        L += ["", f"### Matched seeds: {TREATMENT}_s − {CONTROL}_s (composite exact, parent macro)", "",
              "| seed | in | ho_tpl |", "|---|---|---|"]
        L += [f"| {s} | {v[surfs[0]]:+.3f} | {v[surfs[1]]:+.3f} |" for s, v in sec["matched_seed_differences"].items()]
        ti = sec["seed_t_interval"]
        if all(ti[s] for s in surfs):
            L.append(f"| t interval (df = {ti[surfs[0]]['df']}) | {ti[surfs[0]]['mean']:+.3f} "
                     f"[{ti[surfs[0]]['lo']:+.3f}, {ti[surfs[0]]['hi']:+.3f}] | {ti[surfs[1]]['mean']:+.3f} "
                     f"[{ti[surfs[1]]['lo']:+.3f}, {ti[surfs[1]]['hi']:+.3f}] |")
    ss = sec["seed_sensitivity"]
    if ss["endpoints"]:
        L += ["", f"### Seed-0 sensitivity: C1/C2 on seeds {', '.join(map(str, ss['seeds']))} only (descriptive; p "
              "unadjusted)", "", f"| endpoint | surface | {CONTROL} | {TREATMENT} | diff | 95% CI | p | seeds | "
              f"point ≥ {PRACTICAL:+.2f} and CI lo > 0 |", "|---|---|---|---|---|---|---|---|---|"]
        for ep, x in ss["endpoints"].items():
            L.append(f"| {ep} | {x['surface']} | {x['mean_a']:.3f} | {x['mean_b']:.3f} | {x['diff']:+.3f} | {_ci(x)} | "
                     f"{x['p']:.4f} | {x['n_seeds_a']} / {x['n_seeds_b']} | "
                     f"{_yn(x['point_ge_practical_and_ci_lo_gt_0'])} |")
    if sec["reference_contrasts"]:
        L += ["", "### Reference contrasts (composite exact; seed × parent bootstrap; descriptive CI)", "",
              "| contrast | surface | A | B | B − A | 95% CI | parents |", "|---|---|---|---|---|---|---|"]
        for name, v in sec["reference_contrasts"].items():
            for s in surfs:
                x = v[s]
                L.append(f"| {name} | {s} | {x['mean_a']:.3f} | {x['mean_b']:.3f} | {x['diff']:+.3f} | {_ci(x)} | "
                         f"{x['n_parents']} |")
    for s in surfs:
        L += ["", f"### Per stratum, exact / decision (seed means), surface `{s}`", "",
              "| arm | " + " | ".join(st.split("_")[0] for st in STRATA) + " |", "|---|" + "---|" * len(STRATA)]
        for arm, g in arms.items():
            L.append(f"| {arm} | " + " | ".join(f"{g[s]['strata_exact_mean'][st]:.2f}/"
                                                f"{g[s]['strata_decision_mean'][st]:.2f}" for st in STRATA) + " |")
    L += ["", "### Error directions (counts per cell)", "",
          "| cell | surface | " + " | ".join(ERR_KINDS) + " |", "|---|---|" + "---|" * len(ERR_KINDS)]
    for t, d in cells_d.items():
        for s in surfs:
            L.append(f"| {t} | {s} | " + " | ".join(str(d[s]["errors"][k]) for k in ERR_KINDS) + " |")
    keys = sorted({k for g in arms.values() for k in g["wording_ho_tpl_pooled"]})
    L += ["", "### Held-out wording (`ho_tpl`, all strata, pooled over seeds): no-op items decided no-op · apply "
              "items decided apply", "", "AND/OR compositions render both h0 and h1 with the h0 frame "
              "(`templates.render_rule`).", "",
          "| wording | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for k in keys:
        row = []
        for g in arms.values():
            w = g["wording_ho_tpl_pooled"].get(k)
            row.append("—" if not w else f"{w['noop_ok']}/{w['noop_n']} · {w['apply_ok']}/{w['apply_n']}")
        L.append(f"| {k} | " + " | ".join(row) + " |")
    L += ["", "### M1 / M2 magnitudes (seed means, exact / decision)", "",
          "| arm | " + " | ".join(f"{st.split('_')[0]} `{s}`" for s in surfs for st in ("M1_bal", "M2_amt")) + " |",
          "|---|" + "---|" * (2 * len(surfs))]
    for arm, g in arms.items():
        L.append(f"| {arm} | " + " | ".join(
            f"{g[s]['strata_exact_mean'][st]:.2f} / {g[s]['strata_decision_mean'][st]:.2f}"
            for s in surfs for st in ("M1_bal", "M2_amt")) + " |")
    return "\n".join(L) + "\n"


def _write_new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as f:  # create-only: a confirmation output is never replaced
        f.write(text)


def main_sealed(a):
    if a.resamples != RESAMPLES_SEALED:
        raise SystemExit(f"PREREG_E01N fixes {RESAMPLES_SEALED} bootstrap resamples (got --resamples {a.resamples})")
    out = Path(a.out) if a.out else Path(a.cells).parent / "ANALYSIS_SEALED.md"
    jout = Path(a.json_out) if a.json_out else out.with_suffix(".json")
    existing = [str(p) for p in (out, jout) if p.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite {existing}: SEALED outputs are create-only (PREREG_E01N analyses "
                         "once)")
    battery, arms, meta = load_sealed(a.cells, allow_missing=a.allow_missing)
    res, txt = analyze_sealed(battery, arms, meta)
    _write_new(jout, json.dumps(res, indent=1) + "\n")
    _write_new(out, txt)
    print(txt)
    print(f"DECISION: {res['decision']}  (written {out} and {jout})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="runs/RCX-E01N/probe1")
    ap.add_argument("--cells", default="runs/RCX-E01N/cells")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--battery", default="dev", choices=["dev", "sealed"],
                    help="dev (default): the descriptive DEV analysis above; sealed: the PREREG_E01N confirmation")
    ap.add_argument("--json-out", dest="json_out", default=None,
                    help="sealed: machine-readable output (default: --out with suffix .json)")
    ap.add_argument("--allow-missing", dest="allow_missing", action="store_true",
                    help="sealed: analyse with required cells missing, reporting them as missing (PREREG section 6)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.battery == "sealed":
        return main_sealed(a)
    battery, bsha, arms = load_arms(a.probe, a.cells)
    order = [x for x in ("OFF", "SUPA", "T2") if x in arms] + sorted(k for k in arms if k.startswith("N-"))
    L = [f"# RCX-E01N native analysis (DEV battery sha `{bsha[:16]}…`, {len(battery)} records)", "",
         "**Tiers:** OFF and SUPA are tier A (frozen cores). T2 and N-* are trained students (component adapters on frozen "
         "FP32 Titan-Sparse, unmerged). **DEV, descriptive.** Confirmation requires the frozen PREREG_E01N on a SEALED "
         "battery, 3 seeds, and E33 retention.", "",
         "## Summary", "",
         "| arm | composite in | composite ho_tpl | decision in | no-op when required (varied, in) | V3 sensitivity | "
         "V3 both-twins | verdict (F strata, in) |", "|---|---|---|---|---|---|---|---|"]
    for arm in order:
        P = arms[arm]
        comp = lambda s, f="exact": (sum(cell(battery, P, x, s, f) for x in COMPOSITE) / len(COMPOSITE))  # noqa: E731
        k, n = noop_rate(battery, P, "in")
        lo, hi = ST.wilson(k, n)
        sens, both = rule_sensitivity(battery, P, "in")
        v, _ = verdict(battery, P, "in")
        L.append(f"| {arm} | {fmt(comp('in'))} | {fmt(comp('ho_tpl'))} | {fmt(comp('in', 'decision_ok'))} | "
                 f"{k}/{n} [{lo:.3f}, {hi:.3f}] | {sens:.3f} | {both:.3f} | {v} |")
    L += ["", "## Per-stratum exact / decision (surface `in`)", "",
          "| arm | " + " | ".join(s.split("_")[0] for s in STRATA) + " |", "|---|" + "---|" * len(STRATA)]
    for arm in order:
        P = arms[arm]
        L.append(f"| {arm} | " + " | ".join(
            f"{cell(battery, P, s, 'in'):.2f}/{cell(battery, P, s, 'in', 'decision_ok'):.2f}" for s in STRATA) + " |")
    L += ["", "## Paired contrasts on the primary composite (parent-clustered bootstrap; descriptive on DEV)", "",
          "| A | B | contrast | surface | A | B | A−B | 95% CI | n parents |", "|---|---|---|---|---|---|---|---|---|"]
    for A, Bm, desc in CONTRASTS:
        ka = [k for k in arms if k == A or k.startswith(A + "_s")]
        kb = [k for k in arms if k == Bm or k.startswith(Bm + "_s")]
        if not ka or not kb:
            continue
        for surf in ("in", "ho_tpl"):
            pa = [parents(battery, arms[k], COMPOSITE, surf) for k in ka]
            pb = [parents(battery, arms[k], COMPOSITE, surf) for k in kb]
            res = ST.seed_parent_bootstrap(pb, pa, resamples=a.resamples, seed=20261231)
            L.append(f"| {A} | {Bm} | {desc} | {surf} | {res['mean_b']:.3f} | {res['mean_a']:.3f} | "
                     f"{res['diff']:+.3f} | [{res['lo']:+.3f}, {res['hi']:+.3f}] | {res['n_parents']} |")
    txt = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8", newline="\n")
    print(txt)


if __name__ == "__main__":
    main()
