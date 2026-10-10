"""BWM's independent re-score of PREREG-LB (joint roadmap item 3; co-sign of the report, PREREG-LB section 9.3).

Reads RCWM's frozen draw and episode files READ-ONLY and recomputes every registered number with BWM's own code: its
own ledger simulator (from the registered semantics: a posted add adds, a posted scale multiplies, a posted transfer
moves the amount from the account to the other one, a void event changes nothing), goal test, unmet hook, greedy
needs-lookahead procedure (D_MAX = 6, ties in vocabulary order), exact McNemar (scipy binomial), Holm (m = 2), the
registered branch table, and the paired percentile bootstrap (10,000 resamples, numpy default_rng(17)). Nothing from
RCWM's analysis code is imported. Compares with PREREG_LB_RESULTS.json and writes runs/LB_RESCORE/LB_RESCORE.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

RC = Path("C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1")
ROOT = Path(__file__).resolve().parents[1]
DRAW = RC / "docs/ledger_beam/LB_CONFIRM_DRAW.jsonl"
DRAW_SHA = "4fff07389cb1ac52b9d47773c7fd3a83f2d46681866bbdc82f9370889c3f6e55"
RESULTS = RC / "runs/lbeam/analysis/PREREG_LB_RESULTS.json"
EPISODES = {"T2": "primary/episodes_T2_W2E12.jsonl", "OFF": "primary/episodes_OFF_W2E12.jsonl",
            "ORACLE": "cpu/episodes_ORACLE_W2E12.jsonl", "IGNORE_STATUS": "cpu/episodes_IGNORE_STATUS_W2E12.jsonl",
            "T2_W4E24": "w4e24/episodes_T2_W4E24.jsonl", "ORACLE_W4E24": "cpu/episodes_ORACLE_W4E24.jsonl"}
D_MAX = 6


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def step(bal: dict, a) -> dict:
    kind, acct, other, amt, status, _ = a
    b = dict(bal)
    if status != "posted":
        return b
    if kind == "add":
        b[acct] += amt
    elif kind == "scale":
        b[acct] *= amt
    elif kind == "transfer":
        b[acct] -= amt
        b[other] += amt
    else:
        raise ValueError(kind)
    return b


def met(bal, goal) -> bool:
    return all(bal[a] == t for a, t in goal)


def unmet(bal, goal) -> int:
    return sum(abs(bal[a] - t) for a, t in goal)


def run(inst, plan):
    voc = {tuple(v) for v in inst["vocab"]}
    b = dict(map(tuple, inst["init"]))
    for a in plan:
        a = tuple(a)
        if a not in voc:
            return False
        b = step(b, a)
    return met(b, inst["goal"])


def needs_lookahead(inst) -> bool:
    b, g = dict(map(tuple, inst["init"])), inst["goal"]
    for _ in range(D_MAX):
        if met(b, g):
            break
        b = min((step(b, tuple(a)) for a in inst["vocab"]), key=lambda x: unmet(x, g))  # first minimum = vocab order
    return not met(b, g)


def mcnemar(x, y):
    b = int(sum(1 for i, j in zip(x, y) if i and not j))
    c = int(sum(1 for i, j in zip(x, y) if j and not i))
    p = 1.0 if b + c == 0 else min(1.0, binomtest(b, b + c, 0.5).pvalue)
    return b, c, p


def status(delta, p_holm, b, c):
    if p_holm >= 0.01:
        return "NS"
    if c > b:
        return "REVERSED"
    return "PASS" if delta >= 0.20 - 1e-12 else "SHORT"


def branch(s1, s2, near):
    if s1 == "PASS" and s2 == "REVERSED":
        return "MIXED_PASS"
    if s1 == "SHORT" and s2 == "REVERSED":
        return "MIXED_SHORT"
    if s1 == "REVERSED" and s2 in ("PASS", "SHORT"):
        return "MIXED_LOOKAHEAD_BETTER"
    if s1 == "REVERSED":
        return "REVERSED"
    if s1 == "NS" and s2 == "REVERSED":
        return "REVERSED_LOOKAHEAD"
    if s1 == "PASS" and s2 == "PASS":
        return "YES_NEAR" if near else "YES"
    if s1 == "PASS":
        return "GREEDY_ONLY_NEAR" if near else "GREEDY_ONLY"
    if s2 == "PASS":
        return "LOOKAHEAD_ONLY"
    if "SHORT" in (s1, s2):
        return "DIRECTIONAL"
    return "NO"


def ratio_ci(t2, orc, B=10000, seed=17):
    t2, orc = np.asarray(t2, float), np.asarray(orc, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t2), size=(B, len(t2)))
    a, o = t2[idx].mean(1), orc[idx].mean(1)
    keep = o > 0
    r = a[keep] / o[keep]
    return float(t2.mean() / orc.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    assert sha(DRAW) == DRAW_SHA, "draw differs from the sealed sha256"
    insts = [json.loads(l) for l in DRAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [i["id"] for i in insts]
    succ = {}
    for arm, rel in EPISODES.items():
        eps = {e["id"]: e for e in map(json.loads, (RC / "runs/lbeam" / rel).read_text(encoding="utf-8").splitlines())}
        assert set(eps) == set(ids), f"{arm}: episode ids differ from the draw"
        succ[arm] = [run(i, eps[i["id"]].get("plan") or []) for i in insts]
    nl = [needs_lookahead(i) for i in insts]
    t2, off = succ["T2"], succ["OFF"]
    b1, c1, p1 = mcnemar(t2, off)
    t2n = [x for x, k in zip(t2, nl) if k]
    offn = [x for x, k in zip(off, nl) if k]
    b2, c2, p2 = mcnemar(t2n, offn)
    order = sorted([(p1, 0), (p2, 1)])  # Holm m = 2, monotone step-down
    adj = [0.0, 0.0]
    run_max = 0.0
    for r, (p, k) in enumerate(order):
        run_max = max(run_max, min(1.0, (2 - r) * p))
        adj[k] = run_max
    d1, d2 = (b1 - c1) / len(t2), (b2 - c2) / max(1, len(t2n))
    s1, s2 = status(d1, adj[0], b1, c1), status(d2, adj[1], b2, c2)
    ratio, lo, hi = ratio_ci(t2, succ["ORACLE"])
    near = bool(ratio >= 0.80 and lo >= 0.70)
    mine = {"n": len(insts), "n_needs_lookahead": sum(nl),
            "success": {k: f"{sum(v)}/{len(v)}" for k, v in succ.items()},
            "cp1": {"b": b1, "c": c1, "delta": round(d1, 4), "p_raw": p1, "p_holm": adj[0], "status": s1},
            "cp2": {"n": len(t2n), "t2": sum(t2n), "off": sum(offn), "b": b2, "c": c2, "delta": round(d2, 4),
                    "p_raw": p2, "p_holm": adj[1], "status": s2},
            "ratio": {"ratio": round(ratio, 4), "lo": round(lo, 4), "hi": round(hi, 4), "near": near},
            "branch": branch(s1, s2, near)}
    theirs = json.loads(RESULTS.read_text(encoding="utf-8"))
    dec = theirs["decision"]
    checks = {
        "n_needs_lookahead": mine["n_needs_lookahead"] == theirs["n_needs_lookahead"],
        "needs_lookahead_per_instance": [bool(x) for x in nl] == [bool(x) for x in theirs["per_instance"]["needs_lookahead"]]
        if theirs["per_instance"].get("ids") == ids else "id order differs",
        "cp1_counts": (sum(t2), sum(off), b1, c1) == (dec["cp1"]["t2_success"], dec["cp1"]["off_success"],
                                                      dec["cp1"]["b_t2_only"], dec["cp1"]["c_off_only"]),
        "cp2_counts": (sum(t2n), sum(offn), b2, c2) == (dec["cp2"]["t2_success"], dec["cp2"]["off_success"],
                                                        dec["cp2"]["b_t2_only"], dec["cp2"]["c_off_only"]),
        "p_holm_close": abs(np.log10(adj[0]) - np.log10(dec["cp1"]["p_holm"])) < 1e-6 and
        abs(np.log10(adj[1]) - np.log10(dec["cp2"]["p_holm"])) < 1e-6,
        "statuses": (s1, s2) == (dec["status1"], dec["status2"]),
        "ratio_point": abs(ratio - dec["ratio"]["ratio"]) < 1e-9,
        "ratio_ci_within_0.01": abs(lo - dec["ratio"]["lo"]) < 0.01 and abs(hi - dec["ratio"]["hi"]) < 0.01,
        "near": near == dec["near"],
        "branch": mine["branch"] == dec["branch"],
    }
    checks = {k: (v if isinstance(v, str) else bool(v)) for k, v in checks.items()}
    out = ROOT / "runs/LB_RESCORE"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"draw_sha256": DRAW_SHA, "results_sha256": sha(RESULTS), "bwm_rescore": mine, "agreement": checks,
           "all_agree": all(v is True for v in checks.values())}
    (out / "LB_RESCORE.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
