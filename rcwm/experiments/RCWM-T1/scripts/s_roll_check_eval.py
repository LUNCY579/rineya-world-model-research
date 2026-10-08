"""Independent verifier for scripts/s_roll.py (lens: evaluation equivalence). Read-only on every
existing file; writes runs/bank_pilot/analysis/s_roll_check_eval.{json,log}.

Part A (numpy only, default): re-implements, WITHOUT calling rcwmt, the dynamics, phi, goals (seed 1001),
regret_table (fresh rng 17, 1e-9 tie noise), parent-mean fill and the paired bootstrap, then
  * recomputes Titan chain_one / chain_m1 per-parent regret (k=1,2,4) from the stored generations,
  * recomputes S-roll FULL per-model regret from the chains saved in s_roll_per_model_*.npz,
  * recomputes per-parent nanmean over models for every S-roll arm from the saved per-model vectors,
  * recomputes every paired-bootstrap contrast and every bootstrap-mean CI in s_roll.json,
  * cross-checks against decisions.g_only (called once, as a reference) and PREREG_1_DRYRUN_PER_PARENT.npz,
  * checks the parent set / NaN masks, Titan parse rates, and two sensitivity analyses
    (Titan last-parsed fill; analytic expected regret under uniform tie-breaking).
Part B (--retrain): retrains a subset of S-roll models with s_roll.py's own learner code (CPU, 2 threads)
  and checks that their rollouts, scored by the Part-A pipeline, equal the stored per-model regrets;
  plus a checkpoint-rule sensitivity for the TITAN-MATCHED L/R arms (train to max_steps, no early stop).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["RCWM_THREADS"] = "2"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "runs" / "bank_pilot" / "analysis"
FAMS = ("ledger_replay", "prerequisite_plan", "register_machine")
SH = {"ledger_replay": "L", "prerequisite_plan": "P", "register_machine": "R"}
ACC, REG, MOD, NB = ("A", "B", "C", "D"), ("X", "Y", "Z"), 97, 8
KS = (1, 2, 4)
SEEDS = (17, 18, 19, 20, 21)
LOGF = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n")
    LOGF.flush()


# ------------------------------------------------------------------ independent dynamics / phi

def bits(x):
    return [(x >> i) & 1 for i in range(NB)]


def my_step(fam, s, e):
    if fam == "ledger_replay":
        b = dict(s)
        if e["status"] == "posted":
            a, amt = e["account"], e["amount"]
            if e["kind"] == "add":
                b[a] += amt
            elif e["kind"] == "scale":
                b[a] *= amt
            elif e["kind"] == "transfer":
                b[a] -= amt
                b[e["other"]] += amt
            else:
                raise ValueError(e)
        return b
    if fam == "prerequisite_plan":
        if (s & e["pre_on"]) == e["pre_on"] and (s & e["pre_off"]) == 0:
            return (s & ~e["delete"]) | e["add"]
        return s
    r = dict(s)
    k, a, o, v = e["kind"], e["register"], e.get("other"), e.get("value", 0)
    if k == "swap":
        r[a], r[o] = r[o], r[a]
    elif k == "add":
        r[a] = (r[a] + v) % MOD
    elif k == "mix":
        r[a] = (r[a] + 2 * r[o] + v) % MOD
    elif k == "conditional":
        r[a] = (r[a] + (v if r[o] > e["threshold"] else -v)) % MOD
    else:
        raise ValueError(e)
    return r


def my_phi(fam, s0, fin):
    if fam == "prerequisite_plan":
        return np.array(bits(fin), np.float32)
    if fam == "ledger_replay":
        f = [fin[a] for a in ACC]
        return np.array([float(v < 0) for v in f] + [float(fin[a] != s0[a]) for a in ACC]
                        + [min(3.0, max(-3.0, v / 100.0)) for v in f], np.float32)
    f = [fin[r] for r in REG]
    return np.array([float(v > 48) for v in f] + [float(v % 2) for v in f]
                    + [float(fin[r] != s0[r]) for r in REG], np.float32)


DIM = {"ledger_replay": 12, "prerequisite_plan": 8, "register_machine": 9}


def my_goals(dim, n=256, seed=1001):
    g = np.random.default_rng(seed).standard_normal((n, dim))
    return (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)


def my_regret(Ut, Up, seed=17):
    rng = np.random.default_rng(seed)
    noise = rng.random(Up.shape) * 1e-9
    sel = np.argmax(Up + noise, axis=1)
    P, K, G = Ut.shape
    usel = Ut[np.arange(P)[:, None], sel, np.arange(G)[None, :]]
    ustar, umean = Ut.max(1), Ut.mean(1)
    num, den = (ustar - usel).sum(1), (ustar - umean).sum(1)
    out = np.full(P, np.nan)
    ok = den > 1e-9
    out[ok] = num[ok] / den[ok]
    return out


def my_regret_tie_expected(Ut, Up, atol=1e-6):
    """Expected regret under uniform random choice among candidates tied (within atol) at the max."""
    mx = Up.max(1, keepdims=True)
    tied = Up >= mx - atol                                  # [P,K,G]
    usel = (Ut * tied).sum(1) / tied.sum(1)
    ustar, umean = Ut.max(1), Ut.mean(1)
    num, den = (ustar - usel).sum(1), (ustar - umean).sum(1)
    out = np.full(len(Ut), np.nan)
    ok = den > 1e-9
    out[ok] = num[ok] / den[ok]
    return out, float(((tied.sum(1) > 1)).mean())


def my_fill(U):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = np.nanmean(U, axis=1, keepdims=True)
    return np.where(np.isnan(U), np.where(np.isnan(f), 0.0, f), U).astype(U.dtype)


def my_pboot(a, b, n_boot=10000, seed=17):
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    bs = d[idx].mean(1)
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}


def my_bmean(a, n_boot=10000, seed=17):
    a = a[~np.isnan(a)]
    idx = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))
    bs = a[idx].mean(1)
    return {"n": int(len(a)), "mean": float(a.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}


def nanmean0(x):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x, 0)


# ------------------------------------------------------------------ test decision sets

class Fam:
    def __init__(self, bank, fam):
        self.fam = fam
        ps = [p for p in bank["parents"] if p["family"] == fam and p["split"] == "test"]
        self.pids = sorted(p["parent_id"] for p in ps)
        byp = {p["parent_id"]: p for p in ps}
        self.K = len(ps[0]["candidates"])
        self.cands = []              # bank iteration order (matches s_roll.load_inputs cands order)
        for p in ps:
            for c in p["candidates"]:
                st = [p["s0"]]
                for e in c["events"]:
                    st.append(my_step(fam, st[-1], e))
                self.cands.append({"cid": c["cand_id"], "pid": p["parent_id"], "s0": p["s0"], "events": c["events"],
                                   "gold": st})
        self.cidx = {}
        for c in self.cands:
            pid, ci = c["cid"].rsplit("-c", 1)
            assert pid == c["pid"]
            self.cidx[c["cid"]] = (self.pids.index(pid), int(ci))
        assert all(len(byp[p]["candidates"]) == self.K for p in self.pids)
        self.G = my_goals(DIM[fam])
        self.Ut = {k: self.cube({c["cid"]: c["gold"][k] for c in self.cands}) for k in KS}

    def phis(self, states_by_cid):
        """states_by_cid: cid -> state or None -> [P,K,dim] float32 (NaN rows for None)."""
        P = np.full((len(self.pids), self.K, DIM[self.fam]), np.nan, np.float32)
        for c in self.cands:
            s = states_by_cid.get(c["cid"])
            if s is not None:
                pi, ci = self.cidx[c["cid"]]
                P[pi, ci] = my_phi(self.fam, c["s0"], s)
        return P

    def cube(self, states_by_cid):
        P = self.phis(states_by_cid)
        flat = P.reshape(-1, P.shape[-1]) @ self.G.T
        return flat.reshape(len(self.pids), self.K, -1)

    def regret(self, states_by_cid, k, tie_expected=False):
        U = my_fill(self.cube(states_by_cid))
        if tie_expected:
            return my_regret_tie_expected(self.Ut[k], U)
        return my_regret(self.Ut[k], U)


def titan_states(gen, mode, k, fill="nan"):
    out = {}
    for cid, g in gen[mode].items():
        st = g["states"]
        s = st[k]
        if s is None and fill == "last_parsed":
            prev = [x for x in st[: k + 1] if x is not None]
            s = prev[-1] if prev else None
        out[cid] = None if s is None else json.loads(s)
    return out


def close(a, b, tol=1e-6):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return False
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)))


def maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


# ------------------------------------------------------------------ Part A

def part_a(res):
    bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
    SR = json.loads((AN / "s_roll.json").read_text(encoding="utf-8"))
    dry = np.load(AN / "PREREG_1_DRYRUN_PER_PARENT.npz", allow_pickle=True)
    gen = {}
    for mode in ("chain_one", "chain_m1"):
        p = ROOT / "runs" / "bank_pilot" / "sparse" / f"gen_{mode}.jsonl"
        gen[mode] = {r["cand_id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    fails = []
    F_ = {}
    for fam in FAMS:
        f = SH[fam]
        D = Fam(bank, fam)
        F_[fam] = D
        R = {"n_test_parents": len(D.pids), "n_test_cands": len(D.cands)}
        # gold sanity: my dynamics vs rcwmt bank rows is checked in s_roll; here: oracle rollout through my path
        for k in KS:
            o = D.regret({c["cid"]: c["gold"][k] for c in D.cands}, k)
            assert np.nanmax(np.abs(o)) < 1e-9
        # parent set
        pids_json = SR["per_parent"][fam]["pids"]
        R["pids_match_json"] = pids_json == D.pids
        if not R["pids_match_json"]:
            fails.append(f"{f}: pid list differs")
        # Titan
        tit = {}
        for mode in ("chain_one", "chain_m1"):
            assert set(gen[mode]) == {c["cid"] for c in D.cands} | (set(gen[mode]) - {c["cid"] for c in D.cands})
            s0_ok = all(json.loads(gen[mode][c["cid"]]["states"][0]) == c["s0"] for c in D.cands)
            R[f"titan_{mode}_states0_eq_s0"] = s0_ok
            for k in KS:
                st = titan_states(gen, mode, k)
                pr = float(np.mean([st[c["cid"]] is not None for c in D.cands]))
                pp = D.regret(st, k)
                tit[(mode, k)] = pp
                js = np.array(SR["per_parent"][fam][f"Titan {mode}"][f"k{k}"], np.float64)
                dr = np.asarray(dry[f"{fam}|k{k}|Gtitan_{mode}"], np.float64)
                ex = float(np.mean([st[c["cid"]] == c["gold"][k] for c in D.cands]))
                R[f"Titan {mode}|k{k}"] = {"mean_mine": float(np.nanmean(pp)), "mean_json": SR["regret"][fam][f"Titan {mode}"][str(k)]["mean"],
                                           "maxdiff_vs_json_pp": maxdiff(pp, js), "maxdiff_vs_dryrun_pp": maxdiff(pp, dr),
                                           "nan_mask_eq_json": bool(np.array_equal(np.isnan(pp), np.isnan(js))),
                                           "parse_rate": pr, "exact_state": ex,
                                           "bmean_mine": my_bmean(pp)["ci95"],
                                           "bmean_json": SR["regret"][fam][f"Titan {mode}"][str(k)]["ci95"]}
                if not close(pp, js) or not close(pp, dr, 1e-5):
                    fails.append(f"{f} Titan {mode} k{k} per-parent mismatch {maxdiff(pp, js):.2e} / dry {maxdiff(pp, dr):.2e}")
                if not close(R[f"Titan {mode}|k{k}"]["bmean_mine"], R[f"Titan {mode}|k{k}"]["bmean_json"], 1e-6):
                    fails.append(f"{f} Titan {mode} k{k} bootstrap-mean CI mismatch")
                # sensitivity: last-parsed fill, tie-expected
                pl = D.regret(titan_states(gen, mode, k, fill="last_parsed"), k)
                tit[(mode + "_lastparsed", k)] = pl
                pdry = np.asarray(dry[f"{fam}|k{k}|Gtitan_{mode}_fill_lastparsed"], np.float64)
                R[f"Titan {mode}|k{k}"]["lastparsed_mean"] = float(np.nanmean(pl))
                R[f"Titan {mode}|k{k}"]["lastparsed_maxdiff_vs_dryrun"] = maxdiff(pl, pdry)
                te, tf = D.regret(st, k, tie_expected=True)
                tit[(mode + "_tieexp", k)] = te
                R[f"Titan {mode}|k{k}"]["tie_expected_mean"] = float(np.nanmean(te))
                R[f"Titan {mode}|k{k}"]["frac_parent_goal_with_tie"] = tf
        # S-roll FULL from saved chains
        z = np.load(AN / f"s_roll_per_model_{f}.npz")
        sr = {}
        full_models, full_tie, exk = {k: [] for k in KS}, {k: [] for k in KS}, {k: [] for k in KS}
        n_none, align = 0, []
        for s in SEEDS:
            arr = z[f"{f}|FULL|seed{s}|chains"]
            assert len(arr) == len(D.cands)
            chains = {}
            for c, js_ in zip(D.cands, arr):
                sts = [None if x is None else json.loads(x) for x in json.loads(str(js_))]
                n_none += sum(x is None for x in sts)
                assert sts[0] == c["s0"]
                chains[c["cid"]] = sts
                # alignment: S-roll step output vs true step of THIS candidate's events from its own input
                align += [sts[t + 1] == my_step(fam, sts[t], c["events"][t]) for t in range(4)]
            for k in KS:
                stk = {cid: v[k] for cid, v in chains.items()}
                full_models[k].append(D.regret(stk, k))
                te, tf = D.regret(stk, k, tie_expected=True)
                full_tie[k].append((te, tf))
                exk[k].append(float(np.mean([chains[c["cid"]][k] == c["gold"][k] for c in D.cands])))
        R["S-roll FULL chains: n None states"] = n_none
        R["S-roll FULL chains: own-input step correctness (alignment check)"] = float(np.mean(align))
        for k in KS:
            M = np.stack(full_models[k])
            stored = z[f"{f}|FULL|k{k}|per_model_nregret"]
            pp = nanmean0(M)
            js = np.array(SR["per_parent"][fam]["S-roll FULL"][f"k{k}"], np.float64)
            sr[("FULL", k)] = pp
            sr[("FULL_tieexp", k)] = nanmean0(np.stack([t[0] for t in full_tie[k]]))
            R[f"S-roll FULL|k{k}"] = {"maxdiff_per_model_vs_npz": maxdiff(M, stored), "maxdiff_pp_vs_json": maxdiff(pp, js),
                                      "mean_mine": float(np.nanmean(pp)), "mean_json": SR["regret"][fam]["FULL"][str(k)]["mean"],
                                      "ci_mine": my_bmean(pp)["ci95"], "ci_json": SR["regret"][fam]["FULL"][str(k)]["ci95"],
                                      "nan_mask_eq_titan": bool(np.array_equal(np.isnan(pp), np.isnan(tit[("chain_one", k)]))),
                                      "exact_state_mine": float(np.mean(exk[k])),
                                      "exact_state_json": SR["exact_state"][fam]["FULL"][str(k)]["mean"],
                                      "tie_expected_mean": float(np.nanmean(sr[("FULL_tieexp", k)])),
                                      "frac_parent_goal_with_tie_mean": float(np.mean([t[1] for t in full_tie[k]]))}
            if not close(M, stored) or not close(pp, js):
                fails.append(f"{f} S-roll FULL k{k} mismatch per-model {maxdiff(M, stored):.2e} pp {maxdiff(pp, js):.2e}")
            if not close(R[f"S-roll FULL|k{k}"]["ci_mine"], R[f"S-roll FULL|k{k}"]["ci_json"]):
                fails.append(f"{f} S-roll FULL k{k} CI mismatch")
            if abs(R[f"S-roll FULL|k{k}"]["exact_state_mine"] - R[f"S-roll FULL|k{k}"]["exact_state_json"]) > 1e-9:
                fails.append(f"{f} S-roll FULL k{k} exact-state mismatch")
        # other arms: per-model vectors -> nanmean (prereg1 S rule)
        arms = sorted({kk.split("|")[1] for kk in z.files if kk.endswith("per_model_nregret")})
        for arm in arms:
            for k in KS:
                M = z[f"{f}|{arm}|k{k}|per_model_nregret"]
                pp = nanmean0(M)
                sr[(arm, k)] = pp
                js = np.array(SR["per_parent"][fam][f"S-roll {arm}"][f"k{k}"], np.float64)
                rj = SR["regret"][fam][arm][str(k)]
                bm = my_bmean(pp)
                R[f"S-roll {arm}|k{k}|pm"] = {"n_models": int(M.shape[0]), "maxdiff_pp_vs_json": maxdiff(pp, js),
                                           "mean_mine": bm["mean"], "mean_json": rj["mean"], "ci_mine": bm["ci95"], "ci_json": rj["ci95"],
                                           "per_model_means_match": close([np.nanmean(x) for x in M], rj["per_model_mean"])}
                if not close(pp, js) or not close(bm["ci95"], rj["ci95"]) or not R[f"S-roll {arm}|k{k}|pm"]["per_model_means_match"]:
                    fails.append(f"{f} S-roll {arm} k{k} mismatch")
        # contrasts: every one in s_roll.json
        cm = {}
        for name, cj in SR["contrasts"][fam].items():
            lhs, k = name.split("|")
            k = int(k[1:])
            tmode, arm = lhs.split(" - ")
            tmode = tmode.replace("Titan ", "")
            arm = arm.replace("S-roll ", "")
            mine = my_pboot(tit[(tmode, k)], sr[(arm, k)])
            ok = mine["n"] == cj["n"] and close([mine["mean_diff"]] + mine["ci95"], [cj["mean_diff"]] + cj["ci95"])
            cm[name] = {"mine": mine, "json": cj, "match": ok}
            if not ok:
                fails.append(f"{f} contrast {name} mismatch: mine {mine} json {cj}")
        R["contrasts_checked"] = len(cm)
        R["contrasts_all_match"] = all(v["match"] for v in cm.values())
        # sensitivity contrasts at k=4 (and k=1,2)
        sens = {}
        for k in KS:
            for arm in ("FULL", "MATCHED128"):
                sens[f"k{k}|Titan chain_one(lastparsed) - S-roll {arm}"] = my_pboot(tit[("chain_one_lastparsed", k)], sr[(arm, k)])
            sens[f"k{k}|tie-expected: Titan chain_one - S-roll FULL"] = my_pboot(tit[("chain_one_tieexp", k)], sr[("FULL_tieexp", k)])
        # degenerate parents: every candidate reaches the SAME true phi (indistinguishable under every goal;
        # preregistered as excluded) but float32 rounding leaves |den| ~ 1e-7..1e-6, so den > 1e-9 can keep them in
        degen = {}
        for k in KS:
            Pt = D.phis({c["cid"]: c["gold"][k] for c in D.cands})
            nd = np.array([len({tuple(Pt[p, c]) for c in range(D.K)}) for p in range(len(D.pids))])
            den = (D.Ut[k].max(1) - D.Ut[k].mean(1)).sum(1)
            bad = (nd == 1)
            kept_bad = bad & ~np.isnan(tit[("chain_one", k)])
            info = {"n_indistinguishable": int(bad.sum()), "n_indistinguishable_kept": int(kept_bad.sum()),
                    "den_of_indistinguishable": den[bad].tolist(),
                    "titan_c1_on_kept": tit[("chain_one", k)][kept_bad].tolist(),
                    "sroll_full_on_kept": sr[("FULL", k)][kept_bad].tolist(),
                    "sroll_matched_on_kept": sr[("MATCHED128", k)][kept_bad].tolist()}
            if kept_bad.any():
                msk = lambda x: np.where(bad, np.nan, x)
                info["corrected"] = {
                    "Titan chain_one": float(np.nanmean(msk(tit[("chain_one", k)]))),
                    "S-roll FULL": float(np.nanmean(msk(sr[("FULL", k)]))),
                    "S-roll MATCHED128": float(np.nanmean(msk(sr[("MATCHED128", k)]))),
                    "Titan c1 - FULL": my_pboot(msk(tit[("chain_one", k)]), msk(sr[("FULL", k)])),
                    "Titan c1 - MATCHED128": my_pboot(msk(tit[("chain_one", k)]), msk(sr[("MATCHED128", k)])),
                    "tie-expected Titan c1 - FULL": my_pboot(msk(tit[("chain_one_tieexp", k)]), msk(sr[("FULL_tieexp", k)]))}
            degen[f"k{k}"] = info
            if bad.any():
                log(f"   DEGENERATE {f} k{k}: {int(bad.sum())} parents with one true phi, {int(kept_bad.sum())} kept (den {np.round(den[bad], 9).tolist()}); "
                    f"Titan c1 on kept {np.round(info['titan_c1_on_kept'], 3).tolist()} S-roll FULL {np.round(info['sroll_full_on_kept'], 3).tolist()} "
                    f"MATCHED {np.round(info['sroll_matched_on_kept'], 3).tolist()}")
                if "corrected" in info:
                    cc = info["corrected"]
                    log(f"      corrected (drop them): Titan {cc['Titan chain_one']:.4f} FULL {cc['S-roll FULL']:.4f} MATCHED {cc['S-roll MATCHED128']:.4f} | "
                        f"c1-FULL {cc['Titan c1 - FULL']['mean_diff']:+.4f} {np.round(cc['Titan c1 - FULL']['ci95'], 4).tolist()} n={cc['Titan c1 - FULL']['n']} | "
                        f"c1-MATCHED {cc['Titan c1 - MATCHED128']['mean_diff']:+.4f} {np.round(cc['Titan c1 - MATCHED128']['ci95'], 4).tolist()} | "
                        f"tie-exp c1-FULL {cc['tie-expected Titan c1 - FULL']['mean_diff']:+.4f}")
        R["degenerate_parents"] = degen
        R["sensitivity_contrasts"] = sens
        res[f] = R
        log(f"== {f}: parents {len(D.pids)} | pids==json {R['pids_match_json']} | contrasts {R['contrasts_checked']} all match "
            f"{R['contrasts_all_match']}")
        for k in KS:
            t = R[f"Titan chain_one|k{k}"]
            s = R[f"S-roll FULL|k{k}"]
            log(f"   k{k} Titan c1 mine {t['mean_mine']:.4f} json {t['mean_json']:.4f} pp-diff json {t['maxdiff_vs_json_pp']:.1e} "
                f"dry {t['maxdiff_vs_dryrun_pp']:.1e} parse {t['parse_rate']:.3f} exact {t['exact_state']:.3f} "
                f"lastparsed {t['lastparsed_mean']:.4f} tieexp {t['tie_expected_mean']:.4f} tiefrac {t['frac_parent_goal_with_tie']:.3f}")
            log(f"   k{k} S-roll FULL mine {s['mean_mine']:.4f} json {s['mean_json']:.4f} per-model diff {s['maxdiff_per_model_vs_npz']:.1e} "
                f"pp diff {s['maxdiff_pp_vs_json']:.1e} CI {np.round(s['ci_mine'], 4).tolist()} vs {np.round(s['ci_json'], 4).tolist()} "
                f"exact {s['exact_state_mine']:.3f}/{s['exact_state_json']:.3f} tieexp {s['tie_expected_mean']:.4f} "
                f"tiefrac {s['frac_parent_goal_with_tie_mean']:.3f} nanmask==titan {s['nan_mask_eq_titan']}")
        for arm in arms:
            s = R[f"S-roll {arm}|k4|pm"]
            log(f"   k4 S-roll {arm:<11} ({s['n_models']} models) mine {s['mean_mine']:.4f} json {s['mean_json']:.4f} "
                f"pp diff {s['maxdiff_pp_vs_json']:.1e} CI {np.round(s['ci_mine'], 4).tolist()}")
        for name in ("Titan chain_one - S-roll FULL|k4", "Titan chain_one - S-roll MATCHED128|k4"):
            v = cm[name]
            log(f"   {name}: mine {v['mine']['mean_diff']:+.4f} {np.round(v['mine']['ci95'], 4).tolist()} n={v['mine']['n']} | "
                f"json {v['json']['mean_diff']:+.4f} {np.round(v['json']['ci95'], 4).tolist()}")
        for name, v in sens.items():
            if name.startswith("k4"):
                log(f"   SENS {name}: {v['mean_diff']:+.4f} {np.round(v['ci95'], 4).tolist()}")
        log(f"   S-roll FULL chains None states: {n_none}; own-input step correctness {R['S-roll FULL chains: own-input step correctness (alignment check)']:.3f}")
    res["part_a_failures"] = fails
    log(f"PART A failures: {len(fails)}")
    for x in fails:
        log("  FAIL", x)
    return F_, gen


def part_a_gonly_reference(res, gen):
    """One reference call of the repo pipeline (decisions.g_only) per family/k on Titan chain_one, to confirm
    my re-implementation also equals g_only called on the full modes tuple vs one mode (rng sharing)."""
    import torch                                   # rcwmt.heads imports torch: patch first (I-21)
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT))
    from rcwmt import data, decisions
    assert not torch.cuda.is_available()
    rows = data.candidate_table(data.load_bank("bank_pilot"))
    out = {}
    for fam in FAMS:
        for k in KS:
            one = decisions.g_only(rows, fam, k, gen, modes=("chain_one",))["chain_one"]["nregret"]
            both = decisions.g_only(rows, fam, k, gen, modes=("chain_one", "chain_m1"))
            out[f"{SH[fam]}|k{k}"] = {"g_only_chain_one": float(np.nanmean(one)),
                                     "mine": res[SH[fam]][f"Titan chain_one|k{k}"]["mean_mine"],
                                     "g_only_chain_m1_after_chain_one_shared_rng": float(np.nanmean(both["chain_m1"]["nregret"])),
                                     "chain_m1_fresh_rng_mine": res[SH[fam]][f"Titan chain_m1|k{k}"]["mean_mine"]}
            log(f"   g_only ref {SH[fam]} k{k}: {out[f'{SH[fam]}|k{k}']}")
    res["g_only_reference"] = out


# ------------------------------------------------------------------ Part B (retrain subset with s_roll's learner)

def part_b(res, F_, which, ckpt_sens):
    import torch
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    import s_roll as S  # module import only; it re-applies the same no-GPU guards
    assert not torch.cuda.is_available()
    from rcwmt import env
    bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
    out = {}
    for fam in FAMS:
        f = SH[fam]
        D = F_[fam]
        sel = json.loads((AN / f"s_roll_select_{f}.json").read_text(encoding="utf-8"))["selection"]
        # transitions by split, own construction (same iteration order as s_roll.load_inputs)
        T = {"train": [], "dev": [], "test": []}
        for p in bank["parents"]:
            if p["family"] != fam:
                continue
            for c in p["candidates"]:
                s = p["s0"]
                for t, e in enumerate(c["events"]):
                    s2 = my_step(fam, s, e)
                    assert s2 == env.step(fam, s, e)
                    T[p["split"]].append({"parent_id": p["parent_id"], "cand_id": c["cand_id"], "t": t, "s": s, "e": e, "s2": s2})
                    s = s2
        tr = S.dedupe(fam, T["train"])
        dv = T["dev"]
        z = np.load(AN / f"s_roll_per_model_{f}.npz")

        def roll(model):
            cur = [c["s0"] for c in D.cands]
            hist = [[c["s0"]] for c in D.cands]
            for t in range(4):
                ev = [c["events"][t] for c in D.cands]
                X = np.stack([S.enc(fam, s, e, model.encoding) for s, e in zip(cur, ev)]).astype(np.float32)
                Sa = np.stack([S.st_arr(fam, s) for s in cur])
                nxt = model.predict(X, Sa)
                cur = [S.arr_st(fam, a) for a in nxt]
                for h, s in zip(hist, cur):
                    h.append(s)
            return {c["cid"]: h for c, h in zip(D.cands, hist)}

        jobs = []
        if "matched" in which:
            for r in range(5):
                for si, s in enumerate(SEEDS):
                    if r == 0 or si == 0:            # resample 0 all seeds + seed 17 of every resample (9 models)
                        jobs.append(("MATCHED128", sel["N128"]["chosen"], S.sample(tr, 128, S.RESAMPLE_SEED0 + r), s, r * 5 + si))
        if "full" in which and fam != "ledger_replay":
            jobs.append(("FULL", sel["FULL"]["chosen"], tr, 17, 0))
        if "fullL" in which and fam == "ledger_replay":
            jobs.append(("FULL", sel["FULL"]["chosen"], tr, 17, 0))
        recs = []
        for arm, cfg, trs, seed, row in jobs:
            t0 = time.time()
            m = S.fit_one(fam, cfg, seed, trs, dv, False)
            ch = roll(m)
            reg = {k: D.regret({cid: v[k] for cid, v in ch.items()}, k) for k in KS}
            stored = {k: z[f"{f}|{arm}|k{k}|per_model_nregret"][row] for k in KS}
            d = max(maxdiff(reg[k], stored[k]) for k in KS)
            recs.append({"arm": arm, "row": row, "seed": seed, "n": len(trs), "dev_exact": m.dev_acc, "best_step": m.best_step,
                         "k4_mine": float(np.nanmean(reg[4])), "k4_stored": float(np.nanmean(stored[4])), "maxdiff": d})
            log(f"   retrain {f} {arm} row{row} seed{seed} n={len(trs)} dev={m.dev_acc:.4f} step={m.best_step} "
                f"k4 regret retrained {np.nanmean(reg[4]):.4f} stored {np.nanmean(stored[4]):.4f} maxdiff {d:.1e} ({time.time() - t0:.1f}s)")
        out[f] = {"retrain": recs}
        # checkpoint-rule sensitivity (MATCHED, L and R): no early stopping, final model after max_steps
        if ckpt_sens and fam in ("ledger_replay", "register_machine"):
            sens = []
            for r in range(5):
                cfg = sel["N128"]["chosen"]
                trs = S.sample(tr, 128, S.RESAMPLE_SEED0 + r)
                m = S.make(fam, cfg, 17, len(trs), False)
                m.eval_every = m.max_steps                    # single evaluation at the end -> final weights kept
                t0 = time.time()
                m.fit(*S.arrays(fam, trs, m.encoding), *S.arrays(fam, dv, m.encoding))
                ch = roll(m)
                reg4 = D.regret({cid: v[4] for cid, v in ch.items()}, 4)
                ex4 = float(np.mean([ch[c["cid"]][4] == c["gold"][4] for c in D.cands]))
                sens.append({"resample": r, "dev_exact_final": m.dev_acc, "steps": m.steps_run, "k4": reg4.tolist(),
                             "k4_mean": float(np.nanmean(reg4)), "exact4": ex4,
                             "stored_es_k4_mean": float(np.nanmean(z[f"{f}|MATCHED128|k4|per_model_nregret"][r * 5]))})
                log(f"   ckpt-sens {f} MATCHED r{r} seed17 final@{m.steps_run}: dev_exact {m.dev_acc:.4f} k4 regret {np.nanmean(reg4):.4f} "
                    f"(early-stopped stored {sens[-1]['stored_es_k4_mean']:.4f}) exact4 {ex4:.3f} ({time.time() - t0:.1f}s)")
            pp = nanmean0(np.stack([np.array(s_["k4"]) for s_ in sens]))
            out[f]["ckpt_sens"] = {"models": sens, "pp_mean": float(np.nanmean(pp))}
            out[f]["ckpt_sens"]["contrast_titan_minus_final"] = my_pboot(np.array(res[f]["_titan_c1_k4"]), pp)
            log(f"   ckpt-sens {f}: final-weights MATCHED (5 resamples, seed17) k4 regret {np.nanmean(pp):.4f}; "
                f"Titan c1 - it {out[f]['ckpt_sens']['contrast_titan_minus_final']}")
    res["part_b"] = out


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", default="", help="comma list of: matched,full,fullL")
    ap.add_argument("--ckpt-sens", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    LOGF = open(AN / f"s_roll_check_eval{a.tag}.log", "w", encoding="utf-8")
    res = {}
    t0 = time.time()
    F_, gen = part_a(res)
    part_a_gonly_reference(res, gen)
    # stash Titan chain_one k4 per-parent (mine) for part B contrasts
    for fam in FAMS:
        D = F_[fam]
        res[SH[fam]]["_titan_c1_k4"] = D.regret(titan_states(gen, "chain_one", 4), 4).tolist()
    if a.retrain or a.ckpt_sens:
        part_b(res, F_, set(a.retrain.split(",")) if a.retrain else set(), a.ckpt_sens)
    res["wall_sec"] = round(time.time() - t0, 1)
    (AN / f"s_roll_check_eval{a.tag}.json").write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    log(f"done in {res['wall_sec']}s")


if __name__ == "__main__":
    main()
