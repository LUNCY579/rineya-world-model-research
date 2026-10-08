import sys, json
import numpy as np
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
from pathlib import Path
from rcwmt import data, decide, decisions, env, fidelity
from analyze import load_gen
rows = [r for r in data.candidate_table(data.load_bank("bank_pilot")) if r["split"] == "test"]
gen = {x: load_gen(Path(ROOT) / "runs" / "bank_pilot" / x) for x in ("sparse", "supa")}
def cube_from_states(fam, k, states_fn):
    te = [r for r in rows if r["family"] == fam and r["k"] == k]
    dim = len(te[0]["phi"]); G = decide.goals(dim, 256, 1001)
    pids = sorted({r["parent_id"] for r in te}); pos = {r["item_id"]: i for i, r in enumerate(te)}
    Ut = decisions._cube(pids, 8, k, pos, np.stack([r["phi"] for r in te]) @ G.T)
    ph = []; nfill = 0
    for r in te:
        st = states_fn(r)
        if st is None: nfill += 1
        ph.append(np.full(dim, np.nan, np.float32) if st is None else data.phi_from_state(fam, r["s0"], r["events"], st))
    Uc = decisions._cube(pids, 8, k, pos, np.stack(ph) @ G.T)
    fill = np.nanmean(Uc, axis=1, keepdims=True)
    Uc = np.where(np.isnan(Uc), np.where(np.isnan(fill), 0.0, fill), Uc)
    # tie rate: share of (parent,goal) where argmax is tied
    mx = Uc.max(1, keepdims=True); ties = ((np.abs(Uc - mx) < 1e-6).sum(1) > 1).mean()
    t = decide.regret_table(Ut, Uc, np.random.default_rng(17))
    return t["nregret"], nfill / len(te), ties
def chain_state(arm, mode, j=None):
    def f(r):
        g = gen[arm][mode].get(r["cand_id"])
        if g is None: return None
        s = g["states"][j if j else r["k"]]
        return None if s is None else json.loads(s)
    return f
def e2e_state(arm):
    def f(r):
        g = gen[arm]["e2e"].get(r["item_id"])
        return None if g is None else env.parse_state(r["family"], g["text"], lenient=True)
    return f
def heur(name):
    return lambda r: fidelity.heuristic_states(r["family"], r["s0"], r["events"])[name]
res = {}
for fam in ("ledger_replay", "register_machine", "prerequisite_plan"):
    for k in (1, 2, 4):
        M = {"T_chain": chain_state("sparse", "chain_one"), "T_m1": chain_state("sparse", "chain_m1"), "T_e2e": e2e_state("sparse"),
             "S_chain": chain_state("supa", "chain_one"), "S_m1": chain_state("supa", "chain_m1"), "S_e2e": e2e_state("supa"),
             "F_first": heur("first_only"), "F_last": heur("last_only")}
        if k == 4:
            for j in (1, 2, 3): M[f"T_chain_trunc{j}"] = chain_state("sparse", "chain_one", j)
        out = {}
        for m, fn in M.items():
            nr, fr, ti = cube_from_states(fam, k, fn)
            out[m] = (nr, fr, ti)
        res[(fam, k)] = out
        print(f"\n{fam} k{k}: " + " ".join(f"{m}={np.nanmean(v[0]):.3f}(fill{v[1]:.2f},tie{v[2]:.2f})" for m, v in out.items()))
        if k == 4 or k == 2:
            base = out["T_chain"][0]
            for m in out:
                if m == "T_chain": continue
                d = base - out[m][0]; d = d[~np.isnan(d)]
                print(f"   T_chain - {m}: mean {d.mean():+.3f} sd {d.std(ddof=1):.3f} n {len(d)}")
# exact-state accuracy
print("\nEXACT final state accuracy")
for fam in ("ledger_replay", "register_machine", "prerequisite_plan"):
    for k in (1, 2, 4):
        te = [r for r in rows if r["family"] == fam and r["k"] == k]
        line = []
        for arm in ("sparse", "supa"):
            for nm, fn in (("chain", chain_state(arm, "chain_one")), ("m1", chain_state(arm, "chain_m1")), ("e2e", e2e_state(arm))):
                acc = np.mean([fn(r) == r["states"][-1] for r in te])
                line.append(f"{arm[:2]}_{nm}={acc:.3f}")
        print(fam[:6], k, " ".join(line))
# supa chain breakage
print("\nChain breakage (share of k4 chains with None final) and lenient-recoverable first break")
for arm in ("sparse", "supa"):
    for mode in ("chain_one", "chain_m1"):
        for fam in ("ledger_replay", "register_machine", "prerequisite_plan"):
            recs = [g for cid, g in gen[arm][mode].items() if any(r["cand_id"] == cid and r["family"] == fam and r["k"] == 4 for r in rows[:0])]
            recs = [gen[arm][mode][r["cand_id"]] for r in rows if r["family"] == fam and r["k"] == 4]
            broken = [g for g in recs if g["states"][-1] is None]
            rec = 0; wrapped = 0
            for g in broken:
                t = len(g["texts"]) - 1
                txt = g["texts"][t]
                if env.parse_state(fam, txt, lenient=True) is not None: rec += 1
                if '"after"' in txt or '"before"' in txt: wrapped += 1
            print(f"{arm} {mode} {fam[:6]}: broken {len(broken)}/{len(recs)}, last text lenient-parseable {rec}, wrapped {wrapped}")
