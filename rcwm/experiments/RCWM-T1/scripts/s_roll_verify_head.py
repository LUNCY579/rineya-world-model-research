"""VERIFIER check (adversarial review of s_roll.py; does not change any S-roll output).

Question: s_roll.py picks the output head on DEV one-step EXACT accuracy. The decision endpoint is
regret through phi, which rewards approximately right states (L: sign, changed, clip(bal/100);
R: hi>48, parity, changed). Would picking the head on DEV k=4 REGRET (still DEV-only) change TEST
k=4 regret by > 0.05?  Candidates per (family, budget): the configuration s_roll.py chose, plus
every other head of s_roll.HEADS at the stage-A size (256 x 2, LR 3e-3). Same training code,
same exact-accuracy checkpointing, same rollout and decisions.g_only regret.
Models: N128 = resamples 0-4 with seed 17; FULL = seeds 17, 18, 19.

  python scripts/s_roll_verify_head.py [--family L,R]
Writes runs/bank_pilot/analysis/s_roll_verify_head.json / .log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s_roll as S  # noqa: E402  (CUDA_VISIBLE_DEVICES="", torch.cuda patched, 2 threads)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rcwmt import decide, decisions  # noqa: E402
from s_roll_verify_stop import dev_cands  # noqa: E402

assert not torch.cuda.is_available()


def g4(rows, fam, ch, split):
    return np.asarray(decisions.g_only(rows, fam, 4, {"m": ch}, modes=("m",), split=split)["m"]["nregret"], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="L,R")
    ap.add_argument("--budgets", default="N128,FULL")
    a = ap.parse_args()
    short2fam = {v: k for k, v in S.SHORT.items()}
    log = S.Log(S.OUT / "s_roll_verify_head.log")
    bank, rows, T, cands, *_ = S.load_inputs(log)   # load_inputs also returns DEV candidates
    gen = S.load_gen()
    out_path = S.OUT / "s_roll_verify_head.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for f in a.family.split(","):
        fam = short2fam[f]
        tr, dv = S.dedupe(fam, T[fam]["train"]), T[fam]["dev"]
        dc = dev_cands(bank, fam)
        titan = np.asarray(decisions.g_only(rows, fam, 4, gen, modes=("chain_one",))["chain_one"]["nregret"], np.float64)
        sel = json.loads((S.OUT / f"s_roll_select_{f}.json").read_text(encoding="utf-8"))["selection"]
        out.setdefault(f, {})
        for budget in a.budgets.split(","):
            t0 = time.time()
            chosen = sel[budget]["chosen"]
            cfgs = [("main-chosen", chosen)] + [(f"{h}@stageA", {"head": h, "encoding": "sym", **S.STAGE_A})
                                                for h in S.HEADS[fam] if h != chosen["head"]]
            if budget == "N128":
                sets = [(S.sample(tr, S.MATCHED_N, S.RESAMPLE_SEED0 + r), 17, f"r{r}_seed17") for r in range(S.N_RESAMPLES)]
            else:
                sets = [(tr, s, f"seed{s}") for s in (17, 18, 19)]
            res = {}
            for name, cfg in cfgs:
                te_pp, dv_m, ex = [], [], []
                for trs, seed, tag in sets:
                    m = S.fit_one(fam, cfg, seed, trs, dv, False)
                    ch_te, ch_dv = S.rollout(cands[fam], m.step_fn()), S.rollout(dc, m.step_fn())
                    rt, rd = g4(rows, fam, ch_te, "test"), g4(rows, fam, ch_dv, "dev")
                    te_pp.append(rt)
                    dv_m.append(float(np.nanmean(rd)))
                    ex.append(m.dev_acc)
                    log(f"[{f}] {budget} {name:<20} {tag:<11} devExact {m.dev_acc:.3f} devReg4 {np.nanmean(rd):.3f} "
                        f"testReg4 {np.nanmean(rt):.3f} step {m.best_step}/{m.steps_run} ({time.time() - t0:.0f}s)")
                with np.errstate(all="ignore"):
                    pp = np.nanmean(np.stack(te_pp), 0)
                res[name] = {"config": cfg, "dev_exact_mean": float(np.mean(ex)), "dev_regret_k4_mean": float(np.mean(dv_m)),
                             "test_regret_k4": S.summarize_pp(pp), "per_parent_test_k4": pp.tolist(),
                             "titan_chain_one_minus": decide.paired_bootstrap(titan, pp)}
            best = min(res, key=lambda n: res[n]["dev_regret_k4_mean"])
            main_pp = np.array(res["main-chosen"]["per_parent_test_k4"])
            summ = {"chosen_by_dev_regret": best,
                    "chosen_minus_main": decide.paired_bootstrap(np.array(res[best]["per_parent_test_k4"]), main_pp)}
            out[f][budget] = {"candidates": res, "summary": summ, "wall_sec": round(time.time() - t0, 1)}
            log(f"[{f}] {budget} SUMMARY " + " | ".join(
                f"{n}: devExact {v['dev_exact_mean']:.3f} devReg {v['dev_regret_k4_mean']:.3f} testReg {v['test_regret_k4']['mean']:.3f} "
                f"Titan-it {v['titan_chain_one_minus']['mean_diff']:+.3f} [{v['titan_chain_one_minus']['ci95'][0]:+.3f}, "
                f"{v['titan_chain_one_minus']['ci95'][1]:+.3f}]" for n, v in res.items())
                + f" || DEV-regret pick: {best}; pick - main {summ['chosen_minus_main']['mean_diff']:+.3f} "
                f"[{summ['chosen_minus_main']['ci95'][0]:+.3f}, {summ['chosen_minus_main']['ci95'][1]:+.3f}]")
            out_path.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    log.f.close()


if __name__ == "__main__":
    main()
