"""S-roll robustness check: is the DEV-regret-selected FULL learner still under-trained at its step cap?

Under the DEV-regret rule the FULL regression heads reach their best DEV checkpoint close to the
8,000-step cap. This script refits the DEV-regret-selected FULL configuration of each family with
the schedule doubled (<= 16,000 steps, cosine over the full length, DEV every 100 steps, patience
40), seeds 17-21, and evaluates it exactly like S-roll FULL (same rollout, decisions.g_only regret,
paired contrasts). DEV k=4 regret under the selection goals (seed 1002) is reported for both schedules
so a preference between them can be read on DEV alone. The main protocol and its numbers are unchanged.

  python scripts/s_roll_long.py [--family L,P,R]     (then: python scripts/s_roll.py --report-only)
Writes runs/bank_pilot/analysis/s_roll_long.{json,log}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s_roll as S  # noqa: E402  (sets CUDA_VISIBLE_DEVICES="", patches torch.cuda, 2 threads)
import numpy as np  # noqa: E402
from rcwmt import decide, decisions  # noqa: E402

LONG = dict(batch=256, max_steps=16000, eval_every=100, patience=40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="L,P,R")
    a = ap.parse_args()
    short2fam = {v: k for k, v in S.SHORT.items()}
    fams = [short2fam[x] for x in a.family.split(",")]
    log = S.Log(S.OUT / "s_roll_long.log")
    log(f"S-roll long-schedule check; threads={S.torch.get_num_threads()} cuda_available={S.torch.cuda.is_available()} "
        f"schedule={LONG}")
    bank, rows, T, cands, dcands = S.load_inputs(log)
    gen = S.load_gen()
    out_path = S.OUT / "s_roll_long.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    out["meta"] = {**S.code_meta(), "long_code_sha256": S.sha(Path(__file__).resolve()), "schedule": LONG}
    for fam in fams:
        f = S.SHORT[fam]
        t0 = time.time()
        sel_p = S.OUT / f"s_roll_select_regsel_{f}.json"
        main_p = S.OUT / f"s_roll_regsel_{f}.json"
        cfg = json.loads(sel_p.read_text(encoding="utf-8"))["selection"]["FULL"]["chosen"]
        main = json.loads(main_p.read_text(encoding="utf-8"))
        tr, dv, te = S.dedupe(fam, T[fam]["train"]), T[fam]["dev"], T[fam]["test"]
        per_k = {k: [] for k in S.KS}
        exs, tf, models, dreg = [], [], [], []
        for seed in S.SEEDS:
            t1 = time.time()
            e = cfg.get("encoding", "sym")
            m = S.OneStep(fam, cfg["head"], cfg["hidden"], cfg["depth"], cfg["lr"], seed, encoding=e, **LONG)
            m.fit(*S.arrays(fam, tr, e), *S.arrays(fam, dv, e))
            dr = S.dev_regret(rows, fam, S.rollout(dcands[fam], m.step_fn()))
            ch = S.rollout(cands[fam], m.step_fn())
            rg = S.chain_regrets(rows, fam, ch)
            for k in S.KS:
                per_k[k].append(rg[k][1])
            exs.append(S.exact_at_k(cands[fam], ch))
            tf.append(S.teacher_forced_stats(fam, m, te))
            dreg.append(dr)
            models.append({"seed": seed, "dev_exact": m.dev_acc, "dev_regret_k4": dr, "best_step": m.best_step,
                           "steps_run": m.steps_run})
            log(f"  fit {f} FULL-long seed{seed} dev_exact={m.dev_acc:.4f} dev_regret4={dr:.4f} "
                f"step={m.best_step}/{m.steps_run} ({time.time() - t1:.0f}s)")
        res = {"config": cfg, "models": models, "regret": {}, "contrasts": {}, "per_parent": {},
               "dev_regret_k4_mean": float(np.mean(dreg)),
               "main_fits": main["fits"]["FULL"]}
        for k in S.KS:
            with np.errstate(all="ignore"):
                pp = np.nanmean(np.stack(per_k[k]), 0)
            res["per_parent"][f"k{k}"] = pp.tolist()
            res["regret"][k] = {**S.summarize_pp(pp), "per_model_mean": [float(np.nanmean(x)) for x in per_k[k]]}
            for mode in ("chain_one", "chain_m1"):
                tit = decisions.g_only(rows, fam, k, gen, modes=(mode,))[mode]["nregret"]
                res["contrasts"][f"Titan {mode} - S-roll FULL-long|k{k}"] = decide.paired_bootstrap(
                    np.asarray(tit, np.float64), pp)
            res["contrasts"][f"S-roll FULL-long - S-roll FULL|k{k}"] = decide.paired_bootstrap(
                pp, np.array(main["per_parent"]["S-roll FULL"][f"k{k}"], np.float64))
        res["exact_state"] = {k: {"mean": float(np.mean([e[k] for e in exs]))} for k in S.KS}
        res["onestep"] = {"teacher_forced_test": S.mean_stats(tf)}
        c4 = res["contrasts"]["Titan chain_one - S-roll FULL-long|k4"]
        d4 = res["contrasts"]["S-roll FULL-long - S-roll FULL|k4"]
        log(f"[{f}] FULL-long {cfg} regret k1/2/4 = " + " / ".join(f"{res['regret'][k]['mean']:.3f}" for k in S.KS)
            + f" | DEV regret4 {res['dev_regret_k4_mean']:.4f} | TEST one-step "
            f"{res['onestep']['teacher_forced_test']['ALL']['acc_mean']:.3f} | Titan chain_one - long k4 "
            f"{c4['mean_diff']:+.3f} [{c4['ci95'][0]:+.3f}, {c4['ci95'][1]:+.3f}] | long - main FULL "
            f"{d4['mean_diff']:+.3f} [{d4['ci95'][0]:+.3f}, {d4['ci95'][1]:+.3f}]")
        res["wall_sec"] = round(time.time() - t0, 1)
        out[f] = res
        out_path.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
        log(f"[{f}] done in {res['wall_sec']:.0f}s -> {out_path.name}")
    log.f.close()


if __name__ == "__main__":
    main()
