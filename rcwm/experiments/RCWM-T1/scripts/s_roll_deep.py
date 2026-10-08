"""S-roll robustness check: extend the FULL depth search past the edge of the frozen grid (DEV only).

In the main run (scripts/s_roll.py) the FULL configuration chosen on DEV sat at the deepest point of
the grid (depth 3) for some families, so the grid may have under-powered S-roll. This script keeps
the chosen head / input / width / LR and adds one hidden layer at a time (seed 17, FULL TRAIN, DEV
one-step exact accuracy) while DEV improves, up to depth 6. If a deeper network wins on DEV, it is fitted
with seeds 17-21 and evaluated exactly like S-roll FULL: same rollout, same decisions.g_only regret,
same contrasts. Results go to runs/bank_pilot/analysis/s_roll_deep.{json,log}. They are rendered as a
separate robustness section of docs/s_roll/S_ROLL_PILOT.md. The main protocol and its numbers are
unchanged.

  python scripts/s_roll_deep.py [--family L,R,P]     (then: python scripts/s_roll.py --report-only)
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

MAX_DEPTH = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="L,P,R")
    a = ap.parse_args()
    short2fam = {v: k for k, v in S.SHORT.items()}
    fams = [short2fam[x] for x in a.family.split(",")]
    log = S.Log(S.OUT / "s_roll_deep.log")
    log(f"S-roll deep check; threads={S.torch.get_num_threads()} cuda_available={S.torch.cuda.is_available()}")
    bank, rows, T, cands, *_ = S.load_inputs(log)   # load_inputs also returns DEV candidates
    gen = S.load_gen()
    out_path = S.OUT / "s_roll_deep.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    out["meta"] = {**S.code_meta(), "deep_code_sha256": S.sha(Path(__file__).resolve()), "max_depth": MAX_DEPTH}
    for fam in fams:
        f = S.SHORT[fam]
        t0 = time.time()
        sel = json.loads((S.OUT / f"s_roll_select_{f}.json").read_text(encoding="utf-8"))["selection"]["FULL"]
        base = sel["chosen"]
        base_dev = max(r["dev_exact"] for r in sel["grid"]
                       if all(r[k] == base[k] for k in ("head", "encoding", "hidden", "depth", "lr")))
        tr, dv, te = S.dedupe(fam, T[fam]["train"]), T[fam]["dev"], T[fam]["test"]
        trace = [{**base, "dev_exact": base_dev, "source": "main grid"}]
        best = trace[0]
        d = base["depth"]
        while d < MAX_DEPTH:
            d += 1
            cfg = {**base, "depth": d}
            t1 = time.time()
            m = S.fit_one(fam, cfg, S.SEEDS[0], tr, dv, False)
            rec = {**cfg, "dev_exact": m.dev_acc, "dev_loss": m.dev_loss, "best_step": m.best_step,
                   "steps_run": m.steps_run, "source": "depth extension"}
            trace.append(rec)
            log(f"[{f}] depth {d}: dev_exact {m.dev_acc:.4f} (base depth {base['depth']}: {base_dev:.4f}) "
                f"step {m.best_step}/{m.steps_run} ({time.time() - t1:.0f}s)")
            if m.dev_acc > best["dev_exact"] + 1e-12:
                best = rec
            else:
                break
        res = {"base": base, "base_dev_exact": base_dev, "trace": trace,
               "chosen": {k: best[k] for k in ("head", "encoding", "hidden", "depth", "lr")},
               "chosen_dev_exact": best["dev_exact"], "improved": best is not trace[0]}
        if res["improved"]:
            cfg = res["chosen"]
            per_k = {k: [] for k in S.KS}
            exs, tf, s1 = [], [], []
            for seed in S.SEEDS:
                t1 = time.time()
                m = S.fit_one(fam, cfg, seed, tr, dv, False)
                ch = S.rollout(cands[fam], m.step_fn())
                rg = S.chain_regrets(rows, fam, ch)
                for k in S.KS:
                    per_k[k].append(rg[k][1])
                exs.append(S.exact_at_k(cands[fam], ch))
                tf.append(S.teacher_forced_stats(fam, m, te))
                s1.append(S.onestep_stats(fam, ch, cands[fam], "step1"))
                log(f"  fit {f} FULL-deep seed{seed} dev_exact={m.dev_acc:.4f} step={m.best_step}/{m.steps_run} "
                    f"({time.time() - t1:.0f}s)")
            main_pp = json.loads((S.OUT / f"s_roll_{f}.json").read_text(encoding="utf-8"))["per_parent"]
            res["regret"], res["contrasts"], res["per_parent"] = {}, {}, {}
            for k in S.KS:
                with np.errstate(all="ignore"):
                    pp = np.nanmean(np.stack(per_k[k]), 0)
                res["per_parent"][f"k{k}"] = pp.tolist()
                res["regret"][k] = {**S.summarize_pp(pp), "per_model_mean": [float(np.nanmean(x)) for x in per_k[k]]}
                for mode in ("chain_one", "chain_m1"):
                    tit = decisions.g_only(rows, fam, k, gen, modes=(mode,))[mode]["nregret"]
                    res["contrasts"][f"Titan {mode} - S-roll FULL-deep|k{k}"] = decide.paired_bootstrap(np.asarray(tit, np.float64), pp)
                res["contrasts"][f"S-roll FULL-deep - S-roll FULL|k{k}"] = decide.paired_bootstrap(
                    pp, np.array(main_pp["S-roll FULL"][f"k{k}"], np.float64))
            res["exact_state"] = {k: {"mean": float(np.mean([e[k] for e in exs])), "sd": float(np.std([e[k] for e in exs]))}
                                  for k in S.KS}
            res["onestep"] = {"teacher_forced_test": S.mean_stats(tf), "chain_step1": S.mean_stats(s1)}
            c4 = res["contrasts"]["Titan chain_one - S-roll FULL-deep|k4"]
            log(f"[{f}] FULL-deep regret k1/2/4 = " + " / ".join(f"{res['regret'][k]['mean']:.3f}" for k in S.KS)
                + f" | exact k4 {res['exact_state'][4]['mean']:.3f} | TEST one-step "
                f"{res['onestep']['teacher_forced_test']['ALL']['acc_mean']:.3f} | Titan chain_one - deep k4 "
                f"{c4['mean_diff']:+.3f} [{c4['ci95'][0]:+.3f}, {c4['ci95'][1]:+.3f}]")
        else:
            log(f"[{f}] no DEV gain from depth {base['depth'] + 1}; main FULL configuration stands")
        res["wall_sec"] = round(time.time() - t0, 1)
        out[fam] = res
        out_path.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
        log(f"[{f}] done in {res['wall_sec']:.0f}s -> {out_path.name}")
    log.f.close()


if __name__ == "__main__":
    main()
