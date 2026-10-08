"""VERIFIER check (adversarial review of s_roll.py; does not change any S-roll output).

Question: the MATCHED128 S-roll fits stop at best_step 25-50 (L) and 25 (R, every model) because the
checkpoint is chosen by DEV one-step EXACT accuracy, which at n=128 is maximised by a barely trained
net. Does choosing the checkpoint by DEV LOSS instead (same head/width/depth/LR/schedule, same
resamples, seed 17) change k=4 regret by > 0.05? DEV k=4 regret is reported for both rules so the
choice between them can be made on DEV alone.

  python scripts/s_roll_verify_stop.py [--family L,R,P] [--seeds 17]
Writes runs/bank_pilot/analysis/s_roll_verify_stop.json / .log.
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
from rcwmt import decide, decisions, env  # noqa: E402

assert not torch.cuda.is_available()


class LossStop(S.OneStep):
    """Identical to S.OneStep except the checkpoint / early stopping criterion is DEV loss."""

    def fit(self, X, S_, S2, Xdv, Sdv, S2dv):
        import math
        self.mu, self.sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-3
        ytr = self._targets(S_, S2, fit=True)
        ydv = self._targets(Sdv, S2dv)
        xtr, xdv = self._x(X), self._x(Xdv)
        torch.manual_seed(self.seed)
        net = S._Net(X.shape[1], self._d_out(), self.hidden, self.depth)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, self.max_steps) / self.max_steps)))
        g = torch.Generator().manual_seed(self.seed)
        n, b = len(X), min(self.batch, len(X))
        perm, ptr = torch.randperm(n, generator=g), 0
        best_loss, best_state, best_step, bad, step, best_acc = float("inf"), None, 0, 0, 0, 0.0
        for step in range(1, self.max_steps + 1):
            if ptr + b > n:
                perm, ptr = torch.randperm(n, generator=g), 0
            idx = perm[ptr: ptr + b]
            ptr += b
            net.train()
            opt.zero_grad()
            self._loss(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
            sched.step()
            if step % self.eval_every == 0 or step == self.max_steps:
                net.eval()
                with torch.no_grad():
                    o = net(xdv)
                    dl = float(self._loss(o, ydv))
                acc = float((self._decode(o.numpy(), Sdv) == S2dv).all(1).mean())
                if dl < best_loss - 1e-6:
                    best_loss, best_step, bad, best_acc = dl, step, 0, acc
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        net.load_state_dict(best_state)
        net.eval()
        self.net, self.dev_acc, self.dev_loss, self.best_step, self.steps_run = net, best_acc, best_loss, best_step, step
        return self


def dev_cands(bank, fam):
    out = []
    for p in bank["parents"]:
        if p["family"] != fam or p["split"] != "dev":
            continue
        for c in p["candidates"]:
            out.append({"cand_id": c["cand_id"], "parent_id": p["parent_id"], "s0": p["s0"], "events": c["events"],
                        "gold": env.rollout(fam, p["s0"], c["events"])})
    return out


def regrets(rows, fam, ch, split):
    r = decisions.g_only(rows, fam, 4, {"m": ch}, modes=("m",), split=split)["m"]
    return np.asarray(r["nregret"], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="L,R,P")
    ap.add_argument("--seeds", default="17")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    short2fam = {v: k for k, v in S.SHORT.items()}
    log = S.Log(S.OUT / "s_roll_verify_stop.log")
    bank, rows, T, cands, *_ = S.load_inputs(log)   # load_inputs also returns DEV candidates
    gen = S.load_gen()
    out_path = S.OUT / "s_roll_verify_stop.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for f in a.family.split(","):
        fam = short2fam[f]
        t0 = time.time()
        cfg = json.loads((S.OUT / f"s_roll_select_{f}.json").read_text(encoding="utf-8"))["selection"]["N128"]["chosen"]
        tr, dv, te = S.dedupe(fam, T[fam]["train"]), T[fam]["dev"], T[fam]["test"]
        dc = dev_cands(bank, fam)
        npz = np.load(S.OUT / f"s_roll_per_model_{f}.npz")
        main_pm = npz[f"{f}|MATCHED128|k4|per_model_nregret"]          # rows r0_seed17..r4_seed21
        titan = np.asarray(decisions.g_only(rows, fam, 4, gen, modes=("chain_one",))["chain_one"]["nregret"], np.float64)
        res = {"config": cfg, "seeds": seeds, "models": []}
        pp = {"exact": [], "loss": []}
        for r in range(S.N_RESAMPLES):
            trs = S.sample(tr, S.MATCHED_N, S.RESAMPLE_SEED0 + r)
            for seed in seeds:
                rec = {"resample": r, "seed": seed}
                for rule, cls in (("exact", S.OneStep), ("loss", LossStop)):
                    m = cls(fam, cfg["head"], cfg["hidden"], cfg["depth"], cfg["lr"], seed,
                            encoding=cfg.get("encoding", "sym"), **S.train_kw(len(trs), False))
                    m.fit(*S.arrays(fam, trs, m.encoding), *S.arrays(fam, dv, m.encoding))
                    ch_te = S.rollout(cands[fam], m.step_fn())
                    ch_dv = S.rollout(dc, m.step_fn())
                    rt, rd = regrets(rows, fam, ch_te, "test"), regrets(rows, fam, ch_dv, "dev")
                    X, S0, S2 = S.arrays(fam, te, m.encoding)
                    rec[rule] = {"best_step": m.best_step, "steps_run": m.steps_run, "dev_exact": m.dev_acc,
                                 "test_onestep_exact": float((m.predict(X, S0) == S2).all(1).mean()),
                                 "dev_regret_k4": float(np.nanmean(rd)), "test_regret_k4": float(np.nanmean(rt)),
                                 "test_exact_k4": S.exact_at_k(cands[fam], ch_te)[4]}
                    pp[rule].append(rt)
                    if rule == "exact" and seed in S.SEEDS:
                        j = r * len(S.SEEDS) + S.SEEDS.index(seed)
                        rec["exact"]["matches_main_run_test_regret"] = bool(np.allclose(rt, main_pm[j], equal_nan=True))
                res["models"].append(rec)
                log(f"[{f}] r{r} seed{seed}: EXACT-stop step {rec['exact']['best_step']} devEx {rec['exact']['dev_exact']:.3f} "
                    f"devReg {rec['exact']['dev_regret_k4']:.3f} testReg {rec['exact']['test_regret_k4']:.3f} "
                    f"(=main {rec['exact'].get('matches_main_run_test_regret')}) | LOSS-stop step {rec['loss']['best_step']} "
                    f"devEx {rec['loss']['dev_exact']:.3f} devReg {rec['loss']['dev_regret_k4']:.3f} "
                    f"testReg {rec['loss']['test_regret_k4']:.3f} testOneStep {rec['loss']['test_onestep_exact']:.3f} "
                    f"({time.time() - t0:.0f}s)")
        summ = {}
        for rule in ("exact", "loss"):
            with np.errstate(all="ignore"):
                p = np.nanmean(np.stack(pp[rule]), 0)
            summ[rule] = {"test_regret_k4": S.summarize_pp(p),
                          "dev_regret_k4_mean": float(np.mean([m[rule]["dev_regret_k4"] for m in res["models"]])),
                          "dev_exact_mean": float(np.mean([m[rule]["dev_exact"] for m in res["models"]])),
                          "test_onestep_mean": float(np.mean([m[rule]["test_onestep_exact"] for m in res["models"]])),
                          "titan_chain_one_minus": decide.paired_bootstrap(titan, p)}
        with np.errstate(all="ignore"):
            summ["loss_minus_exact"] = decide.paired_bootstrap(np.nanmean(np.stack(pp["loss"]), 0),
                                                               np.nanmean(np.stack(pp["exact"]), 0))
        res["summary"] = summ
        res["wall_sec"] = round(time.time() - t0, 1)
        log(f"[{f}] SUMMARY exact-stop TEST k4 {summ['exact']['test_regret_k4']['mean']:.3f} (DEV {summ['exact']['dev_regret_k4_mean']:.3f}) | "
            f"loss-stop TEST k4 {summ['loss']['test_regret_k4']['mean']:.3f} (DEV {summ['loss']['dev_regret_k4_mean']:.3f}) | "
            f"loss-exact {summ['loss_minus_exact']['mean_diff']:+.3f} {summ['loss_minus_exact']['ci95']} | "
            f"Titan-loss {summ['loss']['titan_chain_one_minus']['mean_diff']:+.3f} {summ['loss']['titan_chain_one_minus']['ci95']}")
        out[f] = res
        out_path.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    log.f.close()


if __name__ == "__main__":
    main()
