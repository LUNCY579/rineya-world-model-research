"""Probe-verified rollout (PVR) experiment.

  stage1   GPU. For every step t of every chain in --splits, start from the GOLD state s_t, run the
           step prompt (P: multi format with one event, which states the no-op rule; L/R: Titan's
           trained one-step frame), record the answer-slot state at each family's probe layer, the
           greedy answer and N samples (T) with mean token log-probs.
  hstates  GPU. Answer-slot states of another arm (e.g. Supa) on exactly the stage-1 prompts.
  analyze  CPU. Fit verifiers on the pilot's train-split one-step captures (layer chosen on DEV
           log-likelihood), score every candidate, compare selection methods per step.
  stage2   GPU. Closed-loop rollouts from s0 where each step's state is the method's own pick.

  python scripts/run_pvr.py stage1 --bank bank_pilot --arm sparse --splits dev,test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
from rcwmt import data, decide, env, heads, pvr  # noqa: E402

FAMS = env.FAMILIES


def step_messages(fam, before, ev):
    if fam == "prerequisite_plan":
        return env.multi_step_messages(fam, before, [ev])
    return env.one_step_messages(fam, before, ev)


def probe_items(arm: data.Arm, fam: str):
    """One-step training items in the same prompt frame stage 1 uses."""
    suffix = "-k1" if fam == "prerequisite_plan" else "-one"
    return [it for it in arm.items if it["family"] == fam and it["item_id"].endswith(suffix)]


def fit_verifiers(bank_stem: str, armname: str, budget: int | None = None, layers=None, seed: int = 17,
                  fixed_layers: dict | None = None):
    """Per family: choose the layer by DEV mean log-likelihood of the true next state (or use
    fixed_layers[fam] when states were already recorded at a given layer); return
    {fam: (layer, Verifier, dev_loglik)}."""
    arm = data.Arm(bank_stem, armname)
    bank = data.load_bank(bank_stem)
    byc = {c["cand_id"]: (p, c) for p in bank["parents"] for c in p["candidates"]}
    out = {}
    for fam in FAMS:
        its = probe_items(arm, fam)
        sp = {s: [it for it in its if it["split"] == s] for s in ("train", "dev")}
        if budget:
            rng = np.random.default_rng(seed)
            sp["train"] = [sp["train"][i] for i in sorted(rng.choice(len(sp["train"]), min(budget, len(sp["train"])), replace=False))]

        def target(it):
            p, c = byc[it["cand_id"]]
            return pvr.next_phi(fam, p["s0"], env.step(fam, p["s0"], c["events"][0]))
        Y = {s: np.stack([target(it) for it in sp[s]]) for s in sp}
        best = None
        cand_layers = [fixed_layers[fam]] if fixed_layers else (layers or [l for l in arm.layers if l != 0])
        for L in cand_layers:
            X = {s: arm.X_last([it["item_id"] for it in sp[s]], L) for s in sp}
            v = pvr.Verifier(fam).fit(X["train"], Y["train"], X["dev"], Y["dev"])
            post = v.posterior(X["dev"])
            ll = np.mean([v.loglik(post, Y["dev"][i], i) for i in range(len(Y["dev"]))])
            if best is None or ll > best[0]:
                best = (ll, L, v)
        out[fam] = (best[1], best[2], float(best[0]))
        print(f"  verifier[{armname}] {fam}: layer {best[1]} dev mean loglik {best[0]:.3f} (n_train {len(sp['train'])})", flush=True)
    return out


def chains_for(bank, splits):
    return [{"cand_id": c["cand_id"], "parent_id": p["parent_id"], "family": p["family"], "split": p["split"],
             "s0": p["s0"], "events": c["events"]} for p in bank["parents"] if p["split"] in splits for c in p["candidates"]]


def stage1(a):
    from rcwmt import generation, runtime
    bank = data.load_bank(a.bank)
    ver = fit_verifiers(a.bank, a.arm)
    layers = {f: ver[f][0] for f in FAMS}
    chains = chains_for(bank, set(a.splits.split(",")))
    steps = []
    for ch in chains:
        st = env.rollout(ch["family"], ch["s0"], ch["events"])
        for t, ev in enumerate(ch["events"]):
            steps.append({**{k: ch[k] for k in ("cand_id", "parent_id", "family", "split")}, "t": t,
                          "before": st[t], "event": ev, "gold": env.dumps(st[t + 1])})
    out = ROOT / "runs" / a.bank / f"pvr_{a.arm}"
    out.mkdir(parents=True, exist_ok=True)
    tok = runtime.tokenizer()
    with runtime.gpu_lease(label=f"rcwm-t1:pvr1:{a.arm}", job=f"RCWM-T1 PVR stage1 {a.bank}"):
        model, prov = runtime.load_arm(a.arm)
        H = np.zeros((len(steps), 1024), dtype=np.float32)
        recs = [None] * len(steps)
        t0 = time.time()
        for fam in FAMS:
            ix = [i for i, s in enumerate(steps) if s["family"] == fam]
            for b in range(0, len(ix), a.batch):
                bi = ix[b: b + a.batch]
                res = generation.sample_batch(model, tok, [step_messages(fam, steps[i]["before"], steps[i]["event"]) for i in bi],
                                              a.n, a.temp, generation.max_tokens(fam), layers[fam], seed=1000 + b)
                for i, r in zip(bi, res):
                    H[i] = r["h"]
                    for c in r["cands"]:
                        ps = env.parse_state(fam, c["text"])
                        c["parsed"] = None if ps is None else env.dumps(ps)
                        c["correct"] = c["parsed"] == steps[i]["gold"]
                    recs[i] = {**steps[i], "cands": r["cands"]}
                if (b // a.batch) % 20 == 0:
                    print(f"  {fam} {b + len(bi)}/{len(ix)} ({time.time() - t0:.0f}s)", flush=True)
        runtime.unload(model)
    np.save(out / "stage1_h.npy", H)
    with (out / "stage1.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    (out / "stage1_meta.json").write_text(json.dumps({"layers": layers, "n": a.n, "temp": a.temp, "splits": a.splits,
                                                       "provenance": prov}, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"stage1 done: {len(steps)} steps in {time.time() - t0:.0f}s", flush=True)


def hstates(a):
    """Answer-slot states of --arm on the stage-1 prompts recorded by --source-arm."""
    from rcwmt import capture, runtime
    src = ROOT / "runs" / a.bank / f"pvr_{a.source_arm}"
    recs = [json.loads(l) for l in (src / "stage1.jsonl").read_text(encoding="utf-8").splitlines()]
    ver = fit_verifiers(a.bank, a.arm)
    layers = {f: ver[f][0] for f in FAMS}
    tok = runtime.tokenizer()
    enc = [{"ids": tok(capture.render(tok, step_messages(r["family"], r["before"], r["event"])), add_special_tokens=False)["input_ids"]}
           for r in recs]
    for e in enc:
        e["last"], e["events"] = len(e["ids"]) - 1, []
    with runtime.gpu_lease(label=f"rcwm-t1:pvrh:{a.arm}", job=f"RCWM-T1 PVR hstates {a.arm}"):
        model, _ = runtime.load_arm(a.arm)
        used = sorted(set(layers.values()))
        last, _ = capture.capture(model, enc, layers=tuple(used), log_every=0)
        runtime.unload(model)
    H = np.stack([last[i, used.index(layers[r["family"]])].astype(np.float32) for i, r in enumerate(recs)])
    np.save(src / f"hstates_{a.arm}.npy", H)
    (src / f"hstates_{a.arm}_meta.json").write_text(json.dumps({"layers": layers}) + "\n", encoding="utf-8")
    print(f"hstates {a.arm}: {H.shape}", flush=True)


def s_verifier(bank_stem: str, fam: str, budget=None, seed=17):
    """No-LLM verifier: symbolic (before, event) -> next-state verifier features, same labels."""
    bank = data.load_bank(bank_stem)
    rows = {s: [] for s in ("train", "dev")}
    for p in bank["parents"]:
        if p["family"] != fam or p["split"] not in rows:
            continue
        for c in p["candidates"]:
            e = c["events"][0]
            rows[p["split"]].append((p["s0"], e, env.step(fam, p["s0"], e)))
    if budget:
        rng = np.random.default_rng(seed)
        rows["train"] = [rows["train"][i] for i in sorted(rng.choice(len(rows["train"]), min(budget, len(rows["train"])), replace=False))]
    X = {s: np.stack([decide.symbolic(fam, b, [e], kmax=1) for b, e, _ in v]) for s, v in rows.items()}
    Y = {s: np.stack([pvr.next_phi(fam, b, n) for b, _, n in v]) for s, v in rows.items()}
    nb = pvr.VERIFY_SPEC[fam][0]
    mh = heads.MLPHead(seed=seed).fit(X["train"], Y["train"], X["dev"], Y["dev"], nb)
    resid = Y["dev"][:, nb:] - mh.predict(X["dev"])[:, nb:] if Y["dev"].shape[1] > nb else None
    sigma = np.maximum(resid.std(0), 0.05) if resid is not None else None

    def loglik(before, ev, phi):
        P = mh.predict(decide.symbolic(fam, before, [ev], kmax=1)[None])[0]
        p = np.clip(P[:nb], 1e-4, 1 - 1e-4)
        ll = float((phi[:nb] * np.log(p) + (1 - phi[:nb]) * np.log(1 - p)).sum())
        if sigma is not None:
            ll += float(-0.5 * (((phi[nb:] - P[nb:]) / sigma) ** 2).sum())
        return ll
    return loglik


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def analyze(a):
    src = ROOT / "runs" / a.bank / f"pvr_{a.arm}"
    recs = [json.loads(l) for l in (src / "stage1.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = json.loads((src / "stage1_meta.json").read_text(encoding="utf-8"))
    H = {a.arm: np.load(src / "stage1_h.npy")}
    for other in [x for x in a.verifier_arms.split(",") if x and x != a.arm]:
        p = src / f"hstates_{other}.npy"
        if p.exists():
            H[other] = np.load(p)
    report = {"meta": {k: meta[k] for k in ("layers", "n", "temp", "splits")}, "families": {}}
    for budget in [int(x) if x != "all" else None for x in a.budgets.split(",")]:
        btag = f"b{budget or 'all'}"
        vers = {}
        for armname in H:
            fixed = meta["layers"] if armname == a.arm else json.loads((src / f"hstates_{armname}_meta.json").read_text())["layers"]
            v = fit_verifiers(a.bank, armname, budget=budget, fixed_layers=fixed)
            vers[armname] = v
        for fam in FAMS:
            idx = [i for i, r in enumerate(recs) if r["family"] == fam]
            sv = s_verifier(a.bank, fam, budget=budget)
            methods = {m: [] for m in ["greedy", "sc", "logprob", "oracle"] + [f"pvr_{x}" for x in H] + ["pvr_S"]}
            if fam == "prerequisite_plan":
                methods.update({f"decode_{x}": [] for x in H})
            conf, conf_ok, splits, parents = [], [], [], []
            for i in idx:
                r = recs[i]
                cands = r["cands"]
                parsed = [(j, c) for j, c in enumerate(cands) if c["parsed"] is not None]
                pick = lambda j: cands[j]["correct"] if j is not None else False
                methods["greedy"].append(cands[0]["correct"])
                methods["oracle"].append(any(c["correct"] for c in cands))
                if parsed:
                    cnt = Counter(c["parsed"] for _, c in parsed)
                    top = max(cnt.values())
                    mode = next(c["parsed"] for _, c in parsed if cnt[c["parsed"]] == top)
                    methods["sc"].append(mode == r["gold"])
                    methods["logprob"].append(pick(max(parsed, key=lambda jc: jc[1]["mean_logprob"])[0]))
                else:
                    methods["sc"].append(False); methods["logprob"].append(False)
                uniq = {}
                for j, c in parsed:
                    uniq.setdefault(c["parsed"], j)
                phis = {s: pvr.next_phi(fam, r["before"], json.loads(s)) for s in uniq}
                for armname, v in vers.items():
                    L, ver, _ = v[fam]
                    post = ver.posterior(H[armname][i][None])
                    if uniq:
                        best = max(uniq, key=lambda s: ver.loglik(post, phis[s]))
                        methods[f"pvr_{armname}"].append(best == r["gold"])
                        if armname == a.arm:
                            conf.append(ver.loglik(post, phis[best])); conf_ok.append(best == r["gold"])
                    else:
                        methods[f"pvr_{armname}"].append(False)
                        if armname == a.arm:
                            conf.append(-1e9); conf_ok.append(False)
                    if fam == "prerequisite_plan":
                        methods[f"decode_{armname}"].append(env.dumps(pvr.p_state_from_bits(ver.argmax_state_bits(post))) == r["gold"])
                if uniq:
                    best = max(uniq, key=lambda s: sv(r["before"], r["event"], phis[s]))
                    methods["pvr_S"].append(best == r["gold"])
                else:
                    methods["pvr_S"].append(False)
                splits.append(r["split"]); parents.append(r["parent_id"])
            res = {}
            sp = np.array(splits)
            for split in ("dev", "test"):
                m_ = sp == split
                if not m_.any():
                    continue
                pids = sorted({p for p, k in zip(parents, m_) if k})
                pi = {p: n for n, p in enumerate(pids)}

                def parent_mean(v):
                    s_, n_ = np.zeros(len(pids)), np.zeros(len(pids))
                    for p, k, x in zip(parents, m_, v):
                        if k:
                            s_[pi[p]] += x; n_[pi[p]] += 1
                    return s_ / n_
                acc = {m: float(np.mean(np.array(v)[m_])) for m, v in methods.items()}
                pm = {m: parent_mean(np.array(v, float)) for m, v in methods.items()}
                con = {f"pvr_{a.arm} - {b}": decide.paired_bootstrap(pm[f"pvr_{a.arm}"], pm[b]) for b in methods if b != f"pvr_{a.arm}"}
                res[split] = {"n_steps": int(m_.sum()), "n_parents": len(pids), "acc": acc, "contrasts": con,
                              f"auroc_{a.arm}_conf": auroc(np.array(conf)[m_], np.array(conf_ok)[m_])}
            report["families"].setdefault(fam, {})[btag] = res
            t = res.get("test", res.get("dev"))
            print(f"[{btag}] {fam[:6]} " + " ".join(f"{m}={v:.3f}" for m, v in t["acc"].items()) + f" auroc={t[f'auroc_{a.arm}_conf']:.3f}", flush=True)
    (src / "analysis.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")


def stage2(a):
    """Closed-loop rollouts on --splits for each --methods (greedy|sc|logprob|pvr|retro)."""
    from rcwmt import generation, runtime
    bank = data.load_bank(a.bank)
    ver = fit_verifiers(a.bank, a.arm)
    layers = {f: ver[f][0] for f in FAMS}
    chains = chains_for(bank, set(a.splits.split(",")))
    methods = a.methods.split(",")
    tok = runtime.tokenizer()
    out = ROOT / "runs" / a.bank / f"pvr_{a.arm}"
    state = {m: [[c["s0"]] for c in chains] for m in methods}
    from rcwmt import capture
    retro_probes = {}
    if "retro" in methods:
        for fam in FAMS:
            z = np.load(out / f"retro_probe_{a.arm}_{fam}.npz")
            retro_probes[fam] = {k_: z[k_] for k_ in z.files}
    with runtime.gpu_lease(label=f"rcwm-t1:pvr2:{a.arm}", job=f"RCWM-T1 PVR stage2 {a.bank}"):
        model, _ = runtime.load_arm(a.arm)
        t0 = time.time()
        Hmax = max(len(c["events"]) for c in chains)
        for t in range(Hmax):
            for m in methods:
                for fam in FAMS:
                    ix = [i for i, c in enumerate(chains) if c["family"] == fam and state[m][i][-1] is not None]
                    L, v, _ = ver[fam]
                    for b in range(0, len(ix), a.batch):
                        bi = ix[b: b + a.batch]
                        res = generation.sample_batch(model, tok, [step_messages(fam, state[m][i][-1], chains[i]["events"][t]) for i in bi],
                                                      a.n, a.temp, generation.max_tokens(fam), L, seed=5000 + 97 * t + b)
                        for i, r in zip(bi, res):
                            before = state[m][i][-1]
                            parsed = [(c, env.parse_state(fam, c["text"])) for c in r["cands"]]
                            ok = [(c, s) for c, s in parsed if s is not None]
                            if not ok:
                                state[m][i].append(None)
                                continue
                            if m == "greedy":
                                s = parsed[0][1]
                            elif m == "sc":
                                cnt = Counter(env.dumps(s) for _, s in ok)
                                s = json.loads(max(cnt, key=lambda k: (cnt[k], -[env.dumps(x) for _, x in ok].index(k))))
                            elif m == "logprob":
                                s = max(ok, key=lambda cs: cs[0]["mean_logprob"])[1]
                            elif m == "retro":
                                uniq = {}
                                for c, st_ in ok:
                                    uniq.setdefault(env.dumps(st_), (c, st_))
                                keys = list(uniq)
                                pr = retro_probes[fam]
                                enc = []
                                for k_ in keys:
                                    text = capture.render(tok, step_messages(fam, before, chains[i]["events"][t])) + uniq[k_][0]["text"] + "<|im_end|>"
                                    ids_ = tok(text, add_special_tokens=False)["input_ids"]
                                    enc.append({"ids": ids_, "last": len(ids_) - 1, "events": []})
                                Hc, _ = capture.capture(model, enc, layers=(int(pr["layer"]),), log_every=0)
                                Z = (Hc[:, 0].astype(np.float32) - pr["mu"]) / pr["sd"]
                                sc_ = Z @ pr["W"][:, 0] + pr["b"][0]
                                s = uniq[keys[int(np.argmax(sc_))]][1]
                            elif m == "pvr":
                                post = v.posterior(r["h"][None])
                                s = max(ok, key=lambda cs: v.loglik(post, pvr.next_phi(fam, before, cs[1])))[1]
                            else:
                                raise ValueError(m)
                            state[m][i].append(s)
                print(f"  t={t} done ({time.time() - t0:.0f}s)", flush=True)
            for m in methods:
                for i in range(len(chains)):
                    if len(state[m][i]) == t + 1:
                        state[m][i].append(None)
        runtime.unload(model)
    with (out / f"stage2_{a.splits.replace(',', '_')}.jsonl").open("w", encoding="utf-8") as f:
        for i, c in enumerate(chains):
            f.write(json.dumps({"cand_id": c["cand_id"], **{m: [None if s is None else env.dumps(s) for s in state[m][i]] for m in methods}}) + "\n")
    print(f"stage2 done in {time.time() - t0:.0f}s", flush=True)


RETRO_LAYERS = (8, 12, 16, 20, 24, 28)


def retro_pairs(recs):
    """(record index, candidate text) for every unique parsed candidate of every stage-1 step."""
    pairs = []
    for i, r in enumerate(recs):
        seen = set()
        for c in r["cands"]:
            if c["parsed"] is None or c["parsed"] in seen:
                continue
            seen.add(c["parsed"])
            pairs.append({"rec": i, "text": c["text"], "parsed": c["parsed"], "correct": c["parsed"] == r["gold"]})
    return pairs


def retro(a):
    """GPU. Retrospective verifier features: teacher-force prompt + candidate answer + <|im_end|>
    with --arm and record the residual stream at the <|im_end|> token (the model has now READ the
    whole proposed next state). Candidates are the stage-1 samples of --source-arm."""
    from rcwmt import capture, runtime
    src = ROOT / "runs" / a.bank / f"pvr_{a.source_arm}"
    recs = [json.loads(l) for l in (src / "stage1.jsonl").read_text(encoding="utf-8").splitlines()]
    pairs = retro_pairs(recs)
    tok = runtime.tokenizer()
    enc = []
    for p in pairs:
        r = recs[p["rec"]]
        text = capture.render(tok, step_messages(r["family"], r["before"], r["event"])) + p["text"] + "<|im_end|>"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        enc.append({"ids": ids, "last": len(ids) - 1, "events": []})
    print(f"retro {a.arm}: {len(pairs)} candidate sequences from {len(recs)} steps", flush=True)
    with runtime.gpu_lease(label=f"rcwm-t1:pvrr:{a.arm}", job=f"RCWM-T1 PVR retro features {a.arm}"):
        model, prov = runtime.load_arm(a.arm)
        last, _ = capture.capture(model, enc, layers=RETRO_LAYERS, log_every=100)
        runtime.unload(model)
    np.save(src / f"retro_{a.arm}.npy", last)
    with (src / f"retro_{a.arm}_pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    (src / f"retro_{a.arm}_meta.json").write_text(json.dumps({"layers": RETRO_LAYERS, "provenance": prov}, default=str) + "\n", encoding="utf-8")
    print(f"retro {a.arm}: saved {last.shape}", flush=True)


def retro_analyze(a):
    """CPU. Train a logistic correctness probe on DEV-split candidates (layer chosen by 5-fold
    parent-grouped CV inside DEV), then on TEST pick, per step, the candidate with the highest
    P(correct). Compared with greedy / self-consistency / logprob / oracle on the same samples."""
    src = ROOT / "runs" / a.bank / f"pvr_{a.source_arm}"
    recs = [json.loads(l) for l in (src / "stage1.jsonl").read_text(encoding="utf-8").splitlines()]
    report = {}
    for armname in a.verifier_arms.split(","):
        pf = src / f"retro_{armname}.npy"
        if not pf.exists():
            continue
        X = np.load(pf).astype(np.float32)
        pairs = [json.loads(l) for l in (src / f"retro_{armname}_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
        layers = json.loads((src / f"retro_{armname}_meta.json").read_text(encoding="utf-8"))["layers"]
        rep = {}
        for fam in FAMS:
            idx = [n for n, p in enumerate(pairs) if recs[p["rec"]]["family"] == fam]
            dev = [n for n in idx if recs[pairs[n]["rec"]]["split"] == "dev"]
            test = [n for n in idx if recs[pairs[n]["rec"]]["split"] == "test"]
            y = np.array([pairs[n]["correct"] for n in range(len(pairs))], dtype=np.float32)
            # layer by grouped CV on DEV
            groups = np.array([recs[pairs[n]["rec"]]["parent_id"] for n in dev])
            ug = np.unique(groups)
            fold = {g: k_ % 5 for k_, g in enumerate(np.random.default_rng(17).permutation(ug))}
            best = None
            for li, L in enumerate(layers):
                Xd = X[dev, li]
                lls = []
                for f_ in range(5):
                    tr = np.array([fold[g] != f_ for g in groups]); va = ~tr
                    if y[dev][tr].min() == y[dev][tr].max():
                        continue
                    pr = pvr.LogisticProbe().fit(Xd[tr], y[dev][tr, None], Xd[va], y[dev][va, None])
                    p_ = np.clip(pr.proba(Xd[va])[:, 0], 1e-4, 1 - 1e-4)
                    lls.append(float(np.mean(y[dev][va] * np.log(p_) + (1 - y[dev][va]) * np.log(1 - p_))))
                if lls and (best is None or np.mean(lls) > best[0]):
                    best = (np.mean(lls), li, L)
            _, li, L = best
            dev_a = np.array(dev)
            hold = np.array([fold[g] == 0 for g in groups])          # parent-grouped holdout picks C
            pr = pvr.LogisticProbe().fit(X[dev_a[~hold], li], y[dev_a[~hold], None], X[dev_a[hold], li], y[dev_a[hold], None])
            score = {n: float(pr.proba(X[n:n + 1, li])[0, 0]) for n in test}
            np.savez(src / f"retro_probe_{armname}_{fam}.npz", W=pr.W.numpy(), b=pr.b.numpy(), mu=pr.std.mu, sd=pr.std.sd,
                     layer=np.array(L), C=np.array(pr.C))
            # per-step selection on TEST
            by_rec = {}
            for n in test:
                by_rec.setdefault(pairs[n]["rec"], []).append(n)
            acc, conf, ok = {"retro": [], "greedy": [], "sc": [], "logprob": [], "oracle": []}, [], []
            parents = []
            for ri in [i for i, r_ in enumerate(recs) if r_["family"] == fam and r_["split"] == "test"]:
                r = recs[ri]
                cands = r["cands"]
                parsed = [c for c in cands if c["parsed"] is not None]
                acc["greedy"].append(cands[0]["correct"])
                acc["oracle"].append(any(c["correct"] for c in cands))
                if parsed:
                    cnt = Counter(c["parsed"] for c in parsed)
                    mode = max(cnt, key=lambda k_: (cnt[k_], -[c["parsed"] for c in parsed].index(k_)))
                    acc["sc"].append(mode == r["gold"])
                    acc["logprob"].append(max(parsed, key=lambda c: c["mean_logprob"])["correct"])
                else:
                    acc["sc"].append(False); acc["logprob"].append(False)
                ns = by_rec.get(ri, [])
                if ns:
                    nb = max(ns, key=lambda n: score[n])
                    acc["retro"].append(pairs[nb]["correct"]); conf.append(score[nb]); ok.append(pairs[nb]["correct"])
                else:
                    acc["retro"].append(False); conf.append(0.0); ok.append(False)
                parents.append(r["parent_id"])
            pids = sorted(set(parents)); pi = {p: k_ for k_, p in enumerate(pids)}

            def pm(v):
                s_, c_ = np.zeros(len(pids)), np.zeros(len(pids))
                for p, x in zip(parents, v):
                    s_[pi[p]] += x; c_[pi[p]] += 1
                return s_ / c_
            res = {"layer": L, "n_steps": len(parents), "n_parents": len(pids),
                   "acc": {m: float(np.mean(v)) for m, v in acc.items()},
                   "auroc_selected": auroc(conf, ok),
                   "auroc_all_candidates": auroc([score[n] for n in test], [pairs[n]["correct"] for n in test]),
                   "contrasts": {f"retro - {m}": decide.paired_bootstrap(pm(np.array(acc["retro"], float)), pm(np.array(v, float)))
                                 for m, v in acc.items() if m != "retro"}}
            rep[fam] = res
            print(f"[retro:{armname}] {fam[:6]} L{L} " + " ".join(f"{m}={v:.3f}" for m, v in res["acc"].items())
                  + f" auroc_all={res['auroc_all_candidates']:.3f}", flush=True)
        report[armname] = rep
    (src / "retro_analysis.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stage1", "hstates", "analyze", "stage2", "retro", "retro_analyze"])
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--arm", default="sparse")
    ap.add_argument("--source-arm", default="sparse")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--verifier-arms", default="sparse,supa")
    ap.add_argument("--budgets", default="all")
    ap.add_argument("--methods", default="greedy,sc,logprob,pvr")
    a = ap.parse_args()
    {"stage1": stage1, "hstates": hstates, "analyze": analyze, "stage2": stage2, "retro": retro, "retro_analyze": retro_analyze}[a.mode](a)


if __name__ == "__main__":
    main()
