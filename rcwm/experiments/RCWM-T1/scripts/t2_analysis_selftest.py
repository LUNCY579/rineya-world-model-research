"""Selftest for scripts/t2_analysis.py (CPU only, numpy/json; never loads a model).

  python scripts/t2_analysis_selftest.py

Builds three synthetic run dirs under runs/t2/_fixtures/ from the real data files (t2_eval, xframe, supp,
the pilot item list rebuilt from bank_pilot) and the stored Titan-Sparse pilot artefacts, runs the analysis
on each and on all three together, and asserts the expected outcomes.

  null           t2_text == base_text == Titan-like text: P copies `before`; L applies the event ignoring
                 its status; R does swap right, add without mod 97, mix with 1x other, conditional always
                 adds. Chains: stored Titan-Sparse chains for OFF and ON. Off-task: stored log-probs and
                 identical placeholder replies for OFF and ON.
                 -> every delta is 0, every improvement gate FAILS (the non-regression gates G2/G4/G8 pass
                    trivially because nothing changed), rule learned FAILS.
  perfect        t2_text = target, base_text = copy of before. Chains: ON gold states, OFF stored Titan.
                 Off-task: ON == OFF. -> G1-G7 PASS, false-copy falls to 0, rule learned PASS (or PENDING
                 while the Supa off-task reference does not exist).
  copy_collapse  t2_text = copy of before, base_text = Titan-like. Chains: ON copy chains, OFF stored Titan.
                 Off-task: ON log-probs perturbed, half of the ON replies are JSON states.
                 -> void / inapplicable / pre_off-only cells reach 1.0, but app-changed, the minimal pairs,
                    the untouched cells and the false-copy rate expose the collapse; rule learned FAILS.
                    (G5 PASSES: the void gains do carry over to pilot steps 2-4 and |balance|>99; G5 is a
                    carry-over test, the collapse is caught by G1/G2/G3/G4/G6/G7.)
Also checks the statistics helpers against scipy, the file formats against the CPU smoke run
(runs/t2/smoke_cpu, zero-init adapter), determinism (two runs give byte-identical JSON), runtime, and:
  * G5 semantics on gen-only run dirs: per-op gains that do not carry over to the pilot FAIL; a
    significant per-op drop in an untouched stratum that the pilot does not repeat does not affect G5;
  * G4 point-estimate rule: one extra copied row fails;
  * input validation: reordered / missing-text / unreadable gen rows, an edited or incomplete eval.json,
    truncated / duplicated / cross-parent / tampered chain files, a truncated offtask_gen.jsonl, wrong
    off-task meta ids and misaligned substrata/xframe rows all stop the analysis (SystemExit);
  * report text: claim-scope caveats and the provisional-seed note for a single tag; xframe P labels.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RCWM_THREADS"):
    os.environ[_v] = "2"

import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t2_analysis", ROOT / "scripts" / "t2_analysis.py")
t2a = importlib.util.module_from_spec(spec)
sys.modules["t2_analysis"] = t2a
spec.loader.exec_module(t2a)
env = t2a.env

FIX = ROOT / "runs" / "t2" / "_fixtures"
NAMES = ("null", "perfect", "copy_collapse")
FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL:", msg, flush=True)


# ---------------------------------------------------------------- synthetic texts


def titan_like(fam, before, ev):
    if fam == "prerequisite_plan":
        return env.dumps(before)
    if fam == "ledger_replay":
        return env.dumps(env.step(fam, before, {**ev, "status": "posted"}))
    reg, k, r, o, v = dict(before), ev["kind"], ev["register"], ev["other"], ev["value"]
    if k == "swap":
        reg[r], reg[o] = reg[o], reg[r]
    elif k == "add":
        reg[r] = reg[r] + v
    elif k == "mix":
        reg[r] = (reg[r] + reg[o] + v) % env.MOD
    else:
        reg[r] = (reg[r] + v) % env.MOD
    return env.dumps(reg)


def texts(name, r):
    copy, titan = env.dumps(r["before"]), titan_like(r["family"], r["before"], r["event"])
    if name == "null":
        return titan, titan
    if name == "perfect":
        return copy, r["target"]
    return titan, copy


# ---------------------------------------------------------------- fixture builder


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build(name, S, stored_chain_lines):
    d = FIX / name
    if d.exists():
        shutil.rmtree(d)
    (d / "off").mkdir(parents=True)
    sets = {"t2_eval": S.data["t2_eval"], "xframe": S.data["xframe"], "supp": S.data["supp"], "pilot": S.pilot_ref}
    ej = {"tag": name, "fixture": True, "n": {k: len(v) for k, v in sets.items()}}
    for setname, rows in sets.items():
        out = []
        for i, r in enumerate(rows):
            b, t = texts(name, r)
            out.append({"i": i, **{k: v for k, v in r.items() if k != "prompt"}, "base_text": b, "t2_text": t})
        write_jsonl(d / f"gen_{setname}.jsonl", out)
        ej[f"per_op_{setname}_base"] = t2a.per_op_summary(out, [x["base_text"] for x in out])
        ej[f"per_op_{setname}_t2"] = t2a.per_op_summary(out, [x["t2_text"] for x in out])
    ej["per_op_base"], ej["per_op_t2"] = ej["per_op_t2_eval_base"], ej["per_op_t2_eval_t2"]
    ej["provenance"] = {"base_fingerprint": "fixture", "adapter": f"fixture:{name}",
                        "data_sha256": {k: t2a.sha256(ROOT / "data" / f"{k}.jsonl") for k in ("t2_train", "t2_eval", "t2_eval_xframe", "t2_eval_supp")}}
    (d / "eval.json").write_text(json.dumps(ej, indent=1) + "\n", encoding="utf-8")
    # chains
    test = [(p, c) for p in S.bank["parents"] if p["split"] == "test" for c in p["candidates"]]
    for mode in t2a.MODES:
        (d / f"gen_{mode}_base.jsonl").write_text(stored_chain_lines[mode], encoding="utf-8")
        if name == "null":
            (d / f"gen_{mode}.jsonl").write_text(stored_chain_lines[mode], encoding="utf-8")
            continue
        rows = []
        for p, c in test:
            if name == "perfect":
                st = [env.dumps(s) for s in env.rollout(p["family"], p["s0"], c["events"])]
            else:
                st = [env.dumps(p["s0"])] * (len(c["events"]) + 1)
            rows.append({"cand_id": c["cand_id"], "states": st, "texts": st[1:]})
        write_jsonl(d / f"gen_{mode}.jsonl", rows)
    # off-task
    shutil.copyfile(t2a.STORED_SPARSE / "offtask_logprobs.npy", d / "off" / "offtask_logprobs.npy")
    shutil.copyfile(t2a.STORED_SPARSE / "offtask_meta.json", d / "off" / "offtask_meta.json")
    shutil.copyfile(t2a.STORED_SPARSE / "offtask_meta.json", d / "offtask_meta.json")
    if name != "copy_collapse":
        shutil.copyfile(t2a.STORED_SPARSE / "offtask_logprobs.npy", d / "offtask_logprobs.npy")
    else:
        src = np.load(t2a.STORED_SPARSE / "offtask_logprobs.npy", mmap_mode="r")
        dst = np.lib.format.open_memmap(d / "offtask_logprobs.npy", mode="w+", dtype=np.float16, shape=src.shape)
        rng = np.random.default_rng(7)
        for s in range(0, src.shape[0], 8):
            a = np.asarray(src[s:s + 8]).astype(np.float64) + rng.normal(0, 0.7, (min(8, src.shape[0] - s), src.shape[1]))
            dst[s:s + 8] = np.maximum(t2a._lsm(a), -60.0).astype(np.float16)
        dst.flush()
        del dst
    prompts = t2a.load_jsonl(ROOT / "data" / "offtask_prompts.jsonl")
    og = []
    for i, r in enumerate(prompts):
        base = "42" if r["reply_class"] in t2a.NUMERIC_CLASSES else "Sure - here is a short answer."
        t2 = '{"A":1,"B":2,"C":3,"D":4}' if (name == "copy_collapse" and i % 2 == 0) else base
        og.append({"id": r["id"], "reply_class": r["reply_class"], "base_text": base, "t2_text": t2})
    write_jsonl(d / "offtask_gen.jsonl", og)


# ---------------------------------------------------------------- assertions


def all_cells(r):
    out = []
    for blk in ("A", "C"):
        out += r[blk]["cells"] + list(r[blk]["by_stratum"].values())
    out += r["B"]["cells"] + r["B"]["pooled"] + list(r["B"]["g5_cells"].values())
    return [c for c in out if c["n"]]


def cell(r, cid, blk="A"):
    return next(c for c in r[blk]["cells"] if c["id"] == cid)


def assert_null(r):
    G = r["gates"]
    for c in all_cells(r):
        check(c["delta"] == 0 and c["b"] == 0 and c["c"] == 0, f"null: nonzero delta in {c['id']}")
    for blk in ("A", "B", "C"):
        for f, x in r[blk]["false_copy"].items():
            check(x.get("delta", 0) == 0, f"null: false-copy delta {blk} {f}")
    for a in r["D"]["acc"]:
        check(a["delta"] == 0, f"null: chain delta {a['mode']} {a['fam']} k{a['k']}")
    for k, v in r["D"]["repro"].items():
        check(v["identical_chains"] == 1.0, f"null: OFF chains not identical to stored {k}")
    for k, e in r["E"].items():
        check(e["drop"] == 0 and e["drop_ci95"] == [0.0, 0.0], f"null: regret drop {k} = {e['drop']}")
        check(abs(e["off"] - e["stored_sparse"]) < 1e-12, f"null: OFF regret != stored {k}")
    # stored Titan-Sparse reference numbers of docs/t2_review/evalvalidity_check.log (same g_only call pattern)
    review = {"L|chain_one": 0.507, "L|chain_m1": 0.509, "P|chain_one": 0.879, "P|chain_m1": 0.878, "R|chain_one": 0.386, "R|chain_m1": 0.392}
    for k, v in review.items():
        check(round(r["E"][k]["stored_sparse"], 3) == v, f"stored regret {k} {r['E'][k]['stored_sparse']:.4f} != review {v}")
    acc_review = {("chain_one", "L"): (0.399, 0.202, 0.024), ("chain_one", "P"): (0.210, 0.049, 0.000), ("chain_one", "R"): (0.606, 0.361, 0.087),
                  ("chain_m1", "L"): (0.389, 0.192, 0.024), ("chain_m1", "P"): (0.196, 0.036, 0.018), ("chain_m1", "R"): (0.611, 0.365, 0.087)}
    for (m, f), v in acc_review.items():
        got = tuple(round(next(a["off"] for a in r["D"]["acc"] if a["mode"] == m and a["fam"] == f and a["k"] == k), 3) for k in (1, 2, 4))
        check(got == v, f"stored chain accuracy {m} {f}: {got} != review {v}")
    cf = {("chain_one", "P"): (0.612, 0.451, 0.308), ("chain_one", "L"): (0.144, 0.010, 0.000), ("chain_one", "R"): (0.038, 0.034, 0.000)}
    for (m, f), v in cf.items():
        got = tuple(round(next(a["copy_floor"] for a in r["D"]["acc"] if a["mode"] == m and a["fam"] == f and a["k"] == k), 3) for k in (1, 2, 4))
        check(got == v, f"copy-chain floor {m} {f}: {got} != review {v}")
    F = r["F"]
    check(F["kl_off_on"]["mean"] == 0 and F["kl_noise_stored_off"]["mean"] == 0, "null: KL not 0")
    check(F["top1_agree_off_on"] == 1.0 and F["greedy_exact_agree"]["rate"] == 1.0, "null: agreement not 1")
    for g in t2a.IMPROVEMENT_GATES:
        check(G[g]["status"] == "FAIL", f"null: improvement gate {g} is {G[g]['status']}")
    for g in ("G2_untouched", "G4_false_copy"):
        check(G[g]["status"] == "PASS", f"null: non-regression gate {g} is {G[g]['status']}")
    check(G["G8_forgetting"]["status"] in ("PASS", "PENDING"), "null: G8")
    check(G["rule_learned"]["status"] == "FAIL", "null: rule learned not FAIL")
    check(G["G4_false_copy"]["spare_rows"] == {"L": 0, "P": 0, "R": 0}, f"null: G4 spare rows {G['G4_false_copy'].get('spare_rows')}")
    check(not G["G5_pilot_direction"]["rule_strata_gained"], "null: no rule stratum may gain")
    # stored-Titan-like baselines: L posted right, void wrong; P inapplicable right (copy)
    check(cell(r, "L.posted")["off"] == 1.0 and cell(r, "L.void_disc")["off"] == 0.0, "null: Titan-like ledger texts")


def assert_perfect(r):
    G = r["gates"]
    for g in ("G1_targeted", "G2_untouched", "G3_app_del", "G4_false_copy", "G5_pilot_direction", "G6_chains_k4", "G7_decision"):
        check(G[g]["status"] == "PASS", f"perfect: {g} is {G[g]['status']}: {G[g]['detail']}")
    check(G["G8_forgetting"]["status"] in ("PASS", "PENDING"), "perfect: G8")
    check(G["rule_learned"]["status"] == G["G8_forgetting"]["status"], "perfect: rule learned != G8 status")
    for c in r["A"]["cells"]:
        if c["n"]:
            check(c["on"] == 1.0, f"perfect: {c['id']} ON {c['on']}")
    for f, x in r["A"]["false_copy"].items():
        check(x["off"] == 1.0 and x["on"] == 0.0, f"perfect: false-copy {f} {x['off']}->{x['on']}")
    for a in r["D"]["acc"]:
        check(a["on"] == 1.0, f"perfect: chain ON {a['mode']} {a['fam']} k{a['k']} = {a['on']}")
    for f, p in r["D"]["predicted_k4"].items():
        check(p["gold_path_product"]["on"] == 1.0 and p["product_of_step_acc"]["on"] == 1.0, f"perfect: predicted k4 {f}")
        check(p["chains_without_pilot_items"] == 0, f"perfect: chains without pilot items {f}")
    for k, e in r["E"].items():
        check(e["on"] == 0.0, f"perfect: ON regret {k} = {e['on']}")
    check(r["F"]["kl_off_on"]["mean"] == 0, "perfect: KL not 0")
    check(cell(r, "P.app_del")["off"] == 0.0 and cell(r, "P.app_del")["delta_ci95"][0] > 0, "perfect: app-del")
    g5 = G["G5_pilot_direction"]
    check(g5["checks"] and all(c["cell"].rsplit(" ", 1)[0].split(" ", 1)[1] in t2a.RULE_STRATA[c["cell"][0]] for c in g5["checks"]),
          "perfect: G5 must check rule strata only")
    check(all(c["why_not_gated"].startswith("untouched") for c in g5["descriptive"]) and g5["descriptive"],
          "perfect: the untouched-strata gains must be descriptive only")
    # labels: xframe P (one-step frame presumes applicability) and chain_one P inapplicable steps
    xc = {c["id"]: c for c in r["C"]["cells"]}
    check(xc["P.applicable"]["role"] == "TRANSFER" and xc["P.all"]["role"].startswith("descriptive")
          and xc["P.inapp_disc"]["role"].startswith("descriptive-only"), "perfect: xframe P labels")
    check(r["C"]["by_stratum"]["P|inapplicable"]["role"].startswith("descriptive-only")
          and r["C"]["by_stratum"]["P|applicable"]["role"] == "TRANSFER", "perfect: xframe P stratum labels")
    check(r["D"]["p_steps"]["chain_one"]["inapplicable"]["role"].startswith("descriptive-only")
          and r["D"]["p_steps"]["chain_one"]["applicable, state-changing"]["role"] == "TRANSFER"
          and r["D"]["p_steps"]["chain_m1"]["inapplicable"]["role"] == "primary", "perfect: P chain-step labels")


def assert_copy_collapse(r):
    G = r["gates"]
    for cid in ("L.void_disc", "P.inapp_disc", "P.pre_off_only", "L.void_nondisc", "P.app_noop", "P.inapp_nondisc", "R.cond_v0"):
        check(cell(r, cid)["on"] == 1.0, f"copy: {cid} ON {cell(r, cid)['on']} (copy should look good here)")
    for cid in ("P.app_changed", "P.app_addonly", "P.app_del", "L.pairs", "P.pairs", "R.pairs", "R.add_wrap", "R.mix_w0", "R.mix_w1", "R.mix_w2"):
        c = cell(r, cid)
        check(c["on"] == c["copy_floor"] and c["on"] <= 0.05, f"copy: {cid} ON {c['on']} copy {c['copy_floor']} (should expose the copy)")
    for cid in ("P.app_changed", "L.pairs", "P.pairs", "R.pairs"):
        check(cell(r, cid)["on"] == 0.0, f"copy: {cid} ON must be exactly 0")
    check(cell(r, "L.posted")["delta"] < -0.5, "copy: L posted should collapse")
    for f, x in r["A"]["false_copy"].items():
        check(x["on"] == 1.0, f"copy: false-copy ON {f} = {x['on']}")
        if f != "P":     # Titan-like P text is itself a copy, so P OFF is 1.0 as well
            check(x["on"] > x["off"], f"copy: false-copy {f} must rise {x['off']}->{x['on']}")
    for g in ("G1_targeted", "G2_untouched", "G3_app_del", "G4_false_copy", "G6_chains_k4", "G7_decision"):
        check(G[g]["status"] == "FAIL", f"copy: {g} is {G[g]['status']}")
    check(G["rule_learned"]["status"] == "FAIL", "copy: rule learned not FAIL")
    # G5 is a carry-over test: the void gains carry over (PASS); the posted / swap / add-nowrap / conditional-gt
    # collapses are significant per-op DROPS or untouched strata, so they are listed but not gated by G5.
    g5 = G["G5_pilot_direction"]
    check(g5["status"] == "PASS" and sorted(g5["rule_strata_gained"]) == ["L add/void", "L scale/void", "L transfer/void"],
          f"copy: G5 {g5['status']} {g5['rule_strata_gained']}")
    whys = {c["cell"].rsplit(" ", 1)[0]: c["why_not_gated"] for c in g5["descriptive"]}
    check(whys.get("L add/posted", "").startswith("untouched") and whys.get("R conditional/gt", "").startswith("per-op drop"),
          f"copy: G5 descriptive rows {whys}")
    check(G["G4_false_copy"]["spare_rows"]["L"] == -151, "copy: G4 margin")
    p_m1 = next(a for a in r["D"]["acc"] if a["mode"] == "chain_m1" and a["fam"] == "P" and a["k"] == 4)
    check(abs(p_m1["on"] - p_m1["copy_floor"]) < 1e-12, "copy: P chain_m1 k4 ON should equal the copy floor")
    F = r["F"]
    check(F["kl_off_on"]["mean"] > 0.01 and F["top1_agree_off_on"] < 1.0, "copy: KL should be > 0")
    check(F["json_bleed"]["on"]["bleed_rate_nonnumeric"] > 0.3 and F["json_bleed"]["off"]["bleed_rate_nonnumeric"] == 0, "copy: JSON bleed")
    check(F["greedy_exact_agree"]["rate"] == 0.5, "copy: greedy agreement should be 0.5")


# ---------------------------------------------------------------- stats + format checks


def stats_checks():
    from scipy import stats
    for b, c in ((0, 6), (1, 9), (5, 5), (3, 20), (0, 0), (12, 40)):
        ref = 1.0 if b + c == 0 else stats.binomtest(min(b, c), b + c, 0.5).pvalue
        check(abs(t2a.mcnemar_exact(b, c) - ref) < 1e-12, f"mcnemar {b},{c}: {t2a.mcnemar_exact(b, c)} vs {ref}")
    check(abs(t2a.mcnemar_exact(0, 6) - 0.03125) < 1e-15, "mcnemar 0,6")
    for k, n in ((0, 40), (20, 40), (36, 40), (40, 40), (3, 22)):
        ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        got = t2a.wilson(k, n)
        check(abs(got[0] - ref.low) < 1e-9 and abs(got[1] - ref.high) < 1e-9, f"wilson {k}/{n}: {got} vs {ref}")
    rng = np.random.default_rng(3)
    for e, f, g, h in ((20, 10, 4, 6), (5, 30, 0, 5), (40, 0, 0, 0), (0, 0, 0, 40), (10, 10, 10, 10), (0, 40, 0, 0)):
        on = np.array([1] * e + [1] * f + [0] * g + [0] * h, bool)
        off = np.array([1] * e + [0] * f + [1] * g + [0] * h, bool)
        lo, hi = t2a.newcombe_paired(on, off)
        d = on.mean() - off.mean()
        check(-1 <= lo <= d + 1e-12 and d - 1e-12 <= hi <= 1, f"newcombe {e,f,g,h}: [{lo},{hi}] around {d}")
        if 0 < f + g < len(on):
            bs = (on.astype(float) - off)[rng.integers(0, len(on), (20000, len(on)))].mean(1)
            blo, bhi = np.quantile(bs, [0.025, 0.975])
            check(abs(lo - blo) < 0.06 and abs(hi - bhi) < 0.06, f"newcombe {e,f,g,h}: [{lo:.3f},{hi:.3f}] vs bootstrap [{blo:.3f},{bhi:.3f}]")
    # coverage of the Newcombe interval under a known paired model (independent check of the formula)
    for probs in ((0.5, 0.2, 0.1, 0.2), (0.3, 0.05, 0.4, 0.25), (0.7, 0.1, 0.1, 0.1)):
        n, sims, hit = 40, 4000, 0
        true_d = probs[1] - probs[2]
        cells = rng.choice(4, size=(sims, n), p=probs)
        for row in cells:
            on = (row == 0) | (row == 1)
            off = (row == 0) | (row == 2)
            lo, hi = t2a.newcombe_paired(on, off)
            hit += lo <= true_d <= hi
        check(0.925 <= hit / sims <= 0.985, f"newcombe coverage {probs}: {hit / sims:.3f}")
    lo, hi = t2a.newcombe_paired(np.ones(40, bool), np.ones(40, bool))
    check(lo < 0 < hi or (lo <= 0 <= hi), "newcombe all-concordant must contain 0")
    x = np.r_[np.ones(50), np.zeros(50)]
    lo, hi = t2a.cluster_boot(x, [f"p{i // 10}" for i in range(100)])
    check(lo < 0.5 < hi and hi - lo > 0.3, f"cluster bootstrap must widen for perfectly clustered data: [{lo},{hi}]")
    check(t2a.looks_state('{"A":1,"B":2,"C":3,"D":4}') and t2a.looks_state("Answer: {\"X\": 3, \"Y\": 1}")
          and not t2a.looks_state("Sure, tomorrow works.") and t2a.int_only("17") and not t2a.int_only("17 minutes"), "bleed classifier")


def smoke_format_checks(S):
    sm = ROOT / "runs" / "t2" / "smoke_cpu"
    if not sm.exists():
        print("  (smoke_cpu absent; format check skipped)")
        return
    ref = {"t2_eval": S.data["t2_eval"], "xframe": S.data["xframe"], "supp": S.data["supp"], "pilot": S.pilot_ref}
    for name, rows in ref.items():
        p = sm / f"gen_{name}.jsonl"
        if not p.exists():
            continue
        g = t2a.load_jsonl(p)
        errs = t2a.check_rows(g, rows[:len(g)], f"smoke {name}")
        check(not errs, f"smoke {name} rows differ from the reference rebuild: {errs}")
        check(all({"base_text", "t2_text", "i"} <= set(x) for x in g), f"smoke {name}: missing text fields")
    for p in sorted(sm.glob("gen_chain_*.jsonl")):
        for x in t2a.load_jsonl(p):
            check(set(x) == {"cand_id", "states", "texts"} and len(x["states"]) == len(x["texts"]) + 1, f"smoke chain format {p.name}")


# ---------------------------------------------------------------- G5 / G4 semantics and input validation


def fresh(name):
    d = FIX / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def write_gen(d, setname, rows, text_fn):
    write_jsonl(d / f"gen_{setname}.jsonl",
                [{"i": i, **{k: v for k, v in r.items() if k != "prompt"}, **dict(zip(("base_text", "t2_text"), text_fn(r)))}
                 for i, r in enumerate(rows)])


def g5_semantics(S):
    """Gen-only run dirs (no eval.json, chains or off-task: G6-G8 PENDING) probing the G5 rule."""
    sets = {"t2_eval": S.data["t2_eval"], "xframe": S.data["xframe"], "supp": S.data["supp"]}
    # (a) per-op gains that do not carry over: one-step perfect, pilot ON == OFF -> G5 FAIL (zero pilot delta fails)
    d = fresh("_g5_nocarry")
    for s, rows in sets.items():
        write_gen(d, s, rows, lambda r: (env.dumps(r["before"]), r["target"]))
    write_gen(d, "pilot", S.pilot_ref, lambda r: (env.dumps(r["before"]), env.dumps(r["before"])))
    G = t2a.analyse_seed(d, S)["gates"]
    g5 = G["G5_pilot_direction"]
    check(g5["status"] == "FAIL" and g5["checks"] and all(not c["ok"] and c["pilot_delta"] == 0 for c in g5["checks"]),
          f"G5 no-carry-over: {g5['status']} {g5['detail']}")
    check([G[g]["status"] for g in ("G6_chains_k4", "G7_decision", "G8_forgetting")] == ["PENDING"] * 3, "G5 no-carry-over: G6-G8 pending")
    shutil.rmtree(d)
    print(f"  no carry-over: G5 {g5['status']} ({g5['detail'][:90]})", flush=True)

    # (b) rule gains carry over; L posted collapses per-op but not on the pilot -> G5 PASS (G2 FAILS instead)
    def tx(r, pilot):
        titan = titan_like(r["family"], r["before"], r["event"])
        posted = r["family"] == "ledger_replay" and r["stratum"].endswith("/posted")
        if posted:
            return titan, (titan if pilot else env.dumps(r["before"]))
        return titan, r["target"]
    d = fresh("_g5_untouched_drop")
    for s, rows in sets.items():
        write_gen(d, s, rows, lambda r: tx(r, False))
    write_gen(d, "pilot", S.pilot_ref, lambda r: tx(r, True))
    G = t2a.analyse_seed(d, S)["gates"]
    g5 = G["G5_pilot_direction"]
    post = [c for c in g5["descriptive"] if "/posted" in c["cell"]]
    check(g5["status"] == "PASS" and G["G2_untouched"]["status"] == "FAIL", f"G5 untouched drop: G5 {g5['status']} G2 {G['G2_untouched']['status']}")
    check(len(post) == 6 and all(c["pilot_delta"] == 0 and c["perop_delta"] < 0 for c in post), "G5 untouched drop: posted rows descriptive")
    check(not any("/posted" in c["cell"] for c in g5["checks"]), "G5 untouched drop: posted must not be gated")
    shutil.rmtree(d)
    print(f"  untouched drop not repeated on pilot: G5 {g5['status']}, G2 {G['G2_untouched']['status']}", flush=True)


def g4_one_row(res):
    r = copy.deepcopy(res)
    x = r["A"]["false_copy"]["L"]
    x["on_copies"] += 1
    x["on_new_copies"] += 1
    x["on"] = x["on_copies"] / x["n_target_ne_before"]
    G = t2a.gates(r)
    check(G["G4_false_copy"]["status"] == "FAIL" and "over by 1 row" in G["G4_false_copy"]["detail"],
          f"G4 one extra copy must FAIL: {G['G4_false_copy']['detail']}")
    print(f"  G4 with one extra L copy: {G['G4_false_copy']['status']}", flush=True)


def expect_exit(name, setup, phrase, S):
    d = fresh(f"_adv_{name}")
    try:
        setup(d)
        try:
            t2a.analyse_seed(d, S)
            check(False, f"validation {name}: analysis did not stop")
        except SystemExit as e:
            check(phrase in str(e), f"validation {name}: message lacks {phrase!r}: {str(e)[:300]}")
    finally:
        shutil.rmtree(d)


def validation_checks(S):
    src = FIX / "perfect"
    lines = lambda f: (src / f).read_text(encoding="utf-8").splitlines(keepends=True)  # noqa: E731

    def put(d, f, ls):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text("".join(ls), encoding="utf-8")

    def gens(d, skip=()):
        for s in t2a.SETS:
            if s not in skip:
                shutil.copyfile(src / f"gen_{s}.jsonl", d / f"gen_{s}.jsonl")

    def edit_row(f, i, fn):
        def setup(d):
            ls = lines(f)
            r = json.loads(ls[i])
            fn(r)
            ls[i] = json.dumps(r) + "\n"
            put(d, f, ls)
        return setup

    def swap_rows(d):
        ls = lines("gen_t2_eval.jsonl")
        ls[3], ls[4] = ls[4], ls[3]
        put(d, "gen_t2_eval.jsonl", ls)

    def cut_last(f):
        return lambda d: put(d, f, lines(f)[:-1] + [lines(f)[-1][:40]])

    expect_exit("rows_swapped", swap_rows, "do not align", S)
    expect_exit("no_t2_text", edit_row("gen_supp.jsonl", 0, lambda r: r.pop("t2_text")), "t2_text (not a string)", S)
    expect_exit("null_text", edit_row("gen_t2_eval.jsonl", 5, lambda r: r.update(t2_text=None)), "t2_text (not a string)", S)
    expect_exit("no_i", edit_row("gen_xframe.jsonl", 2, lambda r: r.pop("i")), "mismatch in ['i']", S)
    expect_exit("gen_cut_line", cut_last("gen_pilot.jsonl"), "not valid JSONL", S)

    ej = json.loads((src / "eval.json").read_text(encoding="utf-8"))

    def ej_edit(fn, with_all=False):
        def setup(d):
            gens(d)
            e = json.loads(json.dumps(ej))
            fn(e)
            (d / "eval.json").write_text(json.dumps(e), encoding="utf-8")
        return setup

    k0 = next(iter(ej["per_op_t2_eval_t2"]))
    expect_exit("eval_json_edited", ej_edit(lambda e: e["per_op_t2_eval_t2"][k0].update(acc=0.5)), "differs from a recomputation", S)
    expect_exit("eval_json_incomplete", ej_edit(lambda e: None), "files missing although eval.json exists", S)
    expect_exit("eval_json_sha", ej_edit(lambda e: e["provenance"]["data_sha256"].update(t2_eval="0" * 64)), "does not match the current data", S)
    expect_exit("eval_json_n", ej_edit(lambda e: e["n"].update(pilot=3)), "eval.json n[pilot]", S)

    def chains(mod):
        def setup(d):
            for f in t2a.CHAIN_FILES:
                ls = lines(f)
                put(d, f, mod(f, ls) or ls)
        return setup

    def trunc_off(f, ls):
        return ls[:int(0.85 * len(ls))] if f == "gen_chain_one_base.jsonl" else None

    def dup(f, ls):
        return ls + [ls[7]] if f == "gen_chain_m1.jsonl" else None

    def cross_parent(f, ls):
        if f != "gen_chain_one.jsonl":
            return None
        rows = [json.loads(l) for l in ls]
        par = lambda r: r["cand_id"].rsplit("-c", 1)[0]  # noqa: E731
        j = next(j for j in range(1, len(rows)) if par(rows[j]) != par(rows[0]) and rows[j]["cand_id"][:3] == rows[0]["cand_id"][:3])
        rows[0]["states"], rows[j]["states"] = rows[j]["states"], rows[0]["states"]
        rows[0]["texts"], rows[j]["texts"] = rows[j]["texts"], rows[0]["texts"]
        return [json.dumps(r) + "\n" for r in rows]

    def tamper(f, ls):
        if f != "gen_chain_m1_base.jsonl":
            return None
        rows = [json.loads(l) for l in ls]
        r = next(r for r in rows if r["states"][2] is not None and r["states"][2] != r["states"][1])
        r["states"][2] = r["states"][1]
        return [json.dumps(r) + "\n" for r in rows]

    def cut_chain(f, ls):
        return ls[:-1] + [ls[-1][:30]] if f == "gen_chain_m1_base.jsonl" else None

    expect_exit("chain_truncated_off", chains(trunc_off), "cand_id set differs from pilot TEST", S)
    expect_exit("chain_duplicate", chains(dup), "duplicated cand_id", S)
    expect_exit("chain_cross_parent", chains(cross_parent), "is not this candidate's s0", S)
    expect_exit("chain_tampered", chains(tamper), "does not parse to the stored state", S)
    expect_exit("chain_cut_line", chains(cut_chain), "not valid JSONL", S)

    def offtask_cut(d):
        put(d, "offtask_gen.jsonl", lines("offtask_gen.jsonl")[:200])

    def meta_ids(d):
        m = json.loads((src / "off" / "offtask_meta.json").read_text(encoding="utf-8"))
        m["ids"] = m["ids"][::-1]
        put(d, "off/offtask_meta.json", [json.dumps(m)])

    expect_exit("offtask_gen_truncated", offtask_cut, "offtask_gen.jsonl: 200 rows", S)
    expect_exit("offtask_meta_ids", meta_ids, "meta ids differ", S)
    # static data alignment (block A joins substrata by index; block C reuses the labels for xframe)
    subs = t2a.load_jsonl(ROOT / "data" / "t2_eval_substrata.jsonl")
    check(not t2a.static_errors(S.data["t2_eval"], S.data["xframe"], subs), "static data must be aligned")
    s2 = list(subs)
    s2[0], s2[1] = s2[1], s2[0]
    check(len(t2a.static_errors(S.data["t2_eval"], S.data["xframe"], s2)) >= 2, "swapped substrata rows must be detected")
    x2 = list(S.data["xframe"])
    x2[10], x2[11] = x2[11], x2[10]
    check(any("xframe" in e for e in t2a.static_errors(S.data["t2_eval"], x2, subs)), "swapped xframe rows must be detected")
    print("  validation: 16 malformed run dirs stop the analysis (SystemExit); 2 static misalignments detected", flush=True)


# ---------------------------------------------------------------- main


def main():
    t0 = time.time()
    print("[selftest] statistics helpers", flush=True)
    stats_checks()
    S = t2a.Static()
    print("[selftest] smoke-run format checks", flush=True)
    smoke_format_checks(S)
    stored = {m: (t2a.STORED_SPARSE / f"gen_{m}.jsonl").read_text(encoding="utf-8") for m in t2a.MODES}
    FIX.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        t1 = time.time()
        build(name, S, stored)
        print(f"[selftest] built fixture {name} in {time.time() - t1:.0f}s", flush=True)
    results = {}
    for name in NAMES:
        t1 = time.time()
        out = t2a.run([name], FIX, FIX / name / f"T2_RESULTS_{name}.md")
        dt = time.time() - t1
        results[name] = out["results"][name]
        G = results[name]["gates"]
        print(f"[selftest] {name}: analysed in {dt:.0f}s | " + ", ".join(f"{g.split('_')[0]} {G[g]['status']}" for g in t2a.GATE_NAMES)
              + f" | rule learned {G['rule_learned']['status']}", flush=True)
        check(dt < 300, f"{name}: analysis took {dt:.0f}s (> 5 min)")
        check(out["meta"]["torch_imported"] is False, f"{name}: torch was imported")
    print("[selftest] assertions: null", flush=True)
    assert_null(results["null"])
    print("[selftest] assertions: perfect", flush=True)
    assert_perfect(results["perfect"])
    print("[selftest] assertions: copy_collapse", flush=True)
    assert_copy_collapse(results["copy_collapse"])
    print("[selftest] determinism + across-seed path", flush=True)
    a = (FIX / "perfect" / "T2_RESULTS_perfect.json").read_bytes()
    t2a.run(["perfect"], FIX, FIX / "_det_check" / "T2_RESULTS_perfect.md")
    b = (FIX / "_det_check" / "T2_RESULTS_perfect.json").read_bytes()
    check(a.replace(b"_det_check", b"") == b.replace(b"_det_check", b""), "two runs on the same fixture differ")
    md = (FIX / "_det_check" / "T2_RESULTS_perfect.md").read_text(encoding="utf-8")
    for phrase in ("Seeds present: 1 of 3", "self-distillation control", "sealed confirmation bank is not used",
                   "one extra copied row in any family fails it", "row-level, ignores clustering"):
        check(phrase in md, f"single-tag report lacks {phrase!r}")
    shutil.rmtree(FIX / "_det_check")
    out = t2a.run(list(NAMES), FIX, FIX / "T2_RESULTS_fixtures_all.md")
    X = out["meta"]["across"]
    m = {r["metric"]: r for r in X["metrics"]}
    check(X["n_seeds"] == 3 and X["three_seeds"], "across: seed count")
    check(m["A P.app_del delta"]["agree"] == "no" and m["A P.app_del delta"]["signs"] == "0 + 0",
          f"across: P.app_del agreement {m['A P.app_del delta']}")
    check(m["A P.inapp_disc delta"]["agree"] == "no change", "across: P.inapp_disc should be 'no change'")
    check("C xframe P.applicable delta" in m and not any(k.startswith("C xframe P all") for k in m),
          "across: the P transfer metric must exclude the descriptive-only inapplicable rows")
    check(X["rule_learned_pass_seeds"] == (1 if results["perfect"]["gates"]["rule_learned"]["status"] == "PASS" else 0)
          and X["rule_learned_all_seeds"] is False, "across: rule-learned seed count")
    check(X["gates"]["rule_learned"] == {"null": "FAIL", "perfect": results["perfect"]["gates"]["rule_learned"]["status"], "copy_collapse": "FAIL"},
          "across: gate table")
    print("[selftest] G8 with a synthetic Supa reference + partial-run (crash) path", flush=True)
    ref_dir = FIX / "_supa_ref"
    if ref_dir.exists():
        shutil.rmtree(ref_dir)
    ref_dir.mkdir()
    shutil.copyfile(t2a.STORED_SPARSE / "offtask_meta.json", ref_dir / "offtask_meta.json")
    src = np.load(t2a.STORED_SPARSE / "offtask_logprobs.npy", mmap_mode="r")
    dst = np.lib.format.open_memmap(ref_dir / "offtask_logprobs.npy", mode="w+", dtype=np.float16, shape=src.shape)
    rng = np.random.default_rng(11)
    for s in range(0, src.shape[0], 8):
        a = np.asarray(src[s:s + 8]).astype(np.float64) + rng.normal(0, 0.3, (min(8, src.shape[0] - s), src.shape[1]))
        dst[s:s + 8] = np.maximum(t2a._lsm(a), -60.0).astype(np.float16)
    dst.flush()
    del dst, src
    real_supa = t2a.STORED_SUPA
    t2a.STORED_SUPA = ref_dir
    try:
        for name, want in (("perfect", "PASS"), ("copy_collapse", "FAIL"), ("null", "PASS")):
            r = dict(results[name])
            r["F"] = t2a.forgetting_block(FIX / name, S)
            G = t2a.gates(r)
            ref = r["F"]["kl_ref_supa_sparse"]
            check(isinstance(ref, dict) and ref["mean"] > 0, f"synthetic Supa reference KL {ref}")
            check(G["G8_forgetting"]["status"] == want, f"G8 with reference, {name}: {G['G8_forgetting']['detail']}")
            if name == "perfect":
                check(G["rule_learned"]["status"] == "PASS", "perfect with reference: rule learned should PASS")
            print(f"  {name}: {G['G8_forgetting']['detail']} -> {G['G8_forgetting']['status']}; rule learned {G['rule_learned']['status']}")
    finally:
        t2a.STORED_SUPA = real_supa
    shutil.rmtree(ref_dir)
    part = FIX / "_partial"
    if part.exists():
        shutil.rmtree(part)
    part.mkdir()
    for f in ("gen_t2_eval.jsonl", "gen_supp.jsonl", "gen_xframe.jsonl"):
        shutil.copyfile(FIX / "perfect" / f, part / f)
    r = t2a.analyse_seed(part, S)
    G = r["gates"]
    check([G[g]["status"] for g in t2a.GATE_NAMES] == ["PASS"] * 4 + ["PENDING"] * 4 and G["rule_learned"]["status"] == "PENDING",
          f"partial run gates: {[G[g]['status'] for g in t2a.GATE_NAMES]} / {G['rule_learned']['status']}")
    check("B" not in r and "E" not in r and "missing" in r["D"], "partial run: B/D/E should be absent or missing")
    shutil.rmtree(part)
    print("[selftest] G5 carry-over semantics + G4 one-row rule", flush=True)
    g5_semantics(S)
    g4_one_row(results["null"])
    print("[selftest] input validation (malformed run dirs must stop the analysis)", flush=True)
    validation_checks(S)
    print(f"[selftest] total {time.time() - t0:.0f}s", flush=True)
    if FAILS:
        print(f"[selftest] {len(FAILS)} FAILED checks", flush=True)
        sys.exit(1)
    print("[selftest] ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
