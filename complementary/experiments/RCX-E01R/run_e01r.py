"""RCX-E01R step A: DEV2 diagnostic of the existing E01N adapters on fresh, blind-authored rule wordings (P2).

Descriptive only (DESIGN.md section 3); no training, no confirmatory claim. From complementary/:
  python -B experiments/RCX-E01R/run_e01r.py build                       (DEV2 battery -> runs/RCX-E01R/dev2)
  python -B experiments/RCX-E01R/run_e01r.py eval --arm N-CFSD --seed 0   (one cell; refuses below 12 GB free RAM)
  python -B experiments/RCX-E01R/run_e01r.py analyze                      (DIAG_DEV2.md / .json)

Battery: single-atom strata of the E01N DEV generator with a fresh seed; surfaces `in` (canonical t0 wording) and `p2`
(each parent gets one form, rotated within its stratum, and one P2 template of that form per polarity, so V3/V5 twins
keep the same form). The P2 templates are registered in templates.SPOS/SNEG under ids `p2:<id>` at import; nothing in
rcx/rcl is modified on disk.
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "RCX-E01N"))

from rcx import stats as ST  # noqa: E402
from rcx.rcl import gen, score as S, templates as T, world as W  # noqa: E402

HERE = Path(__file__).resolve().parent
P2_PATH = HERE / "pools" / "P2_DEV.json"
OUT = ROOT / "runs" / "RCX-E01R" / "dev2"
DEV2_SEED = 20261261
N_PARENTS = 30
STRATA = ("F1_void", "F2_unseen_real", "F3_unseen_nonce", "F4_nearmiss", "F5_postedlike",
          "V1_pos", "V2_neg", "V3_inverted", "V5_posted_override", "V7_bigset")
FORMS = ("EXPLICIT_LISTED_FIRST", "EXPLICIT_COMPLEMENT_FIRST", "IMPLICIT")
CELLS_DIR = ROOT / "runs" / "RCX-E01N" / "cells"


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_p2(register=True):
    pool = json.loads(P2_PATH.read_text(encoding="utf-8"))["templates"]
    if len({t["id"] for t in pool}) != len(pool):
        raise SystemExit("P2 template ids are not unique")
    for t in pool:
        body = t["text"].replace("{L}", "")
        if t["text"].count("{L}") != 1 or "{" in body or "}" in body:
            raise SystemExit(f"P2 template {t['id']} is not a single-slot format string")
        if register:
            (T.SPOS if t["pol"] == "POS" else T.SNEG)[f"p2:{t['id']}"] = t["text"]
    return pool


def build_battery(pool):
    gen.use_world("arith")
    base = gen.make_battery("dev", N_PARENTS, DEV2_SEED, strata=STRATA, surfaces=("in",))
    by_cell = defaultdict(list)
    for t in pool:
        by_cell[(t["pol"], t["form"])].append(t["id"])
    for k in by_cell:
        by_cell[k].sort()
    parents = defaultdict(list)
    for r in base:
        parents[r["parent_id"]].append(r)
    out = list(base)
    rank = defaultdict(int)
    for pid, recs in parents.items():
        stratum = recs[0]["group"]
        form = FORMS[rank[stratum] % len(FORMS)]
        rank[stratum] += 1
        rng = random.Random(f"{DEV2_SEED}-{pid}")
        pick = {pol: rng.choice(by_cell[(pol, form)]) for pol in ("POS", "NEG")}
        for r in recs:
            op = r["rule_ast"]["op"]
            if op not in ("S+", "S-"):
                raise SystemExit(f"{pid}: non-atomic rule {op} in a single-atom stratum")
            pol = "POS" if op == "S+" else "NEG"
            q = json.loads(json.dumps(r))
            q["surface"], q["tid"], q["p2_form"], q["p2_template"] = "p2", f"p2:{pick[pol]}", form, pick[pol]
            q["input"] = W.render_input(q["rule_ast"], q["state"], q["event"], q["tid"], q["state_order"],
                                        list(q["event"].keys()))
            out.append(q)
    return out


def battery_bytes(battery):
    return "".join(json.dumps(r) + "\n" for r in battery).encode()


def cmd_build(a):
    pool = load_p2()
    battery = build_battery(pool)
    OUT.mkdir(parents=True, exist_ok=True)
    bp = OUT / "battery.jsonl"
    data = battery_bytes(battery)
    if bp.exists() and bp.read_bytes() != data:
        raise SystemExit(f"{bp} exists with different content; DEV2 is built once")
    bp.write_bytes(data)
    meta = {"seed": DEV2_SEED, "n_parents_per_stratum": N_PARENTS, "strata": STRATA, "n": len(battery),
            "by_surface": {s: sum(1 for r in battery if r["surface"] == s) for s in ("in", "p2")},
            "battery_sha256": sha_bytes(data), "p2_sha256": sha_bytes(P2_PATH.read_bytes()),
            "templates_used": len({r["p2_template"] for r in battery if r["surface"] == "p2"}),
            "code_sha256": sha_bytes(Path(__file__).read_bytes())}
    (OUT / "BATTERY_META.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(meta, indent=1))


def load_battery():
    load_p2()
    bp = OUT / "battery.jsonl"
    meta = json.loads((OUT / "BATTERY_META.json").read_text(encoding="utf-8"))
    data = bp.read_bytes()
    if sha_bytes(data) != meta["battery_sha256"]:
        raise SystemExit("DEV2 battery sha mismatch")
    return [json.loads(x) for x in data.decode().splitlines()], meta


def cmd_eval(a):
    import torch
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.set_num_threads(int(os.environ.get("RCX_THREADS", "4")))
    import native_probe as NP
    from rcx.guard import ThermalGuard
    from rcx.native import lora_cpu as LC

    battery, meta = load_battery()
    tag = f"{a.arm}_s{a.seed}" if a.arm.startswith("N-") else a.arm
    res_path = OUT / f"eval_{tag}.json"
    if res_path.exists():
        print("eval exists", flush=True)
        return
    adapter_sha = None
    NP.need_ram()
    model, tok = LC.load_core("sparse")
    if a.arm.startswith("N-"):
        from peft import PeftModel
        ad = CELLS_DIR / tag / "adapter"
        adapter_sha = hashlib.sha256((ad / "adapter_model.safetensors").read_bytes()).hexdigest()
        model = PeftModel.from_pretrained(model, str(ad), is_trainable=False).eval()
    elif a.arm == "T2":
        model = NP.load_t2(model)

    def render(system, user):
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)

    prompts = [render(NP.rcl_system(T.render_rule(r["rule_ast"], r["tid"])), NP.user_json(r)) for r in battery]
    guard = ThermalGuard(log=lambda m: print("guard:", m, flush=True))
    t0 = time.time()
    outs = LC.generate(model, tok, prompts, guard=guard, log=lambda m: print(m, flush=True))
    res = [S.classify(r, NP.parse_answer(o, list(r["gold"]["after"].keys()))) for r, o in zip(battery, outs)]
    with gzip.open(OUT / f"preds_{tag}.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, (o, x) in enumerate(zip(outs, res)):
            f.write(json.dumps({"i": i, "out": o, **x}) + "\n")
    rec = {"arm": a.arm, "seed": a.seed, "battery_sha256": meta["battery_sha256"], "adapter_sha256": adapter_sha,
           "n": len(battery), "wall_s": round(time.time() - t0, 1)}
    res_path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"[{tag}] done {rec['wall_s']}s", flush=True)


# ----------------------------------------------------------------------------- analysis (descriptive)

def load_preds(tag):
    with gzip.open(OUT / f"preds_{tag}.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def cell_key(r):
    pol = "POS" if r["rule_ast"]["op"] == "S+" else "NEG"
    return (r["surface"], pol, r.get("p2_form", "-"))


def tally(battery, preds, select):
    """{(surface, pol, form): {noop: [k, n], apply: [k, n], exact: [k, n]}} over records passing `select`."""
    out = defaultdict(lambda: {"noop": [0, 0], "apply": [0, 0], "exact": [0, 0]})
    for r, p in zip(battery, preds):
        if not select(r):
            continue
        c = out[cell_key(r)]
        c[r["gold"]["decision"]][0] += p["decision_ok"]
        c[r["gold"]["decision"]][1] += 1
        c["exact"][0] += p["exact"]
        c["exact"][1] += 1
    return out


def unpaired_parent_boot(xs, ys, resamples=5000, seed=20261262):
    """Descriptive unpaired bootstrap over parents of mean(ys) - mean(xs); xs, ys: lists of parent means."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        mx = sum(rng.choice(xs) for _ in xs) / len(xs)
        my = sum(rng.choice(ys) for _ in ys) / len(ys)
        diffs.append(my - mx)
    diffs.sort()
    return sum(ys) / len(ys) - sum(xs) / len(xs), diffs[int(0.025 * resamples)], diffs[int(0.975 * resamples) - 1]


def noop_parent_means(battery, per_seed_preds, surface, pol, form):
    """Per parent: mean over seeds and over required-no-op records of decision_ok."""
    acc = defaultdict(list)
    for preds in per_seed_preds:
        for r, p in zip(battery, preds):
            if (r["surface"], ("POS" if r["rule_ast"]["op"] == "S+" else "NEG"), r.get("p2_form", "-")) != \
                    (surface, pol, form) or r["gold"]["decision"] != "noop":
                continue
            acc[r["parent_id"]].append(p["decision_ok"])
    return [sum(v) / len(v) for v in acc.values()]


def fmt(k, n):
    if n == 0:
        return "-"
    lo, hi = ST.wilson(k, n)
    return f"{k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]"


def cmd_analyze(a):
    battery, meta = load_battery()
    arms = {}
    for tag in ("N-CFSD_s0", "N-CFSD_s1", "N-CFSD_s2", "N-VR_s0", "OFF", "T2"):
        if (OUT / f"eval_{tag}.json").exists():
            ev = json.loads((OUT / f"eval_{tag}.json").read_text(encoding="utf-8"))
            if ev["battery_sha256"] != meta["battery_sha256"]:
                raise SystemExit(f"{tag}: battery sha mismatch")
            arms[tag] = load_preds(tag)
    groups = {"N-CFSD (s0-s2 pooled)": [t for t in arms if t.startswith("N-CFSD")],
              "N-VR s0": [t for t in arms if t == "N-VR_s0"], "OFF": [t for t in arms if t == "OFF"],
              "T2": [t for t in arms if t == "T2"]}
    groups = {g: ts for g, ts in groups.items() if ts}
    rows, res = [], {"battery": meta, "arms": sorted(arms), "cells": {}, "contrasts": {}}
    keys = [("in", pol, "-") for pol in ("POS", "NEG")] + [("p2", pol, f) for pol in ("POS", "NEG") for f in FORMS]
    for g, ts in groups.items():
        agg = defaultdict(lambda: {"noop": [0, 0], "apply": [0, 0], "exact": [0, 0]})
        for t in ts:
            for k, v in tally(battery, arms[t], lambda r: True).items():
                for f in v:
                    agg[k][f][0] += v[f][0]
                    agg[k][f][1] += v[f][1]
        res["cells"][g] = {"|".join(k): agg[k] for k in keys}
        for k in keys:
            c = agg[k]
            rows.append(f"| {g} | {k[0]} | {k[1]} | {k[2]} | {fmt(*c['noop'])} | {fmt(*c['apply'])} | "
                        f"{fmt(*c['exact'])} |")
        for f in FORMS:
            xs = noop_parent_means(battery, [arms[t] for t in ts], "p2", "NEG", f)
            ys = noop_parent_means(battery, [arms[t] for t in ts], "p2", "POS", f)
            if xs and ys:
                d, lo, hi = unpaired_parent_boot(xs, ys)
                res["contrasts"][f"{g}|POS-NEG noop|{f}"] = {"diff": d, "lo": lo, "hi": hi,
                                                              "parents_pos": len(ys), "parents_neg": len(xs)}
    lines = ["# RCX-E01R step A: DEV2 diagnostic on fresh blind-authored wordings (descriptive)", "",
             f"Battery: seed {meta['seed']}, sha256 `{meta['battery_sha256']}`, {meta['n']} records "
             f"({meta['by_surface']}), {meta['templates_used']} P2 templates used; P2 sha256 `{meta['p2_sha256']}`.",
             f"Arms: {', '.join(sorted(arms))}. Readings fixed in DESIGN.md section 3 before the run.", "",
             "| arm | surface | polarity | form | no-op items decided no-op | apply items decided apply | exact |",
             "|---|---|---|---|---|---|---|", *rows, "",
             "## POS - NEG no-op accuracy within form (`p2`; unpaired parent bootstrap, descriptive)", "",
             "| arm | form | diff | 95% CI | parents POS / NEG |", "|---|---|---|---|---|"]
    for k, v in res["contrasts"].items():
        g, _, f = k.split("|")
        lines.append(f"| {g} | {f} | {v['diff']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] | "
                     f"{v['parents_pos']} / {v['parents_neg']} |")
    if "N-CFSD (s0-s2 pooled)" in res["cells"]:
        c = res["cells"]["N-CFSD (s0-s2 pooled)"]
        pos = {f: c[f"p2|POS|{f}"]["noop"] for f in FORMS}
        neg = {f: c[f"p2|NEG|{f}"]["noop"] for f in FORMS}
        rate = lambda kn: kn[0] / kn[1] if kn[1] else float("nan")  # noqa: E731
        alt = all(rate(pos[f]) >= 0.80 for f in FORMS)
        main = all(rate(pos[f]) < 0.50 for f in FORMS) and all(rate(neg[f]) >= 0.80 for f in FORMS)
        res["reading"] = "H-alt" if alt else ("H-E01R" if main else "intermediate")
        lines += ["", f"**Pre-stated reading (DESIGN.md section 3), N-CFSD pooled:** {res['reading']}. "
                  "POS no-op by form: " + ", ".join(f"{f} {rate(pos[f]):.2f}" for f in FORMS) +
                  "; NEG no-op by form: " + ", ".join(f"{f} {rate(neg[f]):.2f}" for f in FORMS) + "."]
    (OUT / "DIAG_DEV2.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (OUT / "DIAG_DEV2.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    e = sub.add_parser("eval")
    e.add_argument("--arm", required=True, choices=["N-CFSD", "N-VR", "OFF", "T2"])
    e.add_argument("--seed", type=int, default=0)
    sub.add_parser("analyze")
    a = ap.parse_args()
    {"build": cmd_build, "eval": cmd_eval, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
