"""Stage F: train the goal-blind LM forward model (LoRA on a frozen core) and grade it on exhaustive tables.

Usage: python scripts/train_forward.py --core titan-sparse --rows 4096 --steps 400 --out runs/F_titan_r4096
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from bwmn import lm, rw2  # noqa: E402


def table(limits, only_big=False):
    keys = []
    for L in limits:
        for x in range(-L, L + 1):
            if only_big and abs(x) <= 64:
                continue
            for a in rw2.ACTIONS:
                keys.append((x, a, L))
    return keys


def grade(model, tok, keys, seen=None):
    prompts = [rw2.forward_prompt(x, a, L) for x, a, L in keys]
    gens = lm.generate(model, tok, prompts, max_new_tokens=24)
    res = Counter()
    errors = []
    for (x, a, L), g in zip(keys, gens):
        ref = rw2.exact_step(x, a, L)
        p = rw2.parse_forward(g[0]["text"])
        grp = "seen" if seen is not None and (x, a, L) in seen else "unseen"
        res[f"n_{grp}"] += 1
        res["n"] += 1
        if p is None:
            res["malformed"] += 1
            ok = False
        else:
            ok = p == ref
        res["correct"] += ok
        res[f"correct_{grp}"] += ok
        res[f"n_{a}"] += 1
        res[f"correct_{a}"] += ok
        legal_ref = ref[1]
        res["n_illegal"] += (not legal_ref)
        res["correct_illegal"] += (ok and not legal_ref)
        if not ok and len(errors) < 40:
            errors.append({"x": x, "a": a, "L": L, "ref": ref, "out": g[0]["text"]})
    out = {"acc": res["correct"] / res["n"], **dict(res)}
    for a in rw2.ACTIONS:
        out[f"acc_{a}"] = res[f"correct_{a}"] / max(1, res[f"n_{a}"])
    out["acc_illegal"] = res["correct_illegal"] / max(1, res["n_illegal"])
    if seen is not None:
        out["acc_seen"] = res["correct_seen"] / max(1, res["n_seen"])
        out["acc_unseen"] = res["correct_unseen"] / max(1, res["n_unseen"])
    out["errors_sample"] = errors
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="titan-sparse")
    ap.add_argument("--rows", type=int, default=4096, help="unique transitions sampled for training")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero-shot-only", action="store_true")
    ap.add_argument("--holdout-mod5", action="store_true", help="exclude every key with x %% 5 == 0 from training")
    ap.add_argument("--no-grade", action="store_true", help="train only; grade later on the CPU lane (grade_forward.py)")
    ap.add_argument("--extra-limits", default="", help="comma-separated extra limit values (all <= 64) whose full "
                    "transition tables are added to the training pool, e.g. 16,24,40,56 (rule-parameter variation)")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a", encoding="utf-8")

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    lm.set_determinism(a.seed)
    pools = rw2.build_pools()
    data = rw2.build_training_rows(pools, a.families, seed=1000 + a.seed)
    uniq = {}
    for r in data["forward"]:
        if a.holdout_mod5 and r["x"] % 5 == 0:
            continue
        uniq.setdefault((r["x"], r["action"], r["limit"]), r)
    for L in [int(v) for v in a.extra_limits.split(",") if v]:
        assert L <= 64, "extra limits must keep every number <= 64"
        for x in range(-L, L + 1):
            if a.holdout_mod5 and x % 5 == 0:
                continue
            for act in rw2.ACTIONS:
                nx, legal, term = rw2.exact_step(x, act, L)
                uniq.setdefault((x, act, L), {"x": x, "action": act, "limit": L, "next_x": nx, "legal": legal,
                                              "terminated": term})
    keys = sorted(uniq)
    random.Random(a.seed).shuffle(keys)
    keys = keys[:a.rows]
    rows = [uniq[k] for k in keys]
    seen = set(keys)
    P(f"unique transitions available {len(uniq)}; training on {len(rows)}; illegal {sum(not r['legal'] for r in rows)}; "
      f"finish {sum(r['action']=='finish' for r in rows)}")

    model, tok = lm.load_core(a.core)
    meta = {"core": a.core, "args": vars(a), "core_path": str(lm.MODELS[a.core])}
    from grade_forward import grade_tables
    if a.zero_shot_only:
        res = grade_tables(model, tok, seen=None)
        P(json.dumps(res["strata"]))
        (out / "metrics.json").write_text(json.dumps({"meta": meta, "zero_shot": res}, indent=1))
        return
    pm = lm.attach_lora(model, seed=a.seed)
    sft = rw2.forward_sft(rows)
    t0 = time.time()
    trainlog = lm.train_sft(pm, tok, sft, steps=a.steps, batch_size=a.batch, lr=a.lr, seed=a.seed, logger=P,
                            ckpt_dir=out / "ckpt")
    P("train done", json.dumps({k: v for k, v in trainlog.items() if k != "loss_curve_every50"}))
    lm.save_adapter(pm, out / "adapter", {**meta, "train": trainlog})
    (out / "adapter" / "TRAIN_KEYS.json").write_text(json.dumps(sorted(seen)))
    __import__("shutil").rmtree(out / "ckpt", ignore_errors=True)
    if a.no_grade:
        (out / "metrics.json").write_text(json.dumps({"meta": meta, "train": trainlog, "grade": "deferred-to-cpu"}, indent=1))
        P("trained; grading deferred to the CPU lane")
        return
    res = grade_tables(pm, tok, seen=seen)
    P(json.dumps(res["strata"]))
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "train": trainlog, "grade": res}, indent=1))
    P("total s", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
