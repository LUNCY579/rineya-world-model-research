"""D-54 key-order augmentation (PREREG_D54_KEYORDER.md).

  data  (CPU) runs/BW_data/forward_delta_ko_train.jsonl: the 12,000 transitions of forward_delta_train.jsonl with the
        prompt re-rendered (p 0.5 canonical, else a random TRAINING key order, lists shuffled with p 0.5); targets
        unchanged. Held-out state-key orders never appear.
  eval  (GPU/CPU) F0 and FK on 240 fresh keys in five rendering modes, paired; writes runs/BW_KEYORDER/eval.json.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402

STATE_KEYS = ("clear", "hand", "on", "table")
HELDOUT = {("table", "on", "hand", "clear"), ("hand", "table", "clear", "on"), ("on", "clear", "table", "hand")}
TRAIN_ORDERS = [o for o in itertools.permutations(STATE_KEYS) if o not in HELDOUT]
UNSEEN = list("pqrstuvwxyz") + ["ab", "kx", "mo", "zq", "ty", "uv"]
DATA_SEED, EVAL_SEED = 20261012, 20261013
ADAPTERS = {"F0": "runs/BW_Fdelta_s0/adapter", "FK": "runs/BW_Fdelta_ko_s0/adapter"}


def render(s, a, state_order, state_first: bool, shuffle_lists: bool, rng) -> str:
    st = B.state_json(s)
    if shuffle_lists:
        for k in ("clear", "on", "table"):
            rng.shuffle(st[k])
    st = {k: st[k] for k in state_order}
    user = {"state": st, "action": B.action_str(a)} if state_first else {"action": B.action_str(a), "state": st}
    return B.render_chat(B.SYS_FORWARD_DELTA, json.dumps(user, separators=(",", ":")))


def data(a):
    rng = random.Random(DATA_SEED)
    src = ROOT / "runs/BW_data/forward_delta_train.jsonl"
    dst = ROOT / "runs/BW_data/forward_delta_ko_train.jsonl"
    if dst.exists():
        sys.exit(f"{dst} exists")
    kinds = Counter()
    with open(dst, "w", encoding="utf-8") as out:
        for line in open(src, encoding="utf-8"):
            r = json.loads(line)
            user = json.loads(r["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
            s, act = B.state_from_json(user["state"]), tuple(user["action"].split())
            assert B.forward_prompt_delta(s, act) == r["prompt"]
            if rng.random() < 0.5:
                prompt, kind = r["prompt"], "canon"
            else:
                order = rng.choice(TRAIN_ORDERS)
                prompt = render(s, act, order, rng.random() < 0.5, rng.random() < 0.5, rng)
                kind = "aug"
            assert tuple(json.loads(prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])["state"]) \
                not in HELDOUT
            kinds[kind] += 1
            out.write(json.dumps({"prompt": prompt, "target": r["target"], "legal": r["legal"], "n_blocks": r["n_blocks"],
                                  "render": kind}) + "\n")
    print(dict(kinds), "->", dst.name)


def keys(rng, per_size=120):
    out = []
    for n in (5, 10):
        for _ in range(per_size):
            objs = sorted(rng.sample(B.NAMES, n))
            s = B.towers_to_state(B.random_towers(objs, rng))
            for _ in range(rng.randrange(0, 5)):
                s, _ = B.step(s, rng.choice([x for x in B.ground_actions(objs) if B.applicable(s, x)]))
            ga = B.ground_actions(objs)
            leg = [x for x in ga if B.applicable(s, x)]
            ill = [x for x in ga if not B.applicable(s, x)]
            act = rng.choice(leg) if (rng.random() < 0.5 or not ill) else rng.choice(ill)
            out.append((n, objs, s, act))
    return out


def _cp(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def rename(s, a, objs, names):
    m = dict(zip(objs, names))
    return (frozenset(tuple([f[0]] + [m[x] for x in f[1:]]) for f in s), tuple([a[0]] + [m[x] for x in a[1:]]))


def eval_(a):
    from scipy.stats import binomtest
    from bwmn import lm
    from peft import PeftModel

    rng = random.Random(EVAL_SEED)
    ks = keys(rng)
    rows = []
    for mode in ("canon", "train_order", "heldout_rev", "heldout_other", "heldout_both"):
        for i, (n, objs, s, act) in enumerate(ks):
            s2, a2 = (rename(s, act, objs, rng.sample(UNSEEN, len(objs))) if mode == "heldout_both" else (s, act))
            if mode == "canon":
                p = B.forward_prompt_delta(s2, a2)
            elif mode == "train_order":
                p = render(s2, a2, rng.choice([o for o in TRAIN_ORDERS if o != STATE_KEYS]), rng.random() < 0.5, True, rng)
            elif mode == "heldout_rev":
                p = render(s2, a2, ("table", "on", "hand", "clear"), True, False, rng)
            elif mode == "heldout_other":
                p = render(s2, a2, rng.choice([("hand", "table", "clear", "on"), ("on", "clear", "table", "hand")]),
                           rng.random() < 0.5, False, rng)
            else:
                p = render(s2, a2, ("table", "on", "hand", "clear"), True, True, rng)
            rows.append({"mode": mode, "i": i, "n": n, "s": s2, "a": a2, "prompt": p})
    shas = {k: lm.file_sha256(ROOT / v / "adapter_model.safetensors") for k, v in ADAPTERS.items()}
    assert shas["F0"] != shas["FK"], "F0 and FK are the same adapter file"
    base, tok = lm.load_core("titan-sparse")
    model = PeftModel.from_pretrained(base, str(ROOT / ADAPTERS["F0"]), adapter_name="F0", is_trainable=False)
    model.load_adapter(str(ROOT / ADAPTERS["FK"]), adapter_name="FK", is_trainable=False)
    model.eval()
    ok, items = {}, []
    for arm in ADAPTERS:
        model.set_adapter(arm)
        assert model.active_adapter == arm
        gens = lm.greedy_shared_prefix(model, tok, [r["prompt"] for r in rows], max_new_tokens=40)
        for r, g in zip(rows, gens):
            truth = B.step(r["s"], r["a"])
            ok[(arm, r["mode"], r["i"])] = B.parse_forward_delta(g["text"], r["s"]) == truth
            items.append({"arm": arm, "mode": r["mode"], "i": r["i"], "n_blocks": r["n"], "legal": bool(truth[1]),
                          "prompt_sha": hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest()[:16], "text": g["text"],
                          "n_tokens": g["n_tokens"], "hit_eos": g["hit_eos"], "ok": bool(ok[(arm, r["mode"], r["i"])])})
    res = {}
    for mode in ("canon", "train_order", "heldout_rev", "heldout_other", "heldout_both"):
        f0 = [ok[("F0", mode, i)] for i in range(len(ks))]
        fk = [ok[("FK", mode, i)] for i in range(len(ks))]
        b = sum(x and not y for x, y in zip(fk, f0))
        c = sum(y and not x for x, y in zip(fk, f0))
        res[mode] = {"n": len(ks), "F0": round(sum(f0) / len(ks), 4), "FK": round(sum(fk) / len(ks), 4),
                     "F0_ci95": _cp(sum(f0), len(ks)), "FK_ci95": _cp(sum(fk), len(ks)),
                     "FK_only": b, "F0_only": c, "p_mcnemar": 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue}
    held = ("heldout_rev", "heldout_other")
    # Amendment 1 (D-62a, written before FK existed; wording shared with RCWM's T2b-ledger): absolute bar, canonical
    # non-inferiority, and significantly above the unaugmented arm on every held-out mode.
    gen = all(res[m]["FK"] >= 0.95 for m in held) and res["canon"]["FK"] >= res["canon"]["F0"] - 0.01 and \
        all(res[m]["FK_only"] > res[m]["F0_only"] and res[m]["p_mcnemar"] < 0.05 for m in held)
    near_f0 = all(res[m]["FK"] - res[m]["F0"] < 0.10 or res[m]["p_mcnemar"] >= 0.05 for m in held)
    widened = res["train_order"]["FK"] >= 0.95 and near_f0
    reading = "generalises to unseen key orders" if gen else "trigger widened, not a rule" if widened else "partial"
    out = ROOT / "runs/BW_KEYORDER"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "items.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    os.replace(tmp, out / "items.jsonl")
    import peft
    import torch
    import transformers
    prov = {"adapter_sha256": shas, "rows_sha256": lm.file_sha256(ROOT / "runs/BW_data/forward_delta_ko_train.jsonl"),
            "script_sha256": lm.file_sha256(Path(__file__)), "prereg_sha256": lm.file_sha256(ROOT / "PREREG_D54_KEYORDER.md"),
            "items_sha256": lm.file_sha256(out / "items.jsonl"), "device": str(next(model.parameters()).device),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}}
    (out / "eval.json").write_text(json.dumps({"modes": res, "reading": reading, "eval_seed": EVAL_SEED,
                                               "adapters": ADAPTERS, "provenance": prov}, indent=1))
    print(json.dumps({"modes": res, "reading": reading}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "eval"])
    a = ap.parse_args()
    data(a) if a.stage == "data" else eval_(a)


if __name__ == "__main__":
    main()
