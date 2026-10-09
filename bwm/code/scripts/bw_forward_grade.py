"""Grade a Blocksworld forward adapter (BW_F) by exact match against the exact simulator.

Two key sets (both generated here, never PlanBench test problems):
  val   : the first --val-rows rows of runs/BW_data/forward_val.jsonl (4-6 blocks, ~50% illegal; in-distribution)
  size  : --per-size random (state, action) pairs for each block count in --sizes (states = random towers plus a short
          random walk, names from a..o as in PlanBench; action legal with prob. 0.5), to measure size generalization
          toward PlanBench Blocksworld-hard (6-15 blocks). Training used 4-6 blocks.
Greedy decoding with the shared-prefix decoder (lm.greedy_shared_prefix); --check K first decodes K keys with both
decoders and reports agreement. Outputs are appended per chunk to an fsynced cache so a crash loses at most one chunk.

Usage: python scripts/bw_forward_grade.py --adapter runs/BW_F_s0/adapter --out runs/BW_F_s0/grade_size.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402
from bwmn import lm  # noqa: E402


def prompt_for(fmt, s, a):
    return B.forward_prompt_delta(s, a) if fmt == "delta" else B.forward_prompt(s, a)


def target_for(fmt, s, ns, ok):
    return B.forward_target_delta(s, ns, ok) if fmt == "delta" else B.forward_target(ns, ok)


def parse_for(fmt, text, s):
    return B.parse_forward_delta(text, s) if fmt == "delta" else B.parse_forward(text)


def size_keys(sizes, per_size, seed, fmt="state"):
    rng = random.Random(seed)
    keys = []
    for n in sizes:
        seen = set()
        while sum(1 for k in keys if k["n"] == n) < per_size:
            objs = sorted(rng.sample(B.NAMES, n))
            s = B.towers_to_state(B.random_towers(objs, rng))
            for _ in range(rng.randrange(0, 6)):
                legal = [a for a in B.ground_actions(objs) if B.applicable(s, a)]
                s, _ = B.step(s, rng.choice(legal))
            ga = B.ground_actions(objs)
            legal = [a for a in ga if B.applicable(s, a)]
            illegal = [a for a in ga if not B.applicable(s, a)]
            a = rng.choice(legal) if (rng.random() < 0.5 or not illegal) else rng.choice(illegal)
            if (s, a) in seen:
                continue
            seen.add((s, a))
            ns, ok = B.step(s, a)
            keys.append({"set": "size", "n": n, "legal": ok, "prompt": prompt_for(fmt, s, a),
                         "target": target_for(fmt, s, ns, ok), "s": s, "a": a})
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="titan-sparse")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val-rows", type=int, default=600)
    ap.add_argument("--sizes", default="3,4,5,6,8,10,12,15")
    ap.add_argument("--per-size", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--check", type=int, default=0, help="decode the first K keys with both decoders and compare")
    ap.add_argument("--fmt", choices=["state", "delta"], default="state")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.with_name(out.stem + ".partial.jsonl")

    val = [json.loads(l) for l in open(ROOT / "runs/BW_data/forward_val.jsonl", encoding="utf-8")][:a.val_rows]
    keys = []
    for r in val:  # same transitions in both formats; rebuild (s, a) from the stored state-format prompt
        u = json.loads(r["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
        s0, a0 = B.state_from_json(u["state"]), tuple(u["action"].split())
        ns0, ok0 = B.step(s0, a0)
        keys.append({"set": "val", "n": r["n_blocks"], "legal": ok0, "prompt": prompt_for(a.fmt, s0, a0),
                     "target": target_for(a.fmt, s0, ns0, ok0), "s": s0, "a": a0})
    keys += size_keys([int(x) for x in a.sizes.split(",")], a.per_size, a.seed, a.fmt)
    for i, k in enumerate(keys):
        k["i"] = i

    model, tok = lm.load_core(a.core)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ROOT / a.adapter), is_trainable=False)
    model.eval()

    def max_new(batch):
        return max(len(tok(k["target"], add_special_tokens=False).input_ids) for k in batch) + 16

    check = None
    if a.check:
        sub = keys[:a.check]
        t0 = time.time()
        g1 = lm.greedy_shared_prefix(model, tok, [k["prompt"] for k in sub], max_new_tokens=max_new(sub))
        t1 = time.time()
        g2 = lm.sample_crn(model, tok, [k["prompt"] for k in sub], [None] * len(sub), max_new_tokens=max_new(sub))
        t2 = time.time()
        same = sum(x["text"] == y["text"] for x, y in zip(g1, g2))
        check = {"n": len(sub), "identical": same, "shared_s": round(t1 - t0, 1), "plain_s": round(t2 - t1, 1)}
        print("decoder check", json.dumps(check), flush=True)

    done = {}
    if cache.exists():
        good = []
        for line in cache.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[r["i"]] = r["text"]
                good.append(line)
            except Exception:
                pass
        tmp = cache.with_suffix(".tmp")
        tmp.write_text("".join(l + "\n" for l in good), encoding="utf-8")
        os.replace(tmp, cache)
        print(f"resumed {len(done)} of {len(keys)} keys", flush=True)
    todo = [k for k in keys if k["i"] not in done]
    # group by size so each chunk has a tight max_new_tokens
    todo.sort(key=lambda k: (k["n"], k["i"]))
    t0 = time.time()
    for c0 in range(0, len(todo), a.chunk):
        part = todo[c0:c0 + a.chunk]
        gens = lm.greedy_shared_prefix(model, tok, [k["prompt"] for k in part], max_new_tokens=max_new(part))
        with open(cache, "a", encoding="utf-8") as fh:
            for k, g in zip(part, gens):
                fh.write(json.dumps({"i": k["i"], "text": g["text"]}) + "\n")
                done[k["i"]] = g["text"]
            fh.flush()
            os.fsync(fh.fileno())
        print(f"  {min(c0 + a.chunk, len(todo))}/{len(todo)} keys, {time.time() - t0:.0f}s", flush=True)

    n, c = Counter(), Counter()
    errs = defaultdict(list)
    for k in keys:
        pred = parse_for(a.fmt, done[k["i"]], k["s"])
        ref = B.step(k["s"], k["a"])
        ok = pred is not None and pred == ref
        legal_ok = pred is not None and pred[1] == ref[1]
        for s in (f"{k['set']}", f"{k['set']}_n{k['n']}", f"{k['set']}_n{k['n']}_{'legal' if k['legal'] else 'illegal'}"):
            n[s] += 1
            c[s] += ok
            c[s + ":legality"] += legal_ok
            if not ok and len(errs[s]) < 4:
                errs[s].append({"prompt_tail": k["prompt"][-220:], "pred": done[k["i"]][:300], "ref": k["target"][:300]})
    res = {s: {"n": n[s], "exact": round(c[s] / n[s], 4), "legality": round(c[s + ":legality"] / n[s], 4)}
           for s in sorted(n)}
    print(json.dumps({s: v for s, v in res.items() if s.count("_") <= 1}, indent=0), flush=True)
    out.write_text(json.dumps({"adapter": a.adapter, "args": vars(a), "decoder_check": check, "strata": res,
                               "errors": errs}, indent=1))
    cache.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
