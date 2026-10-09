"""Grade a forward adapter (or zero-shot core) on full transition tables with strata.

Strata: id_seen/id_unseen (vs the adapter's training keys if known), range_small (|x|<=64 at L 96/128; includes the
legality-flip doubles), range_flip (double with 32<|x|<=64 at L=96 or L=128 where 2|x|<=L), range_big (|x|>64),
out3 (true output has 3 digits), illegal, finish. Also a held-out limit table (L=40, L=56) never used in training.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import lm, rw2

def strata(x, a, L, train_limits=(32, 48, 64)):
    nx, legal, term = rw2.exact_step(x, a, L)
    s = []
    if L in train_limits:
        s.append("id")
    elif L in (96, 128):
        s.append("range")
        s.append("range_big" if abs(x) > 64 else "range_small")
        if a == "double" and abs(x) > 32 and abs(2 * x) <= L and abs(2 * x) > 64:
            s.append("range_flip_double")
        if abs(nx) >= 100:
            s.append("out3")
    else:
        s.append(f"limit{L}")
    if not legal:
        s.append("illegal")
    if a == "finish":
        s.append("finish")
    s.append(f"act_{a}")
    return s

def grade_tables(model, tok, seen=None, limits=(32, 48, 64, 96, 128, 40, 56, 21, 37, 45, 53, 61), cache=None,
                 chunk=480):
    """cache (optional, incident I-12): greedy outputs are appended per chunk of keys to this JSONL (fsynced) and
    reused on restart, so a crash loses at most one chunk. Without a cache all keys decode in one call, as before."""
    keys = [(x, act, L) for L in limits for x in range(-L, L + 1) for act in rw2.ACTIONS]
    done = {}
    if cache is not None and cache.exists():
        good = []
        for line in cache.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[tuple(r["k"])] = r["text"]
                good.append(line)
            except Exception:
                pass  # torn line from a crash mid-write
        tmp = cache.with_suffix(".tmp")
        tmp.write_text("".join(l + "\n" for l in good), encoding="utf-8")
        os.replace(tmp, cache)
        print(f"resumed {len(done)} of {len(keys)} keys from {cache}", flush=True)
    todo = [k for k in keys if k not in done]
    step = chunk if cache is not None else max(1, len(todo))
    for i in range(0, len(todo), step):
        part = todo[i:i + step]
        prompts = [rw2.forward_prompt(*k) for k in part]
        if cache is not None:  # CPU-lane grading: shared-prefix decoder (~97 of ~118 prompt tokens are common)
            gens = lm.greedy_shared_prefix(model, tok, prompts, max_new_tokens=24)
        else:
            gens = lm.sample_crn(model, tok, prompts, [None] * len(part), max_new_tokens=24)
        for k, g in zip(part, gens):
            done[k] = g["text"]
        if cache is not None:
            with open(cache, "a", encoding="utf-8") as fh:
                for k, g in zip(part, gens):
                    fh.write(json.dumps({"k": list(k), "text": g["text"]}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(f"  graded {min(i + step, len(todo))}/{len(todo)} new keys", flush=True)
    n, c = Counter(), Counter()
    errs = defaultdict(list)
    content = Counter()
    for k in keys:
        g = {"text": done[k]}
        pred = rw2.parse_forward(g["text"])
        ref = rw2.exact_step(*k)
        ok = pred == ref
        ss = strata(*k)
        if seen is not None and "id" in ss:
            ss.append("id_seen" if k in seen else "id_unseen")
        if seen is not None and k[0] % 5 == 0 and "id" in ss:
            ss.append("id_mod5_heldout")
        for s in ss + ["all"]:
            n[s] += 1
            c[s] += ok
            if not ok and len(errs[s]) < 8:
                errs[s].append([*k, g["text"]])
    return {"strata": {s: {"n": n[s], "acc": round(c[s] / n[s], 4)} for s in sorted(n)}, "errors": errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="titan-sparse")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limits", default="32,48,64,96,128,40,56,21,37,45,53,61")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok = lm.load_core(a.core)
    seen = None
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / a.adapter), is_trainable=False)
        m = ROOT / a.adapter / "TRAIN_KEYS.json"
        if m.exists():
            seen = {tuple(k) for k in json.loads(m.read_text())}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.with_name(out.stem + ".partial.jsonl")
    # decoder agreement check on 48 keys (shared-prefix vs plain greedy), recorded with the results
    ck = [(x, act, L) for L in (32, 96) for x in range(-L, L + 1, 7) for act in rw2.ACTIONS][:48]
    g1 = lm.greedy_shared_prefix(model, tok, [rw2.forward_prompt(*k) for k in ck], max_new_tokens=24)
    g2 = lm.sample_crn(model, tok, [rw2.forward_prompt(*k) for k in ck], [None] * len(ck), max_new_tokens=24)
    check = {"n": len(ck), "identical": sum(x["text"] == y["text"] for x, y in zip(g1, g2))}
    print("decoder check", check, flush=True)
    res = grade_tables(model, tok, seen=seen, limits=tuple(map(int, a.limits.split(","))), cache=cache)
    res["decoder"] = {"grading": "greedy_shared_prefix", "check_vs_plain": check}
    print(json.dumps(res["strata"], indent=0))
    out.write_text(json.dumps({"core": a.core, "adapter": a.adapter, **res}, indent=1))
    cache.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
