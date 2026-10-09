"""A1: factorized forward model for RW2 without training (Track A, D-27).

For every move key (x, action, limit) with action in {add1, sub1, double}:
  (i)   P(legal=true): F_LM's probability of `true` vs `false` right after `{"legal":`        -> F_tau
  (ii)  candidate c: F_LM decoding forced to start `{"legal":true,"terminated":false,"x":`   -> next value if legal
  (iii) verifier (adapter OFF, frozen Titan): "Is {|c|} less than or equal to {L}?"          -> F_fact
        and an all-LM phrasing "Is the absolute value of {c} at most {L}?"                  -> F_fact2
  F_rule (|c| <= L checked in code) is recorded as a CEILING, not a learned model.
halve and finish keep F_LM's table answer. Per-key results go to keys.jsonl (resumable). `--analyze` builds the
forward tables, grades FLIP-EVAL v1 strata, picks tau on calibration limits 80/112 (used by no task), and runs
G1F/G2F/BFS-16/64/256 on the VALIDATION splits only.

Usage (CPU lane):  BWM_DEVICE=cpu python scripts/f_factorized.py --out runs/A1_factorized
                   python scripts/f_factorized.py --out runs/A1_factorized --analyze
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import planner, rw2  # noqa: E402

MOVES = ("add1", "sub1", "double")
CURVE_LIMITS = (76, 84, 100, 116)
CALIB_LIMITS = (80, 112)
ODD_LIMITS = (21, 37, 45, 53, 61)
TABLE_LIMITS = (32, 48, 64, 96, 128)
VERIFIER_SYS = "You are a precise calculator. Reply with the answer only."


def build_keys():
    keys = {}
    for L in TABLE_LIMITS:
        for x in range(-L, L + 1):
            for a in MOVES:
                keys[(x, a, L)] = "table"
    for L in CURVE_LIMITS:
        for x in range(-64, 65):
            keys.setdefault((x, "double", L), "curve")
    for L in CALIB_LIMITS:
        for x in range(-64, 65):
            keys.setdefault((x, "double", L), "calib")
    for L in ODD_LIMITS:
        for x in range(-L, L + 1):
            keys.setdefault((x, "double", L), "odd")
        for x in (L, L - 1, -L, -(L - 1)):
            for a in ("add1", "sub1"):
                keys.setdefault((x, a, L), "odd")
    return keys


LOWCAL_LIMITS = (24, 28)  # below-range calibration limits (F-04 follow-up): used by no task and no grading table


def build_lowcal_keys():
    """Doubles at below-range calibration limits, where the training-range shortcut predicts 'legal' confidently."""
    return {(x, "double", L): "lowcal" for L in LOWCAL_LIMITS for x in range(-L, L + 1)}


VLAB_LIMITS = (72, 80, 88, 104, 112, 120)
GATE_LO, GATE_HI = 0.01, 0.5  # uncertainty band selected on calibration limits (scripts/a1_gated.py)


def build_vlab_keys():
    """A2 self-labelling keys: moves with |x| <= 64 at limits 72..120 (no task uses these limits; no ground truth
    above 64 is ever used: labels come from the frozen backbone)."""
    return {(x, a, L): "vlab" for L in VLAB_LIMITS for x in range(-64, 65) for a in MOVES}


def emit_vlab(a):
    """Training rows for F_vlab from the factorized pass (uncertainty-gated, F-03). Doubles with F_LM's P(legal) in
    (GATE_LO, GATE_HI): verifier legality with F's forced candidate as next value; every other key: F_LM's own
    prediction (add1/sub1 are always legal for |x| <= 64 at L >= 72, a fact the labeller is not told).
    Rows whose label cannot be parsed are dropped and counted."""
    out = ROOT / a.out
    rows, dropped = [], Counter()
    for line in open(out / "keys.jsonl", encoding="utf-8"):
        r = json.loads(line)
        x, act, L = r["key"]
        if act == "double" and GATE_LO < r["p_true"] < GATE_HI:
            # F-03: consult the frozen-backbone verifier only where F_LM is uncertain (band chosen on calibration
            # limits 80/112); the blanket verifier was 0.63 accurate on calibration doubles, the gated rule 0.98.
            leg = _bool(r["ver1"])
            if leg is None or r["cand"] is None:
                dropped["double_unparsed"] += 1
                continue
            nx = r["cand"] if leg else x
        else:
            fr = rw2.parse_forward(r.get("free", "") or "")
            if fr is None:
                dropped["move_unparsed"] += 1
                continue
            nx, leg, _ = fr
        rows.append({"x": x, "action": act, "limit": L, "next_x": nx, "legal": leg, "terminated": False})
    agree = sum((rw.get("next_x"), rw.get("legal")) == rw2.exact_step(rw["x"], rw["action"], rw["limit"])[:2]
                for rw in rows)
    print(f"vlab rows {len(rows)}, dropped {dict(dropped)}, label agreement with exact law (analysis only) "
          f"{agree}/{len(rows)}")
    with open(out / "vlab_rows.jsonl", "w", encoding="utf-8") as fh:
        for rw in rows:
            fh.write(json.dumps(rw) + "\n")
    # SFT file: the exact in-distribution table (limits 32/48/64; F_titan_main saw every one of these keys) plus the
    # self-labelled rows at limits 72..120. No exact label above 64 is used.
    sft = []
    for L in rw2.ID_LIMITS:
        for x in range(-L, L + 1):
            for act in rw2.ACTIONS:
                nx, leg, term = rw2.exact_step(x, act, L)
                sft.append({"prompt": rw2.forward_prompt(x, act, L), "target": rw2.forward_target(nx, leg, term)})
    for rw in rows:
        sft.append({"prompt": rw2.forward_prompt(rw["x"], rw["action"], rw["limit"]),
                    "target": rw2.forward_target(rw["next_x"], rw["legal"], rw["terminated"])})
    with open(out / "vlab_sft.jsonl", "w", encoding="utf-8") as fh:
        for rw in sft:
            fh.write(json.dumps(rw) + "\n")
    print(f"vlab_sft rows {len(sft)}")


def stratum(x, a, L):
    """FLIP-EVAL v1 strata."""
    nx, legal, _ = rw2.exact_step(x, a, L)
    s = []
    if L in (32, 48, 64):
        s.append("id")
    if L in ODD_LIMITS:
        s.append("odd_limit")
    if L in CALIB_LIMITS:
        s.append("calib")
    if a == "double" and legal and abs(2 * x) > 64 and L not in (32, 48, 64):
        s.append("flip_type" if L in (96, 128) + CURVE_LIMITS else "flip_calib")
        if L in (96, 128):
            s.append("flip96")
    if L in (96, 128):
        s.append("range")
        if abs(x) > 64:
            s.append("range_big")
    if not legal:
        s.append("illegal")
    return s


# ----------------------------------------------------------------------------- model passes

def run(a):
    import torch
    from transformers import DynamicCache
    from peft import PeftModel
    from bwmn import lm

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    kf = out / "keys.jsonl"
    if kf.exists():
        for line in open(kf, encoding="utf-8"):
            r = json.loads(line)
            done.add(tuple(r["key"]))
    keyset = {"a1": build_keys, "vlab": build_vlab_keys, "lowcal": build_lowcal_keys}[a.keyset]()
    keys = [k for k in keyset if k not in done]
    if a.max_keys:
        keys = keys[:a.max_keys]
    print(f"{len(done)} keys done, {len(keys)} to go", flush=True)
    base, tok = lm.load_core("titan-sparse")
    model = PeftModel.from_pretrained(base, str(ROOT / a.fwd), adapter_name="fwd", is_trainable=False).eval()
    dev = model.device

    ex_true = tok('{"legal":true,"terminated":false,"x":12}', add_special_tokens=False).input_ids
    ex_false = tok('{"legal":false,"terminated":false,"x":12}', add_special_tokens=False).input_ids
    k = 0
    while k < min(len(ex_true), len(ex_false)) and ex_true[k] == ex_false[k]:
        k += 1
    legal_prefix, true_id, false_id = ex_true[:k], ex_true[k], ex_false[k]
    assert tok.decode(legal_prefix) == '{"legal":', tok.decode(legal_prefix)
    j = max(i for i in range(len(ex_true) + 1) if tok.decode(ex_true[:i]) == '{"legal":true,"terminated":false,"x":')
    forced = ex_true[:j]

    @torch.no_grad()
    def last_logits(id_lists):
        L = max(len(s) for s in id_lists)
        ids = torch.full((len(id_lists), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(id_lists), L), dtype=torch.long)
        for i, s in enumerate(id_lists):
            ids[i, L - len(s):] = torch.tensor(s)
            att[i, L - len(s):] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        o = model(input_ids=ids.to(dev), attention_mask=att.to(dev), position_ids=pos.to(dev), logits_to_keep=1)
        return o.logits[:, -1, :].float()

    @torch.no_grad()
    def greedy(id_lists, max_new):
        L = max(len(s) for s in id_lists)
        B = len(id_lists)
        ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        for i, s in enumerate(id_lists):
            ids[i, L - len(s):] = torch.tensor(s)
            att[i, L - len(s):] = 1
        ids, att = ids.to(dev), att.to(dev)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        cache = DynamicCache()
        o = model(input_ids=ids, attention_mask=att, position_ids=pos, past_key_values=cache, use_cache=True,
                  logits_to_keep=1)
        outs = [[] for _ in range(B)]
        fin = [False] * B
        last = pos[:, -1:]
        for _ in range(max_new):
            nxt = o.logits[:, -1, :].argmax(-1)
            for i in range(B):
                if not fin[i]:
                    t = int(nxt[i])
                    if t in lm.EOS_IDS:
                        fin[i] = True
                    else:
                        outs[i].append(t)
            if all(fin):
                break
            att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device=dev)], -1)
            last = last + 1
            o = model(input_ids=nxt[:, None], attention_mask=att, position_ids=last, past_key_values=cache,
                      use_cache=True, logits_to_keep=1)
        return [tok.decode(s) for s in outs]

    fh = open(kf, "a", encoding="utf-8")
    B = a.batch
    t0 = time.time()
    for b0 in range(0, len(keys), B):
        chunk = keys[b0:b0 + B]
        prompts = [tok(rw2.forward_prompt(*kk), add_special_tokens=False).input_ids for kk in chunk]
        model.set_adapter("fwd")
        lg = last_logits([p + legal_prefix for p in prompts])
        pt = torch.softmax(lg[:, [true_id, false_id]], -1)[:, 0].tolist()
        free_txt = greedy(prompts, 24)
        cand_txt = greedy([p + forced for p in prompts], 8)
        cands = []
        for c in cand_txt:
            try:
                s = c.split("}")[0].strip()
                cands.append(int(s))
            except Exception:
                cands.append(None)
        q1, q2, idx = [], [], []
        for i, (kk, c) in enumerate(zip(chunk, cands)):
            if c is None:
                continue
            L = kk[2]
            q1.append(tok(rw2.render_chat(VERIFIER_SYS, f"Is {abs(c)} less than or equal to {L}? Answer true or false."),
                          add_special_tokens=False).input_ids)
            q2.append(tok(rw2.render_chat(VERIFIER_SYS, f"Is the absolute value of {c} at most {L}? Answer true or false."),
                          add_special_tokens=False).input_ids)
            idx.append(i)
        v1 = [None] * len(chunk)
        v2 = [None] * len(chunk)
        if idx:
            with model.disable_adapter():
                r1 = greedy(q1, 3)
                r2 = greedy(q2, 3)
            for i, x1, x2 in zip(idx, r1, r2):
                v1[i], v2[i] = x1.strip().lower(), x2.strip().lower()
        for kk, p, ft, ct, c, x1, x2 in zip(chunk, pt, free_txt, cand_txt, cands, v1, v2):
            fh.write(json.dumps({"key": list(kk), "p_true": p, "free": ft, "cand_text": ct, "cand": c, "ver1": x1,
                                 "ver2": x2}) + "\n")
        fh.flush()
        lm.gpu_yield()
        n = b0 + len(chunk)
        if (b0 // B) % 10 == 0:
            el = time.time() - t0
            print(f"{n}/{len(keys)} keys, {el:.0f}s, eta {el / n * (len(keys) - n):.0f}s", flush=True)
    print("done", flush=True)


# ----------------------------------------------------------------------------- analysis

def _bool(v):
    if v is None:
        return None
    return True if v.startswith("true") else False if v.startswith("false") else None


def analyze(a):
    out = ROOT / a.out
    rows = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(out / "keys.jsonl", encoding="utf-8")}
    flm = {tuple(k): (tuple(v) if v else None) for k, v in json.loads((ROOT / a.fwd).parent.joinpath("F_TABLE.json").read_text())}

    def pred(kind, key, tau=0.5):
        x, act, L = key
        r = rows.get(key)
        if act in ("halve", "finish") or r is None:
            return flm.get(key)
        c = r["cand"]
        fr = rw2.parse_forward(r.get("free", "") or "")
        lm_pred = fr if fr is not None or key not in flm else flm.get(key)
        if kind == "lm":
            return lm_pred
        # Smoke test (2026-10-06): the verifier fails at |c| == L and the forced decode copies x for illegal add1/sub1,
        # so every factorized variant applies its legality source to DOUBLES only; other moves keep F_LM (99.7%).
        if act != "double" and not kind.endswith("_all"):
            return lm_pred
        kind = kind.replace("_all", "")
        if kind == "tau":
            legal = r["p_true"] > tau
        elif kind == "fact":
            legal = _bool(r["ver1"])
        elif kind == "fact2":
            legal = _bool(r["ver2"])
        elif kind == "rule":
            legal = None if c is None else abs(c) <= L
        if legal is None or (legal and c is None):
            return None
        return (c, True, False) if legal else (x, False, False)

    # tau on calibration limits (no task uses them): maximise accuracy on calib doubles
    calib = [k for k in rows if k[2] in CALIB_LIMITS]
    best = max((sum(pred("tau", k, t) == rw2.exact_step(*k) for k in calib), t) for t in [i / 100 for i in range(1, 100)])
    tau = best[1]
    res = {"tau": tau, "tau_calib_acc": best[0] / len(calib)}
    kinds = ("lm", "tau", "fact", "fact2", "rule", "fact_all", "rule_all")
    strata = defaultdict(lambda: Counter())
    for key in rows:
        ref = rw2.exact_step(*key)
        for s in stratum(*key) + ["all"]:
            strata[s]["n"] += 1
            for kd in kinds:
                p = pred(kd, key, tau) if kd == "tau" else pred(kd, key)
                strata[s][f"n_{kd}"] += 1
                strata[s][kd] += (p == ref)
    res["strata"] = {s: {kd: (round(c[kd] / c[f"n_{kd}"], 4) if c[f"n_{kd}"] else None) for kd in kinds} | {"n": c["n"]}
                     for s, c in sorted(strata.items())}
    for s, v in res["strata"].items():
        print(f"{s:12s} n={v['n']:5d} " + " ".join(f"{kd}={v[kd]}" for kd in kinds))

    def wilson_lb(k, n, z=1.959964):
        if n == 0:
            return None
        p = k / n
        return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    flips = [k for k in rows if "flip_type" in stratum(*k)]
    for kd in ("fact", "fact2", "tau", "rule"):
        kk = sum((pred(kd, k, tau) if kd == "tau" else pred(kd, k)) == rw2.exact_step(*k) for k in flips)
        res[f"gate_{kd}"] = {"flip_type": f"{kk}/{len(flips)}", "wilson_lb": round(wilson_lb(kk, len(flips)), 4)}
    print(json.dumps({k: v for k, v in res.items() if k.startswith("gate") or k.startswith("tau")}))

    # planners on VALIDATION splits only, tables over the RW2 limits
    tables = {}
    for kd in kinds:
        t = {}
        for L in TABLE_LIMITS:
            for x in range(-L, L + 1):
                for act in rw2.ACTIONS:
                    key = (x, act, L)
                    t[key] = pred(kd, key, tau) if kd == "tau" else pred(kd, key)
        tables[kd] = t
    pools = rw2.build_pools()
    plan = {}
    for split, (pool, n) in {"id_val": ("id", 400), "long_val": ("long", 200), "range_val": ("range", 300),
                             "limitonly_val": ("limitonly", 300)}.items():
        tasks = rw2.eval_tasks(pools, pool, "validation", n, seed=0)
        for kd in kinds:
            for name, fn in (("G2F", lambda f: planner.greedy_under_f(tasks, f, 2)),
                             ("BFS64F", lambda f: planner.bfs_under_f(tasks, f, 64)),
                             ("BFS256F", lambda f: planner.bfs_under_f(tasks, f, 256))):
                f = planner.TableForward(tables[kd])
                sm = planner.summarize(fn(f))
                plan[f"{split}/{name}/{kd}"] = {"success": sm["success"], "f_calls": sm["f_calls_per_task"]}
        print(split, {k.split("/", 1)[1]: round(v["success"], 3) for k, v in plan.items() if k.startswith(split)}, flush=True)
    res["planning_validation"] = plan
    (out / "analysis.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", default="runs/F_titan_main/adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--max-keys", type=int, default=0, help="smoke test: only the first N keys")
    ap.add_argument("--keyset", choices=["a1", "vlab", "lowcal"], default="a1")
    ap.add_argument("--emit-vlab", action="store_true")
    a = ap.parse_args()
    if a.emit_vlab:
        emit_vlab(a)
    elif a.analyze:
        analyze(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
