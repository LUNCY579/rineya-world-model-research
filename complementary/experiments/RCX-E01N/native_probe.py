"""RCX-E01N: native tier-A/B rule-reading probe on frozen Rineya cores (inference only, CPU).

Arms:
  OFF   Titan-Sparse FP32, no adapter (tier A, frozen core)
  T2    Titan-Sparse FP32 + registered T2 s0 LoRA, unmerged (RCWM PREREG-2 artifact; read-only)
  SUPA  Supa 1.7 (tier A, the pre-transition core)
Prompts mirror Titan's verbatim ledger one-step frame (rcwmt/env.py SYSTEM_ONE) with only the status clause
replaced by the supplied RCL rule sentence. User turn = compact sorted JSON {"before":..,"event":..} exactly as
Titan was trained. For fixed-rule strata a FRAME=verbatim control uses SYSTEM_ONE itself.

CPU only (single lane, <= 4 threads); refuses to load a model unless >= 12 GB host RAM is available (D-39);
calls rcx.guard between batches. Writes only under the given --out directory.
Usage (from complementary/):
  python -B experiments/RCX-E01N/native_probe.py --out runs/RCX-E01N/probe1 --arms OFF,T2 --n-parents 40
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.set_num_threads(int(os.environ.get("RCX_THREADS", "4")))

from rcx.guard import ThermalGuard, available_ram_gb  # noqa: E402
from rcx.rcl import gen, score as S, templates as T  # noqa: E402

CORES = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
PATHS = {"sparse": CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze",
         "supa": CORES / "Rineya-Supa-1.7.0-devfreeze"}
T2_DIR = REPO / "rcwm" / "experiments" / "RCWM-T1" / "runs" / "t2" / "t2_s0" / "adapter"
T2_SHA = "92bc7c7497c3cb2fc0c7c917b4f3b6bb931f0ed108144d969b26b18b42670b8a"
EOS_IDS = [151645, 151643]
PAD_ID = 151643
RAM_FLOOR_GB = 12.0
BATTERY_SEED = 20261240

VERBATIM_SYSTEM = ("Apply one event to the supplied account balances. If status is not posted, leave all "
                   "balances unchanged. add: add amount to account. scale: multiply account by amount. "
                   "transfer: subtract amount from account and add it to other, simultaneously. Return only "
                   "the complete resulting balances as a JSON object.")


def rcl_system(rule_sentence):
    return ("Apply one event to the supplied account balances. " + rule_sentence + " add: add amount to account. "
            "scale: multiply account by amount. transfer: subtract amount from account and add it to other, "
            "simultaneously. Return only the complete resulting balances as a JSON object.")


def user_json(rec):
    ev = dict(rec["event"])
    ev["tick"] = 1
    return json.dumps({"before": rec["state"], "event": ev}, separators=(",", ":"), sort_keys=True)


def parse_answer(text, keys):
    """RCWM-style lenient parse (PREREG-1 registered leniency subset): strip, drop trailing '.', JSON object,
    unwrap {"after":...}; require exactly the state's keys with integer values. Returns dict or None."""
    t = text.strip()
    if t.endswith("."):
        t = t[:-1]
    try:
        v = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(v, dict) and "after" in v and isinstance(v["after"], dict):
        v = v["after"]
    if not isinstance(v, dict) or set(v) != set(keys):
        return None
    if not all(isinstance(v[k], int) and not isinstance(v[k], bool) for k in keys):
        return None
    return {k: v[k] for k in keys}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def need_ram():
    ram = available_ram_gb()
    if ram is None or ram < RAM_FLOOR_GB:
        raise SystemExit(f"refusing to load a 0.6B model: available RAM {ram} GB < {RAM_FLOOR_GB} GB (D-39)")
    return ram


def load_core(name):
    from transformers import AutoModelForCausalLM
    need_ram()
    m = AutoModelForCausalLM.from_pretrained(str(PATHS[name]), dtype=torch.float32, local_files_only=True,
                                             trust_remote_code=False, low_cpu_mem_usage=True,
                                             attn_implementation="sdpa").eval()
    assert all(p.dtype == torch.float32 for p in m.parameters())
    return m


@torch.no_grad()
def generate(model, tok, prompts, guard, max_new=64, max_batch_tokens=3000, log=print):
    """Greedy decoding, left padding, batches bounded by tokens. Returns list of strings (prompt order)."""
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    outs = [None] * len(enc)
    i = 0
    t0 = time.time()
    while i < len(order):
        L = len(enc[order[i]]) + max_new
        bs = max(1, min(32, max_batch_tokens // L))
        idx = order[i:i + bs]
        guard.check()
        maxlen = max(len(enc[j]) for j in idx)
        ids = torch.full((len(idx), maxlen), PAD_ID, dtype=torch.long)
        att = torch.zeros((len(idx), maxlen), dtype=torch.long)
        for r, j in enumerate(idx):
            ids[r, maxlen - len(enc[j]):] = torch.tensor(enc[j])
            att[r, maxlen - len(enc[j]):] = 1
        out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=EOS_IDS, pad_token_id=PAD_ID)
        for r, j in enumerate(idx):
            outs[j] = tok.decode(out[r, maxlen:], skip_special_tokens=True)
        i += bs
        if (i // bs) % 10 == 0:
            log(f"  {i}/{len(order)} done, {time.time() - t0:.0f}s")
    return outs


def score_arm(battery, outs):
    res = []
    for r, o in zip(battery, outs):
        keys = list(r["gold"]["after"].keys())
        a = parse_answer(o, keys)
        res.append(S.classify(r, a))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="OFF,T2,SUPA")
    ap.add_argument("--n-parents", type=int, default=40)
    ap.add_argument("--surfaces", default="in")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N prompts")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen.use_world("arith")
    battery = [r for r in gen.make_battery("dev", args.n_parents, BATTERY_SEED)
               if r["surface"] in args.surfaces.split(",")]
    if args.limit:
        battery = battery[:args.limit]
    bpath = out / "battery.jsonl"
    with open(bpath, "w", encoding="utf-8", newline="\n") as f:
        for r in battery:
            f.write(json.dumps(r) + "\n")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(PATHS["sparse"]), local_files_only=True)

    def render(system, user):
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)

    prompts = [render(rcl_system(T.render_rule(r["rule_ast"], "t0")), user_json(r)) for r in battery]
    fixed_idx = [i for i, r in enumerate(battery) if r["group"].startswith("F")]
    prompts_verbatim = [render(VERBATIM_SYSTEM, user_json(battery[i])) for i in fixed_idx]
    guard = ThermalGuard(log=lambda m: print("guard:", m, flush=True))
    manifest = {"battery_sha256": sha_file(bpath), "n": len(battery), "surfaces": args.surfaces,
                "code_sha256": sha_file(Path(__file__)), "paths": {k: str(v) for k, v in PATHS.items()},
                "t2_adapter": str(T2_DIR), "arms": {}}
    arms = args.arms.split(",")
    model = None
    if "OFF" in arms or "T2" in arms:
        model = load_core("sparse")
        if "T2" in arms:
            got = sha_file(T2_DIR / "adapter_model.safetensors")
            assert got == T2_SHA, f"T2 adapter sha mismatch {got}"
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(T2_DIR), adapter_name="t2", is_trainable=False).eval()
        for arm in [a for a in ("OFF", "T2") if a in arms]:
            t0 = time.time()
            print(f"[{arm}] generating {len(prompts)} + {len(prompts_verbatim)} verbatim", flush=True)
            if arm == "OFF" and "T2" in arms:
                with model.disable_adapter():
                    outs = generate(model, tok, prompts, guard)
                    outs_v = generate(model, tok, prompts_verbatim, guard)
            else:
                outs = generate(model, tok, prompts, guard)
                outs_v = generate(model, tok, prompts_verbatim, guard)
            write_arm(out, arm, battery, outs, fixed_idx, outs_v, manifest, time.time() - t0)
        del model
        model = None
    if "SUPA" in arms:
        model = load_core("supa")
        t0 = time.time()
        outs = generate(model, tok, prompts, guard)
        outs_v = generate(model, tok, prompts_verbatim, guard)
        write_arm(out, "SUPA", battery, outs, fixed_idx, outs_v, manifest, time.time() - t0)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")


def write_arm(out, arm, battery, outs, fixed_idx, outs_v, manifest, wall):
    res = score_arm(battery, outs)
    summ = S.aggregate(battery, res)
    sub = [battery[i] for i in fixed_idx]
    res_v = score_arm(sub, outs_v)
    summ_v = S.aggregate(sub, res_v)
    with gzip.open(out / f"preds_{arm}.jsonl.gz", "wt", encoding="utf-8") as f:
        for i, (o, x) in enumerate(zip(outs, res)):
            f.write(json.dumps({"i": i, "out": o, **x}) + "\n")
        for k, (o, x) in enumerate(zip(outs_v, res_v)):
            f.write(json.dumps({"i": fixed_idx[k], "frame": "verbatim", "out": o, **x}) + "\n")
    comp = S.composite(summ, "in")
    manifest["arms"][arm] = {"wall_s": wall, "composite_in": comp, "cells": summ, "verbatim_fixed_cells": summ_v}
    (out / f"summary_{arm}.json").write_text(json.dumps(manifest["arms"][arm], indent=1), encoding="utf-8",
                                             newline="\n")
    print(f"[{arm}] done {wall:.0f}s composite_in {comp:.3f}", flush=True)


if __name__ == "__main__":
    main()
