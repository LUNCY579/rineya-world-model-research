"""CPU preview of the mode-vector steering test (run_steer.py) on a random TEST subset.

Runs while PREREG-1 holds the GPU: FP32 on CPU, CUDA hidden (zero visible devices), no lease, below-normal priority,
RCWM_THREADS torch threads (default 4). Offsets, hook, conditions and scoring are imported from
run_steer.py unchanged; only the device and the item subset differ. Per-item correctness is saved so
the subset can be compared paired against the full GPU run in master_chain.sh, which stays
authoritative.

  set LOGN=steer_cpu&& scripts\\cpu_job_light.cmd scripts\\run_steer_cpu.py --per-family 48
"""
from __future__ import annotations

import os

# Zero visible devices, set before torch is imported. NOT "-1": with torch 2.6.0+cu124 on this driver, "-1"
# corrupts the heap and the process dies with an access violation in a later CPU op (DECISIONS I-21).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_THREADS = int(os.environ.get("RCWM_THREADS", "4"))
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = str(_THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

# Never query the driver (keeps this process off the GPU entirely; no lease needed).
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rcwmt import bank as bankmod, env, generation, runtime  # noqa: E402
from run_steer import offsets, score, steer  # noqa: E402

FINGERPRINTS = {"supa": "a9952c55f5a1", "sparse": "818be9c6b6a3"}


def load_cpu(arm):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        str(runtime.PATHS[arm]), dtype=torch.float32, local_files_only=True, trust_remote_code=False,
        low_cpu_mem_usage=True, device_map={"": "cpu"}, attn_implementation="sdpa").eval()
    fp = runtime.param_fingerprint(m)
    assert fp.startswith(FINGERPRINTS[arm]), (arm, fp)
    return m, fp


@torch.no_grad()
def gen_with(model, tok, items, layer, vecs, batch=16):
    """run_steer.gen_with on CPU."""
    res = {}
    for fam in env.FAMILIES:
        fi = [it for it in items if it["family"] == fam]
        for s in range(0, len(fi), batch):
            b = fi[s: s + batch]
            enc = [tok(generation.render(tok, it["messages"]), add_special_tokens=False)["input_ids"] for it in b]
            Lm = max(len(x) for x in enc)
            ids = torch.full((len(b), Lm), generation.PAD, dtype=torch.long)
            att = torch.zeros((len(b), Lm), dtype=torch.long)
            for r, x in enumerate(enc):
                ids[r, Lm - len(x):] = torch.tensor(x); att[r, Lm - len(x):] = 1
            v = None if vecs is None else torch.tensor(np.stack([vecs[fam]] * len(b)), dtype=torch.float32)
            with steer(model, layer, v):
                g = model.generate(input_ids=ids, attention_mask=att, do_sample=False,
                                   max_new_tokens=generation.max_tokens(fam), eos_token_id=generation.EOS,
                                   pad_token_id=generation.PAD, temperature=None, top_p=None, top_k=None)
            for r, it in enumerate(b):
                new = g[r, Lm:].tolist()
                cut = next((j for j, t in enumerate(new) if t in generation.EOS), len(new))
                res[it["item_id"]] = tok.decode(new[:cut], skip_special_tokens=True).strip()
    return res


def correct(items, texts):
    return {it["item_id"]: bool((st := env.parse_state(it["family"], texts[it["item_id"]], lenient=True)) is not None
                                and env.dumps(st) == it["gold"]) for it in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bank_pilot")
    ap.add_argument("--layers", default="16,20,24")
    ap.add_argument("--per-family", type=int, default=48)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-name", default="steer_cpu")
    a = ap.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    torch.set_num_threads(_THREADS)
    torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    out_dir = ROOT / "runs" / a.bank / a.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = json.loads((ROOT / "data" / f"{a.bank}.json").read_text(encoding="utf-8"))
    pool = [it for it in bankmod.items(bank) if it["split"] == "test" and it["format"] == "one"]
    rng = np.random.default_rng(a.seed)
    items = []
    for fam in env.FAMILIES:
        fi = [it for it in pool if it["family"] == fam]
        items += [fi[i] for i in sorted(rng.choice(len(fi), size=min(a.per_family, len(fi)), replace=False))]
    V = offsets(a.bank)
    layers = [int(x) for x in a.layers.split(",")]
    tok = runtime.tokenizer()
    rrng = np.random.default_rng(17)                 # same random-direction stream as run_steer.py
    res_path = out_dir / "steer_cpu_results.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    ids = [it["item_id"] for it in items]
    assert results.get("_meta", {}).get("items", ids) == ids, "existing results are for a different subset"
    results["_meta"] = {"bank": a.bank, "items": ids, "threads": _THREADS,
                        "device": "cpu", "dtype": "float32", "seed": a.seed}

    def run_cond(model, name, layer, vecs):
        if name in results:
            print(f"{name:<16} done earlier; skipping", flush=True)
            return
        t0 = time.time()
        texts = gen_with(model, tok, items, layer, vecs)
        results[name] = {"score": score(items, texts), "correct": correct(items, texts)}
        (out_dir / f"gen_{name.replace(':', '_')}.json").write_text(json.dumps(texts) + "\n", encoding="utf-8")
        res_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"{name:<16} " + " | ".join(f"{f[:6]} len {v['lenient']:.3f} chg {v['changed_lenient']:.3f} (n {v['n']}/{v['changed_n']})"
                                           for f, v in results[name]["score"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)

    model, fp = load_cpu("supa")
    results["_meta"]["fingerprint_supa"] = fp
    run_cond(model, "base:supa", layers[0], None)
    rand = {}
    for L in layers:
        run_cond(model, f"add:{L}:1.0", L, {f: V[(f, L)] for f in env.FAMILIES})
        for f in env.FAMILIES:                        # drawn in run_steer.py's order whether or not the condition runs
            r = rrng.standard_normal(V[(f, L)].shape)
            rand[(f, L)] = r / np.linalg.norm(r) * np.linalg.norm(V[(f, L)])
        run_cond(model, f"rand:{L}", L, {f: rand[(f, L)] for f in env.FAMILIES})
    del model
    model, fp = load_cpu("sparse")
    results["_meta"]["fingerprint_sparse"] = fp
    run_cond(model, "base:sparse", layers[0], None)
    for L in layers:
        run_cond(model, f"sub:{L}", L, {f: -V[(f, L)] for f in env.FAMILIES})
    results["_meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print("[steer_cpu] finished", flush=True)


if __name__ == "__main__":
    main()
