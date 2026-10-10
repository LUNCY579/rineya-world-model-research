"""Short CPU benchmark of char_lm: train-step time and greedy-decode throughput (4 threads).

Run: python -B -m rcx.models.bench_char_lm   (about a minute; writes rcx/models/BENCH_char_lm.json)
Shape: batch 32, input 200 + target 30 random printable chars (232 positions per row).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import platform
import random
import statistics
import time

import torch

THREADS = 4
torch.set_num_threads(THREADS)

from rcx.models import char_lm as lm

CONFIGS = [(128, 3, 4, 512), (160, 4, 4, 640), (192, 4, 6, 768)]
BATCH, IN_LEN, TGT_LEN, MAX_LEN = 32, 200, 30, 256
TRAIN_STEPS, WARM_STEPS = 8, 2
DECODE_ROWS = 64
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BENCH_char_lm.json")


def _records(rng, n):
    pick = lambda k: "".join(chr(rng.randint(32, 126)) for _ in range(k))
    return [{"input": pick(IN_LEN), "target": pick(TGT_LEN)} for _ in range(n)]


def bench_config(d, L, H, ff, rng):
    """Time TRAIN_STEPS train() steps (median after warm-up) and one batched greedy decode of DECODE_ROWS rows."""
    mcfg = lm.CharLMConfig(d_model=d, n_layers=L, n_heads=H, d_ff=ff, max_len=MAX_LEN)
    cfg = lm.TrainConfig(model=mcfg, steps=TRAIN_STEPS, batch_size=BATCH, lr=1e-3, warmup=WARM_STEPS, log_every=1)
    model, hist = lm.train(_records(rng, 4 * BATCH), cfg, seed=0)
    step_s = [r["step_s"] for r in hist["log"][WARM_STEPS:]]
    med = statistics.median(step_s)
    tok_per_step = hist["tokens"] / hist["steps"]

    torch.manual_seed(0)
    dec = lm.CharLM(mcfg).eval()
    with torch.no_grad():
        dec.emb.weight[lm.EOS] = 0.0
    inputs = [r["input"] for r in _records(rng, DECODE_ROWS)]
    lm.generate(dec, inputs[:4], 4, batch_size=BATCH)
    t0 = time.perf_counter()
    texts, done = lm.generate(dec, inputs, TGT_LEN + 1, batch_size=BATCH, return_done=True)
    dt = time.perf_counter() - t0
    gen_tokens = sum(map(len, texts)) + sum(done)
    return {"d_model": d, "n_layers": L, "n_heads": H, "d_ff": ff, "params": lm.param_count(model),
            "train_step_s_median": round(med, 4), "train_step_s": [round(s, 4) for s in step_s],
            "train_tokens_per_step": tok_per_step, "train_tok_s": round(tok_per_step / med),
            "train_target_tok_s": round(hist["target_tokens"] / hist["steps"] / med),
            "est_s_per_1000_steps": round(1000 * med, 1),
            "decode_rows": DECODE_ROWS, "decode_batch": BATCH, "decode_new_tokens": gen_tokens,
            "decode_s": round(dt, 3), "decode_tok_s": round(gen_tokens / dt), "decode_rows_s": round(DECODE_ROWS / dt, 1)}


def main():
    """Run all configs, print a table and write BENCH_char_lm.json."""
    rng = random.Random(0)
    t0 = time.perf_counter()
    results = []
    for d, L, H, ff in CONFIGS:
        r = bench_config(d, L, H, ff, rng)
        results.append(r)
        print(f"d={d} L={L} H={H} ff={ff} params={r['params']:,} train_step={r['train_step_s_median']:.3f}s "
              f"({r['train_tok_s']:,} tok/s) decode={r['decode_tok_s']:,} new tok/s "
              f"({r['decode_rows_s']} rows/s, {r['decode_s']:.2f}s)", flush=True)
    report = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__, "threads": THREADS,
              "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
              "shape": {"batch": BATCH, "input_len": IN_LEN, "target_len": TGT_LEN,
                        "positions_per_row": IN_LEN + TGT_LEN + 2, "max_len": MAX_LEN},
              "method": (f"train(): {TRAIN_STEPS} steps, median step time after {WARM_STEPS} warm-up steps "
                         "(collate + fwd + bwd + AdamW, fp32). Decode: generate() greedy KV-cache on a "
                         f"random-init model with the EOS row zeroed so every row emits {TGT_LEN + 1} tokens; "
                         "time includes prefill."),
              "total_s": round(time.perf_counter() - t0, 1), "results": results}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"total {report['total_s']}s -> {OUT}")


if __name__ == "__main__":
    main()
