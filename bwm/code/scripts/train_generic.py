"""Generic LoRA SFT on a prompt/target JSONL (frozen core, D-03 recipe, chunked CE, duty-capped).

Usage: python scripts/train_generic.py --rows runs/BW_data/forward_train.jsonl --steps 600 --seed 0 --out runs/BW_F_s0
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import lm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="titan-sparse")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--filter", default="", help="optional key=value filter on rows, e.g. outcome=success")
    ap.add_argument("--init-adapter", default="", help="continue training this saved LoRA adapter (B3) instead of a fresh one")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def P(*s):
        m = " ".join(str(x) for x in s)
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    rows = [json.loads(l) for l in open(ROOT / a.rows, encoding="utf-8")]
    if a.filter:
        k, v = a.filter.split("=", 1)
        rows = [r for r in rows if str(r.get(k)) == v]
    rng = random.Random(10007 * a.seed + 17)
    order = []
    while len(order) < a.steps * a.batch:
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        order.extend(idx)
    P(f"rows {len(rows)} from {a.rows}; steps {a.steps} x {a.batch} = {a.steps * a.batch / len(rows):.2f} epochs")
    model, tok = lm.load_core(a.core)
    if a.init_adapter:
        from peft import PeftModel

        init_sha = lm.file_sha256(ROOT / a.init_adapter / "adapter_model.safetensors")
        P(f"init adapter {a.init_adapter} sha256 {init_sha}")
        pm = PeftModel.from_pretrained(model, str(ROOT / a.init_adapter), is_trainable=True)
    else:
        pm = lm.attach_lora(model, seed=a.seed)
    t0 = time.time()
    log = lm.train_sft(pm, tok, rows, steps=a.steps, batch_size=a.batch, lr=a.lr, seed=a.seed, logger=P,
                       row_order=order, max_len=a.max_len, ckpt_dir=out / "ckpt")
    meta = {"core": a.core, "args": vars(a), "rows": len(rows), "train": log}
    if a.init_adapter:
        meta["init_adapter_sha256"] = init_sha
    lm.save_adapter(pm, out / "adapter", meta)
    shutil.rmtree(out / "ckpt", ignore_errors=True)
    (out / "metrics.json").write_text(json.dumps(meta, indent=1))
    P("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
