"""Read-only calibration: reproduce recorded BW training token counts with the standalone `tokenizers` library
(no torch / transformers), and derive chars/token ratios. Mirrors scripts/train_generic.py row order."""
import json
import random
import sys

from tokenizers import Tokenizer

ROOT = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
TOKJ = ("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
        "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
tok = Tokenizer.from_file(TOKJ)
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))


def n(text):
    return len(tok.encode(text, add_special_tokens=False).ids)


for run in ("BW_F_s0", "BW_Fdelta_s0", "BW_Pa_v2_s1"):
    m = json.load(open(f"{ROOT}/runs/{run}/metrics.json"))
    a, t = m["args"], m["train"]
    rows = [json.loads(l) for l in open(f"{ROOT}/{a['rows']}", encoding="utf-8")]
    rng = random.Random(10007 * a["seed"] + 17)
    order = []
    while len(order) < a["steps"] * a["batch"]:
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        order.extend(idx)
    order = order[:a["steps"] * a["batch"]]
    pt = tt = pc = tc = 0
    mx = 0
    cache = {}
    for i in order:
        if i not in cache:
            r = rows[i]
            cache[i] = (n(r["prompt"]), n(r["target"]), len(r["prompt"]), len(r["target"]))
        a1, b1, c1, d1 = cache[i]
        pt += a1; tt += b1; pc += c1; tc += d1
        mx = max(mx, a1 + b1 + 1)
    allmax = max(n(r["prompt"]) + n(r["target"]) + 1 for r in rows)
    N = len(order)
    print(run, "recorded tokens_processed", t["tokens_processed"], "reproduced", pt + tt + N,
          "| recorded supervised", t["supervised_tokens"], "reproduced", tt + N)
    print("   chars/token prompt %.3f target %.3f ; mean prompt tok %.1f target tok %.1f ; max row tok (sampled) %d (all rows) %d"
          % (pc / pt, tc / tt, pt / N, tt / N, mx, allmax))
