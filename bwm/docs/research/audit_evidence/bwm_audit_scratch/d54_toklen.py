import json, random, sys
sys.dont_write_bytecode = True
from tokenizers import Tokenizer
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
tk = Tokenizer.from_file(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
def n(t): return len(tk.encode(t, add_special_tokens=False).ids)
rng = random.Random(10007 * 0 + 17)
order = []
while len(order) < 600 * 16:
    idx = list(range(12000)); rng.shuffle(idx); order.extend(idx)
for f in ("forward_delta_train.jsonl", "forward_delta_ko_train.jsonl"):
    rows = [json.loads(l) for l in open(ROOT + "/runs/BW_data/" + f, encoding="utf-8")]
    L = [n(r["prompt"]) + n(r["target"]) + 1 for r in rows]
    T = [n(r["target"]) + 1 for r in rows]
    used = order[:9600]
    print(f, "max_len", max(L), "n>256", sum(x > 256 for x in L), "tokens_processed(sim)", sum(L[i] for i in used),
          "supervised(sim)", sum(T[i] for i in used))
    if "ko" in f:
        aug = [L[i] for i, r in enumerate(rows) if r["render"] == "aug"]
        print("  aug max", max(aug), "mean", sum(aug)/len(aug), "canon-mean", sum(L[i] for i, r in enumerate(rows) if r["render"] == "canon")/sum(r["render"] == "canon" for r in rows))
