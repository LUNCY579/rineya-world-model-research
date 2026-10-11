import json, random, os, sys
sys.dont_write_bytecode = True
from tokenizers import Tokenizer
T = Tokenizer.from_file(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
EOS = 151645
def enc(t): return T.encode(t, add_special_tokens=False).ids
assert enc("<|im_end|>") == [EOS], enc("<|im_end|>")
for name in ("forward_delta_train.jsonl", "forward_delta_ko_train.jsonl"):
    rows = [json.loads(l) for l in open(f"{ROOT}/runs/BW_data/{name}", encoding="utf-8")]
    L = []; S = []
    for r in rows:
        p = enc(r["prompt"]); t = enc(r["target"]) + [EOS]
        L.append(len(p) + len(t)); S.append(len(t))
    rng = random.Random(10007 * 0 + 17); order = []
    while len(order) < 600 * 16:
        idx = list(range(len(rows))); rng.shuffle(idx); order.extend(idx)
    order = order[:9600]
    print(name, "max len", max(L), "over256", sum(x > 256 for x in L), "tokens_processed", sum(L[i] for i in order), "supervised", sum(S[i] for i in order))
prompts = json.load(open(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_audit_scratch/v_d54/fake_root/prompts.json"))
ids = [enc(p) for p in prompts]
P = min(len(os.path.commonprefix(ids)), min(len(x) for x in ids) - 1)
print("eval prompts", len(ids), "shared prefix", P, "max prompt", max(map(len, ids)))
sys.path.insert(0, ROOT)
from bwmn import blocks as B
mx = 0
for p in prompts:
    u = json.loads(p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
    s = B.state_from_json(u["state"]); a = tuple(u["action"].split())
    ns, lg = B.step(s, a)
    mx = max(mx, len(enc(B.forward_target_delta(s, ns, lg))) + 1)
print("max gold target tokens incl eos", mx)
