import sys, json, hashlib, importlib.util, random
from collections import Counter
from pathlib import Path
NV = Path("C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
spec = importlib.util.spec_from_file_location("bko", NV/"scripts/bw_keyorder.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
B = m.B
rng = random.Random(m.DATA_SEED)
src = NV/"runs/BW_data/forward_delta_train.jsonl"
h = hashlib.sha256(); kinds=Counter(); orders=set(); ident=0; maxlen=0; same_target=True
real = open(NV/"runs/BW_data/forward_delta_ko_train.jsonl", encoding="utf-8")
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    user = json.loads(r["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
    s, act = B.state_from_json(user["state"]), tuple(user["action"].split())
    if rng.random() < 0.5:
        prompt, kind = r["prompt"], "canon"
    else:
        order = rng.choice(m.TRAIN_ORDERS)
        prompt = m.render(s, act, order, rng.random() < 0.5, rng.random() < 0.5, rng)
        kind = "aug"
        if prompt == r["prompt"]: ident += 1
        u = json.loads(prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
        orders.add((tuple(u.keys()), tuple(u["state"].keys())))
    kinds[kind]+=1
    out = json.dumps({"prompt": prompt, "target": r["target"], "legal": r["legal"], "n_blocks": r["n_blocks"], "render": kind}) + "\n"
    h.update(out.encode("utf-8"))
    rl = real.readline()
    if rl != out: print("MISMATCH"); break
print(kinds, "aug orders", len(orders), "aug==canon", ident, "sha", h.hexdigest()[:8])
