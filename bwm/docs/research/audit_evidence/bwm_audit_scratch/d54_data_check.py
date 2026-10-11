import sys, json, hashlib, itertools, random, importlib.util
from collections import Counter
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.dont_write_bytecode = True
sys.path.insert(0, ROOT)
from bwmn import blocks as B
spec = importlib.util.spec_from_file_location("bko", ROOT + "/scripts/bw_keyorder.py")
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)

def user_of(p):
    return p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
def sys_of(p):
    return p.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]

src = [json.loads(l) for l in open(ROOT + "/runs/BW_data/forward_delta_train.jsonl", encoding="utf-8")]
ko = [json.loads(l) for l in open(ROOT + "/runs/BW_data/forward_delta_ko_train.jsonl", encoding="utf-8")]
print("rows", len(src), len(ko))
assert len(src) == len(ko)
bad = Counter(); kinds = Counter(); orders = Counter(); top = Counter(); shuf = Counter(); exact_canon_aug = 0
heldout_hits = 0; canon_mismatch = 0
for i, (r, k) in enumerate(zip(src, ko)):
    for f in ("target", "legal", "n_blocks"):
        if r[f] != k[f]: bad[f] += 1
    if set(k) != {"prompt", "target", "legal", "n_blocks", "render"}: bad["keys"] += 1
    if sys_of(k["prompt"]) != B.SYS_FORWARD_DELTA: bad["sys"] += 1
    if not k["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"): bad["tail"] += 1
    ur, uk = json.loads(user_of(r["prompt"])), json.loads(user_of(k["prompt"]))
    sr, sk = B.state_from_json(ur["state"]), B.state_from_json(uk["state"])
    if sr != sk: bad["state"] += 1
    if ur["action"] != uk["action"]: bad["action"] += 1
    # target consistent with simulator
    act = tuple(uk["action"].split())
    ns, leg = B.step(sk, act)
    if B.forward_target_delta(sk, ns, leg) != k["target"] or leg != k["legal"]: bad["target_vs_sim"] += 1
    kinds[k["render"]] += 1
    so = tuple(uk["state"]); tl = tuple(uk)
    if so in K.HELDOUT: heldout_hits += 1
    if k["render"] == "canon":
        if k["prompt"] != r["prompt"]: canon_mismatch += 1
    else:
        orders[so] += 1; top[tl] += 1
        sorted_lists = all(uk["state"][x] == sorted(uk["state"][x]) for x in ("clear", "on", "table"))
        shuf["sorted" if sorted_lists else "shuffled"] += 1
        if k["prompt"] == r["prompt"]: exact_canon_aug += 1
    if k["render"] == "canon":
        if not all(uk["state"][x] == sorted(uk["state"][x]) for x in ("clear", "on", "table")): bad["canon_unsorted"] += 1
print("bad", dict(bad)); print("kinds", dict(kinds)); print("heldout_hits", heldout_hits, "canon_mismatch", canon_mismatch)
print("aug rows identical to canonical text", exact_canon_aug)
print("aug distinct state orders", len(orders), "min/max count", min(orders.values()), max(orders.values()))
print("aug canonical-state-order rows", orders.get(K.STATE_KEYS, 0))
print("aug top-level", dict(top)); print("aug lists", dict(shuf))
print("train orders in script", len(K.TRAIN_ORDERS), "missing from data", [o for o in K.TRAIN_ORDERS if o not in orders])
# exact regeneration (in memory) with the script's logic and seed
rng = random.Random(K.DATA_SEED); buf = []
for r in src:
    user = json.loads(user_of(r["prompt"]))
    s, act = B.state_from_json(user["state"]), tuple(user["action"].split())
    assert B.forward_prompt_delta(s, act) == r["prompt"]
    if rng.random() < 0.5:
        prompt, kind = r["prompt"], "canon"
    else:
        order = rng.choice(K.TRAIN_ORDERS)
        prompt = K.render(s, act, order, rng.random() < 0.5, rng.random() < 0.5, rng); kind = "aug"
    buf.append(json.dumps({"prompt": prompt, "target": r["target"], "legal": r["legal"], "n_blocks": r["n_blocks"], "render": kind}) + "\n")
regen = "".join(buf).replace(chr(10), chr(13) + chr(10)).encode("utf-8")
filebytes = open(ROOT + "/runs/BW_data/forward_delta_ko_train.jsonl", "rb").read()
print("regen identical to file:", regen == filebytes, hashlib.sha256(filebytes).hexdigest()[:16], len(filebytes), len(regen))
print("n_blocks dist", Counter(r["n_blocks"] for r in src))
print("legal frac", sum(r["legal"] for r in src) / len(src))
