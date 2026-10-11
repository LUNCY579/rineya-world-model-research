import sys, json, random, hashlib, itertools
from collections import Counter
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from bwmn import blocks as B
# import script's constants/render without running main (no torch import at module level)
import importlib.util
spec = importlib.util.spec_from_file_location("ko", ROOT / "scripts/bw_keyorder.py")
ko = importlib.util.module_from_spec(spec); spec.loader.exec_module(ko)
assert "torch" not in sys.modules and "peft" not in sys.modules and "transformers" not in sys.modules
print("TRAIN_ORDERS", len(ko.TRAIN_ORDERS), "HELDOUT", ko.HELDOUT)
src = ROOT / "runs/BW_data/forward_delta_train.jsonl"
dst = ROOT / "runs/BW_data/forward_delta_ko_train.jsonl"
rng = random.Random(ko.DATA_SEED)
lines = []
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    user = json.loads(r["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
    s, act = B.state_from_json(user["state"]), tuple(user["action"].split())
    assert B.forward_prompt_delta(s, act) == r["prompt"]
    if rng.random() < 0.5:
        prompt, kind = r["prompt"], "canon"
    else:
        order = rng.choice(ko.TRAIN_ORDERS)
        prompt = ko.render(s, act, order, rng.random() < 0.5, rng.random() < 0.5, rng)
        kind = "aug"
    lines.append(json.dumps({"prompt": prompt, "target": r["target"], "legal": r["legal"], "n_blocks": r["n_blocks"], "render": kind}) + "\n")
regen = "".join(lines).replace("\n", "\r\n").encode("utf-8")
fb = dst.read_bytes()
print("file sha", hashlib.sha256(fb).hexdigest(), "regen sha", hashlib.sha256(regen).hexdigest(), "identical", fb == regen)
print("src sha", hashlib.sha256(src.read_bytes()).hexdigest())
# independent checks
S = [json.loads(l) for l in open(src, encoding="utf-8")]
K = [json.loads(l) for l in open(dst, encoding="utf-8")]
print("rows", len(S), len(K))
kinds = Counter(); bad = Counter(); held = 0; canon_diff = 0; canon_unsorted = 0; aug_same_text = 0
state_orders = Counter(); top = Counter(); shuffled_aug = 0; shuffled_canon = 0
sysbad = 0
for a, b in zip(S, K):
    kinds[b["render"]] += 1
    for f in ("target", "legal", "n_blocks"):
        if a[f] != b[f]: bad[f] += 1
    ua = json.loads(a["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
    ub = json.loads(b["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
    if a["prompt"].split("<|im_start|>user\n")[0] != b["prompt"].split("<|im_start|>user\n")[0]: sysbad += 1
    if not b["prompt"].endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"): bad["suffix"] += 1
    if B.state_from_json(ua["state"]) != B.state_from_json(ub["state"]): bad["state"] += 1
    if ua["action"] != ub["action"]: bad["action"] += 1
    if set(ub) != {"state", "action"} or set(ub["state"]) != set(ko.STATE_KEYS): bad["keys"] += 1
    so = tuple(ub["state"]); 
    if so in ko.HELDOUT: held += 1
    lists_sorted = all(ub["state"][k] == sorted(ub["state"][k]) for k in ("clear", "on", "table"))
    if b["render"] == "canon":
        if a["prompt"] != b["prompt"]: canon_diff += 1
        if not lists_sorted: shuffled_canon += 1
    else:
        state_orders[so] += 1; top[tuple(ub)] += 1
        if not lists_sorted: shuffled_aug += 1
        if a["prompt"] == b["prompt"]: aug_same_text += 1
print("kinds", dict(kinds), "bad", dict(bad), "sysprompt diffs", sysbad)
print("heldout hits", held, "canon rows differing", canon_diff, "canon rows with unsorted lists", shuffled_canon)
print("aug rows with unsorted lists", shuffled_aug, "aug identical to canonical text", aug_same_text)
print("aug state orders used", len(state_orders), min(state_orders.values()), max(state_orders.values()))
print("top-level", dict(top))
print("canonical-text share", (kinds["canon"] + aug_same_text) / len(K))
# extra: HELDOUT really are 3 distinct perms and all 21 others in TRAIN_ORDERS
allp = set(itertools.permutations(ko.STATE_KEYS))
print("heldout subset of perms", ko.HELDOUT <= allp, "train+held==24", set(ko.TRAIN_ORDERS) | ko.HELDOUT == allp, len(set(ko.TRAIN_ORDERS) & ko.HELDOUT))
