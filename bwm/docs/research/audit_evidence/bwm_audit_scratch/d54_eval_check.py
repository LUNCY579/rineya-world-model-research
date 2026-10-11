import sys, json, random, importlib.util, os
from collections import Counter
sys.dont_write_bytecode = True
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
spec = importlib.util.spec_from_file_location("bko", ROOT + "/scripts/bw_keyorder.py")
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
from tokenizers import Tokenizer
tk = Tokenizer.from_file(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
ids = lambda t: tk.encode(t, add_special_tokens=False).ids

# replicate eval_ row construction verbatim
rng = random.Random(K.EVAL_SEED)
ks = K.keys(rng)
rows = []
for mode in ("canon", "train_order", "heldout_rev", "heldout_other", "heldout_both"):
    for i, (n, objs, s, act) in enumerate(ks):
        s2, a2 = (K.rename(s, act, objs, rng.sample(K.UNSEEN, len(objs))) if mode == "heldout_both" else (s, act))
        if mode == "canon":
            p = B.forward_prompt_delta(s2, a2)
        elif mode == "train_order":
            p = K.render(s2, a2, rng.choice([o for o in K.TRAIN_ORDERS if o != K.STATE_KEYS]), rng.random() < 0.5, True, rng)
        elif mode == "heldout_rev":
            p = K.render(s2, a2, ("table", "on", "hand", "clear"), True, False, rng)
        elif mode == "heldout_other":
            p = K.render(s2, a2, rng.choice([("hand", "table", "clear", "on"), ("on", "clear", "table", "hand")]), rng.random() < 0.5, False, rng)
        else:
            p = K.render(s2, a2, ("table", "on", "hand", "clear"), True, True, rng)
        rows.append({"mode": mode, "i": i, "n": n, "s": s2, "a": a2, "prompt": p, "s0": s, "a0": act, "objs": objs})
print("keys", len(ks), Counter(k[0] for k in ks), "legal frac", sum(B.applicable(k[2], k[3]) for k in ks) / len(ks))
# F-08 key_order render, copied from scripts/bw_surface_audit.py lines 30-43 (not imported: it imports bwmn.lm/torch)
def f08_render(s, a, mode, rng):
    st = B.state_json(s)
    if mode in ("list_order", "both"):
        for k in ("clear", "on", "table"):
            rng.shuffle(st[k])
    user = {"action": B.action_str(a), "state": st}
    if mode in ("key_order", "both"):
        user = {"state": dict(reversed(list(st.items()))), "action": B.action_str(a)}
        text = json.dumps(user, separators=(",", ":"))
    elif mode == "list_order":
        text = json.dumps(user, separators=(",", ":"), sort_keys=True)
    else:
        text = B.canonical(user)
    return B.render_chat(B.SYS_FORWARD_DELTA, text)
U = lambda p: json.loads(p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
stats = {}
for mode in ("canon", "train_order", "heldout_rev", "heldout_other", "heldout_both"):
    R = [r for r in rows if r["mode"] == mode]
    so = Counter(tuple(U(r["prompt"])["state"]) for r in R)
    tl = Counter(tuple(U(r["prompt"])) for r in R)
    parse_ok = sum(B.state_from_json(U(r["prompt"])["state"]) == r["s"] and U(r["prompt"])["action"] == B.action_str(r["a"]) for r in R)
    in_held = sum(o in K.HELDOUT for o in so.elements())
    in_train = sum(o in K.TRAIN_ORDERS for o in so.elements())
    shuffled = sum(any(U(r["prompt"])["state"][x] != sorted(U(r["prompt"])["state"][x]) for x in ("clear", "on", "table")) for r in R)
    # oracle answer parsed back with the (renamed) state
    oracle = sum(B.parse_forward_delta(B.forward_target_delta(r["s"], *B.step(r["s"], r["a"])), r["s"]) == B.step(r["s"], r["a"]) for r in R)
    tmax = max(len(ids(B.forward_target_delta(r["s"], *B.step(r["s"], r["a"])))) for r in R)
    pmax = max(len(ids(r["prompt"])) for r in R)
    names = Counter(x for r in R for f in r["s"] for x in f[1:])
    unseen_only = all(x not in B.NAMES for x in names)
    print(mode, "| state orders", dict(so) if len(so) <= 3 else f"{len(so)} distinct", "| top", dict(tl), "| in_heldout", in_held,
          "in_train", in_train, "| shuffled rows", shuffled, "| parse-back", parse_ok, "| oracle ok", oracle,
          "| max target toks", tmax, "| max prompt toks", pmax, "| names unseen-only", unseen_only)
f08_eq = sum(f08_render(r["s"], r["a"], "key_order", random.Random(0)) == r["prompt"] for r in rows if r["mode"] == "heldout_rev")
print("heldout_rev == F-08 key_order render:", f08_eq, "/ 240")
# check both-mode vs F-08 'both' structure (same key order & top-level)
b = [r for r in rows if r["mode"] == "heldout_both"]
print("heldout_both state order", Counter(tuple(U(r["prompt"])["state"]) for r in b), "top", Counter(tuple(U(r["prompt"])) for r in b))
# shared prefix
allids = [ids(r["prompt"]) for r in rows]
P = min(len(os.path.commonprefix(allids)), min(len(x) for x in allids) - 1)
print("shared prefix tokens", P)
# overlap of eval keys with training transitions
train = set()
for l in open(ROOT + "/runs/BW_data/forward_delta_train.jsonl", encoding="utf-8"):
    r = json.loads(l); u = U(r["prompt"]); train.add((B.state_from_json(u["state"]), u["action"]))
ov = sum((k[2], B.action_str(k[3])) in train for k in ks)
print("eval keys present in training transitions:", ov)
# overlap with F-08 keys (seed 20261009)
r9 = random.Random(20261009)
f08 = []
for n in (5, 10):
    for _ in range(60):
        objs = sorted(r9.sample(B.NAMES, n)); s = B.towers_to_state(B.random_towers(objs, r9))
        for _ in range(r9.randrange(0, 5)):
            s, _ = B.step(s, r9.choice([x for x in B.ground_actions(objs) if B.applicable(s, x)]))
        ga = B.ground_actions(objs); leg = [x for x in ga if B.applicable(s, x)]; ill = [x for x in ga if not B.applicable(s, x)]
        act = r9.choice(leg) if (r9.random() < 0.5 or not ill) else r9.choice(ill)
        f08.append((s, act))
print("eval keys shared with F-08 keys:", sum((k[2], k[3]) in set(f08) for k in ks))
# legal fraction per size
for nn in (5, 10):
    sub = [k for k in ks if k[0] == nn]; print("n", nn, "legal", sum(B.applicable(k[2], k[3]) for k in sub), "/", len(sub))
