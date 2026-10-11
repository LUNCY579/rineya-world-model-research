import sys, json, random, types, itertools
from collections import Counter, defaultdict
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
SCR = Path(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_audit_scratch/v_d54/fake_root")
sys.path.insert(0, str(ROOT))
import bwmn
from bwmn import blocks as B
calls = []
active = {"name": None}
def parse_prompt(p):
    u = json.loads(p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
    return u, B.state_from_json(u["state"]), tuple(u["action"].split())
def fake_gsp(model, tok, prompts, max_new_tokens=64, **kw):
    calls.append((active["name"], list(prompts), max_new_tokens))
    out = []
    for idx, p in enumerate(prompts):
        u, s, a = parse_prompt(p)
        ns, legal = B.step(s, a)
        txt = B.forward_target_delta(s, ns, legal)
        mode = idx // 240
        if active["name"] == "FK" and mode == 2 and idx % 2 == 0:
            txt = '{"add":[],"del":[],"legal":true}' if not legal else '{"add":[],"del":[],"legal":false}'
        out.append({"text": txt, "n_tokens": 1, "hit_eos": True})
    return out
fake_lm = types.ModuleType("bwmn.lm")
fake_lm.load_core = lambda name: ("BASE", "TOK")
fake_lm.greedy_shared_prefix = fake_gsp
sys.modules["bwmn.lm"] = fake_lm
bwmn.lm = fake_lm
class FakePM:
    loaded = []
    @classmethod
    def from_pretrained(cls, base, path, adapter_name="default", is_trainable=False):
        m = cls(); cls.loaded.append((adapter_name, path)); active["name"] = adapter_name; return m
    def load_adapter(self, path, adapter_name, is_trainable=False):
        FakePM.loaded.append((adapter_name, path))  # peft: does NOT activate
    def set_adapter(self, n): active["name"] = n
    def eval(self): pass
fake_peft = types.ModuleType("peft"); fake_peft.PeftModel = FakePM
sys.modules["peft"] = fake_peft
import importlib.util
spec = importlib.util.spec_from_file_location("ko", ROOT / "scripts/bw_keyorder.py")
ko = importlib.util.module_from_spec(spec); spec.loader.exec_module(ko)
spec2 = importlib.util.spec_from_file_location("sa", ROOT / "scripts/bw_surface_audit.py")
sa = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(sa)
assert "torch" not in sys.modules and "transformers" not in sys.modules
ko.ROOT = SCR  # redirect the eval.json write to scratch
ko.eval_(None)
print("loaded", FakePM.loaded)
print("calls", [(c[0], len(c[1]), c[2]) for c in calls])
assert calls[0][1] == calls[1][1]
prompts = calls[0][1]
ks = ko.keys(random.Random(ko.EVAL_SEED))
print("keys", len(ks), Counter(k[0] for k in ks), "legal", sum(B.applicable(k[2], k[3]) for k in ks))
modes = ("canon", "train_order", "heldout_rev", "heldout_other", "heldout_both")
st = defaultdict(Counter)
H1 = ("table", "on", "hand", "clear")
for idx, p in enumerate(prompts):
    mode, i = modes[idx // 240], idx % 240
    n, objs, s, act = ks[i]
    u, s2, a2 = parse_prompt(p)
    so, top = tuple(u["state"]), tuple(u)
    lists_sorted = all(u["state"][k] == sorted(u["state"][k]) for k in ("clear", "on", "table"))
    c = st[mode]
    c["so_heldout" if so in ko.HELDOUT else "so_train"] += 1
    c["so_canon"] += so == ko.STATE_KEYS
    c["top_" + top[0]] += 1
    c["unsorted"] += not lists_sorted
    names = set(B.blocks_of(s2)) | set(a2[1:])
    c["unseen_names"] += names <= set(ko.UNSEEN)
    c["orig_names"] += names <= set(B.NAMES)
    if mode != "heldout_both":
        c["same_key"] += (s2 == s and a2 == act)
    else:
        # renamed: isomorphic check via structural count and legality
        c["legal_same"] += B.applicable(s2, a2) == B.applicable(s, act)
        c["nblocks_same"] += len(B.blocks_of(s2)) == len(objs)
    if mode == "canon":
        c["eq_canonical_fn"] += p == B.forward_prompt_delta(s, act)
    if mode == "heldout_rev":
        c["eq_F08_key_order"] += p == sa.render(s, act, "key_order", random.Random(0))
        c["is_H1"] += so == H1
    if mode == "heldout_other":
        c["H2" if so == ("hand", "table", "clear", "on") else "H3" if so == ("on", "clear", "table", "hand") else "other"] += 1
    if mode == "heldout_both":
        c["is_H1"] += so == H1
    if mode == "train_order":
        c["in_train_orders"] += so in ko.TRAIN_ORDERS
    # correct answer parses with this state
    ns, legal = B.step(s2, a2)
    c["gold_parses"] += B.parse_forward_delta(B.forward_target_delta(s2, ns, legal), s2) == (ns, legal)
for m in modes:
    print(m, dict(st[m]))
print(open(SCR / "runs/BW_KEYORDER/eval.json").read())
# overlap with training transitions and F-08 keys
train = set()
for l in open(ROOT / "runs/BW_data/forward_delta_train.jsonl", encoding="utf-8"):
    r = json.loads(l); train.add(r["prompt"])
ov = sum(B.forward_prompt_delta(k[2], k[3]) in train for k in ks)
rng = random.Random(20261009); f08 = []
for n in (5, 10):
    for _ in range(60):
        objs = sorted(rng.sample(B.NAMES, n)); s = B.towers_to_state(B.random_towers(objs, rng))
        for _ in range(rng.randrange(0, 5)):
            s, _ = B.step(s, rng.choice([x for x in B.ground_actions(objs) if B.applicable(s, x)]))
        ga = B.ground_actions(objs); leg = [x for x in ga if B.applicable(s, x)]; ill = [x for x in ga if not B.applicable(s, x)]
        act = rng.choice(leg) if (rng.random() < 0.5 or not ill) else rng.choice(ill); f08.append((s, act))
print("overlap train", ov, "overlap F08", sum((k[2], k[3]) in set(f08) for k in ks), "dup keys", len(ks) - len({(k[2], k[3]) for k in ks}))
json.dump(prompts, open(SCR / "prompts.json", "w"))
