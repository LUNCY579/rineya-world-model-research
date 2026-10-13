"""Read-only: show that proposer_states_* files carry exactly the same prompts (hence the same (state, goal) problems)
as the corresponding proposer_actions_* files, so the actions-file audit applies to them unchanged; and compare the
superseded forward file with the live one."""
import hashlib
import json
from pathlib import Path

ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


for d in ("runs/BW_data", "runs/BW_data_v2", "runs/_superseded/BW_data_v1_suffix_leak"):
    a, s = ROOT / d / "proposer_actions_train.jsonl", ROOT / d / "proposer_states_train.jsonl"
    n = same = 0
    with open(a, encoding="utf-8") as fa, open(s, encoding="utf-8") as fs:
        for la, ls in zip(fa, fs):
            n += 1
            same += json.loads(la)["prompt"] == json.loads(ls)["prompt"]
        extra = sum(1 for _ in fa) + sum(1 for _ in fs)
    print(d, "rows", n, "identical prompts", same, "unpaired extra lines", extra)
print("forward_train sha", sha(ROOT / "runs/BW_data/forward_train.jsonl")[:16],
      "superseded forward_train sha", sha(ROOT / "runs/_superseded/BW_data_v1_suffix_leak/forward_train.jsonl")[:16])
# superseded v1 vs live v1 proposer rows: which prompts were removed
live = set()
for l in open(ROOT / "runs/BW_data/proposer_actions_train.jsonl", encoding="utf-8"):
    live.add(hashlib.sha1(l.encode()).digest())
only_sup = sum(1 for l in open(ROOT / "runs/_superseded/BW_data_v1_suffix_leak/proposer_actions_train.jsonl", encoding="utf-8")
               if hashlib.sha1(l.encode()).digest() not in live)
print("superseded v1 lines not in live v1:", only_sup)
