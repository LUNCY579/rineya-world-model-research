import sys, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from bwmn import lm, rw2
from peft import PeftModel
base, tok = lm.load_core("titan-sparse")
m = PeftModel.from_pretrained(base, str(ROOT / "runs/F_titan_main/adapter"), is_trainable=False).eval()
keys = [(x, a, 64) for x in range(-30, 30) for a in rw2.ACTIONS][:200]
P = [rw2.forward_prompt(*k) for k in keys]
t0 = time.time(); hf = lm.generate(m, tok, P, max_new_tokens=24); t1 = time.time()
cr = lm.sample_crn(m, tok, P, [None] * len(P), max_new_tokens=24); t2 = time.time()
same = sum(h[0]["text"] == c["text"] for h, c in zip(hf, cr))
print(f"greedy parity {same}/{len(P)}  hf {t1-t0:.1f}s crn {t2-t1:.1f}s")
acc = sum(rw2.parse_forward(c["text"]) == rw2.exact_step(*k) for k, c in zip(keys, cr)); print("F acc on these", acc, "/", len(keys))
# CRN: sampling depends only on (prompt, seed)
Q = [rw2.proposer_prompt(x, 13, 64, 7) for x in range(-20, 20)]
with m.disable_adapter():
    s1 = lm.sample_crn(m, tok, Q, [1000 + i for i in range(len(Q))], max_new_tokens=30)
    perm = list(range(len(Q))); random.Random(0).shuffle(perm)
    s2 = lm.sample_crn(m, tok, [Q[i] for i in perm[:25]], [1000 + i for i in perm[:25]], max_new_tokens=30, max_batch_tokens=3000)
same = sum(s1[i]["text"] == s["text"] for i, s in zip(perm[:25], s2))
print(f"CRN invariance {same}/25; distinct samples {len(set(x['text'] for x in s1))}/{len(s1)}")
print(repr(s1[0]["text"][:120]))
