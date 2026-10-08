"""Audit 4: is Supa's 'unparseable' P/R generation a content failure or a format/truncation artifact?"""
import json, re, sys
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import bank as bankmod, data, env, generation, runtime
tok = runtime.tokenizer()
items = {it["item_id"]: it for it in bankmod.items(data.load_bank("bank_pilot"))}
rows = [json.loads(l) for l in open(ROOT / "runs/bank_pilot/supa/gen_e2e.jsonl", encoding="utf-8")]

def lenient(fam, t):
    s = env.parse_state(fam, t)
    if s is not None:
        return s, "strict"
    try:
        v = json.loads(t.strip().rstrip("."))
    except Exception:
        v = None
    if isinstance(v, dict) and "after" in v:
        s = env.parse_state(fam, json.dumps(v["after"]))
        if s is not None:
            return s, "after_key"
    # truncated JSON: grab a complete "after": {...} or "after": int
    m = re.search(r'"after"\s*:\s*(\{[^{}]*\}|-?\d+)(?=\s*[,}])', t)
    if m:
        s = env.parse_state(fam, m.group(1))
        if s is not None:
            return s, "after_regex"
    return None, None

agg = defaultdict(lambda: defaultdict(int))
for r in rows:
    it = items[r["item_id"]]
    fam = it["family"]; kind = r["item_id"].split("-")[-1]
    key = (fam[:6], kind)
    a = agg[key]
    a["n"] += 1
    ntok = len(tok(r["text"], add_special_tokens=False)["input_ids"])
    a["hit_token_cap"] += int(ntok >= generation.max_tokens(fam))
    s, how = lenient(fam, r["text"])
    a["strict_exact"] += int(r["exact"])
    if s is not None:
        a["lenient_parsed"] += 1
        a["lenient_exact"] += int(env.dumps(s) == it["gold"])
        s0 = json.loads(it["gold_prefix_states"][0])
        a["lenient_eq_s0"] += int(env.dumps(s) == env.dumps(s0))
        a["gold_eq_s0"] += int(it["gold"] == env.dumps(s0))
        a["lenient_exact_and_changed"] += int(env.dumps(s) == it["gold"] and it["gold"] != env.dumps(s0))
for key in sorted(agg):
    a = agg[key]; n = a["n"]
    print(key, f"n={n} hit_cap={a['hit_token_cap']/n:.2f} strict_exact={a['strict_exact']/n:.3f} lenient_parsed={a['lenient_parsed']/n:.2f} "
          f"lenient_exact={a['lenient_exact']/n:.3f} (copy-s0 answers {a['lenient_eq_s0']/max(a['lenient_parsed'],1):.2f}; gold==s0 {a['gold_eq_s0']/max(a['lenient_parsed'],1):.2f}; exact&changed {a['lenient_exact_and_changed']/n:.3f})")
