import os, sys, json, collections, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from rcwmt import env
from analyze import load_gen
for arm in ("supa", "sparse"):
    g = load_gen(ROOT / "runs" / "bank_pilot" / arm)
    print("=====", arm)
    # chain texts: first text of each chain for R and L, parse lenient
    for fam, pre in (("register_machine", "T1R"), ("ledger_replay", "T1L")):
        recs = [r for c, r in g["chain_one"].items() if c.startswith(pre)]
        fails = collections.Counter()
        ex = []
        n = 0; nl = 0
        for r in recs:
            for t in r["texts"]:
                n += 1
                if env.parse_state(fam, t, lenient=True) is None:
                    nl += 1
                    shape = re.sub(r"-?\d+", "N", t)[:90]
                    fails[shape] += 1
        print(fam, "chain texts", n, "lenient-unparseable", nl, "chains", len(recs),
              "chains reaching k4", sum(r["states"][4] is not None for r in recs))
        for s, c in fails.most_common(5):
            print("   ", c, repr(s))
        # e2e
        recs = [r for i, r in g["e2e"].items() if i.startswith(pre)]
        fails = collections.Counter(); nl = 0
        for r in recs:
            if env.parse_state(fam, r["text"], lenient=True) is None:
                nl += 1
                fails[re.sub(r"-?\d+", "N", r["text"])[:90]] += 1
        print(fam, "e2e", len(recs), "lenient-unparseable", nl)
        for s, c in fails.most_common(5):
            print("   ", c, repr(s))
