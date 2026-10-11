import json, sys
from pathlib import Path
from fractions import Fraction
from scipy.stats import fisher_exact
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import blocks as B
import blocks_val_large as VL
def gmax_list(insts):
    out=[]
    for o,i,g in insts:
        gt=[h for h in VL.goal_towers(g,o) if h>1]
        out.append(max(gt) if gt else 1)
    return out
hard=[B.parse_problem(p) for p in (B.BENCH/"blocksworld_hard"/"generated").glob("*.pddl")]
mirror=[(r["objs"],frozenset(map(tuple,r["init"])),frozenset(map(tuple,r["goal"]))) for r in json.loads((ROOT/"runs/BW_val_large/instances_mirror_hard.json").read_text())]
for n,s in (("hard",hard),("mirror",mirror)):
    g=gmax_list(s); print(n, sum(g), len(g), Fraction(sum(g),len(g)), sum(g)/len(g), round(sum(g)/len(g),1))
# published refs
for name, rate in (("R1",0.536),("o1-preview",0.23636363636363636),("o1-mini",0.1)):
    k=round(rate*110)
    print(name, k, k/110, fisher_exact([[73,37],[k,110-k]])[1])
