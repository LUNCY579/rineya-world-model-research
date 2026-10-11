import sys, random
from collections import Counter
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from tokenizers import Tokenizer
from bwmn import logistics as L
TK = Tokenizer.from_file("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
n = lambda s: len(TK.encode(s, add_special_tokens=False).ids)
rng = random.Random(11)
lens = Counter(); mxrow = 0; over = 0; k = 0
while k < 3000:
    g = L.generate(rng, 3, n_cities=3, city_size=3, rename=0.5)
    if g is None: continue
    k += 1
    objs, init, goal = g
    p = L.astar_plan(init, goal)
    lens[len(p)] += 1
    row = n(L.proposer_prompt(init, goal)) + n(L.target_actions(p)) + 1
    mxrow = max(mxrow, row); over += row > 640
print("3 pk, 3x3 maps, 3000 inst: max len", max(lens), "len dist tail", sorted(lens.items())[-6:], "max full row", mxrow, "instances with full row > 640:", over)
