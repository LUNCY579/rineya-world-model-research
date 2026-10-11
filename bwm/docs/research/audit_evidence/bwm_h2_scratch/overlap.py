# Renaming-invariant overlap check: distill-240 problems vs the 110 hard instances.
# WL colour refinement on the (init-on, goal-on, init-ontable, goal-ontable) structure; equal hash is necessary for
# isomorphism, so 0 hash collisions => no distill problem is a renaming of a hard instance.
import json, re, glob, hashlib, collections
B = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/"
def parse(p):
    t = open(p).read().lower()
    objs = re.search(r"\(:objects([^)]*)\)", t).group(1).split()
    init = re.search(r"\(:init(.*?)\)\s*\(:goal", t, re.S).group(1)
    goal = re.search(r"\(:goal(.*)", t, re.S).group(1)
    f = lambda s: [tuple(m.split()) for m in re.findall(r"\(([a-z\-]+(?: [a-z0-9]+)*)\)", s) if m.split()[0] in ("on","ontable","clear","handempty","holding")]
    return objs, f(init), f(goal)
def wl(objs, init, goal, it=6):
    col = {o: "x" for o in objs}
    feats = collections.defaultdict(list)
    for tag, F in (("i", init), ("g", goal)):
        for fct in F:
            if fct[0] == "on": feats[fct[1]].append((tag+"above_of", fct[2])); feats[fct[2]].append((tag+"below_of", fct[1]))
            elif len(fct) == 2: feats[fct[1]].append((tag+fct[0], None))
    for _ in range(it):
        new = {}
        for o in objs:
            sig = sorted((k, col[v] if v else "") for k, v in feats[o])
            new[o] = hashlib.sha1((col[o] + repr(sig)).encode()).hexdigest()[:16]
        col = new
    return (len(objs), tuple(sorted(col.values())))
hard = {}
for p in glob.glob(B + "bench/planbench/blocksworld_hard/generated/*.pddl"):
    o, i, g = parse(p); hard[wl(o, i, g)] = p
dist = [json.loads(l) for l in open(B + "runs/BW_DISTILL_r1/solutions.jsonl", encoding="utf-8")]
hits = [d["id"] for d in dist if wl(d["objs"], [tuple(x) for x in d["init"]], [tuple(x) for x in d["goal"]]) in hard]
print("hard distinct WL classes:", len(hard), "| distill problems:", len(dist), "| distill hash collisions with hard:", len(hits), hits[:5])
nb = collections.Counter(len(parse(p)[0]) for p in glob.glob(B + "bench/planbench/blocksworld_hard/generated/*.pddl"))
print("hard block counts:", sorted(nb.items()))
gl = collections.Counter(sum(1 for f in parse(p)[2] if f[0]=="on") for p in glob.glob(B + "bench/planbench/blocksworld_hard/generated/*.pddl"))
print("hard goal on-pairs:", sorted(gl.items()))
