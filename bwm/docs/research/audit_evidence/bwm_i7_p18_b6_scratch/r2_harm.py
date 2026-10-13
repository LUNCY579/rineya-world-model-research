import json, sys, random, math
from collections import Counter, deque
from pathlib import Path
ROOT = Path("C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from bwmn import blocks as B

inst = json.load(open(ROOT / "runs/BW_val_large/instances_seeds_fresh.json"))
I = {f"seeds_fresh/{i}": dict(objs=r["objs"], init=frozenset(map(tuple, r["init"])), goal=frozenset(map(tuple, r["goal"])),
                              n=len(r["objs"])) for i, r in enumerate(inst)}
IDS = [f"seeds_fresh/{i}" for i in range(40)]


def rows(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / p, encoding="utf-8").read().splitlines() if l.strip()}


def myread(path):
    m = {}
    for line in open(ROOT / path, encoding="utf-8").read().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        m[(frozenset(tuple(f) for f in r["s"]), tuple(r["a"]))] = r["text"]
    return m


def own_legal(s, a):
    """Hand-written STRIPS preconditions (independent of B.applicable)."""
    op = a[0]
    if op == "pick-up":
        return {("clear", a[1]), ("ontable", a[1]), ("handempty",)} <= s
    if op == "put-down":
        return ("holding", a[1]) in s
    if op == "stack":
        return a[1] != a[2] and {("holding", a[1]), ("clear", a[2])} <= s
    if op == "unstack":
        return {("on", a[1], a[2]), ("clear", a[1]), ("handempty",)} <= s
    return False


def fparse(text, s):
    try:
        d = json.loads(text.strip())
        if set(d) != {"add", "del", "legal"} or type(d["legal"]) is not bool:
            return ("malformed", None)
        dele = {tuple(x.split()) for x in d["del"]}
        if not d["legal"]:
            return ("said_illegal", None)
        if not dele <= s:
            return ("said_legal_unsupported_delete", None)
        return ("accepted", frozenset((s - dele) | {tuple(x.split()) for x in d["add"]}))
    except Exception:
        return ("malformed", None)


rep = json.load(open(ROOT / "runs/BW_SEEDS/ADDENDUM_A_REPORT.json"))["A1_f_audit"]["sets"]
beam = {s: rows(f"runs/BW_SEEDS/{s}_beam/plans.jsonl") for s in ("R0", "R1", "R2")}
flag = {s: {r["id"]: r["flagged"] for r in rep[f"i7_{s}_beam"]["instances"]} for s in ("R0", "R1", "R2")}

# ---------------- (a) every R2 harmful unit, checked with the exact simulator
files = [f"runs/BW_SEEDS/R2_beam/shard_{i}of4/fcache.jsonl" for i in range(4)]
C = {f: myread(f) for f in files}
allk = {}
for f in files:
    for k, t in C[f].items():
        allk[k] = (t, f)
harm = []
for k, (t, f) in allk.items():
    s, a = k
    ns, legal = B.step(s, a)
    st, pred = fparse(t, s)
    if legal and st != "accepted":
        harm.append((k, t, f, st))
    elif legal and pred != ns:
        harm.append((k, t, f, "wrong_state"))
    elif not legal and st == "accepted":
        harm.append((k, t, f, "false_legal_accepted"))
print("R2 harmful keys found by own scan:", len(harm), Counter(x[3] for x in harm))
agree = sum(own_legal(k[0], k[1]) == B.applicable(k[0], k[1]) for k in allk)
print("own preconditions agree with B.applicable on all R2 keys:", agree, "/", len(allk))
rep_ids = {h["key_id"] for h in rep["i7_R2_beam"]["harmful_keys"]}
import hashlib
def key_id(k):
    s, a = k
    j = {"s": sorted(" ".join(f) for f in s), "a": " ".join(a)}
    return hashlib.sha256(json.dumps(j, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
mine = {key_id(k) for k, *_ in harm}
print("harmful key_id set equals report's:", mine == rep_ids, len(rep_ids))
by_n = Counter(len(B.blocks_of(k[0])) for k, *_ in harm); by_a = Counter(k[1][0] for k, *_ in harm)
leg_n = Counter(len(B.blocks_of(k[0])) for k in allk if B.applicable(*k))
print("R2 harmful by blocks (harm/legal keys):", {n: f"{by_n[n]}/{leg_n[n]}" for n in sorted(leg_n)})
print("R2 harmful by action:", dict(by_a))
leg_a = Counter(k[1][0] for k in allk if B.applicable(*k))
print("R2 legal keys by action:", dict(leg_a))
# sample of 25 (seeded) for the report
random.seed(7)
samp = random.sample(harm, 25)
ok_s = 0
for k, t, f, st in samp:
    s, a = k
    lg = B.applicable(s, a) and own_legal(s, a)
    ok_s += lg and st != "accepted"
print("sample 25: legal under B.step and own preconditions, and F planner-effective illegal:", ok_s, "/ 25")
for k, t, f, st in samp[:8]:
    s, a = k
    pre = {"pick-up": [("clear", a[1]), ("ontable", a[1]), ("handempty",)], "put-down": [("holding", a[1])],
           "stack": [("holding", a[1]), ("clear", a[2])] if len(a) > 2 else [],
           "unstack": [("on", a[1], a[2]), ("clear", a[1]), ("handempty",)] if len(a) > 2 else []}[a[0]]
    print("  ", key_id(k), len(B.blocks_of(s)), "blocks |", " ".join(a), "| preconds in s:", all(p in s for p in pre),
          "|", st, "|", t[:110], "|", f.split("/")[-2])
# all 159 legal?
print("all harmful: legal under both checks:", sum(B.applicable(*k) and own_legal(*k) for k, *_ in harm), "/", len(harm))
print("all harmful: F said legal:false:", sum(st == "said_illegal" for *_, st in harm),
      "; malformed:", sum(st == "malformed" for *_, st in harm), "; legal:true w/ unsupported del:",
      sum(st == "said_legal_unsupported_delete" for *_, st in harm))
# do the fi_flag texts give the true delta otherwise?  (add/del equal to truth but flag false?)
same_delta = 0; empty = 0
for k, t, f, st in harm:
    if st != "said_illegal":
        continue
    d = json.loads(t)
    s, a = k
    ns, _ = B.step(s, a)
    if {tuple(x.split()) for x in d["add"]} == ns - s and {tuple(x.split()) for x in d["del"]} == s - ns:
        same_delta += 1
    if not d["add"] and not d["del"]:
        empty += 1
print("said_illegal keys whose add/del equal the true delta:", same_delta, "; with empty add/del:", empty)
print("malformed texts:", [t[:120] for k, t, f, st in harm if st == "malformed"])

# ---------------- (b) instance level
print()
print("id n | R0 R1 R2 beam (1=solved) | R2 flagged (#harm keys in forest) | R2 fail")
nh = {r["id"]: len(r["harmful_keys"]) for r in rep["i7_R2_beam"]["instances"]}
for i in IDS:
    print(i.split("/")[1], I[i]["n"], "|", int(beam["R0"][i]["val"]), int(beam["R1"][i]["val"]), int(beam["R2"][i]["val"]),
          "|", int(flag["R2"][i]), nh[i], "|", beam["R2"][i]["fail"], "| R1flag", int(flag["R1"][i]))
tab = Counter((flag["R2"][i], beam["R2"][i]["val"]) for i in IDS)
print("R2 crosstab (flagged, solved):", dict(tab))


def fisher_two_sided(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def p(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9))


a = tab[(True, False)]; b = tab[(True, True)]; c = tab[(False, False)]; d = tab[(False, True)]
print("R2 failures in flagged: %d of %d failures; flagged fail rate %d/%d vs unflagged %d/%d; Fisher two-sided p=%.4g"
      % (a, a + c, a, a + b, c, c + d, fisher_two_sided(a, b, c, d)))
for lo_n in (11, 12):
    sub = [i for i in IDS if I[i]["n"] >= lo_n]
    print(f"instances with >= {lo_n} blocks: {len(sub)}; R2 flagged {sum(flag['R2'][i] for i in sub)}; failures R0 "
          f"{sum(not beam['R0'][i]['val'] for i in sub)}, R1 {sum(not beam['R1'][i]['val'] for i in sub)}, R2 "
          f"{sum(not beam['R2'][i]['val'] for i in sub)}")
for s in ("R0", "R1", "R2"):
    sb = Counter(); tb = Counter()
    for i in IDS:
        tb[I[i]["n"]] += 1; sb[I[i]["n"]] += beam[s][i]["val"]
    print(s, "solved by blocks:", {n: f"{sb[n]}/{tb[n]}" for n in sorted(tb)})
# within unflagged-in-R2 instances vs flagged: R0 / R1 success (difficulty control)
for s in ("R0", "R1"):
    fl = [i for i in IDS if flag["R2"][i]]; un = [i for i in IDS if not flag["R2"][i]]
    print(f"{s} success on R2-flagged {sum(beam[s][i]['val'] for i in fl)}/{len(fl)}, on R2-unflagged "
          f"{sum(beam[s][i]['val'] for i in un)}/{len(un)}")
r0_only = [i for i in IDS if beam["R0"][i]["val"] and not beam["R2"][i]["val"]]
r2_only = [i for i in IDS if beam["R2"][i]["val"] and not beam["R0"][i]["val"]]
r1_not_r2 = [i for i in IDS if beam["R1"][i]["val"] and not beam["R2"][i]["val"]]
both = [i for i in IDS if beam["R1"][i]["val"] and beam["R0"][i]["val"] and not beam["R2"][i]["val"]]
print("R0-solved & R2-failed:", r0_only, "flagged in R2:", sum(flag["R2"][i] for i in r0_only), "R2 fail:",
      [beam["R2"][i]["fail"] for i in r0_only])
print("R2-solved & R0-failed:", r2_only, "flagged in R2:", sum(flag["R2"][i] for i in r2_only))
print("R1-solved & R2-failed:", r1_not_r2, "flagged in R2:", sum(flag["R2"][i] for i in r1_not_r2))
print("R0- and R1-solved & R2-failed:", both, "flagged in R2:", sum(flag["R2"][i] for i in both))

# ---------------- (c) do R2's false-illegal keys block moves that R0/R1 actually used?
shard_of = {}
for i_sh in range(4):
    for l in open(ROOT / f"runs/BW_SEEDS/R2_beam/shard_{i_sh}of4/plans.jsonl", encoding="utf-8"):
        if l.strip():
            shard_of[json.loads(l)["id"]] = files[i_sh]
harm_by_inst = {r["id"]: {(frozenset(tuple(x.split()) for x in h["key"]["s"]), tuple(h["key"]["a"].split()))
                          for h in r["harmful_keys"]} for r in rep["i7_R2_beam"]["instances"]}
for i in IDS:
    if beam["R2"][i]["val"]:
        continue
    it = I[i]
    hits = {}
    for s_ in ("R0", "R1"):
        if not beam[s_][i]["val"]:
            continue
        st = it["init"]; keys = []
        for a in beam[s_][i]["plan"]:
            a = tuple(a.split()); keys.append((st, a)); st, _ = B.step(st, a)
        hits[s_] = (len(set(keys) & harm_by_inst[i]), len(set(keys) & set(C[shard_of[i]])), len(keys))
    # last state of R2's submitted prefix: harmful keys there?
    st = it["init"]
    traj = [st]
    for a in beam["R2"][i]["plan"]:
        st, _ = B.step(st, tuple(a.split())); traj.append(st)
    cache = C[shard_of[i]]
    on_path = sum(1 for (s0, a0) in harm_by_inst[i] if s0 in set(traj))
    at_end = [(" ".join(a0)) for (s0, a0) in harm_by_inst[i] if s0 == traj[-1]]
    legal_end = [a0 for (s0, a0) in cache if s0 == traj[-1] and B.applicable(s0, a0)]
    print(f"R2 fail {i} n={it['n']} fail={beam['R2'][i]['fail']} unmet_end={len(it['goal'] - traj[-1])} "
          f"flagged={flag['R2'][i]} harm_in_forest={len(harm_by_inst[i])} harm_at_states_on_submitted_path={on_path} "
          f"harm_at_end_state={at_end} legal_keys_queried_at_end={len(legal_end)} | R0/R1 plan steps that are R2-harmful "
          f"keys / in R2 cache / plan len: {hits}")
