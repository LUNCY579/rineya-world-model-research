import json, sys, importlib.util, hashlib, time
from collections import Counter, deque
from pathlib import Path
t0 = time.time()
ROOT = Path("C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from bwmn import blocks as B
p = ROOT / "scripts/freplay.py"
assert hashlib.sha256(p.read_bytes()).hexdigest() == "866ca5b696b465d0de77d0882b38323df50f1bb3bcc216ff68aaadbb0beb0a3b"
spec = importlib.util.spec_from_file_location("_aud_freplay", p)
FR = importlib.util.module_from_spec(spec); spec.loader.exec_module(FR)
inst = json.load(open(ROOT / "runs/BW_val_large/instances_seeds_fresh.json"))
I = {f"seeds_fresh/{i}": dict(objs=r["objs"], init=frozenset(map(tuple, r["init"])), goal=frozenset(map(tuple, r["goal"])))
     for i, r in enumerate(inst)}


def rows(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8").read().splitlines() if l.strip()]


rep = json.load(open(ROOT / "runs/BW_SEEDS/ADDENDUM_A_REPORT.json"))["A1_f_audit"]["sets"]


# ---------- independent reimplementation (own parser; only B.step from blocks.py)
def myparse(text):
    try:
        d = json.loads(text.strip())
    except Exception:
        return None
    if not isinstance(d, dict) or set(d) != {"add", "del", "legal"} or type(d["legal"]) is not bool:
        return None
    try:
        add = frozenset(tuple(x.split()) for x in d["add"]); de = frozenset(tuple(x.split()) for x in d["del"])
    except Exception:
        return None
    return add, de, d["legal"]


def myclass(s, a, text):
    ns, legal = B.step(s, a)
    q = myparse(text)
    sup = q is not None and q[1] <= s
    pred = frozenset((s - q[1]) | q[0]) if sup else None
    acc = sup and q[2]
    if legal:
        if acc:
            c = "ok" if pred == ns else "wrong_state"
        else:
            c = "false_illegal"
    else:
        if acc:
            c = "false_legal_accepted"
        elif q is None:
            c = "malformed"
        elif q[2] is False:
            c = "ok"
        else:
            c = "false_legal_rejected_by_support_check"
    fk = q is not None; fc = fk and q[2] == legal
    de = fk and q[2] == legal and q[0] == ns - s and q[1] == s - ns
    return dict(cls=c, legal=legal, acc=acc, next=pred if acc else None, fk=fk, fc=fc, de=de)


HARM = {"false_illegal", "false_legal_accepted", "wrong_state"}


def myread(path):
    m = {}
    for line in open(ROOT / path, encoding="utf-8").read().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        m[(frozenset(tuple(f) for f in r["s"]), tuple(r["a"]))] = r["text"]
    return m


for s in ["R0", "R1", "R2"]:
    files = [f"runs/BW_SEEDS/{s}_beam/shard_{i}of4/fcache.jsonl" for i in range(4)]
    # --- path 1: freplay's own functions
    caches = {f: FR.read_cache(str(ROOT / f)) for f in files}
    memo = FR.Memo()
    units, conf, occ = FR.build_units(files, caches, memo)
    st = FR.strata(units)
    shard_of = {}
    for i in range(4):
        for r in rows(f"runs/BW_SEEDS/{s}_beam/shard_{i}of4/plans.jsonl"):
            shard_of[r["id"]] = files[i]
    idx = {}
    for f in files:
        bs = {}
        for (ss, a) in caches[f][0]:
            bs.setdefault(ss, []).append(a)
        idx[f] = {k: sorted(v) for k, v in bs.items()}
    merged = rows(f"runs/BW_SEEDS/{s}_beam/plans.jsonl")
    frows = []
    for r in merged:
        it = I[r["id"]]; f = shard_of[r["id"]]; cache = caches[f][0]
        states, keys = FR.forest(it["init"], idx[f], cache, memo)
        cc = Counter(x["cls"] for _, x in keys)
        harm = [kk for kk, x in keys if x["cls"] in FR.HARMFUL]
        cov = FR.coverage({"init": it["init"], "goal": it["goal"], "plan": [tuple(x.split()) for x in r["plan"]]}, cache, memo)
        frows.append(dict(id=r["id"], success=bool(r["success"]), verified=bool(r["verified"]), flagged=bool(harm),
                          support_exposed=cc["false_legal_rejected_by_support_check"] > 0, cov=cov, nharm=len(harm)))
    R1 = FR.reading_r1(frows); R2 = FR.reading_r2(f"i7_{s}_beam", st["total"])
    t = st["total"]
    # --- path 2: independent
    U = {}
    for f in files:
        for k, txt in myread(f).items():
            U.setdefault(k, set()).add(txt)
    conflicts = sum(len(v) > 1 for v in U.values())
    sev = ["false_legal_accepted", "wrong_state", "false_illegal", "false_legal_rejected_by_support_check", "malformed", "ok"]
    C = Counter(); legal = 0; ill = 0; fk = 0; fc = 0; de = 0
    for k, txts in U.items():
        rs = [myclass(k[0], k[1], t_) for t_ in txts]; r_ = min(rs, key=lambda x: sev.index(x["cls"]))
        C[r_["cls"]] += 1; legal += r_["legal"]; ill += not r_["legal"]; fk += r_["fk"]; fc += r_["fc"]; de += r_["de"]
    myflag = {}; myexp = {}
    caches2 = {f: myread(f) for f in files}
    for r in merged:
        it = I[r["id"]]; cache = caches2[shard_of[r["id"]]]
        bystate = {}
        for (ss, a) in cache:
            bystate.setdefault(ss, []).append(a)
        seen = {it["init"]}; dq = deque([it["init"]]); h = 0; sx = 0
        while dq:
            x = dq.popleft()
            for a in bystate.get(x, ()):
                c = myclass(x, a, cache[(x, a)])
                h += c["cls"] in HARM; sx += c["cls"] == "false_legal_rejected_by_support_check"
                if c["acc"] and c["next"] not in seen:
                    seen.add(c["next"]); dq.append(c["next"])
        myflag[r["id"]] = h > 0; myexp[r["id"]] = sx > 0
    L = sum(r["success"] for r in merged if not myflag[r["id"]]); nfl = sum(myflag.values())
    myR2 = C["false_legal_rejected_by_support_check"] / ill
    print(f"== {s}")
    print(" freplay-path: units", len(units), "conflicts", len(conf), "legal/illegal", t["legal"], t["illegal"], "classes", t["classes"])
    print("   R2", R2["R2"], "trig", R2["wording_change_triggered"], "| R1 L,U", R1["exactF_success_lower_L"],
          R1["exactF_success_upper_U"], "flagged", R1["flagged"], "flagged_solved", R1["flagged_solved"], "supp-exposed",
          R1["support_exposed_instances"])
    print("   harmful", t["rates"]["harmful"], "false_illegal", t["rates"]["false_illegal"], "flag_acc",
          t["secondary"]["flag_accuracy"], "delta_exact", t["secondary"]["delta_exact"])
    print("   subtypes", t["subtypes"])
    print("   coverage fails", [x["id"] for x in frows if not x["cov"]["covered"]], "verified not goal",
          [x["id"] for x in frows if x["verified"] and x["cov"]["goal_at_end"] is not True])
    print(" independent: units", len(U), "conflicts", conflicts, "legal/illegal", legal, ill, "classes", dict(C),
          "R2 %d/%d = %.6f" % (C["false_legal_rejected_by_support_check"], ill, myR2), "flag_acc %d/%d" % (fc, fk),
          "delta_exact %d/%d" % (de, len(U)), "harmful %d/%d" % (sum(C[c] for c in HARM), len(U)))
    print("   L,U", L, L + nfl, "flagged", nfl, "supp-exposed", sum(myexp.values()), "flag sets equal",
          sorted(k for k, v in myflag.items() if v) == sorted(R1["flagged_ids"]))
    rr = rep[f"i7_{s}_beam"]
    print(" report match: strata", rr["strata"] == st, "R1", rr["FREPLAY_R1"] == R1, "R2", rr["FREPLAY_R2"]["R2"] == R2["R2"],
          "trig", rr["FREPLAY_R2"]["wording_change_triggered"] == R2["wording_change_triggered"])
print("wall", round(time.time() - t0, 1))
try:
    import psutil
    print("rss MB", psutil.Process().memory_info().rss / 1e6)
except Exception as e:
    print(e)
