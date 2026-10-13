import json, math, sys, hashlib
from pathlib import Path
from fractions import Fraction
from collections import Counter, defaultdict
from scipy.stats import beta, binomtest, binom

ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
OUT = ROOT / "runs/BW_SEEDS"
N = 40
fresh = json.loads((ROOT / "runs/BW_val_large/instances_seeds_fresh.json").read_text())
print("fresh n", len(fresh), "sha", hashlib.sha256((ROOT / "runs/BW_val_large/instances_seeds_fresh.json").read_bytes()).hexdigest())
ids = [f"seeds_fresh/{i}" for i in range(len(fresh))]
nb = {f"seeds_fresh/{i}": len(r["objs"]) for i, r in enumerate(fresh)}
assert all(len(r["objs"]) == r["n"] for r in fresh), "objs vs n mismatch"
print("block dist", sorted(Counter(nb.values()).items()))


# ---- independent simulator (4-op Blocksworld, VAL semantics: every action applicable + goal at end)
def sim(init, goal, plan):
    on = {}; table = set(); clear = set(); hold = None; hand = False
    for f in init:
        if f[0] == "on": on[f[1]] = f[2]
        elif f[0] == "ontable": table.add(f[1])
        elif f[0] == "clear": clear.add(f[1])
        elif f[0] == "handempty": hand = True
        elif f[0] == "holding": hold = f[1]
    for k, a in enumerate(plan):
        t = a.split()
        op = t[0]
        if op == "pick-up" and len(t) == 2:
            x = t[1]
            if not (x in clear and x in table and hand): return False, False, k
            clear.discard(x); table.discard(x); hand = False; hold = x
        elif op == "put-down" and len(t) == 2:
            x = t[1]
            if hold != x: return False, False, k
            hold = None; hand = True; clear.add(x); table.add(x)
        elif op == "stack" and len(t) == 3:
            x, y = t[1], t[2]
            if not (x != y and y in clear and hold == x): return False, False, k
            clear.discard(y); hold = None; hand = True; clear.add(x); on[x] = y
        elif op == "unstack" and len(t) == 3:
            x, y = t[1], t[2]
            if not (on.get(x) == y and x in clear and hand): return False, False, k
            del on[x]; clear.discard(x); hand = False; hold = x; clear.add(y)
        else:
            return False, False, k
    st = set()
    for x, y in on.items(): st.add(("on", x, y))
    for x in table: st.add(("ontable", x))
    for x in clear: st.add(("clear", x))
    if hand: st.add(("handempty",))
    if hold: st.add(("holding", hold))
    ok = all(tuple(g) in st for g in goal)
    return ok, True, None


def read_rows(p):
    raw = p.read_bytes()
    issues = []
    if not raw.endswith(b"\n"): issues.append("no trailing newline")
    lines = raw.split(b"\n")
    if lines[-1] == b"": lines = lines[:-1]
    rows = []
    for k, l in enumerate(lines):
        if l.endswith(b"\r"): issues.append(f"CR on line {k}")
        try: rows.append(json.loads(l.decode("utf-8")))
        except Exception as e: issues.append(f"bad json line {k}: {e}")
    return rows, issues


FIELDS = Counter()
data = {}
problems = []
for R in ("R0", "R1", "R2"):
    for cfg in ("beam", "direct"):
        d = OUT / f"{R}_{cfg}"
        rows, iss = read_rows(d / "plans.jsonl")
        if iss: problems.append((R, cfg, iss))
        for r in rows: FIELDS[(R, cfg, tuple(sorted(r)))] += 1
        rid = [r["id"] for r in rows]
        dup = [k for k, v in Counter(rid).items() if v > 1]
        if dup: problems.append((R, cfg, "dups", dup))
        if set(rid) != set(ids) or len(rid) != N: problems.append((R, cfg, "idset", set(rid) ^ set(ids), len(rid)))
        simfail = Counter()
        for r in rows:
            if r["n_blocks"] != nb[r["id"]]: problems.append((R, cfg, "n_blocks", r["id"]))
            if "val" not in r: problems.append((R, cfg, "noval", r["id"]))
            if bool(r["val"]) != bool(r["success"]): problems.append((R, cfg, "val!=success", r["id"]))
            if r["len"] != len(r["plan"]): problems.append((R, cfg, "len", r["id"]))
            i = int(r["id"].split("/")[1])
            ok, valid, _ = sim(fresh[i]["init"], fresh[i]["goal"], r["plan"])
            if ok != bool(r["success"]) or valid != bool(r["valid"]) or ok != bool(r["val"]):
                problems.append((R, cfg, "sim mismatch", r["id"], ok, valid, r["success"], r["valid"], r["val"]))
            if r["verified"] and not ok: problems.append((R, cfg, "verified-but-invalid", r["id"]))
            if not ok: simfail["invalid" if not valid else "goal_not_met"] += 1
        succ = {r["id"]: int(bool(r.get("val", r["success"]))) for r in rows}
        data[(R, cfg)] = (rows, succ)
        # shard union
        if cfg == "beam":
            shard_rows = []
            for s in range(4):
                sr, siss = read_rows(d / f"shard_{s}of4" / "plans.jsonl")
                if siss: problems.append((R, cfg, s, siss))
                exp = [x for j, x in enumerate(ids) if j % 4 == s]
                got = [r["id"] for r in sr]
                if got != exp: problems.append((R, cfg, s, "shard ids", got, exp))
                shard_rows += sr
                ss = json.loads((d / f"shard_{s}of4" / "summary.json").read_text())
                a = ss["args"]; sm = ss["summary"]
                if a["shard"] != f"{s}/4" or a["set"] != "seeds_fresh" or a["freeze"] != f"runs/FREEZE_I7_{R}_beam.json" or a["n"] != 0 or not a["val"]:
                    problems.append((R, s, "shard args", a["shard"], a["freeze"]))
                frz = json.loads((ROOT / a["freeze"]).read_text())["args"]
                bad = {k: (a.get(k), v) for k, v in frz.items() if a.get(k) != v}
                if bad: problems.append((R, s, "args vs freeze", bad))
                sk = sum(bool(r["val"]) for r in sr)
                if abs(sm["success"] - sum(r["success"] for r in sr) / len(sr)) > 1e-12 or abs(sm["val_success"] - sk / len(sr)) > 1e-12 or sm["val_disagreements"] != 0 or sm["n"] != len(sr):
                    problems.append((R, s, "shard summary mismatch", sm["success"], sk))
                fc = sum(1 for _ in open(d / f"shard_{s}of4" / "fcache.jsonl", encoding="utf-8"))
                print(f"  {R} beam shard {s}: n={len(sr)} succ={sk} f_model_queries={sm['f_model_queries']} fcache_lines={fc} wall={sm['wall_s']} ptok/inst={sm['proposer_tokens_per_instance']}")
            if [json.dumps(r, sort_keys=True) for r in shard_rows] != [json.dumps(r, sort_keys=True) for r in rows]:
                problems.append((R, cfg, "merged != concat(shards)"))
            else:
                print(f"  {R} beam merged == concatenation of shards 0..3 (row-for-row, all fields)")
        else:
            ss = json.loads((d / "summary.json").read_text())
            a = ss["args"]
            frz = json.loads((ROOT / a["freeze"]).read_text())["args"]
            bad = {k: (a.get(k), v) for k, v in frz.items() if a.get(k) != v}
            if bad or a["freeze"] != f"runs/FREEZE_I7_{R}_direct.json" or a["shard"] != "" or a["n"] != 0 or not a["val"]:
                problems.append((R, cfg, "args vs freeze", bad, a["freeze"]))
            if rid != ids: problems.append((R, cfg, "order", rid[:5]))
        # summary vs plans
        sm = json.loads((d / "summary.json").read_text())["summary"]
        k = sum(succ.values())
        chk = {"n": (sm["n"], len(rows)), "success": (sm["success"], sum(r["success"] for r in rows) / len(rows)),
               "val_success": (sm["val_success"], k / len(rows)),
               "val_dis": (sm["val_disagreements"], sum(bool(r["val"]) != bool(r["success"]) for r in rows)),
               "verified_rate": (sm["verified_rate"], sum(r["verified"] for r in rows) / len(rows)),
               "ptok": (sm["proposer_tokens_per_instance"], round(sum(r["gen_tokens"] for r in rows) / len(rows), 1))}
        by = defaultdict(list)
        for r in rows: by[nb[r["id"]]].append(succ[r["id"]])
        mysbb = {str(b): round(sum(v) / len(v), 4) for b, v in sorted(by.items())}
        chk["sbb"] = (sm["success_by_blocks"], mysbb)
        for key, (x, y) in chk.items():
            if (isinstance(x, float) and abs(x - y) > 1e-9) or (not isinstance(x, float) and x != y):
                problems.append((R, cfg, "summary mismatch", key, x, y))
        print(f"{R} {cfg}: val {k}/{len(rows)}; success-field {sum(bool(r['success']) for r in rows)}; verified {sum(bool(r['verified']) for r in rows)}; "
              f"valid {sum(bool(r['valid']) for r in rows)}; fails {dict(simfail)}; ptok/inst {chk['ptok'][1]}; summary success {sm['success']} val_success {sm['val_success']} val_dis {sm['val_disagreements']}")
        print("   by blocks (k/n):", {b: f"{sum(v)}/{len(v)}" for b, v in sorted(by.items())})
        print("   fail reasons:", dict(Counter(r["fail"] for r in rows if not r["val"])))

print("FIELD SETS:", dict(FIELDS))
print("PROBLEMS:", problems)


# ---- stats (own implementations, cross-checked)
def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def cp_bisect(k, n):
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            m = (lo + hi) / 2
            if f(m): hi = m
            else: lo = m
        return (lo + hi) / 2
    L = 0.0 if k == 0 else solve(lambda p: binom.sf(k - 1, n, p) >= 0.025)
    U = 1.0 if k == n else solve(lambda p: binom.cdf(k, n, p) <= 0.025)
    return L, U


def mcn_exact(b, c):
    n = b + c
    if n == 0: return Fraction(1)
    m = min(b, c)
    tail = sum(Fraction(math.comb(n, i), 2 ** n) for i in range(m + 1))
    return min(Fraction(1), 2 * tail)


rep = json.loads((OUT / "SEEDS_REPORT.json").read_text())
mine = {}
for R in ("R0", "R1", "R2"):
    be, di = data[(R, "beam")][1], data[(R, "direct")][1]
    kb, kd = sum(be.values()), sum(di.values())
    b = sum(1 for i in ids if be[i] and not di[i]); c = sum(1 for i in ids if di[i] and not be[i])
    both = sum(1 for i in ids if be[i] and di[i]); neither = sum(1 for i in ids if not be[i] and not di[i])
    p = mcn_exact(b, c); ps = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    delta = Fraction(kb - kd, N)
    assert delta == Fraction(b - c, N)
    rule = (float(p) < 0.01) and (kb > kd) and (delta >= Fraction(3, 10))
    ciB, ciD = cp(kb, N), cp(kd, N)
    ciB2, ciD2 = cp_bisect(kb, N), cp_bisect(kd, N)
    print(f"{R}: beam {kb}/40 CI [{ciB[0]:.6f},{ciB[1]:.6f}] (bisect [{ciB2[0]:.6f},{ciB2[1]:.6f}]); direct {kd}/40 CI [{ciD[0]:.6f},{ciD[1]:.6f}] (bisect [{ciD2[0]:.6f},{ciD2[1]:.6f}])")
    print(f"    table both={both} beam_only={b} direct_only={c} neither={neither}; delta={float(delta)}; McNemar exact p={p} = {float(p):.10e} (scipy {ps:.10e}); rule={rule}")
    mine[R] = dict(kb=kb, kd=kd, b=b, c=c, p=float(p), delta=float(delta), rule=rule, ciB=ciB, ciD=ciD)
    r = rep["systems"][R]
    cmp = {"s_beam": (r["s_beam"], f"{kb}/{N}"), "s_direct": (r["s_direct"], f"{kd}/{N}"), "beam_only": (r["beam_only"], b),
           "direct_only": (r["direct_only"], c), "delta": (r["delta"], round(float(delta), 4)), "p": (r["p_mcnemar"], float(p)),
           "rep": (r["replicates"], rule), "ciB": (r["s_beam_ci95"], [round(x, 4) for x in ciB]),
           "ciD": (r["s_direct_ci95"], [round(x, 4) for x in ciD])}
    for kk, (x, y) in cmp.items():
        okk = (abs(x - y) < 1e-15) if isinstance(x, float) else (x == y)
        print(f"    {kk}: report={x} mine={y} {'OK' if okk else 'MISMATCH'}")
gate = mine["R1"]["rule"] and mine["R2"]["rule"]
print("release gate:", gate, "| report:", rep["release_gate_reading"])
for R in ("R1", "R2"):
    for cfg in ("beam", "direct"):
        x, y = data[(R, cfg)][1], data[("R0", cfg)][1]
        b = sum(1 for i in ids if x[i] and not y[i]); c = sum(1 for i in ids if y[i] and not x[i])
        p = mcn_exact(b, c)
        r = rep["spread_report_only"][f"{R}_vs_R0_{cfg}"]
        print(f"spread {R} vs R0 {cfg}: only_{R}={b} only_R0={c} p={p}={float(p)} | report {r} | match={r['only_' + R] == b and r['only_R0'] == c and abs(r['p_mcnemar'] - float(p)) < 1e-15}")
        print("    only", R, [i.split('/')[1] for i in ids if x[i] and not y[i]], "only R0", [i.split('/')[1] for i in ids if y[i] and not x[i]])
print("per-instance (idx nb: R0b R0d R1b R1d R2b R2d):")
for i in ids:
    print("  ", i.split('/')[1], nb[i], "".join(str(data[(R, c)][1][i]) for R in ("R0", "R1", "R2") for c in ("beam", "direct")))
# sensitivity: what minimal change flips R2's rule?
for R in ("R1", "R2"):
    m = mine[R]
    print(R, "margin: delta - 0.30 =", round(m["delta"] - 0.3, 4), "-> beam successes above threshold:", m["b"] - m["c"] - 12)
