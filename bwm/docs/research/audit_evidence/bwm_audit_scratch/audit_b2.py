import json, sys, statistics as st
from collections import defaultdict, Counter
from pathlib import Path
from scipy.stats import binomtest, fisher_exact
from scipy.stats import beta

ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import blocks as B
import blocks_val_large as VL

T = ROOT / "runs/BW_TEST_B2"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ok(r):
    return bool(r.get("val", r["success"]))


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


# expected id sets
def ids_for(name):
    if name == "hard":
        ps = sorted((B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl"), key=lambda p: int(p.stem.split("-")[1]))
        return [f"hard/{p.stem}" for p in ps]
    if name == "mirror":
        return [f"mirror/{i}" for i in range(len(json.loads((ROOT / "runs/BW_val_large/instances_mirror_hard.json").read_text())))]
    if name == "planbench":
        out = []
        for sub in ("generated_basic_3", "generated_basic"):
            ps = sorted((B.BENCH / "blocksworld" / sub).glob("*.pddl"), key=lambda p: int(p.stem.split("-")[1]))
            out += [f"{sub}/{p.stem}" for p in ps]
        return out


res = {}
for s in ("hard", "mirror", "planbench"):
    exp = ids_for(s)
    print(f"\n===== {s}: expected {len(exp)} ids, unique {len(set(exp))}")
    for sysn in ("beam", "direct"):
        d = T / f"{s}_{sysn}"
        rows = load(d / "plans.jsonl")
        ids = [r["id"] for r in rows]
        dup = [k for k, v in Counter(ids).items() if v > 1]
        missing = set(exp) - set(ids)
        extra = set(ids) - set(exp)
        noval = sum("val" not in r for r in rows)
        disc = sum(bool(r.get("val", r["success"])) != bool(r["success"]) for r in rows)
        k = sum(ok(r) for r in rows)
        ksucc = sum(bool(r["success"]) for r in rows)
        summ = json.loads((d / "summary.json").read_text())["summary"]
        print(f"[{s}_{sysn}] n={len(rows)} k(val)={k} k(success)={ksucc} noval={noval} val_disagree={disc} dup={dup} missing={len(missing)} extra={len(extra)} order_eq={ids == exp}")
        print(f"   summary: n={summ['n']} success={summ['success']} val_success={summ.get('val_success')} vd={summ.get('val_disagreements')} "
              f"-> recomputed success={ksucc/len(rows)} val_success={k/len(rows)}  match_n={summ['n']==len(rows)} match_s={abs(summ['success']-ksucc/len(rows))<1e-12} match_v={abs(summ['val_success']-k/len(rows))<1e-12} match_vd={summ['val_disagreements']==disc}")
        # tokens per instance
        tok = round(sum(r["gen_tokens"] for r in rows) / len(rows), 1)
        print(f"   proposer_tokens_per_instance recomputed {tok} vs summary {summ.get('proposer_tokens_per_instance')}; verified_rate {sum(r['verified'] for r in rows)/len(rows)} vs {summ.get('verified_rate')}")
        # shards
        shards = sorted(d.glob("shard_*"))
        if shards:
            srows = []
            fq = 0
            wall = 0
            for sh in shards:
                i, kk = sh.name[len("shard_"):].split("of")
                i, kk = int(i), int(kk)
                rr = load(sh / "plans.jsonl")
                ss = json.loads((sh / "summary.json").read_text())
                # shard membership: index % k == i
                expected_sh = [x for j, x in enumerate(exp) if j % kk == i]
                if [r["id"] for r in rr] != expected_sh:
                    print(f"   SHARD MEMBERSHIP MISMATCH {sh.name}: got {len(rr)} expected {len(expected_sh)}")
                # shard summary vs plans
                sk = sum(ok(r) for r in rr)
                sd = sum(bool(r.get('val', r['success'])) != bool(r['success']) for r in rr)
                if ss["summary"]["n"] != len(rr) or abs(ss["summary"]["val_success"] - sk / len(rr)) > 1e-12 or ss["summary"]["val_disagreements"] != sd:
                    print(f"   SHARD SUMMARY MISMATCH {sh.name}")
                # args vs freeze
                frz = json.loads((ROOT / "runs/FREEZE_B2_beam.json").read_text())["args"]
                a = ss["args"]
                bad = {kk2: (a.get(kk2), frz.get(kk2)) for kk2 in frz if a.get(kk2) != frz.get(kk2)}
                if bad or a.get("freeze") != "runs/FREEZE_B2_beam.json" or a.get("set") != s or a.get("shard") != f"{i}/{kk}" or not a.get("val"):
                    print(f"   SHARD ARGS ISSUE {sh.name}: {bad} freeze={a.get('freeze')} set={a.get('set')} shard={a.get('shard')} val={a.get('val')} n={a.get('n')}")
                if a.get("n"):
                    print(f"   SHARD n arg nonzero {sh.name}: {a.get('n')}")
                srows += rr
                fq += ss["summary"]["f_model_queries"] or 0
                wall += ss["summary"]["wall_s"]
            print(f"   shards={len(shards)} concat_equals_merged={[json.dumps(r) for r in srows] == [json.dumps(r) for r in rows]} "
                  f"concat_ids_set_eq={sorted(r['id'] for r in srows) == sorted(exp)} fq_sum={fq} vs {summ['f_model_queries']} wall_sum={round(wall,1)} vs {summ['wall_s']} fq/inst={round(fq/len(rows),1)}")
        else:
            a = json.loads((d / "summary.json").read_text()).get("args", {})
            frz = json.loads((ROOT / f"runs/FREEZE_B2_{sysn}.json").read_text())["args"]
            bad = {kk2: (a.get(kk2), frz.get(kk2)) for kk2 in frz if a.get(kk2) != frz.get(kk2)}
            print(f"   args vs freeze diff: {bad}; freeze arg={a.get('freeze')} n={a.get('n')} val={a.get('val')}")
        res[(s, sysn)] = {r["id"]: ok(r) for r in rows}

print("\n===== P-10 stats")
for s in ("hard", "mirror", "planbench"):
    b, dr = res[(s, "beam")], res[(s, "direct")]
    assert set(b) == set(dr)
    n = len(b)
    kb, kd = sum(b.values()), sum(dr.values())
    plus = sum(b[i] and not dr[i] for i in b)
    minus = sum(dr[i] and not b[i] for i in b)
    p = binomtest(plus, plus + minus, 0.5).pvalue if plus + minus else 1.0
    lo, hi = cp(kb, n)
    lod, hid = cp(kd, n)
    print(f"{s}: beam {kb}/{n}={kb/n:.4f} CI {lo:.4f}-{hi:.4f} | direct {kd}/{n}={kd/n:.4f} CI {lod:.4f}-{hid:.4f} | +{plus}/-{minus} exact McNemar p={p:.4g}")

hb, mb = res[("hard", "beam")], res[("mirror", "beam")]
kh, km = sum(hb.values()), sum(mb.values())
gap = 100 * (kh / len(hb) - km / len(mb))
odds, pf = fisher_exact([[kh, len(hb) - kh], [km, len(mb) - km]])
print(f"gap hard-mirror beam = {gap:.3f} pp; Fisher two-sided p = {pf:.4f}")
for alt in ("greater", "less"):
    print(f"  Fisher {alt}: {fisher_exact([[kh, len(hb)-kh],[km, len(mb)-km]], alternative=alt)[1]:.4f}")
hd, md = res[("hard", "direct")], res[("mirror", "direct")]
print(f"direct gap = {100*(sum(hd.values())/110 - sum(md.values())/110):.2f} pp")

# cost per instance
print("\n===== MIRROR GAP recompute")
hard = []
for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl"):
    o, i, g = B.parse_problem(p)
    hard.append((f"hard/{p.stem}", o, i, g))
mirror = [(f"mirror/{k}", r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"])))
          for k, r in enumerate(json.loads((ROOT / "runs/BW_val_large/instances_mirror_hard.json").read_text()))]
succ = {**hb, **mb}


def must_move(objs, init, goal):
    on_i = {f[1]: f[2] for f in init if f[0] == "on"}
    on_g = {f[1]: f[2] for f in goal if f[0] == "on"}
    memo = {}

    def settled(x):
        if x not in memo:
            below = on_i.get(x)
            if x in on_g and on_g[x] != below:
                memo[x] = False
            else:
                memo[x] = True if below is None else settled(below)
        return memo[x]
    return sum(not settled(x) for x in objs)


def bin_of(h):
    return "<=6" if h <= 6 else "7-9" if h <= 9 else ">=10"


out = {}
bins = {}
for name, insts in (("hard", hard), ("mirror", mirror)):
    lb, gmax, gcount, imax, by = [], [], [], [], defaultdict(list)
    ge10 = 0
    for iid, o, i, g in insts:
        gt = [h for h in VL.goal_towers(g, o) if h > 1]
        it = [h for h in VL.towers_of(i, o) if h > 1]
        lb.append(2 * must_move(o, i, g))
        gmax.append(max(gt) if gt else 1)
        gcount.append(len(gt))
        imax.append(max(it) if it else 1)
        by[bin_of(gmax[-1])].append(int(succ[iid]))
    bins[name] = by
    out[name] = {"n": len(insts), "success": round(sum(succ[x[0]] for x in insts) / len(insts), 4),
                 "mean_move_lower_bound": round(st.mean(lb), 2), "mean_max_goal_tower": round(st.mean(gmax), 2),
                 "mean_goal_towers": round(st.mean(gcount), 2), "mean_max_init_tower": round(st.mean(imax), 2),
                 "success_by_max_goal_tower": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())},
                 "n_goal_tower_ge10": sum(h >= 10 for h in gmax)}
n_h = sum(len(v) for v in bins["hard"].values())
std = sum(len(bins["hard"][k]) * sum(bins["mirror"][k]) / len(bins["mirror"][k]) for k in bins["hard"]) / n_h
out["mirror_standardized_to_hard_goal_tower_mix"] = round(std, 4)
out["mirror_standardized_unrounded"] = std
out["gap_pp_raw"] = round(100 * (out["hard"]["success"] - out["mirror"]["success"]), 1)
out["gap_pp_standardized"] = round(100 * (out["hard"]["success"] - std), 1)
out["gap_pp_standardized_unrounded_inputs"] = 100 * (kh / 110 - std)
print(json.dumps(out, indent=1))
stored = json.loads((T / "MIRROR_GAP_ANALYSIS.json").read_text())
for k in stored:
    if k == "note":
        continue
    if isinstance(stored[k], dict):
        for kk in stored[k]:
            if stored[k][kk] != out[k][kk]:
                print("MISMATCH", k, kk, stored[k][kk], out[k][kk])
    elif stored[k] != out[k]:
        print("MISMATCH", k, stored[k], out[k])
print("stored-vs-recomputed comparison done")
