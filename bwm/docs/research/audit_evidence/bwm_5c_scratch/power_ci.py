# Report-only context for the release card: paired CIs and the guard's power at m = 5 (own code; no ret_analysis import).
import json, math
import numpy as np
S = json.load(open("recompute_summary.json"))


def wald(lost, gained, n):
    d = (gained - lost) / n
    var = ((lost + gained) - (gained - lost) ** 2 / n) / n ** 2
    se = math.sqrt(max(var, 0.0))
    return d, (d - 1.959964 * se) * n, (d + 1.959964 * se) * n


for name in ("x_bwf_vs_off", "x_bwpa_vs_off"):
    for gr in ("primary", "lenient"):
        r = S[name]["res"][gr]
        print(f"{name} {gr}: pooled net {r['pooled']:+d}/375, 95% Wald CI in items "
              f"[{wald(r['lost'], r['gained'], 375)[1]:+.1f}, {wald(r['lost'], r['gained'], 375)[2]:+.1f}] (margin -15)")
        for c, v in r["cats"].items():
            _, lo, hi = wald(v["lost"], v["gained"], 75)
            print(f"    {c:14s} net {v['net']:+3d}/75  CI items [{lo:+.1f}, {hi:+.1f}]  (category margin -11){'  <-- CI crosses -11' if lo < -11 else ''}")

# ---- power of the registered rule at m = 5, n = 75, single grader (seeded simulation)
rng = np.random.default_rng(20261011)
REPS = 40000
ref_correct = np.array([30, 37, 52, 54, 52])           # OFF primary counts on this draw (unit_conv first)
n = 75
m = 5


def mcn(l, g):
    t = l + g
    return 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, i) for i in range(min(l, g) + 1)) / 2 ** t)


P = np.array([[mcn(l, g) for g in range(n + 1)] for l in range(n + 1)])


def sim(D, k_one=0, k_all=0):
    extra = np.full(m, k_all)
    extra[0] += k_one
    ql = (D / 2) / ref_correct
    qg = (D / 2) / (n - ref_correct)
    lost = rng.binomial(ref_correct - extra, ql, size=(REPS, m)) + extra
    gained = rng.binomial(n - ref_correct, qg, size=(REPS, m))
    p = P[lost, gained]
    order = np.argsort(p, axis=1, kind="stable")
    ps = np.take_along_axis(p, order, 1)
    adj_s = np.minimum(1.0, np.maximum.accumulate(ps * (m - np.arange(m)), axis=1))
    adj = np.empty_like(p)
    np.put_along_axis(adj, order, adj_s, 1)
    det = ((lost > gained) & (adj < 0.05)).any(1)
    net = gained - lost
    cat = (net < -11).any(1)
    pooled = net.sum(1) < -0.04 * n * m
    return float((det | cat | pooled).mean())


print("\nP(NO-GO), single grader, m=5, ref counts", ref_correct.tolist())
for D in (8.0, 12.0):
    row = [f"null {sim(D):.3f}"]
    for k in (4, 6, 8, 10, 12, 15):
        row.append(f"k{k}-in-unit_conv {sim(D, k_one=k):.3f}")
    print(f"  background discordance D={D:g}/category:", ", ".join(row))
    row = []
    for j in (1, 2, 3, 4):
        row.append(f"{j}-in-every-cat ({5 * j} pooled) {sim(D, k_all=j):.3f}")
    print("     diffuse:", ", ".join(row))
print("\nobserved discordance per category, x_bwf primary:",
      {c: v["lost"] + v["gained"] for c, v in S["x_bwf_vs_off"]["res"]["primary"]["cats"].items()})
print("observed discordance per category, x_bwpa primary:",
      {c: v["lost"] + v["gained"] for c, v in S["x_bwpa_vs_off"]["res"]["primary"]["cats"].items()})
