from functools import lru_cache
from scipy.stats import binom, binomtest
@lru_cache(maxsize=None)
def pval(b, c):
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
@lru_cache(maxsize=None)
def tri(n, p10, p01):
    out = {}; pd = p10 + p01; q = p10 / pd if pd else 0.5
    for d in range(n + 1):
        wd = binom.pmf(d, n, pd)
        if wd < 1e-13: continue
        for b in range(d + 1):
            w = wd * binom.pmf(b, d, q)
            if w >= 1e-15: out[(b, d - b)] = w
    return out
def two_stage(n1, n2, p10, p01, fut_net=1, alpha=0.05):
    s1, s2 = tri(n1, p10, p01), tri(n2, p10, p01); rej = stop = 0.0
    for (b1, c1), w1 in s1.items():
        if b1 - c1 <= fut_net: stop += w1; continue
        for (b2, c2), w2 in s2.items():
            b, c = b1 + b2, c1 + c2
            if b > c and pval(b, c) < alpha: rej += w1 * w2
    return rej, stop, n1 + (1 - stop) * n2
for n1, n2 in ((40, 40), (50, 50)):
    for name, a, b in [("+15pp (.20/.05)", .20, .05), ("+15pp hi-disc (.23/.08)", .23, .08), ("null hi (.12/.12)", .12, .12)]:
        r, s, en = two_stage(n1, n2, a, b, 1)
        print(f"n1={n1} n2={n2} fut<=1 {name:26s} P(reject+)={r:.3f} P(stop)={s:.3f} E[n]={en:.1f}")
for n in (100, 150):
    worst = 0
    for pd in (0.02, 0.05, 0.1, 0.16, 0.24, 0.3, 0.4):
        t = tri(n, pd/2, pd/2); worst = max(worst, sum(w for (b, c), w in t.items() if pval(b, c) < 0.05))
    print("single-stage two-sided exact size n", n, "worst over discordance", round(worst, 4))
for a in (.05, .1, .15):
    t = tri(60, a, a); print("retention n=60 null disc", a, "P(block)", round(sum(w for (b, c), w in t.items() if c > b and pval(b, c) < .05), 4))
for a, b in ((.05, .15), (.05, .20), (.10, .25)):
    t = tri(60, a, b); print("retention n=60 harm", a, b, "P(block)", round(sum(w for (bb, c), w in t.items() if c > bb and pval(bb, c) < .05), 4))
