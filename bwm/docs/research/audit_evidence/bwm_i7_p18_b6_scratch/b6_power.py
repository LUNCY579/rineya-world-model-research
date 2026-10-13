"""Exact power of the two-sided exact McNemar test (and of the B3-style MET rule: diff >= bar pp AND p < alpha)
under a trinomial model of discordance: P(new-only)=p10, P(ctrl-only)=p01 per instance."""
import math, sys
from functools import lru_cache
from scipy.stats import binom, binomtest

@lru_cache(maxsize=None)
def pval(b, c):
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue

def power(n, p10, p01, alpha=0.05, bar_pp=None):
    pd = p10 + p01
    q = p10 / pd
    pw = met = 0.0
    for d in range(n + 1):
        wd = binom.pmf(d, n, pd)
        if wd < 1e-12:
            continue
        for b in range(d + 1):
            w = wd * binom.pmf(b, d, q)
            if w < 1e-14:
                continue
            c = d - b
            if b > c and pval(b, c) < alpha:
                pw += w
                if bar_pp is None or 100.0 * (b - c) / n >= bar_pp:
                    met += w
    return pw, met

scen = [("+7pp (p10 .10, p01 .03)", .10, .03), ("+10pp (p10 .13, p01 .03)", .13, .03),
        ("round-1 rate (p10 .164, p01 .036; +12.7pp)", .164, .036), ("+15pp (p10 .18, p01 .03)", .18, .03),
        ("+20pp (p10 .23, p01 .03)", .23, .03), ("+10pp high-discord (p10 .18, p01 .08)", .18, .08)]
for n in (60, 74, 110, 150, 200):
    for name, a, b in scen:
        pw, met = power(n, a, b, bar_pp=10)
        print(f"n={n:3d} {name:45s} power(p<.05)={pw:.3f}  P(MET: >=+10pp & p<.05)={met:.3f}")
    print()
# beam-style scenarios (higher discordance): H2-vs-B2 on hard was +20/-5 of 110 (p10 .18, p01 .045)
for n in (60, 74, 110):
    for name, a, b in [("beam +5pp (p10 .10, p01 .05)", .10, .05), ("beam +8pp (p10 .12, p01 .04)", .12, .04),
                       ("beam H2-like (p10 .18, p01 .045)", .18, .045)]:
        pw, _ = power(n, a, b)
        print(f"n={n:3d} {name:40s} power(p<.05)={pw:.3f}")
