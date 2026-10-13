from math import comb, exp, lgamma, log
from functools import lru_cache
@lru_cache(None)
def pmc(b, c):
    m = b + c
    if m == 0: return 1.0
    k = min(b, c)
    s = sum(comb(m, i) for i in range(k + 1)) / 2 ** m
    return min(1.0, 2 * s)
def power(n, p10, p01, alpha=0.05, bar=None):
    q = 1 - p10 - p01
    lp10, lp01, lq = log(p10) if p10 > 0 else None, log(p01) if p01 > 0 else None, log(q)
    tot = met = 0.0
    for b in range(n + 1):
        for c in range(n + 1 - b):
            if b <= c: continue
            if pmc(b, c) >= alpha: continue
            if (p01 == 0 and c > 0): continue
            lw = lgamma(n + 1) - lgamma(b + 1) - lgamma(c + 1) - lgamma(n - b - c + 1) + b * lp10 + (c * lp01 if c else 0) + (n - b - c) * lq
            w = exp(lw)
            tot += w
            if bar is not None and (b - c) / n >= bar - 1e-12: met += w
    return tot, met
def null_reject(n, p, alpha=0.05, bar=0.10):
    return power(n, p, p, alpha, bar)
print("PRIMARY S-direct, n pairs; (p10, p01): P(p<.05 & b>c), P(MET: >= +10pp & p<.05)")
sc = [("null, discordance .04/.04", .04, .04), ("+8pp r1 11-15 analog (.095/.014)", .095, .014), ("+10pp (.13/.03)", .13, .03),
      ("+12.7pp r1 overall (.164/.036)", .164, .036), ("+15pp (.18/.03)", .18, .03), ("+18pp (.21/.03)", .21, .03),
      ("+22pp r1 in-range (.31/.08)", .31, .08)]
for name, a, b in sc:
    row = []
    for n in (74, 110, 148):
        t, m = power(n, a, b, bar=0.10)
        row.append(f"n={n}: {t:.3f} [{m:.3f}]")
    print(f"{name:40s}", "  ".join(row))
print("Holm m=2 at alpha .025 (first step) n=110:")
for name, a, b in sc[1:]:
    t, m = power(110, a, b, alpha=0.025, bar=0.10)
    print(f"  {name:40s} {t:.3f} [{m:.3f}]")
print("H1b (S-direct Pa_v4 vs H1 on hard, n=110):")
for name, a, b in [("+10 at 11-15, -3 at 6-10: .109/.027", 12/110, 3/110), ("+15/-3: .155/.027", 17/110, 3/110), ("+8/-4", 8/110, 4/110)]:
    t, _ = power(110, a, b); print(f"  {name:40s} {t:.3f}")
print("H2b (S-beam Pa_v4 vs H2 88/110 on hard; at most 22 movable):")
for name, a, b in [("+12/-6 (r1 relative rates)", 12/110, 6/110), ("+9/-3", 9/110, 3/110), ("+12/-3", 12/110, 3/110), ("+15/-3", 15/110, 3/110)]:
    t, _ = power(110, a, b); print(f"  {name:40s} {t:.3f}")
print("min b for p<.05 given c:", {c: next(b for b in range(c + 1, 200) if pmc(b, c) < .05) for c in range(0, 8)})
print("--- conditional model (comparator outcomes fixed): b~Bin(F,r), c~Bin(S,l)")
def cpow(F, S, r, l, alpha=0.05):
    pb = [comb(F, b) * r**b * (1-r)**(F-b) for b in range(F+1)]
    pc = [comb(S, c) * l**c * (1-l)**(S-c) for c in range(S+1)]
    return sum(pb[b]*pc[c] for b in range(F+1) for c in range(S+1) if b > c and pmc(b, c) < alpha)
for name, F, S, r, l in [("H2b r1-relative (conv .54, loss .068)", 22, 88, .54, .068), ("H2b conv .40 loss .03", 22, 88, .40, .03),
                         ("H2b conv .30 loss .03", 22, 88, .30, .03), ("H2b conv .55 loss .03", 22, 88, .55, .03),
                         ("H1b conv .12 loss .15", 97, 13, .12, .15), ("H1b conv .17 loss .15", 97, 13, .17, .15)]:
    print(f"  {name:45s} {cpow(F, S, r, l):.3f}   E[b]={F*r:.1f} E[c]={S*l:.1f}")
print("Holm-adjusted (alpha .025) H2b r1-relative:", round(cpow(22, 88, .54, .068, .025), 3), " conv .55 loss .03:", round(cpow(22, 88, .55, .03, .025), 3))
