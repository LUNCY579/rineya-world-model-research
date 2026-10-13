import math
from functools import lru_cache
def binom_two_sided(b, m):
    # exact two-sided binomial test at 0.5 (scipy's method: sum probs <= prob(b) with tolerance)
    if m == 0: return 1.0
    pr = [math.comb(m, k) * 0.5**m for k in range(m+1)]
    pb = pr[b]
    return min(1.0, sum(p for p in pr if p <= pb * (1 + 1e-7)))
P = {}
def pv(b, m):
    if (b, m) not in P: P[(b, m)] = binom_two_sided(b, m)
    return P[(b, m)]
def power(n, p10, p01, met_pp=None, met_net=None):
    p00 = 1 - p10 - p01
    lp10, lp01, lp00 = math.log(p10), math.log(p01), math.log(p00)
    sig = met = 0.0
    lf = [math.lgamma(k+1) for k in range(n+1)]
    for b in range(n+1):
        for c in range(n+1-b):
            lpr = lf[n]-lf[b]-lf[c]-lf[n-b-c] + b*lp10 + c*lp01 + (n-b-c)*lp00
            pr = math.exp(lpr)
            if pr < 1e-15: continue
            if b > c and pv(b, b+c) < 0.05:
                sig += pr
                ok = True
                if met_pp is not None and round(100*(b-c)/n, 1) < met_pp: ok = False
                if met_net is not None and b - c < met_net: ok = False
                if ok: met += pr
    return round(sig, 3), round(met, 3)
sc = {"null.04": (.04,.04), "oor .095/.014": (.095,.014), "+10 .13/.03": (.13,.03), "P11 .164/.036": (.164,.036),
      "+15 .18/.03": (.18,.03), "inrange .31/.08": (.31,.08), "eff central .17/.04": (.17,.04), "+5 .08/.03": (.08,.03)}
for n in (74, 110, 150):
    for k, (a, b) in sc.items():
        print(n, k, "sig/MET10", power(n, a, b, met_pp=10), "rigorMET(+5pp)", power(n, a, b, met_net=math.ceil(0.05*n - 1e-9)) if n == 150 else "")
