from math import comb, lgamma, exp, log
from functools import lru_cache
from scipy.stats import binomtest
@lru_cache(None)
def pv(b,c):
    return 1.0 if b+c==0 else binomtest(b,b+c,0.5).pvalue
def lmult(n,b,c): return lgamma(n+1)-lgamma(b+1)-lgamma(c+1)-lgamma(n-b-c+1)
def power(n,p10,p01,bar_frac=None,min_bc=None):
    p00=1-p10-p01; P=0; M=0
    for b in range(n+1):
        for c in range(n+1-b):
            pr=exp(lmult(n,b,c)+(b*log(p10) if b else 0)+(c*log(p01) if c else 0)+((n-b-c)*log(p00) if n-b-c else 0))
            if pr<1e-14: continue
            if b>c and pv(b,c)<0.05:
                P+=pr
                ok=True
                if bar_frac is not None and round(100*(b-c)/n,1) < 100*bar_frac: ok=False
                if min_bc is not None and b-c<min_bc: ok=False
                if ok: M+=pr
    return round(P,3), round(M,3)
print("faithful n110 +10", power(110,.13,.03,0.10))
print("faithful n110 r1 out", power(110,.095,.014,0.10))
print("faithful n110 r1 overall", power(110,.164,.036,0.10))
print("effect n150 central", power(150,.17,.04,0.10))
print("effect n150 +10", power(150,.13,.03,0.10))
print("effect n150 r1out", power(150,.095,.014,0.10))
print("effect P2 n80 H2like", power(80,.226,.065,0.10))
print("effect P2 n80 +12", power(80,.17,.05,0.10))
print("rigor n150 +8 minbc8", power(150,.095,.014,None,8))
print("rigor n150 +10 minbc8", power(150,.13,.03,None,8))
print("rigor n150 null .12/.12", power(150,.12,.12,None,8))
# what if control rate low and effect concentrated: v4 21%, v3c 10% with high discordance
print("rigor n150 +8 high-disc .14/.06", power(150,.14,.06,None,8))
print("faithful n110 +8 high-disc .14/.06", power(110,.14,.06,0.10))
