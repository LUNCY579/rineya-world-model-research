from math import comb, lgamma, log, exp
def lbin(n,k): return lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)
def pval(b,c):
    m=b+c
    if m==0: return 1.0
    k=min(b,c)
    t=sum(exp(lbin(m,i)-m*log(2)) for i in range(k+1))
    return min(1.0,2*t)
PV={}
def pv(b,c):
    if (b,c) not in PV: PV[(b,c)]=pval(b,c)
    return PV[(b,c)]
def power(n,p10,p01,alpha,bar_pp=None):
    p00=1-p10-p01
    tot=0.0
    lp10,lp01,lp00=log(p10),log(p01),log(p00) if p00>0 else -1e300
    for b in range(n+1):
        for c in range(n+1-b):
            if b<=c: continue
            if bar_pp is not None and 100*(b-c)/n < bar_pp: continue
            if pv(b,c) >= alpha: continue
            r=n-b-c
            tot+=exp(lgamma(n+1)-lgamma(b+1)-lgamma(c+1)-lgamma(r+1)+b*lp10+c*lp01+r*lp00)
    return tot
import sys
print("S-direct scenarios (power at a=0.05 | a=0.025 | P(MET: a=0.05 & >=+10pp))")
for lab,p10,p01 in [("+8pp r1 out-of-range analog (.095/.014)",.095,.014),("+11pp (.15/.04)",.15,.04),("+15pp (.18/.03)",.18,.03),("+22pp r1 in-range analog (.306/.083)",.306,.083),("null-ish 0 (.05/.05)",.05,.05)]:
    row=[]
    for n in (110,150,200):
        row.append(f"n={n}: {power(n,p10,p01,.05):.2f} | {power(n,p10,p01,.025):.2f} | {power(n,p10,p01,.05,10):.2f}")
    print(lab, " ; ".join(row))
print("S-beam scenarios")
for lab,p10,p01 in [("H2-vs-B2 11-15 analog +17.6pp (.23/.054)",.23,.054),("+12pp noisy (.20/.08)",.20,.08),("+12pp (.16/.04)",.16,.04),("+8pp noisy (.17/.09)",.17,.09),("null, seed-noise discordance 25% (.125/.125)",.125,.125)]:
    row=[]
    for n in (60,80,100):
        row.append(f"n={n}: {power(n,p10,p01,.05):.2f} | {power(n,p10,p01,.025):.2f} | {power(n,p10,p01,.05,10):.2f}")
    print(lab, " ; ".join(row))
