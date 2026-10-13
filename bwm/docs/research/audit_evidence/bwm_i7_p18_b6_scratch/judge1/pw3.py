import math
from pw import pv
def bpmf(n,p): return [math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)]
def pw(nb,pb,nc,pc,alpha=0.05):
    B=bpmf(nb,pb); C=bpmf(nc,pc); s=0
    for b,x in enumerate(B):
        for c,y in enumerate(C):
            if b>c and pv(b,b+c)<alpha: s+=x*y
    return round(s,3)
print("H1b .12/.15", pw(97,.12,13,.15), "H1b .17/.15", pw(97,.17,13,.15))
print("H2b .54/.068", pw(22,20/37,88,5/73), "H2b .40/.03", pw(22,.40,88,.03), "H2b .55/.03", pw(22,.55,88,.03))
print("Holm .025:", pw(22,20/37,88,5/73,.025), pw(22,.55,88,.03,.025))
