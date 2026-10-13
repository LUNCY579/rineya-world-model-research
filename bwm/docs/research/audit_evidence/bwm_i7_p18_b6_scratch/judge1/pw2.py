import math, numpy as np
from pw import pv, power
def tri(n, p10, p01):
    M = np.zeros((n+1, n+1)); p00 = 1-p10-p01
    lf = [math.lgamma(k+1) for k in range(n+1)]
    for b in range(n+1):
        for c in range(n+1-b):
            M[b, c] = math.exp(lf[n]-lf[b]-lf[c]-lf[n-b-c] + b*math.log(p10) + c*math.log(p01) + (n-b-c)*math.log(p00))
    return M
def two_stage(p10, p01, n1=50, n2=50, fut=1):
    A = tri(n1, p10, p01); Bm = tri(n2, p10, p01)
    # distribution of final (b,c) given continuation
    N = n1+n2
    F = np.zeros((N+1, N+1)); stop = 0.0
    for b1 in range(n1+1):
        for c1 in range(n1+1-b1):
            pa = A[b1, c1]
            if pa < 1e-14: continue
            if b1 - c1 <= fut: stop += pa; continue
            F[b1:b1+n2+1, c1:c1+n2+1] += pa * Bm
    met = sum(F[b, c] for b in range(N+1) for c in range(N+1-b) if F[b, c] > 1e-15 and b > c and pv(b, b+c) < 0.05)
    return round(met, 3), round(stop, 3)
for k, (a, b) in {"null .05": (.05,.05), "null .12": (.12,.12), "+10 .15/.05": (.15,.05), "+12.5": (.175,.05), "+15 .20/.05": (.20,.05), "H2like .23/.054": (.23,.054), "+15 hi .23/.08": (.23,.08)}.items():
    print("rigor 2-stage", k, two_stage(a, b))
for n in (64, 80, 100):
    for k, (a, b) in {"H2 analog .226/.065": (.226,.065), "+12 .17/.05": (.17,.05), "+12 noisy .20/.08": (.20,.08), "+8 .14/.06": (.14,.06), "null .125": (.125,.125)}.items():
        print("effect P2 n", n, k, power(n, a, b, met_pp=10))
