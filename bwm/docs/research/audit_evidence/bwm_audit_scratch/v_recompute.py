import json, os, glob, collections
from scipy.stats import binomtest, fisher_exact, beta
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_TEST_B2"
def load(p):
    rows=[json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
    return rows
def succ(r): return bool(r['val']) if 'val' in r else bool(r['success'])
def cp(k,n,a=0.05):
    lo = 0.0 if k==0 else beta.ppf(a/2,k,n-k+1)
    hi = 1.0 if k==n else beta.ppf(1-a/2,k+1,n-k)
    return lo,hi
res={}
for s in ['hard','mirror','planbench']:
    B=load(f"{R}/{s}_beam/plans.jsonl"); D=load(f"{R}/{s}_direct/plans.jsonl")
    bi={r['id']:r for r in B}; di={r['id']:r for r in D}
    print(s,'rows',len(B),len(D),'uniq',len(bi),len(di),'same ids',set(bi)==set(di))
    kb=sum(succ(r) for r in B); kd=sum(succ(r) for r in D)
    plus=sum(1 for i in bi if succ(bi[i]) and not succ(di[i])); minus=sum(1 for i in bi if not succ(bi[i]) and succ(di[i]))
    p=binomtest(plus,plus+minus,0.5).pvalue if plus+minus else 1
    # disagreement success vs val
    dis_b=sum(1 for r in B if 'val' in r and bool(r['val'])!=bool(r['success']))
    dis_d=sum(1 for r in D if 'val' in r and bool(r['val'])!=bool(r['success']))
    novb=sum(1 for r in B if 'val' not in r); novd=sum(1 for r in D if 'val' not in r)
    print(f"  beam {kb}/{len(B)} CI {cp(kb,len(B))} direct {kd}/{len(D)} CI {cp(kd,len(D))} +{plus}/-{minus} p={p:.4g}  dis {dis_b},{dis_d} noval {novb},{novd}")
    tok=sum(r.get('gen_tokens',0) for r in B)/len(B); tokd=sum(r.get('gen_tokens',0) for r in D)/len(D)
    ver=sum(1 for r in B if r.get('verified'))/len(B)
    print(f"  tok/inst beam {tok:.1f} direct {tokd:.1f} verified_rate {ver:.4f}")
    res[s]=(kb,len(B))
kh,nh=res['hard']; km,nm=res['mirror']
print('gap pp', 100*(kh/nh-km/nm))
print('fisher', fisher_exact([[kh,nh-kh],[km,nm-km]]))
