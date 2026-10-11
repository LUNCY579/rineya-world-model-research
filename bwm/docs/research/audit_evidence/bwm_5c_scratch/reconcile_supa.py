import json, math, collections
S='C:/Users/clclc/AppData/Local/Temp/claude/bwm_5c_scratch/'
c1=[json.loads(l) for l in open(S+'my_prompt_grades.jsonl')]
cats=['unit_conv','pct_ratio','kv_recall','unknown_twin','format_follow']
def items(arm,key):
    d={}
    for r in c1:
        pid=r['pid']; c=r['category']
        base=pid.rsplit('/',1)[0] if c=='unknown_twin' else pid
        d.setdefault((c,base),True)
        d[(c,base)] = d[(c,base)] and r[arm][key]
    return d
def mcn(l,g):
    n=l+g
    if n==0: return 1.0
    k=min(l,g); return min(1.0, 2*sum(math.comb(n,i) for i in range(k+1))/2**n)
def holm(ps):
    o=sorted(range(len(ps)),key=lambda i:ps[i]); m=len(ps); out=[0]*m; run=0
    for r,i in enumerate(o):
        run=max(run,min(1,(m-r)*ps[i])); out[i]=run
    return out
for ref,arm in [('supa','x_bwf'),('supa','x_bwpa'),('off','x_bwf'),('off','x_bwpa')]:
    for key in ('P','L'):
        R=items(ref,key); A=items(arm,key); rows=[]
        for c in cats:
            ks=[k for k in R if k[0]==c]
            l=sum(R[k] and not A[k] for k in ks); g=sum(A[k] and not R[k] for k in ks)
            rows.append((c,sum(R[k] for k in ks),sum(A[k] for k in ks),l,g,g-l,mcn(l,g)))
        ph=holm([r[-1] for r in rows])
        print(arm,'vs',ref,key,'pooled',sum(r[5] for r in rows),
              [(r[0],r[1],r[2],r[3],r[4],r[5],round(r[6],4),round(h,4),'DET' if r[3]>r[4] and h<0.05 else '') for r,h in zip(rows,ph)])
