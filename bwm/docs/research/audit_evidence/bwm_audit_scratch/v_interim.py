import json, collections
from scipy.stats import binomtest, beta
N=r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_TEST_B2"
def load(p): return [json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
def succ(r): return bool(r['val']) if 'val' in r else bool(r['success'])
def sh(s,shards):
    ids=[]; out={}
    for i in shards:
        for r in load(f"{N}/{s}_beam/shard_{i}of8/plans.jsonl"): out[r['id']]=succ(r)
    return out
D={s:{r['id']:succ(r) for r in load(f"{N}/{s}_direct/plans.jsonl")} for s in ['hard','mirror']}
h=sh('hard',[0,1,2]); pl=sum(1 for i in h if h[i] and not D['hard'][i]); mi=sum(1 for i in h if not h[i] and D['hard'][i])
print('P-07',sum(h.values()),len(h),'direct',sum(D['hard'][i] for i in h),'+',pl,'-',mi,binomtest(pl,pl+mi).pvalue, beta.ppf(.025,25,18),beta.ppf(.975,26,17))
m=sh('mirror',range(5)); pl=sum(1 for i in m if m[i] and not D['mirror'][i]); mi=sum(1 for i in m if not m[i] and D['mirror'][i])
h5=sh('hard',range(5))
print('P-09',sum(m.values()),len(m),'direct',sum(D['mirror'][i] for i in m),'+',pl,'-',mi,binomtest(pl,pl+mi).pvalue,'hard',sum(h5.values()),len(h5),'gap',100*(sum(h5.values())/70-sum(m.values())/70))
hb={r['id']:r for r in load(f"{N}/hard_beam/plans.jsonl")}
st=collections.defaultdict(lambda:[0,0,0])
for i,r in hb.items():
    s=st[r['n_blocks']]; s[0]+=succ(r); s[1]+=D['hard'][i]; s[2]+=1
print('P-08 strata',dict(sorted(st.items())))
fails=[r for r in hb.values() if not succ(r)]; print('fails',len(fails),'verified among fails',sum(1 for r in fails if r['verified']))
