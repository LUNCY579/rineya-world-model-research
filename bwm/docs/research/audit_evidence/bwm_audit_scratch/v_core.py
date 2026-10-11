import json, collections
from scipy.stats import binomtest, beta
N=r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
C=json.load(open(N+"/release/Rineya-Titan-Sparse-BWM-Blocksworld-0.1.0-devfreeze/CORE.json"))['measured']
def load(p): return [json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
def succ(r): return bool(r['val']) if 'val' in r else bool(r['success'])
def cp(k,n):
    lo=0.0 if k==0 else beta.ppf(.025,k,n-k+1); hi=1.0 if k==n else beta.ppf(.975,k+1,n-k); return [round(lo,4),round(hi,4)]
bad=0
for s in ['hard','mirror','planbench']:
    P={}; NB={r['id']:r['n_blocks'] for r in load(f"{N}/runs/BW_TEST_B2/{s}_beam/plans.jsonl")}
    for arm,key in [('beam','S_beam'),('direct','S_direct')]:
        rows=load(f"{N}/runs/BW_TEST_B2/{s}_{arm}/plans.jsonl"); P[arm]={r['id']:succ(r) for r in rows}
        summ=json.load(open(f"{N}/runs/BW_TEST_B2/{s}_{arm}/summary.json"))['summary']
        k=sum(P[arm].values()); n=len(rows)
        by=collections.defaultdict(list)
        for r in rows: by[str(NB[r['id']])].append(succ(r))
        sbb={b:round(sum(v)/len(v),4) for b,v in by.items()}
        c=C[s][key]
        exp={'solved':k,'n':n,'rate':round(k/n,4),'clopper_pearson_95':cp(k,n),'val_disagreements':summ['val_disagreements'],'f_queries_per_instance':summ['f_queries_per_instance'],'proposer_tokens_per_instance':summ['proposer_tokens_per_instance'],'success_by_blocks':sbb}
        for x in exp:
            if exp[x]!=c.get(x): bad+=1; print('MISMATCH',s,key,x,exp[x],c.get(x))
        # summary vs plans
        if summ.get('success_by_blocks') and summ['success_by_blocks']!=sbb: print('summ sbb mismatch',s,arm,summ['success_by_blocks'],sbb)
        if abs(summ['success']-k/n)>1e-12 or summ['n']!=n: print('summ mismatch',s,arm)
        vs=sum(1 for r in rows if r.get('val'))/n
        if abs(summ['val_success']-vs)>1e-12: print('val_success mismatch',s,arm)
        ver=[r for r in rows if r.get('verified')]
        if 'verification_precision' in summ: print(s,arm,'verified n',len(ver),'prec stored',summ['verification_precision'])
        else: print(s,arm,'verified',len(ver),'valid among verified',sum(succ(r) for r in ver))
    pl=sum(1 for i in P['beam'] if P['beam'][i] and not P['direct'][i]); mi=sum(1 for i in P['beam'] if not P['beam'][i] and P['direct'][i])
    pp=C[s]['paired_beam_vs_direct']
    if (pl,mi,binomtest(pl,pl+mi).pvalue,len(P['beam']))!=(pp['beam_only'],pp['direct_only'],pp['exact_mcnemar_p'],pp['n']): bad+=1; print('PAIR MISMATCH',s)
print('mismatches',bad)
