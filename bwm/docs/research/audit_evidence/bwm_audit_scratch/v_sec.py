import json, collections
from scipy.stats import binomtest
N=r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
def load(p): return [json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
def succ(r): return bool(r['val']) if 'val' in r else bool(r['success'])
info=json.load(open(N+"/bench/planbench/blocksworld_hard/info.json"))
L={("hard/"+k.replace('.pddl','')):v['length'] for k,v in info.items()}
print('info n',len(L),'min',min(L.values()),'max',max(L.values()),'mean',sum(L.values())/len(L))
B={r['id']:succ(r) for r in load(N+"/runs/BW_TEST_B2/hard_beam/plans.jsonl")}
D={r['id']:succ(r) for r in load(N+"/runs/BW_TEST_B2/hard_direct/plans.jsonl")}
print('ids match info', set(B)==set(L))
def b(l): return '20-24' if l<=24 else '25-29' if l<=29 else '30-34' if l<=34 else '35-40'
st=collections.defaultdict(lambda:[0,0,0])
for i in B:
    s=st[b(L[i])]; s[0]+=B[i]; s[1]+=D[i]; s[2]+=1
print(dict(sorted(st.items())))
led=json.load(open(N+"/bench/planbench/LEDGER.json"))
PB={r['id']:succ(r) for r in load(N+"/runs/BW_TEST_B2/planbench_beam/plans.jsonl")}
PD={r['id']:succ(r) for r in load(N+"/runs/BW_TEST_B2/planbench_direct/plans.jsonl")}
print('pb id sample',list(PB)[:3])
zs=set()
for k,v in led.items():
    if 'zero_shot' in k and 'per_instance' in v: zs|=set(v['per_instance'])
print('zero-shot union ids',len(zs))
def norm(i): return i.split('planbench/')[-1] if 'planbench/' in i else i
PBn={norm(i):v for i,v in PB.items()}; PDn={norm(i):v for i,v in PD.items()}
print('pb ids not in zs', sorted(set(PBn)-zs)[:10], 'zs not in pb', sorted(zs-set(PBn))[:5])
z=[i for i in PBn if i in zs]
print('zero-shot 600: beam',sum(PBn[i] for i in z),'direct',sum(PDn[i] for i in z),'n',len(z))
pl=sum(1 for i in z if PBn[i] and not PDn[i]); mi=sum(1 for i in z if not PBn[i] and PDn[i]); print('+',pl,'-',mi,binomtest(pl,pl+mi).pvalue)
for k in ['deepseek-r1-api|task_1_plan_generation_zero_shot','o1-preview_chat|task_1_plan_generation_zero_shot']:
    pi=led[k]['per_instance']; pl=sum(1 for i in pi if PBn[i] and not pi[i]); mi=sum(1 for i in pi if not PBn[i] and pi[i])
    print(k,led[k]['val_success'],len(pi),'+',pl,'-',mi,binomtest(pl,pl+mi).pvalue)
