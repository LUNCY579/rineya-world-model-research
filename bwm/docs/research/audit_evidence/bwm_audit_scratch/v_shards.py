import json, glob, sys
from pathlib import Path
N=Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0,str(N)); sys.path.insert(0,str(N/"scripts"))
from bwmn import blocks as B
# replicate load_set id order without importing blocks_eval (which may import torch)
def ids(name):
    if name=='hard':
        return [f"hard/{p.stem}" for p in sorted((B.BENCH/"blocksworld_hard"/"generated").glob("*.pddl"),key=lambda p:int(p.stem.split('-')[1]))]
    if name=='mirror':
        return [f"mirror/{i}" for i in range(len(json.loads((N/"runs/BW_val_large/instances_mirror_hard.json").read_text())))]
    if name=='planbench':
        o=[]
        for sub in ("generated_basic_3","generated_basic"):
            o+= [f"{sub}/{p.stem}" for p in sorted((B.BENCH/"blocksworld"/sub).glob("*.pddl"),key=lambda p:int(p.stem.split('-')[1]))]
        return o
fb=json.load(open(N/"runs/FREEZE_B2_beam.json"))['args']; fd=json.load(open(N/"runs/FREEZE_B2_direct.json"))['args']
keys=("proposer","pa","mode","fwd","fwd_adapter","fwd_fmt","K","eval_seed","planner","W","E","max_depth","step_cap","R","M","P")
for s in ['hard','mirror','planbench']:
    allids=ids(s)
    shards=sorted(glob.glob(str(N/f"runs/BW_TEST_B2/{s}_beam/shard_*")), key=lambda x:int(Path(x).name.split('_')[1].split('of')[0]))
    k=len(shards); concat=[]; ok=True
    for d in shards:
        i=int(Path(d).name.split('_')[1].split('of')[0])
        sj=json.load(open(d+"/summary.json")); a=sj['args']
        bad={x:(a.get(x),fb.get(x)) for x in keys if a.get(x)!=fb.get(x)}
        lines=open(d+"/plans.jsonl",encoding='utf-8').read().splitlines()
        rid=[json.loads(l)['id'] for l in lines if l.strip()]
        exp=[x for j,x in enumerate(allids) if j%k==i]
        if bad or set(rid)!=set(exp) or len(rid)!=len(set(rid)) or not a.get('val') or a.get('freeze')!='runs/FREEZE_B2_beam.json' or a.get('n'):
            ok=False; print('PROBLEM',d,bad,len(rid),len(exp),a.get('val'),a.get('freeze'),a.get('n'),a.get('shard'))
        concat+= [l for l in lines if l.strip()]
    merged=[l for l in open(N/f"runs/BW_TEST_B2/{s}_beam/plans.jsonl",encoding='utf-8').read().splitlines() if l.strip()]
    print(s,'shards',k,'all ok',ok,'concat==merged (as lines, order)',concat==merged,'as set',sorted(concat)==sorted(merged),'ids cover',sorted(json.loads(l)['id'] for l in merged)==sorted(allids))
    dj=json.load(open(N/f"runs/BW_TEST_B2/{s}_direct/summary.json"))
    a=dj.get('args',{}); bad={x:(a.get(x),fd.get(x)) for x in keys if a.get(x)!=fd.get(x)}
    print('  direct args bad',bad,'val',a.get('val'),'freeze',a.get('freeze'),'n',a.get('n'),'shard',a.get('shard'))
    dids=[json.loads(l)['id'] for l in open(N/f"runs/BW_TEST_B2/{s}_direct/plans.jsonl",encoding='utf-8') if l.strip()]
    print('  direct ids cover',sorted(dids)==sorted(allids), len(dids))
