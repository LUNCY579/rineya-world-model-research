import json,glob
R=r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_TEST_B2"
for s in ['hard_beam','mirror_beam','planbench_beam']:
    tot=0; totu=0
    for d in sorted(glob.glob(f"{R}/{s}/shard_*")):
        keys=[]
        for l in open(d+"/fcache.jsonl",encoding='utf-8'):
            if not l.strip(): continue
            r=json.loads(l); keys.append(json.dumps([r['s'],r['a']],sort_keys=True))
        tot+=len(keys); totu+=len(set(keys))
        print(d.split('/')[-1] if '/' in d else d, len(keys), len(set(keys)))
    print(s,'lines',tot,'unique',totu,'per inst',totu/ (110 if s!='planbench_beam' else 602))
