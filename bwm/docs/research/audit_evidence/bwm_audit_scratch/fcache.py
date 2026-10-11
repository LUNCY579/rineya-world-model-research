import json, glob, os
os.chdir(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_TEST_B2")
tot = {}
for f in sorted(glob.glob('*/shard_*/fcache.jsonl')):
    rows = [json.loads(l) for l in open(f, encoding='utf-8')]
    keys = {(json.dumps(r['s']), json.dumps(r['a'])) for r in rows}
    sm = json.load(open(os.path.join(os.path.dirname(f), 'summary.json')))['summary']
    t = os.path.normpath(f).split(os.sep)[0]
    a = tot.setdefault(t, [0, 0, 0]); a[0] += len(keys); a[1] += sm['f_model_queries']; a[2] += sm['n']
    print(f, 'lines', len(rows), 'unique', len(keys), 'summary_fq', sm['f_model_queries'])
for t, (u, q, n) in tot.items():
    print(t, 'unique_cached', u, 'reported', q, 'n', n, 'unique/inst', round(u / n, 1), 'reported/inst', round(q / n, 1))
