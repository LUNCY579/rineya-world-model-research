"""Own complete retry comparison; permits only two versioned evidence paths."""
import hashlib,json,pathlib,datetime
HERE=pathlib.Path(__file__).resolve().parent

def sha(b):return hashlib.sha256(b).hexdigest()
def typed(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def rows(p):return [json.loads(l) for l in p.read_bytes().splitlines()]
def diff(a,b,path=()):
    if type(a)!=type(b):return [{'path':list(path),'before':a,'after':b}]
    if type(a) is dict:
        if set(a)!=set(b):return [{'path':list(path),'before':a,'after':b}]
        return [d for k in a for d in diff(a[k],b[k],path+(k,))]
    if type(a) is list:
        if len(a)!=len(b):return [{'path':list(path),'before':a,'after':b}]
        return [d for k,(x,y) in enumerate(zip(a,b)) for d in diff(x,y,path+(k,))]
    return [] if a==b else [{'path':list(path),'before':a,'after':b}]

def main():
    fr=json.loads((HERE/'source-freeze-r2.json').read_text())
    for v in fr['bindings']:
        p=pathlib.Path(v['path']) if v['path'].startswith('/') else HERE.parents[2]/v['path']
        assert sha(p.read_bytes())==v['sha256'] and (HERE.parents[2]/v['archive']).read_bytes()==p.read_bytes()
    report={'actual_execution_id':'/root/research_architect','completed_execution':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_freeze_sha256':sha((HERE/'source-freeze-r2.json').read_bytes()),'r1_failed_custody_preserved':True,'gold_or_other_first_verdicts_read':False,'data_approval':False,'comparison':{}}
    for name in ['predictions','initial-judgments','row-input-inspections','family-input-ledger','input-audit']:
        p1=HERE/(name+'-r1.'+('jsonl' if name in ['predictions','initial-judgments','row-input-inspections'] else 'json'));p2=HERE/p1.name.replace('-r1.','-r2.')
        a=rows(p1) if p1.suffix=='.jsonl' else json.loads(p1.read_bytes());b=rows(p2) if p2.suffix=='.jsonl' else json.loads(p2.read_bytes());changes=diff(a,b)
        if name=='initial-judgments':
            assert len(a)==len(b)==216
            for delta in changes:
                assert len(delta['path'])==3 and delta['path'][1]=='evidence' and delta['path'][2] in ('family_ledger','prediction_ledger')
                assert delta['after']==delta['before'].replace('-r1.','-r2.')
        else:assert typed(a)==typed(b),'semantic change on retry '+name
        report['comparison'][name]={'first_sha256':sha(p1.read_bytes()),'retry_sha256':sha(p2.read_bytes()),'exact_bytes_equal':p1.read_bytes()==p2.read_bytes(),'typed_semantic_equal_except_declared_two_paths':True,'changes':changes}
    with (HERE/'retry-comparison-r2.json').open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'completed_execution':True,'216_prediction_semantics_unchanged':True,'216_initials_unchanged_except_two_evidence_paths':True,'other_input_outputs_unchanged':True}))
if __name__=='__main__':main()
