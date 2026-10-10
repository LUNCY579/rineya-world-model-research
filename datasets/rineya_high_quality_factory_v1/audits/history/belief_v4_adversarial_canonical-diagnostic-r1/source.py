#!/usr/bin/env python3
"""Own narrow histogram failure localization, no private/other reviewer access."""
import pathlib,hashlib,json,importlib.util,collections
ROOT=pathlib.Path(__file__).resolve().parents[1]
PRE=ROOT/'audits/execution/belief-gap-v4-canonical-adversarial-diagnostic-r1.preexec.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def firstdiff(a,b,path=''):
 if type(a) is not type(b):return {'path':path,'own_type':type(a).__name__,'supplied_type':type(b).__name__,'own':a,'supplied':b}
 if isinstance(a,dict):
  if set(a)!=set(b):return {'path':path,'only_own':sorted(set(a)-set(b)),'only_supplied':sorted(set(b)-set(a))}
  for k in sorted(a):
   d=firstdiff(a[k],b[k],path+'/'+str(k))
   if d:return d
 elif isinstance(a,list):
  if len(a)!=len(b):return {'path':path,'own_len':len(a),'supplied_len':len(b)}
  for i,(x,y) in enumerate(zip(a,b)):
   d=firstdiff(x,y,path+'/'+str(i))
   if d:return d
 elif a!=b:return {'path':path,'own':a,'supplied':b}
 return None
pre=json.loads(PRE.read_text())
assert {p:sha(ROOT/p) for p in pre['bindings']}==pre['bindings']
spec=importlib.util.spec_from_file_location('own_frozen_canonical_r1',ROOT/'audits/belief_v4_adversarial_canonical_r1.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
data=C.rows(ROOT/C.CAND);pred={r['sample_id']:r for r in C.rows(ROOT/C.PRED)};manifest=C.loadjson((ROOT/C.MAN).read_bytes());issues=[]
for e in manifest['actual_conditional_query_histograms']:
 key=(e['source_family_key'],e['controlled_context'],e['auxiliary_scalar_variant'],e['renderer']);rs=[r for r in data if (r['source_family_key'],r['controlled_context'],r['auxiliary_scalar_variant'],r['renderer'])==key];computed={}
 for names in [('id',),('position',),('fact',),('id','position'),('id','fact'),('position','fact'),('id','position','fact')]:
  roles=collections.defaultdict(collections.Counter);outs=collections.defaultdict(collections.Counter)
  for r in rs:
   for assignment in r['query_role_assignments']:
    i=assignment['position'];q=r['semantic_task']['queries'][i];values={'id':q['query_id'],'position':i,'fact':q['fact']};bucket=C.canon([values[n] for n in names]);roles[bucket][assignment['role']]+=1;outs[bucket][C.canon(C.payload(q,pred[r['sample_id']]['expected_target']['queries'][i],pred[r['sample_id']]['prefix_supports']))]+=1
  computed[','.join(names)]={'actual_role_histograms':{k:dict(v) for k,v in roles.items()},'actual_full_typed_outcome_provenance_histograms':{k:dict(v) for k,v in outs.items()}}
 if C.canon(computed)!=C.canon(e['histograms']):
  diff=firstdiff(computed,e['histograms']);issues.append({'group':key,'first_difference':diff,'sample_id':rs[0]['sample_id'],'record_hash':rs[0]['record_hash']})
report={'actual_actor':'/root/adversarial_auditor','source_sha256':sha(pathlib.Path(__file__)),'all128groups_examined':len(manifest['actual_conditional_query_histograms']),'mismatching_groups':len(issues),'differences':issues,'before':pre['bindings'],'after':{p:sha(ROOT/p) for p in pre['bindings']},'other_review_private_access':False}
assert report['before']==report['after']
C.freeze(ROOT/'audits/execution/belief-gap-v4-canonical-adversarial-diagnostic-r1.json',C.jsonbytes(report))
print(json.dumps({'mismatching_groups':len(issues),'first_two':issues[:2],'report_sha256':sha(ROOT/'audits/execution/belief-gap-v4-canonical-adversarial-diagnostic-r1.json')},ensure_ascii=False))
