#!/opt/anaconda3/bin/python3
"""Separate own post-first mismatch diagnosis, no other reviewer/checker reads."""
import pathlib,json,hashlib,importlib.util,sys,datetime
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert';NAME='belief-gap-v4.canonical-histogram-inspect';ROLE='/root/expert_reviewer'
BIND={'reviews/expert/belief-gap-v4-canonical-review-attempt02.py':'5efe9f4a85e168dbaf03c202902f1f0e100d535432086b9900132fbad9fe80a8',
 'reviews/expert/belief_gap_v2_metadata_support.py':'54324adf2332c7d85651f6217dc51d1ac6f6610ddd68d1819f43416f9f1d9e84',
 'data/RBM-001/candidate_renderings/belief-gap-v4.jsonl':'38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c',
 'data/RBM-001/manifests/belief-gap-v4.family_controls.json':'6b91a54b60dcf5a702889452a29ec5348acb4eb8c0ef62b1a03408f04fad2631',
 'reviews/expert/belief-gap-v4.canonical-first-judgments.jsonl':'b1cd43c4c0e54acce5964e2010bb8ae431dfd53e6b7971d842770a7579c6b6fc'}
def hp(p):
 h=hashlib.sha256();n=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b);n+=len(b)
 return {'sha256':h.hexdigest(),'length_bytes':n}
def binding():
 r={p:hp(ROOT/p) for p in BIND};assert all(r[p]['sha256']==h for p,h in BIND.items());return r
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def imp(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def diff(a,b,path='$'):
 if type(a)!=type(b):return [{'path':path,'supplied_type':type(a).__name__,'own_type':type(b).__name__}]
 if type(a) is dict:
  out=[]
  for k in set(a)|set(b):
   if k not in a or k not in b:out.append({'path':path+'.'+k,'supplied_key_exists':k in a,'own_key_exists':k in b})
   else:out+=diff(a[k],b[k],path+'.'+k)
  return out
 if type(a) is list:
  if len(a)!=len(b):return [{'path':path,'supplied_length':len(a),'own_length':len(b)}]
  return [x for i,(u,v) in enumerate(zip(a,b)) for x in diff(u,v,path+'['+str(i)+']')]
 return [] if a==b else [{'path':path,'supplied':a,'own':b}]
source=pathlib.Path(__file__).resolve();pre=OUT/(NAME+'.preexecution.json')
if sys.argv[1]=='freeze':
 with (OUT/(NAME+'.source.preexecution.py')).open('xb') as f:f.write(source.read_bytes())
 write(pre,{'actual_author':ROLE,'reviewer_execution_id':ROLE,'source':hp(source),'bindings':binding(),'intent':'identify first own diagnostic mismatch only; first judgments immutable'})
 print(json.dumps(hp(pre)))
else:
 p=json.loads(pre.read_text());before=binding();assert p['source']==hp(source)==hp(OUT/(NAME+'.source.preexecution.py')) and before==p['bindings']
 audit=imp('own_frozen_v4_comparison_diagnostic',ROOT/'reviews/expert/belief-gap-v4-canonical-review-attempt02.py');own=imp('own_frozen_typed_solver',ROOT/'reviews/expert/belief_gap_v2_metadata_support.py')
 manifest=json.loads((ROOT/'data/RBM-001/manifests/belief-gap-v4.family_controls.json').read_text());g=manifest['actual_conditional_query_histograms'][0]
 group=[]
 for line in (ROOT/'data/RBM-001/candidate_renderings/belief-gap-v4.jsonl').open():
  r=json.loads(line)
  if all(r[k]==g[k] for k in ['source_family_key','controlled_context','auxiliary_scalar_variant','renderer']):group.append({'row':r,'task':r['semantic_task'],'facts':own.row_facts(r['semantic_task'])})
 actual=audit.conditional_histogram(group);differences=diff(g['histograms'],actual)
 ax=next(iter(actual));cell=next(iter(actual[ax]['actual_full_typed_outcome_provenance_histograms']))
 supplied_signatures=[json.loads(k) for k in g['histograms'][ax]['actual_full_typed_outcome_provenance_histograms'][cell]]
 own_signatures=[json.loads(k) for k in actual[ax]['actual_full_typed_outcome_provenance_histograms'][cell]]
 result={'actual_author':ROLE,'reviewer_execution_id':ROLE,'source':hp(source),'bindings_before':before,'bindings_after':binding(),'first_judgments_changed':False,
 'group':{k:v for k,v in g.items() if k!='histograms'},'actual_rows':len(group),'typed_signature_examples':{'supplied':supplied_signatures,'own':own_signatures},
 'differences_count':len(differences),'first_differences':differences[:6],'dictionary_signature_key_text_may_have_distinct_semantic_spelling':True,
 'own_histogram_sha256':audit.dig(actual),'supplied_histogram_sha256':audit.dig(g['histograms']),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':0}
 assert before==result['bindings_after'] and p['source']==hp(source)
 target=OUT/(NAME+'.execution.json');write(target,result);print(json.dumps({'execution':hp(target),'differences':differences[:2],'examples':result['typed_signature_examples']}))
