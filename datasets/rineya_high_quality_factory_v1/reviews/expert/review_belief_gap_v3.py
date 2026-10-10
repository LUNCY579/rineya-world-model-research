#!/opt/anaconda3/bin/python3
"""Actual guided-corrective expert review v3; independent input-only freeze first.
Fresh bit-mask solver/calibration. Own v2 public-derived property/probe routines
are explicitly reused, never producer/reference/oracle/checker source.
"""
import argparse, collections, copy, datetime, gzip, hashlib, itertools, json, pathlib, runpy, sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'reviews/expert'; ROLE='/root/expert_reviewer'
sha=lambda b:hashlib.sha256(b).hexdigest()
SUP=OUT/'belief_gap_v2_metadata_support.py'
SUPSHA='54324adf2332c7d85651f6217dc51d1ac6f6610ddd68d1819f43416f9f1d9e84'
assert sha(SUP.read_bytes())==SUPSHA
OWN=runpy.run_path(str(SUP)); canon,equal=OWN['canon'],OWN['equal']
PUBLIC={
'schemas/EXTENDED_SEMANTICS.md':'2a67e05e10f99d1fd242ee1691d5b93755093fa701c48040c2d90f47422e7e23',
'schemas/BELIEF_CONTROL_PROPERTIES.md':'200631973256bbe5dcd7b99f3d52cb95f89c442acd29ea4de1fe82dd8b5f3034',
'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md':'46e3120dfd1ee931eec9d1e5e0e75f0b2aedb1d85764fd755afe9f7d36cf121f',
'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md':'3a223952d9664577d7b637ddfb3e18d329863a3d5371a6c47cad233a1f50ff11',
'research/curricula/RBM-EVIDENCE-GAP-REVISION-V3.md':'7e9c9f9160425ab7e455f57d72385e54f399835dfa351dcf75f6fc2842b1dc33',
'schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json':'611656b37d7fe917d64eeb191d901d272a32d8d3c72c9f8865476c6889a3e51a'}
VIEW_SHA='50b3f47bce29e8a639f298ece104744be1583d04c04d3041d940219d6998b694'
CAND_SHA='7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c'
KEYS={'sample_id','record_hash','semantic_parent_id','semantic_family_id','track_id','split','semantic_task','rendered_input'}
NEW={'auxiliary_scalar_transposition_invariance','same_observations_auxiliary_table_counterfactual','context_auxiliary_scalar_variant_crossing'}
GUIDANCE='Corrective public v1/v2/v3 designs and frozen public shortcut hypotheses; own prior expert v1/v2 knowledge and prospective gap guidance; shared underlying model/architect role overlap. Not fully blind. No producer/reference/oracle/checker/private verdict reads.'
def bindings():
 out={}
 for p,h in PUBLIC.items():
  b=(ROOT/p).read_bytes();assert sha(b)==h;out[p]={'sha256':h,'length':len(b)}
 return out
def save(path,obj):
 assert not path.exists(),str(path)
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');return sha(path.read_bytes())
def compress(path):
 raw=path.read_bytes();b=gzip.compress(raw,compresslevel=9,mtime=0);p=pathlib.Path(str(path)+'.gz');assert not p.exists();p.write_bytes(b)
 assert gzip.decompress(p.read_bytes())==raw and len(b)<=25*1024*1024
 return {'raw_path':str(path),'raw_sha256':sha(raw),'raw_length':len(raw),'compressed_path':str(p),'compressed_sha256':sha(b),'compressed_length':len(b),'gzip_mtime':0,'decoded_sha256':sha(gzip.decompress(b)),'roundtrip_byte_equal':True}
def encode(v):
 assert v is None or type(v) in (bool,int,str)
 return json.dumps(v,ensure_ascii=False,separators=(',',':'))
def solve(task):
 """New bit-set posting intersection; output bins independently enumerated."""
 OWN['validate_task'](task);ws=task['worlds'];posting=collections.defaultdict(int)
 for i,w in enumerate(ws):
  for k,v in w['facts'].items():posting[(k,encode(v))]|=1<<i
 active=(1<<len(ws))-1;prefix=[sorted(w['world_id'] for w in ws)]
 for o in task['observations']:
  active &= posting[(o['fact'],encode(o['equals']))]
  prefix.append(sorted(w['world_id'] for i,w in enumerate(ws) if active>>i&1))
 qs=[]
 for q in task['queries']:
  bins=collections.Counter(encode(w['facts'][q['fact']]) for i,w in enumerate(ws) if active>>i&1)
  counts=[{'value':json.loads(k),'count':n} for k,n in sorted(bins.items())]
  certain=len(counts)==1
  qs.append({'query_id':q['query_id'],'counts':counts,'distinct_values':len(counts),'entailed':certain,'guaranteed':counts[0]['value'] if certain else None,'can_reduce_uncertainty':len(counts)>1})
 return {'surviving_worlds':prefix[-1],'belief_sizes':[len(p) for p in prefix[1:]],'queries':qs},prefix
def f(task):
 target,p=solve(task);facts=OWN['row_facts'](task)
 assert equal(target,facts['expected']) and equal(p,facts['prefix_survivors'])
 return facts
def transpose(task,fact,b,g):
 t=copy.deepcopy(task)
 def swap(v):return g if type(v) is str and v==b else b if type(v) is str and v==g else v
 for w in t['worlds']:w['facts'][fact]=swap(w['facts'][fact])
 for o in t['observations']:
  if o['fact']==fact:o['equals']=swap(o['equals'])
 return t
def diagnostic(task):
 vals=[o['equals'] for o in task['observations']]
 marker=any(type(v) is str and v in {'stone','unknown'} for v in vals)
 linen=any(type(v) is str and v=='linen' for v in vals)
 rules={'v1_original_scalar_marker_empty':marker,'prospective_marker_and_auxiliary_token_empty':marker and linen}
 cores={}
 for name,p in rules.items():
  N,M=len(task['worlds']),len(vals)
  cores[name]={'surviving_worlds':[],'belief_sizes':[1 if N==8 else 2]+[0]*(M-1),'queries':[{'query_id':q['query_id'],'counts':[],'distinct_values':0,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':False} for q in task['queries']]} if p and N in (8,12) and M else None
 return {'empty_predictions':rules,'abstaining_full_output_predictions':cores,'final_gold_or_final_support_used_as_predictor':False}
def prepare():
 pubs=bindings()
 ws=[{'world_id':str(i),'facts':{'x':x,'y':y,'q':q,'z':z}} for i,(x,y,q,z) in enumerate([('stone','silk',False,None),('stone','silk',0,None),('reed','linen',None,True),('reed','silk',True,1),('fern','linen','opal','silk')])]
 qs=[{'query_id':k,'fact':v} for k,v in [('c','q'),('a','z'),('b','x')]]
 obs=[[],[{'fact':'q','equals':False}],[{'fact':'x','equals':'stone'},{'fact':'y','equals':'linen'}],[{'fact':'x','equals':'stone'},{'fact':'y','equals':'silk'}]]
 obs += [x+[copy.deepcopy(x[0])] for x in obs[2:4]]
 fixtures=[]
 for o in obs:
  t={'kind':'belief','worlds':copy.deepcopy(ws),'observations':o,'queries':copy.deepcopy(qs)};tf=transpose(t,'y','linen','silk')
  facts=f(t);assert equal(solve(t),solve(tf)) and equal(t,transpose(tf,'y','linen','silk'))
  fixtures.append({'task':t,'facts':facts,'transposed_task':tf,'transposition_full_output_prefix_invariance':True,'probes':OWN['metamorphics'](t),'frozen_rules':diagnostic(t)})
 assert fixtures[1]['facts']['expected']['surviving_worlds']==['0']
 assert fixtures[2]['facts']['expected']['belief_sizes']==[2,0]
 assert fixtures[3]['facts']['expected']['queries'][0]['counts']==[{'value':0,'count':1},{'value':False,'count':1}]
 assert equal(fixtures[2]['task']['observations'],fixtures[3]['transposed_task']['observations'])
 assert not equal(False,0) and not equal(True,1)
 report={'report_type':'V3_OWN_PREPARATION_CALIBRATION','actual_author':ROLE,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'public_inputs':pubs,'source_sha256':sha(pathlib.Path(__file__).read_bytes()),'own_property_support_sha256':SUPSHA,'guidance':GUIDANCE,'new_solver':'bitset posting intersections using exact typed scalar JSON, independently compared with own previous conjunction/sequential property solver','future_candidate_or_gold_reads':0,'fixtures':fixtures,'fixture_count':len(fixtures),'toy_invariance_probes':sum(len(x['probes']['invariance_probes'])+len(x['probes']['typed_equals_counterfactual_probes']) for x in fixtures),'dataset_verdicts':0}
 p=OUT/'belief-gap-v3.preparation.json';h=save(p,report)
 save(OUT/'belief-gap-v3.preparation_execution.json',{'actual_author':ROLE,'argv':sys.argv,'python_executable':sys.executable,'exit_code':0,'source_sha256':report['source_sha256'],'preparation_sha256':h,'public_inputs':pubs,'candidate_reads':0,'gold_reads':0})
 print(canon({'preparation_sha256':h,'source_sha256':report['source_sha256'],'fixtures':len(fixtures)}))
def predict():
 prep=json.loads((OUT/'belief-gap-v3.preparation.json').read_text());assert prep['source_sha256']==sha(pathlib.Path(__file__).read_bytes())
 p=ROOT/'reviews/input_only/belief-gap-v3.jsonl';b=p.read_bytes();assert sha(b)==VIEW_SHA and len(b)==17534736
 rows=[]
 for line in b.splitlines():
  r=json.loads(line);assert set(r)==KEYS;OWN['refuse_targets'](r)
  t=OWN['parse_prompt'](r);assert equal(t,r['semantic_task']);facts=f(t)
  rows.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'visible_identity':{k:r[k] for k in KEYS-{'semantic_task','rendered_input'}},'task':t,'task_sha256':sha(canon(t).encode()),'facts':facts,'probes':OWN['metamorphics'](t),'input_only_frozen_diagnostics':diagnostic(t)})
 assert len(rows)==len({r['sample_id'] for r in rows})==3456
 rep={'report_type':'V3_INPUT_ONLY_TYPED_PREDICTION_FREEZE','actual_author':ROLE,'reviewer_execution_id':ROLE,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'view_sha256':VIEW_SHA,'original_candidate_sha256':CAND_SHA,'public_inputs':bindings(),'source_sha256':sha(pathlib.Path(__file__).read_bytes()),'own_property_support_sha256':SUPSHA,'guidance':GUIDANCE,'metadata_and_gold_not_seen':True,'no_dataset_verdict':True,'records':rows}
 out=OUT/'belief-gap-v3.input_only_predictions.json';h=save(out,rep);comp=compress(out)
 rec={'report_type':'PREDICTION_FREEZE','actual_author':ROLE,'python_executable':sys.executable,'actual_argv':sys.argv,'exit_code':0,'created_utc':rep['created_utc'],'source_sha256':rep['source_sha256'],'own_property_support_sha256':SUPSHA,'view_sha256':VIEW_SHA,'original_candidate_sha256':CAND_SHA,'prediction_sha256':h,'record_count':len(rows),'metadata_and_gold_not_seen':True,'no_dataset_verdict':True,'compression':comp}
 save(pathlib.Path(str(out)+'.freeze_receipt.json'),rec);print(canon(rec))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','predict']);args=a.parse_args()
 prepare() if args.mode=='prepare' else predict()
