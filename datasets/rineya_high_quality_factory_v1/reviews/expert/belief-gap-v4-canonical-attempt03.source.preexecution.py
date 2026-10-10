#!/opt/anaconda3/bin/python3
"""Expert-owned V4 canonical audit AFTER the completed input-only freeze.

Only public contracts, actual authorized canonical/manifest assertions and own
prior source/evidence are used.  No formal/producer/reference/private verdict
implementation is imported.  Sources and failures are immutable by attempt.
"""
import argparse, collections, copy, datetime, gzip, hashlib, importlib.util, itertools, json, pathlib, resource, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/expert';ROLE='/root/expert_reviewer'
PREFIX='belief-gap-v4-canonical-attempt03'
CAND='data/RBM-001/candidate_renderings/belief-gap-v4.jsonl';CSHA='38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c'
MAN='data/RBM-001/manifests/belief-gap-v4.family_controls.json';MSHA='6b91a54b60dcf5a702889452a29ec5348acb4eb8c0ef62b1a03408f04fad2631'
OLD='data/RBM-001/candidate_renderings/belief-gap-v3.jsonl'
INHERITED={'same_prefix_empty_nonempty_counterfactual','contradiction_equals_mixed_outcomes'}
NEW={'auxiliary_scalar_transposition_invariance','same_observations_auxiliary_table_counterfactual','context_auxiliary_scalar_variant_crossing'}
MINIMUM={'name','scope','source_family_key','auxiliary_scalar_variant','observation_length','control_id','pair_sample_ids','exact_candidate_sha256'}
CLAIM_FIELDS={'name','property','scope','source_family_key','evidence_manifest','control_id','pair_sample_ids','probe_id'}
COORDS=('source_family_key','auxiliary_scalar_variant','schedule_index','renderer')
PUBLIC={
 'schemas/EXTENDED_SEMANTICS.md':'2a67e05e10f99d1fd242ee1691d5b93755093fa701c48040c2d90f47422e7e23',
 'schemas/BELIEF_CONTROL_PROPERTIES.md':'200631973256bbe5dcd7b99f3d52cb95f89c442acd29ea4de1fe82dd8b5f3034',
 'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md':'46e3120dfd1ee931eec9d1e5e0e75f0b2aedb1d85764fd755afe9f7d36cf121f',
 'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md':'3a223952d9664577d7b637ddfb3e18d329863a3d5371a6c47cad233a1f50ff11',
 'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md':'40b552ecf47c11bfffb2669283640dda991d6dc6e7094ed3051fb9cad2489fce',
 'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md':'925ccae4e6fdc79fec2d0778d871e5d316257c3026b59c6e166556da71eb346c',
 'research/curricula/BELIEF-METADATA-V4-CLARIFICATION-EXECUTION.json':'2d100ce2ad27ded95275316991f3ec8ace9ef9daf2a40c50c3f06946c777ec0d',
 'research/curricula/BELIEF_METADATA_V4_ENCODING_ROOT_FREEZE.json':'2bbd9d577e0b7eb488143bd7f532f9ce3fd1c3c51be918d90470b29bc7b61625',
 'schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json':'611656b37d7fe917d64eeb191d901d272a32d8d3c72c9f8865476c6889a3e51a',
 'research/curricula/RBM-EVIDENCE-GAP.md':'004fb0ec0c1a1104ba052fbda4882f48523d7d34dd45b2b88e8c890713b4a120',
 'research/curricula/RBM-EVIDENCE-GAP-REVISION-V3.md':'7e9c9f9160425ab7e455f57d72385e54f399835dfa351dcf75f6fc2842b1dc33'}
BIND={**PUBLIC,CAND:CSHA,MAN:MSHA,OLD:'7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c',
 'reviews/expert/belief-gap-v4-canonical-review-attempt02.py':'5efe9f4a85e168dbaf03c202902f1f0e100d535432086b9900132fbad9fe80a8',
 'reviews/expert/belief-gap-v4-canonical-attempt02.execution.json':'6eb1480097ade83f0473f0f02926949d8d5ea9f56641dc56ae52aaef04179f56',
 'reviews/expert/belief-gap-v4.canonical-first-judgments.jsonl':'b1cd43c4c0e54acce5964e2010bb8ae431dfd53e6b7971d842770a7579c6b6fc',
 'reviews/expert/belief-gap-v4.canonical-histogram-inspect.py':'6b358e5c414b830c2b9767455b9a95b0f07736d305c769e81a353b41b8541baf',
 'reviews/expert/belief-gap-v4.canonical-histogram-inspect.execution.json':'b7f5719a0d2f1e171b0ce1bdde1519e4fa4ca174920b6613f039db3a4d5d6008',
 'reviews/expert/belief-gap-v4-canonical-review-attempt01.py':'8129bcebfa9a53cb5a86f37d3c6fcd8b86c7b5e9476a3609c9bc70e9f771174f',
 'reviews/expert/belief-gap-v4-canonical-attempt01.failure.json':'8b75c389cc6d2c2f18c0b880469c9a02cdff7bec2eccd6888149c05d54b32598',
 'data/RBM-001/manifests/belief-gap-v3.family_controls.json':'d197e82d9b6c35e1f7cdac7a212f5e9973446dda986b4b73669b5120a282d2a2',
 'reviews/input_only/belief-gap-v4.jsonl':'ad3c3f79a3547474476a6572c14d1e12bbe3f94e18468dd610638efdbe6d54b5',
 'reviews/expert/belief-gap-v4-input-attempt01.predictions.jsonl':'d9a55125b127edc6ab691f4981ea61edfa60b03b333e90ad4fc7c865b1ccc9ed',
 'reviews/expert/belief-gap-v4-input-attempt01.first-input-judgments.jsonl':'5a0ba9a30c7fbec74c8c69322ba592941046e5b98d375f36bda2118343fe7815',
 'reviews/expert/belief-gap-v4-input-attempt01.FINAL.json':'a234382d8e1116de7dd81c78f3e0746afa46d9495f3ed571f553d460103eebce',
 'reviews/expert/belief-gap-v4-input-attempt01.semantic-inspection.md':'d99b31bcf62882f28f3b89280392813c2e80539a27493033f77d9d278f6b9402',
 'reviews/expert/belief_gap_v2_metadata_support.py':'54324adf2332c7d85651f6217dc51d1ac6f6610ddd68d1819f43416f9f1d9e84',
 'reviews/expert/prepare_belief_gap_v1.py':'e55b7ee49b25b0c4857a3387c730501bfda4ad0572906ce13be9c23a9c11d784',
 'reviews/expert/belief-v4-prospective-interface-toys.py':'4480f30fef7bcad66f50b659d75f807fbc7b55dc42232df1dc37f9db6c9f9c8b',
 'reviews/expert/belief-v4-prospective-review-procedure.md':'ac8df46f90f8126a803f20020d7dd4c1e51ea6f44de259f485dd889f5805c700'}
EXPOSURE=['Canonical access authorized only after all3456 own input predictions/first judgments/INPUT FINAL freeze',
 'public-guided corrective same-case metadata repair; prior V3 canonical gold exposure',
 'same underlying model and public design overlap; no model-independent or unseen blind claim',
 'own public-derived V1 solver/V2 property and metadata routines reused; fresh per-row recomputation',
 'old V3 canonical/manifest allowed for exact preservation checks only; no other reviewer/private checker/producer code',
 'four existing families, two exposed-dev families, repeated worlds and nuisance views; no growth or model-efficacy claim']
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def eq(a,b):return canon(a)==canon(b)
def dig(v):return hashlib.sha256(canon(v).encode()).hexdigest()
def hp(path):
 h=hashlib.sha256();n=0
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b);n+=len(b)
 return {'sha256':h.hexdigest(),'length_bytes':n}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(text):
 def pairs(items):
  out={}
  for k,v in items:
   if k in out:raise ValueError('duplicate JSON key '+k)
   out[k]=v
  return out
 def fail(v):raise ValueError('nonfinite scalar '+v)
 return json.loads(text,object_pairs_hook=pairs,parse_constant=fail)
def save(path,value):
 with path.open('x',encoding='utf-8') as f:f.write(canon(value)+'\n')
 return hp(path)
def bindings():
 out={p:hp(ROOT/p) for p in BIND}
 for p,h in BIND.items():assert out[p]['sha256']==h,('binding changed',p)
 return out
def readrows(path):
 out={}
 for line in path.open(encoding='utf-8'):
  r=load(line);assert r['sample_id'] not in out;out[r['sample_id']]=r
 return out
def coord(r):return tuple(r[k] for k in ('source_family_key','controlled_context','auxiliary_scalar_variant','schedule_index','renderer'))
def reverse(v):
 if type(v) is dict:return {k:reverse(v[k]) for k in reversed(list(v))}
 if type(v) is list:return [reverse(x) for x in v]
 return v
def order(v):
 if type(v) is dict:return [list(v),[order(x) for x in v.values()]]
 if type(v) is list:return [order(x) for x in v]
 return v
def transpose(task,fact,b,g):
 t=copy.deepcopy(task)
 def swap(v):return g if type(v) is str and v==b else b if type(v) is str and v==g else v
 for w in t['worlds']:w['facts'][fact]=swap(w['facts'][fact])
 for o in t['observations']:
  if o['fact']==fact:o['equals']=swap(o['equals'])
 return t
def shape(claim,owner,known):
 assert type(claim) is dict and set(claim)<=CLAIM_FIELDS and ('name' in claim)+('property' in claim)==1
 name=claim.get('name',claim.get('property'));assert name in known and claim.get('scope') in ('corpus_variant','audit_probe')
 if 'source_family_key' in claim:assert claim['source_family_key']==owner['source_family_key']
 for k in ('control_id','probe_id','evidence_manifest'):
  if k in claim:assert type(claim[k]) is str and claim[k]
 if 'evidence_manifest' in claim:
  p=pathlib.PurePosixPath(claim['evidence_manifest']);assert not p.is_absolute() and '..' not in p.parts and claim['evidence_manifest']==MAN
 if 'pair_sample_ids' in claim:
  ids=claim['pair_sample_ids'];assert claim['scope']=='corpus_variant' and type(ids) is list and len(ids)==2 and all(type(x) is str and x for x in ids) and len(set(ids))==2
 if 'probe_id' in claim:assert claim['scope']=='audit_probe'
 if name in INHERITED and claim['scope']=='corpus_variant':assert {'pair_sample_ids','evidence_manifest','control_id'}<=set(claim)
 return name
def inherited(owner,claim,records,resolver,known):
 name=shape(claim,owner,known);assert name in INHERITED and claim['scope']=='corpus_variant'
 ids=claim['pair_sample_ids'];assert all(i in records for i in ids)
 l,r=records[ids[0]],records[ids[1]];a,b=l['task'],r['task'];n=len(owner['semantic_task']['observations'])
 assert n in (2,3) and len(a['observations'])==len(b['observations'])==n
 assert (l['row']['controlled_context'],r['row']['controlled_context'])==(('E','G') if n==2 else ('F','H'))
 assert all(eq(z['row'][k],owner[k]) for z in (l,r) for k in COORDS)
 assert eq(a['worlds'],b['worlds']) and eq(a['queries'],b['queries'])
 assert a['observations'][0]['fact']!=a['observations'][1]['fact']
 assert [i for i,(x,y) in enumerate(zip(a['observations'],b['observations'])) if not eq(x,y)]==[1]
 assert a['observations'][1]['fact']==b['observations'][1]['fact']
 if n==3:assert eq(a['observations'][2],a['observations'][0]) and eq(b['observations'][2],b['observations'][0])
 ap,bp=l['facts']['prefix_survivors'],r['facts']['prefix_survivors'];ao,bo=l['facts']['expected'],r['facts']['expected']
 assert ap[1] and eq(ap[1],bp[1]) and not ap[-1] and bp[-1]
 for ob in a['observations'][:2]:assert any(eq(w['facts'][ob['fact']],ob['equals']) for w in a['worlds'])
 key=(claim['control_id'],name,tuple(ids));matches=resolver.get(key,[]);assert len(matches)==1
 e=matches[0];assert MINIMUM<=set(e) and e['name']==name and e['scope']=='corpus_variant'
 assert e['source_family_key']==owner['source_family_key'] and e['auxiliary_scalar_variant']==owner['auxiliary_scalar_variant']
 assert type(e['observation_length']) is int and e['observation_length']==n and e['exact_candidate_sha256']==CSHA
 # These observed evidence fields are interpreted as assertions, not new public
 # minimum keys.  A different public evidence spelling needs its own decoder.
 evidence={
  'first_prefix_survivor_ids':ap[1],'left_prefix_counts':list(map(len,ap)),
  'right_prefix_counts':list(map(len,bp)),'left_prefix_survivor_ids':ap,'right_prefix_survivor_ids':bp,
  'left_complete_query_payloads':ao['queries'],'right_complete_query_payloads':bo['queries'],
  'typed_first_observation':a['observations'][0],'changed_fields':['observations[1].equals']}
 assert any(k in e for k in ['left_prefix_survivor_ids','first_prefix_survivor_ids']), 'no actual parsed prefix evidence supplied'
 assert any(k in e for k in ['left_complete_query_payloads','right_complete_query_payloads']), 'no actual parsed comparison evidence supplied'
 for k,v in evidence.items():
  if k in e:assert eq(e[k],v),(k,'false supplied comparison evidence')
 if 'schedule_index' in e:assert type(e['schedule_index']) is int and e['schedule_index']==owner['schedule_index']
 if 'renderer' in e:assert e['renderer']==owner['renderer']
 if 'pair_record_hashes' in e:assert eq(e['pair_record_hashes'],[l['row']['record_hash'],r['row']['record_hash']])
 return {'pair_sample_ids':ids,'name':name,'owner_length':n,'owner_nonmember':owner['sample_id'] not in ids,
         'entry_sha256':dig(e),'independent_prefix_evidence_sha256':dig(evidence),'control_id':claim['control_id']}
def payload_signature(q,prefix):
 category='empty' if not q['counts'] else 'entailed' if q['entailed'] else 'ambiguous'
 def typ(v):return 'null' if v is None else 'boolean' if type(v) is bool else 'integer' if type(v) is int else 'string'
 return {'audit_category':category,'full_query_payload':{k:v for k,v in q.items() if k!='query_id'},
  'guaranteed_null_context':category if q['guaranteed'] is None else 'nonnull',
  'value_types':[typ(x['value']) for x in q['counts']], 'final_survivors':prefix[-1],
  'prefix_survivors':prefix,'prefix_counts':list(map(len,prefix))}
def conditional_histogram(rows):
 axes={'id':(0,),'position':(1,),'fact':(2,),'id,position':(0,1),'id,fact':(0,2),'position,fact':(1,2),'id,position,fact':(0,1,2)}
 h={a:{'actual_role_histograms':collections.defaultdict(collections.Counter),'actual_full_typed_outcome_provenance_histograms':collections.defaultdict(collections.Counter)} for a in axes}
 role_triples=collections.defaultdict(collections.Counter)
 for rec in rows:
  md,t,fs=rec['row'],rec['task'],rec['facts'];inverse={v:k for k,v in md['fact_role_assignments'].items()}
  for pos,(query,q) in enumerate(zip(t['queries'],fs['expected']['queries'])):
   role=inverse[query['fact']];role='observed_control' if role=='observation_control' else role
   triple=(query['query_id'],pos,query['fact']);role_triples[role][canon(list(triple))]+=1
   sig=canon(payload_signature(q,fs['prefix_survivors']))
   for ax,indices in axes.items():
    cell=canon([triple[i] for i in indices]);h[ax]['actual_role_histograms'][cell][role]+=1;h[ax]['actual_full_typed_outcome_provenance_histograms'][cell][sig]+=1
 h={a:{k:{cell:dict(counter) for cell,counter in v.items()} for k,v in item.items()} for a,item in h.items()}
 assert all(len(c)==27 and set(c.values())=={1} for c in role_triples.values()) and len(role_triples)==3
 for ax,x in h.items():
  cells=3 if len(axes[ax])==1 else 9 if len(axes[ax])==2 else 27
  expected_count=9 if cells==3 else 3 if cells==9 else 1
  assert len(x['actual_role_histograms'])==cells and all(set(v)==set(role_triples) and set(v.values())=={expected_count} for v in x['actual_role_histograms'].values())
  assert len({canon(v) for v in x['actual_full_typed_outcome_provenance_histograms'].values()})==1
 return h
def normalize_histogram_encoding(hist):
 """Decode diagnostic coordinate labels only; never coerce task/output scalars.

 Public schemas require factual typed histogram equality but do not prescribe
 whether a position coordinate inside a textual dictionary key is written as
 0 or "0". Accept only the canonical labels 0/1/2, with no colliding decoded
 buckets. Query IDs and fact labels remain strings; outcome signatures retain
 JSON types and all prefix provenance.
 """
 axes={'id':(0,),'position':(1,),'fact':(2,),'id,position':(0,1),'id,fact':(0,2),'position,fact':(1,2),'id,position,fact':(0,1,2)}
 assert type(hist) is dict and set(hist)==set(axes)
 result={}
 for axis,dimensions in axes.items():
  metrics=hist[axis]
  assert type(metrics) is dict and set(metrics)=={'actual_role_histograms','actual_full_typed_outcome_provenance_histograms'}
  result[axis]={}
  for kind,cells in metrics.items():
   assert type(cells) is dict
   decoded={}
   for cell,counters in cells.items():
    coords=load(cell);assert type(coords) is list and len(coords)==len(dimensions)
    for j,d in enumerate(dimensions):
     if d==1:
      value=coords[j]
      if type(value) is str:assert value in ('0','1','2');value=int(value)
      assert type(value) is int and 0<=value<=2
      coords[j]=value
     else:assert type(coords[j]) is str and coords[j]
    key=canon(coords);assert key not in decoded,'ambiguous duplicate decoded histogram bucket'
    assert type(counters) is dict
    dst={}
    for label,count in counters.items():
     assert type(count) is int and count>0
     if kind=='actual_full_typed_outcome_provenance_histograms':
      signature=load(label);assert type(signature) is dict
      label=canon(signature)
     else:assert type(label) is str and label
     assert label not in dst,'ambiguous duplicate decoded histogram signature'
     dst[label]=count
    decoded[key]=dst
   result[axis][kind]=decoded
 return result
def histogram_encoding_tests(supplied,actual):
 results=[]
 def check(a,b):assert eq(normalize_histogram_encoding(a),normalize_histogram_encoding(b))
 def pos(name,fn):fn();results.append({'name':name,'expected':'accept','passed':True})
 def neg(name,fn):
  try:fn()
  except (AssertionError,ValueError,KeyError,TypeError):results.append({'name':name,'expected':'reject','passed':True});return
  raise AssertionError('negative histogram probe accepted: '+name)
 pos('actual supplied textual diagnostic positions match independent integer positions',lambda:check(supplied,actual))
 pos('independent integer diagnostic positions remain valid',lambda:check(actual,actual))
 for value in [True,1.0,3,'00','0x0','true']:
  h=copy.deepcopy(actual);cells=h['position']['actual_role_histograms'];key=next(iter(cells));cells[canon([value])]=cells.pop(key)
  neg('malformed position coordinate '+canon(value),lambda h=h:check(h,actual))
 h=copy.deepcopy(actual);cells=h['position']['actual_role_histograms'];key=next(iter(cells));value=load(key)[0];cells[canon([str(value)])]=copy.deepcopy(cells[key])
 neg('integer and text positions cannot create duplicate decoded buckets',lambda:check(h,actual))
 h=copy.deepcopy(actual);cells=h['id']['actual_role_histograms'];key=next(iter(cells));cells[canon([0])]=cells.pop(key)
 neg('query ID coordinate cannot be scalar-coerced',lambda:check(h,actual))
 for count in [True,0,1.0]:
  h=copy.deepcopy(actual);counter=next(iter(h['fact']['actual_role_histograms'].values()));counter[next(iter(counter))]=count
  neg('histogram count has positive integer type '+canon(count),lambda h=h:check(h,actual))
 for action in ('boolean-query-label','prefix-world-removal','typed-null-query-bin'):
  h=copy.deepcopy(actual);counter=next(iter(h['id']['actual_full_typed_outcome_provenance_histograms'].values()));key=next(iter(counter));sig=load(key)
  if action=='boolean-query-label':sig['full_query_payload']['entailed']=not sig['full_query_payload']['entailed']
  elif action=='prefix-world-removal':sig['prefix_survivors'][0]=sig['prefix_survivors'][0][1:]
  else:
   counts=sig['full_query_payload']['counts']
   if counts:counts[0]['value']=False if counts[0]['value'] is None else None
   else:sig['full_query_payload']['guaranteed']=False
  n=counter.pop(key);counter[canon(sig)]=n
  neg('full typed outcome and provenance corruption '+action,lambda h=h:check(h,actual))
 return results
def make_resolver(entries):
 out=collections.defaultdict(list)
 for e in entries:out[(e.get('control_id'),e.get('name'),tuple(e.get('pair_sample_ids',[])))].append(e)
 return out
def tests(records,resolver,known):
 results=[];owner=next(r['row'] for r in records.values() if r['row']['controlled_context']=='A')
 claim=next(c for c in owner['family_control_claims'] if c.get('name') in INHERITED)
 key=(claim['control_id'],claim['name'],tuple(claim['pair_sample_ids']));entry=resolver[key][0]
 def pos(name,fn):fn();results.append({'name':name,'expected':'accept','passed':True})
 def neg(name,fn):
  try:fn()
  except (AssertionError,ValueError,KeyError,TypeError,IndexError):results.append({'name':name,'expected':'reject','passed':True});return
  raise AssertionError('negative probe accepted: '+name)
 pos('actual valid nonmember owner',lambda:inherited(owner,claim,records,resolver,known))
 pos('same pair ID distinct names both resolve',lambda:[inherited(owner,c,records,resolver,known) for c in owner['family_control_claims'] if c.get('name') in INHERITED])
 for k in ('pair_sample_ids','evidence_manifest','control_id'):
  c=copy.deepcopy(claim);del c[k];neg('missing mandatory inline '+k,lambda c=c:inherited(owner,c,records,resolver,known))
 for k,v in [('pair_sample_ids',claim['pair_sample_ids'][::-1]),('pair_sample_ids',['absent',claim['pair_sample_ids'][1]]),('scope','audit_probe'),('property',claim['name'])]:
  c=copy.deepcopy(claim);c[k]=v;neg('incorrect claim '+k+canon(v),lambda c=c:inherited(owner,c,records,resolver,known))
 for k,v in [('source_family_key','foreign'),('auxiliary_scalar_variant','transposed'),('schedule_index',1),('renderer','reverse')]:
  o=copy.deepcopy(owner);o[k]=v;neg('owner coordinate '+k,lambda o=o:inherited(o,claim,records,resolver,known))
 o=copy.deepcopy(owner);o['semantic_task']['observations'].append(copy.deepcopy(o['semantic_task']['observations'][0]));neg('owner actual length3 cannot use length2 pair',lambda:inherited(o,claim,records,resolver,known))
 rr={**resolver,key:[entry,copy.deepcopy(entry)]};neg('duplicate exact resolver key',lambda:inherited(owner,claim,records,rr,known))
 for k in sorted(MINIMUM):
  e=copy.deepcopy(entry);del e[k];rr={**resolver,key:[e]};neg('missing minimum entry '+k,lambda rr=rr:inherited(owner,claim,records,rr,known))
 for k,v in [('exact_candidate_sha256','0'*64),('name','foreign'),('observation_length',True),('first_prefix_survivor_ids',[])]:
  e=copy.deepcopy(entry);e[k]=v;rr={**resolver,key:[e]};neg('false entry '+k,lambda rr=rr:inherited(owner,claim,records,rr,known))
 # Public optionality is tested explicitly, not only rejection cases.
 e={k:v for k,v in entry.items() if k in MINIMUM|{'first_prefix_survivor_ids','left_complete_query_payloads','right_complete_query_payloads'}}
 rr={**resolver,key:[e]};pos('minimum entry plus actual comparison evidence; optional schedule renderer pair hashes absent',lambda:inherited(owner,claim,records,rr,known))
 c=copy.deepcopy(claim);c['property']=c.pop('name');pos('exclusive property alias',lambda:inherited(owner,c,records,resolver,known))
 o=records[claim['pair_sample_ids'][0]]['row'];pos('owner may itself be empty pair member',lambda:inherited(o,claim,records,resolver,known))
 for name in sorted(NEW):pos('aggregate encoding admitted pending independently reconstructed full group '+name,lambda name=name:shape({'name':name,'scope':'corpus_variant','evidence_manifest':MAN,'control_id':'arbitrary descriptive ID'},owner,known))
 return results
def features(rec):
 t=rec['task'];p=rec['facts']['prefix_survivors'];cols={k:sorted(collections.Counter(canon(w['facts'][k]) for w in t['worlds']).items()) for k in t['worlds'][0]['facts']}
 observations=[o['equals'] for o in t['observations']]
 return {'ordered_observation_scalars':observations,'coarse_counts':[len(t['worlds']),len(t['worlds'][0]['facts']),len(t['observations']),len(t['queries'])],
  'first_prefix_ids':p[1],'coarse_first_prefix':[len(t['worlds']),len(t['observations']),len(p[1])],
  'column_frequency_plus_observations':[cols,t['observations']], 'task_character_length':len(canon(t)),
  'prompt_character_length':len(rec['row']['rendered_input']), 'scalar_and_character_length':[observations,len(rec['row']['rendered_input'])],
  'full_table_conjunction_lookup':[t['worlds'],t['observations']]}
def freeze():
 source=pathlib.Path(__file__).resolve();b=bindings()
 with (OUT/(PREFIX+'.source.preexecution.py')).open('xb') as f:f.write(source.read_bytes())
 p=OUT/(PREFIX+'.preexecution.json');save(p,{'actual_author':ROLE,'reviewer_execution_id':ROLE,'created_utc':utc(),'source':hp(source),'dependencies_before_import_test_execution':b,
  'python_executable':str(pathlib.Path(sys.executable).resolve()),'python_binary':hp(pathlib.Path(sys.executable).resolve()),'exposure':EXPOSURE,
  'own_correction':'Own initial comparison encoded histogram position dictionary labels as integers; supplied diagnostic labels are strings. Normalize only canonical coordinate labels and preserve full typed targets/signatures. Original first REVISE receipts immutable.',
  'scope':'fresh corrective canonical comparisons against already frozen own3456 pre-gold expectations; no release approval',
  'all_data_reads_now_authorized':['V4 canonical','V4 manifest','V3 canonical and manifest preservation only'],
  'own_dependency_use':'Import own V2 metadata/property support only after this freeze; prior prospective source frozen/read but not executed again',
  'old_v3_manifest_use':'hash-bound; actual prior candidate fields compared; manifest preserved but not used as semantic expectations'})
 print(canon({'source':hp(source),'preexecution':hp(p)}))
def execute():
 start=utc();timer=time.monotonic();source=pathlib.Path(__file__).resolve();prepath=OUT/(PREFIX+'.preexecution.json');pre=load(prepath.read_text());before=bindings()
 assert before==pre['dependencies_before_import_test_execution'] and hp(source)==pre['source']==hp(OUT/(PREFIX+'.source.preexecution.py'))
 assert hp(pathlib.Path(sys.executable).resolve())==pre['python_binary']
 sp=importlib.util.spec_from_file_location('expert_owned_v2_metadata_for_v4',ROOT/'reviews/expert/belief_gap_v2_metadata_support.py');own=importlib.util.module_from_spec(sp);sp.loader.exec_module(own)
 own.FAMILY_CONTROLS.update(NEW);known=own.FAMILY_CONTROLS
 view=readrows(ROOT/'reviews/input_only/belief-gap-v4.jsonl');pred=readrows(OUT/'belief-gap-v4-input-attempt01.predictions.jsonl');rows=readrows(ROOT/CAND)
 assert len(rows)==len(view)==len(pred)==3456 and set(rows)==set(view)==set(pred)
 manifest=load((ROOT/MAN).read_text());assert manifest['exact_candidate_sha256']==CSHA
 records={};local_errors=collections.defaultdict(list);propc=collections.Counter();metadata_diff=collections.Counter();preservation=collections.Counter();lookup={}
 for sid,row in rows.items():
  p,v=pred[sid],view[sid];es=local_errors[sid]
  for k in v:assert eq(v[k],row[k]),(sid,'stripped canonical join',k)
  assert row['record_hash']==p['record_hash']
  hash_ok=dig({k:x for k,x in row.items() if k!='record_hash'})==row['record_hash']
  if not hash_ok:es.append('full canonical record hash differs')
  task=own.parse_prompt(row);strict_task=load(row['rendered_input'].split('Task JSON:\n',1)[1]);assert eq(task,strict_task) and eq(task,row['semantic_task'])
  facts=own.row_facts(task);assert eq(facts['expected'],p['expected_target']) and eq(facts['prefix_survivors'],p['prefix_survivors'])
  if not eq(facts['expected'],row['verified_target']):es.append('canonical target differs from frozen full typed expectation')
  md,errors=own.metadata_audit(row,task,facts);es.extend(errors)
  for k in ['auxiliary_scalar_variant','auxiliary_scalar_control']:md[k]=own.locate(row,k)
  ctrl=row['auxiliary_scalar_control'];y=row['fact_role_assignments']['observation_auxiliary']
  if type(ctrl) is not dict or set(ctrl)!={'fact','baseline_value','supported_value'}:es.append('wrong auxiliary scalar control shape')
  else:
   b,g=ctrl['baseline_value'],ctrl['supported_value']
   if ctrl['fact']!=y or y in [q['fact'] for q in task['queries']]:es.append('auxiliary control role/queried exclusion fails')
   if type(b) is not str or type(g) is not str or b!='linen' or g in {'stone','unknown','linen'}:es.append('public targeted scalar literal condition fails')
   if not all(any(eq(w['facts'][y],x) for w in task['worlds']) for x in (b,g)):es.append('both scalar strings not occupied in full y column')
  if row['auxiliary_scalar_variant'] not in ('identity','transposed'):es.append('unknown auxiliary scalar variant')
  allowed={2} if row['controlled_context'] in 'GH' else {2,3}
  if type(row['difficulty']) is not int or row['difficulty'] not in allowed or not eq(row['difficulty'],row['structural_difficulty']):es.append('unsupported declared component difficulty')
  if facts['metrics']['duplicate_visible_payloads']:es.append('duplicate full fact-payload worlds')
  if row['split']!=('train' if row['source_family_index'] in (12,13) else 'dev_challenge') or row['semantic_parent_id']!=row['semantic_family_id']:es.append('source split/family-parent mismatch')
  if row['review_status']!='PENDING' or row['review_receipt_ids']:es.append('producer claims reviewed status before review')
  for k in ['auxiliary_scalar_control','auxiliary_scalar_variant','family_control_claims','property_witnesses','verified_target','schedule_index','structural_family_profile','source_family_key']:
   if k in row['rendered_input']:es.append('private metadata field appears in full prompt: '+k)
  for prop in set(row['skill_tags'])&own.PROPERTIES:propc[prop]+=1
  for k,x in p['independently_computed_properties'].items():assert x['witness_count']==len(facts['properties'][k]) and x['actual_witnesses_sha256']==dig(facts['properties'][k])
  records[sid]={'row':row,'task':task,'facts':facts,'metadata':md,'hash_ok':hash_ok}
  co=coord(row);assert co not in lookup;lookup[co]=sid
 # Exact old/new preservation is established before running any old fitted-rule ideas.
 oldrows=readrows(ROOT/OLD);oldlookup={coord(r):r for r in oldrows.values()};assert len(oldlookup)==len(records)
 stable=['semantic_task','rendered_input','verified_target','split','difficulty','structural_difficulty','skill_tags','property_witnesses','fact_role_assignments','query_role_assignments','auxiliary_scalar_control','auxiliary_scalar_variant','structural_family_profile','source_family_index','source_family_key']
 for sid,rec in records.items():
  row=rec['row'];old=oldlookup[coord(row)]
  assert row['sample_id']==view[sid]['sample_id'] and old['sample_id']==pred[sid]['join_evidence']['prior_sample_id']
  assert dig({k:v for k,v in old.items() if k!='record_hash'})==old['record_hash']
  for k in stable:
   if not eq(old[k],row[k]):local_errors[sid].append('V4 preservation changed '+k)
   else:preservation[k]+=1
  for k in set(old)|set(row):
   if not eq(old.get(k),row.get(k)):metadata_diff[k]+=1
 del oldrows,oldlookup
 families=sorted({r['row']['source_family_key'] for r in records.values()});batch_errors=[];ledger=[];counts=collections.Counter();profiles={};null_contrasts={}
 def error(message,ids=(),evidence=None):
  entry={'error':message,'sample_ids':list(ids),'evidence':evidence};batch_errors.append(entry)
  for sid in ids:
   if sid in records:local_errors[sid].append(message)
 def get(f,c,v,s,ren):return records[lookup[(f,c,v,s,ren)]]
 def addpair(name,l,r,okay,detail=None):
  ids=[l['row']['sample_id'],r['row']['sample_id']];ledger.append({'name':name,'pair_sample_ids':ids,'passed':bool(okay),'evidence_sha256':dig(detail or {})});counts[name]+=1
  if not okay:error(name+' actual pair relationship failed',ids,detail)
  return ids
 for fam in families:
  fr=[r for r in records.values() if r['row']['source_family_key']==fam]
  if len(fr)!=864 or len({r['row']['split'] for r in fr})!=1 or len({r['row']['semantic_family_id'] for r in fr})!=1:error('incomplete family or crossed split custody',[r['row']['sample_id'] for r in fr])
  # Literal values and pools fixed; control fact names are fixed unqueried roles.
  for k in ['query_id_pool','queried_fact_name_pool','auxiliary_scalar_control']:
   if len({canon(r['row'][k]) for r in fr})!=1:error('unstable family '+k,[r['row']['sample_id'] for r in fr])
  for variant in ('identity','transposed'):
   rep=get(fam,'A',variant,0,'forward');graph=own.incidence(rep['task'],rep['row']['fact_role_assignments']);perrs=own.profile_errors(rep['row']['structural_family_profile'],graph,rep['facts'])
   profiles[fam+':'+variant]={'profile':rep['row']['structural_family_profile'],'incidence':graph,'representative':rep['row']['sample_id']}
   for e in perrs:error(e,[r['row']['sample_id'] for r in fr])
   for s in range(27):
    for ren in ('forward','reverse'):
     group={c:get(fam,c,variant,s,ren) for c in 'ABCDEFGH'};A,B,C,D,E,F,G,H=[group[c] for c in 'ABCDEFGH']
     if any(not eq(A['task']['worlds'],r['task']['worlds']) or not eq(A['task']['queries'],r['task']['queries']) for r in group.values()):error('within-variant contexts altered table/ordered queries',[r['row']['sample_id'] for r in group.values()])
     for name,l,r in [('equal_support_usefulness_counterfactual',A,B),('observation_reorder_final_invariance',A,C),('nonempty_repetition_final_invariance',A,D),('cross_fact_contradiction_counterfactual',A,E),('post_empty_append_absorption',E,F),('nonempty_repetition_final_invariance',G,H)]:
      a,b=l['task']['observations'],r['task']['observations'];ao,bo=l['facts']['expected'],r['facts']['expected'];change=own.single_equals_change(a,b)
      if name=='equal_support_usefulness_counterfactual':ok=change is not None and len(ao['surviving_worlds'])==len(bo['surviving_worlds'])>=2 and any(x['can_reduce_uncertainty']!=y['can_reduce_uncertainty'] for x,y in zip(ao['queries'],bo['queries'])) and bool(r['facts']['properties']['multiworld_all_queries_redundant'])
      elif name=='observation_reorder_final_invariance':ok=sorted(map(canon,a))==sorted(map(canon,b)) and own.same_final(ao,bo) and not eq(ao['belief_sizes'],bo['belief_sizes'])
      elif name=='nonempty_repetition_final_invariance':ok=len(b)==len(a)+1 and eq(b[:-1],a) and any(eq(b[-1],x) for x in a) and bool(ao['surviving_worlds']) and own.same_final(ao,bo) and bo['belief_sizes'][-1]==len(ao['surviving_worlds'])
      elif name=='cross_fact_contradiction_counterfactual':ok=change is not None and bool(r['facts']['properties']['cross_fact_inconsistent_conjunction'])
      else:ok=len(b)==len(a)+1 and eq(b[:-1],a) and not ao['surviving_worlds'] and own.same_final(ao,bo) and bo['belief_sizes'][-1]==0 and bool(r['facts']['properties']['post_empty_absorption'])
      addpair(name,l,r,ok,{'changed_index':change,'left_prefix':l['facts']['prefix_survivors'],'right_prefix':r['facts']['prefix_survivors'],'left_queries':ao['queries'],'right_queries':bo['queries']})
      if name=='equal_support_usefulness_counterfactual' and change is not None:
       av,bv=a[change]['equals'],b[change]['equals']
       typed=any(eq(alias,bv) for alias in own.aliases(av)) and l['facts']['prefix_survivors'][change+1] and r['facts']['prefix_survivors'][change+1] and not eq(l['facts']['prefix_survivors'][change+1],r['facts']['prefix_survivors'][change+1])
       if typed:addpair('typed_equals_counterfactual',l,r,True,{'changed_index':change,'typed_values':[av,bv],'left_prefix':l['facts']['prefix_survivors'][change+1],'right_prefix':r['facts']['prefix_survivors'][change+1]})
     if not A['facts']['properties']['conjunction_required_invariant'] or not A['facts']['properties']['useful_after_conjunctive_evidence']:error('A lacks essential conjunction/useful witness',[A['row']['sample_id']])
     if not eq(F['task']['observations'][-1],E['task']['observations'][0]):error('F append not exact first-observation repeat',[F['row']['sample_id']])
     if not eq(G['task']['observations'][1]['equals'],G['row']['auxiliary_scalar_control']['supported_value'] if variant=='identity' else G['row']['auxiliary_scalar_control']['baseline_value']):error('identity/transposed G scalar control differs',[G['row']['sample_id']])
   nulls={k:[] for k in ['entailed_null','ambiguous_null','empty']}
   for r in fr:
    if r['row']['auxiliary_scalar_variant']!=variant:continue
    if r['facts']['properties']['multiworld_entailed_null']:nulls['entailed_null'].append(r['row']['sample_id'])
    if r['facts']['properties']['ambiguous_null_partition']:nulls['ambiguous_null'].append(r['row']['sample_id'])
    if not r['facts']['expected']['surviving_worlds']:nulls['empty'].append(r['row']['sample_id'])
   # A family may lack a property/control that it does not claim.  Preserve the
   # actual per-family support instead of inventing universal null ambiguity.
   null_contrasts[(fam,variant)]={k:bool(v) for k,v in nulls.items()}
 # All conditional query triples and producer-published histogram assertions.
 grouped=collections.defaultdict(list)
 for r in records.values():grouped[(r['row']['source_family_key'],r['row']['controlled_context'],r['row']['auxiliary_scalar_variant'],r['row']['renderer'])].append(r)
 supplied={tuple(g[k] for k in ['source_family_key','controlled_context','auxiliary_scalar_variant','renderer']):g for g in manifest['actual_conditional_query_histograms']}
 if len(supplied)!=len(manifest['actual_conditional_query_histograms']) or set(supplied)!=set(grouped):error('missing/duplicate manifest conditional groups',list(records))
 histogram_receipts=[];decoded_tables=set();identity_tables=set();visible_tables=set();decoded_cases=set()
 for key,group in grouped.items():
  ids=[r['row']['sample_id'] for r in group]
  try:
   assert sorted(r['row']['schedule_index'] for r in group)==list(range(27))
   assert len({canon(own.decoded_task(r['task'],r['row'])) for r in group})==1
   actual=conditional_histogram(group);g=supplied[key]
   assert eq(normalize_histogram_encoding(g['histograms']),normalize_histogram_encoding(actual)) and eq(g['actual_schedule_indices'],list(range(27)))
   assert g['complete_positive_triple_support'] is True and type(g['measured_bucket_histogram_difference']) is int and g['measured_bucket_histogram_difference']==0
   histogram_receipts.append({'group':list(key),'rows':len(group),'actual_histograms_sha256':dig(actual),'supplied_normalized_histograms_sha256':dig(normalize_histogram_encoding(g['histograms'])),'diagnostic_coordinate_text_only_normalized':True,'seven_axes_equal':True,'three_role_triples_complete':True})
  except (AssertionError,ValueError,KeyError,TypeError) as e:error('conditional typed histogram/control assertion failed',ids,str(e))
 for rec in records.values():
  row,t,fs=rec['row'],rec['task'],rec['facts'];fam,c,v,s,ren=coord(row);control=row['auxiliary_scalar_control'];y,b,g=control['fact'],control['baseline_value'],control['supported_value']
  norm=transpose(t,y,b,g) if v=='transposed' else t;dt=own.decoded_task(t,row);di=own.decoded_task(norm,row)
  decoded_tables.add(canon(dt['worlds']));identity_tables.add(canon(di['worlds']));visible_tables.add(canon(t['worlds']));decoded_cases.add(canon([fam,c,v,dt]))
  if v=='identity':
   other=get(fam,c,'transposed',s,ren);changed=transpose(t,y,b,g)
   okay=eq(changed,other['task']) and eq(t,transpose(changed,y,b,g)) and not eq(t['worlds'],changed['worlds']) and eq(fs['expected'],other['facts']['expected']) and eq(fs['prefix_survivors'],other['facts']['prefix_survivors'])
   addpair('auxiliary_scalar_transposition_invariance',rec,other,okay,{'control':control,'prefix_ids':fs['prefix_survivors'],'target':fs['expected']})
  if c in 'EF':
   other=get(fam,{'E':'G','F':'H'}[c],{'identity':'transposed','transposed':'identity'}[v],s,ren)
   okay=eq(t['observations'],other['task']['observations']) and eq(t['queries'],other['task']['queries']) and eq(transpose(t,y,b,g)['worlds'],other['task']['worlds']) and not eq(t['worlds'],other['task']['worlds']) and eq(fs['prefix_survivors'][1],other['facts']['prefix_survivors'][1]) and not fs['prefix_survivors'][-1] and bool(other['facts']['prefix_survivors'][-1])
   addpair('same_observations_auxiliary_table_counterfactual',rec,other,okay,{'first_prefix':fs['prefix_survivors'][1],'observations':t['observations'],'queries':t['queries'],'final_supports':[fs['prefix_survivors'][-1],other['facts']['prefix_survivors'][-1]]})
  if ren=='forward':
   other=get(fam,c,v,s,'reverse');lead,body=row['rendered_input'].split('Task JSON:\n');ol,ob=other['row']['rendered_input'].split('Task JSON:\n')
   okay=lead==ol and body!=ob and eq(task:=t,other['task']) and eq(order(reverse(load(body))),order(load(ob))) and eq(fs['expected'],other['facts']['expected'])
   addpair('field_order_invariance',rec,other,okay,{'ordered_objects_reversed_only':okay,'task_sha256':dig(task),'target_sha256':dig(fs['expected'])})
 expected_cross={'conditional_triple_groups':len(grouped),'contexts':list('ABCDEFGH'),'source_families':len(families),'variants':['identity','transposed']}
 if not eq(manifest['context_auxiliary_scalar_variant_crossing'],expected_cross):error('false manifest scalar/context coverage summary',list(records))
 # Fresh specialized named entries; same IDs under different names are allowed.
 resolver=make_resolver(manifest['supplied_same_table_v2_pairs']);entry_uses=collections.Counter();inherited_receipts=[];nonmembers=0
 for sid,rec in records.items():
  claims=rec['row']['family_control_claims'];names={c.get('name',c.get('property')) for c in claims}
  if not INHERITED|NEW<=names:error('missing required inherited/V3 family identifier',[sid])
  for c in claims:
   try:
    name=shape(c,rec['row'],known)
    if c['scope']=='corpus_variant' and name in INHERITED:
     witness=inherited(rec['row'],c,records,resolver,known);entry_uses[(c['control_id'],name,tuple(c['pair_sample_ids']))]+=1;nonmembers+=witness['owner_nonmember'];inherited_receipts.append({'owner_sample_id':sid,'owner_record_hash':rec['row']['record_hash'],**witness})
    elif c['scope']=='corpus_variant' and 'pair_sample_ids' in c:
     ids=c['pair_sample_ids'];assert all(i in records and records[i]['row']['source_family_key']==rec['row']['source_family_key'] for i in ids)
     assert any(x['name']==name and x['passed'] and set(x['pair_sample_ids'])==set(ids) for x in ledger)
    elif c['scope']=='corpus_variant':
     if name=='null_guarantee_contrast':assert all(null_contrasts[(rec['row']['source_family_key'],rec['row']['auxiliary_scalar_variant'])].values())
     elif name not in ['query_id_position_fact_crossing','context_auxiliary_scalar_variant_crossing']:
      assert any(x['name']==name and x['passed'] and records[x['pair_sample_ids'][0]]['row']['source_family_key']==rec['row']['source_family_key'] for x in ledger)
   except (AssertionError,ValueError,KeyError,TypeError) as e:error('family claim invalid: '+str(c.get('name',c.get('property'))),[sid],str(e))
 if set(entry_uses)!=set(resolver) or any(len(v)!=1 for v in resolver.values()):error('unused/missing/ambiguous inherited named entries',list(records))
 # Independently execute every supplied audit task; no credit from its reference output.
 probe_receipts=[];probe_map={}
 for e in manifest['executed_audit_probes']:
  try:
   assert e['scope']=='audit_probe' and e['added_records']==e['added_families']==0
   r=records[e['source_sample_id']];t=e['probe_task'];own.validate_task(t);out,p=own.solve(t);other,op=own.solve_conjunction(t);assert eq(out,other) and eq(p,op)
   original=r['task'];base=r['facts']['expected'];name=e['name']
   assert r['row']['source_family_key']==e['source_family_key'] and r['row']['auxiliary_scalar_variant']==e['auxiliary_scalar_variant']
   assert eq(t['observations'],original['observations']) and eq(t['queries'],original['queries'])
   if name=='world_order_invariance':assert sorted(map(canon,t['worlds']))==sorted(map(canon,original['worlds'])) and not eq(t['worlds'],original['worlds']) and eq(out,base) and eq(p,r['facts']['prefix_survivors'])
   elif name=='world_id_rename_equivariance':
    assert len(t['worlds'])==len(original['worlds']) and all(eq(a['facts'],b['facts']) for a,b in zip(t['worlds'],original['worlds']))
    rename={a['world_id']:b['world_id'] for a,b in zip(original['worlds'],t['worlds'])};assert len(set(rename.values()))==len(rename)
    wanted=copy.deepcopy(base);wanted['surviving_worlds']=sorted(rename[i] for i in base['surviving_worlds']);assert eq(out,wanted) and eq(p,[sorted(rename[i] for i in xs) for xs in r['facts']['prefix_survivors']])
   else:raise AssertionError('unsupported advertised probe semantics '+name)
   assert eq(out,e['producer_reference_target'])
   key=(e['source_family_key'],e['auxiliary_scalar_variant'],name,e['probe_id']);assert key not in probe_map;probe_map[key]=True
   probe_receipts.append({'probe_id':e['probe_id'],'name':name,'source_sample_id':e['source_sample_id'],'actual_expert_execution':ROLE,'probe_task_sha256':dig(t),'own_output_sha256':dig(out),'own_prefix_sha256':dig(p),'passed':True,'corpus_units_added':0})
  except (AssertionError,ValueError,KeyError,TypeError) as ex:error('supplied audit probe failed',[],{'probe_id':e.get('probe_id'),'error':str(ex)})
 for sid,rec in records.items():
  for c in rec['row']['family_control_claims']:
   if c['scope']=='audit_probe':
    key=(rec['row']['source_family_key'],rec['row']['auxiliary_scalar_variant'],c.get('name',c.get('property')),c.get('probe_id'))
    if key not in probe_map:error('advertised audit probe not actually resolved and executed',[sid],key)
 if manifest['probes_add_corpus_records']!=0 or manifest['probes_add_source_families']!=0:error('probe corpus-growth assertion false',list(records))
 # Verify every diagnostic supplied pair payload against own executable witnesses.
 ledgers={(x['name'],tuple(x['pair_sample_ids'])):x for x in ledger}
 for collection,name in [('within_context_scalar_pairs','auxiliary_scalar_transposition_invariance'),('same_observation_opposite_status_pairs','same_observations_auxiliary_table_counterfactual'),('field_order_pairs','field_order_invariance')]:
  actual_pairs=[tuple(e['pair_sample_ids']) for e in manifest[collection]]
  derived_pairs={tuple(e['pair_sample_ids']) for e in ledger if e['name']==name}
  if len(actual_pairs)!=len(set(actual_pairs)) or set(actual_pairs)!=derived_pairs:error('incomplete or duplicated actual manifest pair collection '+collection,list(records))
 def rpair(e):return [records[i] for i in e['pair_sample_ids']]
 for e in manifest['within_context_scalar_pairs']:
  a,b=rpair(e);ids=e['pair_sample_ids'];key=(e['name'],tuple(ids))
  try:
   assert key in ledgers and ledgers[key]['passed'] and e['scope']=='corpus_variant' and e['candidate_sha256']==CSHA
   assert eq(e['auxiliary_scalar_control'],a['row']['auxiliary_scalar_control']) and e['exact_changed_world_column']==a['row']['auxiliary_scalar_control']['fact'] and e['involution_verified'] is True
   fs=a['facts'];wanted={'complete_target':fs['expected'],'prefix_counts':list(map(len,fs['prefix_survivors'])),'prefix_survivor_ids':fs['prefix_survivors']};assert eq(e['prefix_and_target_evidence'],wanted)
  except (AssertionError,KeyError,TypeError) as ex:error('false supplied scalar-pair evidence',ids,str(ex))
 for e in manifest['same_observation_opposite_status_pairs']:
  a,b=rpair(e);ids=e['pair_sample_ids']
  try:
   assert (e['name'],tuple(ids)) in ledgers and ledgers[(e['name'],tuple(ids))]['passed'] and e['scope']=='corpus_variant' and e['candidate_sha256']==CSHA
   assert e['observation_length']==len(a['task']['observations']) and eq(e['ordered_identical_observations'],a['task']['observations']) and eq(e['ordered_identical_queries'],a['task']['queries']) and eq(e['auxiliary_scalar_control'],a['row']['auxiliary_scalar_control'])
   for side,r in [('left',a),('right',b)]:assert eq(e[side],{'complete_target':r['facts']['expected'],'prefix_counts':list(map(len,r['facts']['prefix_survivors'])),'prefix_survivor_ids':r['facts']['prefix_survivors']})
  except (AssertionError,KeyError,TypeError) as ex:error('false supplied opposite-status evidence',ids,str(ex))
 for e in manifest['field_order_pairs']:
  a,b=rpair(e);ids=e['pair_sample_ids']
  try:assert ('field_order_invariance',tuple(ids)) in ledgers and ledgers[('field_order_invariance',tuple(ids))]['passed'] and e['typed_task_sha256']==dig(a['task']) and e['complete_target_sha256']==dig(a['facts']['expected'])
  except (AssertionError,KeyError,TypeError) as ex:error('false supplied field-order evidence',ids,str(ex))
 for e in manifest['supplied_original_pairs']:
  if e['name']=='null_guarantee_contrast':
   ids=e.get('evidence_sample_ids',[])
   try:
    assert ids and len(set(ids))==len(ids) and e['scope']=='corpus_variant'
    rr=[records[i] for i in ids]
    assert all(r['row']['source_family_key']==e['source_family_key'] and r['row']['auxiliary_scalar_variant']==e['auxiliary_scalar_variant'] for r in rr)
    assert any(r['facts']['properties']['multiworld_entailed_null'] for r in rr)
    assert any(r['facts']['properties']['ambiguous_null_partition'] for r in rr)
    assert any(not r['facts']['expected']['surviving_worlds'] for r in rr)
   except (AssertionError,KeyError,TypeError) as ex:error('false supplied original null-family evidence',ids,str(ex))
   continue
  a,b=rpair(e);ids=e['pair_sample_ids'];name=e['name']
  try:
   assert e['scope']=='corpus_variant' and all(r['row']['source_family_key']==e['source_family_key'] and r['row']['auxiliary_scalar_variant']==e['auxiliary_scalar_variant'] for r in (a,b))
   if name=='typed_equals_counterfactual':
    i=own.single_equals_change(a['task']['observations'],b['task']['observations']);assert i is not None and eq(a['task']['worlds'],b['task']['worlds']) and eq(a['task']['queries'],b['task']['queries'])
    av,bv=a['task']['observations'][i]['equals'],b['task']['observations'][i]['equals'];assert any(eq(x,bv) for x in own.aliases(av)) and a['facts']['prefix_survivors'][i+1] and b['facts']['prefix_survivors'][i+1] and not eq(a['facts']['prefix_survivors'][i+1],b['facts']['prefix_survivors'][i+1])
   elif name in INHERITED:assert a['row']['controlled_context'] in 'EF' and b['row']['controlled_context'] in 'GH'
   else:assert (name,tuple(ids)) in ledgers and ledgers[(name,tuple(ids))]['passed']
  except (AssertionError,KeyError,TypeError) as ex:error('false original supplied family-pair evidence',ids,str(ex))
 component_tests=tests(records,resolver,known)
 firstkey=next(iter(grouped));component_tests+=histogram_encoding_tests(supplied[firstkey]['histograms'],conditional_histogram(grouped[firstkey]))
 # Measured input-only features and lookup fits, labels solely own exact facts.
 buckets=collections.defaultdict(lambda:collections.defaultdict(collections.Counter));by_split=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(collections.Counter)))
 scalarhist=collections.defaultdict(collections.Counter);rules=collections.defaultdict(collections.Counter);counterexamples=[];feature_rows=[]
 for sid,rec in records.items():
  row,t,fs=rec['row'],rec['task'],rec['facts'];empty=not bool(fs['expected']['surviving_worlds']);status='empty' if empty else 'nonempty';vals=[o['equals'] for o in t['observations']]
  marker=any(type(v) is str and v in {'stone','unknown'} for v in vals);linen=any(type(v) is str and v=='linen' for v in vals)
  for name,predict in [('v1_original_scalar_marker_empty',marker),('prospective_marker_and_auxiliary_token_empty',marker and linen)]:
   groups=['overall','split:'+row['split'],'family:'+row['source_family_key'],'context:'+row['controlled_context'],'variant:'+row['auxiliary_scalar_variant'],'length:'+str(len(vals))]
   for g in groups:rules[name+'|'+g]['rows']+=1;rules[name+'|'+g]['status_errors']+=predict!=empty
   if predict and len(t['worlds']) in (8,12):
    guess={'surviving_worlds':[],'belief_sizes':[1 if len(t['worlds'])==8 else 2]+[0]*(len(vals)-1),'queries':[{'query_id':q['query_id'],'counts':[],'distinct_values':0,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':False} for q in t['queries']]}
    rules[name+'|overall']['full_output_predicted_rows']+=1;rules[name+'|overall']['full_output_correct_rows']+=eq(guess,fs['expected'])
   else:rules[name+'|overall']['full_output_abstentions']+=1
  inverse={v:k for k,v in row['fact_role_assignments'].items()}
  for i,o in enumerate(t['observations']):
   for scope in ['overall','family:'+row['source_family_key'],'split:'+row['split'],'length:'+str(len(vals))]:
    for key in [('position_scalar',[i,o['equals']]),('role_scalar',[inverse[o['fact']],o['equals']]),('name_position_scalar',[o['fact'],i,o['equals']])]:scalarhist[canon([scope,*key])][status]+=1
  for v in {canon(v) for v in vals}:
   for scope in ['overall','family:'+row['source_family_key'],'split:'+row['split'],'length:'+str(len(vals))]:scalarhist[canon([scope,'anywhere_scalar',v])][status]+=1
  feat=features(rec);feature_rows.append((sid,row['split'],status,feat))
  for name,value in feat.items():buckets[name][canon(value)][status]+=1;by_split[name][row['split']][canon(value)][status]+=1
 lookups={}
 def winner(counter):return sorted(counter,key=lambda k:(-counter[k],k))[0]
 for name,table in buckets.items():
  train=by_split[name]['train'];dev=by_split[name]['dev_challenge'];covered=correct=0
  for key,c in dev.items():
   if key in train:covered+=sum(c.values());correct+=c.get(winner(train[key]),0)
  lookups[name]={'cohort_fitted_correct':sum(max(c.values()) for c in table.values()),'cohort_rows':3456,'distinct_feature_buckets':len(table),'mixed_buckets':sum(len(c)>1 for c in table.values()),
   'train_fitted_correct':sum(max(c.values()) for c in train.values()),'train_rows':1728,'exposed_dev_fitted_correct':sum(max(c.values()) for c in dev.values()),'exposed_dev_rows':1728,
   'train_lookup_on_exposed_dev_covered':covered,'train_lookup_on_exposed_dev_correct':correct,'train_lookup_on_exposed_dev_abstentions':1728-covered,
   'scope':'Finite fitted component-status audit, no learned model/full-target/generalization claim; labels used only to fit/analyze buckets, never as input features'}
 # Four valid-task column-frequency/length counterexamples to cohort fits.
 for fam in families:
  r=get(fam,'E','identity',0,'forward');t=r['task'];x=r['row']['fact_role_assignments']['observation_control'];y=r['row']['fact_role_assignments']['observation_auxiliary'];e=t['observations'][0]['equals'];b=t['observations'][1]['equals']
  left=next(w for w in t['worlds'] if eq(w['facts'][x],e));right=next(w for w in t['worlds'] if eq(w['facts'][y],b) and not eq(w['facts'][x],e))
  altered=copy.deepcopy(t);byid={w['world_id']:w for w in altered['worlds']};byid[left['world_id']]['facts'][y],byid[right['world_id']]['facts'][y]=right['facts'][y],left['facts'][y]
  out,p=own.solve(altered);oo,pp=own.solve_conjunction(altered);assert eq(out,oo) and eq(p,pp) and len(out['surviving_worlds'])==1 and eq(p[1],r['facts']['prefix_survivors'][1])
  beforecols={k:sorted(collections.Counter(canon(w['facts'][k]) for w in t['worlds']).items()) for k in t['worlds'][0]['facts']};aftercols={k:sorted(collections.Counter(canon(w['facts'][k]) for w in altered['worlds']).items()) for k in altered['worlds'][0]['facts']}
  assert eq(beforecols,aftercols) and eq(t['observations'],altered['observations']) and eq(t['queries'],altered['queries']) and len(canon(t))==len(canon(altered))
  counterexamples.append({'actual_expert_execution':ROLE,'source_sample_id':r['row']['sample_id'],'source_record_hash':r['row']['record_hash'],'changed_task':altered,'changed_world_ids':[left['world_id'],right['world_id']],
    'operation':'Swap only two y values between actual worlds; all typed column frequencies, ordered observations, queries, first-prefix IDs and canonical JSON character length stay fixed, empty becomes singleton nonempty.',
    'source_output_sha256':dig(r['facts']['expected']),'own_changed_output':out,'own_changed_prefixes':p,'new_corpus_record':False,'target_candidate_changed':False})
 counts.update({'records':len(records),'families':len(families),'conditional_groups':len(grouped),'histogram_axes':7*len(histogram_receipts),
  'inherited_claims_checked':len(inherited_receipts),'inherited_named_entries':len(resolver),'distinct_inherited_pairs':len({key[2] for key in resolver}),
  'nonmember_owner_inherited_claims':nonmembers,'component_decoder_tests':len(component_tests),'actual_supplied_probes_executed':len(probe_receipts),
  'same_observation_pairs_disjoint_record_count':len({i for e in ledger if e['name']=='same_observations_auxiliary_table_counterfactual' for i in e['pair_sample_ids']}),
  'decoded_identity_source_tables':len(identity_tables),'decoded_scalar_tables':len(decoded_tables),'visible_tables':len(visible_tables),'decoded_scalar_context_cases':len(decoded_cases),
  'column_frequency_counterexamples':len(counterexamples),'published_original_pair_entries':len(manifest['supplied_original_pairs'])})
 analysis={'report_type':'EXPERT_V4_POST_INPUT_FREEZE_CANONICAL_ANALYSIS','actual_author':ROLE,'reviewer_execution_id':ROLE,'created_utc':utc(),'candidate_sha256':CSHA,'manifest_sha256':MSHA,
  'original_first_judgments_preserved_sha256':BIND['reviews/expert/belief-gap-v4.canonical-first-judgments.jsonl'],'correction_scope':'diagnostic positional dictionary labels only; no candidate/target/claim changes',
  'pre_gold_prediction_sha256':BIND['reviews/expert/belief-gap-v4-input-attempt01.predictions.jsonl'],'exposure':EXPOSURE,'counts':dict(counts),'preservation_counts':dict(preservation),
  'actual_changed_top_level_fields_vs_v3':dict(metadata_diff),'added_semantic_families':0,'added_student_inputs':0 if preservation['semantic_task']==preservation['rendered_input']==3456 else None,
  'row_errors':{sid:es for sid,es in local_errors.items() if es},'batch_errors':batch_errors,'actual_attached_property_tag_counts':dict(propc),'profiles':profiles,
  'actual_null_family_category_support':{canon(k):v for k,v in null_contrasts.items()},
  'histogram_receipts':histogram_receipts,'semantic_pair_ledger':ledger,'inherited_owner_claim_receipts':inherited_receipts,'manifest_named_entry_usage_histogram':dict(collections.Counter(entry_uses.values())),
  'own_actual_probe_executions':probe_receipts,'decoder_component_tests':component_tests,'frozen_public_rule_actual_counts':dict(rules),'typed_observation_histograms':dict(scalarhist),
  'fitted_input_status_lookups':lookups,'independently_solved_frequency_counterexamples':counterexamples,'no_release_authority':True,'root_software_test_count_not_used_as_data_verdict':True}
 apath=OUT/(PREFIX+'.analysis.json');save(apath,analysis)
 first=OUT/'belief-gap-v4.canonical-corrective-attempt03.jsonl';verdicts=collections.Counter()
 with first.open('x',encoding='utf-8') as f:
  for sid,rec in records.items():
   row,fs=rec['row'],rec['facts'];errors=local_errors[sid];issues=[]
   for e in errors:issues.append({'category':'metadata_or_correctness','severity':'high','evidence':e,'fix':'Create a new candidate/version with the exact failed target/hash/metadata/control assertion corrected; preserve this initial candidate and receipt.'})
   issues.extend([{'category':'generalization','severity':'moderate','evidence':'Same V3 student input/target/split/difficulty/tags; four existing finite families and two exposed-dev families; field/scalar/name rotations add no independent mechanisms.','fix':'Report zero family/task growth and scope results to finite noiseless consistency filtering; use a separate public design for genuinely new mechanisms.'},
    {'category':'shortcut_resistance','severity':'moderate','evidence':'Cohort has75% nonempty and87.5% usefulness-false; repeated typed tables and fitted input associations remain. G/H can have one effective filter. Shared-model/public-guided/prior-gold exposure is disclosed.','fix':'Preserve measured component-status majority/fitted-lookup limits and G/H level2 scope; do not present control crossing as complete shortcut removal or sealed generalization.'}])
   verdict='REVISE' if errors else 'ACCEPT';verdicts[verdict]+=1
   relevant=[c for c in inherited_receipts if c['owner_sample_id']==sid]
   receipt={'receipt_id':'expert-belief-gap-v4-canonical-attempt03-'+sid,'sample_id':sid,'record_hash':row['record_hash'],'reviewer_role':'expert','reviewer_execution_id':ROLE,
    'review_status':'COMPLETED_CORRECTIVE_CANONICAL_EXPERT_JUDGMENT','scores':{'semantic_correctness':5 if not errors else 3,'target_correctness':5 if eq(fs['expected'],row['verified_target']) else 1,
     'instruction_clarity':5,'reasoning_validity':5,'generalization_value':2,'difficulty_quality':4 if not any('difficulty' in e for e in errors) else 2,'diversity':2,'shortcut_resistance':2},
    'critical_defects':errors,'issues':issues,'decision':verdict,'confidence':0.99 if not errors else 0.98,
    'evidence':{'candidate_sha256':CSHA,'manifest_sha256':MSHA,'own_pre_gold_prediction_sha256':BIND['reviews/expert/belief-gap-v4-input-attempt01.predictions.jsonl'],
     'full_record_hash_recomputed':rec['hash_ok'],'whole_typed_target_matches':eq(fs['expected'],row['verified_target']),'parsed_prompt_task_matches':True,
     'own_complete_target_sha256':dig(fs['expected']),'own_prefix_survivors_sha256':dig(fs['prefix_survivors']),'independent_actual_metrics':fs['metrics'],
     'all_attached_property_locators_checked':not any('property' in e or 'locator' in e or 'witness' in e for e in errors),'actual_property_names':[p for p,w in fs['properties'].items() if w],
     'owner_scoped_inherited_claims':relevant,'all_required_cohort_controls_analysis':str(apath.relative_to(ROOT)),'canonical_read_after_input_FINAL':True,
     'v3_unchanged_task_prompt_target_split_difficulty_tags_verified':not any('preservation' in e for e in errors),'declared_difficulty':row['difficulty'],
     'source_family_key':row['source_family_key'],'context':row['controlled_context'],'auxiliary_scalar_variant':row['auxiliary_scalar_variant'],'schedule_index':row['schedule_index'],'renderer':row['renderer']},
    'exposure':EXPOSURE,'no_dataset_publication_authority':True}
   f.write(canon(receipt)+'\n')
 after=bindings();assert before==after and hp(source)==pre['source']
 execution={'report_type':'EXPERT_V4_CANONICAL_ACTUAL_EXECUTION','actual_author':ROLE,'reviewer_execution_id':ROLE,'start_utc':start,'finish_utc':utc(),'elapsed_seconds':round(time.monotonic()-timer,6),
  'source_before_after':hp(source),'preexecution':hp(prepath),'bindings_before':before,'bindings_after':after,'outputs':{apath.name:hp(apath),first.name:hp(first)},
  'decisions':dict(verdicts),'counts':dict(counts),'row_error_count':sum(bool(v) for v in local_errors.values()),'batch_error_count':len(batch_errors),'ru_maxrss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'exit_code':0,'exposure':EXPOSURE,'canonical_corrective_judgments_frozen':True,'original_first_judgments_preserved':BIND['reviews/expert/belief-gap-v4.canonical-first-judgments.jsonl'],'publication_approval':False}
 ep=OUT/(PREFIX+'.execution.json');save(ep,execution);save(OUT/(PREFIX+'.first.freeze.json'),{'actual_author':ROLE,'reviewer_execution_id':ROLE,'created_utc':utc(),'source':hp(source),'execution':hp(ep),'analysis':hp(apath),'corrective_judgments':hp(first),'decisions':dict(verdicts),'no_release_approval':True})
 print(canon({'corrective_judgments':hp(first),'analysis':hp(apath),'execution':hp(ep),'decisions':dict(verdicts),'counts':dict(counts),'row_errors':sum(bool(v) for v in local_errors.values()),'batch_errors':len(batch_errors)}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','execute']);a=p.parse_args()
 if a.stage=='freeze':freeze()
 else:
  try:execute()
  except BaseException as e:
   save(OUT/(PREFIX+'.failure.json'),{'actual_author':ROLE,'reviewer_execution_id':ROLE,'created_utc':utc(),'source':hp(pathlib.Path(__file__).resolve()),'error_type':type(e).__name__,'error':str(e),'no_release_approval':True});raise
