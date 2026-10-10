#!/opt/anaconda3/bin/python3
"""Prospective, public-only V4 expert interface component toys.

Actual author /root/expert_reviewer. No production candidate/gold/manifest reads.
Reuses own frozen typed belief solver; prior canonical/model/design exposure.
This is not a complete production metadata decoder or whole-cohort approval.
"""
import pathlib,json,copy,hashlib,datetime,importlib.util,time,collections
ROOT=pathlib.Path(__file__).resolve().parents[2];ROLE='/root/expert_reviewer'
OWN='reviews/expert/prepare_belief_gap_v1.py'
BINDINGS={
 OWN:'e55b7ee49b25b0c4857a3387c730501bfda4ad0572906ce13be9c23a9c11d784',
 'schemas/EXTENDED_SEMANTICS.md':'2a67e05e10f99d1fd242ee1691d5b93755093fa701c48040c2d90f47422e7e23',
 'schemas/BELIEF_CONTROL_PROPERTIES.md':'200631973256bbe5dcd7b99f3d52cb95f89c442acd29ea4de1fe82dd8b5f3034',
 'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md':'46e3120dfd1ee931eec9d1e5e0e75f0b2aedb1d85764fd755afe9f7d36cf121f',
 'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md':'3a223952d9664577d7b637ddfb3e18d329863a3d5371a6c47cad233a1f50ff11',
 'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md':'40b552ecf47c11bfffb2669283640dda991d6dc6e7094ed3051fb9cad2489fce',
 'research/curricula/BELIEF-METADATA-V4-CLARIFICATION-EXECUTION.json':'2d100ce2ad27ded95275316991f3ec8ace9ef9daf2a40c50c3f06946c777ec0d',
 'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md':'925ccae4e6fdc79fec2d0778d871e5d316257c3026b59c6e166556da71eb346c',
 'research/curricula/BELIEF_METADATA_V4_ENCODING_ROOT_FREEZE.json':'2bbd9d577e0b7eb488143bd7f532f9ce3fd1c3c51be918d90470b29bc7b61625'}
def bind(p):
 b=(ROOT/p).read_bytes();return {'path':p,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
assert all(bind(p)['sha256']==h for p,h in BINDINGS.items())
sp=importlib.util.spec_from_file_location('own_original_public_belief_solver',ROOT/OWN);own=importlib.util.module_from_spec(sp);sp.loader.exec_module(own)
eq=own.equal;canon=own.canon
INHERITED={'same_prefix_empty_nonempty_counterfactual','contradiction_equals_mixed_outcomes'}
AGGREGATE={'auxiliary_scalar_transposition_invariance','same_observations_auxiliary_table_counterfactual','context_auxiliary_scalar_variant_crossing'}
ALLOWED={'name','property','scope','source_family_key','evidence_manifest','control_id','pair_sample_ids','probe_id'}
MINIMUM={'name','scope','source_family_key','auxiliary_scalar_variant','observation_length','control_id','pair_sample_ids','exact_candidate_sha256'}
COORDS=('source_family_key','auxiliary_scalar_variant','schedule_index','renderer')
MANIFEST='reviews/expert/belief-v4-prospective-handmade-toy-manifest.json'

def task_of(row):
 assert row['rendered_input'].count('Task JSON:\n')==1
 parsed=json.loads(row['rendered_input'].split('Task JSON:\n',1)[1])
 assert eq(parsed,row['semantic_task']);own.validate_task(parsed)
 observed='forward' if list(parsed)[0]=='kind' else 'reverse' if list(parsed)[0]=='queries' else None
 assert observed==row['renderer']
 assert row['record_hash']==hashlib.sha256(canon({k:v for k,v in row.items() if k!='record_hash'}).encode()).hexdigest()
 return parsed

def semantic_pair(left,right):
 a,b=task_of(left),task_of(right);n=len(a['observations'])
 assert n in (2,3) and len(b['observations'])==n
 assert (left['controlled_context'],right['controlled_context'])==(('E','G') if n==2 else ('F','H'))
 assert eq(a['worlds'],b['worlds']) and eq(a['queries'],b['queries'])
 assert a['observations'][0]['fact']!=a['observations'][1]['fact']
 changes=[i for i,(x,y) in enumerate(zip(a['observations'],b['observations'])) if not eq(x,y)]
 assert changes==[1]
 assert a['observations'][1]['fact']==b['observations'][1]['fact']
 assert not eq(a['observations'][1]['equals'],b['observations'][1]['equals'])
 if n==3:assert eq(a['observations'][2],a['observations'][0]) and eq(b['observations'][2],b['observations'][0])
 ao,ap=own.solve(a);bo,bp=own.solve(b);ad,adp=own.solve_conjunction(a);bd,bdp=own.solve_conjunction(b)
 assert eq(ao,ad) and eq(ap,adp) and eq(bo,bd) and eq(bp,bdp)
 assert ap[1] and eq(ap[1],bp[1]) and not ap[-1] and bp[-1]
 # Each nominated contradictory observation is independently supported.
 for ob in a['observations'][:2]:
  solo=copy.deepcopy(a);solo['observations']=[ob];assert own.solve(solo)[0]['surviving_worlds']
 return {'observation_length':n,'changed_observation_indices':[1],'first_prefix_ids':ap[1],
         'all_empty_prefix_ids':ap,'all_nonempty_prefix_ids':bp,
         'empty_complete_queries':ao['queries'],'nonempty_complete_queries':bo['queries']}

def review_claim_shape(claim,owner):
 assert type(claim) is dict and set(claim)<=ALLOWED
 assert ('name' in claim)+('property' in claim)==1
 name=claim.get('name',claim.get('property'));assert name in own.FAMILY_CONTROLS|INHERITED|AGGREGATE
 assert claim.get('scope') in ('corpus_variant','audit_probe')
 if 'source_family_key' in claim:assert claim['source_family_key']==owner['source_family_key']
 for key in ('evidence_manifest','control_id','probe_id'):
  if key in claim:assert type(claim[key]) is str and claim[key]
 if 'evidence_manifest' in claim:
  p=pathlib.PurePosixPath(claim['evidence_manifest']);assert not p.is_absolute() and '..' not in p.parts
 if 'pair_sample_ids' in claim:
  ids=claim['pair_sample_ids'];assert claim['scope']=='corpus_variant' and type(ids) is list and len(ids)==2 and all(type(i) is str and i for i in ids) and len(set(ids))==2
 if 'probe_id' in claim:assert claim['scope']=='audit_probe'
 if name in INHERITED and claim['scope']=='corpus_variant':assert {'pair_sample_ids','evidence_manifest','control_id'}<=set(claim)
 return name

def inherited_claim(claim,owner,records,manifests,candidate_sha):
 name=review_claim_shape(claim,owner);assert name in INHERITED and claim['scope']=='corpus_variant'
 assert owner['sample_id'] in records and eq(owner,records[owner['sample_id']]);ot=task_of(owner)
 ids=claim['pair_sample_ids'];assert all(i in records for i in ids)
 left,right=(records[i] for i in ids)
 assert all(eq(member[k],owner[k]) for member in (left,right) for k in COORDS)
 assert len(ot['observations'])==len(task_of(left)['observations'])==len(task_of(right)['observations'])
 facts=semantic_pair(left,right)
 path=claim['evidence_manifest'];assert path in manifests and (ROOT/path).is_file()
 manifest=manifests[path];collection=manifest['supplied_same_table_v2_pairs'];assert type(collection) is list
 matches=[e for e in collection if e.get('control_id')==claim['control_id'] and e.get('name')==name and eq(e.get('pair_sample_ids'),ids)]
 assert len(matches)==1;entry=matches[0];assert MINIMUM<=set(entry)
 assert entry['scope']=='corpus_variant' and entry['source_family_key']==owner['source_family_key']
 assert entry['auxiliary_scalar_variant']==owner['auxiliary_scalar_variant']
 assert type(entry['observation_length']) is int and entry['observation_length']==facts['observation_length']
 assert entry['exact_candidate_sha256']==candidate_sha
 # This handmade toy's chosen diagnostic spelling is NOT a public closed-schema key.
 assert eq(entry['toy_comparison_evidence'],facts)
 return facts

def transpose(task):
 t=copy.deepcopy(task);sw={'blocked':'supported','supported':'blocked'}
 for w in t['worlds']:w['facts']['y']=sw.get(w['facts']['y'],w['facts']['y'])
 for o in t['observations']:
  if o['fact']=='y':o['equals']=sw.get(o['equals'],o['equals'])
 return t
def reverse_objects(value):
 if type(value) is dict:return {k:reverse_objects(v) for k,v in reversed(list(value.items()))}
 if type(value) is list:return [reverse_objects(v) for v in value]
 return value
def fixtures():
 worlds=[{'world_id':i,'facts':{'x':x,'y':y,'a':a,'z':None}} for i,x,y,a in [('w0',0,'blocked',None),('w1',1,'supported',True),('w2',1,'supported',1),('w3',0,'supported',None),('wb',True,'supported',False)]]
 queries=[{'query_id':'q0','fact':'a'},{'query_id':'q1','fact':'x'},{'query_id':'q2','fact':'z'}]
 rows={}
 for variant in ('identity','transposed'):
  for style in ('forward','reverse'):
   for context in ('A','D','E','F','G','H'):
    observations=[{'fact':'x','equals':0 if context in ('A','D') else 1},{'fact':'y','equals':'supported' if context in ('G','H') else 'blocked'}]
    if context in ('D','F','H'):observations.append(copy.deepcopy(observations[0]))
    task={'kind':'belief','worlds':copy.deepcopy(worlds),'observations':observations,'queries':copy.deepcopy(queries)}
    if variant=='transposed':task=transpose(task)
    presentation=reverse_objects(task) if style=='reverse' else task
    sid='opaque '+context+' '+variant+' '+style
    row={'sample_id':sid,'source_family_key':'handmade component family','controlled_context':context,'auxiliary_scalar_variant':variant,'schedule_index':0,'renderer':style,'semantic_task':task,'rendered_input':'Handmade public component toy. Task JSON:\n'+json.dumps(presentation,ensure_ascii=False,separators=(',',':'))}
    row['record_hash']=hashlib.sha256(canon(row).encode()).hexdigest();rows[sid]=row
 raw=''.join(canon(r)+'\n' for r in rows.values()).encode();candidate_sha=hashlib.sha256(raw).hexdigest();entries=[];claims=[]
 for variant in ('identity','transposed'):
  for style in ('forward','reverse'):
   for owner_ctx,lctx,rctx in [('A','E','G'),('D','F','H')]:
    ids=['opaque '+x+' '+variant+' '+style for x in (lctx,rctx)];control='shared ID '+variant+' '+style+' length'+str(len(rows[ids[0]]['semantic_task']['observations']))
    facts=semantic_pair(rows[ids[0]],rows[ids[1]])
    for name in sorted(INHERITED):
     entry={'name':name,'scope':'corpus_variant','source_family_key':rows[ids[0]]['source_family_key'],'auxiliary_scalar_variant':variant,'observation_length':facts['observation_length'],'control_id':control,'pair_sample_ids':ids,'exact_candidate_sha256':candidate_sha,'toy_comparison_evidence':facts,'optional_diagnostic':'manifest is not closed; no mandatory pair_record_hashes'};entries.append(entry)
     claims.append(({'name':name,'scope':'corpus_variant','pair_sample_ids':ids,'evidence_manifest':MANIFEST,'control_id':control},'opaque '+owner_ctx+' '+variant+' '+style))
 return rows,raw,candidate_sha,{MANIFEST:{'supplied_same_table_v2_pairs':entries,'handmade_fixture_only':True}},claims

def run():
 started=time.monotonic();before={p:bind(p) for p in BINDINGS};rows,raw,candidate_sha,manifests,claims=fixtures();tests=[]
 fixture_path='reviews/expert/belief-v4-prospective-handmade-toy-inputs.jsonl'
 with (ROOT/fixture_path).open('xb') as output:output.write(raw)
 with (ROOT/MANIFEST).open('x') as output:json.dump(manifests[MANIFEST],output,indent=2);output.write('\n')
 assert bind(fixture_path)['sha256']==candidate_sha
 def positive(name,fn):
  try:result=fn();tests.append({'name':name,'expected':'accept_component','pass':True,'result':result})
  except Exception as e:tests.append({'name':name,'expected':'accept_component','pass':False,'actual_exception':type(e).__name__+': '+str(e)})
 def negative(name,fn):
  try:fn();tests.append({'name':name,'expected':'reject_component','pass':False,'actual_exception':None})
  except (AssertionError,ValueError,KeyError,TypeError) as e:tests.append({'name':name,'expected':'reject_component','pass':True,'actual_exception':type(e).__name__+': '+str(e)})
 for i,(claim,sid) in enumerate(claims):
  positive('ordered_same_family_pair_nonmember_owner_'+str(i),lambda claim=claim,sid=sid:inherited_claim(claim,rows[sid],rows,manifests,candidate_sha))
 positive('actual_toy_manifest_file_parsed_exact',lambda:assertion(eq(json.loads((ROOT/MANIFEST).read_text()),manifests[MANIFEST])))
 base,ownerid=claims[0];owner=rows[ownerid]
 for key in ('pair_sample_ids','evidence_manifest','control_id'):
  bad=copy.deepcopy(base);del bad[key];negative('missing_required_claim_'+key,lambda b=bad:inherited_claim(b,owner,rows,manifests,candidate_sha))
 for key,value in [('pair_sample_ids',list(reversed(base['pair_sample_ids']))),('pair_sample_ids',[base['pair_sample_ids'][0],base['pair_sample_ids'][0]]),('pair_sample_ids',['absent',base['pair_sample_ids'][1]]),('scope','audit_probe'),('evidence_manifest','/absolute/path'),('evidence_manifest','../outside'),('undocumented_status','empty'),('property',base['name'])]:
  bad=copy.deepcopy(base);bad[key]=value;negative('claim_wrong_'+key+'_'+str(value),lambda b=bad:inherited_claim(b,owner,rows,manifests,candidate_sha))
 for coord,value in [('source_family_key','foreign family'),('auxiliary_scalar_variant','transposed'),('schedule_index',1),('renderer','reverse')]:
  rr=copy.deepcopy(rows);rr[ownerid][coord]=value;rr[ownerid]['record_hash']=hashlib.sha256(canon({k:v for k,v in rr[ownerid].items() if k!='record_hash'}).encode()).hexdigest()
  negative('owner_mismatch_'+coord,lambda rr=rr:inherited_claim(base,rr[ownerid],rr,manifests,candidate_sha))
 # A same-length family claim is allowed even when owner is E or G itself.
 positive('pair_member_owner_also_permitted',lambda:inherited_claim(base,rows[base['pair_sample_ids'][0]],rows,manifests,candidate_sha))
 property_claim=copy.deepcopy(base);property_claim['property']=property_claim.pop('name');positive('property_alias_exclusively_permitted',lambda:inherited_claim(property_claim,owner,rows,manifests,candidate_sha))
 wrong_length=copy.deepcopy(base);wrong_length['pair_sample_ids']=[i.replace('opaque E ','opaque F ').replace('opaque G ','opaque H ') for i in base['pair_sample_ids']]
 negative('length3_pair_cannot_replace_length2_owner',lambda:inherited_claim(wrong_length,owner,rows,manifests,candidate_sha))
 for key in sorted(MINIMUM):
  mm=copy.deepcopy(manifests);entry=next(e for e in mm[MANIFEST]['supplied_same_table_v2_pairs'] if e['name']==base['name'] and e['control_id']==base['control_id']);del entry[key]
  negative('missing_manifest_minimum_'+key,lambda mm=mm:inherited_claim(base,owner,rows,mm,candidate_sha))
 for label,mutate in [('wrong_candidate_hash',lambda e:e.update(exact_candidate_sha256='0'*64)),('wrong_named_entry',lambda e:e.update(name='auxiliary_scalar_transposition_invariance')),('wrong_pair_order',lambda e:e.update(pair_sample_ids=list(reversed(e['pair_sample_ids'])))),('bool_length',lambda e:e.update(observation_length=True)),('false_prefix_evidence',lambda e:e['toy_comparison_evidence'].update(first_prefix_ids=['w0'])),('wrong_variant',lambda e:e.update(auxiliary_scalar_variant='transposed'))]:
  mm=copy.deepcopy(manifests);entry=next(e for e in mm[MANIFEST]['supplied_same_table_v2_pairs'] if e['name']==base['name'] and e['control_id']==base['control_id']);mutate(entry)
  negative('entry_'+label,lambda mm=mm:inherited_claim(base,owner,rows,mm,candidate_sha))
 mm=copy.deepcopy(manifests);entry=next(e for e in mm[MANIFEST]['supplied_same_table_v2_pairs'] if e['name']==base['name'] and e['control_id']==base['control_id']);mm[MANIFEST]['supplied_same_table_v2_pairs'].append(copy.deepcopy(entry))
 negative('duplicate_exact_named_resolver_key',lambda:inherited_claim(base,owner,rows,mm,candidate_sha))
 for label,mutate in [('change_first_observation',lambda t:t['observations'][0].update(equals=True)),('change_query_order',lambda t:t['queries'].reverse()),('change_world_query_value',lambda t:t['worlds'][1]['facts'].update(a=1)),('change_only_annotation_not_task',lambda t:t['observations'][1].update(equals='blocked'))]:
  rr=copy.deepcopy(rows);right=base['pair_sample_ids'][1];mutate(rr[right]['semantic_task']);t=rr[right]['semantic_task'];rr[right]['rendered_input']='Handmade public component toy. Task JSON:\n'+json.dumps(t,separators=(',',':'));rr[right]['record_hash']=hashlib.sha256(canon({k:v for k,v in rr[right].items() if k!='record_hash'}).encode()).hexdigest()
  negative('actual_semantic_'+label,lambda rr=rr:inherited_claim(base,owner,rr,manifests,candidate_sha))
 # Three aggregate identifiers may omit pair IDs syntactically; none is semantically
 # accepted by this local inherited-pair component. Whole groups remain deferred.
 for name in sorted(AGGREGATE):
  claim={'name':name,'scope':'corpus_variant','evidence_manifest':MANIFEST,'control_id':'aggregate group'}
  positive('aggregate_shape_allowed_pending_full_group_'+name,lambda claim=claim:review_claim_shape(claim,owner))
 probe={'name':base['name'],'scope':'audit_probe','probe_id':'not executed'}
 negative('probe_cannot_replace_required_supplied_pair',lambda:inherited_claim(probe,owner,rows,manifests,candidate_sha))
 # Independent hand-derived full typed outputs, not fixture gold copied from solve.
 gt=rows['opaque G identity forward']['semantic_task'];gout,gprefix=own.solve(gt)
 hand={'surviving_worlds':['w1','w2'],'belief_sizes':[2,2],'queries':[{'query_id':'q0','counts':[{'value':1,'count':1},{'value':True,'count':1}],'distinct_values':2,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':True},{'query_id':'q1','counts':[{'value':1,'count':2}],'distinct_values':1,'entailed':True,'guaranteed':1,'can_reduce_uncertainty':False},{'query_id':'q2','counts':[{'value':None,'count':2}],'distinct_values':1,'entailed':True,'guaranteed':None,'can_reduce_uncertainty':False}]}
 positive('hand_full_typed_G_target_null_context_and_bool_int',lambda:assertion(eq(gout,hand) and gprefix[1]==['w1','w2']))
 positive('typed_boolean_integer_join_distinct',lambda:assertion(not eq({'equals':True},{'equals':1})))
 positive('fact_local_transposition_involutive_full_target',lambda:assertion(eq(transpose(transpose(gt)),gt) and eq(own.solve(transpose(gt)),own.solve(gt))))
 e=rows['opaque E identity forward']['semantic_task'];tg=rows['opaque G transposed forward']['semantic_task']
 positive('same_observations_different_table_opposite_status',lambda:assertion(eq(e['observations'],tg['observations']) and eq(e['queries'],tg['queries']) and not eq(e['worlds'],tg['worlds']) and not own.solve(e)[0]['surviving_worlds'] and bool(own.solve(tg)[0]['surviving_worlds'])))
 after={p:bind(p) for p in BINDINGS};assert before==after
 report={'actual_author':ROLE,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Prospective handmade V4 inherited-interface component preparation only; no production inputs/gold/manifests inspected and no candidate verdict','source':bind(str(pathlib.Path(__file__).relative_to(ROOT))),'bindings_before':before,'bindings_after':after,'unchanged':True,'actual_tests':len(tests),'passed':sum(t['pass'] for t in tests),'failed':sum(not t['pass'] for t in tests),'tests':tests,'handmade_fixture_rows':len(rows),'toy_candidate_sha256':candidate_sha,'toy_candidate_bytes':len(raw),'actual_toy_input':bind(fixture_path),'actual_toy_manifest':bind(MANIFEST),'prospective_full_cohort_certified':False,'toy_protocol_choice':'toy_comparison_evidence is my fixture representation of required actual parsed/prefix evidence, NOT a published closed field or future production decoder assumption. Additional manifest fields allowed; pair_record_hashes deliberately absent. Negative resolver probes use isolated in-memory mutation copies of the actual base fixture file. Schedule/profile/group completeness is separately specified in the procedure, not established by these local fixtures.','aggregate_controls':'All three permitted aggregate claim shapes are recognized; full-group semantics are deferred, not passed from pointers.','actual_elapsed_seconds':time.monotonic()-started,'actual_exit_code':0,'prior_original_canonical_exposure':True,'own_original_solver_reuse':True,'common_model_public_guided_risk':True,'public_docs_summarize_prior_adjudication':True,'other_private_review_formal_producer_future_candidate_read':False,'no_dataset_or_publication_approval':True}
 report['actual_exit_code']=0 if report['failed']==0 else 1
 for path,value in [('reviews/expert/belief-v4-prospective-toy-execution.json',report)]:
  with (ROOT/path).open('x') as output:json.dump(value,output,indent=2);output.write('\n')
 print(json.dumps({'execution':bind('reviews/expert/belief-v4-prospective-toy-execution.json'),'tests':len(tests),'passed':report['passed'],'failed':report['failed']}))
 if report['failed']:raise SystemExit(1)
def assertion(value):
 assert value
 return True
if __name__=='__main__':run()
