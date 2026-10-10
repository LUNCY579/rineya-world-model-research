#!/usr/bin/env python3
"""Public-only handmade V4 preparation, actual author /root/adversarial_auditor.
No actual belief candidate/manifest, producer/formal implementation or other verdict read.
Copied only own typed solver and hand-derived toys; fresh resolver and set-filter authored here.
"""
import collections,copy,hashlib,itertools,json,pathlib,re,time
ROOT=pathlib.Path(__file__).resolve().parents[1];EXECUTION='/root/adversarial_auditor'
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(v):return hashlib.sha256(canon(v).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(v,msg):
 if not v:raise ValueError(msg)
def jb(v):return (json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()
def freeze(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():require(p.read_bytes()==data,'immutable artifact differs '+str(p))
 else:p.write_bytes(data)
def scalar(v): return v is None or type(v) in (str,int,bool)
def typed(v): require(scalar(v),'Invalid JSON scalar'); return canon(v)
def payload_count(t,ids): return len({canon(w['facts']) for w in t['worlds'] if w['world_id'] in ids})
def partition(t,ids,f):
    counts=collections.Counter(typed(w['facts'][f]) for w in t['worlds'] if w['world_id'] in ids)
    return [{'value':json.loads(v),'count':counts[v]} for v in sorted(counts)]
def answer(q,bins,n):
    entailed=bool(n) and len(bins)==1
    return {'query_id':q['query_id'],'counts':bins,'distinct_values':len(bins),'entailed':entailed,'guaranteed':bins[0]['value'] if entailed else None,'can_reduce_uncertainty':len(bins)>1}

def own_prior_bitmask_solve(t):
    # Own third implementation uses typed equality keys and a prefix bitmask over world positions.
    require(type(t) is dict and set(t)=={'kind','worlds','observations','queries'} and t['kind']=='belief','Belief shape')
    worlds,obs,queries=t['worlds'],t['observations'],t['queries']
    require(all(type(x) is list for x in (worlds,obs,queries)),'Lists')
    require(all(type(w) is dict and set(w)=={'world_id','facts'} and type(w['world_id']) is str and type(w['facts']) is dict for w in worlds),'World shape')
    fields=set(worlds[0]['facts']) if worlds else set();require(len({w['world_id'] for w in worlds})==len(worlds),'Unique worlds')
    require(all(set(w['facts'])==fields and all(type(f) is str and scalar(v) for f,v in w['facts'].items()) for w in worlds),'Typed consistent facts')
    require(all(type(o) is dict and set(o)=={'fact','equals'} and type(o['fact']) is str and o['fact'] in fields and scalar(o['equals']) for o in obs),'Observations')
    require(all(type(q) is dict and set(q)=={'query_id','fact'} and type(q['query_id']) is str and q['fact'] in fields for q in queries),'Queries');require(len({q['query_id'] for q in queries})==len(queries),'Unique queries')
    def equal(v,u):return (type(v),v)==(type(u),u)
    mask=(1<<len(worlds))-1
    def ids(mask):return sorted(w['world_id'] for j,w in enumerate(worlds) if mask&(1<<j))
    prefixes=[ids(mask)];individual=[]
    for o in obs:
        support=sum(1<<j for j,w in enumerate(worlds) if equal(w['facts'][o['fact']],o['equals']))
        individual.append(ids(support));mask&=support;prefixes.append(ids(mask))
    def count_at(mask,f):
        bins=collections.Counter(canon(w['facts'][f]) for j,w in enumerate(worlds) if mask&(1<<j))
        return [{'value':json.loads(k),'count':bins[k]} for k in sorted(bins)]
    final=ids(mask);outputs=[]
    for q in queries:
        bins=count_at(mask,q['fact']);ent=bool(mask) and len(bins)==1
        outputs.append({'query_id':q['query_id'],'counts':bins,'distinct_values':len(bins),'entailed':ent,'guaranteed':bins[0]['value'] if ent else None,'can_reduce_uncertainty':len(bins)>1})
    result={'surviving_worlds':final,'belief_sizes':[len(x) for x in prefixes[1:]],'queries':outputs}
    detail={'prefix_survivors':prefixes,'individual_observation_supports':individual,'prefix_query_partitions':[[{'query_id':q['query_id'],'fact':q['fact'],'bins':partition(t,p,q['fact'])} for q in queries] for p in prefixes],'world_count':len(worlds),'fact_count':len(fields),'observation_count':len(obs),'query_count':len(queries),'distinct_complete_world_payloads':payload_count(t,prefixes[0])}
    return result,detail

def toys():
    cases=[]
    def probe(name,worlds,obs,qs,expected):
        t={'kind':'belief','worlds':[{'world_id':i,'facts':f} for i,f in worlds],'observations':obs,'queries':qs}
        actual,detail=solve(t);require(canon(actual)==canon(expected),'Toy '+name);cases.append({'name':name,'task':t,'expected_target':expected,'actual_target':actual,'prefix_survivors':detail['prefix_survivors'],'passed':True})
    q=[{'query_id':'q','fact':'v'}]
    probe('boolean_true_is_not_integer_one',[('a',{'x':True,'v':'yes'}),('b',{'x':1,'v':'no'})],[{'fact':'x','equals':True}],q,{'surviving_worlds':['a'],'belief_sizes':[1],'queries':[{'query_id':'q','counts':[{'value':'yes','count':1}],'distinct_values':1,'entailed':True,'guaranteed':'yes','can_reduce_uncertainty':False}]})
    probe('boolean_false_is_not_integer_zero',[('a',{'x':False,'v':False}),('b',{'x':0,'v':0})],[{'fact':'x','equals':0}],q,{'surviving_worlds':['b'],'belief_sizes':[1],'queries':[{'query_id':'q','counts':[{'value':0,'count':1}],'distinct_values':1,'entailed':True,'guaranteed':0,'can_reduce_uncertainty':False}]})
    probe('typed_counts_multiplicity',[('a',{'v':True}),('b',{'v':1}),('c',{'v':1}),('d',{'v':None})],[],q,{'surviving_worlds':['a','b','c','d'],'belief_sizes':[],'queries':[{'query_id':'q','counts':[{'value':1,'count':2},{'value':None,'count':1},{'value':True,'count':1}],'distinct_values':3,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':True}]})
    probe('entailed_null_multiworld',[('a',{'x':0,'v':None}),('b',{'x':1,'v':None})],[],q,{'surviving_worlds':['a','b'],'belief_sizes':[],'queries':[{'query_id':'q','counts':[{'value':None,'count':2}],'distinct_values':1,'entailed':True,'guaranteed':None,'can_reduce_uncertainty':False}]})
    probe('empty_absorbs_individually_supported_repeat',[('a',{'x':'stone','y':'silk','v':1}),('b',{'x':'reed','y':'linen','v':2})],[{'fact':'x','equals':'stone'},{'fact':'y','equals':'linen'},{'fact':'x','equals':'stone'}],q,{'surviving_worlds':[],'belief_sizes':[1,0,0],'queries':[{'query_id':'q','counts':[],'distinct_values':0,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':False}]})
    probe('stone_and_linen_supported_not_empty',[('a',{'x':'stone','y':'linen','v':'unknown'}),('b',{'x':'reed','y':'silk','v':None})],[{'fact':'x','equals':'stone'},{'fact':'y','equals':'linen'}],q,{'surviving_worlds':['a'],'belief_sizes':[1,1],'queries':[{'query_id':'q','counts':[{'value':'unknown','count':1}],'distinct_values':1,'entailed':True,'guaranteed':'unknown','can_reduce_uncertainty':False}]})
    probe('reorder_changes_trace_preserves_final',[('a',{'x':'r','y':'s','v':0}),('b',{'x':'r','y':'t','v':1}),('c',{'x':'q','y':'t','v':2})],[{'fact':'y','equals':'t'},{'fact':'x','equals':'q'}],q,{'surviving_worlds':['c'],'belief_sizes':[2,1],'queries':[{'query_id':'q','counts':[{'value':2,'count':1}],'distinct_values':1,'entailed':True,'guaranteed':2,'can_reduce_uncertainty':False}]})
    bads=[]
    for name,t in [('float_scalar',{'kind':'belief','worlds':[{'world_id':'a','facts':{'x':1.0}}],'observations':[],'queries':[]}),('duplicate_world',{'kind':'belief','worlds':[{'world_id':'a','facts':{'x':1}},{'world_id':'a','facts':{'x':2}}],'observations':[],'queries':[]}),('missing_fact',{'kind':'belief','worlds':[{'world_id':'a','facts':{'x':1}}],'observations':[{'fact':'y','equals':1}],'queries':[]})]:
        try:solve(t)
        except Exception as e:bads.append({'name':name,'task':t,'rejected':True,'error':str(e)})
        else:raise ValueError('Malformed probe accepted '+name)
    return {'manually_authored_exact_toys':cases,'malformed_probes':bads}



PUBLIC={'schemas/EXTENDED_SEMANTICS.md':'2a67e05e10f99d1fd242ee1691d5b93755093fa701c48040c2d90f47422e7e23','schemas/BELIEF_CONTROL_PROPERTIES.md':'200631973256bbe5dcd7b99f3d52cb95f89c442acd29ea4de1fe82dd8b5f3034','schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md':'46e3120dfd1ee931eec9d1e5e0e75f0b2aedb1d85764fd755afe9f7d36cf121f','schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md':'3a223952d9664577d7b637ddfb3e18d329863a3d5371a6c47cad233a1f50ff11','research/curricula/RBM-EVIDENCE-GAP.md':'004fb0ec0c1a1104ba052fbda4882f48523d7d34dd45b2b88e8c890713b4a120','research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md':'c2eb841ed546848088db4e0499e8e8f7087aa3d213af333ff39237ae6558bd33','research/curricula/RBM-EVIDENCE-GAP-REVISION-V3.md':'7e9c9f9160425ab7e455f57d72385e54f399835dfa351dcf75f6fc2842b1dc33','schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md':'40b552ecf47c11bfffb2669283640dda991d6dc6e7094ed3051fb9cad2489fce','research/curricula/BELIEF-METADATA-V4-CLARIFICATION-EXECUTION.json':'2d100ce2ad27ded95275316991f3ec8ace9ef9daf2a40c50c3f06946c777ec0d','schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md':'925ccae4e6fdc79fec2d0778d871e5d316257c3026b59c6e166556da71eb346c','research/curricula/BELIEF_METADATA_V4_ENCODING_ROOT_FREEZE.json':'2bbd9d577e0b7eb488143bd7f532f9ce3fd1c3c51be918d90470b29bc7b61625'}
OWN='audits/belief_gap_v3_adversarial_audit.py';OWN_HASH='6feaa467cb7049cfe01928feba3dee6516449a7d42a2525b480f6dc06f393725'
RECIPE='reviews/adversarial/belief-v4-prospective-fixtures-r1.json';RECIPE_HASH='27b5296bbe48be4f94a962a2ec5918811d25e285988e2672a8c3cd76bc96bcda'
MANIFEST_PATH='reviews/adversarial/belief-v4-prospective-handmade-manifest-r1.json'
INHERITED={'same_prefix_empty_nonempty_counterfactual','contradiction_equals_mixed_outcomes'}
AGGREGATE={'auxiliary_scalar_transposition_invariance','same_observations_auxiliary_table_counterfactual','context_auxiliary_scalar_variant_crossing'}
CLAIM_FIELDS={'name','property','scope','source_family_key','evidence_manifest','control_id','pair_sample_ids','probe_id'}
ENTRY_FIELDS={'name','scope','source_family_key','auxiliary_scalar_variant','observation_length','control_id','pair_sample_ids','exact_candidate_sha256'}
ROLES=('invariant_query','variation_query','observed_control')
DISCLOSURE='Prospective public-guided handmade preparation only; prior original belief V1/V2/V3 gold/metadata exposure and own previous solver/toy reuse. Common model/design and directed revision context correlate this execution. No actual V4 dataset/gold/producer/formal/other reviewer inspection or approval.'
def solve(t):
 # New direct set filtering independently agrees with our prior positional bitmask implementation.
 require(type(t) is dict and set(t)=={'kind','worlds','observations','queries'} and t['kind']=='belief','typed belief shape')
 worlds,obs,qs=t['worlds'],t['observations'],t['queries'];require(all(type(x) is list for x in (worlds,obs,qs)),'typed lists')
 require(all(type(w) is dict and set(w)=={'world_id','facts'} and type(w['world_id']) is str and type(w['facts']) is dict for w in worlds),'world records')
 facts=set(worlds[0]['facts']) if worlds else set();require(len({w['world_id'] for w in worlds})==len(worlds),'distinct world IDs')
 require(all(set(w['facts'])==facts and all(type(k) is str and scalar(v) for k,v in w['facts'].items()) for w in worlds),'typed fact maps')
 require(all(type(o) is dict and set(o)=={'fact','equals'} and type(o['fact']) is str and o['fact'] in facts and scalar(o['equals']) for o in obs),'typed observations')
 require(all(type(q) is dict and set(q)=={'query_id','fact'} and type(q['query_id']) is str and type(q['fact']) is str and q['fact'] in facts for q in qs),'typed queries');require(len({q['query_id'] for q in qs})==len(qs),'distinct query IDs')
 current={w['world_id'] for w in worlds};prefixes=[sorted(current)];individual=[]
 for o in obs:
  support={w['world_id'] for w in worlds if typed(w['facts'][o['fact']])==typed(o['equals'])};individual.append(sorted(support));current&=support;prefixes.append(sorted(current))
 result={'surviving_worlds':sorted(current),'belief_sizes':[len(p) for p in prefixes[1:]],'queries':[answer(q,partition(t,current,q['fact']),len(current)) for q in qs]}
 return result,{'prefix_survivors':prefixes,'individual_observation_supports':individual}
def record_hash(r):return digest({k:v for k,v in r.items() if k!='record_hash'})
def raw_candidate(rows):return ''.join(canon(r)+'\n' for r in rows.values()).encode()
def candidate_hash(rows):return hashlib.sha256(raw_candidate(rows)).hexdigest()
def reverse_fields(v):
 if type(v) is dict:return {k:reverse_fields(x) for k,x in reversed(list(v.items()))}
 if type(v) is list:return [reverse_fields(x) for x in v]
 return v

def decode(r):
 t=r['semantic_task'];solve(t);require(r['record_hash']==record_hash(r),'actual full record hash')
 require(type(r['source_family_key']) is str and bool(r['source_family_key']),'family string')
 require(r['controlled_context'] in 'ABCDEFGH' and len(r['controlled_context'])==1,'A-H context')
 require(r['auxiliary_scalar_variant'] in ('identity','transposed') and r['renderer'] in ('forward','reverse'),'variant/style')
 vals=[r[k] for k in ('id_shift','position_shift','fact_name_shift')];require(all(type(x) is int and 0<=x<=2 for x in vals),'typed shifts');u,v,w=vals
 require(type(r['schedule_index']) is int and r['schedule_index']==9*u+3*v+w,'exact schedule equation')
 for key in ('query_id_pool','queried_fact_name_pool'):
  require(type(r[key]) is list and len(r[key])==3 and all(type(s) is str for s in r[key]) and len(set(r[key]))==3,'ordered distinct pools')
 rolemap=r['fact_role_assignments'];required={'observation_control','observation_auxiliary','invariant_query','variation_query','unoffered_feature'}
 require(type(rolemap) is dict and required<=set(rolemap) and all(k in required or re.fullmatch(r'aux_extra_[0-9]+',k) for k in rolemap),'declared roles')
 require(all(type(x) is str for x in rolemap.values()) and len(set(rolemap.values()))==len(rolemap),'typed role bijection')
 require(set(rolemap.values())==set(t['worlds'][0]['facts']),'exact visible fact roles')
 assignments=r['query_role_assignments'];require(type(assignments) is list and len(assignments)==3 and {a['role'] for a in assignments}==set(ROLES),'three query roles')
 for a in assignments:
  require(type(a) is dict and {'role','position','query_id'}<=set(a)<= {'role','position','query_id','fact'},'assignment public shape')
  i=ROLES.index(a['role']);pos=(i+v)%3;name=r['queried_fact_name_pool'][(i+w)%3];qid=r['query_id_pool'][(i+u)%3]
  require(type(a['position']) is int and a['position']==pos and a['query_id']==qid,'decoded position/id')
  require(t['queries'][pos]=={'query_id':qid,'fact':name},'actual decoded query')
  frole='observation_control' if a['role']=='observed_control' else a['role'];require(rolemap[frole]==name and ('fact' not in a or a['fact']==name),'coherent fact rename')
 control=r['auxiliary_scalar_control'];require(type(control) is dict and set(control)=={'fact','baseline_value','supported_value'},'control public shape')
 require(control['fact']==rolemap['observation_auxiliary'] and control['fact'] not in {q['fact'] for q in t['queries']},'auxiliary unqueried fact')
 require(type(control['baseline_value']) is type(control['supported_value']) is str and control['baseline_value']!=control['supported_value'],'distinct string transposition')
 require(control['baseline_value']=='linen' and control['supported_value'] not in {'linen','stone','unknown'},'targeted public literal bounds')
 return t

def pair_fact_evidence(left,right):
 lt,rt=decode(left),decode(right);la,ld=solve(lt);ra,rd=solve(rt)
 require(canon(lt['worlds'])==canon(rt['worlds']) and canon(lt['queries'])==canon(rt['queries']),'typed same worlds and ordered queries')
 require(len(lt['observations'])==len(rt['observations']) in (2,3),'same actual length2/3')
 differences=[(i,k) for i,(a,b) in enumerate(zip(lt['observations'],rt['observations'])) for k in set(a)|set(b) if canon(a.get(k))!=canon(b.get(k))]
 require(differences==[(1,'equals')],'only observation1 typed equals differs')
 require(canon(lt['observations'][0])==canon(rt['observations'][0]) and ld['prefix_survivors'][1]==rd['prefix_survivors'][1],'matched exact first prefix')
 require(not la['surviving_worlds'] and bool(ra['surviving_worlds']),'empty-first/nonempty-second')
 require(lt['observations'][0]['fact']!=lt['observations'][1]['fact'] and all(ld['individual_observation_supports'][i] for i in (0,1)),'individually supported cross-fact contradiction')
 return {'pair_sample_ids':[left['sample_id'],right['sample_id']],'actual_observation_length':len(lt['observations']),'typed_changed_fields':differences,'first_prefix_world_ids':ld['prefix_survivors'][1],'empty_prefix_ids':ld['prefix_survivors'],'nonempty_prefix_ids':rd['prefix_survivors'],'empty_complete_target':la,'nonempty_complete_target':ra}

def resolve(owner,claim,rows,manifest,exact_hash,manifest_path=MANIFEST_PATH):
 require(type(claim) is dict and set(claim)<=CLAIM_FIELDS and len(set(claim)&{'name','property'})==1 and 'scope' in claim,'closed canonical claim vocabulary')
 name=claim.get('name',claim.get('property'));require(name in INHERITED,'inherited supplied resolver identifier')
 require(claim['scope']=='corpus_variant' and 'probe_id' not in claim,'audit probe cannot replace supplied pair')
 require({'pair_sample_ids','evidence_manifest','control_id'}<=set(claim),'specialized inline pair/pointer/id required')
 require(type(claim['evidence_manifest']) is str and claim['evidence_manifest']==manifest_path and not pathlib.PurePosixPath(claim['evidence_manifest']).is_absolute(),'actual relative manifest resolution')
 require(type(claim['control_id']) is str and bool(claim['control_id']),'control descriptive string')
 require('source_family_key' not in claim or claim['source_family_key']==owner['source_family_key'],'optional owner family')
 ids=claim['pair_sample_ids'];require(type(ids) is list and len(ids)==2 and all(type(x) is str for x in ids) and len(set(ids))==2 and all(x in rows for x in ids),'two actual distinct pair IDs')
 require(owner['sample_id'] in rows and canon(owner)==canon(rows[owner['sample_id']]),'actual owner binding')
 ot=decode(owner);left,right=[rows[x] for x in ids]
 for r in (left,right):
  decode(r)
  for k in ('source_family_key','auxiliary_scalar_variant','schedule_index','renderer'):require(type(r[k]) is type(owner[k]) and r[k]==owner[k],'owner coordinate '+k)
 length=len(ot['observations']);want=('E','G') if length==2 else ('F','H') if length==3 else None
 require(want is not None and (left['controlled_context'],right['controlled_context'])==want,'owner actual length chooses E/G or F/H')
 require(all(len(r['semantic_task']['observations'])==length for r in (left,right)),'pair and owner actual lengths')
 require(type(manifest) is dict and type(manifest.get('supplied_same_table_v2_pairs')) is list,'published collection, open top shape')
 matches=[e for e in manifest['supplied_same_table_v2_pairs'] if e.get('control_id')==claim['control_id'] and e.get('name')==name and canon(e.get('pair_sample_ids'))==canon(ids)]
 require(len(matches)==1,'exact one ID+name+orderedpair lookup');entry=matches[0]
 require(ENTRY_FIELDS<=set(entry),'published minimum entry fields, open extra fields')
 require(entry['scope']=='corpus_variant' and entry['source_family_key']==owner['source_family_key'] and entry['auxiliary_scalar_variant']==owner['auxiliary_scalar_variant'],'resolved entry scope/family/variant')
 require(type(entry['observation_length']) is int and entry['observation_length']==length,'actual typed entry length')
 require(entry['exact_candidate_sha256']==exact_hash and exact_hash==candidate_hash(rows),'exact complete handmade candidate hash')
 evidence=pair_fact_evidence(left,right)
 return {'owner_sample_id':owner['sample_id'],'owner_not_pair_member':owner['sample_id'] not in ids,'name':name,'control_id':entry['control_id'],'derived_pair_evidence':evidence,'supplied_pair_credit':True}


def make_fixture(recipe):
 rows={};coordinate={};queries=recipe['query_names'];ids=recipe['query_ids'];fields={'invariant_query':'q0','variation_query':'q1','observed_control':'x'}
 for ctx,variant,u,v,w,style in itertools.product('ABCDEFGH',('identity','transposed'),range(3),range(3),range(3),('forward','reverse')):
  mp={fields[role]:queries[(i+w)%3] for i,role in enumerate(ROLES)};mp.update(y='y',u='u')
  swap=lambda x:recipe['supported_value'] if x==recipe['baseline_value'] else recipe['baseline_value'] if x==recipe['supported_value'] else x
  worlds=copy.deepcopy(recipe['worlds']);observations=copy.deepcopy(recipe['observations'][ctx])
  if variant=='transposed':
   for world in worlds:world['facts']['y']=swap(world['facts']['y'])
   for o in observations:
    if o['fact']=='y':o['equals']=swap(o['equals'])
  for world in worlds:world['facts']={mp[k]:val for k,val in world['facts'].items()}
  for o in observations:o['fact']=mp[o['fact']]
  q=[None]*3;qa=[]
  for i,role in enumerate(ROLES):
   pos=(i+v)%3;qid=ids[(i+u)%3];name=mp[fields[role]];q[pos]={'query_id':qid,'fact':name};qa.append({'role':role,'position':pos,'query_id':qid,'fact':name})
  t={'kind':'belief','worlds':worlds,'observations':observations,'queries':q};order=t if style=='forward' else reverse_fields(t)
  sid='opaque-'+hashlib.sha256(canon([ctx,variant,u,v,w,style]).encode()).hexdigest()[:20]
  row={'sample_id':sid,'source_family_key':'reviewer-handmade-family','controlled_context':ctx,'auxiliary_scalar_variant':variant,'schedule_index':9*u+3*v+w,'id_shift':u,'position_shift':v,'fact_name_shift':w,'renderer':style,'query_id_pool':ids,'queried_fact_name_pool':queries,'fact_role_assignments':{'observation_control':mp['x'],'observation_auxiliary':'y','invariant_query':mp['q0'],'variation_query':mp['q1'],'unoffered_feature':'u'},'query_role_assignments':qa,'auxiliary_scalar_control':{'fact':'y','baseline_value':recipe['baseline_value'],'supported_value':recipe['supported_value']},'semantic_task':t,'rendered_input':canon(order) if style=='forward' else json.dumps(order,separators=(',',':'),ensure_ascii=False)}
  # Forward raw map order is deliberately made explicit; canonical JSON would erase it.
  row['rendered_input']=json.dumps(order,separators=(',',':'),ensure_ascii=False);row['record_hash']=record_hash(row);rows[sid]=row;coordinate[(ctx,variant,9*u+3*v+w,style)]=sid
 candhash=candidate_hash(rows);entries=[];claims={}
 for variant,k,style,length,name in itertools.product(('identity','transposed'),range(27),('forward','reverse'),(2,3),sorted(INHERITED)):
  left,right=('E','G') if length==2 else ('F','H');pair=[coordinate[(c,variant,k,style)] for c in (left,right)];cid='shared-evidence-'+digest([variant,k,style,length])[:12]
  entries.append({'name':name,'scope':'corpus_variant','source_family_key':'reviewer-handmade-family','auxiliary_scalar_variant':variant,'observation_length':length,'control_id':cid,'pair_sample_ids':pair,'exact_candidate_sha256':candhash})
 for sid,r in rows.items():
  length=len(r['semantic_task']['observations']);pairctx=('E','G') if length==2 else ('F','H');pair=[coordinate[(c,r['auxiliary_scalar_variant'],r['schedule_index'],r['renderer'])] for c in pairctx];cid='shared-evidence-'+digest([r['auxiliary_scalar_variant'],r['schedule_index'],r['renderer'],length])[:12]
  claims[sid]=[{'name':name,'scope':'corpus_variant','source_family_key':r['source_family_key'],'pair_sample_ids':pair,'evidence_manifest':MANIFEST_PATH,'control_id':cid} for name in sorted(INHERITED)]
 manifest={'supplied_same_table_v2_pairs':entries,'optional_diagnostics':{'handmade_only':True},'no_closed_top_level_shape_required':True}
 return rows,coordinate,manifest,claims,candhash


def manual_target(row,recipe):
 ctx=row['controlled_context'];prefix=recipe['manually_derived_prefix_ids'][ctx];support=prefix[-1];byrole={}
 if ctx in ('E','F'):byrole={role:[] for role in ROLES}
 else:
  q0='known' if ctx=='B' else None;q1=[{'value':'red','count':2}] if ctx=='B' else [{'value':1,'count':1},{'value':True,'count':1}] if ctx in ('G','H') else [{'value':0,'count':1},{'value':False,'count':1}]
  byrole={'invariant_query':[{'value':q0,'count':2}],'variation_query':q1,'observed_control':[{'value':'stone' if ctx in ('G','H') else 'reed','count':2}]}
 output=[None]*3
 for a in row['query_role_assignments']:output[a['position']]=answer({'query_id':a['query_id']},byrole[a['role']],len(support))
 return {'surviving_worlds':support,'belief_sizes':[len(p) for p in prefix],'queries':output}


def crossing_checks(rows,coordinate,targets,details):
 groups=collections.defaultdict(list)
 for r in rows.values():groups[(r['source_family_key'],r['controlled_context'],r['auxiliary_scalar_variant'],r['renderer'])].append(r)
 counters=collections.Counter();evidence=[]
 for key,rs in groups.items():
  require(len(rs)==27 and {r['schedule_index'] for r in rs}==set(range(27)),'all27 actual schedules')
  triples=collections.defaultdict(collections.Counter);hist=collections.defaultdict(collections.Counter)
  for r in rs:
   for a in r['query_role_assignments']:
    triple=(r['query_id_pool'].index(a['query_id']),a['position'],r['queried_fact_name_pool'].index(a['fact']));triples[a['role']][triple]+=1
    q=copy.deepcopy(targets[r['sample_id']]['queries'][a['position']]);q.pop('query_id');q['prefix_ids']=details[r['sample_id']]['prefix_survivors'];hist[triple][canon(q)]+=1
  require(all(set(c)==set(itertools.product(range(3),repeat=3)) and set(c.values())=={1} for c in triples.values()),'complete role triples once')
  for role,c in triples.items():
   for axis in range(3):require(set(collections.Counter(t[axis] for t in c).values())=={9},'marginals9')
   for axes in ((0,1),(0,2),(1,2)):require(set(collections.Counter(tuple(t[i] for i in axes) for t in c).values())=={3},'pairs3')
  require(len({canon(dict(h)) for h in hist.values()})==1,'full typed outcome and prefix provenance histogram equals across nuisance triples')
  evidence.append({'conditional_group':key,'rows':len(rs),'role_triples_per_role':27,'matching_typed_payload_provenance_histograms':True});counters['complete_conditional_groups']+=1
 for ctx,variant,k in itertools.product('ABCDEFGH',('identity','transposed'),range(27)):
  a,b=[rows[coordinate[(ctx,variant,k,s)]] for s in ('forward','reverse')];require(canon(json.loads(a['rendered_input']))==canon(json.loads(b['rendered_input'])) and a['rendered_input']!=b['rendered_input'],'actual field orders preserve typed tasks');require(canon(targets[a['sample_id']])==canon(targets[b['sample_id']]),'field target invariant');counters['field_order_pairs']+=1
 for ctx,k,style in itertools.product('ABCDEFGH',range(27),('forward','reverse')):
  a,b=[rows[coordinate[(ctx,s,k,style)]] for s in ('identity','transposed')];require(canon(targets[a['sample_id']])==canon(targets[b['sample_id']]) and details[a['sample_id']]['prefix_survivors']==details[b['sample_id']]['prefix_survivors'],'scalar transposition target/prefix invariant')
  t=copy.deepcopy(b['semantic_task']);control=a['auxiliary_scalar_control'];swap=lambda v:control['supported_value'] if v==control['baseline_value'] else control['baseline_value'] if v==control['supported_value'] else v
  for world in t['worlds']:world['facts'][control['fact']]=swap(world['facts'][control['fact']])
  for o in t['observations']:
   if o['fact']==control['fact']:o['equals']=swap(o['equals'])
  require(canon(t)==canon(a['semantic_task']),'actual T squared involution');counters['scalar_transposition_pairs']+=1
 for lctx,rctx,lvar,rvar,k,style in itertools.product(('E','F'),('G','H'),('identity','transposed'),('identity','transposed'),range(27),('forward','reverse')):
  if (lctx,rctx) not in (('E','G'),('F','H')) or lvar==rvar:continue
  a,b=[rows[coordinate[(ctx,var,k,style)]] for ctx,var in ((lctx,lvar),(rctx,rvar))];at,bt=a['semantic_task'],b['semantic_task'];require(canon(at['observations'])==canon(bt['observations']) and canon(at['queries'])==canon(bt['queries']),'opposite status identical visible observations/queries');require(details[a['sample_id']]['prefix_survivors'][1]==details[b['sample_id']]['prefix_survivors'][1] and not targets[a['sample_id']]['surviving_worlds'] and targets[b['sample_id']]['surviving_worlds'],'cross-table same-prefix opposite outcomes');counters['same_observations_opposite_status_pairs']+=1
 return dict(counters),evidence


def negative_tests(rows,coord,manifest,claims,candhash):
 # Every test starts with real handmade rows; rehash owner changes before checking relations.
 ownerid=coord[('A','identity',0,'forward')];original_claim=claims[ownerid][0];records=[]
 def run(name,change,keep_bad_hash=False):
  owner=copy.deepcopy(rows[ownerid]);rs=dict(rows);rs[ownerid]=owner;c=copy.deepcopy(original_claim);m=copy.deepcopy(manifest);context={'owner':owner,'rows':rs,'claim':c,'manifest':m};change(context)
  if not keep_bad_hash:owner['record_hash']=record_hash(owner)
  current_hash=candidate_hash(rs)
  for entry in m['supplied_same_table_v2_pairs']:entry['exact_candidate_sha256']=current_hash
  if name=='incorrect_complete_candidate_hash':
   for entry in m['supplied_same_table_v2_pairs']:entry['exact_candidate_sha256']='0'*64
  try:resolve(owner,c,rs,m,current_hash)
  except (ValueError,TypeError,KeyError,IndexError) as e:records.append({'name':name,'rejected':True,'error':str(e),'claim':c,'owner_sample_id':ownerid})
  else:raise ValueError('Negative fixture silently accepted '+name)
 def cmut(**kw):return lambda x:x['claim'].update(kw)
 def omut(**kw):return lambda x:x['owner'].update(kw)
 run('owner_foreign_family',omut(source_family_key='foreign-family'))
 run('owner_wrong_scalar_variant',omut(auxiliary_scalar_variant='transposed'))
 run('owner_wrong_schedule',omut(schedule_index=1,fact_name_shift=1))
 run('owner_wrong_renderer',omut(renderer='reverse'))
 run('owner_actual_length3_with_length2_pair',lambda x:x['owner']['semantic_task']['observations'].append(copy.deepcopy(x['owner']['semantic_task']['observations'][0])))
 run('reversed_pair_IDs',lambda x:x['claim']['pair_sample_ids'].reverse())
 for label,context in [('foreign_scalar',('E','transposed',0,'forward')),('foreign_schedule',('E','identity',1,'forward')),('foreign_renderer',('E','identity',0,'reverse')),('wrong_length_pair',('F','identity',0,'forward'))]:
  run(label,lambda x,context=context:x['claim'].update(pair_sample_ids=[coord[context],x['claim']['pair_sample_ids'][1]]))
 run('identical_pair_IDs',lambda x:x['claim'].update(pair_sample_ids=[x['claim']['pair_sample_ids'][0]]*2))
 run('missing_inline_pair_IDs',lambda x:x['claim'].pop('pair_sample_ids'))
 run('one_pair_ID',lambda x:x['claim'].update(pair_sample_ids=x['claim']['pair_sample_ids'][:1]))
 run('unknown_pair_ID',cmut(pair_sample_ids=['not-member','not-member-2']))
 run('pair_ID_boolean_type',cmut(pair_sample_ids=[True,'not-member']))
 run('both_name_and_property',lambda x:x['claim'].update(property=x['claim']['name']))
 run('unknown_identifier',cmut(name='invented_property'))
 run('undeclared_canonical_claim_key',cmut(observation_length=2))
 run('probe_as_supplied_pair',cmut(probe_id='actually-executed-extra-probe'))
 run('unexecuted_probe_substitution',cmut(scope='audit_probe',probe_id='not-executed'))
 run('null_scope',cmut(scope=None))
 run('empty_control_ID',cmut(control_id=''))
 run('absolute_manifest_pointer',cmut(evidence_manifest='/tmp/manifest.json'))
 run('missing_named_manifest_pointer',cmut(evidence_manifest='reviews/adversarial/nonexistent.json'))
 run('optional_foreign_claim_family',cmut(source_family_key='foreign-family'))
 def match(x):return next(e for e in x['manifest']['supplied_same_table_v2_pairs'] if e['name']==x['claim']['name'] and e['control_id']==x['claim']['control_id'] and e['pair_sample_ids']==x['claim']['pair_sample_ids'])
 run('duplicate_ID_name_orderedpair_lookup',lambda x:x['manifest']['supplied_same_table_v2_pairs'].append(copy.deepcopy(match(x))))
 run('missing_named_entry',lambda x:x['manifest']['supplied_same_table_v2_pairs'].remove(match(x)))
 run('named_entry_reversed_pair',lambda x:match(x)['pair_sample_ids'].reverse())
 run('wrong_entry_name',lambda x:match(x).update(name='typed_equals_counterfactual'))
 run('wrong_entry_family',lambda x:match(x).update(source_family_key='foreign-family'))
 run('wrong_entry_variant',lambda x:match(x).update(auxiliary_scalar_variant='transposed'))
 run('wrong_entry_length',lambda x:match(x).update(observation_length=3))
 run('boolean_entry_length',lambda x:match(x).update(observation_length=True))
 run('missing_entry_length',lambda x:match(x).pop('observation_length'))
 run('entry_scope_probe',lambda x:match(x).update(scope='audit_probe'))
 run('incorrect_complete_candidate_hash',lambda x:None)
 run('incorrect_owner_record_hash',lambda x:x['owner'].update(record_hash='0'*64),True)
 for key in ('schedule_index','id_shift','position_shift','fact_name_shift'):run('bool_metadata_'+key,omut(**{key:True}))
 # Change an evidence record's actual typed input while preserving metadata. Update its hash.
 def altered_world(x):
  sid=x['claim']['pair_sample_ids'][1];r=copy.deepcopy(x['rows'][sid]);r['semantic_task']['worlds'][0]['facts']['y']=True;r['record_hash']=record_hash(r);x['rows'][sid]=r
 run('typed_world_change_boolean_versus_string',altered_world)
 def unsupported_nonempty(x):
  sid=x['claim']['pair_sample_ids'][1];r=copy.deepcopy(x['rows'][sid]);r['semantic_task']['observations'][1]['equals']=None;r['record_hash']=record_hash(r);x['rows'][sid]=r
 run('null_equals_cannot_fake_nonempty',unsupported_nonempty)
 # Foreign-family pair with truthful worlds remains unauthorized for this owner.
 def foreign_pair(x):
  ids=[]
  for sid in x['claim']['pair_sample_ids']:
   r=copy.deepcopy(x['rows'][sid]);r['sample_id']=sid+'foreign';r['source_family_key']='foreign-family';r['record_hash']=record_hash(r);x['rows'][r['sample_id']]=r;ids.append(r['sample_id'])
  x['claim']['pair_sample_ids']=ids
 run('actual_foreign_family_pair',foreign_pair)
 return records


def main():
 source=pathlib.Path(__file__);sourcehash=sha(source)
 for rel,h in PUBLIC.items():require(sha(ROOT/rel)==h,'public hash '+rel)
 require(sha(ROOT/OWN)==OWN_HASH and sha(ROOT/RECIPE)==RECIPE_HASH,'own source and hand recipe hash')
 freeze(ROOT/f'audits/history/belief_v4_adversarial_prospective-{sourcehash}.py',source.read_bytes())
 pre=ROOT/'audits/execution/belief-v4-adversarial-prospective-r1.preexec.json';freeze(pre,jb({'actual_execution_id':EXECUTION,'source_sha256':sourcehash,'fixture_sha256':RECIPE_HASH,'own_prior_solver_source_sha256':OWN_HASH,'public_source_bindings':PUBLIC,'stage':'SOURCE_AND_HANDMADE_EXPECTATIONS_FROZEN_BEFORE_ANY_EXECUTION','disclosure':DISCLOSURE,'read_scope':'Public definitions/freezes, own frozen source and new handmade recipe only. The explicitly authorized author freeze contains historical adjudication summary, but its linked verdicts/private code were not read.','approval':False}))
 print(canon({'stage':'source_and_fixtures_frozen','source_sha256':sourcehash,'preexec_sha256':sha(pre),'fixture_sha256':RECIPE_HASH}),flush=True);start=time.perf_counter()
 reused=toys();recipe=json.loads((ROOT/RECIPE).read_text());rows,coord,manifest,claims,candhash=make_fixture(recipe);require(len(rows)==864,'one-family handmade complete product')
 manifestfile=ROOT/MANIFEST_PATH;freeze(manifestfile,jb(manifest));loaded=json.loads(manifestfile.read_text());require(canon(loaded)==canon(manifest),'actual relative manifest bytes')
 targets={};details={};checks=collections.Counter()
 for sid,r in rows.items():
  decode(r);a,d=solve(r['semantic_task']);b,bd=own_prior_bitmask_solve(r['semantic_task']);require(canon(a)==canon(b) and d['prefix_survivors']==bd['prefix_survivors'],'new set filter vs own frozen bitmask');require(canon(a)==canon(manual_target(r,recipe)),'hand-derived exact typed target and prefix counts');require(d['prefix_survivors'][1:]==recipe['manually_derived_prefix_ids'][r['controlled_context']],'hand-derived exact IDs');targets[sid]=a;details[sid]=d;checks['hand_derived_full_targets']+=1;checks['set_vs_own_prior_bitmask_full_targets_prefixes']+=1
 for sid,r in rows.items():
  for c in claims[sid]:resolve(r,c,rows,loaded,candhash);checks['actual_owner_scoped_inherited_resolutions']+=1
 allpairs=[]
 for ctx,var,k,style in itertools.product(('A','D'),('identity','transposed'),range(27),('forward','reverse')):
  sid=coord[(ctx,var,k,style)];out=resolve(rows[sid],claims[sid][0],rows,loaded,candhash);require(out['owner_not_pair_member'],'legitimate nonmember owner');checks['legitimate_nonmember_owner_resolutions']+=1
 for length,var,k,style in itertools.product((2,3),('identity','transposed'),range(27),('forward','reverse')):
  ctx='E' if length==2 else 'F';sid=coord[(ctx,var,k,style)];a,b=claims[sid];require(a['control_id']==b['control_id'] and a['name']!=b['name'],'shared IDs across names');resolve(rows[sid],a,rows,loaded,candhash);resolve(rows[sid],b,rows,loaded,candhash);checks['same_ID_two_truthful_named_entries']+=1
  allpairs.append(resolve(rows[sid],a,rows,loaded,candhash)['derived_pair_evidence'])
 # Property alias is also legal; extra manifest diagnostics are not a rejection criterion.
 sid=coord[('A','identity',0,'forward')];c=copy.deepcopy(claims[sid][0]);c['property']=c.pop('name');resolve(rows[sid],c,rows,loaded,candhash);checks['property_alias_claim_without_name']+=1
 require(all('pair_record_hashes' not in e for e in loaded['supplied_same_table_v2_pairs']),'optional hashes deliberately omitted');checks['valid_without_optional_pair_record_hashes_or_global_ID_uniqueness']+=1
 negatives=negative_tests(rows,coord,loaded,claims,candhash);cross,crossevidence=crossing_checks(rows,coord,targets,details)
 # Executed typed probe evidence exists, but never earns supplied-pair coverage.
 probe={'kind':'belief','worlds':[{'world_id':'a','facts':{'x':True,'q':None}},{'world_id':'b','facts':{'x':1,'q':True}},{'world_id':'c','facts':{'x':False,'q':0}},{'world_id':'d','facts':{'x':0,'q':False}}],'observations':[{'fact':'x','equals':True}],'queries':[{'query_id':'probe-q','fact':'q'}]};true,td=solve(probe);other=copy.deepcopy(probe);other['observations'][0]['equals']=1;integer,idd=solve(other);require(true['surviving_worlds']==['a'] and integer['surviving_worlds']==['b'] and true['queries'][0]['entailed'] and true['queries'][0]['guaranteed'] is None,'typed probe cannot coercetrue1/null');probe_evidence={'probe_id':'reviewer-executed-typed-probe','actual_execution_id':EXECUTION,'executed':True,'true_task':probe,'int_task':other,'actual_true_target':true,'actual_int_target':integer,'supplied_pair_credit':False};checks['executed_typed_probe_only_no_supplied_credit']+=1
 # Actual fixture shortcuts, explicitly not real-corpus conclusions.
 baselines=[]
 for r in rows.values():
  t=r['semantic_task'];truth=not targets[r['sample_id']]['surviving_worlds'];marker=any(type(o['equals']) is str and o['equals'] in {'stone','unknown'} for o in t['observations']);linen=any(type(o['equals']) is str and o['equals']=='linen' for o in t['observations']);first=t['observations'][0];second=t['observations'][1];second_support=sum(typed(w['facts'][second['fact']])==typed(second['equals']) for w in t['worlds']);count_rule=marker and second_support==3
  baselines.append({'sample_id':r['sample_id'],'context':r['controlled_context'],'variant':r['auxiliary_scalar_variant'],'empty':truth,'original_marker_predict_empty':marker,'marker_AND_linen_predict_empty':marker and linen,'toy_fitted_table_column_count_predict_empty':count_rule})
 baseline_summary={key:{'correct':sum(v[key]==v['empty'] for v in baselines),'total':864} for key in ('original_marker_predict_empty','marker_AND_linen_predict_empty','toy_fitted_table_column_count_predict_empty')};require(baseline_summary['original_marker_predict_empty']['correct']==648 and baseline_summary['marker_AND_linen_predict_empty']['correct']==648,'manual fixture predicate75percent');require(baseline_summary['toy_fitted_table_column_count_predict_empty']['correct']==864,'intentional miniature fittedtable vulnerability')
 for rel,h in PUBLIC.items():require(sha(ROOT/rel)==h,'public unchanged '+rel)
 require(sha(ROOT/OWN)==OWN_HASH and sha(ROOT/RECIPE)==RECIPE_HASH and sha(source)==sourcehash,'source/test bytes unchanged')
 report={'actual_execution_id':EXECUTION,'completed':True,'stage':'PROSPECTIVE_HANDMADE_V4_PREPARATION_ONLY','source_sha256':sourcehash,'preexec_sha256':sha(pre),'fixture_sha256_before_after':RECIPE_HASH,'public_sources_sha256_before_after':PUBLIC,'own_prior_source_sha256_before_after':OWN_HASH,'actual_seconds':time.perf_counter()-start,'own_reused_toys':reused,'handmade_fixture_rows_in_memory':len(rows),'handmade_fixture_candidate_sha256_actual':candhash,'handmade_relative_manifest_sha256':sha(manifestfile),'handmade_relative_manifest_named_entries':len(loaded['supplied_same_table_v2_pairs']),'actual_counts':dict(checks),'actual_negative_tests':negatives,'crossing_counts':cross,'crossing_evidence':crossevidence,'actual216_typed_pair_evidence':allpairs,'executed_probe_evidence':probe_evidence,'handmade_only_baselines':baseline_summary,'future_input_stage_plan':['Bind fresh stripped view/source before actual every-row targets; preserve allfirsts before fresh canonical access.','Decode metadata only after later authorization and verify every actual owner length/family/variant/schedule/style/pair lookup and exact full candidate hash.','Independently audit all128 conditional triple groups/1728stylepairs/1728scalarpairs/864same-observation pairs only if actual full cohort supplies them.','Measure typed scalar-anywhere/position/fact-role/pairs, ID/name/position joint counts, prompt lengths, fullworld/count/character-length/table-fingerprint and partial/full table lookup train-to-exposed-dev fits.','Compare original real tasks/renderings/full typed targets from an authorized stripped matching view before claiming zero semantic change; new IDs never prove equality.'],'residual_limits':['One7world handmade family/864fixture rows is a prospective test product, not four source profiles or a real V4 candidate; no source-growth or production inspection.','Hand fixture majority75percent and both old scalar rules75percent do not establish actual V4 counts; the intentionally fitted count rule100percent illustrates remaining world-table associations.','The miniature usefulness histogram differs from real public scope; the actual87.5percent usefulness-false majority must be remeasured later.','Actual pair comparison/prefix evidence fields lack a separately published exact closed shape; this resolver does not invent mandatory annotation keys. Own derived evidence is separately emitted.','Three V3 aggregate controls are verified here by reconstructing full handmade groups; pointers alone cannot establish actual future group truth.','Record fields here are focused resolver/schedule fixtures, not full canonical schema or production/publication gates.','No global uniquecontrolID, closedmanifestshape or mandatorypair_record_hashes requirement is imposed. No actual candidate or private production/formal/other reviewer read.'], 'disclosure':DISCLOSURE,'data_approval':False}
 out=ROOT/'reviews/adversarial/belief-v4-prospective-preparation-r1.json';freeze(out,jb(report));ex=ROOT/'audits/execution/belief-v4-adversarial-prospective-r1.json';freeze(ex,jb({'actual_execution_id':EXECUTION,'completed':True,'source_sha256':sourcehash,'preexec_sha256':sha(pre),'fixture_sha256_before_after':RECIPE_HASH,'report_sha256':sha(out),'public_sources_sha256_before_after':PUBLIC,'actual_counts':dict(checks),'negative_tests_executed_and_rejected':len(negatives),'crossing_counts':cross,'actual_reused_hand_toys':len(reused['manually_authored_exact_toys']),'actual_reused_malformed_toys':len(reused['malformed_probes']),'private_production_and_actual_candidate_not_read':True,'data_approval':False,'disclosure':DISCLOSURE}))
 print(canon({'completed':True,'source_sha256':sourcehash,'preexec_sha256':sha(pre),'fixture_sha256':RECIPE_HASH,'report_sha256':sha(out),'execution_sha256':sha(ex),'negative_checks':len(negatives),'counts':dict(checks),'crossings':cross,'toy_baselines':baseline_summary}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  source=pathlib.Path(__file__);f=ROOT/f'audits/execution/belief-v4-adversarial-prospective-r1.failure-{sha(source)[:16]}.json';freeze(f,jb({'actual_execution_id':EXECUTION,'source_sha256':sha(source),'completed':False,'error_type':type(e).__name__,'error':str(e),'public_old_candidate_review_bytes_unchanged':True}));print(canon({'failure_sha256':sha(f),'path':str(f.relative_to(ROOT)),'error':str(e)}),flush=True);raise
