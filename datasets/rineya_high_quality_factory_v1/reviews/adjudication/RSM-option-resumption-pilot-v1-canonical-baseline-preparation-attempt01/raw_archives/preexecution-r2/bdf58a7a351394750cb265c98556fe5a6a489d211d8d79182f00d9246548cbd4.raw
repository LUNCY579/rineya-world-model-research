"""Prepared independently owned canonical FIRST, NEVER executed during preparation.
No official/private producer/formal semantic implementation or other review imports.
"""
import argparse,collections,copy,datetime,hashlib,importlib.util,itertools,json,pathlib,re,sys,time,traceback
D=pathlib.Path(__file__).resolve().parent;B=D.parents[2]
NAMES={'uninterrupted_resume_suffix':('F','R'),'resume_restart_replay':('R','T'),'budget_pause_after_effect':('R','P'),'second_checkpoint_suffix':('P','U'),'positive_bound_argument_counterfactual':('F','A'),'initiation_argument_counterfactual':('F','I'),'success_contract_counterfactual':('F','S'),'active_checkpoint_tamper_rejection':('R','X'),'object_field_order_invariance':None}
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def equal(a,b):return enc(a)==enc(b)
def integer(v):return type(v) is int
def text(v):return type(v) is str and bool(v)
def hashtext(v):return type(v) is str and re.fullmatch('[0-9a-f]{64}',v) is not None
def rowhash(v):return sha(enc({k:x for k,x in v.items() if k!='record_hash'}))
def unique(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate JSON key '+k)
  d[k]=v
 return d
def loads(raw):return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def readjson(p):return loads(p.read_bytes())
def readrows(p):return [loads(l) for l in p.read_bytes().splitlines()]
def newjson(p,v):
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
def newjsonl(p,vs):
 with p.open('x') as f:
  for v in vs:f.write(enc(v).decode()+'\n')
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def normalize_name(v):
 n=v.get('name');p=v.get('property')
 if 'name' not in v and 'property' not in v:raise ValueError('missing identifier')
 if 'name' in v and (not text(n) or n not in NAMES):raise ValueError('identifier name')
 if 'property' in v and (not text(p) or p not in NAMES):raise ValueError('identifier property')
 if 'name' in v and 'property' in v and n!=p:raise ValueError('conflicting identifier')
 return n or p

def safe_path(value):
 if not text(value) or '\x00' in value or '\\' in value:return False
 p=pathlib.PurePosixPath(value)
 if p.is_absolute() or '..' in p.parts or not p.parts or value!=p.as_posix():return False
 current=B
 if any(a.is_symlink() for a in (B,*B.parents)):return False
 for part in p.parts:
  current=current/part
  if current.is_symlink():return False
 return True

def pairkey(v):
 ids=v.get('pair_sample_ids')
 if not text(v.get('control_id')) or type(ids) is not list or len(ids)!=2 or not all(type(x) is str for x in ids) or ids[0]==ids[1]:raise ValueError('control ID or ordered pair IDs')
 return v['control_id'],normalize_name(v),tuple(ids)

def verify_freeze(fr):
 if str(pathlib.Path(sys.executable).resolve())!=fr['resolved_python']:raise ValueError('resolved Python differs')
 checks=[]
 for v in fr['bindings']:
  path=pathlib.Path(v['path']) if v['path'].startswith('/') else B/v['path'];raw=path.read_bytes();archive=(B/v['archive']).read_bytes()
  if sha(raw)!=v['sha256'] or len(raw)!=v['bytes'] or raw!=archive:raise ValueError('frozen byte drift '+v['path'])
  checks.append({'path':v['path'],'sha256':sha(raw),'bytes':len(raw),'opaque':v.get('scope')=='opaque-source-custody'})
 return checks

def loaded_runtime(fr):
 frozen={str((pathlib.Path(v['path']) if v['path'].startswith('/') else B/v['path']).resolve()):v for v in fr['bindings']}
 checked=[]
 for name,m in sorted(sys.modules.items()):
  source=getattr(m,'__file__',None)
  if source is None:continue
  p=pathlib.Path(source)
  if p.suffix=='.pyc':
   try:p=pathlib.Path(importlib.util.source_from_cache(str(p)))
   except ValueError:pass
  if not p.is_file():raise ValueError('loaded module lacks byte custody '+name+':'+str(p))
  p=p.resolve();v=frozen.get(str(p))
  if v is None:raise ValueError('loaded module outside frozen closure '+name+':'+str(p))
  raw=p.read_bytes()
  if sha(raw)!=v['sha256'] or len(raw)!=v['bytes']:raise ValueError('loaded runtime differs '+name)
  checked.append({'module':name,'path':str(p),'sha256':v['sha256'],'bytes':v['bytes']})
 return checked

def object_shape(v):
 if type(v) is dict:return [[k,object_shape(x)] for k,x in v.items()]
 if type(v) is list:return [object_shape(x) for x in v]
 return ['scalar',type(v).__name__,v]
def reversed_objects(v):
 if type(v) is dict:return {k:reversed_objects(v[k]) for k in reversed(v)}
 if type(v) is list:return [reversed_objects(x) for x in v]
 return v

def metrics(t,events,engine):
 active=0;depth=0;branches=0
 for count in range(len(events)+1):
  cp=engine.fold_events(t,events[:count]);active=sum(f['frame_kind']=='option' for f in cp['frames']);depth=max(depth,active);branches=max(branches,sum(f['frame_kind']=='branch' for f in cp['frames']))
 return dict(engine.primitives(t),maximum_option_depth=depth,maximum_selected_branch_depth=branches,total_F_events=len(events),total_F_effects=sum(e['event']=='effect' for e in events))

def family_evidence(rows,pred,solver,engine,activity):
 byrole=collections.defaultdict(dict)
 for row in rows:byrole[row['renderer']][row['controlled_context']]=row
 issues=[];facts={}
 for style,ctx in byrole.items():
  if set(ctx)!=set('FRTPUAISX'):issues.append('MISSING_CONTEXT:'+style);continue
  F,R,T,P,U,A,I,S,X=(ctx[k]['semantic_task'] for k in 'FRTPUAISX');o={k:pred[ctx[k]['sample_id']]['expected_output'] for k in ctx};fchecks={};history=R['history'];frames=R['checkpoint']['frames']
  fchecks['F_success_and_budget']=o['F']['status']=='completed' and F['mode']=='fresh' and F['history']==[] and F['checkpoint'] is None and F['event_budget']==48
  fchecks['full_budgets']=all(ctx[k]['semantic_task']['event_budget']==48 for k in 'FRUA')
  fchecks['F_R_suffix']=equal(history+o['R']['new_events'],o['F']['new_events']) and equal(o['R']['checkpoint'],o['F']['checkpoint'])
  fchecks['R_T_only_mode']=engine.diff(R,T)==[{'path':['mode'],'before':'resume','after':'restart'}]
  fchecks['root_restart_passes']=o['T']['new_events'][0]['event']=='enter' and o['T']['new_events'][0]['initiation_ok']
  fchecks['restart_trace_and_terminal_differ']=o['T']['status'] in ('completed','initiation_failed','success_failed') and not equal(o['R']['new_events'],o['T']['new_events']) and (o['R']['status']!=o['T']['status'] or not equal(o['R']['checkpoint']['state'],o['T']['checkpoint']['state']))
  delta=engine.diff(R,P);fchecks['P_only_budget_and_committed_effect']=len(delta)==1 and delta[0]['path']==['event_budget'] and o['P']['status']=='paused' and o['P']['new_effects']>=1
  fchecks['U_second_snapshot_suffix']=equal(U['history'],P['history']+o['P']['new_events']) and equal(U['checkpoint'],o['P']['checkpoint']) and equal(o['U']['checkpoint'],o['F']['checkpoint']) and equal(U['history']+o['U']['new_events'],o['F']['new_events'])
  da=engine.diff(F,A);di=engine.diff(F,I);ds=engine.diff(F,S);dx=engine.diff(R,X)
  fchecks['A_one_argument_and_later_behavior']=o['F']['status']=='completed' and len(da)==1 and da[0]['path'][:2]==['entry','args'] and o['A']['status']=='completed' and not equal([e for e in o['A']['new_events'] if e['event']!='enter'],[e for e in o['F']['new_events'] if e['event']!='enter'])
  fchecks['I_one_argument_and_no_effect']=len(di)==1 and di[0]['path'][:2]==['entry','args'] and o['I']['status']=='initiation_failed' and o['I']['new_effects']==0
  root=F['entry']['option'];ip=[solver.test({'all':[a]},I['initial_state'],{p:solver.evaluate(e,I['initial_state'],{}) for p,e in I['entry']['args'].items()}) for a in I['options'][root]['initiation']['all']];fchecks['I_selective_one_false_atom']=ip.count(False)==1
  fchecks['S_one_success_literal_same_effects']=len(ds)==1 and ds[0]['path'][:3]==['options',root,'success'] and o['S']['status']=='success_failed' and equal([e for e in o['F']['new_events'] if e['event']=='effect'],[e for e in o['S']['new_events'] if e['event']=='effect'])
  sb={p:solver.evaluate(e,S['initial_state'],{}) for p,e in S['entry']['args'].items()};sp=[solver.test({'all':[a]},o['S']['checkpoint']['state'],sb) for a in S['options'][root]['success']['all']];fchecks['S_selective_one_false_atom']=sp.count(False)==1
  fchecks['X_one_cp_scalar_rejected']=len(dx)==1 and integer(dx[0]['before']) and integer(dx[0]['after']) and dx[0]['path'][0]=='checkpoint' and not isinstance(dx[0]['before'],(dict,list)) and o['X']==solver.invalid('invalid_checkpoint')
  dependency={};w=ctx['R'].get('property_witnesses',{})
  # Optional locators are facts if supplied; absence does not remove the required
  # relation, which is reconstructed by probing actual active frame copies.
  locators=[]
  for fi,f in enumerate(frames):
   for name in f['bindings']:locators.append(['frames',fi,'bindings',name])
  for locator in locators:
   result=activity.activity(R,R['checkpoint'],locator,solver);dependency[json.dumps(locator)]=result
  actual_used=[v for v in dependency.values() if v['status']=='ACTIVE'];fchecks['some_pending_binding_consequential']=bool(actual_used)
  # Trace each retained argument copy back through actual committed enter events.
  # A direct register expression or a forwarded register-derived parameter is
  # evidence; repeated parameter spelling and annotation alone are not.
  origins={}
  for event in history:
   if event['event']!='enter':continue
   site=event['call_site'];caller=site['caller_instance'] if site is not None else None
   if site is None:args=R['entry']['args']
   else:
    previous=next(e for e in history if e['event']=='enter' and e['instance_id']==caller)
    args=engine.code_body(R,previous['option'],site['body_path'])[site['index']]['args']
   origins[event['instance_id']]={name:('reg' in expr or ('param' in expr and origins.get(caller,{}).get(expr['param'],False))) for name,expr in args.items()}
  derived=[]
  for item in actual_used:
   fi=item['locator'][1];name=item['locator'][-1];instance=frames[fi]['instance_id']
   if origins.get(instance,{}).get(name,False):derived.append(item['locator'])
  fchecks['register_derived_pending_saved_argument']=bool(derived)
  dependency['register_derived_active_locators']=derived
  false_init=[];opposite=[]
  for f in frames:
   if f['frame_kind']=='option' and f['instance_id']!=0:false_init.append(not solver.test(R['options'][f['option']]['initiation'],R['checkpoint']['state'],f['bindings']))
   if f['frame_kind']=='branch':
    ins=engine.code_body(R,f['option'],f['body_path'][:-2])[f['body_path'][-2]];now=solver.test(ins['test'],R['checkpoint']['state'],f['bindings']);opposite.append(now!=(f['body_path'][-1]=='then'))
  fchecks['nonroot_init_now_false']=any(false_init);fchecks['selected_branch_guard_reversed']=any(opposite)
  fchecks['R_retained_caller_branch_and_committed_effect']=sum(f['frame_kind']=='option' for f in frames)>=2 and any(f['frame_kind']=='branch' for f in frames) and sum(e['event']=='effect' for e in history)>=1
  if w.get('saved_binding_locator') is not None:
   declared=activity.activity(R,R['checkpoint'],w['saved_binding_locator'],solver);fchecks['declared_saved_copy_consequential']=declared['status']=='ACTIVE'
  if len(dx)==1:
   loc=dx[0]['path'][1:];changed=copy.deepcopy(R['checkpoint']);node=changed
   for k in loc[:-1]:node=node[k]
   before=node[loc[-1]];after=dx[0]['after'];node[loc[-1]]=after
   try:baseobs=activity.observe(R,R['checkpoint'],solver);mutobs=activity.observe(R,changed,solver);fchecks['X_consequential_future_observable']=not activity.equal(baseobs['observable_projection'],mutobs['observable_projection'])
   except (KeyError,IndexError,AssertionError):fchecks['X_consequential_future_observable']=False
  replay=[];state=copy.deepcopy(R['initial_state']);bindings={}
  for e in history:
   if e['event']=='enter':bindings[e['instance_id']]=e['bindings']
   if e['event']=='effect':
    ins=engine.code_body(R,e['option'],e['body_path'])[e['index']];saved=bindings[e['instance_id']]
    def transform(z):
     n=copy.deepcopy(z)
     if ins['op']=='swap':n['x'],n['y']=n['y'],n['x']
     else:
      v=solver.evaluate(ins['value'],n,saved);reg=ins['register'];n[reg]=n[reg]+v if ins['op']=='add' else n[reg]*v
     return n
    a=transform(state);b=transform(a);hits=[j for j,n in enumerate(o['T']['new_events']) if n['event']=='effect' and (n['option'],n['body_path'],n['index'])==(e['option'],e['body_path'],e['index'])]
    if not equal(a,b) and hits:replay.append({'option':e['option'],'body_path':e['body_path'],'index':e['index'],'original_prestate':state,'once':a,'twice':b,'T_event_indices':hits})
    state=e['state']
  fchecks['actual_replayed_nonidempotent_effect']=bool(replay)
  fmetrics=metrics(F,o['F']['new_events'],engine);limits=readjson(B/'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json')['bounded_valid_programs']
  fchecks['program_structure_bounds']=limits['option_count_min']<=fmetrics['options']<=limits['option_count_max'] and all(limits['parameters_per_option_min']<=n<=limits['parameters_per_option_max'] for n in fmetrics['parameters_per_option'].values()) and fmetrics['declared_instructions']<=limits['maximum_declared_instructions_both_branches'] and fmetrics['syntactic_conditional_depth']<=limits['maximum_syntactic_conditional_depth'] and fmetrics['maximum_option_depth']<=limits['maximum_simultaneous_option_depth'] and fmetrics['total_F_events']<=limits['maximum_complete_semantic_events'] and fmetrics['total_F_effects']>=limits['minimum_executed_F_effects']
  fchecks['actual_valid_run_range_bounds']=all(limits['initial_integer_range'][0]<=v<=limits['initial_integer_range'][1] for v in F['initial_state'].values()) and all(limits['valid_run_intermediate_integer_range'][0]<=v<=limits['valid_run_intermediate_integer_range'][1] for k in 'FRTPUAIS' for e in o[k]['new_events'] if e['event']=='effect' for v in e['state'].values())
  fmetrics['history_effects_at_R']=sum(e['event']=='effect' for e in history);fmetrics['caller_frames_at_R']=sum(f['frame_kind']=='option' for f in frames)-1;fmetrics['selected_branch_frames_at_R']=sum(f['frame_kind']=='branch' for f in frames)
  witnesses={}
  for row in ctx.values():
   p=row.get('property_witnesses',{})
   if type(p) is not dict:issues.append('WITNESS_OBJECT:'+row['sample_id']);continue
   if p.get('replay_locator') is not None and not any(equal(p['replay_locator'],{k:v[k] for k in ('option','body_path','index')}) for v in replay):issues.append('FALSE_REPLAY_LOCATOR:'+row['sample_id'])
   if p.get('checkpoint_tamper_locator') is not None and (len(dx)!=1 or p['checkpoint_tamper_locator']!=dx[0]['path'][1:]):issues.append('FALSE_TAMPER_LOCATOR:'+row['sample_id'])
   if p.get('saved_binding_locator') is not None:
    try:
     if activity.activity(R,R['checkpoint'],p['saved_binding_locator'],solver)['status']!='ACTIVE':issues.append('INACTIVE_SAVED_LOCATOR:'+row['sample_id'])
    except (ValueError,KeyError,IndexError,TypeError):issues.append('FALSE_SAVED_LOCATOR:'+row['sample_id'])
   declared=p.get('dependency_metrics',{});supported={'maximum_option_depth','maximum_selected_branch_depth','declared_instructions','total_F_events','total_F_effects','history_effects_at_R','caller_frames_at_R','selected_branch_frames_at_R'}
   if type(declared) is not dict:issues.append('METRICS_OBJECT:'+row['sample_id']);declared={}
   bad={k:v for k,v in declared.items() if not integer(v) or v<0 or k not in supported or v!=fmetrics[k]}
   if bad:issues.append('FALSE_OR_UNTYPED_METRICS:'+row['sample_id']);witnesses[row['sample_id']]=bad
   for name in ('saved_binding_locator','checkpoint_tamper_locator'):
    if name in p:
     locator=p[name]
     if type(locator) is not list or any(not (type(k) is str or type(k) is int) for k in locator):issues.append('UNTYPED_LOCATOR:'+row['sample_id'])
  facts[style]={'argument_register_origins':origins,'checks':fchecks,'dependency_observables':dependency,'replay':replay,'measured_metrics':fmetrics,'witness_errors':witnesses}
  issues.extend('CONTROL_FALSE:'+style+':'+k for k,v in fchecks.items() if not v)
 return {'issues':issues,'styles':facts}

def manifest_and_claims(manifest,rows,candidate_sha):
 index=collections.defaultdict(list);errors=[];byid={r['sample_id']:r for r in rows}
 required={'scope','control_id','pair_sample_ids','source_family_key','source_family_index','controlled_contexts','renderers','candidate_sha256'}
 for entry in manifest.get('control_pairs',[]):
  try:
   key=pairkey(entry);index[key].append(entry);left,right=(byid[s] for s in key[2]);name=key[1]
   if not required<=set(entry):raise ValueError('missing public pair entry field')
   if entry['scope']!='corpus_variant' or entry['candidate_sha256']!=candidate_sha:raise ValueError('scope/candidate SHA')
   if not text(entry['source_family_key']) or not integer(entry['source_family_index']):raise ValueError('typed manifest family coordinates')
   if any(r['source_family_key']!=entry['source_family_key'] or r['source_family_index']!=entry['source_family_index'] for r in (left,right)):raise ValueError('family coordinate')
   if entry['controlled_contexts']!=[left['controlled_context'],right['controlled_context']] or entry['renderers']!=[left['renderer'],right['renderer']]:raise ValueError('member context/renderer coordinates')
   if left['split']!=right['split']:raise ValueError('pair split')
   if name=='object_field_order_invariance':
    if left['controlled_context']!=right['controlled_context'] or [left['renderer'],right['renderer']]!=['forward','reverse'] or not equal(left['semantic_task'],right['semantic_task']) or not equal(left['verified_target'],right['verified_target']):raise ValueError('field-order member relation')
   elif (left['controlled_context'],right['controlled_context'])!=NAMES[name] or left['renderer']!=right['renderer']:raise ValueError('named relation orientation/style')
   hashes=entry.get('pair_record_hashes')
   if hashes is not None and (type(hashes) is not list or hashes!=[left['record_hash'],right['record_hash']] or not all(hashtext(h) for h in hashes)):raise ValueError('external ordered hashes')
  except (ValueError,KeyError,TypeError) as exc:errors.append({'type':'MANIFEST_ENTRY','error':str(exc),'entry':entry})
 for key,entries in index.items():
  if len(entries)!=1:errors.append({'type':'AMBIGUOUS_NAMED_RESOLVER','key':list(key),'entries':len(entries)})
 perrow={};allowed={'name','property','scope','evidence_manifest','control_id','pair_sample_ids','source_family_key','source_family_index','controlled_context','renderer','diagnostics'}
 for owner in rows:
  reasons=[];names=[];claims=owner.get('family_control_claims')
  if type(claims) is not list:reasons.append('MISSING_CLAIMS');claims=[]
  for claim in claims:
   try:
    if type(claim) is not dict or not {'scope','evidence_manifest','control_id','pair_sample_ids'}<=set(claim) or set(claim)-allowed:raise ValueError('unsupported/missing claim field, canonical hashes forbidden')
    key=pairkey(claim);names.append(key[1]);path=claim['evidence_manifest']
    if claim['scope']!='corpus_variant' or not safe_path(path):raise ValueError('unsafe scope/manifest')
    if path!=readjson(D/'config-r1.json')['control_manifest']:raise ValueError('supplied manifest unresolved')
    entries=index[key]
    if len(entries)!=1:raise ValueError('named entry not uniquely resolved')
    entry=entries[0];members=[byid[s] for s in key[2]]
    if any(r['source_family_key']!=owner['source_family_key'] or r['source_family_index']!=owner['source_family_index'] or r['split']!=owner['split'] for r in members):raise ValueError('owner/member coordinate mismatch')
    if key[1]=='object_field_order_invariance':
     if owner['sample_id'] not in key[2] or any(r['controlled_context']!=owner['controlled_context'] for r in members):raise ValueError('renderer owner exception')
    elif any(r['renderer']!=owner['renderer'] for r in members):raise ValueError('owner renderer mismatch')
    for field in ('source_family_key','source_family_index','controlled_context','renderer'):
     if field in claim and (type(claim[field]) is not type(owner[field]) or claim[field]!=owner[field]):raise ValueError('false/untyped optional owner coordinate')
    if 'diagnostics' in claim and type(claim['diagnostics']) is not dict:raise ValueError('diagnostics object')
   except (ValueError,KeyError,TypeError) as exc:reasons.append('CLAIM:'+str(exc))
  if collections.Counter(names)!=collections.Counter(NAMES.keys()):reasons.append('INCOMPLETE_OR_DUPLICATE_NINE_NAMES')
  perrow[owner['sample_id']]=reasons
 return {'manifest_errors':errors,'per_row_errors':perrow,'entries':len(manifest.get('control_pairs',[])),'claims':sum(len(r.get('family_control_claims',[])) for r in rows)}

def run(expected,expected_config):
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();fr_raw=(D/'SOURCE_PREEXECUTION-r1.json').read_bytes()
 if sha(fr_raw)!=expected:raise ValueError('source freeze differs')
 fr=readjson(D/'SOURCE_PREEXECUTION-r1.json');before=verify_freeze(fr);runtime_before=loaded_runtime(fr);config_raw=(D/'config-r1.json').read_bytes()
 if sha(config_raw)!=expected_config:raise ValueError('external expected config differs')
 cfg=loads(config_raw)
 # Actual execution requires an external grant; --execute-after-separate-root-grant
 # is only a deliberate invocation marker, never inferred approval.
 for name in cfg['planned_outputs_after_separate_grant']:
  if (D/name).exists():raise ValueError('would overwrite output '+name)
 dep=B/cfg['own_public_preparation'];first=B/cfg['immutable_input_first'];solver=load('own_canon_solver',dep/'solver-r1.py');engine=load('own_input_engine',first/'input-review-r2.py');activity=load('own_canon_observable',D/'future-observables-r1.py');novelty=load('own_canon_novelty',D/'novelty-r1.py')
 candidate_raw=(B/cfg['candidate']).read_bytes();rows=readrows(B/cfg['candidate']);parentrows=readrows(B/cfg['semantic_parents']);legacy=readrows(B/cfg['legacy_inputs']);manifest=readjson(B/cfg['control_manifest']);predrows=readrows(first/'predictions-r2.jsonl');pred={r['sample_id']:r for r in predrows};parents={r['semantic_parent_id']:r for r in parentrows};newjudgments=[];joins=[];globalissues=[];families=collections.defaultdict(list)
 if sha(candidate_raw)!=cfg['candidate_sha256']:raise ValueError('candidate bytes differ')
 if len(rows)!=216 or len(pred)!=216 or len({r['sample_id'] for r in rows})!=216:globalissues.append('FULL_ROW_COMPLETENESS')
 sources=[]
 for path,asserted in manifest.get('source_bindings',{}).items():
  frozen=[v for v in fr['bindings'] if v['path']==path]
  if len(frozen)!=1:sources.append({'path':path,'asserted_sha256':asserted,'status':'UNRESOLVED_SOURCE'});globalissues.append('SOURCE_UNRESOLVED:'+path)
  else:
   observed=frozen[0]['sha256'];sources.append({'path':path,'asserted_sha256':asserted,'observed_sha256':observed,'matches':asserted==observed,'opaque':frozen[0].get('scope')=='opaque-source-custody'})
   if asserted!=observed:globalissues.append('SOURCE_BINDING_MISMATCH:'+path)
 public=readjson(B/'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json');indices={i for values in public['source_family_indices_by_stratum'].values() for i in values}
 required_public=[v['path'] for v in fr['bindings'] if v.get('scope')=='operative-public-source']
 if not integer(manifest.get('metadata_version')) or manifest.get('metadata_version')!=1 or manifest.get('batch')!=public['proposed_batch'] or manifest.get('candidate_sha256')!=cfg['candidate_sha256']:globalissues.append('MANIFEST_TOPLEVEL_CUSTODY')
 if type(manifest.get('source_bindings')) is not dict or not set(required_public)<=set(manifest.get('source_bindings',{})):globalissues.append('MANIFEST_PUBLIC_SOURCE_COMPLETENESS')
 if any(not safe_path(k) for k in manifest.get('source_bindings',{})):globalissues.append('MANIFEST_SOURCE_PATH_SAFETY')
 if any(not hashtext(v) for v in manifest.get('source_bindings',{}).values()):globalissues.append('MANIFEST_SOURCE_HASH_TYPING')
 m=manifest_and_claims(manifest,rows,cfg['candidate_sha256']);globalissues.extend('MANIFEST:'+e['type'] for e in m['manifest_errors'])
 for row in rows:families[row['source_family_key']].append(row)
 if len(parentrows)!=108 or len(families)!=12 or m['entries']!=300 or m['claims']!=1944:globalissues.append('COHORT_COMPLETENESS')
 for family,members in families.items():
  if len(members)!=18 or {(r['controlled_context'],r['renderer']) for r in members}!={(c,v) for c in 'FRTPUAISX' for v in ('forward','reverse')} or len({r['split'] for r in members})!=1:globalissues.append('FAMILY_CONTEXT_STYLE_SPLIT_COMPLETENESS:'+family)
 familyfacts={};signatures={};legacy_sigs={view:collections.defaultdict(list) for view in ('full','all_coarse','reachable_coarse')}
 for old in legacy:
  for view in legacy_sigs:legacy_sigs[view][novelty.signature(old['semantic_task'],view)].append(old['sample_id'])
 for family,members in families.items():
  familyfacts[family]=family_evidence(members,pred,solver,engine,activity)
  frows=[r for r in members if r['controlled_context']=='F'];program=frows[0]['semantic_task'];signatures[family]={view:novelty.signature(program,view) for view in legacy_sigs}
 for view in legacy_sigs:
  seen=collections.defaultdict(list)
  for family,v in signatures.items():seen[v[view]].append(family)
  for sig,names in seen.items():
   if len(names)>1:globalissues.append('NOVELTY_CURRENT_COLLISION:'+view+':'+','.join(names))
   if sig in legacy_sigs[view]:globalissues.append('NOVELTY_LEGACY_COLLISION:'+view+':'+','.join(names))
 schema=readjson(B/'schemas/canonical/v1.schema.json');contract=(B/'schemas/RSM_OPTION_RESUMPTION_V1.md').read_text();style_inputs={}
 for row in rows:
  sid=row['sample_id'];issues=[];p=pred.get(sid);gold=row['verified_target'];parent=parents.get(row['semantic_parent_id']);familiesplit=('train' if row['source_family_index']%10<8 else 'dev_challenge') if integer(row['source_family_index']) else None
  if not set(schema['required'])<=set(row):issues.append('CANONICAL_REQUIRED_FIELDS')
  for key,definition in schema['properties'].items():
   if key not in row:continue
   value=row[key];kind=definition.get('type');expected_type={'string':str,'integer':int,'object':dict,'array':list}.get(kind)
   if expected_type is not None and type(value) is not expected_type:issues.append('CANONICAL_FIELD_TYPE:'+key)
   if kind=='array' and definition.get('items',{}).get('type')=='string' and (type(value) is not list or any(type(v) is not str for v in value)):issues.append('CANONICAL_ARRAY_ITEM:'+key)
   if 'enum' in definition and value not in definition['enum']:issues.append('CANONICAL_ENUM:'+key)
  try:
   prefix,tail=row['rendered_input'].split('\nTask JSON:\n');decoded=loads(tail)
   if prefix.rstrip()!=contract.rstrip() or not equal(decoded,row['semantic_task']):issues.append('PUBLIC_RENDERED_TASK_JOIN')
   style_inputs[(row['source_family_key'],row['controlled_context'],row['renderer'])]=decoded
  except (ValueError,TypeError):issues.append('PUBLIC_RENDERED_TASK_DECODE')
  if 'family_claims' in row:issues.append('SUPERSEDED_PARALLEL_CLAIM_ALIAS')
  if row['track_id']!='RSM-001':issues.append('TRACK_ID')
  if rowhash(row)!=row['record_hash']:issues.append('FULL_RECORD_HASH_MISMATCH')
  if p is None or p['record_hash']!=row['record_hash'] or p['task_sha256']!=sha(engine.canonical(row['semantic_task'])) or p['rendered_input_sha256']!=sha(row['rendered_input'].encode()):issues.append('FROZEN_FIRST_JOIN_MISMATCH')
  targetmatch=p is not None and equal(p['expected_output'],gold)
  if not targetmatch:issues.append('FROZEN_EXPECTATION_GOLD_MISMATCH')
  if parent is None or not equal(parent['semantic_task'],row['semantic_task']) or any(type(parent[k]) is not type(row[k]) or parent[k]!=row[k] for k in ('semantic_family_id','source_family_key','source_family_index','split','controlled_context')):issues.append('SEMANTIC_PARENT_JOIN')
  if not integer(row['source_family_index']) or row['source_family_index'] not in indices or row['split']!=familiesplit:issues.append('SPLIT_INDEX')
  if not integer(row['difficulty']) or not integer(row['structural_difficulty']) or not 2<=row['difficulty']<=4 or row['structural_difficulty']!=row['difficulty']:issues.append('DIFFICULTY_TYPING')
  if not text(row.get('source_provenance',{}).get('curriculum_difficulty_explanation')):issues.append('MISSING_DIFFICULTY_EXPLANATION')
  if type(row['skill_tags']) is not list or not all(type(t) is str for t in row['skill_tags']):issues.append('DESCRIPTOR_TYPING')
  if row['review_status']!='PENDING' or row['review_receipt_ids']!=[]:issues.append('PENDING_CUSTODY')
  if row.get('source_provenance',{}).get('sealed_evaluation_used') is not False:issues.append('SEALED_PROVENANCE_UNCONFIRMED')
  if row['renderer']=='reverse':
   forward=style_inputs.get((row['source_family_key'],row['controlled_context'],'forward'))
   if forward is None:
    matching=[r for r in rows if r['source_family_key']==row['source_family_key'] and r['controlled_context']==row['controlled_context'] and r['renderer']=='forward']
    if len(matching)==1:forward=loads(matching[0]['rendered_input'].split('\nTask JSON:\n')[1])
   if forward is None or object_shape(decoded)!=object_shape(reversed_objects(forward)):issues.append('RECURSIVE_OBJECT_ORDER_RELATION')
  issues.extend(m['per_row_errors'].get(sid,[]));issues.extend(familyfacts[row['source_family_key']]['issues']);issues.extend(globalissues)
  recommendation='QUARANTINE' if any(x in issues for x in ['FULL_RECORD_HASH_MISMATCH','FROZEN_FIRST_JOIN_MISMATCH']) else 'REVISE' if issues else 'PASS_FACTUAL_CHECKS_ONLY'
  joins.append({'sample_id':sid,'record_hash':row['record_hash'],'FIRST_expected_sha256':sha(enc(p['expected_output'])) if p else None,'actual_gold_sha256':sha(enc(gold)),'typed_target_match':targetmatch,'FIRST_prediction_unchanged':True,'issues':issues})
  newjudgments.append({'receipt_id':'research_architect-canonical-machine:'+sid,'sample_id':sid,'record_hash':row['record_hash'],'reviewer_role':'independent_adjudicator_machine_comparison_only','reviewer_execution_id':'/root/research_architect','stage':'MACHINE_BASELINE_ACTUAL_AGENT_QUALITY_INSPECTION_PENDING','decision':'UNASSESSED_PENDING_ACTUAL_AGENT_INSPECTION','machine_check_recommendation':recommendation,'issues':issues,'typed_target_match':targetmatch,'reason':'Factual comparison of original frozen expectation without revision. This output is not an actual-agent quality vote; the agent must inspect the completed baseline and separately freeze its assessment before other reviews are read.','data_approval':False,'publication_eligible':None,'original_FIRST_preserved':True,'actual_agent_quality_assessment_completed':False})
 newjsonl(D/'canonical-prediction-join-r1.jsonl',joins);newjsonl(D/'canonical-machine-findings-r1.jsonl',newjudgments)
 audit={'actual_execution_id':'/root/research_architect','rows':len(rows),'parent_records':len(parentrows),'family_keys':len(families),'manifest_claims':m,'source_custody':sources,'family_evidence':familyfacts,'novelty_signatures':signatures,'legacy_signature_counts':{v:len(s) for v,s in legacy_sigs.items()},'global_issues':globalissues,'decision_counts':dict(collections.Counter(j['decision'] for j in newjudgments)),'machine_check_recommendation_counts':dict(collections.Counter(j['machine_check_recommendation'] for j in newjudgments)),'actual_agent_quality_assessment_completed':False,'publication_eligible':None,'other_reviews_read':False,'data_approval':False,'limitations':['finite supplied-program/state/control repetition','exposed dev and correlated descendants','same-model/design/adjudicator overlap','opaque source hashes are custody only','graph signatures are explicit chosen public normalization, not proof of all semantic mechanisms']};newjson(D/'canonical-baseline-audit-r1.json',audit)
 runtime_after=loaded_runtime(fr);after=verify_freeze(fr);receipt={'completed_execution':True,'actual_execution_id':'/root/research_architect','started_utc':start,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_freeze_sha256':sha(fr_raw),'bindings_before':before,'bindings_after':after,'runtime_modules_before':runtime_before,'runtime_modules_after':runtime_after,'all_frozen_bindings_unchanged':before==after,'canonical_rows':len(rows),'FIRST_expectations_compared':len(joins),'decision_counts':audit['decision_counts'],'other_reviews_or_root_tickets_read':False,'data_approval':False,'publication_eligible':None};newjson(D/'canonical-baseline-execution-r1.json',receipt)
 artifacts=[]
 for name in ['canonical-prediction-join-r1.jsonl','canonical-machine-findings-r1.jsonl','canonical-baseline-audit-r1.json','canonical-baseline-execution-r1.json']:
  raw=(D/name).read_bytes();artifacts.append({'path':str((D/name).relative_to(B)),'sha256':sha(raw),'bytes':len(raw)})
 newjson(D/'CANONICAL_COMPARISON-r1.json',{'actual_execution_id':'/root/research_architect','completed_execution':True,'stage':'MACHINE_COMPARISON_FROZEN_AGENT_ASSESSMENT_PENDING','actual_agent_quality_assessment_completed':False,'source_freeze_sha256':sha(fr_raw),'bindings':artifacts,'data_approval':False,'publication_eligible':None,'later_adjudication_requires_separate_grant':True})
 print(json.dumps({'completed_execution':True,'unassessed_counts':audit['decision_counts'],'machine_check_recommendation_counts':audit['machine_check_recommendation_counts'],'data_approval':False}))

def main():
 p=argparse.ArgumentParser();p.add_argument('--expected-freeze',required=True);p.add_argument('--expected-config',required=True);p.add_argument('--execute-after-separate-root-grant',action='store_true');a=p.parse_args()
 if not a.execute_after_separate_root_grant:raise SystemExit('prepared only; execution requires separate root grant')
 try:run(a.expected_freeze,a.expected_config)
 except Exception:
  newjson(D/'FAILED_EXECUTION-r1.json',{'actual_execution_id':'/root/research_architect','completed_execution':False,'traceback':traceback.format_exc(),'data_approval':False});raise
if __name__=='__main__':main()
