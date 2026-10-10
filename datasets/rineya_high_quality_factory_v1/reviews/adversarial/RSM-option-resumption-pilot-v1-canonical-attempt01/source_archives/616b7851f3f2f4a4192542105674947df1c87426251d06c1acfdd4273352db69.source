"""Owned RSM canonical review, prepared AFTER immutable216 input predictions.
DO NOT execute/import until a separate root grant. No private implementation reads.
The public schema is evaluated directly, and private asserted sources use opaque
byte hashes only. No other quality vote or expected-output source is consulted.
"""
import copy, datetime, hashlib, itertools, json, re, sys, time
from collections import Counter,defaultdict
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
ACTOR='/root/adversarial_auditor'
PREP=ROOT/'reviews/adversarial/RSM-option-resumption-public-preparation-20261010T1112Z'
CORRECTED=ROOT/'reviews/adversarial/RSM-option-resumption-public-interface-correction-20261010/attempt01'
INPUT=ROOT/'reviews/adversarial/RSM-option-resumption-pilot-v1-input-first-attempt01'

def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def semantic_hash(v):return hashlib.sha256(canon(v).encode()).hexdigest()
def bind(p):
    p=Path(p);raw=p.read_bytes()
    return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def record_hash(row):return semantic_hash({k:v for k,v in row.items() if k!='record_hash'})
def write(name,value,jsonl=False):
    p=OUT/name
    with p.open('x') as f:
        if jsonl:
            for r in value:f.write(canon(r)+'\n')
        else:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    if p.stat().st_size>25000000:raise ValueError('artifact exceeds cap')
    return bind(p)
def assert_true(v,msg):
    if not v:raise ValueError(msg)
def safe_file(raw):
    assert_true(type(raw) is str and raw and '\0' not in raw,'literal path')
    p=PurePosixPath(raw)
    assert_true(not p.is_absolute() and p.as_posix()==raw and all(x not in {'.','..'} for x in p.parts),'safe relative path')
    actual=ROOT
    for part in p.parts:
        actual=actual/part;assert_true(not actual.is_symlink(),'no symlink components')
    assert_true(actual.is_relative_to(ROOT) and actual.is_file(),'existing bound file')
    return actual
def validate_schema(value,schema):
    """Interpret the actual public canonical schema keywords, no private gate.
    Strict integers excludebool, consistent with task and public metadata contract.
    """
    types={'object':dict,'array':list,'string':str,'integer':int,'boolean':bool,'null':type(None)}
    if 'type' in schema:assert_true(type(value) is types[schema['type']],'schema type')
    if 'enum' in schema:assert_true(any(type(value) is type(v) and value==v for v in schema['enum']),'schema enum')
    if 'minimum' in schema:assert_true(value>=schema['minimum'],'schema minimum')
    if 'maximum' in schema:assert_true(value<=schema['maximum'],'schema maximum')
    if 'pattern' in schema:assert_true(re.search(schema['pattern'],value) is not None,'schema pattern')
    if type(value) is dict:
        assert_true(set(schema.get('required',[]))<=set(value),'schema required')
        if schema.get('additionalProperties') is False:assert_true(set(value)<=set(schema.get('properties',{})),'schema extra keys')
        for k,sub in schema.get('properties',{}).items():
            if k in value:validate_schema(value[k],sub)
    if type(value) is list and 'items' in schema:
        for v in value:validate_schema(v,schema['items'])
    return True
def path_value(value,path):
    assert_true(type(path) is list,'locator list')
    current=value
    for key in path:
        if type(current) is dict:assert_true(type(key) is str and key in current,'literal object locator');current=current[key]
        elif type(current) is list:assert_true(type(key) is int and 0<=key<len(current),'typed list index');current=current[key]
        else:raise ValueError('locator traverses scalar')
    return current
def set_leaf(value,path,new):
    assert_true(path,'nonempty leaf path');parent=path_value(value,path[:-1]);parent[path[-1]]=new
def trusted_suffix(task,machine,frame):
    current=copy.deepcopy(machine);events=[]
    while current['phase']=='running' and len(events)<100:events.append(frame.event_step(task,current))
    assert_true(current['phase']!='running','finite diagnostic')
    return events,current
def lift_legacy(task):
    """Lift only supplied target-free legacy primitives/main into common grammar."""
    assert_true(task['kind']=='skills','legacy kind')
    root='__own_legacy_main__'
    while root in task['procedures']:root+='x'
    def body(old):
        out=[]
        for ins in old:
            op=ins['op']
            if op in {'add','scale'}:out.append({'op':op,'register':ins['register'],'value':{'const':ins['value']}})
            elif op=='swap':out.append({'op':op})
            elif op=='call':out.append({'op':op,'option':ins['name'],'args':{}})
            elif op=='if':
                t=ins['test'];out.append({'op':op,'test':{'all':[{'left':{'reg':t['register']},'cmp':t['cmp'],'right':{'const':t['value']}}]},'then':body(ins['then']),'else':body(ins['else'])})
            else:raise ValueError('unsupported public legacy instruction')
        return out
    def definition(instructions):return {'params':[],'initiation':{'all':[]},'success':{'all':[]},'body':body(instructions)}
    options={name:definition(instructions) for name,instructions in task['procedures'].items()};options[root]=definition(task['main'])
    return {'kind':'option_resume','version':1,'initial_state':task['state'],'options':options,'entry':{'option':root,'args':{}},'mode':'fresh','history':[],'checkpoint':None,'event_budget':100}
def witness_facts(row,family,outputs,frame,recursive,controls,owninput):
    style=row['renderer'];F=family[('F',style)]['semantic_task'];R=family[('R',style)]['semantic_task'];X=family[('X',style)]['semantic_task']
    f=outputs[family[('F',style)]['sample_id']];r=outputs[family[('R',style)]['sample_id']]
    all_events,snapshots=recursive.full_execution(F);g=owninput.geometry(F,frame)
    metrics={'maximum_option_depth':max(sum(fr['frame_kind']=='option' for fr in cp['frames']) for cp in snapshots),
      'maximum_selected_branch_depth':max(sum(fr['frame_kind']=='branch' for fr in cp['frames']) for cp in snapshots),
      'declared_instructions':g['declared_instructions'],'total_F_events':len(all_events),'total_F_effects':sum(e['event']=='effect' for e in all_events),
      'history_effects_at_R':sum(e['event']=='effect' for e in R['history']),
      'caller_frames_at_R':sum(fr['frame_kind']=='option' for fr in R['checkpoint']['frames'])-1,
      'selected_branch_frames_at_R':sum(fr['frame_kind']=='branch' for fr in R['checkpoint']['frames'])}
    witness=row['property_witnesses'];assert_true(type(witness) is dict,'witness object')
    assert_true(set(witness)<= {'saved_binding_locator','checkpoint_tamper_locator','replay_locator','dependency_metrics','diagnostics'},'supported witness interface')
    if 'diagnostics' in witness:assert_true(type(witness['diagnostics']) is dict,'witness diagnostics')
    if 'dependency_metrics' in witness:
        asserted=witness['dependency_metrics'];assert_true(type(asserted) is dict and set(asserted)<=set(metrics),'supported metric keys')
        for name,value in asserted.items():assert_true(type(value) is int and value>=0 and value==metrics[name],'metric '+name)
    facts={'recomputed_metrics':metrics}
    if 'saved_binding_locator' in witness:
        p=witness['saved_binding_locator'];v=path_value(R['checkpoint'],p)
        assert_true(type(v) is int and len(p)==4 and p[0]=='frames' and type(p[1]) is int and p[2]=='bindings','saved-copy locator shape')
        modified=copy.deepcopy(R['checkpoint']);set_leaf(modified,p,v+1);changed,final=trusted_suffix(F,modified,frame)
        assert_true(not frame.typed_equal(changed,r['new_events']),'particular saved copy actually read in pending execution')
        facts['saved_binding_probe']={'actual_path':p,'original_value':v,'changed_value':v+1,'changes_actual_suffix':True,'diagnostic_final_state':final['state']}
    if 'checkpoint_tamper_locator' in witness:
        p=witness['checkpoint_tamper_locator'];path_value(R['checkpoint'],p);diff=controls.differences(R,X)
        assert_true(diff==[('checkpoint',)+tuple(p)],'actual tamper locator equals changed active scalar')
        changed,final=trusted_suffix(F,X['checkpoint'],frame)
        assert_true(not frame.typed_equal(changed,r['new_events']),'actual tamper affects pending work')
        facts['checkpoint_tamper_probe']={'actual_path':p,'diagnostic_final_state':final['state'],'canonical_error':outputs[family[('X',style)]['sample_id']]['error']}
    if 'replay_locator' in witness:
        p=witness['replay_locator'];assert_true(type(p) is dict and set(p)=={'option','body_path','index'} and type(p['option']) is str and type(p['index']) is int and p['index']>=0,'replay locator exact type')
        ins=controls.instruction_at(F,{'option':p['option'],'body_path':p['body_path'],'index':p['index']})
        assert_true(ins['op'] in {'add','scale','swap'},'replay points primitive')
        replay=controls.check_controls({c:family[(c,style)]['semantic_task'] for c in 'FRTPUAISX'}, {c:outputs[family[(c,style)]['sample_id']] for c in 'FRTPUAISX'})['repeated_effects']
        assert_true(any(e['option']==p['option'] and e['body_path']==p['body_path'] and e['index']==p['index'] for e in replay),'actual committed repeated nonidempotent primitive')
        facts['replay_actual_witnesses']=replay
    return facts

def main():
    config=json.loads((OUT/'RUN_CONFIG.json').read_text());freeze=json.loads((OUT/'SOURCE_FREEZE.json').read_text())
    def all_bindings():return [bind(ROOT/b['path'] if not Path(b['path']).is_absolute() else b['path']) for b in freeze['operative_bindings']]
    before=all_bindings();assert before==[{k:b[k] for k in ('path','size_bytes','sha256')} for b in freeze['operative_bindings']]
    assert_true((OUT/'EXECUTION_AUTHORIZATION.json').is_file(),'separate explicit ROOT execution grant required; source preparation is not grant')
    authorization=json.loads((OUT/'EXECUTION_AUTHORIZATION.json').read_text());assert_true(authorization['authorized_by']=='/root' and authorization['phase']=='canonical-comparison-execution','actual grant')
    start=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    sys.path[:0]=[str(PREP),str(CORRECTED),str(INPUT)]
    import frame_solver_r1 as frame
    import recursive_simulator_r1 as recursive
    import control_probes_r2 as controls
    import run_input_review_r1 as owninput
    read=lambda path:[frame.strict_json(x) for x in (ROOT/path).read_text().splitlines()]
    rows=read(config['candidate_path']);parents=read(config['parent_path']);legacy=read(config['legacy_input_path'])
    manifest=frame.strict_json((ROOT/config['manifest_path']).read_text());schema=frame.strict_json((ROOT/'schemas/canonical/v1.schema.json').read_text())
    first={r['sample_id']:r for r in read(str((INPUT/'frame_predictions.jsonl').relative_to(ROOT)))}
    second={r['sample_id']:r for r in read(str((INPUT/'recursive_predictions.jsonl').relative_to(ROOT)))}
    view={r['sample_id']:r for r in read('reviews/input_only/RSM-option-resumption-pilot-v1.jsonl')}
    checks=[];failures=[];issues=defaultdict(list);outputs={};members={};families=defaultdict(dict)
    def check(name,fn,sid=None,severity='HIGH',fix='Correct source in a fresh version; preserve original bytes and independently rereview.'):
        try:result=fn();assert_true(result is not False,name);okay=True;error=None
        except Exception as e:okay=False;error=type(e).__name__+': '+str(e);result=None
        item={'check':name,'passed':okay,'sample_id':sid}
        if not okay:
            item['evidence']=error;failures.append(item)
            if sid:issues[sid].append({'severity':severity,'code':name.split(':')[0],'evidence':error,'fix':fix})
        checks.append(item);return result
    check('216_rows',lambda:len(rows)==216);check('108_parents',lambda:len(parents)==108)
    assert_true(bind(ROOT/config['candidate_path'])['sha256']==config['candidate_sha256'],'frozen candidate SHA')
    for row in rows:
        sid=row['sample_id'];check('unique_id:'+sid,lambda:sid not in members,sid);members[sid]=row
        check('public_canonical_schema:'+sid,lambda:validate_schema(row,schema),sid)
        check('full_record_hash:'+sid,lambda:record_hash(row)==row['record_hash'],sid)
        check('pending_unapproved:'+sid,lambda:row['review_status']=='PENDING' and row['review_receipt_ids']==[],sid)
        check('exact_stripped_join:'+sid,lambda:all(frame.typed_equal(row[k],view[sid][k]) for k in view[sid]),sid)
        check('first_frame_prediction_join:'+sid,lambda:first[sid]['record_hash']==row['record_hash'] and first[sid]['typed_task_sha256']==semantic_hash(row['semantic_task']),sid)
        check('first_recursive_prediction_join:'+sid,lambda:second[sid]['record_hash']==row['record_hash'] and second[sid]['typed_task_sha256']==semantic_hash(row['semantic_task']),sid)
        check('immutable_frame_target:'+sid,lambda:frame.typed_equal(row['verified_target'],first[sid]['expected_output']),sid)
        check('immutable_recursive_target:'+sid,lambda:frame.typed_equal(row['verified_target'],second[sid]['expected_output']),sid)
        fresh=frame.solve(row['semantic_task']);fresh_recursive=recursive.solve(row['semantic_task']);outputs[sid]=fresh
        check('fresh_own_solver_agreement:'+sid,lambda:frame.typed_equal(fresh,fresh_recursive) and frame.typed_equal(fresh,first[sid]['expected_output']),sid)
        def cohortbounds():
            t=row['semantic_task'];geom=owninput.geometry(t,frame);events,snapshots=recursive.full_execution(t)
            assert_true(2<=geom['options']<=4 and all(1<=n<=3 for n in geom['parameters']) and geom['declared_instructions']<=24 and geom['conditional_depth']<=2,'declared cohort bounds')
            assert_true(all(type(n) is int and -6<=n<=6 for n in t['initial_state'].values()),'initial integer bounds')
            assert_true(len(events)<=48 and max(sum(fr['frame_kind']=='option' for fr in snap['frames']) for snap in snapshots)<=3,'actual complete event/option depth bounds')
            assert_true(all(-4096<=v<=4096 for snap in snapshots for v in snap['state'].values()),'actual complete register bounds')
            if row['controlled_context'] in {'F','R','U','A'}:assert_true(t['event_budget']==48,'full context budget')
            return True
        check('actual_all_context_cohort_bounds:'+sid,cohortbounds,sid)
        check('typed_metadata_coordinates:'+sid,lambda:type(row['source_family_index']) is int and type(row['source_family_key']) is str and row['controlled_context'] in controls.CONTEXTS and row['renderer'] in {'forward','reverse'},sid)
        key=(row['controlled_context'],row['renderer']);check('unique_family_coordinates:'+sid,lambda:key not in families[row['semantic_family_id']],sid);families[row['semantic_family_id']][key]=row
        check('difficulty_strict_range:'+sid,lambda:type(row['difficulty']) is int and type(row['structural_difficulty']) is int and 2<=row['difficulty']<=4 and row['difficulty']==row['structural_difficulty'],sid)
        prov=row['source_provenance']
        check('difficulty_explanation_present:'+sid,lambda:type(prov.get('curriculum_difficulty_explanation')) is str and bool(prov['curriculum_difficulty_explanation'].strip()),sid)
        check('target_format_structured:'+sid,lambda:row['target_format']=='structured',sid)
        check('source_origin_disclosure:'+sid,lambda:prov.get('origin')=='original_synthetic' and prov.get('common_model_prior_context') is True and prov.get('sealed_evaluation_used') is False,sid)
        # Finite proxy descriptors are not treated as property proofs.
        check('skill_descriptors_typed:'+sid,lambda:type(row['skill_tags']) is list and all(type(t) is str for t in row['skill_tags']),sid)
        if fresh['status']=='initiation_failed' and not row['semantic_task']['history'] and len(fresh['new_events'])==1 and fresh['new_effects']==0 and row['difficulty']>2:
            issues[sid].append({'severity':'MODERATE','code':'difficulty_dependency_not_demonstrated',
              'evidence':{'declared_level':row['difficulty'],'actual_visible_execution':'Fresh root fails initiation afteroneenter, no history/caller/branch/saved continuation needs execution','actual_event':fresh['new_events'][0],
                          'explanation':prov.get('curriculum_difficulty_explanation')},
              'fix':'Label this one-guard component Level2 or supply a specific defensible dependency-based explanation under the public curriculum; program grammar validation alone does not demonstrate joint retained-frame execution difficulty.'})
    parent_map={p['semantic_parent_id']:p for p in parents}
    for row in rows:
        sid=row['sample_id']
        def parentcheck():
            p=parent_map[row['semantic_parent_id']]
            assert_true(all(frame.typed_equal(p[k],row[k]) for k in ['semantic_family_id','semantic_task','source_family_index','source_family_key','controlled_context','split','generator_version']),'actual parent lineage/task binding')
            assert_true(p['status']=='PENDING','parent status');return True
        check('semantic_parent_binding:'+sid,parentcheck,sid)
    source_bytes={}
    for path,h in manifest['source_bindings'].items():
        p=safe_file(path);b=p.read_bytes();assert_true(hashlib.sha256(b).hexdigest()==h,'opaque/source binding '+path);source_bytes[path]=b
    check('manifest_public_header',lambda:controls.manifest_header(manifest,source_bytes,config['candidate_sha256'],'option-resumption-pilot-v1'))
    entries=manifest['control_pairs'];check('300_named_entries',lambda:len(entries)==300)
    keys=[];nonmember=0;claim_evidence=[]
    for row in rows:
        sid=row['sample_id'];keys_row=check('nine_owner_claims:'+sid,lambda:controls.owner_claims(row),sid)
        for claim in row['family_control_claims']:
            key=controls.claim_key(claim);keys.append(key);nonmember+=sid not in claim['pair_sample_ids']
            def resolve():
                assert_true(safe_file(claim['evidence_manifest'])==ROOT/config['manifest_path'],'actual manifest reference')
                return controls.resolve_claim(row,claim,entries,members,config['candidate_sha256'])
            e=check('actual_owner_lookup:'+sid+':'+key[1],resolve,sid)
            claim_evidence.append({'owner_sample_id':sid,'owner_record_hash':row['record_hash'],'lookup_key':[key[0],key[1],list(key[2])],'resolved':e is not None,'nonmember_owner':sid not in claim['pair_sample_ids']})
    check('1944_owner_claims',lambda:len(keys)==1944);check('1344_nonmember_owner_claims',lambda:nonmember==1344)
    named_keys=[]
    for e in entries:
        name=controls.identifier(e);pair=e['pair_sample_ids'];named_keys.append((e['control_id'],name,tuple(pair)))
        check('entry_pair_types:'+e['control_id']+':'+name,lambda:type(pair) is list and len(pair)==2 and all(type(v) is str for v in pair) and pair[0]!=pair[1])
    check('unique_named_lookup_keys',lambda:len(set(named_keys))==300)
    check('all_entries_used_by_actual_claims',lambda:set(named_keys)==set(keys))
    cfg=json.loads((ROOT/'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json').read_text());byindex={i:s for s,indices in cfg['source_family_indices_by_stratum'].items() for i in indices}
    structural={};family_evidence=[];witness_evidence=[]
    check('12_actual_families',lambda:len(families)==12)
    for fam,group in families.items():
        check('18_family_coordinates:'+fam,lambda:set(group)==set(itertools.product(controls.CONTEXTS,{'forward','reverse'})))
        if set(group)!=set(itertools.product(controls.CONTEXTS,{'forward','reverse'})):continue
        frow=group[('F','forward')];index=frow['source_family_index'];expected_split='train' if index%10<=7 else 'dev_challenge'
        structural[fam]=controls.program_signatures(frow['semantic_task'])
        for row in group.values():
            sid=row['sample_id']
            check('same_family_lineage:'+sid,lambda:row['source_family_key']==frow['source_family_key'] and row['source_family_index']==index and row['split']==expected_split,sid)
            check('source_stratum_actual_assignment:'+sid,lambda:row['source_provenance'].get('source_stratum')==byindex[index],sid)
            facts=check('actual_optional_witnesses:'+sid,lambda:witness_facts(row,group,outputs,frame,recursive,controls,owninput),sid)
            witness_evidence.append({'sample_id':sid,'record_hash':row['record_hash'],'facts':facts})
        for style in ('forward','reverse'):
            tasks={c:group[(c,style)]['semantic_task'] for c in controls.CONTEXTS};out={c:outputs[group[(c,style)]['sample_id']] for c in controls.CONTEXTS}
            facts=check('actual_eight_semantic_controls:'+fam+':'+style,lambda:controls.check_controls(tasks,out))
            if facts is None:
                for row in group.values():issues[row['sample_id']].append({'severity':'HIGH','code':'actual_family_control_failure','evidence':fam+' '+style,'fix':'Repair affected family in a fresh version and re-review all controls.'})
            family_evidence.append({'family':fam,'renderer':style,'actual_relations':facts})
        for c in controls.CONTEXTS:
            a,b=group[(c,'forward')],group[(c,'reverse')]
            pa=frame.strict_json(a['rendered_input'].rsplit('Task JSON:\n',1)[1]);pb=frame.strict_json(b['rendered_input'].rsplit('Task JSON:\n',1)[1])
            check('actual_renderer_typed_pair:'+a['sample_id'],lambda:frame.typed_equal(pa,pb) and frame.typed_equal(pa,a['semantic_task']) and frame.typed_equal(pb,b['semantic_task']),a['sample_id'])
            check('actual_recursive_order_reversal:'+a['sample_id'],lambda:json.dumps(controls.reverse_objects(pa),separators=(',',':'))==json.dumps(pb,separators=(',',':')),a['sample_id'])
    legacy_sigs={r['sample_id']:controls.program_signatures(lift_legacy(r['semantic_task'])) for r in legacy}
    novelty={}
    for name in ('full','coarse_all','coarse_reachable'):
        newset={s[name] for s in structural.values()};oldset={s[name] for s in legacy_sigs.values()};collisions=newset&oldset
        novelty[name]={'new_unique_graphs':len(newset),'legacy_unique_graphs':len(oldset),'legacy_views':len(legacy),'legacy_lineages':len({r['semantic_family_id'] for r in legacy}),
                       'cross_collision_signatures':sorted(collisions),'new_signatures':{f:s[name] for f,s in structural.items()},'legacy_signatures':{i:s[name] for i,s in legacy_sigs.items()}}
        check('12distinct_no_legacy_collision:'+name,lambda:len(newset)==12 and not collisions)
        if len(newset)!=12 or collisions:
            for row in rows:issues[row['sample_id']].append({'severity':'HIGH','code':'normalized_family_growth_not_established','evidence':novelty[name],'fix':'Correct novelty/source-family claims; semantic variants cannot establish new lineages.'})
    # Actual mutations of in-memory copies only, and handmade interface controls.
    regression=[]
    def rejection(name,fn):
        try:fn();okay=False;error='unexpected acceptance'
        except Exception as e:okay=True;error=type(e).__name__+': '+str(e)
        regression.append({'check':name,'passed':okay,'actual_result':error})
    owner=rows[0];claim=copy.deepcopy(owner['family_control_claims'][0]);entry=next(e for e in entries if (e['control_id'],controls.identifier(e),tuple(e['pair_sample_ids']))==controls.claim_key(claim))
    for label,mutator in [('reverse_pair',lambda c:c['pair_sample_ids'].reverse()),('bool_index',lambda c:c.update(source_family_index=True)),('empty_control_id',lambda c:c.update(control_id='')),('forbidden_canonical_hashes',lambda c:c.update(pair_record_hashes=['0'*64]*2)),('missing_member',lambda c:c['pair_sample_ids'].__setitem__(0,'own-missing-member')),('wrong_context',lambda c:c.update(controlled_context='X'))]:
        c=copy.deepcopy(claim);mutator(c);rejection(label,lambda:controls.resolve_claim(owner,c,entries,members,config['candidate_sha256']))
    rejection('duplicate_complete_lookup_key',lambda:controls.resolve_claim(owner,claim,entries+[copy.deepcopy(entry)],members,config['candidate_sha256']))
    e2=copy.deepcopy(entries);selected=next(e for e in e2 if e['control_id']==entry['control_id'] and controls.identifier(e)==controls.identifier(entry));selected['candidate_sha256']='0'*64
    rejection('wrong_actual_candidate_SHA',lambda:controls.resolve_claim(owner,claim,e2,members,config['candidate_sha256']))
    e2=copy.deepcopy(entries);selected=next(e for e in e2 if e['control_id']==entry['control_id'] and controls.identifier(e)==controls.identifier(entry));selected['pair_record_hashes'].reverse()
    rejection('reversed_external_hashes',lambda:controls.resolve_claim(owner,claim,e2,members,config['candidate_sha256']))
    allowed_claim=copy.deepcopy(claim);allowed_claim['pair_sample_ids']=['','nonempty'];allowed_claim['control_id']=' '
    regression.append({'check':'empty_but_distinct_IDs_and_literal_whitespace_controlID','passed':controls.claim_key(allowed_claim)==(' ',controls.identifier(claim),('','nonempty'))})
    optional=copy.deepcopy(entries)
    for e in optional:e.pop('pair_record_hashes',None)
    regression.append({'check':'optional_external_hash_absence_valid','passed':controls.resolve_claim(owner,claim,optional,members,config['candidate_sha256']) is not None})
    bool_row=copy.deepcopy(owner);bool_row['difficulty']=True;rejection('canonical_bool_integer_type',lambda:validate_schema(bool_row,schema))
    altered=copy.deepcopy(owner);altered['verified_target']['status']='invalid';altered['record_hash']=record_hash(altered)
    rejection('rehashed_wrong_target_still_fails_first_prediction',lambda:assert_true(frame.typed_equal(altered['verified_target'],first[owner['sample_id']]['expected_output']),'wrong target'))
    for r in regression:checks.append({'check':'own_regression:'+r['check'],'passed':r['passed'],'evidence':r})
    failures += [r for r in regression if not r['passed']]
    batch_failures=[f for f in failures if f.get('sample_id') is None]
    if batch_failures:
        for row in rows:issues[row['sample_id']].append({'severity':'HIGH','code':'unresolved_batch_or_review_regression_defect','evidence':batch_failures,'fix':'Resolve with fresh source/batch and retain first failure artifacts.'})
    evidence=write('CANONICAL_CONTROL_WITNESS_NOVELTY_EVIDENCE.json',{'actor':ACTOR,'witnesses':witness_evidence,'family_controls':family_evidence,'claims':claim_evidence,'novelty':novelty,'regressions':regression,'read_scope':'authorized canonical gold after frozen own FIRST predictions; private source bytes opaque only'})
    receipts=[]
    for row in rows:
        sid=row['sample_id'];local=issues[sid];strong=any(i['severity'] in {'HIGH','CRITICAL'} for i in local)
        moderate=any(i['code']=='difficulty_dependency_not_demonstrated' for i in local)
        decision='QUARANTINE' if strong else 'REVISE' if moderate else 'ACCEPT'
        target_ok=frame.typed_equal(row['verified_target'],first[sid]['expected_output']) and frame.typed_equal(row['verified_target'],second[sid]['expected_output'])
        local += [{'severity':'MODERATE','code':'partial_status_shortcut_scope','evidence':{'actual_own_input_supplement_sha256':'ca59f56ff9129af23213b23286e0904619fd9461495336987435ea19b3e512cb','status_baseline_dev_correct':68,'dev_rows':72,'full_targets_solved':False,'true_predicate_counterexamples':40},
                  'fix':'For future competency evaluation cross superficial success/argument/cut patterns with statuses; report exact-output versus status-only scores separately and expose finite12family/dev scope.'}]
        receipts.append({'receipt_id':'red-rsm-canonical-r1-'+semantic_hash([sid,row['record_hash']])[:24],'sample_id':sid,'record_hash':row['record_hash'],
          'reviewer_role':'adversarial','reviewer_execution_id':ACTOR,'scores':{'semantic_correctness':1 if strong else 4,'target_correctness':5 if target_ok else 1,
          'instruction_clarity':4,'reasoning_validity':1 if strong else 4,'generalization_value':2,'difficulty_quality':1 if moderate else 3,'diversity':3,'shortcut_resistance':2},
          'critical_defects':[i['code'] for i in local if i['severity'] in {'HIGH','CRITICAL'}],'issues':local,'decision':decision,'confidence':0.92,
          'evidence':{'immutable_first_prediction_sha256':'d67500ab37d1835d21511fe8cf6a9934cf8dd78cd019abddc5d13846567d466b','recursive_first_sha256':'231679b1cdc1b31785612a9bf4baab5131ff32d5c2841903835015afd19059c6','canonical_review_evidence_sha256':evidence['sha256']},
          'scope_disclosure':'Same underlying model/shared public design/own grammar and handmade reuse; canonical gold only after immutable216input predictions. Other reviewer/formal/producer semantic implementations notread. No sealedOOD, efficacy, fullpublicationapproval or status-baseline exact-output claim.'})
    first_receipts=write('FIRST_CANONICAL_JUDGMENTS.jsonl',receipts,True);final_receipts=write('FINAL_CANONICAL_JUDGMENTS.jsonl',receipts,True)
    designated=ROOT/config['final_receipt_path']
    with designated.open('x') as f:
        for r in receipts:f.write(canon(r)+'\n')
    after=all_bindings();assert before==after
    execution=write('EXECUTION.json',{'actor':ACTOR,'started_at_utc':started,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
        'operative_freeze':bind(OUT/'SOURCE_FREEZE.json'),'grant':bind(OUT/'EXECUTION_AUTHORIZATION.json'),'actual_checks':checks,'failed_checks':failures,
        'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),'candidate':bind(ROOT/config['candidate_path']),
        'decisions':dict(Counter(r['decision'] for r in receipts)),'outputs':[evidence,first_receipts,final_receipts,bind(designated)],
        'before_bindings':before,'after_bindings':after,'all_frozen_bytes_unchanged':True,'no_other_review_quality_vote':True,
        'opaque_private_bindings_never_inspected_or_imported':True,'initial_shortcut_evidence_preserved':True,'dataset_publication_approval':False})
    print(canon({'execution':execution,'decisions':dict(Counter(r['decision'] for r in receipts)),'designated':bind(designated)}))
if __name__=='__main__':main()
