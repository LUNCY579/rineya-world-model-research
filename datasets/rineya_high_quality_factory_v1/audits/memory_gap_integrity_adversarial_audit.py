#!/usr/bin/env python3
"""Supplement frozen blind judgments: public metadata, robust motifs and typed joins.
Never opens producer/oracle/reviewer code or generation manifests.
Requested pipeline/export_validation functions are AST-extracted with OWN solver injection.
"""
import ast,copy,collections,hashlib,json,pathlib,tempfile,sys
from adversarial_audit import ROOT,EXECUTION,canon,digest,require
from extended_adversarial_audit import episodic
from memory_gap_adversarial_audit import BATCH,P,OUT,decode,answer_sem,sha,typename,baseline

REPORT=ROOT/'audits/execution'/f'{BATCH}-integrity-adversarial.json'

def write_report(o):
    if REPORT.exists() and REPORT.read_text()!=json.dumps(o,indent=2,ensure_ascii=False)+'\n':
        hist=REPORT.parent/'history'/f'{REPORT.stem}-{sha(REPORT)}.json';hist.parent.mkdir(parents=True,exist_ok=True)
        if not hist.exists():hist.write_bytes(REPORT.read_bytes())
    REPORT.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')

def event_order_motif(t):
    _,_,z=decode(t);bykey=collections.defaultdict(list)
    for e in z['events']:bykey[e['key']].append(e)
    # Sort by meaningful key/version/time; retain original order only within exact tuple ties.
    events=sorted(z['events'],key=lambda e:(e['key'],e['version'],e['at']))
    version_rank={k:{v:i for i,v in enumerate(sorted({e['version'] for e in es}))} for k,es in bykey.items()}
    time_rank={k:{v:i for i,v in enumerate(sorted({e['at'] for e in bykey[k]}|{q['as_of'] for q in z['queries'] if q['key']==k}))} for k in {q['key'] for q in z['queries']}}
    symbols={};n=collections.Counter();norm=[]
    for e in events:
        v=e['value']
        if v is not None:
            c=canon(v);typ=typename(v)
            if c not in symbols:symbols[c]=typ+'_'+str(n[typ]);n[typ]+=1
            v=symbols[c]
        norm.append({**e,'version':version_rank[e['key']][e['version']],'at':time_rank[e['key']][e['at']],'value':v})
    return {'kind':'episodic','events':norm,'queries':[{**q,'as_of':time_rank[q['key']][q['as_of']]} for q in z['queries']]}

def own_schema_extract(tree):
    names=('schema_validate',);nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    # No factory.reference imports execute; record hashing and canonical strings are own helpers.
    scope={'json':json,'ROOT':ROOT,'canonical':canon,'record_hash':lambda r:digest({k:v for k,v in r.items() if k!='record_hash'}),'REQUIRED':{'sample_id','semantic_parent_id','semantic_family_id','track_id','skill_tags','difficulty','generator_version','oracle_version','generation_seed','source_provenance','semantic_task','rendered_input','verified_target','target_format','split','review_status','review_receipt_ids','record_hash'}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'requested_pipeline_schema_extract','exec'),scope)
    return scope['schema_validate']

def own_specialized_extract(tree):
    from jsonschema import Draft202012Validator
    def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    def own_solve(t):require(t['kind']=='episodic','This code review test covers episodic joins only');return episodic(t)
    scope={'json':json,'ROOT':ROOT,'canonical':canon,'digest':digest,'load_jsonl':load,'Draft202012Validator':Draft202012Validator,'solve':own_solve}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_specialized']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'requested_specialized_validation_extract','exec'),scope)
    return scope['validate_specialized']

def main():
    bound={str(p.relative_to(ROOT)):sha(p) for p in [P,OUT/f'{BATCH}.input_predictions.jsonl',OUT/f'{BATCH}.first_judgments.jsonl',OUT/f'{BATCH}.jsonl']}
    rows=[json.loads(s) for s in P.read_text().splitlines()];pred={r['sample_id']:r for r in [json.loads(s) for s in (OUT/f'{BATCH}.input_predictions.jsonl').read_text().splitlines()]}
    final={r['sample_id']:r for r in [json.loads(s) for s in (OUT/f'{BATCH}.jsonl').read_text().splitlines()]}
    require(len(rows)==len(pred)==len(final)==768,'exact one receipt/prediction per row')
    for r in rows:require(pred[r['sample_id']]['record_hash']==final[r['sample_id']]['record_hash']==r['record_hash'],'frozen exact hash binding')
    doc=ROOT/'schemas/MEMORY_CONTROL_METADATA.md';public_hash={str(doc.relative_to(ROOT)):sha(doc)}
    legal_controls={'query_role_id_position_crossing','key_role_crossing','field_order_invariance','lower_version_append_invariance','other_key_edit_invariance','future_event_edit_invariance','version_priority_counterfactual','active_input_tie_swap_counterfactual','null_lifecycle_counterfactual','query_key_counterfactual'}
    corpus={'query_role_id_position_crossing','key_role_crossing','field_order_invariance','version_priority_counterfactual','active_input_tie_swap_counterfactual','null_lifecycle_counterfactual'}
    probe={'lower_version_append_invariance','other_key_edit_invariance','future_event_edit_invariance','version_priority_counterfactual','active_input_tie_swap_counterfactual','null_lifecycle_counterfactual','query_key_counterfactual'}
    metadata_errors=[];scopes=collections.Counter();role_signature=collections.defaultdict(set);schedule=collections.defaultdict(list);role_tables=[];orderchecks=[];motifs=collections.defaultdict(list);weak_motifs=collections.defaultdict(list);obs=[]
    trainkeys=set();devkeys=set();trainvalues=collections.defaultdict(set);devvalues=collections.defaultdict(set)
    for r in rows:
        t=json.loads(r['rendered_input'].split('Task JSON:\n',1)[1]);a=pred[r['sample_id']]['input_derived_answer'];group,roles,norm=decode(t)
        try:
            require(type(r['family_control_claims']) is list,'claim list')
            for c in r['family_control_claims']:
                require(type(c) is dict and set(c)<={'name','property','scope','source_family_key','evidence_manifest'},'declared claim fields')
                identifiers=[k for k in ('name','property') if k in c];require(len(identifiers)==1,'one identifier encoding');name=c[identifiers[0]]
                require(name in legal_controls and c.get('scope') in ('corpus_variant','audit_probe'),'known property/scope')
                require('source_family_key' not in c or c['source_family_key']==r['source_family_key'],'same-family claim')
                if 'evidence_manifest' in c:
                    pointer=pathlib.PurePosixPath(c['evidence_manifest']);require(type(c['evidence_manifest']) is str and not pointer.is_absolute() and '..' not in pointer.parts,'relative provenance pointer')
                require(name in (corpus if c['scope']=='corpus_variant' else probe),'declared coverage instantiated at scope');scopes[(name,c['scope'])]+=1
            groups=r['key_role_assignments'];require(set(groups)=={'absent','delayed','index_tie','lifecycle'} and len(set(groups.values()))==4,'four key assignments')
            translated={'absent':'absence','delayed':'version_recency','index_tie':'active_index_tie','lifecycle':'null_lifecycle'}
            for k,v in groups.items():require(group[v]==translated[k],'declared key group actual input mismatch')
            keysizes=[len(v) for v in groups.values()]
            require(all(type(v) is str and v.islower() and v.isalpha() for v in groups.values()),'same simple lowercase alphabetic style')
            assigned=r['query_role_assignments'];require(len(assigned)==8 and len({x['role'] for x in assigned})==8,'eight declared distinct roles')
            require({x['position'] for x in assigned}==set(range(8)),'all positions')
            for x in assigned:
                require(set(x)=={'role','query_id','position'} and type(x['position']) is int and 0<=x['position']<8 and type(x['role']) is str,'query assignment shape')
                q=t['queries'][x['position']];require(x['query_id']==q['query_id'],'visible query ID assignment')
                role_signature[(r['source_family_key'],r['controlled_variant'],x['role'])].add((group[q['key']],q['as_of']))
            schedule[(r['source_family_key'],r['controlled_variant'],r['renderer'])].append(r)
        except Exception as e:metadata_errors.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'error':str(e)})
        # Event groups can move, but equal-key/version/time groups retain internal input order.
        tiegroup=collections.defaultdict(list)
        for i,e in enumerate(t['events']):tiegroup[(e['key'],e['version'],e['at'])].append(i)
        perm=[i for k in sorted(tiegroup,reverse=True) for i in tiegroup[k]];inverse={old:new for new,old in enumerate(perm)};z=copy.deepcopy(t);z['events']=[z['events'][i] for i in perm];expected=copy.deepcopy(a)
        for x in expected['results']:
            if x['chosen_event'] is not None:x['chosen_event']=inverse[x['chosen_event']]
        require(canon(episodic(z))==canon(expected),'event group order invariance and required reindexing');orderchecks.append(r['sample_id'])
        motifs[digest(event_order_motif(t))].append(r)
        # Operation/query mechanism only: keep null lifecycle + tied event order + cutoff relations.
        weak=event_order_motif(t)
        for e in weak['events']:
            if e['value'] is not None:e['value']='nonnull_'+str(e['value']).rsplit('_',1)[-1]
        weak_motifs[digest(weak)].append(r)
        keyset={e['key'] for e in t['events']}|{q['key'] for q in t['queries']};(trainkeys if r['split']=='train' else devkeys).update(keyset)
        for e in t['events']:(trainvalues if r['split']=='train' else devvalues)[typename(e['value'])].add(canon(e['value']))
        counts=collections.Counter(e['key'] for e in t['events'])
        for j,(q,x) in enumerate(zip(t['queries'],a['results'])):obs.append({'split':r['split'],'id':q['query_id'],'position':j,'key':q['key'],'event_count':counts[q['key']],'input_derived_semantic_group':group[q['key']],'answer':x})
    for key,v in role_signature.items():
        if len(v)!=1:metadata_errors.append({'family_history_role':key,'unstable_decoded_signatures':sorted(v)})
    for k,rs in sorted(schedule.items()):
        role_tables.append({'family':k[0],'history':k[1],'renderer':k[2],'assignments':[{'sample_id':r['sample_id'],'record_hash':r['record_hash'],'query_rotation':r['query_rotation'],'key_rotation':r['key_rotation'],'key_role_assignments':r['key_role_assignments'],'query_role_assignments':r['query_role_assignments']} for r in sorted(rs,key=lambda r:r['query_rotation'])]})
    train=[o for o in obs if o['split']=='train'];dev=[o for o in obs if o['split']!='train']
    semantic_baselines={n:{kind:baseline(train,dev,fn,kind) for kind in ('status','type','present_null','full_query_target')} for n,fn in {'visible_exact_key_event_count':lambda o:o['event_count'],'input_derived_semantic_group_requires_pattern_decoding':lambda o:o['input_derived_semantic_group']}.items()}
    # Previously reviewed public candidate inputs; never memory-gap-v1 or prior reviewer files.
    prior_paths=[ROOT/'data/REM-001/candidate_renderings'/f'{b}.jsonl' for b in ('extended-pilot-v2','extended-pilot-v3')]
    prior_tasks=set();prior_families=set();prior_n=0
    for p in prior_paths:
        if not p.exists():continue
        for s in p.read_text().splitlines():
            r=json.loads(s);t=json.loads(r['rendered_input'].split('Task JSON:\n',1)[1]);prior_tasks.add(digest(t));prior_families.add(r.get('source_family_key',r['semantic_family_id']));prior_n+=1
    exact_old=[r['sample_id'] for r in rows if digest(json.loads(r['rendered_input'].split('Task JSON:\n',1)[1])) in prior_tasks]
    family_old=sorted({r['source_family_key'] for r in rows}&prior_families)
    # Review the two requested generic code files only after blind first judgments frozen.
    codefiles=[ROOT/'factory/pipeline.py',ROOT/'factory/export_validation.py',ROOT/'factory/common.py'];codehash={str(p.relative_to(ROOT)):sha(p) for p in codefiles};pipeline_ast=ast.parse(codefiles[0].read_text());export_ast=ast.parse(codefiles[1].read_text())
    schema_check=own_schema_extract(pipeline_ast)
    for r in rows:schema_check(r)
    svchecks=[]
    # Mutations chosen to pass Python == while altering exact JSON scalar types.
    boolrow=next(r for r in rows if any(type(e['value']) is bool for e in r['semantic_task']['events']))
    for name,field in [('boolean_value_to_integer','value'),('integer_timestamp_to_float','at'),('integer_version_to_float','version')]:
        z=copy.deepcopy(boolrow)
        if field=='value':i=next(i for i,e in enumerate(z['semantic_task']['events']) if type(e['value']) is bool);z['semantic_task']['events'][i]['value']=int(z['semantic_task']['events'][i]['value'])
        else:i=0;z['semantic_task']['events'][i][field]=float(z['semantic_task']['events'][i][field])
        require(json.loads(z['rendered_input'].split('Task JSON:\n',1)[1])==z['semantic_task'],'mutation must demonstrate Python equality weakness');z['record_hash']=digest({k:v for k,v in z.items() if k!='record_hash'})
        try:schema_check(z)
        except Exception as e:svchecks.append({'mutation':name,'rejected':True,'reason':str(e)})
        else:raise ValueError('typed rendering mutation accepted '+name)
    specialized_check=own_specialized_extract(export_ast);specialchecks=[]
    row=next(r for r in rows if any(type(x['value']) is bool for x in episodic(r['semantic_task'])['results']) and r['split']=='train');answer=episodic(row['semantic_task']);item={'input':row['rendered_input'],'target':answer}
    with tempfile.TemporaryDirectory(prefix='red-memory-joins-') as tmp:
        d=pathlib.Path(tmp);f=d/'train-memory_updates.jsonl';f.write_text(canon(item)+'\n');counts=specialized_check(d,[row]);require(counts[f.name]==1,'valid memory export rejected');specialchecks.append({'test':'own typed full memory export','passed':True})
        z=copy.deepcopy(item);i=next(i for i,x in enumerate(z['target']['results']) if type(x['value']) is bool);z['target']['results'][i]['value']=int(z['target']['results'][i]['value']);require(z==item,'mutation must pass Python equality');f.write_text(canon(z)+'\n')
        try:specialized_check(d,[row])
        except AssertionError:specialchecks.append({'test':'memory exported boolean label changed to equal integer','rejected':True})
        else:raise ValueError('typed memory export mutation accepted')
        # All ordering and values remain fixed; a mismatched split must also fail.
        f.unlink();f=d/'dev_challenge-memory_updates.jsonl';f.write_text(canon(item)+'\n')
        try:specialized_check(d,[row])
        except AssertionError:specialchecks.append({'test':'memory export outside canonical split','rejected':True})
        else:raise ValueError('wrong split memory export accepted')
    # Static inspected join sites; own tests cover typed schema + specialized memory target.
    joins=[]
    for file,tree,names in [('factory/pipeline.py',pipeline_ast,{'schema_validate','audit','validate_release'}),('factory/export_validation.py',export_ast,{'validate_specialized'})]:
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in names:
                for n in ast.walk(fn):
                    if isinstance(n,ast.Compare) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='canonical' for x in ast.walk(n)):
                        joins.append({'file':file,'function':fn.name,'line':n.lineno,'expression':ast.unparse(n)})
    unchanged={str(p.relative_to(ROOT)):sha(p) for p in [P,OUT/f'{BATCH}.input_predictions.jsonl',OUT/f'{BATCH}.first_judgments.jsonl',OUT/f'{BATCH}.jsonl']};require(unchanged==bound,'frozen candidate/receipt bytes changed');require({str(p.relative_to(ROOT)):sha(p) for p in codefiles}==codehash,'requested gate source changed during review')
    report={'batch':BATCH,'reviewer_execution_id':EXECUTION,'status':'completed','decision':'PASS_WITH_EXPLICIT_SCOPE_LIMITS' if not metadata_errors else 'REVISE_METADATA','frozen_candidate_and_receipt_hashes':bound,'script_sha256':sha(pathlib.Path(__file__)),'public_metadata_doc_hash':public_hash,'requested_gate_code_hashes':codehash,'metadata_interface_errors':metadata_errors,'family_control_claim_scope_counts':{canon(k):v for k,v in sorted(scopes.items())},'actual_declared_role_signature_groups':len(role_signature),'declared_roles_retain_stable_decoded_signatures':not metadata_errors,'producer_expectation_sources_not_read':['property_witnesses','counterfactual_baseline','evidence_manifest contents','generator/reference/oracle code','prior memory-gap-v1','expert/formal/adjudication verdicts'],'public_metadata_assignments_checked_after_initial_judgments_freeze':True,'assignment_schedules':role_tables,'typed_schema_records_passed':len(rows),'typed_rendering_join_mutation_rejections':svchecks,'typed_specialized_memory_join_tests':specialchecks,'static_canonical_join_sites':joins,'event_group_permutation_with_ties_preserved_and_winner_reindexed':len(orderchecks),'order_invariant_typed_relation_motifs':len(motifs),'order_invariant_typed_relation_motifs_cross_split':sum(len({r['split'] for r in rs})>1 for rs in motifs.values()),'order_invariant_typed_relation_motif_groups':[{'motif_hash':h,'records':len(rs),'families':sorted({r['source_family_key'] for r in rs}),'splits':sorted({r['split'] for r in rs})} for h,rs in motifs.items()],'operation_query_relation_motifs_with_nonnull_types_also_stripped':len(weak_motifs),'operation_query_relation_motifs_cross_split':sum(len({r['split'] for r in rs})>1 for rs in weak_motifs.values()),'operation_query_relation_motif_groups':[{'motif_hash':h,'records':len(rs),'families':sorted({r['source_family_key'] for r in rs}),'splits':sorted({r['split'] for r in rs})} for h,rs in weak_motifs.items()],'semantic_partial_label_baselines':semantic_baselines,'vocabulary_controls':{'train_key_count':len(trainkeys),'dev_key_count':len(devkeys),'shared_keys':sorted(trainkeys&devkeys),'shared_typed_values':{k:sorted(trainvalues[k]&devvalues[k]) for k in sorted(set(trainvalues)|set(devvalues))},'scope':'Held-out keys and string/integer values checked explicitly; null and boolean literals remain shared semantic types, not a sealed mechanism holdout.'},'prior_reviewed_public_candidate_lineage':{'candidate_files':[str(p.relative_to(ROOT)) for p in prior_paths if p.exists()],'rows':prior_n,'exact_task_reuse':exact_old,'source_family_key_reuse':family_old,'scope':'Only previously own-reviewed extended pilot inputs; no memory-gap-v1 or external/protected benchmarks inspected.'},'material_blockers':metadata_errors,'scope_limits':['All 12 families use the same public max-tuple retrieval mechanism. Variants and numeric/vocabulary instances are correlated; motif count is not independent mechanism count.','Event-order-conservative initial metric reports 0 shared cross-split shapes; the additional event-group-order-invariant relation metric above is the appropriate broader template overlap disclosure.','Status majority baseline is 65.625%; IDs/positions and key names do not improve status. Coarse event-count group predicts partial labels but does not recover exact values/version/index and is public semantic content.','All supplied absent queries are never-key absence; future-only absence is uncovered. Eight independent queries do not establish memory composition.','Incorrect version-then-index resolver is correct on 94.79% of individual query outputs, despite failing the targeted timestamp controls; evaluate complete tasks and property subsets to avoid masked semantic defects.','Same-model reviewers and exposed two-family dev are not independent model evidence or sealed efficacy evaluation.','Generic validator code tests inject own public-contract episodic solver, never call/import private oracle/reference. Static canonical comparisons inspected elsewhere, not a full multi-track semantic rerun.','No candidate/release/history/other-reviewer bytes changed; final semantic receipts remain frozen.']}
    write_report(report);print(canon({'report':str(REPORT),'sha256':sha(REPORT),'metadata_errors':len(metadata_errors),'typed_motifs':len(motifs),'typed_motifs_cross_split':report['order_invariant_typed_relation_motifs_cross_split'],'type_stripped_motifs':len(weak_motifs),'type_stripped_motifs_cross_split':report['operation_query_relation_motifs_cross_split'],'code_hashes':codehash,'partial_label_baselines':semantic_baselines,'vocabulary':report['vocabulary_controls']}))

if __name__=='__main__':main()
