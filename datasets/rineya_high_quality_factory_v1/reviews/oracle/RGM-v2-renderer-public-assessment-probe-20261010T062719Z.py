"""Bounded guided input-only diagnostic; actual author /root/formal_oracle.

This is a NEW frozen diagnostic, not a revised gate. No source or candidate
mutation. Actual gold/reviewer predictions are never accessed.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import traceback

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from oracles import rgm_planning_controls_v2_independent as oracle
from oracles import rgm_planning_controls_independent as v1
from factory import planning_controls as integration

AUTHORIZED = {
 'oracles/rgm_planning_controls_v2_independent.py':'69944b8e06fb984947d0d686296d0bb886aecc358ae6f7405d279d0ba5a40fdf',
 'oracles/rgm_planning_controls_independent.py':'cd92cff9bb92eb38afd5a989109718cd600390a9609a13fc070f9cc1e1677bf3',
 'tests/oracle_rgm_planning_v2_test.py':'b21a64de06d58fba6711a704508bf31babab619f9c48821b2b0116e9f327ce85',
 'data/RGM-001/candidate_renderings/planning-gap-v2.jsonl':'57dfa90e3946fbb657d03c33388e2d67fb3fa9107e4b4de4ffddef0b16289c23',
 'data/RGM-001/manifests/planning-gap-v2.family_controls.json':'9c4497cf61e1d77cea24d808eca965c00e10e975cf235bd7d1aa2667e259da17',
 'audits/diversity/planning-gap-v2-planning-controls.json':'20212a488c949b4f5d9cbc920399eec444c976d335d5aca637d302cb4527ba26',
 'quality_reports/RGM_V2_RENDERER_CONTROL_FAILURE_DIAGNOSTIC.json':'b3578cbc2a41f8bc2a33e1a23902b16d806606fc2c1bfe7f7c2f9a720be39e74',
 'factory/planning_controls.py':'352f5fc0247b339293206958144e25f7d4437477f4a60702d926b805f25c18c3',
 'factory/pipeline.py':'d3d328e4b604f85fdd0c88d5f39aee2773689626412f72f7aeae9768429c5b62',
 **oracle.SPECIFICATIONS}

def encode(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(raw): return hashlib.sha256(raw).hexdigest()
def payload(row): return json.loads(v1._render_payload(row['rendered_input'],row['semantic_task']))

def toy():
    action=lambda block,to:{'block':block,'to':to}
    initial={'a':'b','b':'c','c':'TABLE','d':'TABLE','e':'TABLE'}
    base={'kind':'blocks','on':initial,'goal_on':{'a':'d','b':'TABLE'},
          'plan':[action('a','d'),action('b','TABLE')]}
    extended=deepcopy(base)
    # The same old goal facts appear in a different insertion order; one new
    # obligation changes semantics exactly as a public D0/D1 extension requires.
    extended['goal_on']={'b':'TABLE','a':'d','c':'e'}
    minimum=deepcopy(extended);minimum['plan'].append(action('c','e'))
    detour=deepcopy(minimum);detour['plan'].insert(0,action('a','TABLE'))
    tasks=[base,extended,minimum,detour]
    rows={};mapping={};controls=[]
    for i,task in enumerate(tasks):
        ids=[]
        for style in (0,1):
            sid='hand-D'+str(i)+'-'+str(style);ids.append(sid)
            # Two explicit deterministic renderer functions: preserve source
            # insertion order, or reverse object field orders. They do not
            # change plan sequence, support facts or any other input value.
            shown=deepcopy(task)
            if style:
                shown=dict(reversed(list(shown.items())))
                for key in ('on','goal_on'):shown[key]=dict(reversed(list(shown[key].items())))
                shown['plan']=[dict(reversed(list(x.items()))) for x in shown['plan']]
            rows[sid]={'sample_id':sid,'semantic_task':deepcopy(task),
                'rendered_input':'Task JSON:\n'+json.dumps(shown),
                'generator_version':'handmade-public-style-proof','skill_tags':[],'track_id':'RGM-001'}
        mapping['D'+str(i)]=(task,oracle.analyze(task),ids)
    for name,left,right,field in (
          ('partial_goal_extension_counterfactual','D0','D1','goal_on'),
          ('same_world_valid_goal_detour','D2','D3','plan')):
        for style in (0,1):
            controls.append({'control_id':'hand-'+name+'-'+str(style),'name':name,'scope':'corpus_variant',
                'sample_ids':[mapping[left][2][style],mapping[right][2][style]],
                'changed_task_fields':[field],'provenance':'supplied_tasks'})
    return rows,mapping,{'primary_stratum':'D','controls':controls,'source_index':216}

def projection_diagnostic(mapping,by_id,stratum):
    bases=[payload(by_id[sid]) for sid in mapping[stratum+'0'][2]]
    details=[];diagnostic_styles={}
    for context,(task,_,ids) in sorted(mapping.items()):
        common_edges={tuple(e) for e in bases[0].get('edges',[])} & {tuple(e) for e in task.get('edges',[])}
        common_goal=set(bases[0].get('goal_on',{})) & set(task.get('goal_on',{}))
        references=[oracle._presentation(b,common_edges,common_goal) for b in bases]
        for sid in ids:
            measured=oracle._presentation(payload(by_id[sid]),common_edges,common_goal)
            exact=[i for i,x in enumerate(references) if encode(x)==encode(measured)]
            # Fault isolation only; NEVER used as a substitute acceptance gate.
            small=lambda x:{k:v for k,v in x.items() if k!='goal_key_order'}
            less=[i for i,x in enumerate(references) if encode(small(x))==encode(small(measured))]
            if len(less)==1:diagnostic_styles[sid]=less[0]
            details.append({'sample_id':sid,'context':context,'exact_matches':exact,
                'diagnostic_matches_without_shared_goal_order':less,
                'actual_shared_goal_order':measured.get('goal_key_order'),
                'base_shared_goal_orders':[x.get('goal_key_order') for x in references]})
    return details,diagnostic_styles

def execute():
    report={'actual_execution_id':'/root/formal_oracle','role':'Guided formal contract assessment; no data approval',
        'at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':{},'errors':[],
        'read_boundary':'Public seven contracts, own frozen V1/V2 source, explicitly authorized candidate/public manifest/root fault localization and root generic integration only. Stored verified_target/reviewer predictions/verdicts/producer source/private expectations are not accessed.',
        'child_delegation':'Separate fresh public-only interpretation spawn attempted; tool rejected actual agent thread limit. No second-role conclusion claimed.'}
    report_path=BASE/'reviews/oracle/RGM-v2-renderer-public-assessment-execution-20261010T062719Z.json'
    try:
        for p,expected in AUTHORIZED.items():
            actual=digest((BASE/p).read_bytes());assert actual==expected,(p,actual,expected)
            report['source_sha256'][p]=actual
        own_relative=str(Path(__file__).resolve().relative_to(BASE))
        report['source_sha256'][own_relative]=digest(Path(__file__).read_bytes())
        blob=BASE/'reproducibility/source_blobs'/(report['source_sha256'][own_relative]+'.source')
        assert blob.exists() and blob.read_bytes()==Path(__file__).read_bytes(), 'diagnostic source must be frozen before execution'
        manifest=json.loads((BASE/'data/RGM-001/manifests/planning-gap-v2.family_controls.json').read_bytes())
        # Explicit public input projection. No stored target/reference comparison.
        whitelist=('sample_id','semantic_task','rendered_input','skill_tags','generator_version','track_id','split',
            'source_provenance','semantic_family_id','semantic_parent_id','controlled_context','primary_stratum')
        rows=[]
        for line in (BASE/'data/RGM-001/candidate_renderings/planning-gap-v2.jsonl').read_bytes().splitlines():
            decoded=json.loads(line);rows.append({key:decoded[key] for key in whitelist if key in decoded})
        by_id={x['sample_id']:x for x in rows};assert len(rows)==len(by_id)==408
        row_failures={r['sample_id']:oracle.check(r) for r in rows if oracle.check(r)}
        assert not row_failures,row_failures
        report['actual_input_row_checks']={'rows':len(rows),'errors':row_failures,'gold_used':False}
        failures=[];accepted_controls=[];details=[];declared_controls_without_goal_projection=[];seen=set()
        for family in manifest['families']:
            mapping={}
            for context in family['contexts']:
                ids=context['sample_ids'];task=by_id[ids[0]]['semantic_task']
                mapping[context['context_id']]=(task,oracle.analyze(task),ids)
            try:
                declarations,_,_,_=oracle._supplied_controls(family,mapping,by_id,seen)
                accepted_controls.extend(declarations)
            except oracle.OracleError as exc:
                failures.append({'source_family_key':family['source_family_key'],'error':str(exc)})
            if family['primary_stratum']=='D':
                measurements,diagnostic_styles=projection_diagnostic(mapping,by_id,'D')
                for item in measurements:item['source_family_key']=family['source_family_key']
                details.extend(measurements)
                # Independently verify pair facts and diagnostic style matches;
                # this does not waive the unchanged frozen style gate.
                for control in family['controls']:
                    left,right=control['sample_ids'];assert diagnostic_styles[left]==diagnostic_styles[right]
                    fields=v1._pair(control['name'],by_id[left]['semantic_task'],by_id[right]['semantic_task'])
                    assert fields==sorted(control['changed_task_fields'])
                    declared_controls_without_goal_projection.append({'control_id':control['control_id'],
                        'input_derived_fields':fields,'diagnostic_style':diagnostic_styles[left]})
        root_audit=json.loads((BASE/'audits/diversity/planning-gap-v2-planning-controls.json').read_bytes())['report']
        assert [x['source_family_key']+': '+x['error'] for x in failures]==root_audit['errors']
        root_localization=json.loads((BASE/'quality_reports/RGM_V2_RENDERER_CONTROL_FAILURE_DIAGNOSTIC.json').read_bytes())
        exact_failures=sum(len(x['exact_matches'])!=1 for x in details)
        less_matches=sum(len(x['diagnostic_matches_without_shared_goal_order'])==1 for x in details)
        assert exact_failures==root_localization['rows_not_matching_full_projection']
        assert less_matches==root_localization['rows_matching_after_removing_only_shared_goal_key_order']
        report['actual_frozen_style_checks']={'families_attempted':len(manifest['families']),
            'failures':failures,'renderer_control_declarations_checked':len(accepted_controls),
            'D_rows':len(details),'D_exact_projection_failures':exact_failures,
            'D_unique_diagnostic_matches_without_goal_order':less_matches,
            'D_pair_fields_and_diagnostic_same_style_declarations':len(declared_controls_without_goal_projection),
            'diagnostic_pair_checks':declared_controls_without_goal_projection,'D_presentation_details':details,
            'current_gate_result':'FAIL remains unchanged; diagnostic relaxed projection is not an approval criterion.'}
        hand_rows,mapping,family=toy();snap=encode([hand_rows,mapping,family])
        assert all(not oracle.check(x) for x in hand_rows.values())
        assert mapping['D0'][1]['target']['valid'] and mapping['D0'][1]['target']['reaches_goal']
        assert mapping['D0'][1]['search']['minimum_moves']==2
        assert mapping['D1'][1]['target']['valid'] and not mapping['D1'][1]['target']['reaches_goal']
        assert mapping['D1'][1]['search']['minimum_moves']==3
        for context in ('D2','D3'):assert mapping[context][1]['target']['valid'] and mapping[context][1]['target']['reaches_goal']
        assert v1._pair('partial_goal_extension_counterfactual',mapping['D0'][0],mapping['D1'][0])==['goal_on']
        assert v1._pair('same_world_valid_goal_detour',mapping['D2'][0],mapping['D3'][0])==['plan']
        expected_failures=[]
        for name,call in (('renderer_styles',lambda:oracle._renderer_styles(mapping,hand_rows,'D')),
                          ('supplied_controls',lambda:oracle._supplied_controls(family,mapping,hand_rows,set()))):
            try:call()
            except oracle.OracleError as exc:expected_failures.append({'probe':name,'error':str(exc)})
        assert len(expected_failures)==2,expected_failures
        hand_details,styles=projection_diagnostic(mapping,hand_rows,'D');assert len(styles)==8
        # Control experiment changes only synthetic rendering order, not tasks
        # or any actual candidate file. The exact frozen gate then accepts.
        canonical=deepcopy(hand_rows)
        for sid,row in canonical.items():
            parsed=payload(row);order=sorted(parsed['goal_on'])
            if sid.endswith('-1'):order.reverse()
            parsed['goal_on']={b:parsed['goal_on'][b] for b in order}
            row['rendered_input']='Task JSON:\n'+json.dumps(parsed)
            assert encode(oracle.solve(row['semantic_task']))==encode(oracle.solve(hand_rows[sid]['semantic_task']))
        checked,_,_,_=oracle._supplied_controls(family,mapping,canonical,set());assert len(checked)==4
        assert snap==encode([hand_rows,mapping,family])
        report['minimal_synthetic_reproduction']={'tasks':{k:v[0] for k,v in mapping.items()},
            'rendered_inputs':{sid:r['rendered_input'] for sid,r in hand_rows.items()},
            'explicit_renderers':['preserve source object insertion order','reverse source object field orders; preserve action sequence'],
            'expected_unchanged_checker_failures':expected_failures,'diagnostic_details':hand_details,
            'semantics_and_D_pair_obligations':{'D0_minimum':2,'D1_fresh_minimum':3,'D2_minimum':3,'D3_length':4,
                'typed_parsed_payload_equality':True,'input_derived_targets_reorder_invariant':True},
            'synthetic_shared_goal_sorting_control_declarations_checked':len(checked)}
        selected=integration.planning_control_rows(rows,'planning-gap-v2')
        assert len(selected)==408
        assert integration.planning_checker(selected,'planning-gap-v2',manifest=manifest) is oracle
        missing_c4=[r for r in rows if r.get('controlled_context')!='C4' and oracle.NEW_PROPERTY not in r.get('skill_tags',[])]
        assert integration.planning_checker(missing_c4,'planning-gap-v2',manifest=manifest) is oracle
        alt=[dict(r,generator_version='alternative-public-fresh-version') for r in missing_c4]
        assert integration.planning_checker(alt,'planning-gap-v2',manifest=manifest) is oracle
        assert integration.planning_checker([],None,manifest={'version':oracle.VERSION}) is oracle
        assert integration.planning_checker([],None,manifest={'version':'rgm-planning-controls-1.0.0'}) is v1
        assert integration.V2_CHECKER_SHA256==AUTHORIZED['oracles/rgm_planning_controls_v2_independent.py']
        assert integration.CHECKER_SHA256==AUTHORIZED['oracles/rgm_planning_controls_independent.py']
        assert len(oracle.SPECIFICATIONS)==7
        report['bounded_integration_checks']={'actual_selected_rows':408,'actual_bound_manifest_routes_frozen_V2':True,
            'missing_C4_still_routes_V2_by_batch_manifest':True,'alternative_generator_spelling_still_routes_V2':True,
            'explicit_V1_manifest_routes_frozen_V1':True,'source_constants_match_exact_frozen_hashes':True,
            'code_inspection':'planning_controls.py lines84-123 binds V1/V2 source, seven public docs, exact prior/parent bytes and fails before family execution on mismatches. Lines124-129 recheck evidence. pipeline.py lines241-252 propagates family errors into all_machine_quality_gates_pass; lines355-356 and448-449 require complete family controls after approval filtering and release verification.',
            'not_executed':'No pipeline/publication/release or planning_control_report invocation; no parent/prior inventory or review/private producer reads.'}
        assert all(digest((BASE/p).read_bytes())==h for p,h in report['source_sha256'].items())
        report['frozen_source_candidate_manifest_unchanged_after_execution']=True
    except Exception as exc:
        report['errors'].append({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
    assert not report_path.exists()
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(encode({'report_path':str(report_path.relative_to(BASE)),'sha256':digest(report_path.read_bytes()),
        'bytes':report_path.stat().st_size,'execution_errors':len(report['errors']),
        'style_failure_families':len(report.get('actual_frozen_style_checks',{}).get('failures',[])),
        'synthetic_expected_gate_failures':len(report.get('minimal_synthetic_reproduction',{}).get('expected_unchanged_checker_failures',[]))}))
    if report['errors']:raise SystemExit(1)

if __name__=='__main__':execute()
