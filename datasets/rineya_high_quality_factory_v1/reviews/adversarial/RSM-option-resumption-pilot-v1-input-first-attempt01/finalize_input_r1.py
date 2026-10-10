"""Freeze final INPUT-ONLY assessment. No new corpus execution or gold access."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
ACTOR='/root/adversarial_auditor'
def bind(p):
    p=Path(p);b=p.read_bytes()
    return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(name,value,rows=False):
    p=OUT/name
    with p.open('x') as f:
        if rows:
            for r in value:f.write(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')
        else:json.dump(value,f,indent=2);f.write('\n')
    return bind(p)
def main():
    pre=json.loads((OUT/'FINALIZATION_PREEXEC.json').read_text())
    for b in pre['bindings']:
        p=ROOT/b['path'] if not Path(b['path']).is_absolute() else Path(b['path']);assert bind(p)['sha256']==b['sha256']
    audit=json.loads((OUT/'INPUT_DIVERSITY_SHORTCUT_AUDIT.json').read_text());num=json.loads((OUT/'NUMERIC_SHORTCUT_EXECUTION.json').read_text());run=json.loads((OUT/'EXECUTION.json').read_text())
    rows=[json.loads(x) for x in (OUT/'INITIAL_INPUT_JUDGMENTS.jsonl').read_text().splitlines()]
    assert len(rows)==216 and run['failed']==0 and sum(not c['passed'] for c in num['actual_transform_checks'])==0
    supplement=bind(OUT/'NUMERIC_SHORTCUT_EXECUTION.json');cohort=bind(OUT/'INPUT_DIVERSITY_SHORTCUT_AUDIT.json')
    for row in rows:
        row['receipt_id']='red-rsm-final-input-r1-'+hashlib.sha256((row['sample_id']+row['record_hash']).encode()).hexdigest()[:24]
        row['stage']='final_initial_input_only_before_gold'
        row['decision']='INPUT_STAGE_COMPLETE_PENDING_CANONICAL'
        row['scores']['shortcut_resistance']=2
        row['issues'] += [
          {'severity':'MODERATE','code':'status_only_coarse_shortcut_susceptibility',
           'evidence':{'supplement_sha256':supplement['sha256'],'train_only_dev_status_correct':68,'dev_rows':72,
                       'coarse_keys':'mode/history length/budget/root argument form/root success max literal +3 shallow checkpoint flags',
                       'exact_full_output_solved_by_baseline':False},
           'fix':'For a later prospective batch, cross success-literal magnitudes/argument forms/cut budgets with statuses, include truthful irrelevant predicate controls, and evaluate exact traces/states with status reported separately. Preserve current matched execution contrasts and disclose exposed tiny-cohort scope.'},
          {'severity':'LOW','code':'finite_exposed_curriculum_scope',
           'evidence':{'audit_sha256':cohort['sha256'],'input_family_lineages':12,'broad_strata':4,'typed_tasks':108,
                       'field_order_views':216,'all_strata_exposed_on_dev':True,'within_cohort_graphs_distinct':12,
                       'prior_active_graph_novelty':'NOT_VERIFIED_IN_THIS_READ_SCOPE'},
           'fix':'Retain family-level splits and correlated-view accounting; do not characterize these descendants as additional families, sealed OOD, broad planning/option discovery, or measured efficacy.'}]
        row['scope_limits'] += ['Canonical target/control claims/full hashes/difficulty/provenance remain unreviewed; this is not ACCEPT or publication approval.',
                               'Input-only shortcut issue is actual status predictability from legitimate public fields; no outcome annotation was found in prompt.']
        row['evidence']['actual_cohort_execution_sha256']=bind(OUT/'EXECUTION.json')['sha256']
        row['evidence']['numeric_supplement_sha256']=supplement['sha256']
    final_rows=write('FINAL_INITIAL_INPUT_JUDGMENTS.jsonl',rows,True)
    report={
      'schema':'rsm-actual-input-first-review-final-v1','actual_author_execution':ACTOR,
      'status':'INPUT_STAGE_COMPLETE_PENDING_SEPARATE_CANONICAL_AUTHORIZATION','data_approval':False,
      'gold_access':False,'other_reviewer_verdict_access':False,'private_implementation_access':False,
      'packet':bind(ROOT/'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl'),
      'counts':{'actual_rows':216,'own_frame_predictions':216,'own_recursive_predictions':216,'two_solver_disagreements':0,
                'initial_row_judgments':216,'final_initial_row_judgments':216,'typed_tasks':108,'prompt_views':216,
                'input_family_lineages':12,'within_new_cohort_full_coarse_reachable_graphs':12,
                'train_views':144,'exposed_dev_views':72,'original_hand_calibration_checks_reexecuted':11,
                'actual_main_checks':7509,'actual_main_passed':7509,'actual_main_failed':0,
                'cut_mode_budget_descendants':2192,'malformed_descendants':840,'truthful_success_tautology_descendants':12,
                'supplement_checks_passed':12,'supplement_checks_failed':0,'numeric_counterexample_records':40,
                'actual_semantic_control_groups':24,'eight_semantic_relations_each_group':8,'actual_field_order_pairs':108,
                'new_source_families_from_audit_descendants':0},
      'scope_findings':[
        {'severity':'MODERATE','finding':'Partial status predictability from legitimate student-visible fields',
         'evidence':'Train-only exact coarse lookup +3 shallow checkpoint consistency flags gets68/72 exposed-dev statuses, macro0.9. Mode/history/budget alone48/72; majority40/72. No full program continuation executed by baseline.',
         'interpretation':'A real shortcut threat to status-only competency inference in this bounded cohort, not proof of metadata leakage or exact trace/output solution. Some flags are legitimate semantic preflight. Dev is already exposed and exploratory feature selection limits statistical independence.',
         'fix':'Prospectively cross superficial numeric/argument/cut patterns with both statuses and score full typed events/state separately.'},
        {'severity':'MODERATE','finding':'Audit numeric baseline fails irrelevant true-predicate invariance',
         'minimal_actual_parent':'RSM-001-option-resumption-pilot-v1-p0012-F-forward',
         'parent_record_hash':'d990b3013f5b405bbffc7011baf84968dad215b9f9d8479088b64056efe74e33',
         'mutation':{'path':['options','start','success','all','append'],'atom':{'left':{'const':33},'cmp':'eq','right':{'const':33}}},
         'evidence':'Own exact full typed output remains completed and byte-normalized equal, while four fitted numeric/coarse baselines predictsuccess_failed.12 truthful descendants preserve outputs;40 baseline counterexamples retained.',
         'interpretation':'This is a demonstrated failure of audit shortcut baselines, not an incorrect production target or false supplied control. It establishes why numeric associations cannot certify execution competence.'},
        {'severity':'LOW','finding':'Context joins directly encode status outside student boundary',
         'evidence':'Train ID-context lookup72/72dev statuses; every cohort context has one outcome. All216 prompts have exact same frozen public-contract lead and typed TaskJSON; sample/context IDs are excluded from the student prompt.',
         'interpretation':'No actual prompt-level target label leakage observed. Conditional direct leakage would occur if consumers fed sample IDs/context metadata to students; this review does not inspect exports.'},
        {'severity':'LOW','finding':'Bounded scientific scope and family lineage',
         'evidence':'12 distinct within-cohort structural graphs across own full/coarse/reachable normalizations;108 typed tasks/216renderings; no exact prompt duplicates and no cross-split exact typed duplicates.12pairs-of-styles/family are correlated.',
         'interpretation':'No legacy target-free inputs authorized/read in this stage; actual novelty against prior-active releases remains unknown. Four strata exposed ondev, all initial registers2,2; no sealed OOD/efficacy or general skill-discovery claim.'}],
      'actual_control_evidence':bind(OUT/'ACTUAL_CONTROL_AND_ATTACK_EVIDENCE.json'),
      'first_prediction_freeze':bind(OUT/'PREDICTION_FREEZE.json'),'final_initial_judgments':final_rows,
      'all_initial_sources_and_outputs_preserved':True,
      'correlation_disclosure':'Same underlying model/common public design/root guidance; own previous handmade preparation and grammar reuse. Recursive and frame execution differ but share grammar validation. Earlier backup opaque-byte/hash/span exposure toRSM receipts only. Current full semantics first derived from stripped actualinputs, before canonicalgold; no unseen-blind or model-diversity claim.',
      'remaining_unknowns':['Stored target correctness','Full canonical record hashes','Actual metadata/difficulty/property/control manifest truth and custody','Prior-active graph novelty','Adjudication/release suitability'],
      'input_only_preexec_source_and_runtime':bind(OUT/'PREEXECUTION_FREEZE.json'),
      'additional_runtime_freeze':bind(OUT/'RUNTIME_DEPENDENCY_FREEZE.json'),
      'main_actual_execution':bind(OUT/'EXECUTION.json'),'numeric_actual_execution':supplement,
      'source_and_input_before_after_equal':run['all_before_after_equal'] and num['all_bound_bytes_after_equal']}
    final=write('FINAL_INPUT_REVIEW.json',report)
    before=[{k:v for k,v in b.items() if k!='archive'} for b in pre['bindings']]
    for b in pre['bindings']:
        p=ROOT/b['path'] if not Path(b['path']).is_absolute() else Path(b['path']);assert bind(p)['sha256']==b['sha256']
    for b in json.loads((OUT/'RUNTIME_DEPENDENCY_FREEZE.json').read_text())['bindings']:
        if 'sha256' in b:assert bind(Path(b['path']))['sha256']==b['sha256']
    owned_files=[bind(p) for p in sorted(OUT.rglob('*')) if p.is_file()]
    assert max(b['size_bytes'] for b in owned_files)<25000000
    first=write('FIRST_INPUT_PHASE_RECEIPT.json',{'actual_author_execution':ACTOR,'phase':'complete all-input first review before gold',
        'final_review':final,'all_completed_owned_artifacts':owned_files,'gold_access':False,'data_approval':False,
        'root_is_not_quality_vote':True,'stop_writes_after_final_custody':True})
    custody_files=[bind(p) for p in sorted(OUT.rglob('*')) if p.is_file()]
    custody=write('FINAL_INPUT_CUSTODY.json',{'actual_author_execution':ACTOR,'status':'FROZEN_INPUT_STAGE_ONLY',
        'first_input_phase_receipt':first,'final_review':final,'files':custody_files,
        'no_gold_access_before_freeze':True,'no_candidate_or_core_edits':True,'all_files_below25MB':True,
        'semantic_target_correctness':'UNKNOWN','approval':False,'await_separate_comparison_grant':True})
    print(json.dumps({'final_review':final,'first_receipt':first,'custody':custody,'final_initial_judgments':final_rows,'frozen_files':len(custody_files),'stop_writes':True}))
if __name__=='__main__':main()
