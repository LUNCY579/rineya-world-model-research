"""Join completed actual reviewer evidence and our own immutable REM audit.

No sample, producer, contract, release or status writes are performed.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reviews/adjudication'
ACTOR='/root/research_architect'
BATCH='memory-gap-v2'
EXPECTED='ca17f8a460f9a8909522901d2ef39b5f096b27a2cc56e0e3fb49ca465c8e96b1'
SCORE_KEYS=['semantic_correctness','target_correctness','instruction_clarity','reasoning_validity',
            'generalization_value','shortcut_resistance','diversity','difficulty_quality']

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def write_new(p,data):
    with p.open('x',encoding='utf-8') as f:f.write(data);f.flush();os.fsync(f.fileno())
def load(p):return json.loads(p.read_text())

def main():
    final=OUT/f'{BATCH}.jsonl'
    if final.exists():print('PRESERVED_EXISTING_FINAL');return
    cp=ROOT/f'data/REM-001/candidate_renderings/{BATCH}.jsonl'
    rp={k:ROOT/f'reviews/{k}/{BATCH}.jsonl' for k in ['expert','adversarial','oracle']}
    expertp=ROOT/'reviews/expert/memory-gap-v2.completed_execution.json'
    redp=ROOT/'audits/execution/memory-gap-v2-adversarial.json'
    integrityp=ROOT/'audits/execution/memory-gap-v2-integrity-adversarial.json'
    codep=ROOT/'audits/execution/memory-gap-v2-export-completeness-adversarial.json'
    ownp=OUT/'memory-gap-v2.independent_adjudicator_audit.json'
    formalp=ROOT/'reviews/oracle/memory-gap-v2.controls-interface-aligned.json'
    formalexecp=ROOT/'reviews/oracle/memory-gap-v2.controls-interface-aligned.execution.json'
    leakp=ROOT/'audits/leakage/memory-gap-v2.json'
    required=[cp,*rp.values(),expertp,redp,integrityp,codep,ownp,formalp,formalexecp,leakp,
              ROOT/'reviews/expert/memory-gap-v2.first_judgments.jsonl',
              ROOT/'reviews/expert/semantic_inspection_memory_gap_v2.md',
              ROOT/'reviews/expert/memory-gap-v2.post_first_metadata_interface.json',
              ROOT/'reviews/expert/semantic_inspection_memory_gap_v1.md',
              ROOT/'reviews/expert/memory-gap-v1.jsonl',
              ROOT/'reviews/expert/memory-gap-v1.first_judgments.jsonl',
              ROOT/'reviews/adversarial/memory-gap-v2.first_judgments.jsonl',
              ROOT/'schemas/EXTENDED_SEMANTICS.md',ROOT/'schemas/MEMORY_CONTROL_PROPERTIES.md',
              ROOT/'schemas/MEMORY_CONTROL_METADATA.md',ROOT/'schemas/canonical/v1.schema.json',
              OUT/'audit_memory_gap_v2.py',Path(__file__).resolve(),
              ROOT/'reviews/oracle/memory-gap-v2.controls.json',
              ROOT/'data/REM-001/candidate_renderings/memory-gap-v1.jsonl',
              ROOT/'data/REM-001/quarantine/memory-gap-v1.jsonl',
              ROOT/'audits/leakage/memory-gap-v2_cross_release.json']
    assert all(p.exists() for p in required)
    snapshots={str(p.relative_to(ROOT)):sha(p) for p in required}
    assert sha(cp)==EXPECTED
    expert,red,integrity,code,own,formal,formalexec,leak=map(load,[expertp,redp,integrityp,codep,ownp,formalp,formalexecp,leakp])
    assert expert['reviewer_execution_id']=='/root/expert_reviewer' and expert['completed_receipts']==768
    assert expert['source_sha256']==EXPECTED and expert['receipt_sha256']==sha(rp['expert'])
    assert expert['first_final_identical'] and expert['exact_candidate_hash_bound'] and expert['candidate_bytes_unchanged']
    assert sha(ROOT/'reviews/expert/memory-gap-v2.first_judgments.jsonl')==sha(rp['expert'])
    assert expert['inspection_sha256']==sha(ROOT/'reviews/expert/semantic_inspection_memory_gap_v2.md')
    assert expert['post_first_interface_report_sha256']==sha(ROOT/'reviews/expert/memory-gap-v2.post_first_metadata_interface.json')
    assert red['reviewer_execution_id']=='/root/adversarial_auditor' and red['status']=='completed'
    assert red['candidate_file_sha256']==EXPECTED and red['records']==768
    assert red['final_receipts_sha256']==sha(rp['adversarial'])
    assert red['first_judgments_sha256']==sha(ROOT/'reviews/adversarial/memory-gap-v2.first_judgments.jsonl')
    assert red['first_predictions_and_judgments_preserved'] and red['target_mismatches']==red['unsupported_property_rows']==0
    assert integrity['status']=='completed' and integrity['reviewer_execution_id']=='/root/adversarial_auditor'
    for rel,h in integrity['frozen_candidate_and_receipt_hashes'].items():assert sha(ROOT/rel)==h
    assert not integrity['metadata_interface_errors'] and integrity['declared_roles_retain_stable_decoded_signatures']
    assert integrity['order_invariant_typed_relation_motifs']==9
    assert integrity['order_invariant_typed_relation_motifs_cross_split']==6
    assert code['status']=='completed' and code['decision']=='REVISE_CODE'
    assert code['candidate_semantic_receipts_unchanged_sha256']==sha(rp['adversarial'])
    assert integrity['separate_completed_export_coverage_review']['sha256']==sha(codep)
    assert own['execution_id']==ACTOR and own['completed_independent_check'] and own['candidate_sha256']==EXPECTED
    assert own['counts']['records']==768 and own['counts']['source_families']==12
    for rel,h in own['source_hashes'].items():assert sha(ROOT/rel)==h
    for rel,h in own['cross_release_lineage']['historical_release_sha256'].items():assert sha(ROOT/rel)==h
    assert formal['execution_id']=='/root/formal_oracle' and formal['candidate_sha256']==EXPECTED and not formal['errors']
    assert formalexec['execution_id']=='/root/formal_oracle'
    assert formalexec['actual_candidate_execution']['report_sha256']==sha(formalp)
    assert formalexec['actual_candidate_execution']['candidate_sha256']==EXPECTED
    assert formalexec['actual_candidate_execution']['errors']==formalexec['actual_candidate_execution']['unsatisfied_claim_checks']==0
    assert leak['records']==768 and leak['oracle_failures']==leak['metadata_failures']==leak['validation_failures']==leak['memory_control_error_count']==0
    assert not leak['split_leakage'] and not leak['cross_parent_semantic_duplicates']
    oldfirst=ROOT/'reviews/expert/memory-gap-v1.first_judgments.jsonl'
    assert sha(oldfirst)==sha(ROOT/'reviews/expert/memory-gap-v1.jsonl')
    rows=read(cp);ids={r['sample_id'] for r in rows}
    reviews={k:{r['sample_id']:r for r in read(p)} for k,p in rp.items()}
    facts={r['sample_id']:r for r in own['rows']}
    assert len(ids)==768 and all(len(v)==768 and set(v)==ids for v in [*reviews.values(),facts])
    own_scores=dict(semantic_correctness=5,target_correctness=5,instruction_clarity=5,reasoning_validity=5,
                    generalization_value=3,shortcut_resistance=3,diversity=3,difficulty_quality=3)
    receipt_rows=[];agreements=Counter();score_disagreements=Counter()
    semantic_baseline=integrity['semantic_partial_label_baselines']['visible_exact_key_event_count']
    assert semantic_baseline['status']['accuracy']==0.84375
    assert semantic_baseline['full_query_target']['accuracy']==0.125
    added_issues=[
      dict(category='shared_relation_motifs',severity='low',evidence='Order-invariant typed relation normalization finds9 motifs,6 shared across train/dev. Exact source/task/history lineage remains disjoint; structural OOD is not established.',fix='Preserve motif disclosure and develop genuinely new mechanism families before structural-generalization claims.'),
      dict(category='semantic_partial_label_shortcut',severity='low',evidence='Exposed two-family dev lookup using visible same-key event count gets84.375% status accuracy against65.625% majority, but12.5% exact core answer accuracy. It uses legitimate semantic content and does not recover the full tuple-resolved target.',fix='Vary key-history multiplicity within status/type strata and report full-target plus property-subset checks.'),
      dict(category='future_only_absence_missing',severity='low',evidence='All768 absent corpus answers use missing keys; zero future-only absent queries are supplied. Future-filter witnesses on other queries do not fill that subtype gap.',fix='Add independently reviewed same-key future-only absent before/at controls in new source families.'),
      dict(category='joint_nuisance_associations',severity='low',evidence='Own complete rotation-set audit reproduces joint empirical status majority fractions0.625/0.75/0.875, despite identical marginal full typed/provenance histograms by ID, position and key.',fix='Cross query-ID, position and key schedules independently when joint nuisance independence is required.'),
      dict(category='bounded_proxy_scope',severity='low',evidence='768 correlated records represent12 families/48 base histories of12-14 events and eight independent queries. No persistent learned memory, associative relevance, unresolved conflict or efficacy is demonstrated.',fix='Keep current finite-retrieval claims and plan new public semantics before multi-record or persistent-memory tasks.')]
    for row in rows:
        sid,h=row['sample_id'],row['record_hash'];e,a,o=reviews['expert'][sid],reviews['adversarial'][sid],reviews['oracle'][sid];f=facts[sid]
        assert all(r['record_hash']==h for r in [e,a,o,f])
        assert e['reviewer_execution_id']=='/root/expert_reviewer' and a['reviewer_execution_id']=='/root/adversarial_auditor'
        assert e['decision']==a['decision']=='ACCEPT' and not e['critical_defects'] and not a['critical_defects']
        assert o['pass'] and not o['errors'] and not o['metadata_errors']
        assert f['all_hash_schema_prompt_target_typed_checks'] and f['supplied_witnesses_verified'] and f['family_claims_scope_valid']
        assert set(f['status_counts'])=={'present','deleted','absent'}
        assert set(e['scores'])==set(a['scores'])==set(SCORE_KEYS)
        scores={k:min(e['scores'][k],a['scores'][k],own_scores[k]) for k in SCORE_KEYS}
        for k in SCORE_KEYS:
            assert type(scores[k]) is int and 1<=scores[k]<=5
            if e['scores'][k]!=a['scores'][k]:score_disagreements[k]+=1
        issues=[dict(x,origin_reviewer_execution_id=e['reviewer_execution_id']) for x in e['issues']]
        issues += [dict(x,origin_reviewer_execution_id=a['reviewer_execution_id']) for x in a['issues']]
        issues += [dict(x,origin_reviewer_execution_id=ACTOR) for x in added_issues]
        receipt_rows.append(dict(
            receipt_id='adj-'+BATCH+'-'+digest([ACTOR,sid,h])[:20],sample_id=sid,record_hash=h,
            reviewer_role='independent_adjudicator',reviewer_execution_id=ACTOR,
            scores=scores,critical_defects=[],issues=issues,decision='ACCEPT',confidence=0.97,
            reason='Both completed actual-execution reviewers ACCEPT this exact record. Own independent prompt-derived typed retrieval, schema/hash/full-target/provenance, all nine in-row property and supplied witness checks, complete family crossing/field pairs, supplied counterfactuals and scoped audit probes pass. Historical source-family/exact task/history lineage is disjoint. Low conditional correlations, familiar cross-split motifs and finite proxy/absence-subtype limits remain disclosed.',
            evidence=dict(expert_receipt_id=e['receipt_id'],adversarial_receipt_id=a['receipt_id'],
                expert_decision=e['decision'],adversarial_decision=a['decision'],disagreement_preserved=False,
                expert_scores=e['scores'],adversarial_scores=a['scores'],adjudicator_scores=own_scores,
                expert_completed_execution=str(expertp.relative_to(ROOT)),adversarial_completed_execution=str(redp.relative_to(ROOT)),
                oracle_receipt_key={'sample_id':sid,'record_hash':h},oracle_author_execution=o['oracle_author_execution'],
                oracle_runner_execution=o['check_runner_execution'],
                formal_control_execution=str(formalexecp.relative_to(ROOT)),
                own_independent_audit=str(ownp.relative_to(ROOT)),own_independent_audit_sha256=sha(ownp),
                own_derived_target_sha256=f['derived_target_sha256'],own_property_witnesses=f['properties'],
                supplied_witnesses_verified=True,family_claim_scopes_verified=True,all_three_statuses=True,
                source_family_key=row['source_family_key'],all_family_descendants_same_split=True,
                cross_release_source_task_history_overlap=0,property_tags_in_row_only=True,
                corpus_controls_count_as_family_variants=True,audit_probes_add_records_and_families=0,
                broader_order_invariant_relation_motifs=9,broader_relation_motifs_shared_across_split=6,
                semantic_partial_label_baseline=semantic_baseline,
                separate_original_generic_export_code_finding=str(codep.relative_to(ROOT)),
                generic_release_gate_clearance_issued_by_this_receipt=False),
            scope_limitations=['Adjudicator previously authored the public REM-GAP-003 design but generated no samples and authored no initial expert/adversarial verdicts.',
                              'Expert review is corrective, informed by its own v1 findings; public metadata interface checked after frozen first judgments changed neither candidate nor verdict.',
                              'Adversarial public metadata/generic code followup followed frozen initial predictions/judgments; final data receipts remain exact-bound.',
                              'All separate actual roles share an underlying model and can have correlated errors; two exposed dev families are not sealed evaluation.',
                              'Original separate REVISE_CODE export coverage finding remains preserved; this approves data only and does not adjudicate a subsequent generic code correction.']))
        agreements[(e['decision'],a['decision'])]+=1
    assert len(receipt_rows)==len({r['receipt_id'] for r in receipt_rows})==768
    assert sha(cp)==EXPECTED
    assert all(sha(ROOT/r)==h for r,h in snapshots.items())
    write_new(final,'\n'.join(canon(r) for r in receipt_rows)+'\n')
    execution={'execution_id':ACTOR,'reviewer_execution_id':ACTOR,'reviewer_role':'independent_adjudicator',
               'status':'completed','completed_execution':True,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
               'batch':BATCH,'candidate_sha256':EXPECTED,'records':768,'decision_counts':dict(Counter(r['decision'] for r in receipt_rows)),
               'final_receipts_sha256':sha(final),'independent_audit_sha256':sha(ownp),'input_sha256':snapshots,
               'all_input_sha256_unchanged_during_execution':True,
               'reviewer_decision_agreements':{str(k):v for k,v in agreements.items()},
               'reviewer_score_disagreement_counts':dict(score_disagreements),
               'reviewer_decisions_and_original_issues_preserved':True,
               'counts':own['counts'],'data_quality_release_eligible':True,
               'generic_release_code_fix_clearance_issued':False,
               'original_export_code_finding_preserved':{'path':str(codep.relative_to(ROOT)),'sha256':sha(codep),'decision':'REVISE_CODE','severity':'high'},
               'role_custody':{'actual_identity':ACTOR,'prior_public_design_author':True,'sample_generation':False,
                               'initial_expert_or_adversarial_review_author':False,'producer_or_release_writes':False,
                               'same_underlying_model_correlated_error_risk':True},
               'scope':'Exact data quality approval only. Release assembly and backup require separately passing corrected generic export gates. No candidate/generator/contract/release/status edits.',
               'remaining_priorities':['Future-only absence cutoff controls.','Independently crossed ID/position/key schedules and semantic event-count/status/type strata.',
                                       'New order-invariant retrieval histories/mechanisms beyond9 recurrent relation motifs.',
                                       'Separately specified multi-record relevance/dependent queries, unresolved conflicts and procedure read/write protection.',
                                       'Separate authorized model/effectiveness evaluation; current data establishes no memory efficacy.']}
    ep=OUT/'execution-memory-gap-v2.json';write_new(ep,json.dumps(execution,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'decision_counts':execution['decision_counts'],'records':768,'receipt_sha256':sha(final),'execution_sha256':sha(ep),'data_quality_release_eligible':True,'generic_code_fix_clearance':False},indent=2))

if __name__=='__main__':main()
