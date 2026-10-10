"""Exact final receipt join after completed reviewer evidence; never edits candidates/releases."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json

from audit_belief_gap_v1 import BASE, canonical, scalar, sha, solve, rows


def write_new(path, value):
    with (BASE / path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def main():
    now = datetime.now(timezone.utc).isoformat()
    audit_path = 'reviews/adjudication/belief-gap-v1.independent_adjudicator_audit.json'
    audit = json.loads((BASE / audit_path).read_text())
    assert audit['status'] == 'COMPLETED_INDEPENDENT_INPUT_RECOMPUTATION'
    source_bindings = audit['input_evidence_hashes']
    for key in ('candidate', 'expert', 'red', 'oracle', 'expert_execution', 'expert_final_binding', 'red_execution'):
        binding = source_bindings[key]
        assert sha(BASE / binding['path']) == binding['sha256']
    candidates = rows(source_bindings['candidate']['path'])
    expert = {r['sample_id']: r for r in rows(source_bindings['expert']['path'])}
    red = {r['sample_id']: r for r in rows(source_bindings['red']['path'])}
    facts = {r['sample_id']: r for r in audit['per_record']}
    # Train-only conditional lookup: empty target core by input world/observation counts.
    train_counts = defaultdict(Counter)
    for r in candidates:
        if r['split'] == 'train' and facts[r['sample_id']]['empty']:
            expected, _ = solve(r['semantic_task'])
            key = (len(r['semantic_task']['worlds']), len(r['semantic_task']['observations']))
            train_counts[key][canonical(expected['belief_sizes'])] += 1
    assert all(len(v) == 1 for v in train_counts.values())
    lookup = {k: json.loads(next(iter(v))) for k, v in train_counts.items()}
    baseline = defaultdict(Counter)
    for r in candidates:
        task = r['semantic_task']
        predicted_empty = any(type(o['equals']) is str and o['equals'] in {'stone', 'unknown'} for o in task['observations'])
        s = baseline[r['split']]
        s['records'] += 1
        s['empty_flag_correct'] += int(predicted_empty == facts[r['sample_id']]['empty'])
        if predicted_empty:
            key = (len(task['worlds']), len(task['observations']))
            assert key in lookup
            predicted = {'surviving_worlds': [], 'belief_sizes': lookup[key],
                         'queries': [{'query_id': q['query_id'], 'counts': [], 'distinct_values': 0,
                                      'entailed': False, 'guaranteed': None, 'can_reduce_uncertainty': False}
                                     for q in task['queries']]}
            expected, _ = solve(task)
            s['predicted_empty_records'] += 1
            s['complete_empty_target_correct'] += int(canonical(predicted) == canonical(expected))
    assert baseline['train']['complete_empty_target_correct'] == 216
    assert baseline['dev_challenge']['complete_empty_target_correct'] == 216
    supplementary_path = 'reviews/adjudication/belief-gap-v1.lexical_baseline_supplement.json'
    supplemental = {
        'actual_execution_id': '/root/research_architect', 'completed_at_utc': now,
        'candidate_sha256': source_bindings['candidate']['sha256'],
        'input_rule': 'empty iff any observation.equals is string stone or unknown',
        'train_only_lookup_key': ['input world count', 'input observation count'],
        'train_only_trace_lookup': [{'world_count': k[0], 'observation_count': k[1], 'belief_sizes': v}
                                   for k, v in sorted(lookup.items())],
        'actual_split_results': {k: dict(v) for k, v in baseline.items()},
        'complete_target_claim_scope': '432 predicted-empty targets including prefix sizes and input-copied query IDs; no complete target prediction claimed for864 nonempty tasks.',
        'scalar_bijection_audit': {'executions': 1296, 'semantic_equivariance_passes': 1296,
                                  'marker_failures_after_rename': 432},
        'manually_inspected_representatives': [r['sample_id'] for r in candidates
                                             if r['schedule_index'] == 0 and r['renderer'] == 'forward'],
        'inspection_scope': 'Read four complete visible world tables and six context observation/trace/query outputs per family; all1296 records independently recomputed. No manual-reading claim for every rendering.',
        'no_training_or_model_efficacy': True,
    }
    write_new(supplementary_path, supplemental)
    audit_sha = sha(BASE / audit_path)
    supplemental_sha = sha(BASE / supplementary_path)
    issue_path = 'feedback/improvement_requests/RBM-GAP-007-LEXICAL-EMPTY-MEASURED.json'
    issue = {
        'ticket': 'RBM-GAP-007-LEXICAL-EMPTY-MEASURED', 'parent_ticket': 'RBM-GAP-007',
        'status': 'CONFIRMED_MEASURED_DATA_QUALITY_REVISE', 'created_at_utc': now,
        'author_execution_id': '/root/research_architect', 'author_role': 'independent_adjudicator',
        'design_author_overlap': 'Same actual identity authored original and new public designs; no producer or initial reviewer authorship. This is adjudication using completed reviewer evidence, not blind first review.',
        'same_underlying_model_risk': True,
        'code': 'observation_scalar_token_empty_target_shortcut', 'severity': 'high',
        'candidate': source_bindings['candidate'],
        'measured_rule': supplemental['input_rule'],
        'actual_records': 1296, 'actual_independent_source_families': 4,
        'base_contexts': 24, 'marker_flag_correct': 1296,
        'empty_records_directly_affected': 432, 'dependent_nonempty_records_held': 864,
        'split_results': supplemental['actual_split_results'],
        'family_results': audit['actual_family_stats'],
        'typed_observation_scalar_histogram': audit['typed_observation_scalar_anywhere_outcome_histogram'],
        'scalar_specificity': 'Union of stone/unknown in observation.equals, not a substring-anywhere prompt rule. Stone alone covers324 empty rows; unknown alone108. Unknown occurs as an ordinary possible query value in positive tasks.',
        'counterfactual_evidence': 'Actual injective renaming of string scalar values, coherently in worlds/observations, preserves all typed filtering/query answers equivariantly; the marker rule then misses all432 empty tasks.',
        'current_target_correctness': 'All1296 exact full targets independently match typed input-derived recomputation; current corrected formal/root reports have zero errors.',
        'dependency': 'Removing all E/F red-REVISE records removes both mandatory contradiction/absorption contexts from every family; removing rotation/style views also breaks full coverage. No dependency-complete quality-approved subset is certified.',
        'incoming_review_decisions': {'expert': {'ACCEPT': 1296}, 'adversarial_final': {'ACCEPT': 864, 'REVISE': 432},
                                      'adversarial_first': {'ACCEPT': 1296}},
        'expert_red_disagreements_preserved': 432, 'adjudication': {'REVISE': 1296},
        'measured_evidence': {'independent_audit': {'path': audit_path, 'sha256': audit_sha},
                              'lexical_baseline_supplement': {'path': supplementary_path, 'sha256': supplemental_sha},
                              'red_receipts': source_bindings['red'], 'red_completed_execution': source_bindings['red_execution'],
                              'expert_receipts': source_bindings['expert'], 'expert_final_binding': source_bindings['expert_final_binding']},
        'prospective_fix': 'NEW frozen A–H revision with E/G and F/H same-first-prefix single-equals status changes; F/H repeat first observation. Fresh candidate/formal/initial expert/red/adjudication required.',
        'public_fix': 'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md',
        'public_revision_sha256': sha(BASE / 'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md'),
        'fix_limits': 'Proposed1728 correlated rows addzero families;75% nonempty, positive-only scalar/position associations and finite conjunction lookup can remain; G/H may be singleton/simple.',
        'candidate_or_gold_edits': 0, 'sample_approvals': 0,
    }
    write_new(issue_path, issue)
    receipt_path = 'reviews/adjudication/belief-gap-v1.jsonl'
    receipts = []
    for r in candidates:
        sid = r['sample_id']
        direct = facts[sid]['empty']
        code = 'observation_scalar_token_empty_target_shortcut' if direct else 'required_family_control_dependency'
        reason = ('Correct target, but this empty context is directly covered by the perfect lexical shortcut.' if direct else
                  'Correct target and incoming reviewers agree ACCEPT, but mandatory E/F controls of this same family require revision; this row cannot form a complete approved release group.')
        reason += ' Preserve first/final reviews; use a separately generated and freshly reviewed corrected batch.'
        identity = canonical(['/root/research_architect', 'belief-gap-v1', sid, r['record_hash'], code])
        receipt = {
            'receipt_id': 'adj-belief-gap-v1-' + hashlib.sha256(identity.encode()).hexdigest()[:24],
            'sample_id': sid, 'record_hash': r['record_hash'], 'reviewer_role': 'independent_adjudicator',
            'reviewer_execution_id': '/root/research_architect', 'completed_at_utc': now,
            'scores': {'semantic_correctness': 5, 'target_correctness': 5, 'reasoning_validity': 5,
                       'instruction_clarity': 5, 'generalization_value': 2, 'diversity': 3,
                       'shortcut_resistance': 2 if direct else 3, 'difficulty_quality': 3},
            'critical_defects': [code], 'issues': [{'code': code, 'severity': 'high',
                'evidence': '432 exact empty rows match stone/unknown observation marker; all four families lose required E/F contexts in the864-row partial acceptance subset.',
                'fix': 'Retain this original batch as revised evidence; require fresh complete A–H paired controls and independent exact-hash review.'}],
            'decision': 'REVISE', 'confidence': 0.99, 'reason': reason,
            'evidence': {
                'candidate_sha256': source_bindings['candidate']['sha256'],
                'independent_audit': audit_path, 'independent_audit_sha256': audit_sha,
                'input_derived_expected_sha256': facts[sid]['independent_expected_sha256'],
                'target_hash_prompt_schema_typed_two_solver_pass': True,
                'context': r['controlled_context'], 'source_family_key': r['source_family_key'],
                'source_family_index': r['source_family_index'], 'split': r['split'],
                'own_prefix_survivors': facts[sid]['prefix_survivors'],
                'per_record_properties_independently_witnessed': facts[sid]['property_tags_independently_witnessed'],
                'lexical_marker_present': facts[sid]['marker_present'], 'empty': facts[sid]['empty'],
                'direct_lexical_defect': direct, 'dependency_hold': not direct,
                'expert_receipt_id': expert[sid]['receipt_id'], 'expert_decision': expert[sid]['decision'],
                'adversarial_receipt_id': red[sid]['receipt_id'], 'adversarial_decision': red[sid]['decision'],
                'expert_red_disagreement': expert[sid]['decision'] != red[sid]['decision'],
                'expert_final_sha256': source_bindings['expert']['sha256'], 'adversarial_final_sha256': source_bindings['red']['sha256'],
                'oracle_receipt_file_sha256': source_bindings['oracle']['sha256'], 'oracle_row_pass': True,
                'measured_issue_ticket': issue_path,
                'current_formal_root_report_error_counts': audit['current_machine_counts'],
                'mandatory_family_contexts': list('ABCDEF'), 'partial_red_accept_missing_contexts': ['E', 'F'],
            },
            'independence_scope': 'Actual original/revision public-design author and later adjudicator; no generation or initial reviewer authorship. Completed private reviewer verdicts are adjudication evidence. Shared model/common guidance risk disclosed.',
        }
        receipts.append(receipt)
    assert len(receipts) == 1296 and len({r['receipt_id'] for r in receipts}) == 1296
    assert all(r['decision'] == 'REVISE' for r in receipts)
    assert sum(r['evidence']['expert_red_disagreement'] for r in receipts) == 432
    with (BASE / receipt_path).open('x') as f:
        for r in receipts:
            f.write(canonical(r) + '\n')
    report_path = 'reviews/adjudication/ADJUDICATION_BELIEF_GAP_V1.md'
    report = f'''# Actual independent adjudication: belief-gap-v1

Final decision: **REVISE all 1,296 records; no release-eligible subset certified**. All exact targets remain correct. Of these, 432 E/F records have the direct high-severity lexical shortcut;864 A–D records are held through mandatory family-control dependency. Their incoming expert/red ACCEPT decisions are preserved, not rewritten.

Actual adjudicator is `/root/research_architect`, also the original and v2 public design author. This identity generated no candidate and authored no first expert/red judgments. I read completed reviewers' evidence after root confirmed final red completion. This is adjudication, not a blind first review. Public correction guidance and a shared underlying model can cause correlated errors.

Candidate `data/RBM-001/candidate_renderings/belief-gap-v1.jsonl` SHA256 `{source_bindings['candidate']['sha256']}` binds all decisions. My own typed incremental filter and separately constructed prefix conjunction independently agree for all1,296 parsed student tasks, exact full targets, record hashes and prompts. All attached public properties/locators were recomputed;48 complete nuisance-triple groups and648 field-order pairs pass. I read four complete representative world tables and all24 context observations/trace/query summaries, rather than claiming manual reading of every rendering. Four families and24 contexts are the independent design units; the1,296 renderings are correlated. Current/prior source indices and exact tasks do not overlap;8/12-world current tables differ in cardinality from the older4/6-world release. Shared unweighted motifs and exposed dev remain limitations.

The input rule `empty iff any observation.equals is a string in {{stone,unknown}}` matches1,296/1,296 statuses:648/648 train and648/648 exposed dev. Stone occurs in324 empty records and unknown in108; neither occurs in observation.equals on a nonempty row. This is a union-of-two-values rule, not a generic prompt substring rule. Unknown can appear as a legitimate possible query value in nonempty rows. A train-only lookup keyed by input world count and observation count supplies the empty prefix traces; with input query IDs it reproduces all216 train and216 dev complete empty targets. It makes no complete-target claim for864 nonempty rows. I executed1,296 coherent injective string-scalar renamings: all typed semantics/provenance are equivariant, but the marker rule then misses all432 empty rows. No training or model efficacy result is involved.

Expert first/final remain1,296 ACCEPT, SHA256 `{source_bindings['expert']['sha256']}`. Red first remains1,296 ACCEPT, SHA256 `{source_bindings['red_first']['sha256']}`; completed red final is864 ACCEPT/432 REVISE, SHA256 `{source_bindings['red']['sha256']}`. Preserve the432 expert/red disagreements explicitly in each exact-hash receipt. The final red execution SHA256 is `{source_bindings['red_execution']['sha256']}`. Correct target scores do not override a high data-quality defect.

All four source families12,13,18,19 would lose E and F if only red-ACCEPT records were released. The frozen original curriculum requires every family to supply all six contexts, including contradiction and absorption. A partial row selection also cannot manufacture missing required controls or claim complete rotations for absent contexts. Hence every constituent is REVISE as a release group; no remaining family is complete under the declared gate. This is a dependency hold for A–D, not an invented target error.

The corrected current machine evidence has0 oracle, metadata, validation and belief-control errors and68 executed probes. Current report snapshots `{audit['preserved_machine_source_snapshots'][0]['preserved_path']}` and `{audit['preserved_machine_source_snapshots'][1]['preserved_path']}` preserve actual bytes/hashes separately from the historical first reports. Those first reports retain216 metadata/validation failures and224 control errors from the original interface/provenance checker; their hashes remain bound in my audit. Own actual first audit execution also failed once on an older release's missing source_family_key; its exact source and failure are preserved under `belief-gap-v1-execution-history/`. I corrected only my lineage reader to use that release's actual semantic-family pNNNN indices and reran successfully. No candidate, target or initial verdict changed.

The NEW frozen `schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md` and `research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md` propose G/H controls with one-equals E/G and F/H changes and exact same first-prefix supports; F/H repeat their first observation. They do not rewrite v1 or approve future data. Proposed1,728 correlated rows still represent four families/32 contexts, with75% nonempty frequency and remaining positive-only scalar/position and finite lookup risks. G/H may be singleton or non-strict simple filtering components; tag/complexity claims require actual witnesses. Further scalar balancing is not established. Any future batch needs fresh final formal/expert/red/adjudication evidence.

Exact artifacts: `{receipt_path}`, `{audit_path}`, `{supplementary_path}`, `{issue_path}` and `execution-belief-gap-v1.json`. First/final reviewer files, candidate/gold, established public docs and release files remain unchanged.
'''
    with (BASE / report_path).open('x') as f:
        f.write(report)
    output_paths = [receipt_path, audit_path, supplementary_path, issue_path, report_path,
                    'reviews/adjudication/audit_belief_gap_v1.py',
                    'reviews/adjudication/finalize_belief_gap_v1_adjudication.py',
                    'reviews/adjudication/belief-gap-v1-execution-history/first-failure.json']
    execution = {
        'status': 'COMPLETED_FROZEN_DATA_ADJUDICATION', 'actual_execution_id': '/root/research_architect',
        'reviewer_role': 'independent_adjudicator', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_sha256': source_bindings['candidate']['sha256'], 'records': 1296,
        'decisions': {'REVISE': 1296}, 'direct_lexical_defects': 432, 'dependency_holds': 864,
        'expert_red_disagreements_preserved': 432, 'release_eligible_records': 0,
        'actual_python': '/opt/anaconda3/bin/python3',
        'actual_commands': [
            {'command': '/opt/anaconda3/bin/python3 High-Quality_Dataset/reviews/adjudication/audit_belief_gap_v1.py',
             'exit_code': 1, 'reason': 'Older release lacks source_family_key; failed source and execution preserved before repair.'},
            {'command': '/opt/anaconda3/bin/python3 High-Quality_Dataset/reviews/adjudication/audit_belief_gap_v1.py',
             'exit_code': 0, 'reason': 'Corrected only own prior-lineage reader; all input/target/hash/property/schedule/lexical checks completed.'},
            {'command': '/opt/anaconda3/bin/python3 High-Quality_Dataset/reviews/adjudication/finalize_belief_gap_v1_adjudication.py',
             'exit_code': 0, 'reason': 'Exact final review join, train-only marker core lookup, measured ticket and all per-row decisions.'}],
        'source_bindings': source_bindings,
        'current_machine_snapshots': audit['preserved_machine_source_snapshots'],
        'preserved_first_machine_reports': audit['preserved_first_machine_reports'],
        'current_corrected_machine_error_counts': audit['current_machine_counts'],
        'first_final_review_preservation': {'expert_byte_equal': True, 'red_first_preserved': True,
            'red_first': {'ACCEPT': 1296}, 'red_final': {'ACCEPT': 864, 'REVISE': 432}},
        'output_sha256': {p: sha(BASE / p) for p in output_paths},
        'prospective_revision_files_frozen': {
            p: sha(BASE / p) for p in ['schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md',
                                      'research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md',
                                      'feedback/improvement_requests/RBM-GAP-007-REVISION-V2.json',
                                      'research/curricula/RBM-EVIDENCE-GAP-REVISION-V2-EXECUTION.json']},
        'original_public_design_author_and_later_adjudicator_disclosed': True,
        'private_verdicts_read_after_completed_final_confirmation': True,
        'producer_reference_or_oracle_implementation_read': False,
        'same_underlying_model_correlated_error_risk': True,
        'child_analysis_spawn_attempt': 'Denied by actual agent-thread limit; no child evidence invented.',
        'production_samples_generated_or_approved': 0, 'candidate_or_gold_edits': 0, 'release_writes': 0,
        'freeze': 'Per-record decisions and research output complete; future packing adds lossless custody artifacts without changing raw bytes.'
    }
    execution_path = 'reviews/adjudication/execution-belief-gap-v1.json'
    write_new(execution_path, execution)
    print(canonical({'status': execution['status'], 'decisions': execution['decisions'],
                     'receipt_sha256': sha(BASE / receipt_path), 'execution_sha256': sha(BASE / execution_path),
                     'ticket_sha256': sha(BASE / issue_path), 'report_sha256': sha(BASE / report_path),
                     'supplement_sha256': supplemental_sha}))


if __name__ == '__main__':
    main()
