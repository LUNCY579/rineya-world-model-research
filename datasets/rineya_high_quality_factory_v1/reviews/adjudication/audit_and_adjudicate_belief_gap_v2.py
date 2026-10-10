"""Own v2 adjudication; public-derived own solver reuse, no producer/oracle/reviewer imports."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import copy
import hashlib
import json
import gzip

from audit_belief_gap_v1 import BASE, canonical, scalar, sha, solve, property_audit, ids, rows


def new_json(path, value):
    with (BASE / path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def marker(task):
    return any(type(o['equals']) is str and o['equals'] in {'stone', 'unknown'} for o in task['observations'])


def compound(task):
    return marker(task) and any(type(o['equals']) is str and o['equals'] == 'linen' for o in task['observations'])


def main():
    now = datetime.now(timezone.utc).isoformat()
    paths = {
        'candidate': 'data/RBM-001/candidate_renderings/belief-gap-v2.jsonl',
        'expert': 'reviews/expert/belief-gap-v2.jsonl',
        'red': 'reviews/adversarial/belief-gap-v2.jsonl',
        'expert_first': 'reviews/expert/belief-gap-v2.first_judgments.jsonl',
        'red_first': 'reviews/adversarial/belief-gap-v2.first_judgments.jsonl',
        'expert_execution': 'reviews/expert/belief-gap-v2.completed_execution.json',
        'expert_inspection': 'reviews/expert/semantic_inspection_belief_gap_v2.md',
        'red_execution': 'audits/execution/belief-gap-v2-adversarial.json',
        'oracle': 'reviews/oracle/belief-gap-v2.jsonl',
        'current_formal_report': 'audits/diversity/belief-gap-v2-belief-controls.json',
        'current_root_audit': 'audits/leakage/belief-gap-v2.json',
        'reproduction': 'quality_reports/BELIEF_GAP_V2_REPRODUCTION.json',
        'reproduction_supplement': 'quality_reports/BELIEF_GAP_V2_REPRODUCTION_SUPPLEMENT.json',
        'reproduction_issue': 'feedback/improvement_requests/RBM-REPRO-010.json',
        'original_shortcut_report': 'data/RBM-001/manifests/belief-gap-v2.shortcut_audit.json',
        'reproduced_shortcut_report': 'reproducibility/failed_belief_gap_v2/data/RBM-001/manifests/belief-gap-v2.shortcut_audit.json',
        'public_predicates': 'schemas/BELIEF_SHORTCUT_DIAGNOSTICS.json',
        'public_properties': 'schemas/BELIEF_CONTROL_PROPERTIES.md',
        'public_v2_properties': 'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md',
        'public_v2_curriculum': 'research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md',
        'own_solver_source': 'reviews/adjudication/audit_belief_gap_v1.py',
        'own_execution_source': 'reviews/adjudication/audit_and_adjudicate_belief_gap_v2.py',
        'prior_candidate': 'data/RBM-001/candidate_renderings/belief-gap-v1.jsonl',
    }
    bindings = {k: {'path': p, 'sha256': sha(BASE / p)} for k, p in paths.items()}
    assert bindings['candidate']['sha256'] == '589119d69290cf91f8cb00adf49d280f7499a6121e9064655ae523222e9e72df'
    assert bindings['expert']['sha256'] == '91b8c18ed026f6d296d7d4ace73b35420e5b99100c25790e47c815b5479bd5e1'
    assert bindings['red']['sha256'] == '21ec9d792759d6f793eb346c893b25b44bb9e27ede560028d46ffb7397fbdde0'
    rex = json.loads((BASE / paths['red_execution']).read_text())
    assert rex['status'] == 'completed' and rex['candidate_sha256'] == bindings['candidate']['sha256']
    assert rex['final_receipts_sha256'] == bindings['red']['sha256']
    assert bindings['expert']['sha256'] == bindings['expert_first']['sha256']
    machine = json.loads((BASE / paths['current_formal_report']).read_text())
    root = json.loads((BASE / paths['current_root_audit']).read_text())
    assert not machine['report']['errors']
    assert machine['checker_sha256'] == '5b97b73df0b9248b1ad155307e75e2a73b995dff64e81232382a6c05a2fe4984'
    assert machine['report']['counts']['executed_probes'] == 84
    zero_keys = ('oracle_failures', 'metadata_failures', 'validation_failures', 'belief_control_error_count')
    assert all(root[k] == 0 for k in zero_keys) and root['records'] == 1728
    reproduction = json.loads((BASE / paths['reproduction']).read_text())
    supplement = json.loads((BASE / paths['reproduction_supplement']).read_text())
    assert reproduction['pass'] is False and supplement['exact_artifacts'] == 6 and supplement['total_artifacts'] == 7
    assert supplement['candidate_and_parents_byte_identical'] is True
    original_diagnostic = json.loads((BASE / paths['original_shortcut_report']).read_text())
    reproduced_diagnostic = json.loads((BASE / paths['reproduced_shortcut_report']).read_text())
    assert original_diagnostic == reproduced_diagnostic
    assert bindings['original_shortcut_report']['sha256'] != bindings['reproduced_shortcut_report']['sha256']
    candidates = rows(paths['candidate'])
    expert = {r['sample_id']: r for r in rows(paths['expert'])}
    red = {r['sample_id']: r for r in rows(paths['red'])}
    oracle = {r['sample_id']: r for r in rows(paths['oracle'])}
    assert len(candidates) == 1728 and len({r['sample_id'] for r in candidates}) == 1728
    assert all(len(x) == 1728 and set(x) == {r['sample_id'] for r in candidates} for x in (expert, red, oracle))
    index = {(r['source_family_index'], r['controlled_context'], r['schedule_index'], r['renderer']): r for r in candidates}
    prior = rows(paths['prior_candidate'])
    prior_index = {(r['source_family_index'], r['controlled_context'], r['schedule_index'], r['renderer']): r for r in prior}
    properties, stats, scalar_hist, axis_groups = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    details, expected, prefixes_by_id = [], {}, {}
    new_contexts, changed_F, unchanged_core = 0, 0, 0
    full_core_correct, renamed_exact, renamed_compound_misses = 0, 0, 0
    for r in candidates:
        sid = r['sample_id']
        assert hashlib.sha256(canonical({k: v for k, v in r.items() if k != 'record_hash'}).encode()).hexdigest() == r['record_hash']
        assert r['rendered_input'].count('\nTask JSON:\n') == 1
        task = json.loads(r['rendered_input'].split('\nTask JSON:\n', 1)[1])
        assert canonical(task) == canonical(r['semantic_task'])
        assert 'controlled_context' not in r['rendered_input'] and 'property_witnesses' not in r['rendered_input']
        target, prefixes = solve(task)
        expected[sid], prefixes_by_id[sid] = target, prefixes
        assert canonical(target) == canonical(r['verified_target'])
        tags = property_audit(r, task, target, prefixes)
        properties.update(tags)
        assert expert[sid]['record_hash'] == red[sid]['record_hash'] == oracle[sid]['record_hash'] == r['record_hash']
        assert oracle[sid]['pass'] is True and not oracle[sid]['errors'] and not oracle[sid]['metadata_errors']
        assert expert[sid]['decision'] == red[sid]['decision'] == ('REVISE' if r['controlled_context'] in 'EF' else 'ACCEPT')
        empty = not prefixes[-1]
        p1, p2 = marker(task), compound(task)
        assert p2 == empty
        groups = ['overall', f"split:{r['split']}", f"family:{r['source_family_index']}",
                  f"family_length:{r['source_family_index']}:{len(task['observations'])}"]
        for key in groups:
            stats[key].update(records=1, empty_records=int(empty), original_rule_errors=int(p1 != empty),
                              compound_status_correct=int(p2 == empty))
        if p2:
            assert len(task['worlds']) in (8, 12)
            fixed = {'surviving_worlds': [],
                     'belief_sizes': [1 if len(task['worlds']) == 8 else 2] + [0] * (len(task['observations']) - 1),
                     'queries': [{'query_id': q['query_id'], 'counts': [], 'distinct_values': 0, 'entailed': False,
                                  'guaranteed': None, 'can_reduce_uncertainty': False} for q in task['queries']]}
            assert canonical(fixed) == canonical(target)
            full_core_correct += 1
            for key in groups:
                stats[key]['compound_complete_empty_target_correct'] += 1
        auxiliary = r['fact_role_assignments']['observation_auxiliary']
        assert auxiliary not in {q['fact'] for q in task['queries']}
        renamed = copy.deepcopy(task)
        replacement = '__actual_adjudicator_aux_opaque_linen__'
        assert all(w['facts'][auxiliary] != replacement for w in task['worlds'])
        for w in renamed['worlds']:
            if scalar(w['facts'][auxiliary]) == scalar('linen'):
                w['facts'][auxiliary] = replacement
        for o in renamed['observations']:
            if o['fact'] == auxiliary and scalar(o['equals']) == scalar('linen'):
                o['equals'] = replacement
        rt, rp = solve(renamed)
        assert canonical(rt) == canonical(target) and [ids(p) for p in rp] == [ids(p) for p in prefixes]
        renamed_exact += 1
        renamed_compound_misses += int(compound(renamed) != empty)
        for value in {scalar(o['equals']) for o in task['observations']}:
            scalar_hist[value]['empty' if empty else 'nonempty'] += 1
        assert r['split'] == ('train' if r['source_family_index'] % 10 < 8 else 'dev_challenge')
        assert r['schedule_index'] == 9 * r['id_shift'] + 3 * r['position_shift'] + r['fact_name_shift']
        axis_groups[(r['source_family_index'], r['controlled_context'], r['renderer'])].append(r)
        key = (r['source_family_index'], r['controlled_context'], r['schedule_index'], r['renderer'])
        if r['controlled_context'] in 'GH':
            new_contexts += 1
            assert r['difficulty'] == 2
            assert len([i for i in range(len(task['observations'])) if 0 < len(prefixes[i + 1]) < len(prefixes[i])]) == 1
        else:
            old = prior_index[key]
            assert canonical(old['semantic_task']['worlds']) == canonical(task['worlds'])
            assert canonical(old['semantic_task']['queries']) == canonical(task['queries'])
            if r['controlled_context'] == 'F':
                assert canonical(old['semantic_task']['observations'][:2]) == canonical(task['observations'][:2])
                assert canonical(old['semantic_task']['observations'][2]) != canonical(task['observations'][2])
                assert canonical(task['observations'][2]) == canonical(task['observations'][0])
                changed_F += 1
            else:
                assert canonical(old['semantic_task']) == canonical(task)
                unchanged_core += 1
        details.append({'sample_id': sid, 'record_hash': r['record_hash'], 'context': r['controlled_context'],
                        'source_family_index': r['source_family_index'], 'split': r['split'], 'empty': empty,
                        'original_marker_prediction': p1, 'compound_marker_prediction': p2,
                        'independent_target_sha256': hashlib.sha256(canonical(target).encode()).hexdigest(),
                        'prefix_survivors': [ids(p) for p in prefixes], 'properties': sorted(tags),
                        'auxiliary_rename_complete_output_exact': True,
                        'compound_wrong_on_auxiliary_rename': compound(renamed) != empty})
    assert full_core_correct == 432 and renamed_exact == 1728 and renamed_compound_misses == 432
    assert new_contexts == 432 and changed_F == 216 and unchanged_core == 1080
    assert len(axis_groups) == 64
    for key, group in axis_groups.items():
        assert len(group) == 27 and {r['schedule_index'] for r in group} == set(range(27))
        histogram = defaultdict(list)
        for r in group:
            for role_idx, role in enumerate(('invariant_query', 'variation_query', 'observed_control')):
                loc = next(x for x in r['query_role_assignments'] if x['role'] == role)
                assert loc['position'] == (role_idx + r['position_shift']) % 3
                assert loc['query_id'] == r['query_id_pool'][(role_idx + r['id_shift']) % 3]
                q = r['semantic_task']['queries'][loc['position']]
                assert q['query_id'] == loc['query_id'] and q['fact'] == r['queried_fact_name_pool'][(role_idx + r['fact_name_shift']) % 3]
                a = copy.deepcopy(expected[r['sample_id']]['queries'][loc['position']]); del a['query_id']
                histogram[(q['query_id'], loc['position'], q['fact'])].append(canonical({
                    'answer': a, 'sizes': expected[r['sample_id']]['belief_sizes'],
                    'survivors': expected[r['sample_id']]['surviving_worlds']}))
        assert len(histogram) == 27 and len({canonical(sorted(v)) for v in histogram.values()}) == 1
    pair_count = 0
    for family in (12, 13, 18, 19):
        for schedule in range(27):
            for renderer in ('forward', 'reverse'):
                for left, right in (('E', 'G'), ('F', 'H')):
                    a, c = index[(family, left, schedule, renderer)], index[(family, right, schedule, renderer)]
                    at, ct = a['semantic_task'], c['semantic_task']
                    assert canonical(at['worlds']) == canonical(ct['worlds']) and canonical(at['queries']) == canonical(ct['queries'])
                    assert len(at['observations']) == len(ct['observations'])
                    differences = [i for i in range(len(at['observations'])) if canonical(at['observations'][i]) != canonical(ct['observations'][i])]
                    assert differences == [1] and at['observations'][1]['fact'] == ct['observations'][1]['fact']
                    assert ids(prefixes_by_id[a['sample_id']][1]) == ids(prefixes_by_id[c['sample_id']][1])
                    assert not prefixes_by_id[a['sample_id']][-1] and prefixes_by_id[c['sample_id']][-1]
                    if left == 'F':
                        assert canonical(at['observations'][2]) == canonical(at['observations'][0])
                        assert canonical(ct['observations'][2]) == canonical(ct['observations'][0])
                    pair_count += 1
    assert pair_count == 432
    for r in candidates:
        if r['renderer'] == 'forward':
            c = index[(r['source_family_index'], r['controlled_context'], r['schedule_index'], 'reverse')]
            assert canonical(r['semantic_task']) == canonical(c['semantic_task'])
            assert r['rendered_input'] != c['rendered_input']
    prior_families = {r['source_family_key'] for r in prior}
    assert {r['source_family_key'] for r in candidates} == prior_families and len(prior_families) == 4
    historical_paths = [
        'audits/diversity/history/belief-gap-v2-belief-controls-6654a6fcaeee4e3e5cc6154009ee67e266fcfc8b829f718c68350bbe49d2b3af.json',
        'audits/leakage/history/belief-gap-v2-24e73dad24495d532f8a18fbb78409e5cbd00b14b6cff9e8208a62917e7ede6d.json',
        'reviews/oracle/history/belief-gap-v2-360a119b3265305f8fee2ba8d13752957f43fdcd01c45eae506523bfe7ae83dc.jsonl',
    ]
    audit_path = 'reviews/adjudication/belief-gap-v2.independent_adjudicator_audit.json'
    audit = {'status': 'COMPLETED_INPUT_DERIVED_V2_ADJUDICATOR_AUDIT', 'actual_execution_id': '/root/research_architect',
             'completed_at_utc': now, 'source_bindings': bindings, 'records': 1728,
             'exact_target_prompt_record_hash_schema_pass': 1728, 'independent_filter_direct_prefix_agreement': 1728,
             'property_counts': dict(sorted(properties.items())), 'full_typed_triple_histogram_groups': 64,
             'field_order_pairs': 864, 'matched_empty_positive_pairs': 432, 'source_families': 4,
             'semantic_base_contexts': 32, 'new_families_relative_to_v1': 0,
             'unchanged_A_to_E_tasks': unchanged_core, 'changed_F_tasks': changed_F, 'added_GH_tasks': new_contexts,
             'frozen_rule_statistics': {k: dict(v) for k, v in stats.items()},
             'typed_observation_scalar_outcome_histogram': {k: dict(v) for k, v in scalar_hist.items()},
             'compound_status_correct': 1728, 'compound_complete_empty_targets': 432,
             'compound_complete_target_abstentions': 1296,
             'auxiliary_column_scalar_rename_probes': 1728, 'auxiliary_rename_exact_full_output_prefix_pass': renamed_exact,
             'compound_rule_errors_after_auxiliary_rename': renamed_compound_misses,
             'counterexample_scope': 'Actual v2 audit probes, not prospective v3 table transpositions or production records.',
             'current_machine_counts': {k: root[k] for k in ('records', *zero_keys)},
             'current_formal_counts': machine['report']['counts'],
             'current_formal_checker_sha256': machine['checker_sha256'],
             'current_checker_dependency_sha256': machine['checker_dependency_sha256'],
             'first_failed_machine_report_bindings': [{'path': p, 'sha256': sha(BASE / p)} for p in historical_paths],
             'reproduction_scope': {'root_reported_pass': False, 'exact_artifacts': 6, 'total_artifacts': 7,
                                    'candidate_parents_byte_identical': True, 'own_parsed_diagnostic_equality': True,
                                    'diagnostic_byte_equal': False, 'reproduction_by_this_execution': False},
             'expert_decisions': dict(Counter(r['decision'] for r in expert.values())),
             'red_decisions': dict(Counter(r['decision'] for r in red.values())), 'expert_red_disagreements': 0,
             'partial_accept_missing_family_contexts': {str(f): ['E', 'F'] for f in (12, 13, 18, 19)},
             'same_actual_original_v1_v2_v3_design_author_and_later_adjudicator': True,
             'same_model_common_guidance_risk': True, 'producer_oracle_reviewer_implementation_read': False,
             'per_record': details}
    new_json(audit_path, audit)
    audit_sha = sha(BASE / audit_path)
    issue_path = 'feedback/improvement_requests/RBM-GAP-007-COMPOUND-EMPTY-MEASURED.json'
    issue = {'ticket': 'RBM-GAP-007-COMPOUND-EMPTY-MEASURED', 'status': 'CONFIRMED_DATA_QUALITY_REVISE',
             'author_execution_id': '/root/research_architect', 'created_at_utc': now, 'severity': 'high',
             'code': 'residual_observation_scalar_conjunction_empty_shortcut', 'candidate': bindings['candidate'],
             'rule': 'any typed string observation.equals in{stone,unknown} AND any typed string observation.equals==linen',
             'frozen_public_rule_source': bindings['public_predicates'],
             'independent_status_correct': 1728, 'actual_split_status': {'train': '864/864', 'dev_challenge': '864/864'},
             'complete_empty_targets_reproduced': 432, 'covered_query_rows_correct': 1296,
             'complete_target_abstentions': 1296,
             'full_target_prediction_inputs': 'Only observation scalars, world count8/12, observation count and query IDs/order; no world fact payloads.',
             'empty_trace_decoder': '[1 if world_count==8 else2] followed by observation_count-1 zeros',
             'counterexamples': '1728 coherent fact-local auxiliary linen->opaque string renamings preserve all full targets/prefix IDs exactly; compound rule then misses432emptyrows.',
             'input_target_truth': 'All1728 full typed targets/hash/prompt/property checks pass; corrected current formal report has0errors.',
             'factual_coverage': 'Four unchanged v1 families,32contexts,864parsed tasks,1728correlated renderings; G/H432rows simple, no new families.',
             'incoming_final_reviews': {'expert': {'ACCEPT': 1296, 'REVISE': 432}, 'adversarial': {'ACCEPT': 1296, 'REVISE': 432}},
             'disagreements': 0, 'adjudication': {'REVISE': 1728}, 'direct_lexical_defects': 432, 'dependency_holds': 1296,
             'dependency': 'Every family loses mandatory E/F if only individually ACCEPTed rows are released, so no complete A–H family/cohort is quality-approved.',
             'measured_evidence': {'path': audit_path, 'sha256': audit_sha},
             'reproduction_limit': {'issue': bindings['reproduction_issue'], 'report': bindings['reproduction'],
                                    'supplement': bindings['reproduction_supplement'], '6_of_7_exact_bytes': True,
                                    'candidate_and_parents_exact': True, 'diagnostic_parsed_equal_byte_different': True,
                                    'interpretation': 'Separate diagnostic custody limitation, not a target error.'},
             'prospective_fix': 'Frozen NEW v3 public fact-local auxiliary transposition design; fresh generation/formal/expert/red/adjudication required. No future efficacy or complete-shortcut-removal claim.',
             'design_role_overlap_disclosed': True, 'same_underlying_model_risk': True,
             'candidate_edits': 0, 'approvals': 0}
    new_json(issue_path, issue)
    by_detail = {r['sample_id']: r for r in details}
    receipt_path = 'reviews/adjudication/belief-gap-v2.jsonl'
    with (BASE / receipt_path).open('x') as f:
        for r in candidates:
            sid = r['sample_id']; d = by_detail[sid]; direct = d['empty']
            code = 'residual_observation_scalar_conjunction_empty_shortcut' if direct else 'required_family_control_dependency'
            receipt_id = 'adj-belief-gap-v2-' + hashlib.sha256(canonical(['/root/research_architect', sid, r['record_hash'], code]).encode()).hexdigest()[:24]
            receipt = {'receipt_id': receipt_id, 'sample_id': sid, 'record_hash': r['record_hash'],
                       'reviewer_role': 'independent_adjudicator', 'reviewer_execution_id': '/root/research_architect',
                       'completed_at_utc': now, 'decision': 'REVISE', 'confidence': 0.99,
                       'scores': {'semantic_correctness': 5, 'target_correctness': 5, 'reasoning_validity': 5,
                                  'instruction_clarity': 5, 'generalization_value': 2, 'diversity': 3,
                                  'shortcut_resistance': 2 if direct else 3, 'difficulty_quality': 3},
                       'critical_defects': [code],
                       'issues': [{'code': code, 'severity': 'high',
                                   'evidence': 'Frozen compound rule fits1728statuses and432full emptytargets; all4families need mandatory E/F corrections.',
                                   'fix': 'Preserve original batch and completed reviews; require a freshly reviewed complete corrected cohort.'},
                                  {'code': 'diagnostic_byte_reproduction_limitation', 'severity': 'medium',
                                   'evidence': '6/7artifact bytes reproduce; diagnostic JSON parsed equal but byte order differs. Candidate/parent bytes exact.',
                                   'fix': 'Preserve first failure/source; sort fresh diagnostic feature insertion and verify separate-process exact reproduction.'}],
                       'reason': ('Exact target is correct; compound observation-token rule bypasses this full empty target.' if direct else
                                  'Exact target is correct and both final reviewers ACCEPT; mandatory E/F of this family require revision, so this row has a cohort dependency hold.') + ' No partial A–H release is certified.',
                       'evidence': {'candidate_sha256': bindings['candidate']['sha256'], 'independent_audit': audit_path,
                                    'independent_audit_sha256': audit_sha, 'input_derived_target_sha256': d['independent_target_sha256'],
                                    'target_prompt_hash_schema_pass': True, 'direct_prefix_agreement': True,
                                    'context': d['context'], 'source_family_index': d['source_family_index'], 'split': d['split'],
                                    'prefix_survivors': d['prefix_survivors'], 'properties_independently_witnessed': d['properties'],
                                    'compound_prediction_empty': d['compound_marker_prediction'], 'empty': direct,
                                    'direct_lexical_defect': direct, 'dependency_hold': not direct,
                                    'expert_receipt_id': expert[sid]['receipt_id'], 'expert_decision': expert[sid]['decision'],
                                    'adversarial_receipt_id': red[sid]['receipt_id'], 'adversarial_decision': red[sid]['decision'],
                                    'expert_red_disagreement': False, 'expert_final_sha256': bindings['expert']['sha256'],
                                    'adversarial_final_sha256': bindings['red']['sha256'], 'oracle_file_sha256': bindings['oracle']['sha256'],
                                    'current_corrected_formal_report': bindings['current_formal_report'],
                                    'current_corrected_root_audit': bindings['current_root_audit'],
                                    'measured_issue_ticket': issue_path, 'required_family_contexts': list('ABCDEFGH'),
                                    'partial_incoming_accept_missing_contexts': ['E', 'F']},
                       'independence_scope': 'Actual v1/v2/v3 public-design author and later adjudicator; no generation or first reviewer authorship. Authorized completed reviews read, not blind first review; own earlier public-derived solver reused; shared-model/common-guidance risks.'}
            f.write(canonical(receipt) + '\n')
    report_path = 'reviews/adjudication/ADJUDICATION_BELIEF_GAP_V2.md'
    report = f'''# Actual v2 adjudication

**REVISE all1,728 records; zero release-eligible rows certified.** The432 E/F rows have a direct compound lexical shortcut;1,296 nonempty rows are held through mandatory A–H family dependencies. Their individual final ACCEPT decisions remain preserved. Exact targets are correct.

Actual author/adjudicator `/root/research_architect` also wrote v1/v2/v3 public designs. No production or initial expert/red judgments were authored by this identity. Completed review evidence was read after root's explicit final-completion confirmation. This is guided adjudication, with own frozen earlier public-derived solver reuse and shared-model/common-guidance risk; it is not a blind first review.

Candidate SHA256 `{bindings['candidate']['sha256']}` binds all1,728 decisions. Independent student-task typed filtering, direct-prefix conjunction agreement, full target/hash/prompt/schema and row property/locator checks pass. All64 actual full typed nuisance-triple histogram groups,864 field-order pairs and432 E/G or F/H same-first-prefix pairs pass. A–E's1,080 tasks remain identical to v1;216 F tasks change only their appended equality to repeat first;432 G/H tasks are new. Four source families/32contexts and864parsed tasks underpin1,728correlated renderings; zero families are added to v1. G/H's non-strict/singleton components retain honest Level2, not added complexity.

The exact frozen compound predicate (stone/unknown observation string AND linen observation string) predicts all1,728 statuses,864/864 per split. The public world-count8/12 and observation-length decoder plus visible ordered query IDs reproduces all432 complete empty targets and1,296query rows while abstaining on1,296nonempty complete targets. Whole-task conditional exactness is100% on25%coverage, not100%whole-batch complete-target accuracy. The original single-marker rule makes432positive G/H errors. I executed1,728 fact-local auxiliary linen-to-opaque bijections: all full outputs/prefix IDs remain exactly identical, yet the compound predicate misses every432empty row. These are actual audit probes, not future v3 records or extra families. Exact status and coverage statistics appear in the own audit/ticket.

Both final reviewers independently report1,296 ACCEPT/432 REVISE, so there is zero final expert/red disagreement. Expert first/final are byte-identical SHA256 `{bindings['expert']['sha256']}`. Red first1,728 ACCEPT remains SHA256 `{bindings['red_first']['sha256']}`; final SHA256 `{bindings['red']['sha256']}` records the later whole-batch finding. Actual completed execution hashes are expert `{bindings['expert_execution']['sha256']}` and red `{bindings['red_execution']['sha256']}`. Preserve all first/final evidence and prior failed execution sources.

Every family loses required E/F under an individually accepted-only selection. The frozen design requires full A–H controls and complete shift/style groups in each family. Thus no complete approved cohort remains; positive G/H and A–D targets are not themselves mislabeled, but their release dependencies are unmet. Correct target scores do not erase the separate high quality defect.

Root confirmed its corrective current audit completed before this adjudication. The actual report SHA256 `{bindings['current_formal_report']['sha256']}` uses checker `{machine['checker_sha256']}` and separately binds its unchanged prior dependency. It reports0errors,64histograms and84executed probes; root audit SHA256 `{bindings['current_root_audit']['sha256']}` has0oracle/metadata/validation/control errors. Historical6654/24e7/oracle360a reports retain the first1,728metadata failures/1,744control errors caused by an unpublished uniqueness assumption; those actual paths/hashes are separately bound. Do not present those failed diagnostics as current target errors. No corrective source implementation was read by this identity.

The actual coordinator reproduction remains FAILED:6of7 exact artifacts, with candidate/parents byte-identical. Only the diagnostic JSON byte order differs; own parsed comparison is equal. Original/current diagnostic SHA256 `{bindings['original_shortcut_report']['sha256']}`, failed reproduction SHA256 `{bindings['reproduced_shortcut_report']['sha256']}`. `RBM-REPRO-010.json` and the actual reproduction/supplement are exact-bound separately. Root reports unsorted set insertion as cause; I did not inspect producer source or rerun it. This is a diagnostic custody limitation, not a target discrepancy, and cannot be advertised as all-artifact byte reproduction success.

Frozen v3 public design remains unchanged and receives no approval here. Its actual production/review is a separate task. Future controls may challenge the frozen predicates but cannot establish all-shortcut removal, independent family expansion, model efficacy or sealed OOD. Existing75%nonempty distribution, positive-only literals, finite table lookup and repeated motifs remain limitations.

Final artifacts: `belief-gap-v2.jsonl`, `belief-gap-v2.independent_adjudicator_audit.json`, `execution-belief-gap-v2.json` and `feedback/improvement_requests/RBM-GAP-007-COMPOUND-EMPTY-MEASURED.json`. No candidate/gold/producer/release file or frozen v3 design was edited.
'''
    with (BASE / report_path).open('x') as f:
        f.write(report)
    outputs = [audit_path, issue_path, receipt_path, report_path, paths['own_execution_source']]
    receipt_rows = rows(receipt_path)
    assert len(receipt_rows) == 1728 and len({r['receipt_id'] for r in receipt_rows}) == 1728
    assert sum(r['evidence']['direct_lexical_defect'] for r in receipt_rows) == 432
    execution = {'status': 'COMPLETED_FROZEN_DATA_ADJUDICATION', 'actual_execution_id': '/root/research_architect',
                 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'records': 1728,
                 'decisions': {'REVISE': 1728}, 'direct_defects': 432, 'dependency_holds': 1296,
                 'final_reviewer_disagreements': 0, 'release_eligible_records': 0,
                 'actual_command': '/opt/anaconda3/bin/python3 High-Quality_Dataset/reviews/adjudication/audit_and_adjudicate_belief_gap_v2.py',
                 'exit_code': 0, 'source_bindings': bindings,
                 'output_sha256': {p: sha(BASE / p) for p in outputs},
                 'own_execution_failures': [], 'first_machine_failures_separately_preserved': audit['first_failed_machine_report_bindings'],
                 'current_machine_counts': audit['current_machine_counts'],
                 'current_formal_checker_sha256': machine['checker_sha256'],
                 'checker_dependency_sha256': machine['checker_dependency_sha256'],
                 'expert_first_final_preserved_equal': True, 'red_first_final_preserved_distinct': True,
                 'reproduction_failed_6of7_exact': True, 'diagnostic_parsed_equal_byte_different': True,
                 'producer_implementation_read': False, 'oracle_implementation_read': False,
                 'original_revision_design_and_later_adjudicator_overlap_disclosed': True,
                 'same_underlying_model_risk': True, 'candidate_or_release_edits': 0, 'approvals': 0,
                 'frozen_v3_sources': {p: sha(BASE / p) for p in ['schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md',
                     'research/curricula/RBM-EVIDENCE-GAP-REVISION-V3.md', 'research/curricula/RBM-EVIDENCE-GAP-REVISION-V3-EXECUTION.json']}}
    execution_path = 'reviews/adjudication/execution-belief-gap-v2.json'
    new_json(execution_path, execution)
    print(canonical({'status': execution['status'], 'decisions': execution['decisions'],
                     'output_sha256': {**execution['output_sha256'], execution_path: sha(BASE / execution_path)}}))


if __name__ == '__main__':
    main()
