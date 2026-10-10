"""Verify completed independent evidence bytes and identities; no quality vote."""
from pathlib import Path
import collections
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'BELIEF_V4_FINAL_REVIEW_JOIN_ATTEMPT01'
FROZEN = ROOT / 'quality_reports' / (PREFIX + '_SOURCE_FREEZE.json')
OUTPUT = ROOT / 'quality_reports' / (PREFIX + '.json')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while part := source.read(65536):
            digest.update(part)
    return digest.hexdigest()


def path(name):
    value = Path(name)
    return value if value.is_absolute() else ROOT / value


def verify(rows):
    for row in rows:
        p = path(row['path'])
        assert p.is_file() and not p.is_symlink(), row['path']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], row['path']


def read(name):
    return json.loads(path(name).read_bytes())


def jsonl(name):
    with path(name).open(encoding='utf-8') as source:
        return [json.loads(line) for line in source]


def main():
    frozen = read(str(FROZEN))
    assert frozen['source_sha256'] == sha(Path(__file__))
    verify(frozen['bindings'])
    execution = read('reviews/adjudication/belief-gap-v4-execution-r1.json')
    before, after = execution['bindings_before'], execution['bindings_after']
    assert before == after and len(before) == 126
    assert execution['source_and_inputs_unchanged'] is True
    assert execution['completed_execution'] is True and execution['exit_code'] == 0
    assert execution['decision_counts'] == {'ACCEPT': 3456}
    assert execution['publication_eligible'] is True and execution['whole_cohort_publication_eligible'] is True
    assert execution['actual_author_and_execution_id'] == '/root/research_architect'
    pre = read('reviews/adjudication/belief-gap-v4-preexecution-r1.json')
    assert pre['bindings'] == before
    assert datetime.datetime.fromisoformat(pre['at_utc']) < datetime.datetime.fromisoformat(execution['started_at_utc'])
    verify(before)
    verify(execution['outputs'])
    final = read('reviews/adjudication/belief-gap-v4-FINAL-r1.json')
    audit = read('reviews/adjudication/belief-gap-v4-audit-r1.json')
    completion = read('reviews/adjudication/belief-gap-v4-completion-receipt-r1.json')
    verify(final['files'] + [final['candidate'], final['manifest'], final['preexecution'], final['completed_execution']])
    for value in (final, audit, completion):
        assert value['publication_eligible'] is True and value['whole_cohort_publication_eligible'] is True
        assert value['publication_performed'] is False
    assert audit['all_required_data_gates_pass'] is True
    assert audit['full_old_new_typed_preservation'] == 3456
    assert audit['added_semantic_tasks'] == audit['added_source_families'] == 0
    assert completion['actual_adjudication_tool_execution']['exit_code'] == 0
    assert completion['all126_original_preexecution_bindings_unchanged'] is True
    assert completion['all_final_output_bindings_rechecked'] is True
    rows = jsonl('data/RBM-001/candidate_renderings/belief-gap-v4.jsonl')
    candidate = {r['sample_id']: r['record_hash'] for r in rows}
    assert len(candidate) == len(rows) == 3456
    roles = {'expert': ('/root/expert_reviewer', 'expert'),
             'adversarial': ('/root/adversarial_auditor', 'Adversarial Red-Team'),
             'adjudication': ('/root/research_architect', 'independent_adjudicator')}
    all_ids, counts = [], {}
    for role, (actor, role_name) in roles.items():
        reviews = jsonl('reviews/' + role + '/belief-gap-v4.jsonl')
        assert len(reviews) == len({r['sample_id'] for r in reviews}) == 3456
        assert {r['sample_id'] for r in reviews} == set(candidate)
        assert all(r['record_hash'] == candidate[r['sample_id']] for r in reviews)
        assert all(r['reviewer_execution_id'] == actor and r['reviewer_role'] == role_name for r in reviews)
        assert all(r['decision'] == 'ACCEPT' and not r.get('critical_defects') for r in reviews)
        assert not any(i.get('severity') in ('high', 'critical') for r in reviews for i in r.get('issues', []))
        counts[role] = dict(collections.Counter(r['decision'] for r in reviews))
        all_ids.extend(r['receipt_id'] for r in reviews)
    assert len(all_ids) == len(set(all_ids)) == 10368
    first = jsonl('reviews/expert/belief-gap-v4.canonical-first-judgments.jsonl')
    first_counts = dict(collections.Counter(r['decision'] for r in first))
    assert first_counts == {'REVISE': 3456}
    verify(frozen['bindings'])
    verify(before)
    value = {'status': 'ROOT_VERIFIED_COMPLETED_INDEPENDENT_REVIEW_CUSTODY_JOIN',
             'runner_execution': '/root', 'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source_sha256': sha(Path(__file__)), 'source_freeze_sha256': sha(FROZEN),
             'bindings': frozen['bindings'], 'original126_adjudicator_bindings_unchanged': True,
             'exact_records': 3456, 'actual_unique_role_receipt_joins': 10368,
             'final_role_decision_counts': counts, 'first_expert_counts_preserved': first_counts,
             'independent_publication_eligible': True, 'authoritative_adjudicator_execution': '/root/research_architect',
             'publication_performed': False, 'root_supplies_quality_vote': False,
             'remote_status': 'These post-cut completed review/custody artifacts require a later exact backup.',
             'disclosures': final['disclosures'], 'remaining_limits': audit['remaining_limits']}
    with OUTPUT.open('x', encoding='utf-8') as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.write('\n')
    print(json.dumps({'status': value['status'], 'sha256': sha(OUTPUT), 'bindings': len(frozen['bindings']), 'joins': 10368}))


if __name__ == '__main__':
    main()
