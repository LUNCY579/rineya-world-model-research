"""Own stdlib-only finished review metadata/hash readback. Never imports targets."""
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path('/Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset')
OUT = ROOT / 'quality_reports/RSM_CANONICAL_QUALITY_INDEPENDENT_READBACK_ATTEMPT01'
E = ROOT / 'reviews/expert/RSM-option-resumption-pilot-v1-canonical-preparation-attempt03'
R = ROOT / 'reviews/adversarial/RSM-option-resumption-pilot-v1-canonical-attempt03'
READS = {}
ERRORS = []
PAIRS = []
RUBRICS = {'semantic_correctness', 'target_correctness', 'instruction_clarity', 'reasoning_validity',
           'generalization_value', 'difficulty_quality', 'diversity', 'shortcut_resistance'}

def need(condition, label):
    if not condition:
        ERRORS.append(label)

def read(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if any(x.is_symlink() for x in [p, *p.parents]) or not p.is_file():
        raise ValueError('Nonregular or symlink byte identity: ' + str(p))
    before = p.stat()
    fd = os.open(p, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as f:
        opened = os.fstat(f.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError('Replaced before opening: ' + str(p))
        raw = f.read()
    after = p.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if tuple(getattr(before, k) for k in keys) != tuple(getattr(after, k) for k in keys) or len(raw) != before.st_size:
        raise ValueError('Changed during read: ' + str(p))
    READS[str(p)] = {'path': str(p), 'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return raw

def strict(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def obj(path, expected=None):
    raw = read(path)
    if expected is not None:
        need(hashlib.sha256(raw).hexdigest() == expected, 'External SHA: ' + str(path))
    return strict(raw)

def rows(path):
    return [strict(line) for line in read(path).splitlines() if line]

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def same(left, right):
    return canonical(left) == canonical(right)

def epair(row, label):
    p = Path(row['absolute_path'])
    raw = read(p)
    stored = read(E / row['archive_path'])
    need(type(row['bytes']) is int and row['bytes'] >= 0 and len(raw) == row['bytes']
         and hashlib.sha256(raw).hexdigest() == row['sha256'] and raw == stored,
         'Expert exact current/raw: ' + str(p))
    if 'path' in row:
        need(ROOT / row['path'] == p, 'Expert relative/absolute identity: ' + str(p))
    PAIRS.append({'group': label, 'current': str(p), 'archive': str(E / row['archive_path']),
                  'size': row['bytes'], 'sha256': row['sha256']})

def module_snapshot(snapshot, sourcefreeze, main, own_paths=()):
    declared = {m['module']: m for m in sourcefreeze['runtime']['module_closure']}
    runtime_paths = {r['absolute_path'] for r in sourcefreeze['runtime']['file_bindings']}
    source_paths = {r['absolute_path'] for r in sourcefreeze['bindings']}
    seen = set()
    for row in snapshot:
        name = row['module']
        need(name not in seen, 'Duplicate actual expert runtime module: ' + name)
        seen.add(name)
        p = row.get('absolute_path')
        if name == '__main__':
            need(p == str(main), 'Exact expert actual main identity')
        elif p in own_paths:
            need(p in source_paths and row['origin'] == p, 'Expert actual helper source: ' + name)
        else:
            wanted = declared.get(name)
            need(wanted is not None and wanted['origin'] == row['origin'], 'Expert runtime declared origin: ' + name)
            if p:
                need(wanted is not None and wanted.get('absolute_path') == p and p in runtime_paths,
                     'Expert runtime exact source membership: ' + name)
        if row.get('file_backed') is False:
            need(row['origin'] in ('built-in', 'frozen'), 'Expert nonfile runtime classification: ' + name)
        cache = row.get('cached_path')
        if row.get('cached_file_exists') is True:
            need(cache in runtime_paths or cache in source_paths, 'Expert actual existing cache membership: ' + name)
    return len(seen)

own_sha = sys.argv[1]
need(hashlib.sha256(read(Path(__file__).absolute())).hexdigest() == own_sha, 'Own reader source anchor')
need(hashlib.sha256(read(OUT / 'source_blobs' / (own_sha + '.source'))).hexdigest() == own_sha,
     'Own reader archived source anchor')
efinal = obj(E / 'QUALITY-PHASE-FINAL.json', '25a748d81a81dfe5508f65255295bb86ba2ceff8cd3629531f504702ec668c79')
rf = obj(R / 'FINAL_QUALITY_CUSTODY.json', 'b78d6967cd3f4930c226ee9c29194c5fab9228651c89a6da4147815d79142cc0')
eq = obj(E / 'quality-map.first.json', '2cdc099f1fb036063e929d101144d59832eeacec6e90fa5c8bfc38279be9ecdb')
rq = obj(R / 'QUALITY_MAP.json', 'a9e8e42965ca3c22117898a7ab5a7ceeb1e2fbeb6cb8207269a2dd23cad8b704')
efirst = obj(E / 'QUALITY-FIRST.json', '9ba8cdc217661a445d63c03d0efe97a88be1becce3a7900e7ecf54aed09c839d')
eqf = obj(E / 'QUALITY-MAP-FREEZE.json', '53bd7fa8d4cbce7fe693f6791d65708138933dd5a243c0d9c31666985c8e12dd')
esf = obj(E / 'SOURCE-PREPARATION-FREEZE.json', '91e91429d941a58790db4d9e716e85f332d66b29a9887547efa59530a8df6936')
pre = obj(E / 'QUALITY-AUTHORSHIP-PREEXECUTION.json', 'ef597abc513d76166a690134349aa11b0379aae0d7e21878fbaa12ff7c4655e5')
ex = obj(E / 'QUALITY-AUTHORSHIP-EXECUTION.json', 'cd741f890aa7693e8537b1a09eb127f549485b1f52be9ace3ccd89a8167064d6')
ec = obj(E / 'config.json', 'a3dbb827bdb386c4e54660c1d14b5a8d0173dfdc3b4fcdf2a2959841a9621d94')
em = obj(E / 'MANUAL-QUALITY-AUTHORSHIP.json', '8f6c22f179c04bdc9876c59580e9c5280fd64722a633557806559e24502b4f95')
rm = obj(R / 'MANUAL_QUALITY_LEDGER.json', 'd5faf123bad5dd714c629e43bc205663e4fe79b6510a5731591ee2b7ca2ffccd')
candidate = rows(ROOT / ec['candidate'])
need(READS[str(ROOT / ec['candidate'])]['sha256'] == 'c20eebd1eba211bd069ae9424d2d5a8094f657432fbe25ffffdcc5d307756bef',
     'Exact canonical candidate whole bytes')
by_id = {r['sample_id']: r for r in candidate}
need(len(candidate) == len(by_id) == 216, 'Exact unique original216 candidate IDs')
for row in candidate:
    need(hashlib.sha256(canonical({k: v for k, v in row.items() if k != 'record_hash'}).encode()).hexdigest()
         == row['record_hash'], 'Full original record hash: ' + row['sample_id'])
    need(row['review_status'] == 'PENDING' and row['review_receipt_ids'] == []
         and row['source_provenance']['sealed_evaluation_used'] is False, 'Original PENDING unapproved custody')
efactual = rows(E / 'canonical-comparison.first.jsonl')
rfactual = rows(R / 'FACTUAL_FIRST.jsonl')
joins = {sid: row['record_hash'] for sid, row in by_id.items()}
for label, data in [('expert-factual', efactual), ('adversarial-factual', rfactual), ('adversarial-quality', rq['rows'])]:
    actual = {r['sample_id']: r['record_hash'] for r in data}
    need(len(data) == len(actual) == 216 and same(actual, joins), label + ' complete exact original ID/hash join')
need(set(eq['per_sample']) == set(by_id), 'Expert quality map complete original IDs')
efacts = {row['sample_id']: row for row in efactual}
rfacts = {row['sample_id']: row for row in rfactual}
rquality = {row['sample_id']: row for row in rq['rows']}
rfamilies = {row['family_index']: row for row in rm['family_reviews']}
histograms = {'expert': {}, 'adversarial': {}}
for actor in histograms:
    histograms[actor] = {k: collections.Counter() for k in RUBRICS}
for sid, source in by_id.items():
    context = source['controlled_context']
    family = source['source_family_index']
    erow = eq['per_sample'][sid]
    rrow = rquality[sid]
    eprofile = em['context_assessments'][context]
    escores = dict(zip(em['rubric_order'], eprofile['scores']))
    escores.update(em['family_score_overrides'].get(str(family), {}))
    eissues = [dict(em['moderate_issues_authored'][name], issue_id=name)
               for name in em['context_specific_issue_selection'][context]]
    need(erow['sample_id'] == sid and erow['record_hash'] == joins[sid], 'Expert exact map row ID/hash')
    need(same(erow['scores'], escores) and erow['decision'] == em['all_sample_decision_authored']
         and same(erow['issues'], eissues) and same(erow['confidence'], eprofile['confidence'])
         and erow['critical_defects'] == [], 'Expert exact authored expansion: ' + sid)
    need(same(rrow['scores'], rm['context_profiles'][context]['scores'])
         and rrow['decision'] == rfamilies[family]['decisions'][context]
         and same(rrow['issues'], rm['shared_issues']) and same(rrow['confidence'], rm['confidence'])
         and same(rrow['critical_defects'], rm['critical_defects']), 'Adversarial exact authored expansion: ' + sid)
    for label, q in [('expert', erow), ('adversarial', rrow)]:
        need(set(q['scores']) == RUBRICS and all(type(n) is int and 1 <= n <= 5 for n in q['scores'].values()),
             label + ' exact eight typed rubric values')
        need(q['decision'] in ('ACCEPT', 'REVISE', 'REJECT', 'QUARANTINE') and type(q['issues']) is list
             and type(q['critical_defects']) is list, label + ' exact quality fields')
        for key, value in q['scores'].items():
            histograms[label][key][value] += 1
    need(efacts[sid]['failures'] == [] and efacts[sid]['target_equal_to_immutable_first'] is True
         and efacts[sid]['quality_decision'] == 'UNASSESSED_PENDING_ACTUAL_AGENT_INSPECTION',
         'Expert factual FIRST remains factual and complete')
    need(all(check['passed'] is True for check in rfacts[sid]['checks']), 'Adversarial factual check status')
    status = source['verified_target']['status']
    need(erow['evidence']['actual_status'] == status
         and rfacts[sid]['actual_execution_for_later_difficulty_inspection']['status'] == status,
         'Review factual status agrees with exact candidate')
for group, descriptors in [('publication', eqf['publication_input_bindings']), ('preexecution', pre['bindings']),
                           ('comparison-source', esf['bindings']), ('runtime', esf['runtime']['file_bindings'])]:
    need(len({r['absolute_path'] for r in descriptors}) == len(descriptors), 'Expert unique ' + group + ' descriptors')
    for row in descriptors:
        epair(row, group)
module_counts = {}
for label in ('before', 'after', 'postwrite'):
    module_counts['quality_' + label] = module_snapshot(ex[label]['loaded_module_membership'], esf, E / 'serialize_authored_quality.py')
    need(ex[label]['source_runtime_raw_equality'] is True, 'Expert actual quality raw guard status ' + label)
compare = obj(E / 'COMPARISON-EXECUTION.json', '315019906c5fbd694ce1a599c837990b2a04227d21c6b04f61245f1b2d386461')
own = {str(ROOT / ec[k]) for k in ('own_solver', 'own_calibration')} | {str(E / n) for n in ('own_normalization.py', 'new_calibrations.py')}
for label in ('custody_before', 'custody_after'):
    module_counts['comparison_' + label] = module_snapshot(compare[label]['loaded_module_membership'], esf, E / 'comparison_review.py', own)
    need(compare[label]['typed_SHA_size_exact_raw_archives_verified'] is True, 'Expert actual comparison raw guard ' + label)
for label in ('runtime_equality_and_membership_before', 'runtime_equality_and_membership_after'):
    module_counts['seal_' + label] = module_snapshot(eqf[label]['modules'], esf, E / 'freeze_quality_custody.py')
    need(eqf[label]['current_raw_equality'] is True, 'Expert actual seal raw guard ' + label)
need(ex['exit_code'] == compare['exit_code'] == 0 and compare['all_bound_bytes_unchanged'] is True,
     'Expert actual successful quality/comparison status')
need(efirst['all216_complete'] is True and efirst['publication_approval'] is False and efirst['no_emitter_run'] is True
     and eq['publication_approval'] is False and efinal['emitter_run'] is False, 'Expert stop before emission/approval')
checks = obj(E / 'canonical-check-ledger.json')
need(checks['passed'] == 7644 and checks['failed'] == 0
     and sum(check['holds'] is True for check in checks['checks']) == 7644
     and all(check['holds'] is True for check in checks['checks']), 'Expert actual recorded check-ledger counts')
report = {
    'schema': 'independent-canonical-quality-readback-v1',
    'actual_reviewer_execution': '/root/rsm_core_integration_auditor',
    'status': 'MATCHED_FINISHED_QUALITY_SOURCE_AND_CUSTODY' if not ERRORS else 'READBACK_MISMATCHES',
    'reader_source_sha256': own_sha, 'errors': ERRORS, 'candidate_count': len(candidate),
    'all216_current_records_remain_PENDING_and_unapproved': all(row['review_status'] == 'PENDING' and row['review_receipt_ids'] == [] for row in candidate),
    'exact_original_full_record_hashes_and_both_factual_quality_joins': not ERRORS,
    'exact_both_manual_ledger_to_quality_map_expansions': not ERRORS,
    'expert_unique_publication_current_raw_pairs': len(eqf['publication_input_bindings']),
    'expert_preexecution_current_raw_pairs': len(pre['bindings']),
    'expert_source_current_raw_pairs': len(esf['bindings']),
    'expert_runtime_current_raw_pairs': len(esf['runtime']['file_bindings']),
    'expert_actual_loaded_module_snapshot_counts': module_counts,
    'recorded_factual_check_counts_only': {'expert': 7644, 'adversarial': rf['factual_checks']['passed']},
    'score_histograms': {actor: {key: dict(value) for key, value in mapping.items()} for actor, mapping in histograms.items()},
    'actual_bytes_hashed': sorted(READS.values(), key=lambda row: row['path']),
    'expert_current_raw_pair_rows': PAIRS,
    'scope': 'Finished reviewers own manual judgments and custody readback only; original producer/reference source is hash bytes only, never inspected semantically.',
    'target_imports_tests_helpers_emitters_reviews_rerun': 0, 'Git_network_release_or_designated_receipt_writes': 0,
    'new_semantic_sample_vote_or_data_approval': False,
    'same_model_disclosure': 'Independent work with inherited same underlying model; child separately checks adversarial raw/runtime custody.',
}
with (OUT / 'QUALITY_READBACK.json').open('x') as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write('\n')
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({key: value for key, value in report.items()
                  if key not in ('actual_bytes_hashed', 'expert_current_raw_pair_rows', 'score_histograms')}, sort_keys=True))
if ERRORS:
    sys.exit(1)
