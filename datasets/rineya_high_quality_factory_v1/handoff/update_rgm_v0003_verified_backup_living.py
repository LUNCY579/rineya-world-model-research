#!/usr/bin/env python3
"""Preserve all ten living files, then document exact completed archival custody."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTHOR = '/root/backup_manager'
FREEZE = 'handoff/RGM_V0003_VERIFIED_BACKUP_LIVING_SOURCE_FREEZE.json'
MILESTONE = 'handoff/RGM_V0003_VERIFIED_BACKUP_LIVING_MILESTONE.json'
REPORT = 'quality_reports/RGM_V0003_VERIFIED_BACKUP_LIVING_UPDATE.json'
HISTORY = 'handoff/report_history/rgm-v0003-verified-backup-living'
COMMIT = '82787689fef4e50f7a35cfee505c290b2f6a6d78'
LIVING = [
    'PROJECT_STATUS.md', 'RESEARCH_GAPS.md', 'DATASET_QUALITY_REPORT.md',
    'GENERATOR_IMPROVEMENTS.md', 'CURRICULUM_COVERAGE.md',
    'TRAINING_READINESS.md', 'GITHUB_BACKUP_LOG.md',
    'NEXT_SESSION_HANDOFF.md', 'handoff/NEXT_SESSION_HANDOFF.md',
    'research/role_registry.json',
]
EXPECTED = {
    'backup_logs/rgm-v0003-independent-release_SPEC.json': 'caf6e0e4d0a0569206cec8c725130cbb4150160eb79f74be8113f18f8d496708',
    'backup_logs/rgm-v0003-independent-release_STAGE.json': '9ddf03eecbcc13d603659b951f6404b267ce7795bac392ebab22004af840f97d',
    'backup_logs/rgm-v0003-independent-release_RECOVERY_VERIFIED_RECEIPT.json': '83f5ad7628b3748504a8dac73fafb045d8c7f7acdb5c7d1c481205741d092f2f',
    'backup_logs/RGM_V0003_FINAL_REVIEW_JOINED_CUSTODY.json': '3ec1e2f75e42e827eff483fff190f349ceedee2d6b20dcc4619a315f16fa0bb1',
    'quality_reports/RGM_V0003_REMOTE_LOSSLESS.json': '21912e37c1e959e009a0eb80497be30a213a85d813e71d330868eff12d853de1',
    'handoff/RGM_V0003_BACKUP_COMPLETED_ADDENDUM.md': 'd8ef660956b15e72bf59e983a2cad7386cdedf40b7108f4aefdcc9feeb756638',
    'quality_reports/RGM_RENDERER_R1_RELEASE_EXECUTION.json': '6d7a318650b6133a8d6f3b60da6cf94a6351e8e2eccf879fe8c67304ef6b4ccb',
    'handoff/RGM_RENDERER_PUBLICATION_MILESTONE.json': 'dc71e31c53ffc0effc291315a13b6239a8a48f3c19cbaffcf0424aeada2377c2',
    'manifests/ACTIVE_RELEASE_CATALOG.json': '85a6b220aa0996717886fca76794aa2e1298a89047effb98d787c03094b2a852',
    'data/RGM-001/releases/v0003/canonical.jsonl': '83c03ee0af0f4bb7301198a3edc6c9ec6321ecd5687088a51a7edbf09cb0ef0c',
    'data/RGM-001/releases/v0003/MANIFEST.json': 'ab67c2b14d9a395d8e50cd078c4fff71c5698a9b2c4e5457d66705cd80943768',
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def proof(path):
    require(path.is_file() and not path.is_symlink(), 'Regular exact file required: ' + str(path))
    size = 0
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': size, 'sha256': digest.hexdigest()}


def exclusive_bytes(path, content):
    with path.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def exclusive_json(path, value):
    exclusive_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + '\n').encode())


def load(name):
    return json.loads((ROOT / name).read_bytes())


def main():
    started = now()
    source = proof(Path(__file__))
    freeze = load(FREEZE)
    require(freeze['source'] == source and freeze['author_execution'] == AUTHOR,
            'New updater was not source-frozen before execution')
    require(freeze['source_frozen_before_execution'] is True, 'Missing preexecution source freeze')
    freeze_proof = proof(ROOT / FREEZE)
    require(len(freeze['living_before']) == 10 and set(freeze['living_before']) == set(LIVING),
            'Exact ten-file preservation baseline required')
    evidence = {}
    for name, digest in EXPECTED.items():
        entry = proof(ROOT / name)
        require(entry['sha256'] == digest, 'Bound completed evidence changed: ' + name)
        evidence[name] = entry
    for name, entry in freeze['additional_public_or_completed_metadata_evidence'].items():
        require(proof(ROOT / name) == entry, 'Public/report evidence changed: ' + name)
        evidence[name] = entry
    require(not (ROOT / MILESTONE).exists() and not (ROOT / REPORT).exists(),
            'Fresh milestone/report outputs required')
    require(not (ROOT / HISTORY).exists(), 'Fresh immutable history directory required')

    spec = load('backup_logs/rgm-v0003-independent-release_SPEC.json')
    stage = load('backup_logs/rgm-v0003-independent-release_STAGE.json')
    full = load('backup_logs/rgm-v0003-independent-release_RECOVERY_VERIFIED_RECEIPT.json')
    joined = load('backup_logs/RGM_V0003_FINAL_REVIEW_JOINED_CUSTODY.json')
    gzip = load('quality_reports/RGM_V0003_REMOTE_LOSSLESS.json')
    catalog = load('manifests/ACTIVE_RELEASE_CATALOG.json')
    release = load('quality_reports/RGM_RENDERER_R1_RELEASE_EXECUTION.json')
    require(joined['status'] == 'BACKED_UP_FROZEN_INVENTORY_AND_REMOTE_LOSSLESS_PROOF_JOINED'
            and full['status'] == 'BACKED_UP' and gzip['status'] == 'PASS_ACTUAL_REMOTE_GZIP_DECODE',
            'Actual completed full and gzip joined proof required')
    require(joined['commit_sha'] == full['commit_sha'] == stage['commit_sha'] == gzip['commit_sha'] == COMMIT,
            'Completed commit binding differs')
    require(full['inventory'] == spec['files'] and len(spec['files']) == 2303
            and sum(x['size'] for x in spec['files']) == 928914914, 'Exact frozen scope differs')
    require(full['remote_verification']['verified_unique_blobs'] == 1895
            and joined['actual_full_unique_remote_fetches'] == 1895, 'Actual full remote fetch count differs')
    require(gzip['actual_remote_gzip_fetches'] == 24
            and gzip['raw_bytes_exactly_recovered'] == 1188634377
            and joined['actual_new_gzip_payload_fetches'] == 24, 'Actual new lossless proof count differs')
    require(joined['immutable_release_files_actual_bytes_verified'] == 210
            and joined['all_parent2095_paths_preserved'] is True
            and len(joined['six_protected_git_baselines']) == 6, 'Completed custody guards differ')
    require(joined['grants_data_approval'] is False and gzip['grants_data_approval'] is False,
            'Archival backup cannot grant data approval')
    require(release['status'] == 'COMPLETED_VALIDATED_LOCAL_RELEASE'
            and release['exact_inputs_unchanged'] is True
            and release['historical_release_files_unchanged'] is True, 'Original release execution differs')
    require(catalog['active_records'] == 3748 and catalog['active_source_families'] == 696,
            'Approved catalog changed')
    require(len(catalog['releases']) == 18
            and sum(x['active_records'] > 0 for x in catalog['releases']) == 16, 'Release counts changed')
    tracks = collections.Counter()
    for item in catalog['releases']:
        tracks[item['track']] += item['active_records']
    require(len(tracks) == 10 and sum(tracks.values()) == 3748 and tracks['RGM-001'] == 840,
            'Approved ten-track totals changed')
    require(any(x['track'] == 'RGM-001' and x['release'] == 'v0003' and x['active_records'] == 408
                for x in catalog['releases']), 'Approved RGM v0003 catalog entry absent')

    # Read only the authorized living files; no prospective candidate/checker/private fixtures.
    old_bytes = {}
    for name in LIVING:
        require(proof(ROOT / name) == freeze['living_before'][name], 'Living file changed before preservation: ' + name)
        old_bytes[name] = (ROOT / name).read_bytes()
    history = ROOT / HISTORY
    history.mkdir()
    archives = {}
    # Every one of the ten original byte streams is archived and verified BEFORE any living write.
    for name in LIVING:
        archive = history / (freeze['living_before'][name]['sha256'] + '-' + name.replace('/', '__'))
        exclusive_bytes(archive, old_bytes[name])
        entry = proof(archive)
        require((entry['sha256'], entry['bytes']) ==
                (freeze['living_before'][name]['sha256'], freeze['living_before'][name]['bytes']),
                'Preserved original bytes differ: ' + name)
        archives[name] = entry
    preserved = {'status': 'ALL10_ORIGINAL_LIVING_BYTES_PRESERVED_BEFORE_ANY_UPDATE',
                 'author_execution': AUTHOR, 'actual_runner_execution': AUTHOR, 'at_utc': now(),
                 'source': source, 'source_freeze': freeze_proof, 'originals': freeze['living_before'],
                 'archives': archives, 'all_originals_byte_exact': True, 'local_postinventory': True}
    exclusive_json(ROOT / 'backup_logs/RGM_V0003_LIVING_ALL10_PREWRITE_PRESERVATION.json', preserved)
    for name in LIVING:
        require(proof(ROOT / name) == freeze['living_before'][name], 'Living file changed before write: ' + name)

    section = '''# Current approved catalog and completed archival backup — 2026-10-10 UTC

The approved catalog remains **3,748 records / 696 source families across all ten tracks**, with 16 active releases and 18 preserved historical releases. **RGM-001/v0003** contains 408 independently accepted records (340 train / 68 exposed development); RGM contributes 840 active records / 228 families. The release and its completed expert, adversarial and adjudication evidence are included in the completed checkpoint below. This documentation update creates no release, approval or additional approved record. RGM v0003 reuses 204 semantic tasks / 48 families from original quarantined V2, adding zero semantic tasks or families relative to that cohort. Exposed development, repeated mechanism grammars, class imbalance, uncalibrated difficulty, same underlying model and public-design/adjudicator overlap remain disclosed. Learning benefit and sealed OOD transfer remain untested. No model training has occurred.

The latest fully verified private GitHub checkpoint is **82787689fef4e50f7a35cfee505c290b2f6a6d78**, sole parent `bc5198e11273cb6ea69659b6d3a8b5dc312b6f00`: **2,303 exact frozen files / 928,914,914 bytes**, with **1,895 new actual GitHub unique-blob fetches** by `/root`. A separate `/root/backup_manager` execution made **24 new actual remote gzip fetches**, decoded and compared byte-for-byte with **1,188,634,377 preserved local raw bytes**. Fresh source/target/index/commit checks, remote tree `37a927fb912199351bc6dcc3683c7860403e1ce4`, private ADMIN, both clean checkouts, all 2,095 parent paths, all 210 historical release files and six protected original Git byte/existence guards passed. Backup preserves bytes and grants no data-quality approval.

The exact authority is spec `caf6e0e4d0a0569206cec8c725130cbb4150160eb79f74be8113f18f8d496708`, stage `9ddf03eecbcc13d603659b951f6404b267ce7795bac392ebab22004af840f97d`, full remote proof `83f5ad7628b3748504a8dac73fafb045d8c7f7acdb5c7d1c481205741d092f2f`, joined custody `3ec1e2f75e42e827eff483fff190f349ceedee2d6b20dcc4619a315f16fa0bb1` and separate lossless proof `21912e37c1e959e009a0eb80497be30a213a85d813e71d330868eff12d853de1`. Their paths are `backup_logs/rgm-v0003-independent-release_SPEC.json`, `backup_logs/rgm-v0003-independent-release_STAGE.json`, `backup_logs/rgm-v0003-independent-release_RECOVERY_VERIFIED_RECEIPT.json`, `backup_logs/RGM_V0003_FINAL_REVIEW_JOINED_CUSTODY.json` and `quality_reports/RGM_V0003_REMOTE_LOSSLESS.json`.

**Only the frozen 2,303-file inventory is claimed backed up.** The new checkpoint spec, its later stage/full/gzip proofs and ledgers, joined receipt/addendum, this updater/source freeze/report history/current living update and postinventory V4 artifacts remain local until the next exact verified checkpoint. A receipt cannot include its own final bytes. Previously completed expert V4 component preparation (62 hand checks), the completed 12-file formal prospective preparation and public V4 encoding/addendum were included in 827. Official V4 work under `reviews/oracle/RBM-v4-official-20261010T073119Z/` and the newborn adversarial prefixes were explicitly deferred at scan start; later FINALs do not retroactively enter that frozen spec. Every earlier FAILED or PUSHED_PENDING receipt remains unchanged historical evidence.

Belief V1/V2/V3 remain quarantined. Per root's current scope attestations, the public-only official V4 checker has an actual 25-test FINAL, adversarial prospective preparation exercised 864 toy cases, and architect prospective preparation completed 28 cases; those later preparations remain local and approve no production data. The separately completed expert's 62 component checks are already included in 827. Producer V4 metadata repair and formal factory integration are currently running, with no completed production FINAL or data approval at this reporting boundary. This updater did not inspect prospective private producer expectations, reviewer procedures, fixtures or candidates. Public V4 work concerns ordered claim pairs, declared lengths, owner coordinates and unique named resolution. It is expected to reuse 3,456 V3 presentations with zero new semantic growth relative to V3; that expectation is not a verified production result. Approved V4 records remain zero and the active catalog remains 3,748 / 696.

Next: finish the independently owned formal integration and producer's exact source-frozen replay, then freeze a fresh stripped-input review view and obtain separate input-first reviews and actual independent adjudication before any V4 publication. Preserve all attempts, failures and original quarantine evidence. Carry the new local proof/report/preparation bytes into a later exact archival inventory; require actual remote-byte and lossless-decode completion for its backup claim. Dataset-only CPU work continues, with protected research, model weights/architecture, training and sealed evaluation outside this operation. All actors use the same underlying model; role separation does not imply model diversity.

---

## Preserved historical report text

All current-status and pending-backup statements below are preserved historical snapshots, superseded by the current section above. Their exact previous bytes are separately archived before this update.

'''
    registry = json.loads(old_bytes['research/role_registry.json'])
    rgm = registry['rgm_planning_wave']
    rgm.setdefault('v1_historical_candidate_records', rgm.get('candidate_records'))
    rgm.setdefault('v1_historical_semantic_tasks', rgm.get('semantic_tasks'))
    rgm.update({
        'stage': 'RENDERER_R1_ALL408_INDEPENDENT_ACCEPT_VALIDATED_AND_REMOTELY_BACKED_UP_V0003',
        'candidate_records': 408, 'prospective_records': 408, 'semantic_tasks': 204,
        'source_families': 48, 'approved_records': 408,
        'total_active_rgm_records': 840, 'total_active_rgm_source_families': 228,
        'latest_verified_backup_commit': COMMIT,
        'backup': '82787689 exact2303-file/928914914-byte scope fully verified through1895 new actual remote blobs and24 separate fresh gzip decodes; release/completed r1 reviews included. Later proofs and current living/V4 work remain local.',
        'renderer_revision_release_remotely_backed_up': True,
        'renderer_revision_completed_reviews_remotely_backed_up': True,
        'new_release_and_completed_reviews_local_only': False,
        'newer_milestone_artifacts_local_only': True,
    })
    registry['belief_evidence_wave'].update({
        'stage': 'V3_QUARANTINE_V4_PUBLIC_ONLY_OFFICIAL_FINAL_PRODUCER_AND_FORMAL_INTEGRATION_RUNNING',
        'v4_data_approval': False, 'v4_approved_records': 0,
        'v4_production_records': None,
        'v4_production_record_count_claim': 'Unknown at this attested ongoing boundary; no completed producer FINAL or approval.',
        'v4_official_checker_public_only_source_sha256': '0ff69cc8499d5f1fa8cfd2515d118f3a3a1ad43359bdf5c22bd3d4a81cab8613',
        'v4_official_checker_tests': 25, 'v4_official_checker_final_status': 'ROOT_ATTESTED_ACTUAL_FINAL_PUBLIC_ONLY_NO_DATA_APPROVAL_LOCAL',
        'v4_official_checker_prefix': 'reviews/oracle/RBM-v4-official-20261010T073119Z/',
        'v4_expert_component_preparation_checks': 62,
        'v4_expert_component_preparation_included_commit': COMMIT,
        'v4_adversarial_prospective_toy_cases': 864,
        'v4_architect_prospective_preparation_cases': 28,
        'v4_adversarial_and_architect_preparation_local_only': True,
        'v4_producer_status': 'ROOT_ATTESTED_RUNNING_NO_PRODUCTION_FINAL_OR_APPROVAL',
        'v4_formal_factory_integration_status': 'ROOT_ATTESTED_RUNNING_NOT_A_VALIDATED_PRODUCTION_GATE',
        'v4_expected_reused_presentations': 3456,
        'v4_expected_semantic_growth_relative_v3': 0,
        'v4_expectation_is_actual_production_result': False,
        'v4_status_evidence_boundary': 'Root scope attestations only. Backup manager did not inspect prospective private producer/reviewer internals.',
    })
    registry['current_backup_custody'] = {
        'commit_sha': COMMIT, 'sole_parent_sha': joined['sole_parent_sha'],
        'frozen_files': 2303, 'frozen_bytes': 928914914,
        'actual_full_unique_remote_fetches': 1895, 'actual_new_gzip_fetches': 24,
        'raw_bytes_exactly_recovered': 1188634377,
        'copy_stage_push_full_verification_runner': '/root',
        'gzip_and_join_author_runner': AUTHOR,
        'spec': evidence['backup_logs/rgm-v0003-independent-release_SPEC.json'],
        'stage': evidence['backup_logs/rgm-v0003-independent-release_STAGE.json'],
        'full_proof': evidence['backup_logs/rgm-v0003-independent-release_RECOVERY_VERIFIED_RECEIPT.json'],
        'joined_custody': evidence['backup_logs/RGM_V0003_FINAL_REVIEW_JOINED_CUSTODY.json'],
        'separate_gzip_proof': evidence['quality_reports/RGM_V0003_REMOTE_LOSSLESS.json'],
        'postinventory_new_proofs_reports_and_v4_work_local_only': True,
        'backup_grants_data_approval': False,
    }
    registry['backup_execution_note'] = 'Backup manager authored/froze/tested helpers; /root ran actual copy/stage/commit/nonforce push and full remote verification. /root/backup_manager ran separate fresh gzip proofs, custody joins and this living update. Actual roles are bound in receipts.'
    registry['living_report_refresh_execution'] = {
        'actual_author': AUTHOR, 'actual_runner': AUTHOR, 'at_utc': started,
        'milestone': MILESTONE, 'source': source, 'source_freeze': freeze_proof,
        'all_prior10_living_bytes_archived_before_any_write': True,
        'new_living_bytes_local_postinventory': True,
    }
    contents = {name: (json.dumps(registry, sort_keys=True, indent=2) + '\n').encode()
                if name.endswith('.json') else section.encode() + old_bytes[name] for name in LIVING}
    changes = []
    for name in LIVING:
        require(proof(ROOT / name) == freeze['living_before'][name], 'Living file changed during preservation: ' + name)
        with (ROOT / name).open('wb') as stream:
            stream.write(contents[name])
            stream.flush()
            os.fsync(stream.fileno())
        actual = proof(ROOT / name)
        require(actual['sha256'] == hashlib.sha256(contents[name]).hexdigest(), 'Actual written living bytes differ')
        changes.append({'path': name, 'before': freeze['living_before'][name],
                        'preserved': archives[name], 'after': actual})
    for name, item in evidence.items():
        require(proof(ROOT / name) == item, 'Bound evidence changed during documentation update: ' + name)
    require(proof(Path(__file__)) == source and proof(ROOT / FREEZE) == freeze_proof,
            'Frozen updater/source-freeze changed during execution')
    for name in LIVING:
        require(proof(ROOT / archives[name]['path']) == archives[name], 'Archived original bytes changed')
    receipt = {
        'status': 'COMPLETED_HASH_BOUND_LIVING_REPORT_REFRESH_LOCAL_POSTINVENTORY',
        'author_execution': AUTHOR, 'actual_runner_execution': AUTHOR,
        'start_utc': started, 'finish_utc': now(), 'source': source, 'source_freeze': freeze_proof,
        'approved_records': 3748, 'active_source_families': 696,
        'active_track_records': dict(tracks), 'active_releases': 16, 'historical_releases': 18,
        'existing_release_documented': 'data/RGM-001/releases/v0003',
        'new_release_created_by_updater': False, 'data_approval_granted_by_updater': False,
        'approved_v4_records': 0, 'v4_production_record_count_claimed': False,
        'v4_scope_attestations': freeze['root_v4_scope_attestations'],
        'v4_private_producer_reviewer_internals_inspected': False,
        'living_report_updates': changes, 'all10_originals_preserved_before_any_write': True,
        'historical_report_bytes_retained_below_current_section': True,
        'evidence': evidence, 'latest_verified_backup_commit': COMMIT,
        'exact_backup_claim_files': 2303, 'exact_backup_claim_bytes': 928914914,
        'actual_full_unique_remote_fetches_by_root': 1895,
        'actual_new_gzip_fetches_by_backup_manager': 24,
        'raw_bytes_exactly_recovered': 1188634377,
        'current_update_and_postinventory_proofs_v4_work_local_only': True,
        'no_git_mutation': True, 'no_model_training': True,
        'model_disclosure': 'Inherited same underlying model; separate roles do not establish model diversity.',
        'self_inventory_boundary': 'New updater/archive/freeze/prewrite receipt/living outputs/milestone cannot be claimed inside an earlier frozen spec; include actual final bytes in a later checkpoint.',
    }
    exclusive_json(ROOT / MILESTONE, receipt)
    exclusive_json(ROOT / REPORT, receipt)
    print(json.dumps({'status': receipt['status'], 'updated_reports': len(changes),
                      'approved_records': 3748, 'active_source_families': 696,
                      'milestone': proof(ROOT / MILESTONE), 'report': proof(ROOT / REPORT)}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        path = ROOT / 'backup_logs' / ('RGM_V0003_LIVING_NEW_EXECUTION_FAILURE_' +
                                     datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        exclusive_json(path, {'status': 'FAILED_LIVING_REFRESH_EVIDENCE_PRESERVED',
                              'author_execution': AUTHOR, 'actual_runner_execution': AUTHOR,
                              'at_utc': now(), 'source': proof(Path(__file__)),
                              'error_type': type(error).__name__, 'error': str(error),
                              'original_archive_directory': HISTORY, 'no_git_mutation': True,
                              'grants_data_approval': False})
        raise
