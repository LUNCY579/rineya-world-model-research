"""Preserve ten living byte streams before documenting independently accepted V4."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import os
import traceback

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'BELIEF_V4_LOCAL_RELEASE_LIVING_ATTEMPT01'
FREEZE = ROOT / 'handoff' / (PREFIX + '_SOURCE_FREEZE.json')
HISTORY = ROOT / 'handoff/report_history/belief-v4-local-release-living-attempt01'
LIVING = ['PROJECT_STATUS.md', 'RESEARCH_GAPS.md', 'DATASET_QUALITY_REPORT.md',
          'GENERATOR_IMPROVEMENTS.md', 'CURRICULUM_COVERAGE.md', 'TRAINING_READINESS.md',
          'GITHUB_BACKUP_LOG.md', 'NEXT_SESSION_HANDOFF.md', 'handoff/NEXT_SESSION_HANDOFF.md',
          'research/role_registry.json']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def proof(name):
    path = ROOT / name
    assert isinstance(name, str) and not Path(name).is_absolute() and '..' not in Path(name).parts
    assert path.resolve().is_relative_to(ROOT) and path.is_file() and not path.is_symlink()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(65536):
            digest.update(block)
    return {'path': name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def main():
    started = now()
    frozen = json.loads(FREEZE.read_bytes())
    source = proof(Path(__file__).relative_to(ROOT).as_posix())
    assert frozen['source'] == source and frozen['runner_execution'] == '/root'
    assert frozen['frozen_before_first_import_execution'] is True
    assert sorted(frozen['living_before']) == sorted(LIVING)
    for binding in frozen['evidence_bindings']:
        assert proof(binding['path']) == binding, binding['path']
    for name, binding in frozen['all222_release_files'].items():
        assert proof(name) == binding, name
    assert len(frozen['all222_release_files']) == 222
    release = json.loads((ROOT / 'quality_reports/BELIEF_V4_RELEASE_ATTEMPT01_EXECUTION.json').read_bytes())
    assert release['status'] == 'COMPLETED_VALIDATED_LOCAL_RELEASE'
    assert release['exact_inputs_unchanged'] and release['historical_release_files_unchanged']
    assert release['release_code_set_unchanged'] and release['result']['validation']['pass']
    cat = json.loads((ROOT / 'manifests/ACTIVE_RELEASE_CATALOG.json').read_bytes())
    assert cat['active_records'] == 7204 and cat['active_source_families'] == 700
    tracks = collections.Counter()
    for item in cat['releases']:
        tracks[item['track']] += item['active_records']
    assert len(tracks) == 10 and sum(tracks.values()) == 7204 and tracks['RBM-001'] == 3480
    active = sum(item['active_records'] > 0 for item in cat['releases'])
    assert len(cat['releases']) == 19 and active == 17
    backup = json.loads((ROOT / 'backup_logs/BELIEF_V4_R2_JOINED_CUSTODY.json').read_bytes())
    assert backup['status'] == 'BACKED_UP_FROZEN_INVENTORY_AND_REMOTE_LOSSLESS_PROOF_JOINED'
    assert backup['commit_sha'] == '4d0b5eff1af3184fedf7fa25ada729a6cc909268'
    assert backup['frozen_files'] == 2685 and backup['actual_full_unique_remote_fetches'] == 2153
    assert backup['actual_new_gzip_payload_fetches'] == 26
    assert not HISTORY.exists()
    milestone = ROOT / 'handoff' / (PREFIX + '_MILESTONE.json')
    report = ROOT / 'quality_reports' / (PREFIX + '_EXECUTION.json')
    assert not milestone.exists() and not report.exists()
    old = {}
    for name in LIVING:
        assert proof(name) == frozen['living_before'][name], name
        old[name] = (ROOT / name).read_bytes()
    HISTORY.mkdir()
    archives = {}
    for name in LIVING:
        path = HISTORY / (frozen['living_before'][name]['sha256'] + '-' + name.replace('/', '__'))
        with path.open('xb') as output:
            output.write(old[name])
            output.flush()
            os.fsync(output.fileno())
        binding = proof(path.relative_to(ROOT).as_posix())
        assert binding['sha256'] == frozen['living_before'][name]['sha256']
        assert binding['bytes'] == frozen['living_before'][name]['bytes']
        archives[name] = binding
    new(ROOT / 'backup_logs' / (PREFIX + '_ALL10_PREWRITE_PRESERVATION.json'),
        {'at_utc': now(), 'status': 'ALL10_PRIOR_LIVING_BYTES_ARCHIVED_BEFORE_ANY_WRITE',
         'actual_runner_execution': '/root', 'source': source, 'originals': frozen['living_before'],
         'archives': archives, 'all_originals_byte_exact': True, 'data_approval': False})
    text = f'''# Current independently accepted V4 release and exact backup scope — {started}

The approved active catalog contains **7,204 records / 700 source families across all ten tracks**, with 17 active and 19 preserved historical releases. **RBM-001/v0002** contains 3,456 independently accepted V4 records (1,728 train / 1,728 exposed development), with four reused source families. It corrects supplied control evidence for the preserved V3 quarantine tasks, adding **zero semantic tasks, source families or student inputs relative to that V3 cohort**. V1/V2/V3 quarantine and every first failure, first review, corrected review and source version remain preserved. These 3,456 records are genuine completed review acceptance, not evidence of new semantic coverage or model efficacy.

Actual expert `/root/expert_reviewer`, adversarial `/root/adversarial_auditor` and adjudicator `/root/research_architect` each completed 3,456 exact canonical judgments. All 10,368 distinct receipt joins passed; actual independent adjudication authorized publication. Root executed that decision and supplies no data-quality vote. The existing source-bound 445-test CPU suite passed before publication, and mandatory whole-cohort audit, targets, control evidence, review joins, exports and checksums passed actual release validation. All 210 earlier release files stayed byte exact; there are now 222 immutable release files. See `quality_reports/BELIEF_V4_RELEASE_ATTEMPT01_EXECUTION.json` and the independent adjudication FINAL. Exposed development, prior V3 gold exposure, own-solver reuse, common model/public guidance, public-design/adjudicator overlap, four finite reused families, length associations, 75% nonempty belief sets and 87.5% false query-usefulness labels remain limitations. No model training, learning benefit or sealed transfer is claimed.

The latest fully byte-verified private GitHub checkpoint is **4d0b5eff1af3184fedf7fa25ada729a6cc909268**, sole parent `82787689fef4e50f7a35cfee505c290b2f6a6d78`: **2,685 frozen files / 1,080,978,676 bytes**. Root fetched and verified **2,153 actual unique GitHub blobs**. Backup manager separately fetched **26 actual remote gzip payloads**, decoded and compared **1,353,229,114 preserved raw bytes**. Joined custody `backup_logs/BELIEF_V4_R2_JOINED_CUSTODY.json` binds actual full/gzip proof ledgers, all prior 2,303 paths, 210 historical release files, private ADMIN, clean checkouts and six protected original Git guards. The original strict diff check exited 2 with exactly five preserved official-source whitespace warnings; its narrowly bound exception and restricted strict PASS are recorded honestly. Backup proves preservation and grants no data approval.

**The new RBM v0002 release and post-cut final expert/adjudication, publication, backup-proof, RSM and current living-update bytes are LOCAL pending their own fresh exact remote-byte checkpoint.** They are not retroactively inside the frozen 2,685-file inventory. Current receipts cannot self-inventory their final bytes. Later proof statuses never overwrite earlier FAILED or PUSHED_PENDING historical receipts.

RSM prospective option resumption has public task/curriculum/config, independently tested task-solver feasibility and target-free 24-row legacy contrast. Existing 12 legacy lineage families collapse to eight structures in the published normalization; legacy tasks already contain nested calls. Handwritten prospective examples establish no production novelty. The new r2 public metadata amendment forbids circular canonical member hashes, retains externally verified manifest hashes and binds six public sources. Public-only review found no new blocking interface issue. The isolated whole-corpus checker remains under development at this reporting cut; no RSM producer or new RSM release is authorized or counted. Actual generic-validator integration, twelve actual producer graph comparisons in all three views, all pair controls, fresh independent input-first predictions/reviews and adjudication remain later gates.

Active track records: {json.dumps(dict(sorted(tracks.items())), sort_keys=True)}. Next preserve and losslessly package every oversized approved/raw artifact outside immutable release directories, refresh the complete archival inventory from parent4d0b, handle only exact newly observed whitespace fixtures with a separately frozen exception, then perform actual full GitHub byte and gzip proofs. Continue CPU dataset work on measured gaps while preserving all ten tracks, read-only research, model weights/architecture and sealed evaluations.

---

## Preserved historical report text

Earlier current/pending statements below are immutable historical snapshots superseded by this current section. Their original exact bytes were archived before this update.

'''
    registry = json.loads(old['research/role_registry.json'])
    registry['belief_evidence_wave'].update({
        'stage': 'V4_ALL3456_INDEPENDENT_ACCEPT_VALIDATED_LOCAL_V0002_REMOTE_PENDING',
        'v4_data_approval': True, 'v4_approved_records': 3456, 'v4_production_records': 3456,
        'v4_release': 'data/RBM-001/releases/v0002', 'v4_release_remotely_backed_up': False,
        'v4_actual_independent_receipt_joins': 10368, 'v4_full_cpu_suite_tests': 445,
        'v4_added_semantic_tasks_relative_v3_quarantine': 0,
        'v4_added_source_families_relative_v3_quarantine': 0,
        'v4_producer_status': 'ACTUAL_COMPLETED_FINAL_PRESERVED',
        'v4_formal_factory_integration_status': 'ACTUAL_COMPLETED_VALIDATED_AND_USED_FOR_RELEASE',
        'v4_data_approval_actor': '/root/research_architect', 'v4_publication_runner': '/root'})
    registry['current_backup_custody'] = {
        'commit_sha': backup['commit_sha'], 'sole_parent_sha': backup['sole_parent_sha'],
        'frozen_files': 2685, 'frozen_bytes': 1080978676,
        'actual_full_unique_remote_fetches': 2153, 'actual_new_gzip_fetches': 26,
        'raw_bytes_exactly_recovered': 1353229114,
        'joined_custody': proof('backup_logs/BELIEF_V4_R2_JOINED_CUSTODY.json'),
        'new_v4_release_and_postcut_artifacts_local_only': True, 'backup_grants_data_approval': False}
    registry['rsm_option_resumption_prospective_wave'] = {
        'stage': 'R2_PUBLIC_INTERFACE_REVIEWED_ISOLATED_CORPUS_CHECKER_IN_DEVELOPMENT',
        'production_authorized': False, 'new_approved_records': 0,
        'proposed_source_families': 12, 'proposed_typed_tasks': 108, 'proposed_views': 216,
        'actual_generated_counts': None, 'legacy_lineage_families': 12, 'legacy_structural_signatures': 8,
        'operative_metadata_amendment': 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md'}
    registry['living_report_refresh_execution'] = {
        'actual_author': '/root', 'actual_runner': '/root', 'at_utc': started,
        'source': source, 'all10_prior_byte_streams_archived_before_any_write': True,
        'new_living_bytes_local_postinventory': True, 'data_approval_by_updater': False}
    changes = []
    for name in LIVING:
        assert proof(name) == frozen['living_before'][name]
        content = (json.dumps(registry, indent=2, sort_keys=True) + '\n').encode() if name.endswith('.json') else text.encode() + old[name]
        with (ROOT / name).open('wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        after = proof(name)
        assert after['sha256'] == hashlib.sha256(content).hexdigest()
        changes.append({'path': name, 'before': frozen['living_before'][name],
                        'archive': archives[name], 'after': after})
    for binding in frozen['evidence_bindings']:
        assert proof(binding['path']) == binding
    for name, binding in frozen['all222_release_files'].items():
        assert proof(name) == binding
    for binding in archives.values():
        assert proof(binding['path']) == binding
    assert proof(source['path']) == source
    result = {'status': 'COMPLETED_ALL10_PRESERVED_LOCAL_RELEASE_LIVING_UPDATE',
              'actual_runner_execution': '/root', 'started_at_utc': started, 'finished_at_utc': now(),
              'source': source, 'source_freeze': proof(FREEZE.relative_to(ROOT).as_posix()),
              'approved_active_records': 7204, 'active_source_families': 700,
              'active_track_records': dict(tracks), 'historical_releases': 19, 'active_releases': 17,
              'all10_old_living_bytes_preserved_before_write': True, 'living_changes': changes,
              'all222_release_files_unchanged': True, 'latest_verified_backup_commit': backup['commit_sha'],
              'new_release_and_current_update_remotely_verified': False, 'data_approval_by_updater': False}
    new(milestone, result)
    new(report, result)
    print(json.dumps({'status': result['status'], 'reports': len(changes), 'approved_records': 7204}))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        path = ROOT / 'quality_reports' / (PREFIX + '_FAILED_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        new(path, {'status': 'FAILED_LIVING_UPDATE_PRESERVED', 'at_utc': now(),
                   'actual_runner_execution': '/root', 'error_type': type(error).__name__,
                   'error': str(error), 'traceback': traceback.format_exc(), 'data_approval': False})
        raise
