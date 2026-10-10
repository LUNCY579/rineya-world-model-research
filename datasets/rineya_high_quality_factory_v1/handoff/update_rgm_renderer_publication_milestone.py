"""Archive living reports before documenting independently approved RGM publication."""
from pathlib import Path
import collections
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new(path, value):
    with path.open('x') as out:
        json.dump(value, out, sort_keys=True, indent=2)
        out.write('\n')


def main():
    from factory.catalog import catalog
    freeze = json.loads((ROOT / 'handoff/RGM_RENDERER_PUBLICATION_REPORT_SOURCE_FREEZE.json').read_bytes())
    assert sha(Path(__file__)) == freeze['source_sha256']
    release = json.loads((ROOT / 'quality_reports/RGM_RENDERER_R1_RELEASE_EXECUTION.json').read_bytes())
    assert release['status'] == 'COMPLETED_VALIDATED_LOCAL_RELEASE'
    assert release['exact_inputs_unchanged'] and release['historical_release_files_unchanged']
    adjud = json.loads((ROOT / 'reviews/adjudication/planning-gap-v2-renderer-r1.final_execution_receipt.json').read_bytes())
    assert adjud['publication_eligible'] is True and adjud['actual_decision_counts'] == {'ACCEPT': 408}
    backup = json.loads((ROOT / 'backup_logs/RGM_V2_RENDERER_FINAL_REVIEW_JOINED_CUSTODY.json').read_bytes())
    assert backup['status'] == 'BACKED_UP_FROZEN_INVENTORY_AND_REMOTE_LOSSLESS_PROOF_JOINED'
    assert backup['commit_sha'] == 'bc5198e11273cb6ea69659b6d3a8b5dc312b6f00'
    active, report = catalog()
    assert len(active) == 3748 and report['active_source_families'] == 696
    evidence_names = [
        'quality_reports/RGM_RENDERER_R1_RELEASE_EXECUTION.json',
        'quality_reports/RGM_RENDERER_R1_RELEASE_SOURCE_FREEZE.json',
        'reviews/expert/planning-gap-v2-renderer-r1.jsonl',
        'reviews/adversarial/planning-gap-v2-renderer-r1.jsonl',
        'reviews/adjudication/planning-gap-v2-renderer-r1.jsonl',
        'reviews/adjudication/planning-gap-v2-renderer-r1.final_execution_receipt.json',
        'reviews/adjudication/planning-gap-v2-renderer-r1.audit.json',
        'quality_reports/RGM_RENDERER_R1_FROZEN_REVIEW_EVIDENCE_JOIN.json',
        'quality_reports/RGM_V2_RENDERER_R1_AUDIT_EXECUTION.json',
        'manifests/planning-gap-v2-renderer-r1_PRODUCER_FINAL_CUSTODY.json',
        'data/RGM-001/releases/v0003/MANIFEST.json',
        'data/RGM-001/releases/v0003/CHECKSUMS.sha256',
        'manifests/ACTIVE_RELEASE_CATALOG.json',
        'data/RGM-001/quarantine/planning-gap-v2.jsonl',
        'feedback/improvement_requests/RGM-RENDERER-R1-RESIDUALS.json',
        'backup_logs/RGM_V2_RENDERER_FINAL_REVIEW_JOINED_CUSTODY.json',
        'backup_logs/rgm-v2-renderer-review-milestone_RECOVERY_VERIFIED_RECEIPT.json',
        'quality_reports/RGM_V2_RENDERER_REMOTE_LOSSLESS.json',
        'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md',
        'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md',
        'research/curricula/BELIEF_METADATA_V4_ENCODING_ROOT_FREEZE.json',
        'reviews/oracle/RBM-v4-prospective-20261010T071308Z/final_receipt.json',
        'backup_logs/PROVISIONAL_ARCHIVAL_RUNNER_V2_READINESS.json',
    ]
    evidence = {name: {'sha256': sha(ROOT / name), 'bytes': (ROOT / name).stat().st_size} for name in evidence_names}
    section = '''# Current independently approved planning release — 2026-10-10 UTC

The active catalog now contains **3,748 records / 696 source families across all ten tracks**, in 16 active releases and 18 preserved historical releases. Fresh **RGM-001/v0003** contains 408 accepted planning records (340 train / 68 exposed development), with validated SFT, structured and trajectory exports. Its canonical hashes, three actual reviewer receipt joins, exact targets, whole-family controls, source archives and checksums passed release validation. All 198 prior historical release files remain byte-identical. RGM now contributes 840 active records from 228 source families. No model training has occurred.

This publication executes the actual independent adjudicator's decision for `planning-gap-v2-renderer-r1`. Expert, adversarial and adjudication executions each independently returned 408 ACCEPT decisions with no unresolved high or critical issue. Their 1,224 unique receipts bind the exact same candidate hashes. Each actor froze all fresh input predictions before fresh canonical access and then independently checked targets and metadata. The adjudicator separately verified 510 factual witnesses, 192 renderer declarations, 96 semantic control pairs and all 48 complete families. Prior original gold, reused solvers, public-design/adjudicator overlap, common underlying model and root-guided revision are disclosed. These executions do not establish unseen blind semantics or independent model error distributions.

The fresh candidate uses producer `rgm-planning-gap-2.0.5`, with frozen source and byte-exact reproduction across seeds 17 and 971. It reuses **204 semantic tasks / 48 source families** from the quarantined original V2, adding **zero new semantic tasks or families relative to that original cohort**. Only 22 visible inputs change goal-map key order. All semantic tasks, targets, splits, contexts, difficulty and skill tags remain exact. The unchanged V2 whole-cohort gate now passes; no historical specification or checker was relaxed. Original planning V1 (384 records) and V2 (408 records) remain quarantined with their original candidates, first judgments, failure sources and execution receipts preserved.

Measured residual limitations remain: repeated block arrangements, conditional label lookup associations, a tiny exposed development set, class imbalance, shared mechanism grammars and structural difficulty labels uncalibrated to model performance. The revision's difficulty distribution remains 32 level 2, 234 level 3 and 142 level 4 records. `RGM-RENDERER-R1-RESIDUALS.json` records the future diversification work. This release provides exact development supervision; transfer, learning benefit and sealed OOD efficacy are untested.

The latest fully verified private GitHub snapshot is **bc5198e11273cb6ea69659b6d3a8b5dc312b6f00**: 2,095 frozen files / 886,553,408 bytes, verified through 1,740 actual remote unique blob fetches and 24 separate new actual remote gzip decodes recovering 1,188,634,377 raw bytes. Joined custody verifies private ADMIN, exact parent, historical bytes, both clean checkouts and all six protected original Git byte/existence baselines. **The new v0003 release, completed fresh reviews/adjudication, post-snapshot proof records and this report update are local pending the next exact checkpoint.** A commit or push alone cannot change that boundary. The next frozen inventory must include all completed renderer-r1 artifacts and preserve every prior tracked path.

Belief V1/V2/V3 remain quarantined. Prospective V4 public encoding and a separate addendum are frozen before official implementation or production. They require explicit length-bound ordered pair claims, owner-coordinate matching, exact candidate-byte custody and unique named-entry resolution while permitting nonmember family owners and IDs shared across distinct claim names. A formal prospective preparation completed 15 handmade tests plus actual toy-byte custody joins. A separate official checker and expert preparation are underway; no V4 production candidate has been generated, validated or approved. The intended metadata repair would reuse 3,456 presentations, with no new semantic diversity. Original V3 evidence is preserved.

Next: back up this validated release and completed reviews on the dedicated private branch, fetch and verify the actual remote bytes, and independently decode all oversized evidence. Continue the diagnosed belief interface repair using a separately frozen public-only checker, fresh generator version, exact reproduction, fresh input-first reviews and independent adjudication. Dataset-only CPU execution continues; model weights, architectures, protected research and sealed evaluation instruments remain outside this work.

---

'''
    living = ['PROJECT_STATUS.md', 'RESEARCH_GAPS.md', 'DATASET_QUALITY_REPORT.md',
              'GENERATOR_IMPROVEMENTS.md', 'CURRICULUM_COVERAGE.md', 'TRAINING_READINESS.md',
              'GITHUB_BACKUP_LOG.md', 'NEXT_SESSION_HANDOFF.md', 'handoff/NEXT_SESSION_HANDOFF.md',
              'research/role_registry.json']
    changes = []
    for name in living:
        path = ROOT / name
        old = path.read_bytes()
        digest = hashlib.sha256(old).hexdigest()
        archive = ROOT / 'handoff/report_history/rgm-renderer-r1-publication-living' / (digest + '-' + path.name)
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            assert archive.read_bytes() == old
        else:
            with archive.open('xb') as out:
                out.write(old)
        if name.endswith('.json'):
            registry = json.loads(old)
            registry['rgm_planning_wave'].update({
                'stage': 'RENDERER_R1_ALL408_INDEPENDENT_ACCEPT_VALIDATED_LOCAL_V0003',
                'approved_records': 408,
                'renderer_revision_fresh_review_status': 'ACTUAL_THREE_ROLES_ALL408_ACCEPT',
                'renderer_revision_adjudication': 'reviews/adjudication/planning-gap-v2-renderer-r1.final_execution_receipt.json',
                'renderer_revision_release': 'data/RGM-001/releases/v0003',
                'renderer_revision_release_validated': True,
                'renderer_revision_independent_review_receipts': 1224,
                'latest_verified_backup_commit': backup['commit_sha'],
                'backup': 'bc5198 exact 2095-file snapshot fully verified; new v0003 and completed reviews remain local until next exact checkpoint',
                'newer_milestone_artifacts_local_only': True,
            })
            registry['belief_evidence_wave'].update({
                'stage': 'V3_WHOLE_QUARANTINE_V4_PUBLIC_ENCODING_FROZEN_OFFICIAL_CHECKER_PREPARATION',
                'v4_public_specification': 'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md',
                'v4_prospective_encoding_addendum': 'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md',
                'v4_root_public_encoding_freeze': 'research/curricula/BELIEF_METADATA_V4_ENCODING_ROOT_FREEZE.json',
                'v4_actual_formal_preparation': 'reviews/oracle/RBM-v4-prospective-20261010T071308Z/final_receipt.json',
                'v4_production_records': 0,
                'v4_data_approval': False,
            })
            content = (json.dumps(registry, sort_keys=True, indent=2) + '\n').encode()
        else:
            content = section.encode() + old
        path.write_bytes(content)
        changes.append({'path': name, 'before_sha256': digest, 'preserved_path': str(archive.relative_to(ROOT)), 'after_sha256': sha(path)})
    assert all(sha(ROOT / name) == item['sha256'] for name, item in evidence.items())
    receipt = {'runner_execution': '/root', 'actual_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'source_sha256': sha(Path(__file__)), 'approved_records': len(active), 'active_source_families': 696,
               'active_track_records': dict(collections.Counter(r['track_id'] for r in active)),
               'created_release': 'data/RGM-001/releases/v0003', 'independently_approved_new_records': 408,
               'new_semantic_tasks_relative_to_original_quarantine': 0, 'new_source_families_relative_to_original_quarantine': 0,
               'reviewer_disagreement': 0, 'independent_review_receipts': 1224,
               'evidence': evidence, 'living_report_updates': changes,
               'latest_verified_backup_commit': backup['commit_sha'], 'new_release_and_completed_reviews_local_only': True,
               'belief_v4_production_records': 0, 'no_model_training': True}
    new(ROOT / 'handoff/RGM_RENDERER_PUBLICATION_MILESTONE.json', receipt)
    new(ROOT / 'quality_reports/RGM_RENDERER_PUBLICATION_LIVING_REPORT_UPDATE.json', receipt)
    print(json.dumps({'receipt_sha256': sha(ROOT / 'handoff/RGM_RENDERER_PUBLICATION_MILESTONE.json'), 'updated_reports': len(changes), 'active_records': len(active), 'active_source_families': 696}))


if __name__ == '__main__':
    main()
