"""Root final-stage documentation custody; requires actual frozen executions."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from factory.catalog import catalog
from factory.common import ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2) + '\n')


def main():
    output = ROOT / 'handoff/RGM_V1_FINAL.json'
    assert not output.exists(), 'Final stage is immutable; use a new version'
    suite_path = ROOT / 'quality_reports/RGM_V1_CASE_BOUNDARY_FULL_SUITE_EXECUTION.json'
    suite = json.loads(suite_path.read_text())
    assert suite['exit_code'] == 0 and suite['bound_sources_unchanged']
    assert any('Ran 324 tests' in line for line in suite['actual_output_tail'])
    for relative, expected in suite['source_sha256_before'].items():
        assert sha(ROOT / relative) == expected, relative
    red_path = ROOT / 'reviews/adversarial/planning-publication-code-v3.json'
    red = json.loads(red_path.read_text())
    assert red['decision'] == 'ACCEPT_CODE_WITH_LIMITS'
    assert sha(red_path) == '9b15970d045f38a5654c2357fa1c1ec1b6f945436a6dd6212b83069fcf6134a5'
    for relative, expected in red['code_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    quarantine = json.loads((ROOT / 'quality_reports/RGM_PLANNING_V1_QUARANTINE_EXECUTION.json').read_text())
    assert quarantine['pass'] and quarantine['whole_quarantine_records'] == 384
    assert sha(ROOT / quarantine['quarantine_path']) == quarantine['quarantine_sha256']
    rows, active = catalog()
    assert len(rows) == 3340 and active['active_source_families'] == 648
    actual_utc = datetime.now(timezone.utc).isoformat()
    evidence_paths = [
        'reviews/adjudication/planning-gap-v1.final_execution_receipt.json',
        'reviews/adjudication/planning-gap-v1.jsonl',
        'data/RGM-001/candidate_renderings/planning-gap-v1.jsonl',
        'data/RGM-001/quarantine/planning-gap-v1.jsonl',
        'quality_reports/RGM_V1_CASE_BOUNDARY_FULL_SUITE_EXECUTION.json',
        'quality_reports/RGM_V1_CASE_BOUNDARY_FOCUSED_EXECUTION.json',
        'reviews/adversarial/planning-publication-code-v1.json',
        'reviews/adversarial/planning-publication-code-v2.json',
        'reviews/adversarial/planning-publication-code-v3.json',
        'audits/execution/planning-publication-code-adversarial-v3.json',
        'quality_reports/RGM_V1_FINAL_STAGE_BOUNDARY_PREFLIGHT.json',
        'backup_logs/belief-v3-final-review_RECOVERY_VERIFIED_RECEIPT.json',
        'quality_reports/BELIEF_V3_FINAL_REVIEW_REMOTE_LOSSLESS.json',
        'research/curricula/RGM-PLANNING-GAP-V2-ROOT_FREEZE.json',
        'research/curricula/RGM-PLANNING-GAP-V2-RENDERER-ROOT_FREEZE.json',
        'research/curricula/RGM-PLANNING-GAP-V2-PRODUCER-ROOT_DISPATCH.json',
        'reviews/oracle/RGM-planning-v2-final-freeze-20261010T053030Z.json',
        'backup_logs/RGM_V1_NEXT_CHECKPOINT_READINESS_20261010T053949Z.json',
        'factory/pipeline.py', 'factory/planning_controls.py',
        'tests/test_planning_review_corrections.py',
    ]
    evidence = {name: sha(ROOT / name) for name in evidence_paths}
    section = '''

## RGM v1 final quarantine and publication correction — 2026-10-10 UTC

This completed stage supersedes the earlier in-progress RGM summaries. The approved corpus remains **3,340 records / 648 source families across all ten tracks**. Actual independent adjudication assigns **all 384 planning-gap-v1 rows QUARANTINE**, with zero publication-eligible rows. Root preserved the whole exact-record quarantine and created zero releases. Both reviewers' original first/final 384 ACCEPT judgments remain unchanged: every target, all 434 factual witnesses and 84 supplied controls passed their bounded calculations. The publication hold concerns an undeclared V1 witness value enum and initial-precondition priority in the frozen checker, not an established false target or factual witness. All original candidate, producer, checker, failed audits and reviewer bytes remain preserved. V2 rules do not apply retrospectively.

Independent code review found and retained concrete publication weaknesses: persistent planning routing, exact role actors, known metadata labels, public-name false rejection and private outer-key casing. New root corrections retain routing after changed version/batch metadata, require the exact expert/red/adjudicator role actors, reject known and actual additional private row metadata in the contract lead or case-normalized task outer keys, and allow legitimate public names inside typed payload maps. Earlier REVISE_CODE receipts remain immutable. Latest actual adversarial replay is **ACCEPT_CODE_WITH_LIMITS: all 732 reused checks pass** (687 schema/hash/prompt fixtures, 13 actor joins, 32 routing checks), including all 12 previously accepted casing variants. This guided, same-model fixture replay is scoped code evidence; it provides no data approval or universal encoded-alias/privacy guarantee.

The final source-bound production suite passes **324 actual tests**, and 21 focused correction tests pass. All bound sources stayed unchanged during execution. Earlier 316/320/321/323 passing suites, independent later counterexamples, every correction source and the mistaken nonexistent-test-module invocation remain retained as historical evidence. The original 65 read-only reference files and original research checkout/main remain unchanged. See `handoff/RGM_V1_FINAL.json` and `quality_reports/RGM_V1_CASE_BOUNDARY_FULL_SUITE_EXECUTION.json` for exact custody.

Prospective planning-gap-v2 public rules and renderer addendum are frozen before generation. Its independent public-only checker passed 25 tests with exact full hand-probe replay under two hash seeds. A real fresh producer dispatch has now started for **408 planned rows / 204 tasks / the same 48 revised source families** (340 train, 68 exposed development; zero family growth relative to V1). The public design balances feasible/infeasible A fuel cases in development, selective source/destination C1 negatives, isolated C4 no-ops, four typed clearance witness fields and 192 required per-renderer control declarations. Generation and new input-derived reviews remain pending; these plans/checker tests approve no future candidate. Expert public-only review preparation has also been dispatched. Actual separate roles use the same underlying model, and public-design/later-adjudicator and guided revision overlap remain disclosed. Belief V3 remains wholly quarantined; V4 metadata clarification is prospective only.

The last remotely verified checkpoint is **b46d96a5db0f1cc94e894f15d61942af49e61819**, with actual GitHub bytes verified for 1,485 paths / 818,645,945 bytes / 1,219 unique blobs. All 24 remote gzip artifacts decode exactly to 1,188,634,377 preserved raw bytes. Original failed backup receipts remain FAILED; the successful recovery receipt and separate actual-remote decode proof are authoritative. The dedicated backup checkout has been safely restored to full dataset scope and remains clean. This final RGM evidence, current corrections, recovery receipts and documentation remain local until the next verified archival checkpoint. The narrow prepared wrapper retains all prior data, completed V1 artifacts, every release and public V2 documents, explicitly deferring only named unfinished V2 implementation/candidate/review paths. A specification and its resulting post-push receipt cannot include their own final bytes and are carried forward separately. Storage checks currently permit the checkpoint; no unrelated project was cleaned.

Next: freeze the final V1 archival inventory, copy/commit/non-force-push the dedicated branch and verify actual remote hashes. Keep included source/docs immutable during verification while new deferred V2 paths continue. Then integrate the frozen V2 checker with exact public/prior/parent bindings, execute on the fresh candidate, freeze each reviewer's new input-derived predictions and first judgments before gold/private comparisons, obtain independent adjudication and release only if all quality gates pass. No model training, sealed evaluation, architecture/weight modification or efficacy claim is authorized or performed.
'''
    living = ['PROJECT_STATUS.md', 'RESEARCH_GAPS.md', 'DATASET_QUALITY_REPORT.md',
              'GENERATOR_IMPROVEMENTS.md', 'CURRICULUM_COVERAGE.md', 'TRAINING_READINESS.md',
              'GITHUB_BACKUP_LOG.md', 'NEXT_SESSION_HANDOFF.md', 'handoff/NEXT_SESSION_HANDOFF.md',
              'research/role_registry.json']
    changes = []
    for name in living:
        path = ROOT / name
        prior = path.read_bytes()
        old_sha = hashlib.sha256(prior).hexdigest()
        archive = ROOT / 'handoff/report_history/rgm-v1-final-stage' / (old_sha + '-' + path.name)
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            assert archive.read_bytes() == prior
        else:
            archive.write_bytes(prior)
        if name == 'research/role_registry.json':
            registry = json.loads(prior)
            wave = registry['rgm_planning_wave']
            wave.update({'stage': 'V1_ALL384_QUARANTINE_324_TESTS_PASS_CODE732_SCOPED_PASS_V2_PRODUCER_RUNNING',
                         'approved_records': 0, 'quarantine_records': 384,
                         'adjudication': 'reviews/adjudication/planning-gap-v1.final_execution_receipt.json',
                         'current_tests': {'count': 324, 'execution': str(suite_path.relative_to(ROOT))},
                         'current_code_review': str(red_path.relative_to(ROOT)),
                         'code_review_mode': 'Root-guided same-model reuse of independently frozen732fixtures; earlierfailedcodejudgments retained; no corpus or data approval',
                         'v2_stage': 'ACTUAL_PRODUCER_RUNNING_PUBLIC_ONLY_EXPERT_PREPARATION_DISPATCHED',
                         'v2_planned_records': 408, 'v2_planned_tasks': 204,
                         'v2_revised_families': 48, 'v2_family_growth_relative_v1': 0,
                         'v2_public_freeze': 'research/curricula/RGM-PLANNING-GAP-V2-RENDERER-ROOT_FREEZE.json',
                         'v2_formal_freeze': 'reviews/oracle/RGM-planning-v2-final-freeze-20261010T053030Z.json',
                         'backup': 'b46d96a5db0f1cc94e894f15d61942af49e61819 verified frozen belief scope; completed RGM V1 final evidence awaits verified checkpoint'})
            new = (json.dumps(registry, indent=2, sort_keys=True) + '\n').encode()
        else:
            new = prior + section.encode()
        path.write_bytes(new)
        changes.append({'path': name, 'before_sha256': old_sha, 'preserved_prior_path': str(archive.relative_to(ROOT)),
                        'after_sha256': sha(path), 'after_bytes': len(new)})
    report = {'actual_execution': '/root', 'actual_utc': actual_utc,
              'source_sha256': sha(Path(__file__)), 'active_approved_records': 3340, 'active_source_families': 648,
              'planning_v1_decision': {'QUARANTINE': 384}, 'new_releases': 0,
              'production_tests': 324, 'focused_tests': 21,
              'adversarial_code_decision': 'ACCEPT_CODE_WITH_LIMITS', 'actual_reused_code_checks': 732,
              'evidence_sha256': evidence, 'living_document_changes': changes,
              'v2_status': 'Actualproducer/expertpublic-onlypreparationdispatched;408/204/48 planned;0newfamilies; noapproval',
              'backup': {'last_verified_commit': 'b46d96a5db0f1cc94e894f15d61942af49e61819',
                         'current_rgm_final_checkpoint': 'LOCAL_PENDING_FROZEN_INVENTORY_AND_ACTUAL_REMOTE_VERIFY'},
              'next_actions': ['RootfrozenV1archivalcheckpointwithnamedV2deferral', 'ActualremoteGitHubbyteandgzipverification',
                               'V2integrationandfreshcandidateaudit', 'Separatefreshinputprediction-before-goldreviews/adjudication',
                               'BeliefV4public-onlycheckerandrevisedmetadataafterRGM'],
              'scope': 'Actual stage handoff and history preservation; no new data/model efficacy approval'}
    write_new(output, report)
    write_new(ROOT / 'quality_reports/RGM_V1_FINAL_LIVING_REPORT_UPDATE.json', report)
    write_new(ROOT / 'feedback/resolved_issues/PIPELINE-PRIVACY-002.json',
              {'actual_resolver_execution': '/root', 'at_utc': actual_utc, 'ticket': 'PIPELINE-PRIVACY-002',
               'status': 'OBSERVED_KNOWN_METADATA_BOUNDARY_FAILURES_RESOLVED_WITH_SCOPED_REPLAY',
               'actual_prior_code_failures': ['reviews/adversarial/planning-publication-code-v1.json',
                                            'reviews/adversarial/planning-publication-code-v2.json'],
               'current_code_acceptance': str(red_path.relative_to(ROOT)), 'full_suite': str(suite_path.relative_to(ROOT)),
               'current_source_sha256': sha(ROOT / 'factory/pipeline.py'),
               'limitations': red['publication_scope'], 'dataset_or_release_approval': False})
    print(json.dumps({'updated_living_documents': len(changes), 'handoff': str(output.relative_to(ROOT)),
                      'handoff_sha256': sha(output), 'actual_production_tests': 324, 'active': 3340}))


if __name__ == '__main__':
    main()
