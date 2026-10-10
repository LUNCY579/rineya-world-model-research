"""HAND-only core integration tests; execute only AFTER installation/freeze.

Reference calls are explicitly mocked to own427 for wiring, not interpreted as
independent producer-reference agreement. Synthetic review receipts below are
test fixtures, never other agents' actual judgements or production approvals.
All temporary candidate/audit/release/archive writes stay below a NEW hand root.
The original immutable HAND fixture lacks the shared schema's sealed-evaluation
flag. This revision derives a distinct fixture by adding strict False only,
then recomputes record/candidate/ordered external pair hashes. No task, target,
control declaration or schema/gate changes are included.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from factory import common, pipeline, exports, export_validation, rsm_controls as R
from oracles.independent import solve as independent_solve

ROOT=Path(__file__).resolve().parents[1]
HAND='reviews/oracle/RSM-option-resumption-corpus-20261010T0956Z/'
CANDIDATE=HAND+'actual-hand-candidate-attempt04.jsonl'
MANIFEST=HAND+'hand-control-manifest.json'
REPAIR='quality_reports/RSM_CORE_HAND_FIXTURE_REPAIR_ATTEMPT01/'
DERIVED_CANDIDATE=REPAIR+'HAND_SCHEMA_COMPLETE_PENDING_ATTEMPT01.jsonl'
DERIVED_MANIFEST=REPAIR+'HAND_SCHEMA_COMPLETE_CONTROL_MANIFEST_ATTEMPT01.json'
RESULTS={'scope':'Schema-complete local derivation of immutable HAND216 only; reference mocked, synthetic receipts, no production approval.','cases':[]}


def rebind_manifest(rows, raw, manifest, codec):
    """Bind only candidate SHA and actual ordered member hashes; keep controls."""
    result=deepcopy(manifest)
    result['candidate_sha256']=sha256(raw).hexdigest()
    by_id={row['sample_id']:row for row in rows}
    if len(by_id)!=len(rows):raise ValueError('fixture duplicate sample IDs')
    for entry in result['control_pairs']:
        entry['candidate_sha256']=result['candidate_sha256']
        entry['pair_record_hashes']=[by_id[sid]['record_hash'] for sid in entry['pair_sample_ids']]
    return result


def schema_complete_fixture(candidate_raw, manifest_raw, codec):
    """Local HAND provenance repair; no runtime/schema mocking or semantics edits."""
    originals=[codec.strict_json(line) for line in candidate_raw.splitlines()]
    rows=deepcopy(originals)
    for row in rows:
        provenance=row['source_provenance']
        if type(provenance) is not dict:raise ValueError('fixture provenance object required')
        if 'sealed_evaluation_used' in provenance and provenance['sealed_evaluation_used'] is not False:
            raise ValueError('fixture existing sealed flag must be strict False')
        provenance['sealed_evaluation_used']=False
        row['record_hash']=codec.record_hash(row)
    raw=(''.join(codec.canonical(row)+'\n' for row in rows)).encode('utf-8')
    manifest=rebind_manifest(rows,raw,codec.strict_json(manifest_raw),codec)
    manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
    return rows,raw,manifest_bytes

class RSMCoreIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_raw=(ROOT/CANDIDATE).read_bytes()
        cls.original_manifest=(ROOT/MANIFEST).read_bytes()
        assert sha256(cls.original_raw).hexdigest()=='742364bd1173a76d911a8b43bff70fb30541a46ce5ec637e2ee1477ff34a0bd5'
        assert sha256(cls.original_manifest).hexdigest()=='bb01ac3d44a62ebdbb95750b9d5ad6721c020c5330cebc89e69252c8ad93817f'
        cls.C=R.checker()
        cls.original_rows=[cls.C.strict_json(line) for line in cls.original_raw.splitlines()]
        cls.rows,cls.raw,cls.manifest=schema_complete_fixture(cls.original_raw,cls.original_manifest,cls.C)
        assert cls.raw==(ROOT/DERIVED_CANDIDATE).read_bytes()
        assert cls.manifest==(ROOT/DERIVED_MANIFEST).read_bytes()
        assert sha256(cls.raw).hexdigest()=='db47dffc9607f734d43bc4b703a8ee1b9e1682ac7c8ae33c20cdca54248df6df'
        assert sha256(cls.manifest).hexdigest()=='2d13879e88cfe322f97e8319fe0777a1cb12298265aff680e6c20181f28df831'
        RESULTS['fixture_identity']={'original_candidate_sha256':sha256(cls.original_raw).hexdigest(),
            'original_manifest_sha256':sha256(cls.original_manifest).hexdigest(),
            'derived_candidate_sha256':sha256(cls.raw).hexdigest(),'derived_manifest_sha256':sha256(cls.manifest).hexdigest(),
            'records':len(cls.rows),'public_schema_changed':False,'semantic_or_control_changes':False}

    @contextmanager
    def sandbox(self):
        execution=ROOT/REPAIR/'FUTURE_HAND_SANDBOXES'
        execution.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='case-',dir=execution) as temporary:
            root=Path(temporary)
            paths=set(R.SOURCES)|{R.LEGACY_PATH,R.ADAPTER_PATH,'factory/pipeline.py','factory/exports.py','factory/export_validation.py','oracles/independent.py'}
            for path in paths:
                target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/path).read_bytes())
            candidate=root/f'data/RSM-001/candidate_renderings/{R.BATCH}.jsonl'
            candidate.parent.mkdir(parents=True);candidate.write_bytes(self.raw)
            manifest=root/MANIFEST;manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_bytes(self.manifest)
            with ExitStack() as stack:
                for module in (common,pipeline,export_validation,R):stack.enter_context(patch.object(module,'ROOT',root))
                stack.enter_context(patch.object(R,'_CHECKER',None));stack.enter_context(patch.object(R,'_TASK_SOLVER',None))
                # This is a routing/gate test, explicitly no separate-reference claim.
                stack.enter_context(patch.object(pipeline,'reference_solve',R.option_resume_solve))
                yield root,deepcopy(self.rows)

    def fixture_approvals(self,rows,omit=None):
        evidence={}
        for role,actor in pipeline.APPROVAL_ACTORS.items():
            evidence[role]={r['sample_id']:[{'receipt_id':'HAND_ONLY_'+role+'_'+r['sample_id'],
                'sample_id':r['sample_id'],'record_hash':r['record_hash'],'decision':'ACCEPT',
                'reviewer_execution_id':actor,'critical_defects':False,'issues':[],
                'scope':'SYNTHETIC_HAND_FIXTURE_NOT_AN_ACTUAL_REVIEW'}] for r in rows if r['sample_id']!=omit}
        return evidence

    @contextmanager
    def approvals(self,rows,omit=None):
        evidence=self.fixture_approvals(rows,omit)
        # No prior production inputs are read or claimed to be tested here.
        with patch.object(pipeline,'load_receipts',side_effect=lambda role,batch:evidence[role]), patch('factory.lineage.cross_release',return_value={}):
            yield

    def accepted(self,rows):
        result=deepcopy(rows)
        for row in result:
            row['reviewed_candidate_hash']=row['record_hash'];row['review_status']='ACCEPTED'
            row['review_receipt_ids']=['HAND_ONLY_RECEIPT'];row['record_hash']=self.C.record_hash(row)
        return result

    def write_skill_files(self,path,rows):
        path.mkdir(parents=True)
        for split in ('train','dev_challenge'):
            items=[{'input':r['rendered_input'],'target':R.option_resume_solve(r['semantic_task'])} for r in rows if r['split']==split]
            common.write_jsonl(path/(split+'-skill_demonstrations.jsonl'),items)

    def test_schema_complete_fixture_typed_invariants_and_exact_restoration(self):
        self.assertEqual(len(self.rows),216)
        restored=[]
        for original,revised in zip(self.original_rows,self.rows):
            self.assertNotIn('sealed_evaluation_used',original['source_provenance'])
            self.assertIs(revised['source_provenance']['sealed_evaluation_used'],False)
            self.assertEqual(revised['record_hash'],self.C.record_hash(revised))
            row=deepcopy(revised);row['source_provenance'].pop('sealed_evaluation_used')
            row['record_hash']=original['record_hash']
            self.assertTrue(self.C.eq(row,original))
            restored.append(row)
        reverted=(''.join(self.C.canonical(row)+'\n' for row in restored)).encode('utf-8')
        self.assertEqual(reverted,self.original_raw)
        manifest=self.C.strict_json(self.manifest);old=self.C.strict_json(self.original_manifest)
        self.assertEqual(manifest['candidate_sha256'],sha256(self.raw).hexdigest())
        self.assertEqual(len(manifest['control_pairs']),300)
        by_id={r['sample_id']:r for r in self.rows}
        for entry,prior in zip(manifest['control_pairs'],old['control_pairs']):
            self.assertEqual(entry['candidate_sha256'],sha256(self.raw).hexdigest())
            self.assertTrue(self.C.eq(entry['pair_record_hashes'],[by_id[sid]['record_hash'] for sid in entry['pair_sample_ids']]))
            entry['candidate_sha256']=prior['candidate_sha256'];entry['pair_record_hashes']=prior['pair_record_hashes']
        manifest['candidate_sha256']=old['candidate_sha256']
        self.assertTrue(self.C.eq(manifest,old))
        archives=ROOT/REPAIR/'original_byte_archives'
        for digest,raw in (('742364bd1173a76d911a8b43bff70fb30541a46ce5ec637e2ee1477ff34a0bd5',self.original_raw),
                ('bb01ac3d44a62ebdbb95750b9d5ad6721c020c5330cebc89e69252c8ad93817f',self.original_manifest)):
            self.assertEqual((archives/(digest+'.source')).read_bytes(),raw)
        with self.assertRaisesRegex(ValueError,'strict False'):
            corrupted=deepcopy(self.original_rows);corrupted[0]['source_provenance']['sealed_evaluation_used']=0
            raw=(''.join(self.C.canonical(r)+'\n' for r in corrupted)).encode()
            schema_complete_fixture(raw,self.original_manifest,self.C)
        RESULTS['cases'].append({'case':'HAND-schema-complete-exact-restoration','records':216,'ordered_pairs':300,
            'typed_flag':False,'exact_original_candidate_restored':True,'typed_original_manifest_restored':True})

    def test_missing_sealed_flag_blocks_real_schema_audit_and_release(self):
        for replacement in ('MISSING',0):
            with self.sandbox() as (root,rows):
                if replacement=='MISSING':rows[0]['source_provenance'].pop('sealed_evaluation_used')
                else:rows[0]['source_provenance']['sealed_evaluation_used']=replacement
                rows[0]['record_hash']=self.C.record_hash(rows[0])
                raw=(''.join(self.C.canonical(row)+'\n' for row in rows)).encode()
                candidate=root/f'data/RSM-001/candidate_renderings/{R.BATCH}.jsonl';candidate.write_bytes(raw)
                manifest=rebind_manifest(rows,raw,self.C.strict_json(self.manifest),self.C)
                (root/MANIFEST).write_bytes(self.C.canonical(manifest).encode())
                # Rebinding all declared raw hashes isolates the actual shared schema gate.
                controls=R.require_rsm_controls(rows,R.BATCH)
                self.assertEqual(controls['errors'],[])
                expected_error=KeyError if replacement=='MISSING' else AssertionError
                with self.assertRaises(expected_error):pipeline.schema_validate(rows[0])
                stats=pipeline.audit(R.BATCH)
                self.assertGreater(stats['validation_failures'],0);self.assertFalse(stats['all_machine_quality_gates_pass'])
                checks=common.load_jsonl(root/f'reviews/oracle/{R.BATCH}.jsonl')
                self.assertFalse(next(c for c in checks if c['sample_id']==rows[0]['sample_id'])['pass'])
                with self.approvals(rows):
                    with self.assertRaises(AssertionError):pipeline.release(R.BATCH,'HAND_ONLY_NEW')
                self.assertFalse((root/'data/RSM-001/releases/HAND_ONLY_NEW').exists())
                RESULTS['cases'].append({'case':'shared-schema-sealed-flag-negative','replacement':replacement,
                    'validation_failures':stats['validation_failures'],'rsm_controls_error_count':len(controls['errors']),
                    'release_destination_created':False,'schema_validation_mocked':False})

    def test_standalone_dispatch_all_hand_tasks_and_typed_invalid_version(self):
        with self.sandbox() as (root,rows):
            for row in rows:self.assertTrue(self.C.eq(independent_solve(row['semantic_task']),R.task_solver().solve(row['semantic_task'])))
            task=deepcopy(rows[0]['semantic_task']);task['version']=True
            self.assertEqual(independent_solve(task)['status'],'invalid')
            (root/R.TASK_SOLVER_PATH).write_bytes((root/R.TASK_SOLVER_PATH).read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'source changed'):independent_solve(rows[0]['semantic_task'])

    def test_audit_full_hand_proof_and_actual_source_ledger(self):
        with self.sandbox() as (root,rows):
            stats=pipeline.audit(R.BATCH)
            self.assertEqual(stats['rsm_control_error_count'],0);self.assertTrue(stats['all_machine_quality_gates_pass'])
            self.assertEqual(stats['rsm_control_counts']['proved_named_pairs'],300)
            self.assertIn(R.ADAPTER_PATH,stats['rsm_source_and_evidence']);self.assertIn(MANIFEST,stats['rsm_source_and_evidence'])
            RESULTS['cases'].append({'case':'audit-HAND-full','counts':stats['rsm_control_counts'],'source_and_evidence':stats['rsm_source_and_evidence']})

    def test_audit_history_preserves_first_raw_report_on_corruption(self):
        with self.sandbox() as (root,rows):
            pipeline.audit(R.BATCH);report=root/f'audits/diversity/{R.BATCH}-rsm-controls.json';first=report.read_bytes()
            rows[0]['semantic_task']['version']=False;rows[0]['record_hash']=self.C.record_hash(rows[0])
            common.write_jsonl(root/f'data/RSM-001/candidate_renderings/{R.BATCH}.jsonl',rows)
            stats=pipeline.audit(R.BATCH);self.assertGreater(stats['rsm_control_error_count'],0);self.assertFalse(stats['all_machine_quality_gates_pass'])
            snapshot=report.parent/'history'/f'{report.stem}-{sha256(first).hexdigest()}{report.suffix}'
            self.assertEqual(snapshot.read_bytes(),first)

    def test_original_missing_raw_evidence_blocks_release_before_destination(self):
        for path in (R.LEGACY_PATH,MANIFEST,'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md'):
            with self.sandbox() as (root,rows):
                (root/path).unlink()
                with self.assertRaises(R.RSMGateError):pipeline.release(R.BATCH,'HAND_ONLY_NEW')
                self.assertFalse((root/'data/RSM-001/releases/HAND_ONLY_NEW').exists())

    def test_partial_and_zero_approval_cohorts_block_before_destination(self):
        for all_missing in (False,True):
            with self.sandbox() as (root,rows):
                evidence=self.fixture_approvals(rows,rows[-1]['sample_id']) if not all_missing else {role:{} for role in pipeline.APPROVAL_ACTORS}
                with patch.object(pipeline,'load_receipts',side_effect=lambda role,batch:evidence[role]),patch('factory.lineage.cross_release',return_value={}):
                    with self.assertRaises(R.RSMGateError):pipeline.release(R.BATCH,'HAND_ONLY_NEW')
                self.assertFalse((root/'data/RSM-001/releases/HAND_ONLY_NEW').exists())

    def test_full_hand_release_accepted_projection_validation_and_raw_archives(self):
        with self.sandbox() as (root,rows):
            with self.approvals(rows):created=pipeline.release(R.BATCH,'HAND_ONLY_NEW')
            self.assertEqual(created,['data/RSM-001/releases/HAND_ONLY_NEW'])
            path=root/created[0];manifest=json.loads((path/'MANIFEST.json').read_bytes())
            verdict=pipeline.validate_release(path);self.assertTrue(verdict['pass']);self.assertEqual(verdict['records'],216)
            required=set(R.SOURCES)|{R.ADAPTER_PATH,R.LEGACY_PATH,MANIFEST,f'data/RSM-001/candidate_renderings/{R.BATCH}.jsonl'}
            self.assertTrue(required<=set(manifest['rsm_source_and_evidence']))
            for relative,b in manifest['rsm_source_and_evidence'].items():
                raw=(root/relative).read_bytes();self.assertEqual((root/'reproducibility/source_blobs'/(b['sha256']+'.source')).read_bytes(),raw)
            self.assertEqual((root/f'data/RSM-001/candidate_renderings/{R.BATCH}.jsonl').read_bytes(),self.raw)
            self.assertNotEqual(sha256((path/'canonical.jsonl').read_bytes()).hexdigest(),sha256(self.raw).hexdigest())
            RESULTS['cases'].append({'case':'release-HAND-full-synthetic-receipts','records':216,'counts':manifest['rsm_control_counts'],'raw_ledger':manifest['rsm_source_and_evidence'],'production_approval':False})

    def test_direct_three_apis_reject_subset_before_split_or_write(self):
        with self.sandbox() as (root,rows):
            path=root/'forbidden-direct-export'
            with self.assertRaises(R.RSMGateError):exports.specialized(rows[:-1],'train')
            with self.assertRaises(R.RSMGateError):exports.write_specialized(path,rows[:-1])
            self.assertFalse(path.exists())
            with self.assertRaises(R.RSMGateError):export_validation.validate_specialized(path,rows[:-1])

    def test_explicit_batch_cannot_duck_gate_when_all_row_markers_removed(self):
        with self.sandbox() as (root,rows):
            for row in rows:
                row['semantic_task']['kind']='skills';row['semantic_task']['version']=0;row['track_id']='OTHER';row['generator_version']='tampered'
                row.pop('family_control_claims');row.pop('controlled_context');row['record_hash']=self.C.record_hash(row)
            path=root/'forbidden-direct-export'
            for call in (lambda:exports.specialized(rows,'train',batch=R.BATCH),lambda:exports.write_specialized(path,rows,batch=R.BATCH),lambda:export_validation.validate_specialized(path,rows,batch=R.BATCH)):
                with self.assertRaises(R.RSMGateError):call()
            self.assertFalse(path.exists())

    def test_wrong_explicit_batch_with_domain_markers_fails_closed(self):
        with self.sandbox() as (root,rows):
            for batch in ('wrong',R.BATCH+'-bad'):
                with self.assertRaises(R.RSMGateError):exports.specialized(rows,'train',batch=batch)

    def test_direct_both_skill_splits_and_typed_output_corruption(self):
        with self.sandbox() as (root,rows):
            accepted=self.accepted(rows);path=root/'hand-derived'
            exports.write_specialized(path,accepted,batch=R.BATCH)
            counts=export_validation.validate_specialized(path,accepted,batch=R.BATCH)
            self.assertEqual(counts,{'train-skill_demonstrations.jsonl':144,'dev_challenge-skill_demonstrations.jsonl':72})
            items=common.load_jsonl(path/'train-skill_demonstrations.jsonl');items[0]['target']['checkpoint']['root_pending']=0
            common.write_jsonl(path/'train-skill_demonstrations.jsonl',items)
            with self.assertRaisesRegex(ValueError,'typed specialized'):export_validation.validate_specialized(path,accepted,batch=R.BATCH)

    def test_missing_split_and_duplicate_row_fail_direct_validation(self):
        with self.sandbox() as (root,rows):
            path=root/'hand-derived';self.write_skill_files(path,rows)
            (path/'dev_challenge-skill_demonstrations.jsonl').unlink()
            with self.assertRaises(OSError):export_validation.validate_specialized(path,rows,batch=R.BATCH)
            self.write_skill_files(root/'hand-derived-two',rows)
            path=root/'hand-derived-two';items=common.load_jsonl(path/'train-skill_demonstrations.jsonl');items[0]=deepcopy(items[1]);common.write_jsonl(path/'train-skill_demonstrations.jsonl',items)
            with self.assertRaisesRegex(ValueError,'coverage'):export_validation.validate_specialized(path,rows,batch=R.BATCH)

    def test_original_candidate_not_accepted_serialization(self):
        with self.sandbox() as (root,rows):
            accepted=self.accepted(rows)
            common.write_jsonl(root/f'data/RSM-001/candidate_renderings/{R.BATCH}.jsonl',accepted)
            with self.assertRaises(R.RSMGateError):exports.specialized(accepted,'train',batch=R.BATCH)

    def test_exact_binary_extra_source_archive_and_typed_ledger(self):
        with self.sandbox() as (root,rows):
            extra=root/'hand/one-byte.bin';extra.parent.mkdir();extra.write_bytes(b'\xff')
            manifest=self.C.strict_json((root/MANIFEST).read_bytes());manifest['source_bindings']['hand/one-byte.bin']=sha256(b'\xff').hexdigest()
            (root/MANIFEST).write_bytes(self.C.canonical(manifest).encode())
            report=R.require_rsm_controls(rows,R.BATCH);ledger=R.preserve_rsm_bytes(report)
            self.assertEqual((root/'reproducibility/source_blobs'/(sha256(b'\xff').hexdigest()+'.source')).read_bytes(),b'\xff')
            release={'source_archive':'reproducibility/source_blobs/{sha256}.source','rsm_source_and_evidence':deepcopy(ledger),'reproducibility_code_sha256':{p:b['sha256'] for p,b in ledger.items()}}
            R.validate_rsm_release_bindings(report,release)
            release['rsm_source_and_evidence']['hand/one-byte.bin']['bytes']=True
            with self.assertRaisesRegex(ValueError,'typed'):R.validate_rsm_release_bindings(report,release)

    def test_safe_manifest_and_archive_symlinks_rejected(self):
        with self.sandbox() as (root,rows):
            manifest=root/MANIFEST;raw=manifest.read_bytes();target=root/'raw-manifest';target.write_bytes(raw);manifest.unlink();manifest.symlink_to(target)
            with self.assertRaises(R.RSMGateError):exports.specialized(rows,'train',batch=R.BATCH)
        with self.sandbox() as (root,rows):
            report=R.require_rsm_controls(rows,R.BATCH)
            destination=root/'outside-archive';destination.mkdir();(root/'reproducibility').symlink_to(destination,target_is_directory=True)
            with self.assertRaises(OSError):R.preserve_rsm_bytes(report)
            self.assertEqual(list(destination.iterdir()),[])

    def test_release_cannot_delete_archive_flag_and_regenerate_checksums(self):
        with self.sandbox() as (root,rows):
            with self.approvals(rows):created=pipeline.release(R.BATCH,'HAND_ONLY_NEW')
            path=root/created[0];original=json.loads((path/'MANIFEST.json').read_bytes())
            for missing in (True,False):
                manifest=deepcopy(original)
                if missing:manifest.pop('source_archive')
                else:manifest['source_archive']=''
                common.write_json(path/'MANIFEST.json',manifest)
                common.atomic(path/'CHECKSUMS.sha256',''.join(sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in sorted(path.iterdir()) if f.name!='CHECKSUMS.sha256'))
                with self.assertRaisesRegex(ValueError,'archive format'):pipeline.validate_release(path)

    def test_deleted_official_public_and_binary_archive_bytes_fail_restoration(self):
        with self.sandbox() as (root,rows):
            extra=root/'hand/one-byte.bin';extra.parent.mkdir();extra.write_bytes(b'\xff')
            manifest=self.C.strict_json((root/MANIFEST).read_bytes());manifest['source_bindings']['hand/one-byte.bin']=sha256(b'\xff').hexdigest()
            (root/MANIFEST).write_bytes(self.C.canonical(manifest).encode())
            report=R.require_rsm_controls(rows,R.BATCH);ledger=R.preserve_rsm_bytes(report)
            release={'source_archive':'reproducibility/source_blobs/{sha256}.source','rsm_source_and_evidence':ledger,'reproducibility_code_sha256':{p:b['sha256'] for p,b in ledger.items()}}
            for relative in (R.TASK_SOLVER_PATH,'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md','hand/one-byte.bin'):
                blob=root/'reproducibility/source_blobs'/(ledger[relative]['sha256']+'.source');raw=blob.read_bytes();blob.unlink()
                with self.assertRaises(OSError):R.validate_rsm_release_bindings(report,release)
                blob.write_bytes(raw)
            R.validate_rsm_release_bindings(report,release)

    def test_legacy_direct_api_unchanged_without_new_raw_evidence(self):
        rows=[{'track_id':'RSM-001','split':'train','semantic_task':{'kind':'skills'},'rendered_input':'legacy input','verified_target':{'legacy':True}}]
        self.assertFalse(R.is_rsm_cohort(rows));self.assertIsNone(R.require_rsm_controls(rows))
        self.assertEqual(exports.specialized(rows,'train'),{'train-skill_demonstrations.jsonl':[{'input':'legacy input','target':{'legacy':True}}]})

if __name__=='__main__':unittest.main()
