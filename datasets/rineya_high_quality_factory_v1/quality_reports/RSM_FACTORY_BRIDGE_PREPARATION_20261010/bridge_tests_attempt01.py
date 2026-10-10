"""Hand-only isolated routing/custody/direct-export/package gate regressions."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).parent; ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('isolated_rsm_bridge',HERE/'rsm_bridge_attempt01.py')
B=importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
HAND='reviews/oracle/RSM-option-resumption-corpus-20261010T0956Z/'
CANDIDATE=HAND+'actual-hand-candidate-attempt04.jsonl'
MANIFEST=HAND+'hand-control-manifest.json'
RESULTS={'cases':[]}

class RSMBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.C=B.checker()
        cls.raw=(ROOT/CANDIDATE).read_bytes(); assert sha256(cls.raw).hexdigest()=='742364bd1173a76d911a8b43bff70fb30541a46ce5ec637e2ee1477ff34a0bd5'
        cls.manifest=(ROOT/MANIFEST).read_bytes(); assert sha256(cls.manifest).hexdigest()=='bb01ac3d44a62ebdbb95750b9d5ad6721c020c5330cebc89e69252c8ad93817f'
        cls.rows=[cls.C.strict_json(line) for line in cls.raw.splitlines()]

    def sandbox(self):
        temporary=tempfile.TemporaryDirectory(prefix='actual-hand-case-',dir=HERE); root=Path(temporary.name)
        for path in list(B.SOURCES)+[B.LEGACY_PATH]:
            target=root/path; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes((ROOT/path).read_bytes())
        target=root/('data/RSM-001/candidate_renderings/'+B.BATCH+'.jsonl'); target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(self.raw)
        target=root/MANIFEST; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(self.manifest)
        return temporary,root,deepcopy(self.rows)
    def recustody(self,root,rows):
        for row in rows: row['record_hash']=self.C.record_hash(row)
        raw=b''.join((self.C.canonical(row)+'\n').encode() for row in rows)
        (root/('data/RSM-001/candidate_renderings/'+B.BATCH+'.jsonl')).write_bytes(raw)
        manifest=self.C.strict_json((root/MANIFEST).read_bytes()); manifest['candidate_sha256']=sha256(raw).hexdigest(); by={r['sample_id']:r for r in rows}
        for e in manifest['control_pairs']:
            e['candidate_sha256']=manifest['candidate_sha256']
            if 'pair_record_hashes' in e: e['pair_record_hashes']=[by[sid]['record_hash'] for sid in e['pair_sample_ids']]
        (root/MANIFEST).write_bytes(self.C.canonical(manifest).encode())
    def record(self,name,report):
        RESULTS['cases'].append({'name':name,'errors':report['errors'],'counts':report.get('counts',{}),'route':report.get('route'),'stage':report.get('stage'),'source_hash':report.get('checker_sha256')})

    def test_original_module_location_and_exact_source_package_inputs(self):
        self.assertEqual(Path(self.C.__file__).resolve(),(ROOT/B.CHECKER_PATH).resolve()); self.assertEqual(self.C.ROOT,ROOT)
        self.assertEqual(sha256(Path(self.C.__file__).read_bytes()).hexdigest(),B.CHECKER_SHA)
        self.assertEqual(set(self.rows[0]['semantic_task']),set('kind version initial_state options entry mode history checkpoint event_budget'.split()))
        self.assertEqual(len(self.rows),216)

    def test_legacy_rows_delegate_without_source_or_candidate_access(self):
        rows=[{'track_id':'RSM-001','semantic_task':{'kind':'skills'},'generator_version':'1.0.0'}]
        self.assertFalse(B.is_cohort(rows,'pilot-v2'))
        for stage in ('audit','publish','validate_release','exports'):
            self.assertEqual(B.gate(stage,rows,'pilot-v2')['route'],'LEGACY_DELEGATE')

    def test_routing_survives_kind_version_deleted_metadata_and_track(self):
        rows=deepcopy(self.rows)
        for row in rows:
            row['semantic_task']['kind']='skills'; row['semantic_task']['version']=False; row['track_id']='OTHER'
            row.pop('family_control_claims'); row.pop('controlled_context'); row['generator_version']='tampered'
        self.assertTrue(B.is_cohort(rows,B.BATCH)); self.assertEqual(len(B.cohort_rows(rows,B.BATCH)),216)
        self.assertTrue(B.is_cohort([],B.BATCH)); self.assertTrue(B.is_cohort(rows,B.BATCH+'-tampered'))
        self.assertTrue(B.is_cohort([{'generator_version':'rsm-option-resumption-unknown'}],None))
        self.assertTrue(B.is_cohort([{'family_control_claims':[{'property':'resume_restart_replay'}]}],None))
        self.assertFalse(B.is_cohort([{'family_control_claims':[{'property':'object_field_order_invariance'}]}],None))

    def test_all_four_stage_boundaries_full_hand_pass(self):
        t,root,rows=self.sandbox()
        with t:
            for stage in ('audit','publish','validate_release','exports'):
                report=B.gate(stage,rows,B.BATCH,root); self.assertEqual(report['errors'],[]); self.assertTrue(report['machine_controls_pass'])
                self.assertEqual(report['counts']['proved_named_pairs'],300); self.assertFalse(report['production_data_approval'])
                self.assertEqual(report['reference_agreement'],'NOT_RUN_SEPARATE_AUTHOR_REQUIRED'); self.record(stage,report)

    def test_current_subset_fails_every_stage_without_directory_creation(self):
        t,root,rows=self.sandbox()
        with t:
            for stage in ('audit','publish','validate_release','exports'):
                destination=root/('forbidden-'+stage)
                report=B.gate(stage,rows[:-1],B.BATCH,root); self.assertTrue(report['errors']); self.assertFalse(destination.exists()); self.record('subset-'+stage,report)
            for args in (([],B.BATCH),(rows,None),(rows,B.BATCH+'-bad')):
                self.assertTrue(B.gate('publish',args[0],args[1],root)['errors'])

    def test_deleted_all_current_metadata_and_mutated_labels_fail_closed(self):
        t,root,rows=self.sandbox()
        with t:
            for row in rows:
                row['semantic_task']['kind']='skills'; row['semantic_task']['version']=0; row['family_control_claims']=[]; row.pop('controlled_context'); row['record_hash']=self.C.record_hash(row)
            report=B.gate('audit',rows,B.BATCH,root); self.assertTrue(report['errors']); self.assertEqual(report['route'],'RSM_WHOLE_COHORT'); self.record('deleted-current-markers',report)

    def test_original_pending_subset_and_missing_candidate_unknown(self):
        t,root,rows=self.sandbox()
        with t:
            path=root/('data/RSM-001/candidate_renderings/'+B.BATCH+'.jsonl')
            path.write_bytes(b''.join(line+b'\n' for line in self.raw.splitlines()[:-1]))
            report=B.gate('publish',rows,B.BATCH,root); self.assertTrue(any('PENDING216' in e for e in report['errors'])); self.record('original-subset',report)
            path.unlink(); report=B.gate('validate_release',rows,B.BATCH,root); self.assertTrue(any('UNKNOWN' in e for e in report['errors']))

    def test_original_kind_version_receipts_and_metadata_corruption_unknown(self):
        for name,mutate in [('kind',lambda r:r['semantic_task'].__setitem__('kind','skills')),('version',lambda r:r['semantic_task'].__setitem__('version',True)),('review',lambda r:r.__setitem__('review_status','ACCEPTED')),('receipts',lambda r:r.__setitem__('review_receipt_ids',['unexpected'])),('claims',lambda r:r.__setitem__('family_control_claims',[]))]:
            t,root,rows=self.sandbox()
            with t:
                mutate(rows[0]); self.recustody(root,rows); report=B.gate('audit',rows,B.BATCH,root)
                self.assertTrue(any('UNKNOWN' in e for e in report['errors']),name); self.record('original-'+name,report)

    def test_all_six_sources_missing_or_changed_fail_closed(self):
        for path in self.C.PUBLIC:
            t,root,rows=self.sandbox()
            with t:
                (root/path).write_bytes((root/path).read_bytes()+b'\n'); report=B.gate('audit',rows,B.BATCH,root)
                self.assertTrue(any('hash mismatch' in e for e in report['errors']),path)
        t,root,rows=self.sandbox()
        with t:
            (root/list(self.C.PUBLIC)[-1]).unlink(); self.assertTrue(B.gate('publish',rows,B.BATCH,root)['errors'])

    def test_exact_checker_solver_graph_and_legacy_required(self):
        for path in (B.CHECKER_PATH,*self.C.DEPENDENCIES,B.LEGACY_PATH):
            t,root,rows=self.sandbox()
            with t:
                (root/path).write_bytes((root/path).read_bytes()+b'\n'); report=B.gate('validate_release',rows,B.BATCH,root)
                self.assertTrue(any('hash mismatch' in e for e in report['errors']),path)

    def test_missing_changed_manifest_and_asserted_additional_raw_source(self):
        t,root,rows=self.sandbox()
        with t:
            (root/MANIFEST).unlink(); self.assertTrue(B.gate('exports',rows,B.BATCH,root)['errors'])
        t,root,rows=self.sandbox()
        with t:
            manifest=self.C.strict_json(self.manifest); manifest['source_bindings']['hand/extra.txt']=sha256(b'actual-extra').hexdigest()
            (root/MANIFEST).write_bytes(self.C.canonical(manifest).encode()); report=B.gate('exports',rows,B.BATCH,root); self.assertTrue(report['errors'])
            extra=root/'hand/extra.txt'; extra.parent.mkdir(); extra.write_bytes(b'actual-extra')
            self.assertEqual(B.gate('audit',rows,B.BATCH,root)['errors'],[])
            manifest['candidate_sha256']='0'*64; (root/MANIFEST).write_bytes(self.C.canonical(manifest).encode())
            self.assertTrue(any('whole PENDING candidate SHA' in e for e in B.gate('audit',rows,B.BATCH,root)['errors']))

    def test_safe_paths_traversal_absolute_symlink_nonregular(self):
        t,root,rows=self.sandbox()
        with t:
            for value in ('/tmp/file','../escape','a/../b','a//b','a\\b',''):
                with self.assertRaises(ValueError): B.safe_read(root,value)
            link=root/'link'; link.symlink_to(root/MANIFEST)
            with self.assertRaises(OSError): B.safe_read(root,'link')
            directory=root/'directory'; directory.mkdir()
            with self.assertRaises((OSError,ValueError)): B.safe_read(root,'directory')
            target=root/'manifests-link'; target.symlink_to((root/MANIFEST).parent,target_is_directory=True)
            with self.assertRaises(OSError): B.safe_read(root,'manifests-link/hand-control-manifest.json')

    def test_evidence_changed_during_proof_detected(self):
        t,root,rows=self.sandbox()
        with t:
            original=self.C.check_batch
            def changed(*args,**kwargs):
                result=original(*args,**kwargs); (root/MANIFEST).write_bytes(self.manifest+b'\n'); return result
            with patch.object(self.C,'check_batch',changed): report=B.gate('audit',rows,B.BATCH,root)
            self.assertTrue(any('changed during gate' in e for e in report['errors'])); self.record('mid-gate-manifest-change',report)

    def test_typed_gold_boolean_integer_substitution_not_accepted(self):
        t,root,rows=self.sandbox()
        with t:
            rows[0]['verified_target']['new_events'][0]['instance_id']=False
            self.recustody(root,rows); report=B.gate('audit',rows,B.BATCH,root)
            self.assertTrue(any('typed target mismatch' in e for e in report['errors'])); self.record('bool-int-target',report)

    def test_full_accepted_projection_and_partial_approval(self):
        t,root,rows=self.sandbox()
        with t:
            for row in rows:
                row['reviewed_candidate_hash']=row['record_hash']; row['review_status']='ACCEPTED'; row['review_receipt_ids']=['hand-receipt']; row['record_hash']=self.C.record_hash(row)
            report=B.gate('validate_release',rows,B.BATCH,root); self.assertEqual(report['errors'],[]); self.record('accepted-full',report)
            self.assertTrue(B.gate('publish',rows[:-1],B.BATCH,root)['errors'])
            rows[0]['skill_tags'].append('illicit-change'); rows[0]['record_hash']=self.C.record_hash(rows[0])
            self.assertTrue(any('immutable projection' in e for e in B.gate('validate_release',rows,B.BATCH,root)['errors']))

    def test_accepted_serialization_not_candidate_bytes(self):
        t,root,rows=self.sandbox()
        with t:
            for row in rows:
                row['reviewed_candidate_hash']=row['record_hash']; row['review_status']='ACCEPTED'; row['review_receipt_ids']=['hand-receipt']; row['record_hash']=self.C.record_hash(row)
            (root/('data/RSM-001/candidate_renderings/'+B.BATCH+'.jsonl')).write_bytes(b''.join((self.C.canonical(r)+'\n').encode() for r in rows))
            self.assertTrue(any('original PENDING' in e for e in B.gate('validate_release',rows,B.BATCH,root)['errors']))

    def test_direct_specialized_write_and_validation_block_subsets(self):
        t,root,rows=self.sandbox()
        with t:
            destination=root/'new-export-directory'
            with self.assertRaises(B.GateError): B.specialized(rows[:-1],'train',B.BATCH,root)
            with self.assertRaises(B.GateError): B.write_specialized(destination,rows[:-1],B.BATCH,root)
            self.assertFalse(destination.exists())
            with self.assertRaises(B.GateError): B.validate_specialized({},rows[:-1],B.BATCH,root)
            (root/MANIFEST).unlink()
            with self.assertRaises(B.GateError): B.write_specialized(destination,rows,B.BATCH,root)
            self.assertFalse(destination.exists())

    def test_direct_specialized_both_split_coverage_and_exact_types(self):
        t,root,rows=self.sandbox()
        with t:
            train=B.specialized(rows,'train',B.BATCH,root); dev=B.specialized(rows,'dev_challenge',B.BATCH,root)
            self.assertEqual(len(train['train-skill_demonstrations.jsonl']),144); self.assertEqual(len(dev['dev_challenge-skill_demonstrations.jsonl']),72)
            exports={**train,**dev}; self.assertEqual(B.validate_specialized(exports,rows,B.BATCH,root)['errors'],[])
            exports['train-skill_demonstrations.jsonl'][0]['target']['checkpoint']['root_pending']=0
            with self.assertRaisesRegex(ValueError,'typed specialized'): B.validate_specialized(exports,rows,B.BATCH,root)
            destination=root/'actual-specialized-output'; B.write_specialized(destination,rows,B.BATCH,root)
            self.assertEqual(len((destination/'train-skill_demonstrations.jsonl').read_bytes().splitlines()),144)

    def test_all_derived_formats_coverage_missing_duplicate_and_bool_int(self):
        t,root,rows=self.sandbox()
        with t:
            exports=B._export_payloads(rows,self.C)
            self.assertEqual(B.gate('exports',rows,B.BATCH,root,exports=exports)['errors'],[])
            exports['train-structured.jsonl'][0]['target']['checkpoint']['root_pending']=0
            self.assertTrue(any('typed one-per-row' in e for e in B.gate('exports',rows,B.BATCH,root,exports=exports)['errors']))
            exports=B._export_payloads(rows,self.C); exports['train-sft.jsonl'].pop()
            self.assertTrue(B.gate('exports',rows,B.BATCH,root,exports=exports)['errors'])
            exports=B._export_payloads(rows,self.C); exports.pop('dev_challenge-skill_demonstrations.jsonl')
            self.assertTrue(B.gate('exports',rows,B.BATCH,root,exports=exports)['errors'])
            exports=B._export_payloads(rows,self.C); exports['train-structured.jsonl'][0]=deepcopy(exports['train-structured.jsonl'][1])
            self.assertTrue(B.gate('exports',rows,B.BATCH,root,exports=exports)['errors'])

    def test_source_package_exact_review_paths_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix='actual-package-case-',dir=HERE) as temporary:
            destination=Path(temporary)/'package'; plan=B.package_sources(destination)
            self.assertIn(B.CHECKER_PATH,plan['source_bindings']); self.assertTrue(set(self.C.DEPENDENCIES)<=set(plan['source_bindings']))
            self.assertTrue(set(self.C.PUBLIC)<=set(plan['source_bindings'])); self.assertIn(B.LEGACY_PATH,plan['source_bindings'])
            proof=B.verify_package(destination); self.assertEqual(proof['source_files'],13); self.assertTrue(proof['all_exact'])
            RESULTS['package']={'source_bindings':plan['source_bindings'],'verification':proof}
            with self.assertRaisesRegex(ValueError,'immutable'): B.package_sources(destination)
            p=destination/plan['source_bindings'][B.CHECKER_PATH]['archive']; p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'packaged exact'): B.verify_package(destination)
            with self.assertRaisesRegex(ValueError,'isolated'): B.package_sources(ROOT/'reproducibility/new-forbidden')

if __name__=='__main__': unittest.main()
