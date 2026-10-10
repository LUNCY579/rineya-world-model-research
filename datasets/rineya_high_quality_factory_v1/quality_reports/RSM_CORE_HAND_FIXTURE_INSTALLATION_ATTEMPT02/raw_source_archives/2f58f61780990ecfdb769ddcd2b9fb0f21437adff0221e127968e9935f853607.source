"""Handmade V4 factory custody/routing regression tests; no production reads.

Full 3456-row property validation is covered by the unchanged official25 tests.
The bridge positive tests explicitly isolate byte transport with a small actual
typed pair resolver; they do not pretend that eight rows form a complete cohort.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from factory import belief_controls as bridge
from factory import pipeline
from factory.common import canonical,record_hash
from oracles import belief_controls_v4_independent as v4
spec=importlib.util.spec_from_file_location('own_frozen_official_v4_fixtures',BASE/'tests/oracle_belief_controls_v4_test.py')
hand=importlib.util.module_from_spec(spec);spec.loader.exec_module(hand)
BATCH='belief-gap-v4-handmade'
MEASURED={'isolated_actual_pair_proofs':0,'loader_positive_calls':0,'pipeline_audit_calls':0,'pipeline_partial_release_calls':0,'pipeline_validate_calls':0}


def write(root,relative,raw):
    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);return path


def isolated_pair_checker(rows,manifest_sources=None,candidate_bytes=None,previous_rows=None):
    """Explicit test isolation only, still using actual byte and typed proofs."""
    _,by,sha=v4._candidate(candidate_bytes);v4._bind_rows(rows,by,sha)
    proofs=[]
    for owner in rows:
        for claim in owner['family_control_claims']:
            if claim.get('name',claim.get('property')) in v4.INHERITED and claim['scope']=='corpus_variant':
                proofs.append(v4.resolve_claim(owner,claim,by,json.loads(manifest_sources[claim['evidence_manifest']]),sha))
    retained=v4.preservation(rows,previous_rows);MEASURED['isolated_actual_pair_proofs']+=len(proofs)
    return {'errors':retained['errors'],'counts':{'records':len(rows),'isolated_actual_pairs':len(proofs)},'v4_preservation':retained,'test_scope':'Small transport fixture; full cohort engine explicitly isolated in this unit test.'}


class FactoryBridge(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for relative in bridge.SOURCE_BINDINGS:
            write(self.root,relative,(BASE/relative).read_bytes())
        write(self.root,'factory/belief_controls.py',(BASE/'factory/belief_controls.py').read_bytes())
        self.rows=[hand.row(context=context) for context in hand.v2.CONTEXTS]
        for row in self.rows:row['generator_version']='rbm-belief-gap-4.0.0';row['record_hash']=record_hash(row)
        self.candidate_path='data/RBM-001/candidate_renderings/'+BATCH+'.jsonl'
        self.raw,self.sources,self.manifest=hand.bundle(self.rows)
        write(self.root,self.candidate_path,self.raw)
        for relative,raw in self.sources.items():write(self.root,relative,raw)
        self.prior=deepcopy(self.rows)
        for row in self.prior:row['generator_version']='rbm-belief-gap-3.0.0';row['record_hash']=record_hash(row)
        self.prior_raw=hand.candidate(self.prior)
        self.prior_binding={'path':bridge.PRIOR_BINDING['path'],'sha256':hashlib.sha256(self.prior_raw).hexdigest(),'bytes':len(self.prior_raw)}
        write(self.root,self.prior_binding['path'],self.prior_raw)
        self.prior_patch=patch.object(bridge,'PRIOR_BINDING',self.prior_binding);self.prior_patch.start()

    def tearDown(self):self.prior_patch.stop();self.temp.cleanup()

    def accepted(self):
        rows=deepcopy(self.rows)
        for row in rows:
            row.update(reviewed_candidate_hash=row['record_hash'],review_status='ACCEPTED',review_receipt_ids=['own-handmade-execution'])
            row['record_hash']=record_hash(row)
        return rows

    def test_exact_candidate_and_manifest_ledger_without_unpublished_top_fields(self):
        self.assertNotIn('version',self.manifest);self.assertNotIn('batch',self.manifest)
        loaded=bridge.load_v4_evidence(self.rows,BATCH,self.root);self.assertEqual(loaded['errors'],[])
        self.assertEqual(loaded['candidate_bytes'],self.raw);self.assertEqual(canonical(loaded['previous_rows']),canonical(self.prior));self.assertEqual(loaded['manifest_sources'],self.sources)
        MEASURED['loader_positive_calls']+=1
        with patch.object(v4,'check_batch',side_effect=isolated_pair_checker):report=bridge.belief_v4_control_report(self.rows,BATCH,self.root)
        self.assertEqual(report['errors'],[]);self.assertEqual(report['v4_preservation']['status'],'PROVED');self.assertEqual(report['counts']['isolated_actual_pairs'],16)

    def test_accepted_hashes_bind_preserved_pending_candidate_not_accepted_file(self):
        accepted=self.accepted();self.assertNotEqual(hashlib.sha256(hand.candidate(accepted)).hexdigest(),hashlib.sha256(self.raw).hexdigest())
        loaded=bridge.load_v4_evidence(accepted,BATCH,self.root);self.assertEqual(loaded['errors'],[]);self.assertEqual(loaded['candidate_bytes'],self.raw);MEASURED['loader_positive_calls']+=1
        with patch.object(v4,'check_batch',side_effect=isolated_pair_checker):report=bridge.belief_v4_control_report(accepted,BATCH,self.root)
        self.assertEqual(report['errors'],[]);self.assertEqual(report['evidence_sha256'][self.candidate_path],hashlib.sha256(self.raw).hexdigest())

    def test_changed_candidate_record_and_release_nonreview_mutation_fail(self):
        accepted=self.accepted();accepted[0]['semantic_task']['worlds'][0]['facts']['feature']=True;accepted[0]['record_hash']=record_hash(accepted[0])
        self.assertTrue(bridge.load_v4_evidence(accepted,BATCH,self.root)['errors'])
        altered=deepcopy(self.rows);altered[0]['difficulty']=4;altered[0]['record_hash']=record_hash(altered[0]);write(self.root,self.candidate_path,hand.candidate(altered))
        self.assertTrue(bridge.load_v4_evidence(self.rows,BATCH,self.root)['errors'])

    def test_missing_and_hash_or_size_changed_frozen_prior_fail_before_checker(self):
        path=self.root/self.prior_binding['path'];path.unlink()
        with patch.object(v4,'check_batch',side_effect=AssertionError('proof must not run')):
            self.assertTrue(bridge.belief_v4_control_report(self.rows,BATCH,self.root)['errors'])
        write(self.root,self.prior_binding['path'],self.prior_raw+b' ');self.assertTrue(bridge.load_v4_evidence(self.rows,BATCH,self.root)['errors'])
        write(self.root,self.prior_binding['path'],self.prior_raw)
        with patch.object(bridge,'PRIOR_BINDING',{**self.prior_binding,'bytes':True}):self.assertTrue(bridge.load_v4_evidence(self.rows,BATCH,self.root)['errors'])

    def test_missing_manifest_candidate_or_named_entry_is_not_a_pointer_proof(self):
        (self.root/hand.MANIFEST_PATH).unlink();self.assertTrue(bridge.load_v4_evidence(self.rows,BATCH,self.root)['errors'])
        write(self.root,hand.MANIFEST_PATH,json.dumps({'supplied_same_table_v2_pairs':[]}).encode())
        with patch.object(v4,'check_batch',side_effect=isolated_pair_checker):
            self.assertTrue(bridge.belief_v4_control_report(self.rows,BATCH,self.root)['errors'])
        (self.root/self.candidate_path).unlink();self.assertTrue(bridge.load_v4_evidence(self.rows,BATCH,self.root)['errors'])

    def test_root_escape_absolute_and_symlink_evidence_rejected(self):
        for path in ('../outside.json','/outside.json','C:\\outside.json','sub/../../outside.json'):
            with self.assertRaises(ValueError):bridge.safe_read(self.root,path)
        outside=write(self.root,'actual-target.json',b'{}');(self.root/'leaf-link.json').symlink_to(outside)
        with self.assertRaises(OSError):bridge.safe_read(self.root,'leaf-link.json')
        (self.root/'parent-link').symlink_to(self.root/'schemas',target_is_directory=True)
        with self.assertRaises(OSError):bridge.safe_read(self.root,'parent-link/canonical/v1.schema.json')
        claim=next(c for c in self.rows[0]['family_control_claims'] if c.get('name') in v4.INHERITED);claim['evidence_manifest']='leaf-link.json';self.rows[0]['record_hash']=record_hash(self.rows[0]);self.raw=hand.candidate(self.rows);write(self.root,self.candidate_path,self.raw)
        self.assertTrue(bridge.load_v4_evidence(self.rows,BATCH,self.root)['errors'])

    def test_frozen_source_and_public_document_corruption_fail(self):
        for relative in ('oracles/belief_controls_v4_independent.py','schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md','factory/belief_controls.py'):
            original=(self.root/relative).read_bytes();write(self.root,relative,original+b'\n')
            self.assertTrue(bridge.load_v4_evidence(self.rows,BATCH,self.root)['errors']);write(self.root,relative,original)

    def test_malformed_batch_and_tampered_version_cannot_duck_v4(self):
        for version in ('rbm-belief-gap-3.0.0','rbm-belief-gap-4.1.0','rbm-belief-gap-4.0.0x',True):
            rows=deepcopy(self.rows)
            for row in rows:row['generator_version']=version;row['record_hash']=record_hash(row)
            self.assertEqual(pipeline.belief_checker_name(rows,BATCH),bridge.CHECKER_NAME);self.assertEqual(pipeline.belief_control_rows(rows,BATCH),rows)
            self.assertTrue(bridge.belief_v4_control_report(rows,BATCH,self.root)['errors'])
        self.assertTrue(bridge.belief_v4_control_report(self.rows,'legacy',self.root)['errors'])
        self.assertTrue(bridge.belief_v4_control_report(self.rows,'belief-gap-v4/escape',self.root)['errors'])

    def test_stripped_context_kind_and_claims_still_route_by_batch(self):
        rows=deepcopy(self.rows)
        for row in rows:
            row['generator_version']='legacy';row['semantic_task']['kind']='other'
            for key in ('controlled_context','family_control_claims','auxiliary_scalar_variant','auxiliary_scalar_control'):row.pop(key,None)
            row['record_hash']=record_hash(row)
        self.assertEqual(pipeline.belief_checker_name(rows,BATCH),bridge.CHECKER_NAME);self.assertEqual(len(pipeline.belief_control_rows(rows,BATCH)),8)
        self.assertTrue(pipeline.belief_control_report(rows,BATCH,self.root)['errors'])

    def test_empty_v4_and_missing_prior_preservation_are_fail_closed(self):
        self.assertTrue(pipeline.belief_control_report([],BATCH,self.root)['errors'])
        with patch.object(v4,'check_batch',return_value={'errors':[],'counts':{'records':8},'v4_preservation':{'status':'NOT_RUN'}}):
            report=bridge.belief_v4_control_report(self.rows,BATCH,self.root)
        self.assertTrue(report['errors']);self.assertTrue(any('preservation' in e for e in report['errors']))

    def test_manifest_candidate_hash_and_duplicate_named_key_rejected_by_actual_pair_engine(self):
        for mode in ('wrong-hash','duplicate'):
            parsed=deepcopy(self.manifest)
            if mode=='wrong-hash':
                for entry in parsed['supplied_same_table_v2_pairs']:entry['exact_candidate_sha256']='0'*64
            else:parsed['supplied_same_table_v2_pairs'].append(deepcopy(parsed['supplied_same_table_v2_pairs'][0]))
            write(self.root,hand.MANIFEST_PATH,json.dumps(parsed).encode())
            with patch.object(v4,'check_batch',side_effect=isolated_pair_checker):
                self.assertTrue(bridge.belief_v4_control_report(self.rows,BATCH,self.root)['errors'])

    def test_evidence_changed_during_proof_is_detected(self):
        def change(*args,**kwargs):
            report=isolated_pair_checker(*args,**kwargs);path=self.root/hand.MANIFEST_PATH;path.write_bytes(path.read_bytes()+b'\n');return report
        with patch.object(v4,'check_batch',side_effect=change):report=bridge.belief_v4_control_report(self.rows,BATCH,self.root)
        self.assertTrue(any('changed during audit' in error for error in report['errors']))

    def test_legacy_names_and_single_argument_api_remain_unchanged(self):
        for version,name in [('old','oracles.belief_controls_independent'),('rbm-belief-gap-2.0.0','oracles.belief_controls_v2_independent'),('rbm-belief-gap-3.0.0','oracles.belief_controls_v3_independent')]:
            row={'semantic_task':{'kind':'belief'},'generator_version':version};self.assertEqual(pipeline.belief_checker_name([row]),name)
        self.assertIsNone(pipeline.belief_control_report([{'semantic_task':{'kind':'belief'}}]))

    def test_pipeline_audit_records_missing_prior_as_batch_gate_failure(self):
        (self.root/self.prior_binding['path']).unlink()
        with patch.object(pipeline,'ROOT',self.root),patch.object(pipeline,'candidate_paths',return_value=[self.root/self.candidate_path]),patch.object(pipeline,'reference_solve',side_effect=lambda task:v4.v1.derive(task)['target']):
            stats=pipeline.audit(BATCH)
        MEASURED['pipeline_audit_calls']+=1
        self.assertEqual(stats['oracle_failures'],0);self.assertGreater(stats['belief_control_error_count'],0);self.assertFalse(stats['all_machine_quality_gates_pass'])
        report=json.loads((self.root/f'audits/diversity/{BATCH}-belief-controls.json').read_bytes());self.assertEqual(report['checker_sha256'],bridge.CHECKER_SHA256)

    def test_partial_approval_blocks_before_release_directory_creation(self):
        write(self.root,f'reviews/oracle/{BATCH}.jsonl',hand.candidate([{'sample_id':row['sample_id'],'pass':True} for row in self.rows]))
        stats={'split_leakage':[],'cross_parent_semantic_duplicates':[],'exact_prompt_duplicates':0}
        def evidence(row,_):return ([{'receipt_id':'own-handmade'}],[]) if row['sample_id']!=self.rows[-1]['sample_id'] else ([],['synthetic-pending-role'])
        with patch.object(pipeline,'ROOT',self.root),patch.object(pipeline,'audit',return_value=stats),patch.object(pipeline,'candidate_paths',return_value=[self.root/self.candidate_path]),patch.object(pipeline,'load_receipts',return_value={}),patch.object(pipeline,'approval_evidence',side_effect=evidence),patch.object(v4,'check_batch',side_effect=AssertionError('partial candidate must fail before proof')):
            with self.assertRaisesRegex(AssertionError,'belief family controls'):pipeline.release(BATCH,'own-handmade-release')
        MEASURED['pipeline_partial_release_calls']+=1
        self.assertFalse((self.root/'data/RBM-001/releases/own-handmade-release').exists())

    def test_validate_release_uses_preserved_candidate_custody_and_prior(self):
        accepted=deepcopy(self.rows);receipts=[]
        for row in accepted:
            original=row['record_hash'];own=[]
            for role,actor in pipeline.APPROVAL_ACTORS.items():
                receipt_id='own-handmade-'+role+'-'+row['sample_id'];own.append(receipt_id)
                receipts.append({'receipt_id':receipt_id,'record_hash':original,'decision':'ACCEPT','reviewer_execution_id':actor})
            row.update(reviewed_candidate_hash=original,review_status='ACCEPTED',review_receipt_ids=own);row['record_hash']=record_hash(row)
        destination=self.root/'data/RBM-001/releases/own-handmade';destination.mkdir(parents=True)
        files={'canonical.jsonl':hand.candidate(accepted),'review_receipts.jsonl':hand.candidate(receipts)}
        for split in ('train','dev_challenge'):
            subset=[row for row in accepted if row['split']==split]
            files[split+'-sft.jsonl']=hand.candidate([{'messages':[{'role':'user','content':row['rendered_input']},{'role':'assistant','content':canonical(row['verified_target'])}]} for row in subset])
            files[split+'-structured.jsonl']=hand.candidate([{'input':row['rendered_input'],'target':row['verified_target']} for row in subset])
        for name,raw in files.items():(destination/name).write_bytes(raw)
        release_manifest={'batch':BATCH,'track':'RBM-001','records':len(accepted),'files':{name:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()} for name,raw in files.items()}}
        raw=canonical(release_manifest).encode();files['MANIFEST.json']=raw;(destination/'MANIFEST.json').write_bytes(raw)
        (destination/'CHECKSUMS.sha256').write_text(''.join(hashlib.sha256(raw).hexdigest()+'  '+name+'\n' for name,raw in sorted(files.items())))
        with patch.object(pipeline,'ROOT',self.root),patch('factory.export_validation.validate_specialized',return_value={}),patch.object(v4,'check_batch',side_effect=isolated_pair_checker):report=pipeline.validate_release(destination)
        MEASURED['pipeline_validate_calls']+=1
        self.assertTrue(report['pass']);self.assertEqual(report['records'],8)
        (self.root/self.prior_binding['path']).unlink()
        with patch.object(pipeline,'ROOT',self.root),patch('factory.export_validation.validate_specialized',return_value={}):
            with self.assertRaisesRegex(AssertionError,'belief family controls'):pipeline.validate_release(destination)


if __name__=='__main__':
    program=unittest.main(verbosity=2,exit=False)
    print('ACTUAL_V4_BRIDGE_HANDMADE_COUNTS '+json.dumps(MEASURED,sort_keys=True))
    sys.exit(not program.result.wasSuccessful())
