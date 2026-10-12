"""Frozen-source refusal and evidence-bound approval, without pilot seed reuse."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.auditors import release
from goal_conditioned_decision_data.generators import v2_experiment as experiment
from goal_conditioned_decision_data.pipelines.common import canonical,digest,file_hash


class ExperimentCustodyTests(unittest.TestCase):
    def root(self,directory):
        root=Path(directory).resolve()
        (root/'configs').mkdir();(root/'manifests').mkdir();(root/'reports').mkdir()
        spec=json.loads((experiment.ROOT/'configs/v2_experiment.json').read_text())
        (root/'configs/v2_experiment.json').write_text(canonical(spec)+'\n')
        return root

    def test_freeze_replay_preserves_bytes_and_changed_source_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.root(directory)
            with patch.object(experiment,'ROOT',root),patch.object(experiment,'source_hashes',return_value={'code.py':'a'*64}):
                first=experiment.freeze()
                before=(root/'manifests/V2_EXPERIMENT_FREEZE.json').read_bytes()
                self.assertEqual(experiment.freeze(),first)
                self.assertEqual((root/'manifests/V2_EXPERIMENT_FREEZE.json').read_bytes(),before)
                with patch.object(experiment,'source_hashes',return_value={'code.py':'b'*64}),patch.object(experiment,'generate_goal_triplet') as generator:
                    with self.assertRaisesRegex(ValueError,'source/specification changed'):experiment.measure()
                    generator.assert_not_called()

    def test_completed_result_observation_does_not_rerun_goals(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.root(directory)
            code={'code.py':'a'*64}
            with patch.object(experiment,'ROOT',root),patch.object(experiment,'source_hashes',return_value=code):
                experiment.freeze()
                result={'version':'v2-controlled-result-1','verdict':'PASS','source_digest':digest(code),
                        'freeze_sha256':file_hash(root/'manifests/V2_EXPERIMENT_FREEZE.json'),
                        'goal_parents_attempted':32,'goal_parents_valid':32,'vocabulary_pairs':32,
                        'tasks_verified_in_memory':160,'parent_receipts':[{}]*64,
                        'canonical_records_published':0,'training_launched':False,
                        'oracle_or_label_mismatches':0,'reproduction_mismatches':0}
                (root/'reports/V2_CONTROLLED_EXPERIMENT.json').write_text(canonical(result)+'\n')
                with patch.object(experiment,'generate_goal_triplet') as generator:
                    self.assertEqual(experiment.measure(),result)
                    generator.assert_not_called()
                result['freeze_sha256']='0'*64
                (root/'reports/V2_CONTROLLED_EXPERIMENT.json').write_text(canonical(result)+'\n')
                with self.assertRaisesRegex(ValueError,'does not match'):experiment.measure()

    def install_review(self,root,evidence):
        code={'code.py':'c'*64}
        source=digest(code)
        report=root/'reports/INDEPENDENT_REVIEW.md'
        report.write_text('FINAL_VERDICT: PASS\n'+source+'\n'+canonical(evidence)+'\n')
        receipt={'code_hashes':code,'source_digest':source,'independent_review_sha256':file_hash(report),'required_evidence':evidence}
        (root/'manifests/VALIDATED_CODE.json').write_text(canonical(receipt)+'\n')
        return code

    def test_changed_or_missing_experiment_evidence_invalidates_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.root(directory)
            result=root/'reports/experiment.json';result.write_text('verified result\n')
            code=self.install_review(root,{'reports/experiment.json':file_hash(result)})
            with patch.object(release,'ROOT',root),patch.object(release,'source_hashes',return_value=code):
                self.assertTrue(release.current_independent_review())
                result.write_text('changed result\n')
                self.assertFalse(release.current_independent_review())
                result.unlink()
                self.assertFalse(release.current_independent_review())

    def test_external_or_symlinked_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.root(directory)
            outside=root/'outside.txt';outside.write_text('do not trust redirected evidence')
            evidence=root/'reports/experiment.json';evidence.symlink_to(outside)
            cases=[{'reports/experiment.json':file_hash(outside)}, {'../outside.txt':'d'*64},
                   {str(outside):file_hash(outside)}, [], {'reports/experiment.json':None}]
            for item in cases:
                code=self.install_review(root,item)
                with self.subTest(evidence=item),patch.object(release,'ROOT',root),patch.object(release,'source_hashes',return_value=code):
                    self.assertFalse(release.current_independent_review())

    def test_v2_receipt_requires_both_evidence_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.root(directory)
            (root/'configs/production.json').write_text('{"active_generation_profile":"v2"}')
            freeze=root/'manifests/V2_EXPERIMENT_FREEZE.json';freeze.write_text('frozen')
            result=root/'reports/V2_CONTROLLED_EXPERIMENT.json';result.write_text('checked')
            complete={str(path.relative_to(root)):file_hash(path) for path in (freeze,result)}
            for item in ({}, {next(iter(complete)):next(iter(complete.values()))}, complete):
                code=self.install_review(root,item)
                with self.subTest(evidence=item),patch.object(release,'ROOT',root),patch.object(release,'source_hashes',return_value=code):
                    self.assertEqual(release.current_independent_review(),item==complete)
            # Rehashing altered evidence in only the approval receipt cannot
            # change what the immutable independent statement approved.
            code=self.install_review(root,complete)
            result.write_text('altered result')
            path=root/'manifests/VALIDATED_CODE.json'
            receipt=json.loads(path.read_text())
            receipt['required_evidence']['reports/V2_CONTROLLED_EXPERIMENT.json']=file_hash(result)
            path.write_text(canonical(receipt)+'\n')
            with patch.object(release,'ROOT',root),patch.object(release,'source_hashes',return_value=code):
                self.assertFalse(release.current_independent_review())


if __name__=='__main__':unittest.main()
