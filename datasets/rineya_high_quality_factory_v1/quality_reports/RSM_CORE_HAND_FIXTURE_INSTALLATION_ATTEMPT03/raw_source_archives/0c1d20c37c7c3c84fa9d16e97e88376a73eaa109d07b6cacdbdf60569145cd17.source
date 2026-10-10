"""Public evidence-key interface regressions; no candidate files or private gold."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oracles import belief_controls_v2_independent as v2
spec=importlib.util.spec_from_file_location('own_frozen_v2_fixtures',Path(__file__).with_name('oracle_belief_controls_v2_test.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def shared(item):
    for claim in item['family_control_claims']:
        if claim['name'] in v2.NEW_FAMILY:
            claim['control_id']='physical-pair:'+':'.join(claim['pair_sample_ids'])
    return item


class RowEvidenceKeys(unittest.TestCase):
    def test_shared_physical_pair_identifier_two_properties(self):
        item=shared(fixtures.fixture())
        new=[c for c in item['family_control_claims'] if c['name'] in v2.NEW_FAMILY]
        self.assertEqual(new[0]['control_id'],new[1]['control_id'])
        self.assertEqual(new[0]['pair_sample_ids'],new[1]['pair_sample_ids'])
        self.assertEqual(new[0]['evidence_manifest'],new[1]['evidence_manifest'])
        self.assertEqual(v2.check(item),[])
    def test_descriptive_key_cannot_cross_scopes(self):
        item=shared(fixtures.fixture());claim=deepcopy(item['family_control_claims'][-1]);claim.update(scope='audit_probe',probe_id='actually-required')
        del claim['pair_sample_ids'];item['family_control_claims'].append(claim)
        self.assertTrue(any('scope' in e for e in v2.check(item)))
    def test_shared_key_does_not_admit_unknown_property_or_fields(self):
        for change in ({'name':'unpublished'},{'extra':1}):
            item=shared(fixtures.fixture());item['family_control_claims'][-1].update(change);self.assertTrue(v2.check(item))
    def test_new_pair_scope_and_family_stay_closed(self):
        for change in ({'source_family_key':'foreign'},{'scope':'audit_probe'}):
            item=shared(fixtures.fixture());item['family_control_claims'][-1].update(change);self.assertTrue(v2.check(item))


class BatchEvidenceKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.inputs=[shared(item) for item in fixtures.batch()];cls.report=v2.check_batch(cls.inputs)
    def test_shared_keys_preserve_separately_executed_properties(self):
        self.assertFalse(self.report['errors'],self.report['errors'][:3])
        self.assertEqual(self.report['counts']['records'],1728)
        for name in v2.NEW_FAMILY:self.assertEqual(len(self.report['corpus_controls'][name]),432)
        matched=[c for c in self.report['claim_checks'] if c['claim']['property'] in v2.NEW_FAMILY]
        self.assertTrue(all(c['satisfied'] and c['matched_witness_count']==1 for c in matched))
        self.assertEqual({length for c in matched for length in c['actual_observation_lengths']},{2,3})
    def test_shared_pointer_does_not_supply_wrong_pair_or_foreign_family(self):
        items=deepcopy(self.inputs)
        claims=items[0]['family_control_claims']
        claims[-1]['pair_sample_ids']=['v2fixture-12-A-000-forward','v2fixture-12-B-000-forward']
        claims[-2]['pair_sample_ids']=['v2fixture-13-E-000-forward','v2fixture-13-G-000-forward']
        report=v2.check_batch(items)
        self.assertTrue(any('missing actual scoped claim witness' in e for e in report['errors']))
        self.assertTrue(any('not actual same family' in e for e in report['errors']))
    def test_probe_id_conflicts_remain_execution_bound(self):
        items=deepcopy(self.inputs)
        for i,name in enumerate(('world_order_invariance','world_id_rename_equivariance')):
            items[0]['family_control_claims'].append({'name':name,'scope':'audit_probe','control_id':'independent-probe-claim-'+str(i),'probe_id':'shared-incompatible-execution-id'})
        report=v2.check_batch(items)
        self.assertTrue(any('declared probe ID conflicts with executed property' in e for e in report['errors']))
        self.assertTrue(any('missing actual scoped claim witness' in e for e in report['errors']))
    def test_batch_descriptive_key_scope_conflict_fails(self):
        items=deepcopy(self.inputs);claim=deepcopy(items[0]['family_control_claims'][-1]);claim.update(scope='audit_probe',probe_id='new-probe-id');del claim['pair_sample_ids'];items[1]['family_control_claims'].append(claim)
        report=v2.check_batch(items);self.assertTrue(any('conflicting family claim identifier' in e for e in report['errors']))


if __name__=='__main__':unittest.main()
