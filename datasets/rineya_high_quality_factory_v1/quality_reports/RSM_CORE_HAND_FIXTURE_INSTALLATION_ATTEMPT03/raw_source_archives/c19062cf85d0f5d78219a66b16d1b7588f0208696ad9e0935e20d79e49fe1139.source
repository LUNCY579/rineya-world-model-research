import copy
import unittest
from factory.lineage import cross_release,semantic_fingerprint


class LineageTests(unittest.TestCase):
    def row(self,index=1,split='train'):
        return {'track_id':'RGM-001','generation_seed':20261009,'semantic_parent_id':f'RGM-001-old-p{index:04d}',
                'sample_id':f'r{index}','split':split,'semantic_task':{'kind':'test','test':index}}

    def test_index_relabel_cannot_inflate_new_families(self):
        old=self.row();new=copy.deepcopy(old);new['semantic_parent_id']='RGM-001-new-p0001';new['sample_id']='new'
        result=cross_release([new],[old]);self.assertTrue(result['family_overlap']);self.assertTrue(result['cross_parent_semantic_duplicates'])

    def test_cross_partition_relabel_detected(self):
        old=self.row();new=self.row(split='dev_challenge')
        self.assertTrue(cross_release([new],[old])['split_leakage'])

    def test_actual_new_family_clean(self):
        self.assertFalse(any(cross_release([self.row(31)],[self.row(1)]).values()))

    def test_opaque_id_and_edge_order_do_not_hide_duplicates(self):
        task={'kind':'critic','world':{'kind':'graph','nodes':['B','A'],'edges':[['A','B',1]],'start':'A','goal':'B','fuel':1,'plan':[]},
              'candidates':[{'candidate_id':'c1','plan':['B'],'claimed_after':{'node':'B','fuel':0}}]}
        changed=copy.deepcopy(task);changed['candidates'][0]['candidate_id']='other';changed['world']['nodes'].reverse()
        self.assertEqual(semantic_fingerprint(task),semantic_fingerprint(changed))


if __name__=='__main__':unittest.main()
