import unittest
from unittest.mock import patch
from dsv3wm.recovery import legacy_prompt,legacy_parse,batched_rollout,RecoveryConfig
from dsv3wm.domains import _semantics,rules_for


class RecoveryInterfaceTests(unittest.TestCase):
    def test_original_h1_prompt_is_identical(self):
        inputs={'domain':'register','rules':rules_for('register',limit=32),'s0':{'x':3},'actions':['halve']}
        self.assertEqual(legacy_prompt(inputs),_semantics('register').forward_prompt(3,'halve',32))
    def test_multi_input_never_contains_later_state_or_goal(self):
        inputs={'domain':'register','rules':rules_for('register',limit=32),'s0':{'x':3},'actions':['halve','double']}
        prompt=legacy_prompt(inputs)
        self.assertIn('"actions":["halve","double"]',prompt)
        self.assertNotIn('goal',prompt);self.assertNotIn('"states"',prompt)
    def test_parser_strictly_maps_predicted_flags_only(self):
        value=legacy_parse('{"legal":false,"terminated":false,"x":3}')
        self.assertEqual(value,{'state':{'x':3},'legal':False,'terminated':False,'failed':True})
        for text in ('{"legal":true,"terminated":false,"x":true}','{"legal":true,"terminated":false,"x":3,"x":4}',
                     '{"legal":true,"terminated":false,"x":3} extra'):
            self.assertTrue(legacy_parse(text)['malformed'])
    def test_rollout_never_executes_rules_or_teacher_states(self):
        seen=[]
        class Mock:
            def predict_batch(self,inputs):
                seen.append([r['s0']['x'] for r in inputs])
                return [{'state':{'x':r['s0']['x']+7},'legal':True,'terminated':False,'failed':False} for r in inputs]
        row={'domain':'register','s0':{'x':1},'rules':rules_for('register'),'actions':['add1']*4,
             'labels':{'states':[{'x':999}]*4}}
        with patch('dsv3wm.domains.exact_transition',side_effect=AssertionError('oracle forbidden')):
            outputs=batched_rollout(Mock(),[row],4)
        self.assertEqual(seen,[[1],[8],[15],[22]])
        self.assertEqual(outputs[0][-1]['state'],{'x':29})
    def test_recovery_capacity_and_limits_are_explicit(self):
        self.assertEqual(RecoveryConfig().rank,16)
        with self.assertRaises(ValueError):RecoveryConfig(rank=4)


if __name__=='__main__':unittest.main()
