import unittest
from dsv3wm.evaluation import free_rollout


class RolloutTests(unittest.TestCase):
    def test_uses_own_state_only(self):
        seen = []
        def predictor(inputs):
            self.assertEqual(set(inputs), {'domain','rules','s0','actions'})
            seen.append(inputs['s0']['x'])
            return {'state':{'x':inputs['s0']['x']+7}, 'legal':True, 'terminated':False}
        initial = {'domain':'register', 'rules':{}, 's0':{'x':1}, 'actions':['add1']*4}
        result = free_rollout(predictor, initial, 4)
        self.assertEqual(seen, [1,8,15,22])
        self.assertEqual(result['state'], {'x':29})
        self.assertEqual(initial['s0'], {'x':1})

    def test_malformed_no_fallback(self):
        result = free_rollout(lambda _: {'malformed':True}, {'domain':'register','rules':{},'s0':{'x':1},'actions':['double']*4}, 4)
        self.assertTrue(result['malformed'])
        self.assertIsNone(result['state'])
        self.assertEqual(len(result['outputs']), 1)

    def test_predicted_termination_is_absorbing(self):
        calls = []
        def predict(inputs):
            calls.append(inputs)
            return {'state':{'x':2},'legal':True,'terminated':True}
        result = free_rollout(predict, {'domain':'register','rules':{},'s0':{'x':1},'actions':['finish','double']}, 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['state'], {'x':2})


if __name__ == '__main__':
    unittest.main()
