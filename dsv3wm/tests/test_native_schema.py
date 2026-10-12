import unittest
from dsv3wm.native import parse_prediction


class NativeSchemaTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {'domain':'register','rules':{'limit':32},'s0':{'x':0},'actions':['add1']}
    def test_rejects_numeric_bool_and_float(self):
        for value in ('true','1.0','NaN','Infinity','"1"'):
            output = '{"state":{"x":'+value+'},"legal":true,"terminated":false,"failed":false}'
            self.assertTrue(parse_prediction(output,self.inputs)['malformed'])
    def test_rejects_missing_flags_and_duplicate_keys(self):
        for output in ('{"state":{"x":1},"legal":true}',
                       '{"state":{"x":0,"x":1},"legal":true,"terminated":false,"failed":false}',
                       '{"state":{"x":1},"legal":1,"terminated":false,"failed":false}'):
            self.assertTrue(parse_prediction(output,self.inputs)['malformed'])
    def test_strict_full_and_delta(self):
        output = '{"state":{"x":1},"legal":true,"terminated":false,"failed":false}'
        self.assertEqual(parse_prediction(output,self.inputs)['state'], {'x':1})
        delta = output.replace('"state"','"delta"')
        self.assertEqual(parse_prediction(delta,self.inputs,'delta')['state'], {'x':1})


if __name__=='__main__':
    unittest.main()
