"""Pure-mock cache wrapper accounting/schema tests; no native model or lease.

These do not establish native logit/cache parity. The constructed decoder result
fixtures exercise the wrapper around the separately required native parity gate.
"""
import copy
import json
import sys
import types
import unittest
from unittest.mock import patch

from dsv3wm.cached_native import CachedPredictor, chat_prompt


class MockTokenizer:
    def __init__(self, positions=5):
        self.positions = positions
        self.prompts = []

    def __call__(self, prompt, add_special_tokens=False):
        self.prompts.append(prompt)
        return types.SimpleNamespace(input_ids=list(range(self.positions)))


class MockModel:
    def __init__(self):
        self.config = types.SimpleNamespace(num_hidden_layers=2, num_key_value_heads=3, head_dim=4)
        self.evaluations = 0

    def eval(self):
        self.evaluations += 1


class CachedInstrumentTests(unittest.TestCase):
    def setUp(self):
        self.conditioning = {"domain": "register", "rules": {"limit": 32, "text": "add1 adds one"},
                             "s0": {"x": 1}, "actions": ["add1"]}
        self.config = types.SimpleNamespace(device="cpu", target_mode="full", max_length=100,
                                            generation_tokens=64)
        self.model, self.tokenizer = MockModel(), MockTokenizer()
        self.generated = {"text": json.dumps({"state": {"x": 2}, "legal": True,
                                               "terminated": False, "failed": False}),
                          "n_tokens": 3, "hit_eos": False}
        self.invocations = []
        self.lm = types.ModuleType("bwmn.lm")
        self.lm.DUTY_RATIO = 3.0
        def sample_crn(model, tokenizer, prompts, seeds, *, max_new_tokens):
            self.invocations.append({"prompts": prompts, "seeds": seeds, "limit": max_new_tokens})
            self.assertEqual(seeds, [None])
            return [copy.deepcopy(self.generated)]
        self.lm.sample_crn = sample_crn
        package = types.ModuleType("bwmn")
        package.lm = self.lm
        self.modules = patch.dict(sys.modules, {"bwmn": package, "bwmn.lm": self.lm})
        self.modules.start()
        self.boundaries = patch("dsv3wm.cached_native.boundary", return_value=None)
        self.boundaries.start()
        self.addCleanup(self.boundaries.stop)
        self.addCleanup(self.modules.stop)

    def test_plain_prompt_is_identical_and_label_free(self):
        from dsv3wm.native import response_prompt
        predictor = CachedPredictor(self.model, self.tokenizer, self.config)
        with patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no labels")):
            result = predictor(self.conditioning)
        self.assertEqual(self.invocations[0]["prompts"], [response_prompt(self.conditioning, "full")])
        self.assertEqual(result["state"], {"x": 2})
        self.assertEqual(set(predictor.traces[0]["conditioning"]), {"domain", "rules", "s0", "actions"})
        self.assertNotIn("labels", self.invocations[0]["prompts"][0])
        self.assertNotIn("goal", self.invocations[0]["prompts"][0])

    def test_cap_includes_unused_final_forward_and_resident_kv_only(self):
        predictor = CachedPredictor(self.model, self.tokenizer, self.config)
        predictor(self.conditioning)
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(predictor.input_tokens, 5)
        self.assertEqual(predictor.output_tokens, 3)
        self.assertEqual(predictor.decoder_forwards, 4)
        self.assertEqual(predictor.decoder_token_positions, 8)
        self.assertEqual(predictor.kv_cache_peak_bytes_estimate, 1536)

    def test_eos_stops_without_extra_forward_and_includes_eos_token(self):
        self.generated["hit_eos"] = True
        predictor = CachedPredictor(self.model, self.tokenizer, self.config)
        predictor(self.conditioning)
        self.assertEqual(predictor.output_tokens, 3)
        self.assertEqual(predictor.decoder_forwards, 3)
        self.assertEqual(predictor.decoder_token_positions, 7)
        self.assertEqual(predictor.kv_cache_peak_bytes_estimate, 1344)
        self.generated["n_tokens"] = 1
        first_eos = CachedPredictor(self.model, self.tokenizer, self.config)
        first_eos(self.conditioning)
        self.assertEqual(first_eos.decoder_forwards, 1)
        self.assertEqual(first_eos.decoder_token_positions, 5)

    def test_position_budget_is_conservative_and_refusal_does_not_call_model(self):
        predictor = CachedPredictor(self.model, self.tokenizer, self.config, max_decoder_token_positions=8)
        predictor(self.conditioning)
        self.assertEqual(self.invocations[0]["limit"], 3)
        self.assertEqual(predictor.decoder_token_positions, 8)
        rejected = predictor(self.conditioning)
        self.assertEqual(rejected["error"], "decoder-token-budget-exhausted")
        self.assertTrue(predictor.budget_exhausted)
        self.assertEqual(len(self.invocations), 1)
        self.assertEqual(predictor.calls, 1)

    def test_overlength_skips_decoder(self):
        self.tokenizer.positions = 101
        predictor = CachedPredictor(self.model, self.tokenizer, self.config)
        self.assertEqual(predictor(self.conditioning)["error"], "overlength")
        self.assertEqual(predictor.calls, 0)
        self.assertEqual(self.invocations, [])

    def test_malformed_schema_cannot_be_scored_as_numeric_equal(self):
        predictor = CachedPredictor(self.model, self.tokenizer, self.config)
        examples = [
            '{"state":{"x":2.0},"legal":true,"terminated":false,"failed":false}',
            '{"state":{"x":2},"legal":1,"terminated":false,"failed":false}',
            '{"state":{"x":2},"legal":true}',
            '{"state":{"x":2,"x":3},"legal":true,"terminated":false,"failed":false}',
            '{"state":{"x":NaN},"legal":true,"terminated":false,"failed":false}',
            '```json\n{"state":{"x":2},"legal":true,"terminated":false,"failed":false}\n```',
        ]
        for text in examples:
            with self.subTest(text=text):
                self.generated["text"] = text
                self.assertTrue(predictor(self.conditioning)["malformed"])
        self.assertEqual(predictor.calls, len(examples))
        self.assertEqual(len(predictor.traces), len(examples))

    def test_chat_is_explicit_format_change_without_transition_execution(self):
        with patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no labels")):
            prompt = chat_prompt(self.conditioning, "full")
        self.assertIn('"x":1', prompt)
        self.assertIn('"actions":["add1"]', prompt)
        self.assertNotIn('"x":2', prompt)
        self.assertNotIn("labels", prompt)
        self.assertNotIn("goal", prompt)
        with self.assertRaises(ValueError):
            CachedPredictor(self.model, self.tokenizer, self.config, prompt_style="automatic")


if __name__ == "__main__":
    unittest.main()
