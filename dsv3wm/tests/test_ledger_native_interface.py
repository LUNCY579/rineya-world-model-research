"""Pure/mocked native interfaces; does not import torch or load any model."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_ledger_recovery import (LedgerConfig, normalize_adapter_config, canonical_adapter_config,
                                   qualified_baseline, bind_qualification)
from evaluate_ledger_recovery import LedgerNativePredictor, generate_dynamics, score_dynamics
from dsv3wm.ledger_recovery import T2_SHA, T2_CONFIG_SHA, ledger_rules


def event(tick):
    return {"account": "A", "amount": 2, "kind": "add", "other": "B", "status": "posted", "tick": tick}


def row():
    return {"parent_world_id": "mock-parent", "domain": "ledger", "rules": ledger_rules(),
            "s0": {"A": 1, "B": 2, "C": 3, "D": 4}, "actions": [event(i) for i in range(4)],
            "provenance": {"cohort": "canonical_id"}, "labels": {"states": [{"A": 999, "B": 2, "C": 3, "D": 4}] * 4}}


class LedgerNativeInterfaceTests(unittest.TestCase):
    def test_recipe_is_typed_and_fixed(self):
        self.assertEqual(LedgerConfig().generation_tokens, 48)
        self.assertEqual(LedgerConfig(arm="A5", steps=96).horizons, (1,))
        self.assertEqual(LedgerConfig(arm="A2").horizons, (1, 2, 4))
        for kwargs in ({"arm": "A3"}, {"steps": 4}, {"rank": 4}, {"legality_weight": 0.1},
                       {"generation_tokens": 32}, {"shuffled_aux": True}):
            with self.assertRaises(ValueError):
                LedgerConfig(**kwargs)

    def test_adapter_normalization_preserves_behavior_but_ignores_order_metadata(self):
        original = {"r": 16, "lora_alpha": 32, "lora_dropout": .05,
                    "target_modules": ["q_proj", "k_proj"], "inference_mode": True, "peft_version": "old"}
        reordered = {**original, "target_modules": {"k_proj", "q_proj"}, "inference_mode": False, "peft_version": "new"}
        self.assertEqual(normalize_adapter_config(original), normalize_adapter_config(reordered))
        self.assertNotEqual(normalize_adapter_config(original), normalize_adapter_config({**reordered, "lora_dropout": 0.0}))

    def test_adapter_comparison_rejects_type_aliases_and_binds_auto_mapping(self):
        original = {"r": 16, "lora_alpha": 32, "fan_in_fan_out": False, "auto_mapping": None,
                    "target_modules": ["q_proj", "k_proj"]}
        for key, value in (("r", 16.0), ("lora_alpha", 32.0), ("fan_in_fan_out", 0), ("auto_mapping", {})):
            self.assertNotEqual(canonical_adapter_config(original), canonical_adapter_config({**original, key: value}))
        small = {"scale": 1, "target_modules": ["q_proj"]}
        self.assertNotEqual(canonical_adapter_config(small), canonical_adapter_config({**small, "scale": True}))
        self.assertEqual(canonical_adapter_config(original), canonical_adapter_config(
            {**original, "target_modules": ["k_proj", "q_proj", "q_proj"], "revision": "metadata-only"}))
        for modules in ("q_proj", ["q_proj", 0], [False], {"q_proj", 0}):
            with self.assertRaises(ValueError):
                canonical_adapter_config({**original, "target_modules": modules})

    def test_new_training_freeze_binds_b0_result_traces_and_original_freeze_reference(self):
        from dsv3wm.native import sha256
        original_freeze = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dev, result_path = directory / "dev.jsonl", directory / "B0.json"
            trace_path, freeze_path = result_path.with_suffix(".traces.json"), directory / "TRAIN_FREEZE.json"
            dev.write_text("fixed DEV bytes", encoding="utf-8")
            trace_path.write_text("[]", encoding="utf-8")
            result = {"status": "development-evaluated", "baseline_only": True, "baseline_qualified": True,
                      "adapter_sha256": T2_SHA, "adapter_config_sha256": T2_CONFIG_SHA,
                      "freeze_sha256": original_freeze, "data_sha256": sha256(dev), "traces_sha256": sha256(trace_path),
                      "summary": {"canonical_id": {"1": {"n": 12, "rollout_correct": 10, "rollout_malformed": 1}}}}
            result_path.write_text(json.dumps(result), encoding="utf-8")
            frozen = {"qualification_freeze_sha256": original_freeze,
                      "files": {str(p.resolve()): sha256(p) for p in (result_path, trace_path, dev)}}
            freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
            self.assertNotEqual(sha256(freeze_path), original_freeze)
            loaded, reference = bind_qualification(freeze_path, result_path, dev)
            self.assertEqual(loaded, result); self.assertEqual(reference, original_freeze)
            # An old result cannot authorize training through an unbound path.
            no_result = {**frozen, "files": {k: v for k, v in frozen["files"].items() if k != str(result_path.resolve())}}
            freeze_path.write_text(json.dumps(no_result), encoding="utf-8")
            with self.assertRaises(ValueError):
                bind_qualification(freeze_path, result_path, dev)
            no_trace = {**frozen, "files": {k: v for k, v in frozen["files"].items() if k != str(trace_path.resolve())}}
            freeze_path.write_text(json.dumps(no_trace), encoding="utf-8")
            with self.assertRaises(ValueError):
                bind_qualification(freeze_path, result_path, dev)
            # A changed B0 result or trace fails its bound identity.
            freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
            result_path.write_text(json.dumps({**result, "baseline_qualified": False}), encoding="utf-8")
            with self.assertRaises(ValueError):
                bind_qualification(freeze_path, result_path, dev)
            result_path.write_text(json.dumps(result), encoding="utf-8")
            trace_path.write_text("[\"changed\"]", encoding="utf-8")
            with self.assertRaises(ValueError):
                bind_qualification(freeze_path, result_path, dev)
            # Even recomputed file hashes cannot substitute another B0 freeze.
            trace_path.write_text("[]", encoding="utf-8")
            freeze_path.write_text(json.dumps({**frozen, "qualification_freeze_sha256": "b" * 64}), encoding="utf-8")
            with self.assertRaises(ValueError):
                bind_qualification(freeze_path, result_path, dev)

    def test_qualification_binds_original_weights_freeze_and_dev(self):
        baseline = {"status": "development-evaluated", "baseline_only": True, "baseline_qualified": True,
                    "adapter_sha256": T2_SHA, "adapter_config_sha256": T2_CONFIG_SHA,
                    "freeze_sha256": "freeze", "data_sha256": "dev",
                    "summary": {"canonical_id": {"1": {"n": 12, "rollout_correct": 10, "rollout_malformed": 1}}}}
        self.assertTrue(qualified_baseline(baseline, "freeze", "dev"))
        for key, value in (("data_sha256", "other"), ("adapter_sha256", "other"), ("baseline_only", False)):
            self.assertFalse(qualified_baseline({**baseline, key: value}, "freeze", "dev"))
        invalid = copy.deepcopy(baseline); invalid["summary"]["canonical_id"]["1"]["rollout_malformed"] = 2
        self.assertFalse(qualified_baseline(invalid, "freeze", "dev"))
        for key, value in (("n", 12.0), ("rollout_correct", 10.0), ("rollout_malformed", True)):
            invalid = copy.deepcopy(baseline); invalid["summary"]["canonical_id"]["1"][key] = value
            self.assertFalse(qualified_baseline(invalid, "freeze", "dev"))

    def test_generation_uses_own_states_and_tagged_phases_without_labels(self):
        seen = []
        class Mock:
            def predict_batch(self, inputs, contexts):
                seen.append((copy.deepcopy(inputs), copy.deepcopy(contexts)))
                self_outer.assertTrue(all(set(item) == {"domain", "rules", "s0", "actions"} for item in inputs))
                return [{"state": {**item["s0"], "A": item["s0"]["A"] + 7}} for item in inputs]
        self_outer = self
        values, direct = generate_dynamics(Mock(), [row()], baseline_only=True)
        self.assertEqual([batch[0][0]["s0"]["A"] for batch in seen[:4]], [1, 8, 15, 22])
        self.assertEqual(values[0][-1]["state"]["A"], 29)
        self.assertEqual(set(direct), {1})
        self.assertEqual([contexts[0]["step"] for _, contexts in seen], [1, 2, 3, 4, 1])
        self.assertEqual(seen[-1][1][0]["phase"], "direct")

    def test_malformed_stops_without_gold_reset_or_flag_accuracy_claim(self):
        class Mock:
            def predict_batch(self, inputs, contexts):
                return [{"malformed": True} for _ in inputs]
        values, direct = generate_dynamics(Mock(), [row()], baseline_only=True)
        self.assertEqual(len(values[0]), 1)
        scored = score_dynamics([row()], values, direct)
        self.assertEqual(scored["summary"]["all"]["4"]["rollout_correct"], 0)
        self.assertEqual(scored["summary"]["all"]["4"]["rollout_attempted"], 0)
        self.assertIn("no learned", scored["flag_metrics_scope"])

    def test_native_wrapper_records_started_trace_and_real_forward_counts(self):
        history = []
        class Model:
            def eval(self): pass
            def register_forward_pre_hook(self, callback, with_kwargs):
                self.callback = callback
                return SimpleNamespace(remove=lambda: history.append("removed"))
        model = Model()
        def sample_crn(loaded, tokenizer, prompts, seeds, max_new_tokens):
            self.assertEqual(history[-1], "started")
            loaded.callback(loaded, (), {"input_ids": SimpleNamespace(numel=lambda: 55)})
            loaded.callback(loaded, (), {"input_ids": SimpleNamespace(numel=lambda: 1)})
            return [{"text": json.dumps(row()["s0"]), "n_tokens": 2, "hit_eos": True}]
        package = ModuleType("bwmn"); package.lm = SimpleNamespace(sample_crn=sample_crn, DUTY_RATIO=3)
        tokenizer = lambda prompt, add_special_tokens: SimpleNamespace(input_ids=[1] * 55)
        predictor = LedgerNativePredictor(model, tokenizer, LedgerConfig(), trace_sink=lambda traces: history.append(traces[-1]["status"]))
        item = {key: row()[key] for key in ("domain", "rules", "s0")}; item["actions"] = [event(1)]
        context = {"parent_world_id": "mock-parent", "phase": "rollout", "step": 1}
        with patch.dict(sys.modules, {"bwmn": package}), patch("dsv3wm.native.boundary"):
            value = predictor.predict_batch([item], [context])[0]
        self.assertEqual(value["state"], row()["s0"])
        self.assertEqual((predictor.calls, predictor.decoder_forwards, predictor.decoder_token_positions), (1, 2, 56))
        self.assertEqual(history, ["started", "removed", "complete"])
        self.assertEqual(predictor.traces[0]["phase"], "rollout")
        self.assertNotIn("labels", predictor.traces[0]["conditioning"])


if __name__ == "__main__":
    unittest.main()
