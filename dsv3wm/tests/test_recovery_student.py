import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
from dsv3wm.domains import rules_for, sha256
from dsv3wm.data import make_record
from dsv3wm.planning import verify_plan
from dsv3wm.recovery_student import (StudentConfig, generate_students, parse_registered_plan,
                                     registered_prompt, registered_target, rerank_students,
                                     score_students, shared_orders, validate_corpus, verify_positive,
                                     assert_registered_behavior, assert_adapter_configuration,
                                     fixed_forward_adapter, warm_lineage, admit_readout_lineage,
                                     continuation_slot, combine_student_losses)


class MockForward:
    """Pure fixture, deliberately inaccurate; native efficacy is not tested."""
    def __init__(self, events=None):
        self.traces = []
        self.events = [] if events is None else events

    def token_length(self, inputs):
        return 10

    def __call__(self, inputs):
        self.events.append("forward")
        x, action = inputs["s0"]["x"], inputs["actions"][0]
        if action == "add1":
            x += 1
        elif action == "sub1":
            x -= 1
        value = {"state": {"x": x}, "legal": True, "terminated": action == "finish", "failed": False}
        self.traces.append({"conditioning": copy.deepcopy(inputs),
                            "generated": {"text": "mock", "n_tokens": 3, "hit_eos": True}, "parsed": value})
        return value


class RecoveryStudentTests(unittest.TestCase):
    def setUp(self):
        rules = rules_for("register", limit=32)
        self.source = make_record("register", {"x": 1}, ["halve", "add1", "sub1", "double"], rules, split="train")
        self.task = {"id": "task-fixture", "split": "train", "domain": "register", "rules": rules,
                     "s0": {"x": 1}, "goal": {"kind": "state", "state": {"x": 3}},
                     "parent_world_id": self.source["parent_world_id"],
                     "provenance": {"source_trajectory_id": self.source["id"]}}
        plan = ["add1", "add1", "finish"]
        inputs = {"domain": "register", "rules": rules, "s0": {"x": 1}, "actions": []}
        self.positive = {"split": "train", "domain": "register", "rules": rules, "s0": {"x": 1},
                         "goal": self.task["goal"], "parent_world_id": self.source["parent_world_id"],
                         "source_plan": plan[:-1], "plan": plan, "verified": True,
                         "prompt": registered_prompt(self.task), "target": registered_target(plan),
                         "target_contract": "registered-rw2-actions-final-finish",
                         "verification": verify_plan(inputs, self.task["goal"], plan),
                         "provenance": {"training_input_sha256": sha256(self.source), "task_sha256": sha256(self.task),
                           "selection": "learned-ranked-then-independent-exact", "learned_predicted_goal": True,
                           "all_searches_and_rankings_fixed_before_verification": True,
                           "completion_origin": "deterministic-final-finish-completion-before-verification"}}
        self.dev = {**copy.deepcopy(self.task), "split": "dev"}

    def test_registered_bytes_and_strict_syntax_without_legality_oracle(self):
        from dsv3wm.domains import _semantics
        module = _semantics("register")
        self.assertEqual(registered_prompt(self.task), module.proposer_prompt(1, 3, 32, 6))
        self.assertEqual(registered_target(self.positive["plan"]), module.target_actions(self.positive["plan"]))
        with patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no oracle")):
            self.assertFalse(parse_registered_plan('{"actions":["halve","finish"]}')["malformed"])
            for text in ['["add1","finish"]', '{"actions":[]}', '{"actions":["add1"]}',
                         '{"actions":["finish","add1"]}', '{"actions":["teleport","finish"]}',
                         '{"actions":["finish"],"actions":["add1","finish"]}',
                         '{"actions":[true,"finish"]}', '```json\n{"actions":["finish"]}\n```']:
                self.assertTrue(parse_registered_plan(text)["malformed"])

    def test_train_only_corpus_reverification_and_tamper_fail_closed(self):
        self.assertTrue(verify_positive(self.positive, [self.source], [self.task]))
        gate = validate_corpus([self.positive], [self.source], [self.task], require_gate=False)
        self.assertFalse(gate["passed"])
        with self.assertRaises(ValueError):
            validate_corpus([self.positive], [self.source], [self.task])
        for field, value in [("split", "dev"), ("target", '["add1","add1","finish"]'),
                             ("plan", ["sub1", "sub1", "finish"]), ("source_plan", ["double"]),
                             ("parent_world_id", "forged-parent")]:
            changed = copy.deepcopy(self.positive)
            changed[field] = value
            self.assertFalse(verify_positive(changed, [self.source], [self.task]), field)
        changed = copy.deepcopy(self.positive)
        changed["provenance"]["learned_predicted_goal"] = False
        self.assertFalse(verify_positive(changed, [self.source], [self.task]))
        self.assertFalse(verify_positive(self.positive, [], [self.task]))

    def test_all_generation_then_forward_then_exact_and_no_top1_fallback(self):
        events, texts = [], iter([registered_target(["sub1", "sub1", "finish"]),
                                 registered_target(["add1", "add1", "finish"])] + ["bad", "bad"])
        seeds = []
        def generator(prompt, seed, cap):
            events.append("generate")
            seeds.append(seed)
            self.assertEqual(prompt, self.positive["prompt"])
            self.assertEqual(cap, 80)
            return {"text": next(texts), "n_tokens": 5, "model_calls": 1, "input_tokens": 10,
                    "decoder_token_positions": 15}
        with patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no oracle")):
            generated = generate_students([self.dev], generator, base_seed=3)
            reranked = rerank_students([self.dev], generated, MockForward(events))
        self.assertIsNone(seeds[0])
        self.assertEqual(len(set(seeds[1:])), 3)
        self.assertEqual(reranked[0]["selected_index"], 1)
        self.assertLessEqual(reranked[0]["forward_queries"], 26)
        self.assertLessEqual(reranked[0]["total_model_queries"], 30)
        def logged_verifier(*args):
            events.append("verify")
            return verify_plan(*args)
        scored = score_students([self.dev], generated, reranked, verifier=logged_verifier)[0]
        self.assertEqual(events[:4], ["generate"] * 4)
        self.assertTrue(all(event == "forward" for event in events[4:-2]))
        self.assertEqual(events[-2:], ["verify"] * 2)
        self.assertFalse(scored["top1_success"])
        self.assertTrue(scored["pass_at_4"])
        self.assertEqual(scored["selected_plan"], ["sub1", "sub1", "finish"])
        self.assertTrue(scored["learned_reranking"]["success"])
        self.assertFalse(scored["first_action_optimal"])

    def test_seed_and_shared_order_reproducibility(self):
        def generator(prompt, seed, cap):
            action = "add1" if seed is None else random.Random(seed).choice(["add1", "sub1"])
            return {"text": registered_target([action, "finish"]), "n_tokens": 4}
        self.assertEqual(generate_students([self.dev], generator, base_seed=9),
                         generate_students([self.dev], generator, base_seed=9))
        self.assertEqual(shared_orders(128, 20, seed=2), shared_orders(128, 20, seed=2))
        self.assertEqual(StudentConfig(arm="A3").warm_arm, "A2")
        self.assertEqual(StudentConfig(arm="A4", plan_weight=0, control_extra_one_step=True).warm_arm, "A0")
        with self.assertRaises(ValueError):
            StudentConfig(arm="A3", steps=64)
        with self.assertRaises((ValueError, PermissionError)):
            generate_students([{**self.dev, "split": "confirmation"}], generator)

    def test_no_termination_cannot_count_as_student_success(self):
        generated = generate_students([self.dev], lambda *args: {"text": self.positive["target"], "n_tokens": 5})
        def fake_success(*args):
            return {"success": True, "all_legal": True, "labels": {"terminated": [False, False, False]}}
        result = score_students([self.dev], generated, verifier=fake_success)[0]
        self.assertFalse(result["top1_success"])
        self.assertFalse(result["pass_at_4"])

    def test_failed_corpus_admission_precedes_runtime_and_native_load(self):
        path = Path(__file__).resolve().parents[1] / "scripts/train_recovery_student.py"
        spec = importlib.util.spec_from_file_location("student_admission_test", path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text('{"arm":"A3"}', encoding="utf-8")
            (directory / "manifest.json").write_text('{"output_sha256":{}}', encoding="utf-8")
            (directory / "tasks.jsonl").write_text(json.dumps(self.task) + "\n", encoding="utf-8")
            args = ["train_recovery_student.py", "--config", str(directory / "config.json"),
                    "--train", str(directory / "train.jsonl"), "--tasks", str(directory / "tasks.jsonl"),
                    "--teacher-dir", str(directory), "--warm-dir", str(directory / "warm"),
                    "--qualification", str(directory / "qualification.json"), "--freeze", str(directory / "freeze.json"),
                    "--out", str(directory / "output")]
            with patch.object(sys, "argv", args), patch("dsv3wm.native.verify_freeze", return_value="mock-freeze"), \
                 patch("dsv3wm.provenance.verify_bound_inputs", return_value=None), \
                 patch("dsv3wm.data.load_records", return_value=[self.source]), \
                 patch("dsv3wm.recovery_student.load_teacher_corpus", side_effect=ValueError("unverified corpus")), \
                 patch("dsv3wm.native.load") as load, patch("dsv3wm.native.runtime") as runtime:
                with self.assertRaisesRegex(ValueError, "unverified corpus"):
                    script.main()
                load.assert_not_called()
                runtime.assert_not_called()
            self.assertFalse((directory / "output").exists())

    def test_every_peft_behavior_field_fails_closed(self):
        reference = {"r": 16, "lora_alpha": 32, "lora_dropout": .05, "target_modules": ["q_proj"],
                     "use_rslora": False, "use_dora": False, "rank_pattern": {}, "alpha_pattern": {},
                     "new_reference_field": None, "base_model_name_or_path": "pinned elsewhere", "inference_mode": True}
        metadata = {**copy.deepcopy(reference), "base_model_name_or_path": "metadata-only", "revision": "r1",
                    "peft_version": "changed", "inference_mode": False}
        assert_registered_behavior(metadata, reference)
        for key, value in [("use_rslora", True), ("use_dora", True), ("rank_pattern", {"q_proj": 8}),
                           ("alpha_pattern", {"q_proj": 16}), ("unknown_behavior", False),
                           ("new_reference_field", {"changed": True}), ("use_dora", 0)]:
            changed = copy.deepcopy(reference)
            changed[key] = value
            with self.subTest(field=key, value=value), self.assertRaisesRegex(ValueError, "PEFT behavior"):
                assert_registered_behavior(changed, reference)
        missing = copy.deepcopy(reference)
        del missing["alpha_pattern"]
        with self.assertRaises(ValueError):
            assert_registered_behavior(missing, reference)

    def test_saved_config_hash_prevents_silent_metadata_refreeze(self):
        reference = {"r": 16, "use_dora": False, "inference_mode": True}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, ref = directory / "config.json", directory / "reference.json"
            path.write_text(json.dumps(reference), encoding="utf-8")
            ref.write_text(json.dumps(reference), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert_adapter_configuration(path, reference_path=ref, recorded_sha256=digest)
            path.write_text(json.dumps({**reference, "inference_mode": False}), encoding="utf-8")
            # Metadata-only differences are acceptable without a saved-byte claim.
            assert_adapter_configuration(path, reference_path=ref)
            with self.assertRaisesRegex(ValueError, "hash lineage"):
                assert_adapter_configuration(path, reference_path=ref, recorded_sha256=digest)

    def test_peft_target_module_set_permutation_is_accepted(self):
        reference = {"r": 16, "target_modules": ["q_proj", "v_proj", "o_proj"], "use_dora": False}
        reordered = {**reference, "target_modules": ["o_proj", "q_proj", "v_proj", "q_proj"]}
        assert_registered_behavior(reordered, reference)
        for targets in (["q_proj", "v_proj"], ["q_proj", "v_proj", "k_proj"], ["q_proj", True, "o_proj"]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                assert_registered_behavior({**reference, "target_modules": targets}, reference)
        with self.assertRaises(ValueError):
            assert_registered_behavior({**reordered, "use_dora": 0}, reference)

    def test_warm_lineage_without_saved_config_hash_checks_full_behavior(self):
        from dsv3wm.recovery import INITIAL_F, INITIAL_F_SHA
        from dsv3wm.native import EXPECTED_CORE_SHA
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            adapter = directory / "adapter"
            adapter.mkdir()
            weights, train = adapter / "adapter_model.safetensors", directory / "train.jsonl"
            weights.write_bytes(b"pure test weight fixture")
            train.write_bytes(b"pure test source fixture")
            reference = json.loads((INITIAL_F / "adapter_config.json").read_text(encoding="utf-8"))
            config = adapter / "adapter_config.json"
            config.write_text(json.dumps(reference), encoding="utf-8")
            record = {"status": "pilot-trained", "config": {"arm": "A2"}, "optimizer_steps": 32,
                      "initial_adapter_sha256": INITIAL_F_SHA, "core_sha256": EXPECTED_CORE_SHA,
                      "data_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                      "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
            (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
            warm_lineage(directory, StudentConfig(arm="A3"), train)
            reference["use_rslora"] = True
            config.write_text(json.dumps(reference), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "PEFT behavior"):
                warm_lineage(directory, StudentConfig(arm="A3"), train)

    def test_forward_switch_restores_student_on_success_and_failure(self):
        class Model:
            def __init__(self):
                self.active_adapter = "student"
                self.peft_config = {"student": {}}
                self.events = []
            def load_adapter(self, path, adapter_name, is_trainable):
                self.events.append(("load", adapter_name, is_trainable))
                self.peft_config[adapter_name] = {}
            def set_adapter(self, name):
                self.events.append(("set", name))
                self.active_adapter = name
        for fail in (False, True):
            model = Model()
            try:
                with fixed_forward_adapter(model, "fixture-B0") as switch:
                    self.assertEqual(model.active_adapter, "h3_fixed_B0")
                    self.assertEqual(switch["student_adapter"], "student")
                    if fail:
                        raise ValueError("fixture inference failed")
            except ValueError:
                self.assertTrue(fail)
            self.assertEqual(model.active_adapter, "student")
            self.assertEqual(model.events[-1], ("set", "student"))

    def test_explicit_warm_baseline_preserves_original_training_identity(self):
        from dsv3wm.recovery import INITIAL_F, INITIAL_F_SHA, RecoveryConfig
        from dsv3wm.native import EXPECTED_CORE_SHA
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            adapter = directory / "adapter"
            adapter.mkdir()
            weights, train = adapter / "adapter_model.safetensors", directory / "train.jsonl"
            weights.write_bytes(b"pure warm fixture")
            train.write_bytes(b"pure source TRAIN fixture")
            reference = json.loads((INITIAL_F / "adapter_config.json").read_text(encoding="utf-8"))
            (adapter / "adapter_config.json").write_text(json.dumps(reference), encoding="utf-8")
            config, freeze, qualification = [directory / name for name in ("config.json", "training-freeze.json", "qualification.json")]
            expanded = RecoveryConfig(arm="A2").dict()
            config.write_text(json.dumps(expanded), encoding="utf-8")
            freeze.write_text('{"files":{}}', encoding="utf-8")
            qualification.write_text('{"baseline_qualified":true}', encoding="utf-8")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            record = {"status": "pilot-trained", "config": expanded, "optimizer_steps": 32,
                      "initial_adapter_sha256": INITIAL_F_SHA, "core_sha256": EXPECTED_CORE_SHA,
                      "data_sha256": digest(train), "adapter_sha256": digest(weights),
                      "config_sha256": digest(config), "freeze_sha256": digest(freeze),
                      "qualification_sha256": digest(qualification),
                      "command": ["train_recovery.py", "--config", str(config), "--train", str(train),
                                  "--freeze", str(freeze), "--qualification", str(qualification)]}
            run = directory / "run.json"
            run.write_text(json.dumps(record), encoding="utf-8")
            lineage, extra, kind = admit_readout_lineage(adapter, run, StudentConfig(arm="A3"),
                                                        warm_baseline=True, train_path=train)
            self.assertEqual(kind, "warm-parent-proposer-baseline")
            self.assertEqual(lineage["status"], "pilot-trained")
            self.assertEqual(lineage["config"]["arm"], "A2")
            self.assertEqual(set(extra), {config, train, freeze, qualification})
            with self.assertRaises(ValueError):
                admit_readout_lineage(adapter, run, StudentConfig(arm="A3"))
            with self.assertRaises(ValueError):
                admit_readout_lineage(adapter, run, StudentConfig(arm="A3"), warm_baseline=True)
            with self.assertRaises(ValueError):
                admit_readout_lineage(adapter, run, StudentConfig(arm="A4"), warm_baseline=True, train_path=train)
            freeze.write_text('{"files":{},"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training-freeze"):
                admit_readout_lineage(adapter, run, StudentConfig(arm="A3"), warm_baseline=True, train_path=train)

    def test_student_config_rejects_numeric_booleans(self):
        for override in ({"student_temperature": True}, {"student_k": True}, {"plan_weight": True}, {"steps": True}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                StudentConfig(arm="A3", **override)

    def test_control_requires_explicit_real_extra_h1_slot(self):
        from dsv3wm.data import conditioning
        from dsv3wm.recovery import legacy_prompt
        from dsv3wm.recovery_data import legacy_target
        for override in ({"plan_weight": 0}, {"control_extra_one_step": True},
                         {"plan_weight": 0, "control_extra_one_step": 1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                StudentConfig(arm="A3", **override)
        cfg = StudentConfig(arm="A3", plan_weight=0, control_extra_one_step=True)
        with patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no oracle")):
            prompt, target, kind = continuation_slot(self.source, self.positive, cfg)
        self.assertEqual(kind, "extra_one_step")
        self.assertEqual(prompt, legacy_prompt(conditioning(self.source, 1)))
        self.assertEqual(target, json.dumps(legacy_target(self.source, 1), sort_keys=True, separators=(",", ":")))
        study = continuation_slot(self.source, self.positive, StudentConfig(arm="A3"))
        self.assertEqual(study, (self.positive["prompt"], self.positive["target"], "plan"))
        self.assertEqual(cfg.steps * (len(cfg.horizons) + 1), 128)
        c4 = StudentConfig(arm="A4", plan_weight=0, control_extra_one_step=True)
        self.assertEqual(c4.steps * (len(c4.horizons) + 1), 64)
        self.assertEqual(shared_orders(128, 20, seed=cfg.seed), shared_orders(128, 20, seed=c4.seed))

    def test_control_slot_has_unit_gradient_not_discarded_compute(self):
        import torch
        for arm in ("A3", "A4"):
            cfg = StudentConfig(arm=arm, plan_weight=0, control_extra_one_step=True)
            dynamics = {horizon: torch.tensor(float(horizon), device="cpu", requires_grad=True) for horizon in cfg.horizons}
            slot = torch.tensor(3., device="cpu", requires_grad=True)
            loss = combine_student_losses(dynamics, slot, cfg)
            loss.backward()
            self.assertEqual(float(slot.grad), 1.)
            self.assertEqual(float(dynamics[1].grad), 1.)
            if arm == "A3":
                self.assertEqual(float(dynamics[2].grad), .5)
                self.assertEqual(float(dynamics[4].grad), .5)
                self.assertEqual(float(loss.detach()), 7.)
            else:
                self.assertEqual(float(loss.detach()), 4.)


if __name__ == "__main__":
    unittest.main()
