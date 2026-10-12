"""Singleton H3 admission/geometry/provenance with CPU mocks; no native model."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from dsv3wm import native, native_cpu11, native_cpu105_readout as readout
from dsv3wm.recovery import LegacyPredictor
from dsv3wm.recovery_student import StudentConfig, generate_students, rerank_students
from dsv3wm.proposer import NativePlanGenerator
from dsv3wm.domains import rules_for, parent_world_key
import freeze_recovery_student_cpu105_readout as freezer
import evaluate_recovery_student_guards_cpu105_readout as guard
import test_native_cpu105_training as training_tests
import test_recovery_student_guards as guard_tests

OLD_SHA = {"scripts/evaluate_recovery_student_cpu11.py": "f56dbed907b36960241938e38a8f494e9c729f9852eddb21c97db64b83f3d6b7",
           "scripts/evaluate_recovery_student_guards_cpu11.py": "33f956e7308ec2bf2aec8f3790a993019d5ff575db67d59609a52a4ccde433b8",
           "src/dsv3wm/native_cpu11.py": "075b7fdea2199a5f8f564022e48742b3b3064c58b8bec877ed7c0198d8c4f884"}


def cfg(kind="A3"):
    return StudentConfig.read(ROOT / "configs" / readout.REGISTERED_CONFIGS[kind][0])


def changes(name):
    common = [
        ('from dsv3wm.native_cpu11 import admission_profile, last_admission_check, load',
         'from dsv3wm.native_cpu105_readout import admission_profile, last_admission_check, load, validate_readout_config'),
        ('    cfg = StudentConfig.read(args.config)\n', '    cfg = StudentConfig.read(args.config)\n    validate_readout_config(cfg)\n'),
        ('"native", "native_cpu11", "provenance"', '"native", "native_cpu11", "native_cpu105_readout", "provenance"'),
        ('            model, tokenizer, lm = load(cfg)\n', '            model, tokenizer, lm = load(cfg, readout_only=True)\n')]
    if name.endswith("guards"):
        return common + [
            ('    return [Path(__file__), ROOT / "scripts/evaluate_recovery_student_guards.py",',
             '    return [Path(__file__), ROOT / "scripts/evaluate_recovery_student_guards_cpu11.py",\n'
             '            ROOT / "scripts/freeze_recovery_student_cpu105_readout.py", ROOT / "prereg/RECOVERY_RESOURCE_CLARIFICATION_6.md",\n'
             '            ROOT / "scripts/evaluate_recovery_student_guards.py",'),
            ('    from dsv3wm.recovery import LegacyPredictor, load_warm\n', '    from dsv3wm.recovery import load_warm\n'
             '    from dsv3wm.native_cpu105_readout import SingletonLegacyPredictor, validate_singleton_baseline\n'),
            ('    if baseline is not None:\n        # Refuse wrong baseline', '    if baseline is not None:\n'
             '        validate_singleton_baseline(baseline)\n        # Refuse wrong baseline'),
            ('            predictor = LegacyPredictor(model, tokenizer, cfg)\n',
             '            predictor = SingletonLegacyPredictor(model, tokenizer, cfg)\n'),
            ('                replies = lm.sample_crn(model, tokenizer, prompts, [None] * 6, max_new_tokens=32)\n',
             '                replies = []\n                for prompt in prompts:\n'
             '                    replies.extend(lm.sample_crn(model, tokenizer, [prompt], [None], max_new_tokens=32))\n')]
    return common + [('REPO / "dsv3wm/scripts/evaluate_recovery_student.py"]',
                      'REPO / "dsv3wm/scripts/evaluate_recovery_student.py", REPO / "dsv3wm/scripts/evaluate_recovery_student_cpu11.py",\n'
                      '              REPO / "dsv3wm/scripts/freeze_recovery_student_cpu105_readout.py", REPO / "dsv3wm/prereg/RECOVERY_RESOURCE_CLARIFICATION_6.md"]')]


class ForbiddenTargets(dict):
    def __getitem__(self, key):
        raise AssertionError("No target/label read during generation")


class ReadoutCPU105Tests(unittest.TestCase):
    def test_old_bytes_and_copied_evaluators_allow_only_admission_source_and_geometry_edits(self):
        for relative, digest in OLD_SHA.items():
            self.assertEqual(native.sha256(ROOT / relative), digest)
        for name in ("evaluate_recovery_student", "evaluate_recovery_student_guards"):
            original = (ROOT / "scripts" / (name + "_cpu11.py")).read_text(encoding="utf-8")
            expected = original
            for before, after in changes(name):
                self.assertEqual(expected.count(before), 1); expected = expected.replace(before, after)
            actual = (ROOT / "scripts" / (name + "_cpu105_readout.py")).read_text(encoding="utf-8")
            self.assertEqual(actual, expected)
            def algorithms(source):
                return {node.name: ast.dump(node) for node in ast.walk(ast.parse(source))
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                        and node.name not in {"main", "source_paths", "behavior_generator"}}
            self.assertEqual(algorithms(original), algorithms(actual))
            if name.endswith("guards"):
                self.assertTrue({"TaggedPredictor", "generate_guards", "score_guards", "replay_dynamics", "count_forward", "trace_sink"}
                                <= set(algorithms(actual)))

    def test_load_tail_runtime_boundary_and_old_floors_remain_identical(self):
        def tail(path):
            node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "load")
            start = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                         and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "set_num_threads")
            return ast.dump(ast.Module(body=node.body[start:], type_ignores=[]))
        self.assertEqual(tail(ROOT / "src/dsv3wm/native_cpu11.py"), tail(ROOT / "src/dsv3wm/native_cpu105_readout.py"))
        self.assertIs(readout.runtime, native.runtime); self.assertIs(readout.boundary, native.boundary)
        self.assertEqual(native_cpu11.ADMISSION_PROFILE["load_free_host_ram_floor_gib"], 11)
        fixture = training_tests.CPU105TrainingTests().fixture
        _, modules, _, _ = fixture(10.917744, 18.396648)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "11 GiB"):
            native_cpu11.load(cfg())

    def test_exact_four_config_scope_and_explicit_core_only_readout_admission(self):
        for kind in readout.REGISTERED_CONFIGS:
            self.assertEqual(readout.validate_readout_config(cfg(kind)), kind)
        for candidate in (StudentConfig(arm="A3", seed=1), StudentConfig(arm="A3", ce_chunk=4),
                          StudentConfig(arm="A4", threads=1), SimpleNamespace(**cfg().dict())):
            with self.assertRaises(ValueError): readout.validate_readout_config(candidate)
        for kwargs in ({}, {"readout_only": False}, {"readout_only": 1}, {"readout_only": True, "adapter": "path"}):
            with self.assertRaisesRegex(ValueError, "readout_only=True"):
                readout.load(cfg(), **kwargs)
        with self.assertRaises(TypeError): readout.load(cfg(), training_only=True)
        with patch.object(readout, "sha256", return_value="changed config"), self.assertRaises(ValueError):
            readout.validate_readout_config(cfg())

    def test_physical_commit_admission_and_every_core_pin_before_mock_load(self):
        fixture = training_tests.CPU105TrainingTests().fixture
        for physical, commit in ((10.49, 20.), (11., 10.49)):
            events, modules, digest, _ = fixture(physical, commit)
            with patch.dict(sys.modules, modules), patch.object(readout, "sha256", side_effect=digest), self.assertRaises(RuntimeError):
                readout.load(cfg(), readout_only=True)
            self.assertEqual(events, []); self.assertEqual(readout.last_admission_check()["status"], "refused")
        events, modules, digest, expected = fixture(10.5, 10.5)
        with patch.dict(sys.modules, modules), patch.object(readout, "sha256", side_effect=digest):
            self.assertEqual(readout.load(cfg("C4"), readout_only=True), expected)
        self.assertEqual(readout.last_admission_check()["config_kind"], "C4")
        for name in ("model.safetensors", *native.EXPECTED_SIDECARS):
            events, modules, digest, _ = fixture(changed=name)
            with patch.dict(sys.modules, modules), patch.object(readout, "sha256", side_effect=digest), self.assertRaises(RuntimeError):
                readout.load(cfg(), readout_only=True)
            self.assertFalse(any(event[0] == "core" for event in events))

    def test_original_four_gib_boundary_and_priority_waiter_still_refuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            psutil = ModuleType("psutil"); psutil.virtual_memory = lambda: SimpleNamespace(available=4 * 2**30 - 1)
            with patch.dict(os.environ, {"RINEYA_LANE_DIR": temporary}), patch.dict(sys.modules, {"psutil": psutil}), \
                 self.assertRaisesRegex(RuntimeError, "below 4 GiB"):
                readout.boundary(cfg())
            psutil.virtual_memory = lambda: SimpleNamespace(available=8 * 2**30)
            (Path(temporary) / "rineya-cpu-lane.waiting.bwm").write_text("synthetic waiter")
            with patch.dict(os.environ, {"RINEYA_LANE_DIR": temporary}), patch.dict(sys.modules, {"psutil": psutil}), \
                 self.assertRaisesRegex(RuntimeError, "priority waiter"):
                readout.boundary(cfg())

    def sample_fixture(self):
        calls = []
        def sample(model, tokenizer, prompts, seeds, **kwargs):
            self.assertEqual(len(prompts), 1); self.assertEqual(len(seeds), 1)
            calls.append((prompts[0], seeds[0], kwargs["max_new_tokens"]))
            reply = {"x": int(hashlib.sha256(prompts[0].encode()).hexdigest()[:4], 16), "legal": True, "terminated": False}
            return [{"text": json.dumps(reply), "n_tokens": 2, "hit_eos": True}]
        lm = SimpleNamespace(sample_crn=sample, DUTY_RATIO=1.)
        bwmn = ModuleType("bwmn"); bwmn.lm = lm
        tokenizer = lambda prompt, add_special_tokens: SimpleNamespace(input_ids=[1] * 20)
        return calls, {"bwmn": bwmn}, SimpleNamespace(eval=lambda: None), tokenizer

    def test_singleton_predictor_preserves_order_raw_traces_and_all_174_nominal_queries_without_labels(self):
        fixture = guard_tests.StudentGuardTests()
        rows, fixtures = fixture.rows(), fixture.fixtures()
        for row in rows: row["labels"] = ForbiddenTargets()
        calls, modules, model, tokenizer = self.sample_fixture()
        predictor = readout.SingletonLegacyPredictor(model, tokenizer, cfg())
        def behavior(requests):
            self.assertTrue(all(set(row) == {"id", "domain", "prompt", "messages"} for row in requests))
            for request in requests:
                modules["bwmn"].lm.sample_crn(model, tokenizer, [request["prompt"]], [None], max_new_tokens=32)
            return {"replies": [{"text": "mock", "n_tokens": 2, "hit_eos": True} for _ in requests],
                    "model_queries": 6, "input_tokens": 120, "output_tokens": 12}
        with patch.dict(sys.modules, modules), patch("dsv3wm.recovery.boundary"), \
             patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no oracle")):
            result = guard.generate_guards(rows, fixtures, predictor, behavior)
        self.assertEqual(len(calls), 174); self.assertEqual(predictor.calls, 168)
        self.assertEqual([trace["source_index"] for trace in result["dynamics_traces"][:24]], list(range(24)))
        self.assertEqual([trace["phase"] for trace in result["dynamics_traces"][-72:]], ["direct"] * 72)
        self.assertTrue(result["all_generation_fixed_before_scoring"])
        for index in range(24):
            previous, next_trace = result["dynamics_traces"][index], result["dynamics_traces"][24 + index]
            self.assertEqual(next_trace["conditioning"]["s0"], previous["parsed"]["state"])

    def test_behavior_nested_generator_uses_one_unchanged_sampler_call_per_fixture(self):
        tree = ast.parse((ROOT / "scripts/evaluate_recovery_student_guards_cpu105_readout.py").read_text(encoding="utf-8"))
        function = copy.deepcopy(next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "behavior_generator"))
        calls = []
        def sample(model, tok, prompts, seeds, max_new_tokens):
            self.assertEqual((len(prompts), seeds, max_new_tokens), (1, [None], 32)); calls.extend(prompts)
            return [{"text": "raw-" + prompts[0], "n_tokens": 3, "hit_eos": True}]
        class Tokenizer:
            def __call__(self, prompt, add_special_tokens): return SimpleNamespace(input_ids=[1] * 5)
            def apply_chat_template(self, messages, **kwargs): return messages[0]["content"]
        fixtures = [{"id": str(i), "prompt": str(i), "messages": [{"content": str(i)}]} for i in range(6)]
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.json"; trace.write_text('{"dynamics":[],"behavior":[]}')
            namespace = {"phase": ["dynamics"], "tokenizer": Tokenizer(), "lm": SimpleNamespace(sample_crn=sample),
                         "model": object(), "observed_behavior": [{}], "json": json, "trace_path": trace, "fixtures": fixtures}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "synthetic generator", "exec"), namespace)
            result = namespace["behavior_generator"](fixtures)
            self.assertEqual(calls, list(map(str, range(6)))); self.assertEqual(result["model_queries"], 6)
            self.assertEqual([row["fixture_id"] for row in json.loads(trace.read_text())["behavior"]], calls)

    def test_unchanged_proposer_and_all_fixed_b0_rerank_requests_are_singleton_without_oracle(self):
        rules = rules_for("register", limit=32)
        task = {"id": "mock-task", "split": "dev", "domain": "register", "rules": rules, "s0": {"x": 1},
                "goal": {"kind": "state", "state": {"x": 3}}, "parent_world_id": parent_world_key("register", {"x": 1}, rules), "provenance": {}}
        calls, modules, model, tokenizer = self.sample_fixture()
        def sample(model, tokenizer, prompts, seeds, **kwargs):
            self.assertEqual((len(prompts), len(seeds)), (1, 1)); calls.append(kwargs["max_new_tokens"])
            text = '{"actions":["finish"]}' if kwargs["max_new_tokens"] == 80 else '{"x":1,"legal":true,"terminated":true}'
            return [{"text": text, "n_tokens": 3, "hit_eos": True}]
        modules["bwmn"].lm.sample_crn = sample
        with patch.dict(sys.modules, modules), patch("dsv3wm.native.boundary"), patch("dsv3wm.recovery.boundary"), \
             patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no oracle")):
            proposals = generate_students([task], NativePlanGenerator(model, tokenizer, cfg()))
            forward = LegacyPredictor(model, tokenizer, cfg())
            ranked = rerank_students([task], proposals, forward)
        self.assertEqual(calls, [80] * 4 + [32] * 4)
        self.assertEqual(ranked[0]["forward_queries"], 4); self.assertEqual(ranked[0]["total_model_queries"], 8)
        self.assertEqual(proposals[0]["selected_index"], 0)

    def test_old_warm_guard_refs_refuse_and_new_singleton_metadata_is_required(self):
        record = {"warm_baseline": True, "status": "development-guards-evaluated", "resource_admission_profile": native_cpu11.ADMISSION_PROFILE}
        with self.assertRaisesRegex(ValueError, "New completed singleton"): readout.validate_singleton_baseline(record)
        record["resource_admission_profile"] = readout.ADMISSION_PROFILE
        readout.validate_singleton_baseline(record)
        record["resource_admission_profile"] = {**readout.ADMISSION_PROFILE, "max_sampler_requests_per_call": 2}
        with self.assertRaises(ValueError): readout.validate_singleton_baseline(record)
        record["resource_admission_profile"] = {**readout.ADMISSION_PROFILE, "max_sampler_requests_per_call": True}
        with self.assertRaises(ValueError): readout.validate_singleton_baseline(record)

    def freeze_fixture(self, directory, *, instrument=freezer.READOUT, artifacts=None, bound=()):
        paths = [ROOT / "src/dsv3wm/native.py", ROOT / "src/dsv3wm/native_cpu11.py",
                 ROOT / "scripts/evaluate_recovery_student_cpu11.py", ROOT / "scripts/evaluate_recovery_student_guards_cpu11.py"]
        paths += list(bound)
        record = {"instrument": instrument, "confirmation_access": False,
                  "cpu11_phase_amendment": {"profile": native_cpu11.ADMISSION_PROFILE},
                  "artifacts": artifacts or [{"kind": "A3", "warm_arm": "A2"}],
                  "files": {str(path.resolve()).replace("\\", "/"): native.sha256(path) for path in paths}}
        path = directory / "synthetic-freeze.json"; path.write_text(json.dumps(record), encoding="utf-8")
        return path, record

    def selected_student_fixture(self, directory, kind="A3"):
        directory.mkdir(parents=True, exist_ok=True)
        warm = directory / "warm"; (warm / "adapter").mkdir(parents=True)
        for path in (warm / "run.json", warm / "adapter/adapter_model.safetensors", warm / "adapter/adapter_config.json"):
            path.write_bytes(("synthetic warm artifact " + str(path)).encode())
        selected = directory / "student"; selected.mkdir()
        configuration = cfg(kind)
        proof = {"warm_arm": configuration.warm_arm, "passed": True, "new_result": "synthetic-new-warm.json",
                 "warm_adapter_sha256": native.sha256(warm / "adapter/adapter_model.safetensors"),
                 "warm_adapter_config_sha256": native.sha256(warm / "adapter/adapter_config.json"),
                 "warm_run_sha256": native.sha256(warm / "run.json")}
        student = {"status": "student-pilot-trained", "training_kind": kind, "config": configuration.dict(),
                   "warm_arm": configuration.warm_arm, "command": ["synthetic trainer", "--warm-dir", str(warm)],
                   **{key: proof[key] for key in ("warm_adapter_sha256", "warm_adapter_config_sha256", "warm_run_sha256")}}
        run = selected / "run.json"; run.write_text(json.dumps(student))
        artifact = {"kind": kind, "warm_arm": configuration.warm_arm, "config": configuration.dict(), "directory": str(selected)}
        return artifact, [run, warm / "run.json", warm / "adapter/adapter_model.safetensors", warm / "adapter/adapter_config.json"], proof

    def test_append_only_readout_extension_preserves_pins_and_refuses_training_other_scopes(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            directory = Path(temporary); source, original = self.freeze_fixture(directory)
            before = source.read_bytes(); extended = freezer.extend(source)
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue(original["files"].items() <= extended["files"].items())
            for key in original:
                if key != "files": self.assertEqual(original[key], extended[key])
            self.assertEqual(extended["files"][str(source.resolve()).replace("\\", "/")], native.sha256(source))
            for scope in ("warm-verified-H3-students-controls-v1", "recovery-H3-teacher-v1", "tick-compatible-ledger-development-v1"):
                source.write_text(json.dumps({**original, "instrument": scope}), encoding="utf-8")
                with self.assertRaises(ValueError): freezer.extend(source)
            source.write_text(json.dumps({**original, "cpu105_singleton_readout_amendment": {}}), encoding="utf-8")
            with self.assertRaises(ValueError): freezer.extend(source)

    def test_student_guard_extension_requires_parity_and_warm_stage_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact, paths, report = self.selected_student_fixture(directory / "A3")
            source, _ = self.freeze_fixture(directory, instrument=freezer.GUARDS, artifacts=[artifact], bound=paths)
            with self.assertRaisesRegex(ValueError, "parity-qualified"): freezer.extend(source)
            with patch.object(freezer, "raw_guard_parity", return_value=(report, [])):
                extended = freezer.extend(source, old_warm_guards=["old"], new_warm_guards=["new"])
            self.assertEqual(extended["cpu105_singleton_readout_amendment"]["warm_singleton_raw_parity"], {"A2": report})
            binding = extended["cpu105_singleton_readout_amendment"]["selected_student_warm_bindings"][0]
            self.assertEqual(binding["student_kind"], "A3")
            for field in ("warm_adapter_sha256", "warm_adapter_config_sha256", "warm_run_sha256"):
                self.assertEqual(binding[field], report[field])
            source, _ = self.freeze_fixture(directory, instrument=freezer.GUARDS, artifacts=[{"kind": "warm-A2"}])
            freezer.extend(source)

    def test_self_consistent_parity_wrong_weight_config_or_run_refuses_selected_student(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact, paths, report = self.selected_student_fixture(directory / "A3")
            source, original = self.freeze_fixture(directory, instrument=freezer.GUARDS, artifacts=[artifact], bound=paths)
            before = source.read_bytes()
            for field in ("warm_adapter_sha256", "warm_adapter_config_sha256", "warm_run_sha256"):
                wrong = {**report, field: "0" * 64}
                with self.subTest(field=field), patch.object(freezer, "raw_guard_parity", return_value=(wrong, [])), \
                     self.assertRaisesRegex(ValueError, field):
                    freezer.extend(source, old_warm_guards=["old wrong artifact"], new_warm_guards=["new same wrong artifact"])
                self.assertEqual(source.read_bytes(), before)
            self.assertEqual(json.loads(source.read_text())["files"], original["files"])

    def test_same_warm_arm_parity_must_bind_every_selected_student_not_only_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            a3, a3_paths, a3_proof = self.selected_student_fixture(directory / "A3", "A3")
            c3, c3_paths, _ = self.selected_student_fixture(directory / "C3", "C3")
            source, _ = self.freeze_fixture(directory, instrument=freezer.GUARDS, artifacts=[a3, c3], bound=[*a3_paths, *c3_paths])
            with patch.object(freezer, "raw_guard_parity", return_value=(a3_proof, [])), self.assertRaisesRegex(ValueError, "warm_adapter_sha256"):
                freezer.extend(source, old_warm_guards=["A2 old"], new_warm_guards=["A2 new"])

    def test_student_warm_edge_cannot_be_self_asserted_after_repinning_student_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact, paths, report = self.selected_student_fixture(directory / "A4", "A4")
            run = paths[0]; student = json.loads(run.read_text()); student["warm_run_sha256"] = "0" * 64
            run.write_text(json.dumps(student))
            source, _ = self.freeze_fixture(directory, instrument=freezer.GUARDS, artifacts=[artifact], bound=paths)
            with patch.object(freezer, "raw_guard_parity", return_value=({**report, "warm_run_sha256": "0" * 64}, [])), \
                 self.assertRaisesRegex(ValueError, "warm_run_sha256"):
                freezer.extend(source, old_warm_guards=["old"], new_warm_guards=["new"])

    def test_raw_parity_compares_outputs_not_metrics_and_rejects_any_raw_difference(self):
        old = {"artifact_arm": "A2", "adapter_sha256": "same", "adapter_config_sha256": "same", "run_sha256": "same",
               "config": {"seed": 0}, "config_sha256": "same", "data_sha256": "same", "fixtures_sha256": "same", "traces_sha256": "same",
               "outcomes": {"correct": 0}}
        new = {**old, "outcomes": {"correct": 999}}
        raw = {"status": "generation-complete", "dynamics": [{"conditioning": {"s0": {"x": 1}}, "generated": {"text": "raw"}}],
               "behavior": [{"fixture_id": str(i), "generated": {"text": str(i)}} for i in range(6)]}
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "old.json", Path(temporary) / "new.json"
            a.write_text("old fixture"); b.write_text("new fixture")
            with patch.object(freezer, "_checked_guard", side_effect=[(old, raw, [a]), (new, copy.deepcopy(raw), [b])]):
                report, _ = freezer.raw_guard_parity(a, b)
                self.assertTrue(report["passed"])
                for field, original_field in (("warm_adapter_sha256", "adapter_sha256"),
                                              ("warm_adapter_config_sha256", "adapter_config_sha256"), ("warm_run_sha256", "run_sha256")):
                    self.assertEqual(report[field], new[original_field])
            changed = copy.deepcopy(raw); changed["behavior"][0]["generated"]["text"] = "changed"
            with patch.object(freezer, "_checked_guard", side_effect=[(old, raw, [a]), (new, changed, [b])]), \
                 self.assertRaisesRegex(ValueError, "raw-output parity failed"):
                freezer.raw_guard_parity(a, b)

    def test_checked_guard_binds_config_artifact_freeze_and_raw_trace_before_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); adapter = directory / "warm" / "adapter"; adapter.mkdir(parents=True)
            config, data, fixtures = [directory / name for name in ("config.json", "dev.jsonl", "fixtures.json")]
            config.write_text(json.dumps(cfg().dict()))
            data.write_bytes(b"synthetic public input digest only"); fixtures.write_bytes(b"synthetic fixture digest only")
            run = adapter.parent / "run.json"; run.write_bytes(b"synthetic unchanged warm run")
            for name in ("adapter_model.safetensors", "adapter_config.json"): (adapter / name).write_bytes(b"synthetic artifact only")
            source = ROOT / "scripts/evaluate_recovery_student_guards_cpu105_readout.py"
            freeze, path = directory / "freeze.json", directory / "warm_guard.json"
            bindings = [config, data, fixtures, run, adapter / "adapter_model.safetensors", adapter / "adapter_config.json", source]
            freeze.write_text(json.dumps({"files": {str(p): native.sha256(p) for p in bindings}}))
            trace = path.with_suffix(".traces.json")
            trace.write_text(json.dumps({"status": "generation-complete", "dynamics": [{"generated": {"text": "raw"}}],
                                         "behavior": [{"fixture_id": str(i), "generated": {"text": "raw"}} for i in range(6)]}))
            record = {"schema": "dsv3wm-h3-fixed-development-guards-v1", "status": "development-guards-evaluated", "warm_baseline": True,
                      "artifact_arm": "A2", "oracle_feedback": False, "all_generation_fixed_before_scoring": True,
                      "resource_admission_profile": readout.ADMISSION_PROFILE, "config": cfg().dict(),
                      "source_sha256": {str(source): native.sha256(source)}, "freeze_sha256": native.sha256(freeze),
                      "traces_sha256": native.sha256(trace), "command": [str(source)]}
            flags = ("--config", "--data", "--fixtures", "--run", "--adapter", "--freeze", "--out")
            for flag, value in zip(flags, (config, data, fixtures, run, adapter, freeze, path)):
                record["command"] += [flag, str(value)]
            fields = ("config_sha256", "data_sha256", "fixtures_sha256", "run_sha256", "adapter_sha256", "adapter_config_sha256")
            for field, value in zip(fields, bindings): record[field] = native.sha256(value)
            path.write_text(json.dumps(record))
            _, raw, paths = freezer._checked_guard(path, singleton=True)
            self.assertEqual(len(raw["behavior"]), 6); self.assertIn(trace, paths)
            for field in ("adapter_sha256", "freeze_sha256", "traces_sha256"):
                changed = {**record, field: "0" * 64}; path.write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises((RuntimeError, ValueError)):
                    freezer._checked_guard(path, singleton=True)
            changed = copy.deepcopy(record); changed["config"]["ce_chunk"] = 4; path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "expanded configuration"):
                freezer._checked_guard(path, singleton=True)

    def test_exclusive_output_refuses_before_source_inspection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.json"; path.write_bytes(b"preserved freeze")
            argv = ["freeze_recovery_student_cpu105_readout.py", "--freeze", "missing", "--out", str(path)]
            with patch.object(sys, "argv", argv), patch.object(freezer, "extend") as extend, self.assertRaises(FileExistsError):
                freezer.main()
            extend.assert_not_called(); self.assertEqual(path.read_bytes(), b"preserved freeze")


if __name__ == "__main__":
    unittest.main()
