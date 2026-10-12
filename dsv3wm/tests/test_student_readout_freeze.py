"""Pure synthetic lineage/hash fixtures; no model or native efficacy run."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dsv3wm.native import EXPECTED_CORE_SHA, EXPECTED_SIDECARS, sha256
from dsv3wm.recovery import INITIAL_F, INITIAL_F_SHA, RecoveryConfig
from dsv3wm.recovery_student import StudentConfig, admit_readout_lineage

path = Path(__file__).resolve().parents[1] / "scripts/freeze_recovery_student_readout.py"
spec = importlib.util.spec_from_file_location("student_readout_freezer", path)
freezer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freezer)


class StudentReadoutFreezeTests(unittest.TestCase):
    def fixture(self, directory, *, control=True):
        cfg = StudentConfig(arm="A3", plan_weight=0 if control else 1, control_extra_one_step=control)
        warm, student, teacher = directory / "warm", directory / "student", directory / "teacher"
        for folder in (warm / "adapter", student / "adapter", teacher):
            folder.mkdir(parents=True)
        reference = json.loads((INITIAL_F / "adapter_config.json").read_text(encoding="utf-8"))
        for folder in (warm / "adapter", student / "adapter"):
            (folder / "adapter_model.safetensors").write_bytes(b"synthetic unit weight bytes")
            (folder / "adapter_config.json").write_text(json.dumps(reference), encoding="utf-8")
        train, tasks, qualification = directory / "train.jsonl", directory / "tasks.jsonl", directory / "qualification.json"
        train.write_text("synthetic source bytes", encoding="utf-8")
        tasks.write_text("", encoding="utf-8")
        q_config, q_data, q_freeze = directory / "q-config.json", directory / "dev.jsonl", directory / "q-freeze.json"
        q_expanded = RecoveryConfig(arm="A0").dict()
        q_config.write_text(json.dumps(q_expanded), encoding="utf-8")
        q_data.write_text("synthetic DEV source bytes", encoding="utf-8")
        initial_weights, initial_config = INITIAL_F / "adapter_model.safetensors", INITIAL_F / "adapter_config.json"
        program, trainer = freezer.ROOT / "scripts/evaluate_recovery.py", freezer.ROOT / "scripts/train_recovery.py"
        snapshot_sha = "04f7f231238605ab607480e05aa4569f4f4e85e5a8fcd6e7b9a0071bc88c57dd"
        snapshot = freezer.ROOT / f"prereg/source_snapshots/{snapshot_sha}.py"
        q_pins = {str(p.resolve()): sha256(p) for p in (q_config, q_data, initial_weights, initial_config, program)}
        q_pins[str(trainer.resolve())] = snapshot_sha
        q_freeze.write_text(json.dumps({"files": q_pins}), encoding="utf-8")
        qualification.write_text(json.dumps({"status": "development-evaluated", "baseline_qualified": True,
                                            "adapter_sha256": INITIAL_F_SHA, "oracle_feedback": False,
                                            "config": q_expanded, "data_sha256": sha256(q_data),
                                            "freeze_sha256": sha256(q_freeze),
                                            "command": [str(program), "--config", str(q_config), "--data", str(q_data),
                                                        "--freeze", str(q_freeze), "--baseline-only"]}), encoding="utf-8")
        warm_config, warm_freeze = directory / "warm-config.json", directory / "warm-freeze.json"
        warm_expanded = RecoveryConfig(arm="A2").dict()
        warm_config.write_text(json.dumps(warm_expanded), encoding="utf-8")
        warm_pins = {str(p.resolve()): sha256(p) for p in
                     (warm_config, train, qualification, q_config, q_data, initial_weights, initial_config, program, trainer)}
        warm_freeze.write_text(json.dumps({"files": warm_pins, "qualification_sha256": sha256(qualification),
                                           "qualification_freeze_sha256": sha256(q_freeze)}), encoding="utf-8")
        warm_record = {"status": "pilot-trained", "config": warm_expanded, "optimizer_steps": 32,
                       "initial_adapter_sha256": INITIAL_F_SHA, "core_sha256": EXPECTED_CORE_SHA,
                       "data_sha256": sha256(train), "adapter_sha256": sha256(warm / "adapter/adapter_model.safetensors"),
                       "config_sha256": sha256(warm_config), "freeze_sha256": sha256(warm_freeze),
                       "qualification_sha256": sha256(qualification),
                       "command": ["train_recovery.py", "--config", str(warm_config), "--train", str(train),
                                   "--freeze", str(warm_freeze), "--qualification", str(qualification)]}
        (warm / "run.json").write_text(json.dumps(warm_record), encoding="utf-8")
        verified, source = teacher / "verified.jsonl", directory / "source.py"
        verified.write_text("synthetic admitted corpus bytes", encoding="utf-8")
        source.write_text("# synthetic frozen source fixture\n", encoding="utf-8")
        manifest = {"status": "corpus-qualified", "gate": {"passed": True}, "train_file_sha256": sha256(train),
                    "tasks_file_sha256": sha256(tasks), "output_sha256": {"verified.jsonl": sha256(verified)}}
        (teacher / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        config, freeze = directory / "student-config.json", directory / "student-training-freeze.json"
        config.write_text(json.dumps(cfg.dict()), encoding="utf-8")
        preinputs = [config, train, tasks, qualification, teacher / "manifest.json", verified, warm / "run.json",
                     warm / "adapter/adapter_model.safetensors", warm / "adapter/adapter_config.json", warm_config,
                     warm_freeze, INITIAL_F / "adapter_config.json", source, q_config, q_data, q_freeze,
                     initial_weights, program, trainer, snapshot]
        freeze.write_text(json.dumps({"files": {str(p.resolve()): sha256(p) for p in preinputs}}), encoding="utf-8")
        record = {"status": "student-pilot-trained", "training_kind": "C3" if control else "A3", "config": cfg.dict(),
                  "optimizer_steps": 32, "core_sha256": EXPECTED_CORE_SHA, "warm_arm": "A2",
                  "adapter_sha256": sha256(student / "adapter/adapter_model.safetensors"),
                  "adapter_config_sha256": sha256(student / "adapter/adapter_config.json"),
                  "config_sha256": sha256(config), "train_sha256": sha256(train), "tasks_sha256": sha256(tasks),
                  "qualification_sha256": sha256(qualification), "freeze_sha256": sha256(freeze),
                  "teacher_manifest_sha256": sha256(teacher / "manifest.json"), "verified_corpus_sha256": sha256(verified),
                  "warm_run_sha256": sha256(warm / "run.json"), "warm_adapter_sha256": warm_record["adapter_sha256"],
                  "warm_adapter_config_sha256": sha256(warm / "adapter/adapter_config.json"),
                  "registered_adapter_config_sha256": sha256(INITIAL_F / "adapter_config.json"),
                  "source_sha256": {str(source): sha256(source)},
                  "command": ["train_recovery_student.py", "--config", str(config), "--train", str(train),
                              "--tasks", str(tasks), "--qualification", str(qualification), "--freeze", str(freeze),
                              "--teacher-dir", str(teacher), "--warm-dir", str(warm)]}
        (student / "run.json").write_text(json.dumps(record), encoding="utf-8")
        return cfg, student, train, config, manifest

    def test_full_control_configuration_cannot_masquerade_as_A3(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg, student, train, config, manifest = self.fixture(Path(temporary))
            record, extra, kind = admit_readout_lineage(student / "adapter", student / "run.json", cfg)
            self.assertEqual(kind, "h3-continuation-control-C3")
            with self.assertRaisesRegex(ValueError, "full trained"):
                admit_readout_lineage(student / "adapter", student / "run.json", StudentConfig(arm="A3"))
            config.write_text(json.dumps(StudentConfig(arm="A3").dict()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full trained"):
                freezer.validate_student(student)

    def test_actual_inputs_and_outputs_are_bound_without_native_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg, student, train, config, manifest = self.fixture(Path(temporary))
            # Exact corpus admission is tested separately in recovery_student;
            # this fixture isolates the freezer's file/lineage wiring.
            with patch("dsv3wm.data.load_records", return_value=[]), \
                 patch("dsv3wm.recovery_student.load_teacher_corpus", return_value=([{}] * 16, manifest, {"passed": True})), \
                 patch("dsv3wm.native.load", side_effect=AssertionError("no model")) as load:
                metadata, paths = freezer.validate_student(student)
                self.assertEqual(metadata["kind"], "C3")
                self.assertEqual(metadata["warm_arm"], "A2")
                self.assertIn(student / "adapter/adapter_model.safetensors", paths)
                self.assertIn(student / "adapter/adapter_config.json", paths)
                self.assertIn(student / "run.json", paths)
                load.assert_not_called()
                (student / "adapter/adapter_model.safetensors").write_bytes(b"changed actual output")
                with self.assertRaisesRegex(ValueError, "weight/config-hash"):
                    freezer.validate_student(student)

    def test_unbound_teacher_input_cannot_be_silently_refrozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg, student, train, config, manifest = self.fixture(Path(temporary))
            record = json.loads((student / "run.json").read_text())
            (student.parent / "teacher/verified.jsonl").write_text("changed labels", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "teacher-manifest/corpus"):
                freezer.validate_student(student)

    def test_incomplete_run_and_core_drift_refuse_before_file_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "run.json").write_text('{"status":"started"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed"):
                freezer.validate_student(directory)
            core = directory / "core"
            core.mkdir()
            expected = {"model.safetensors": EXPECTED_CORE_SHA, **EXPECTED_SIDECARS}
            for name in expected:
                (core / name).write_bytes(b"synthetic test core/sidecar")
            with patch("dsv3wm.native.sha256", side_effect=lambda p: expected[Path(p).name]):
                self.assertEqual(len(freezer.validate_core(core)), 4)
            expected["tokenizer.json"] = "wrong-tokenizer-hash"
            with patch("dsv3wm.native.sha256", side_effect=lambda p: expected[Path(p).name]), self.assertRaisesRegex(ValueError, "tokenizer"):
                freezer.validate_core(core)

    def test_historical_membership_and_qualification_edges_are_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg, student, train, config, manifest = self.fixture(Path(temporary))
            warm = student.parent / "warm"
            paths = freezer.validate_warm_history(warm, cfg, train)
            snapshot = freezer.ROOT / "prereg/source_snapshots/04f7f231238605ab607480e05aa4569f4f4e85e5a8fcd6e7b9a0071bc88c57dd.py"
            self.assertIn(snapshot.resolve(), paths)
            self.assertIn((student.parent / "q-freeze.json").resolve(), paths)
            self.assertIn((student.parent / "q-config.json").resolve(), paths)
            # An executed qualification config cannot be silently replaced.
            q_config = student.parent / "q-config.json"
            original = q_config.read_bytes()
            q_config.write_text(json.dumps(RecoveryConfig(arm="A0", seed=1).dict()), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                freezer.validate_warm_history(warm, cfg, train)
            q_config.write_bytes(original)
            # Changing only the old qualifier freeze used to evade warm-file
            # digest checks because that file was not a warm V2 member.
            q_freeze = student.parent / "q-freeze.json"
            q_freeze.write_text('{"files":{},"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "qualification freeze/data"):
                freezer.validate_warm_history(warm, cfg, train)

    def test_only_registered_nonexecuted_snapshot_drift_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg, student, train, config, manifest = self.fixture(Path(temporary))
            snapshot = (freezer.ROOT / "prereg/source_snapshots/04f7f231238605ab607480e05aa4569f4f4e85e5a8fcd6e7b9a0071bc88c57dd.py").resolve()
            actual_hash = sha256
            with patch("dsv3wm.native.sha256", side_effect=lambda p: "bad-snapshot-digest" if Path(p).resolve() == snapshot else actual_hash(p)):
                with self.assertRaisesRegex(ValueError, "outside registered snapshot"):
                    freezer.validate_warm_history(student.parent / "warm", cfg, train)


if __name__ == "__main__":
    unittest.main()
