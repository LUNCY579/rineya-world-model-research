"""Pure/mock conditional legality diagnostics; no torch import or native model."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dsv3wm.domains import parent_world_key, rules_for
from dsv3wm.native import sha256
from dsv3wm.recovery import RecoveryConfig

spec = importlib.util.spec_from_file_location("_legal_calibration_pure_tests", ROOT / "scripts/evaluate_recovery_legal_calibration.py")
calibration = importlib.util.module_from_spec(spec); spec.loader.exec_module(calibration)


class Tokenizer:
    eos_token_id = 9999
    all_special_ids = [9999, 9998]
    def __call__(self, text, add_special_tokens=False):
        if add_special_tokens is not False: raise AssertionError("No implicit special tokens")
        return SimpleNamespace(input_ids=list(text.encode()))


class EncodingLM:
    def __init__(self): self.calls = []
    def encode_pair(self, tokenizer, prompt, target, max_len, eos=True):
        self.calls.append({"prompt": prompt, "target": target, "eos": eos})
        p, t = tokenizer(prompt).input_ids, tokenizer(target).input_ids
        if eos: t += [tokenizer.eos_token_id]
        if len(p) + len(t) > max_len: return None
        return SimpleNamespace(input_ids=p + t, labels=[-100] * len(p) + t, n_target=len(t))


class ForbiddenLabels(dict):
    def __getitem__(self, key): raise AssertionError("Labels accessed before all captures")


def rows():
    result = []
    for index in range(24):
        rule = rules_for("register", limit=32 if index < 12 else 80)
        state = {"x": index - 12}
        result.append({"id": "synthetic-" + str(index), "domain": "register", "split": "dev",
            "parent_world_id": parent_world_key("register", state, rule), "rules": rule, "s0": state,
            "actions": ["add1", "double", "halve", "sub1"],
            "provenance": {"cohort": "recovery_id_teacher_known" if index < 12 else "recovery_ood_limit"},
            "labels": {"legal": [index % 3 != 0, False, False, False], "states": "must never condition on this"}})
    return result


def capture(source, scorer=None, tokenizer=None):
    return calibration.capture_probabilities(source, tokenizer or Tokenizer(), EncodingLM(),
        scorer or (lambda prompt, literal, encoded: math.log(.2 if literal == "true," else .3)))


class LegalCalibrationTests(unittest.TestCase):
    def test_exact_no_eos_encoding_and_response_mask(self):
        lm = EncodingLM(); tokenizer = Tokenizer()
        encoded = calibration.checked_encoding(tokenizer, lm, 'prompt{"legal":', "true,")
        self.assertFalse(lm.calls[0]["eos"])
        self.assertEqual(encoded.n_target, len("true,"))
        self.assertEqual(encoded.labels[-encoded.n_target:], list(b"true,"))
        self.assertNotIn(tokenizer.eos_token_id, encoded.input_ids)

    def test_nonmatching_boundary_special_token_empty_and_overlength_refuse(self):
        class Merged(Tokenizer):
            def __call__(self, text, add_special_tokens=False):
                ids = super().__call__(text, add_special_tokens).input_ids
                if text.endswith(':true,'): ids[-6:] = [888]
                return SimpleNamespace(input_ids=ids)
        class Special(Tokenizer):
            def __call__(self, text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[self.eos_token_id] if text == "true," else list(text.encode()))
        class Empty(Tokenizer):
            def __call__(self, text, add_special_tokens=False):
                return SimpleNamespace(input_ids=[] if text == "true," else list(text.encode()))
        for tokenizer in (Merged(), Special(), Empty()):
            lm = EncodingLM()
            with self.assertRaises(ValueError): calibration.checked_encoding(tokenizer, lm, 'p{"legal":', "true,")
            self.assertEqual(lm.calls, [])
        with self.assertRaisesRegex(ValueError, "<=512"):
            calibration.checked_encoding(Tokenizer(), EncodingLM(), "p" * 512, "true,")

    def test_wrong_no_eos_mask_cannot_pass_encoding_validation(self):
        class WrongMask(EncodingLM):
            def encode_pair(self, *args, **kwargs):
                encoded = super().encode_pair(*args, **kwargs)
                encoded.labels[0] = 1
                return encoded
        with self.assertRaisesRegex(ValueError, "response mask"):
            calibration.checked_encoding(Tokenizer(), WrongMask(), "p", "false,")

    def test_disjoint_complete_paths_and_actual_mass_before_normalization(self):
        value = calibration.binary_mass(math.log(.2), math.log(.3))
        self.assertEqual(value["status"], "captured")
        self.assertAlmostEqual(value["literal_mass_true"], .2)
        self.assertAlmostEqual(value["literal_mass_false"], .3)
        self.assertAlmostEqual(value["valid_literal_mass"], .5)
        self.assertAlmostEqual(value["other_mass"], .5)
        self.assertAlmostEqual(value["conditional_p_true"], .4)
        self.assertAlmostEqual(math.exp(value["conditional_logp_false"]), .6)

    def test_impossible_nonfinite_mass_and_finite_underflow_are_explicit(self):
        for pair in ((float("nan"), -1.), (float("inf"), -1.), (-math.inf, -1.), (.1, -1.), (0., 0.), (True, -1.)):
            value = calibration.binary_mass(*pair)
            self.assertEqual(value["status"], "finite-failure")
            self.assertNotIn("conditional_p_true", value)
            json.dumps(value, allow_nan=False)
        value = calibration.binary_mass(-1000., -1000.)
        self.assertEqual(value["status"], "captured"); self.assertTrue(value["literal_mass_underflow"])
        self.assertEqual(value["valid_literal_mass"], 0.)
        self.assertAlmostEqual(value["conditional_p_true"], .5)
        self.assertAlmostEqual(value["conditional_logp_true"], -math.log(2))
        confident = calibration.binary_mass(math.log1p(-1e-8), math.log(1e-8))
        self.assertEqual(confident["status"], "captured")
        self.assertGreater(confident["conditional_p_true"], .99999998)

    def test_all_48_probability_scores_precede_labels_and_have_only_causal_h1_prefix(self):
        source = rows(); labels = [row["labels"] for row in source]
        for row in source: row["labels"] = ForbiddenLabels()
        lm = EncodingLM(); calls = []
        def score(prompt, literal, encoded):
            self.assertTrue(prompt.endswith(calibration.PREFIX))
            self.assertNotIn("must never condition", prompt)
            self.assertNotIn('"actions":', prompt)
            self.assertNotIn('"goal":', prompt)
            self.assertNotIn('"labels":', prompt)
            calls.append(literal)
            return math.log(.2 if literal == "true," else .3)
        with patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("No oracle")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("No oracle")):
            result = calibration.capture_probabilities(source, Tokenizer(), lm, score)
        self.assertEqual(calls, list(calibration.LITERALS) * 24)
        self.assertTrue(all(call["eos"] is False for call in lm.calls))
        self.assertTrue(all(len(trace["conditioning"]["actions"]) == 1 for trace in result["traces"]))
        for row, gold in zip(source, labels): row["labels"] = gold
        scored = calibration.score_probabilities(source, result)
        self.assertEqual(scored["summary"]["all"]["n"], 24)
        self.assertAlmostEqual(scored["summary"]["all"]["brier"], (16 * .36 + 8 * .16) / 24)
        self.assertAlmostEqual(scored["summary"]["all"]["logloss"], -(16 * math.log(.4) + 8 * math.log(.6)) / 24)

    def test_identical_or_prefix_token_paths_refuse_without_a_score_call(self):
        source = rows(); calls = []
        for paths in (("true,", "true,"), ("true,", "true," + "false,")):
            with patch.object(calibration, "LITERALS", paths):
                result = capture(source, lambda *args: calls.append(args) or -1.)
            self.assertTrue(all(trace["status"] == "capture-failure" for trace in result["traces"]))
        self.assertEqual(calls, [])

    def test_complete_attempts_are_separate_from_complete_probabilities(self):
        for n, successes, attempts, probabilities in ((23, 23, False, False), (24, 23, True, False),
                                                     (24, 0, True, False), (24, 24, True, True)):
            traces = [{"status": "captured" if index < successes else "finite-failure"} for index in range(n)]
            result = calibration.capture_completeness(traces)
            self.assertEqual(result["successful_probability_captures"], successes)
            self.assertEqual(result["complete_capture_attempts"], attempts)
            self.assertEqual(result["complete_probability_capture"], probabilities)
        interrupted = calibration.capture_completeness([{"status": "captured"}] * 23 + [{"status": "capture-aborted"}])
        self.assertEqual(interrupted["capture_parents_started"], 24)
        self.assertEqual(interrupted["capture_rows_completed"], 23)
        self.assertFalse(interrupted["complete_capture_attempts"])

    def test_first_path_success_second_valueerror_preserves_per_path_work_and_score(self):
        source = rows()
        gold = [row["labels"] for row in source]
        for row in source: row["labels"] = ForbiddenLabels()
        counters = {"literal_queries": 0, "literal_queries_returned": 0, "input_tokens": 0, "decoder_token_positions": 0}
        snapshots = []
        def score(prompt, literal, encoded):
            counters["literal_queries"] += 1
            counters["input_tokens"] += len(encoded.input_ids)
            counters["decoder_token_positions"] += len(encoded.input_ids)
            if counters["literal_queries"] == 2: raise ValueError("synthetic second-path scoring failure")
            counters["literal_queries_returned"] += 1
            return math.log(.2 if literal == "true," else .3)
        captured = calibration.capture_probabilities(source, Tokenizer(), EncodingLM(), score,
            sink=lambda traces: snapshots.append(copy.deepcopy(traces)), work_snapshot=lambda: dict(counters))
        first = captured["traces"][0]
        self.assertEqual(first["status"], "capture-failure")
        self.assertEqual(first["literal_paths"][0]["status"], "returned")
        self.assertAlmostEqual(first["literal_paths"][0]["raw_logp"], math.log(.2))
        self.assertEqual(first["literal_paths"][0]["work"]["literal_queries_returned"], 1)
        failed = first["literal_paths"][1]
        self.assertEqual(failed["status"], "score-failure"); self.assertFalse(failed["returned"])
        self.assertEqual(failed["work"]["literal_queries"], 1)
        self.assertEqual(failed["work"]["literal_queries_returned"], 0)
        self.assertGreater(failed["work"]["decoder_token_positions"], 0)
        self.assertTrue(any(len(snapshot) == 1 and snapshot[-1].get("literal_paths")
                            and snapshot[-1]["literal_paths"][0].get("returned") for snapshot in snapshots))
        for row, labels in zip(source, gold): row["labels"] = labels
        scored = calibration.score_probabilities(source, captured)
        self.assertIsNone(scored["per_parent"][0]["brier"])
        self.assertEqual(scored["summary"]["all"]["captured"], 23)

    def test_second_path_operational_abort_flushes_current_parent_without_gold_scoring(self):
        for failure in (RuntimeError("synthetic priority boundary"), KeyboardInterrupt()):
            source = rows()
            for row in source: row["labels"] = ForbiddenLabels()
            snapshots, ordinal = [], [0]
            def score(prompt, literal, encoded):
                ordinal[0] += 1
                if ordinal[0] == 2: raise failure
                return math.log(.2)
            with self.subTest(failure=type(failure).__name__), self.assertRaises(type(failure)):
                calibration.capture_probabilities(source, Tokenizer(), EncodingLM(), score,
                    sink=lambda traces: snapshots.append(copy.deepcopy(traces)))
            current = snapshots[-1][0]
            self.assertEqual(current["status"], "capture-aborted")
            self.assertEqual(current["literal_paths"][0]["status"], "returned")
            self.assertAlmostEqual(current["literal_paths"][0]["raw_logp"], math.log(.2))
            self.assertEqual(current["literal_paths"][1]["status"], "score-aborted")
            self.assertTrue(current["literal_paths"][1]["scoring_attempted"])
            self.assertFalse(current["literal_paths"][1]["returned"])
            self.assertEqual(current["literal_paths"][1]["supervised_literal_tokens"], len("false,"))
            self.assertFalse(calibration.capture_completeness(snapshots[-1])["complete_capture_attempts"])
            self.assertNotIn("conditional_p_true", current)
            json.dumps(snapshots[-1], allow_nan=False)

    def test_returned_nonfinite_path_is_incrementally_json_safe(self):
        snapshots = []
        captured = calibration.capture_probabilities(rows(), Tokenizer(), EncodingLM(), lambda *args: float("nan"),
            sink=lambda traces: snapshots.append(json.loads(json.dumps(traces, allow_nan=False))))
        self.assertEqual(captured["traces"][0]["status"], "finite-failure")
        path = captured["traces"][0]["literal_paths"][0]
        self.assertIsNone(path["raw_logp"]); self.assertEqual(path["raw_logp_repr"], "nan")
        self.assertTrue(path["returned"]); self.assertFalse(path["finite"])

    def test_partial_and_all_failures_keep_denominators_without_zero_filling(self):
        source = rows(); ordinal = [0]
        def score(prompt, literal, encoded):
            ordinal[0] += 1
            return float("nan") if ordinal[0] == 1 else math.log(.2 if literal == "true," else .3)
        result = calibration.score_probabilities(source, capture(source, score))
        self.assertEqual(result["summary"]["all"]["captured"], 23)
        self.assertEqual(result["summary"]["all"]["finite_failures"], 1)
        self.assertIsNone(result["per_parent"][0]["brier"])
        self.assertFalse(result["complete_probability_capture"])
        failed = calibration.score_probabilities(source, capture(source, lambda *args: float("nan")))
        self.assertTrue(failed["complete_capture_attempts"]); self.assertFalse(failed["complete_probability_capture"])
        self.assertEqual(failed["summary"]["all"]["failures"], 24)
        self.assertIsNone(failed["summary"]["all"]["brier"])
        self.assertIsNone(failed["summary"]["all"]["logloss"])

    def test_ten_bins_one_final_bin_and_reporting_clip_preserve_logspace_loss(self):
        source = rows()
        result = calibration.score_probabilities(source, capture(source, lambda p, literal, encoded: 0. if literal == "true," else -1000.))
        bins = result["summary"]["all"]["fixed_bins"]
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[-1]["n"], 24); self.assertTrue(bins[-1]["high_inclusive"])
        self.assertTrue(all(bin["n"] == 0 and bin["mean_p_true"] is None for bin in bins[:-1]))
        negative = result["per_parent"][0]
        self.assertEqual(negative["unclipped_logloss"], 1000.)
        self.assertTrue(negative["logloss_clipped"])
        self.assertGreater(negative["logloss"], 27.); self.assertLess(negative["logloss"], 28.)

    def test_corrupt_probability_or_late_parent_is_rejected_before_any_gold_read(self):
        source = rows(); captured = capture(source)
        for row in source: row["labels"] = ForbiddenLabels()
        for field, value in (("conditional_p_true", .9), ("parent_world_id", "wrong")):
            changed = copy.deepcopy(captured); changed["traces"][-1][field] = value
            with self.assertRaises(ValueError): calibration.score_probabilities(source, changed)
        changed = copy.deepcopy(captured); changed["all_probability_traces_fixed_before_labels"] = False
        with self.assertRaises(ValueError): calibration.score_probabilities(source, changed)

    def test_prospective_freeze_and_protocol_membership_are_mandatory(self):
        self.assertIn(ROOT / "prereg/RECOVERY_LEGAL_CALIBRATION_V1.md", calibration.source_paths())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "freeze.json"
            for record in ({"instrument": "registered-F-recovery-readout-v1"},
                           {"instrument": calibration.SCHEMA, "confirmation_access": True},
                           {"instrument": calibration.SCHEMA, "confirmation_access": False, "probability_outputs_accessed": True}):
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError): calibration.require_phase_freeze(path)
            member = Path(temporary) / "source"; member.write_bytes(b"synthetic non-label input")
            record = {"instrument": calibration.SCHEMA, "confirmation_access": False, "probability_outputs_accessed": False,
                      "calibration_contract": calibration.CONTRACT, "files": {str(member): sha256(member)}}
            path.write_text(json.dumps(record)); self.assertEqual(calibration.require_phase_freeze(path), sha256(path))

    def test_versions_are_read_only_metadata_missing_is_null_not_zero(self):
        before = "torch" in sys.modules
        def version(name):
            if name == "tokenizers": raise calibration.importlib.metadata.PackageNotFoundError(name)
            return "synthetic-version-" + name
        with patch.object(calibration.importlib.metadata, "version", side_effect=version):
            record = calibration.versions_record()
        self.assertEqual("torch" in sys.modules, before)
        self.assertEqual(record["versions"]["python"], sys.version)
        self.assertIsNone(record["versions"]["tokenizers"])
        self.assertEqual(record["unavailable_distributions"], ["tokenizers"])
        self.assertIn("valid alternative", calibration.CONTRACT["other_mass_scope"])

    def lineage_fixture(self, directory, arm="A5"):
        from dsv3wm import native, recovery, recovery_student
        initial, core, output = [directory / name for name in ("initial", "core", arm)]
        initial.mkdir(); core.mkdir(); (output / "adapter").mkdir(parents=True)
        peft = {"r": 16, "lora_alpha": 32, "lora_dropout": .05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "use_rslora": False, "use_dora": False, "rank_pattern": {}, "alpha_pattern": {}, "bias": "none"}
        for path in (initial / "adapter_config.json", output / "adapter/adapter_config.json"):
            path.write_text(json.dumps(peft))
        (initial / "adapter_model.safetensors").write_bytes(b"synthetic initial weights")
        (output / "adapter/adapter_model.safetensors").write_bytes(b"synthetic trained weights")
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
            (core / name).write_bytes(("synthetic core sidecar " + name).encode())
        initial_sha, core_sha = sha256(initial / "adapter_model.safetensors"), sha256(core / "model.safetensors")
        sidecars = {name: sha256(core / name) for name in ("config.json", "tokenizer.json", "tokenizer_config.json")}
        self.enterContext(patch.object(recovery, "INITIAL_F", initial))
        self.enterContext(patch.object(recovery, "INITIAL_F_SHA", initial_sha))
        self.enterContext(patch.object(recovery_student, "INITIAL_F", initial))
        self.enterContext(patch.object(native, "EXPECTED_CORE_SHA", core_sha))
        self.enterContext(patch.object(native, "EXPECTED_SIDECARS", sidecars))
        config, q_config = directory / "config.json", directory / "q_config.json"
        configuration = RecoveryConfig(arm=arm, steps=96 if arm == "A5" else 32)
        config.write_text(json.dumps(configuration.dict())); q_config.write_text(json.dumps(RecoveryConfig().dict()))
        train, data = directory / "train.jsonl", directory / "dev.jsonl"
        train.write_bytes(b"synthetic TRAIN digest only"); data.write_bytes(b"synthetic DEV digest only")
        q_freeze, qualification = directory / "q_freeze.json", directory / "qualified.json"
        q_members = [q_config, data, initial / "adapter_model.safetensors", initial / "adapter_config.json",
                     ROOT / "scripts/evaluate_recovery.py"]
        q_freeze.write_text(json.dumps({"files": {str(path): sha256(path) for path in q_members}}))
        qualified = {"status": "development-evaluated", "baseline_qualified": True, "adapter_sha256": initial_sha,
                     "oracle_feedback": False, "freeze_sha256": sha256(q_freeze), "data_sha256": sha256(data),
                     "config": RecoveryConfig().dict(), "config_sha256": sha256(q_config),
                     "command": [str(ROOT / "scripts/evaluate_recovery.py"), "--baseline-only", "--config", str(q_config),
                                 "--data", str(data), "--freeze", str(q_freeze)]}
        qualification.write_text(json.dumps(qualified))
        training_freeze = directory / "train_freeze.json"
        training_members = [config, train, qualification, *q_members]
        frozen = {"qualification_sha256": sha256(qualification), "qualification_freeze_sha256": sha256(q_freeze),
                  "files": {str(path): sha256(path) for path in training_members}}
        training_freeze.write_text(json.dumps(frozen))
        run = output / "run.json"
        record = {"status": "pilot-trained", "config": configuration.dict(), "optimizer_steps": configuration.steps,
                  "initial_adapter_sha256": initial_sha, "core_sha256": core_sha, "config_sha256": sha256(config),
                  "data_sha256": sha256(train), "freeze_sha256": sha256(training_freeze), "qualification_sha256": sha256(qualification),
                  "adapter_sha256": sha256(output / "adapter/adapter_model.safetensors"),
                  "adapter_config_sha256": sha256(output / "adapter/adapter_config.json"),
                  "command": [str(ROOT / "scripts/train_recovery.py"), "--config", str(config), "--train", str(train),
                              "--freeze", str(training_freeze), "--qualification", str(qualification), "--out", str(output)]}
        run.write_text(json.dumps(record))
        phase = directory / "phase.json"
        members = [config, train, data, qualification, q_config, q_freeze, training_freeze, run,
                   initial / "adapter_model.safetensors", initial / "adapter_config.json", *core.glob("*"),
                   output / "adapter/adapter_model.safetensors", output / "adapter/adapter_config.json", *calibration.source_paths()]
        phase.write_text(json.dumps({"files": {str(path): sha256(path) for path in members}}))
        return [config, output / "adapter", run, train, data, phase, core], record

    def test_actual_lineage_accepts_a5_96_and_rejects_run_arm_or_weight_self_assertion(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments, record = self.lineage_fixture(Path(temporary))
            configuration, _, bound = calibration.admit_lineage(*arguments)
            self.assertEqual(configuration.arm, "A5"); self.assertEqual(configuration.steps, 96)
            self.assertIn(arguments[2], bound)
            for field, value in (("optimizer_steps", 32), ("adapter_sha256", "0" * 64), ("initial_adapter_sha256", "0" * 64)):
                arguments[2].write_text(json.dumps({**record, field: value}))
                with self.subTest(field=field), self.assertRaises((ValueError, RuntimeError)):
                    calibration.admit_lineage(*arguments)
            arguments[2].write_text(json.dumps(record))
            arguments[0].write_text(json.dumps({**record["config"], "legality_weight": False}))
            with self.assertRaisesRegex(ValueError, "legality_weight|configuration"):
                calibration.admit_lineage(*arguments)

    def test_actual_config_bytes_and_full_source_membership_must_be_phase_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments, _ = self.lineage_fixture(Path(temporary), "A1")
            configuration, _, _ = calibration.admit_lineage(*arguments)
            self.assertEqual(configuration.arm, "A1")
            phase = json.loads(arguments[5].read_text())
            del phase["files"][str(ROOT / "prereg/RECOVERY_LEGAL_CALIBRATION_V1.md")]
            arguments[5].write_text(json.dumps(phase))
            with self.assertRaisesRegex(ValueError, "input absent from freeze"):
                calibration.admit_lineage(*arguments)


if __name__ == "__main__":
    unittest.main()
