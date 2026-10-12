"""Original event schemas, inert tick exclusion and paired v1/v2 case invariance."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def reference(name, filename):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


builder = reference("_rp_v2_builder_tests", "build_rcwm_retention_v2.py")
evaluator = reference("_rp_v2_evaluator_tests", "evaluate_rcwm_retention_v2.py")
legacy = builder.legacy_builder()
freezer = reference("_rp_v2_pairing_freezer_tests", "freeze_rcwm_retention_v2.py")


class RCWMRetentionV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = legacy.build_fixtures(234)
        cls.audit = {"available": True, "input_keys": {family: set() for family in legacy.FAMILIES},
                     "parent_keys": {family: set() for family in legacy.FAMILIES},
                     "source": {"sha256": "explicit unit-test input audit", "labels_used": False}}
        cls.v2, cls.mapping = builder.upgrade(cls.v1, audit=cls.audit, ticks=[0, 3, 9])

    def test_every_register_retains_original_six_field_schema_and_prerequisite_seven(self):
        rows, contract = evaluator.validate_document(self.v2)
        self.assertEqual(len(rows), 24)
        self.assertEqual(contract, {"greedy": True, "max_input_tokens": 512, "max_new_tokens": 64})
        for row in rows:
            fields = set(row["input"]["event"])
            if row["family"] == "register_machine":
                self.assertEqual(fields, {"kind", "other", "register", "threshold", "tick", "value"})
                self.assertIs(type(row["input"]["event"]["tick"]), int)
            else:
                self.assertEqual(fields, {"add", "cost", "delete", "kind", "name", "pre_off", "pre_on"})
            self.assertEqual(json.loads(row["messages"][1]["content"]), row["input"])

    def test_paired_cases_targets_parents_graphs_and_semantics_are_unchanged(self):
        original = copy.deepcopy(self.v1)
        v2, mapping = builder.upgrade(self.v1, audit=self.audit, ticks=[0, 3, 9])
        self.assertEqual(self.v1, original)
        self.assertEqual(v2, self.v2)
        self.assertEqual(mapping, self.mapping)
        for before, after, pair in zip(self.v1["fixtures"], v2["fixtures"], mapping):
            for key in ("id", "target", "parent_world_id", "graph_group_id", "semantic_input_id"):
                self.assertEqual(before[key], after[key])
            self.assertTrue(pair["targets_identical"] and pair["operative_key_unchanged"])
            self.assertEqual(pair["v1_id"], pair["v2_id"])
            if before["family"] == "register_machine":
                self.assertNotEqual(pair["v1_exact_input_sha256"], pair["v2_exact_input_sha256"])
            else:
                self.assertEqual(before["prompt"], after["prompt"])
                self.assertEqual(pair["v1_exact_input_sha256"], pair["v2_exact_input_sha256"])

    def test_neutral_tick_cannot_disguise_known_train_operative_input(self):
        row = self.v1["fixtures"][0]
        event = row["input"]["event"]
        self.assertEqual(legacy.input_key(row["family"], row["input"]["before"], event),
                         legacy.input_key(row["family"], row["input"]["before"], dict(event, tick=999999)))
        exposed = copy.deepcopy(self.audit)
        exposed["input_keys"][row["family"]].add(row["semantic_input_id"])
        with self.assertRaisesRegex(ValueError, "operative input is exposed"):
            builder.upgrade(self.v1, audit=exposed, ticks=[0])

    def test_missing_boolean_or_noninteger_tick_and_old_v1_schema_are_refused(self):
        with self.assertRaises(ValueError):
            evaluator.validate_document(self.v1)
        for value in (True, 1.0, "1"):
            changed = copy.deepcopy(self.v2)
            changed["fixtures"][0]["input"]["event"]["tick"] = value
            with self.assertRaises(ValueError):
                evaluator.validate_document(changed)
        changed = copy.deepcopy(self.v2)
        changed["fixtures"][0]["input"]["event"].pop("tick")
        with self.assertRaises(ValueError):
            evaluator.validate_document(changed)
        with self.assertRaises(ValueError):
            builder.upgrade(self.v1, audit=self.audit, ticks=[False])

    def test_strict_v1_scoring_is_reused_without_repair(self):
        self.assertIsNone(evaluator.strict_state("prerequisite_plan", '{"after":17}'))
        self.assertIsNone(evaluator.strict_state("register_machine", '{"X":1,"Y":2,"Z":3,"tick":0}'))
        replies = [{"text": row["target"], "n_tokens": 1, "hit_eos": True} for row in self.v2["fixtures"]]
        self.assertTrue(all(row["correct"] for row in evaluator.score_all(self.v2["fixtures"], replies)))

    def test_first_party_import_closure_is_explicit(self):
        paths = {path.relative_to(builder.REPO).as_posix() for path in builder.source_paths()}
        self.assertTrue({"dsv3wm/src/dsv3wm/__init__.py", "dsv3wm/src/dsv3wm/data.py", "bwm/code/bwmn/__init__.py", "bwm/code/bwmn/flops.py",
                         "bwm/code/bwmn/blocks.py", "bwm/code/bwmn/planbench_extract.py"} <= paths)
        self.assertTrue(all(path.exists() for path in builder.source_paths()))

    def test_freezer_checks_actual_v1_bytes_mapping_and_tick_only_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "v1.json"
            source.write_text(json.dumps(self.v1), encoding="utf-8")
            digest = builder.file_hash(source)
            self.assertEqual(freezer.validate_pairing(source, self.v2, self.mapping, expected_sha=digest), source)
            changed = copy.deepcopy(self.v2)
            changed["fixtures"][0]["input"]["event"]["value"] += 1
            with self.assertRaisesRegex(ValueError, "Only neutral R tick"):
                freezer.validate_pairing(source, changed, self.mapping, expected_sha=digest)
            changed = copy.deepcopy(self.mapping)
            changed[0]["target_sha256"] = "wrong-target-hash"
            with self.assertRaisesRegex(ValueError, "mapping evidence"):
                freezer.validate_pairing(source, self.v2, changed, expected_sha=digest)
            source.write_text("modified original bytes", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Original v1 fixture bytes"):
                freezer.validate_pairing(source, self.v2, self.mapping, expected_sha=digest)

    def test_original_train_prompt_audit_records_p_m1_without_inspecting_labels(self):
        from dsv3wm.domains import _semantics
        env = _semantics("ledger")
        r = next(row for row in self.v1["fixtures"] if row["family"] == "register_machine")
        p = next(row for row in self.v1["fixtures"] if row["family"] == "prerequisite_plan")
        messages = env.multi_step_messages(p["family"], p["input"]["before"], [p["input"]["event"]])
        prompt = "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages)
        rows = [{"family": r["family"], "prompt": r["prompt"], "frame": "one", "target": "POISON_OLD_LABEL"},
                {"family": p["family"], "prompt": prompt, "frame": "m1", "target": "POISON_OLD_LABEL"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t2_train.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            audit = builder.original_frame_audit(path, expected_sha=builder.file_hash(path))
        self.assertFalse(audit["labels_used"])
        self.assertEqual(audit["families"]["register_machine"]["systems"], {"SYSTEM_ONE": 1})
        self.assertEqual(audit["families"]["prerequisite_plan"]["systems"], {"SYSTEM_MULTI": 1})
        self.assertEqual(audit["families"]["prerequisite_plan"]["chain_lengths"], {"1": 1})
        self.assertIn("not T2 trained m1", self.v2["fixtures"][-1]["frame_relation_to_t2_train"])

    def test_no_tokenizer_path_does_not_keep_stale_v1_token_counts(self):
        before = copy.deepcopy(self.v1)
        before["fixtures"][0].update(prompt_tokens=1, target_tokens_including_eos=2)
        upgraded, _ = builder.upgrade(before, audit=self.audit, ticks=[0])
        self.assertNotIn("prompt_tokens", upgraded["fixtures"][0])
        self.assertNotIn("target_tokens_including_eos", upgraded["fixtures"][0])


if __name__ == "__main__":
    unittest.main()
