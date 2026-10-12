"""Causal, semantic, representation and fresh split integrity checks (CPU only)."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsv3wm.data import (causal_prompt, conditioning, generate_splits, integrity_audit, load_records,
                         make_record, target, training_examples, validate_record, write_dataset)
from dsv3wm.domains import (apply_delta, exact_transition, parent_world_key, rollout_labels, rules_for,
                            semantic_key)


class DynamicsTest(unittest.TestCase):
    def test_register_illegal_boundary_and_absorbing_finish(self):
        labels = rollout_labels("register", {"x": 3}, ["halve", "add1", "double", "finish", "sub1"],
                                rules_for("register", limit=4))
        self.assertEqual(labels["states"], [{"x": 3}, {"x": 4}, {"x": 4}, {"x": 4}, {"x": 4}])
        self.assertEqual(labels["legal"], [False, True, False, True, False])
        self.assertEqual(labels["terminated"], [False, False, False, True, True])
        self.assertEqual(labels["executed"], [True, True, True, True, False])
        self.assertEqual(labels["failed"], [True] * 5)
        clean = rollout_labels("register", {"x": 2}, ["finish", "halve"], rules_for("register"))
        self.assertEqual(clean["failed"], [False, False])

    def test_ledger_status_gating_and_transfer_conservation(self):
        state = {"A": 8, "B": -2, "C": 0, "D": 1}
        event = {"kind": "transfer", "account": "A", "other": "B", "amount": 5, "status": "posted"}
        after, legal, done, invalid = exact_transition("ledger", state, event, rules_for("ledger"))
        self.assertEqual(after, {"A": 3, "B": 3, "C": 0, "D": 1})
        self.assertEqual(sum(after.values()), sum(state.values()))
        self.assertEqual((legal, done, invalid), (True, False, False))
        pending = dict(event, status="cancelled")
        self.assertEqual(exact_transition("ledger", state, pending, rules_for("ledger")),
                         (state, True, False, False))
        same = dict(event, other="A")
        self.assertEqual(exact_transition("ledger", state, same, rules_for("ledger"))[0], state)
        scaled = dict(event, kind="scale", amount=-3)
        self.assertEqual(exact_transition("ledger", state, scaled, rules_for("ledger"))[0]["A"], -24)

    def test_blocks_relational_legality_and_effects(self):
        state = {"clear": ["a", "b"], "hand": "empty", "on": [], "table": ["a", "b"]}
        rules = rules_for("blocks", objects=["a", "b"])
        labels = rollout_labels("blocks", state, ["stack a b", "pick-up a", "stack a b", "unstack a b"], rules)
        self.assertEqual(labels["legal"], [False, True, True, True])
        self.assertEqual(labels["states"][0], state)
        self.assertEqual(labels["states"][1], {"clear": ["b"], "hand": "a", "on": [], "table": ["b"]})
        self.assertEqual(labels["states"][2], {"clear": ["a"], "hand": "empty", "on": [["a", "b"]], "table": ["b"]})
        self.assertEqual(labels["states"][3], labels["states"][1])

    def test_invalid_contract_is_noop_in_every_domain(self):
        cases = [("register", {"x": 2}, {"kind": "add1"}, rules_for("register")),
                 ("ledger", {"A": 2, "B": 1, "C": 0, "D": 0},
                  {"kind": "add", "account": "missing", "amount": 2, "status": "posted"}, rules_for("ledger")),
                 ("blocks", {"clear": ["a"], "hand": "empty", "on": [], "table": ["a"]},
                  "pick-up", rules_for("blocks", objects=["a"]))]
        for domain, state, action, rules in cases:
            self.assertEqual(exact_transition(domain, state, action, rules), (state, False, False, True))

    def test_delta_rejects_unknown_and_contradictory_facts(self):
        state = {"clear": ["a", "b"], "hand": "empty", "on": [], "table": ["a", "b"]}
        for delta in ({"add": ["invented a"], "del": []}, {"add": ["holding a"], "del": []},
                      {"add": ["on a a"], "del": ["ontable a"]}):
            with self.assertRaises(ValueError):
                apply_delta("blocks", state, delta)


class CausalProjectionTest(unittest.TestCase):
    def setUp(self):
        self.record = make_record("register", {"x": 2}, ["add1", "double", "sub1", "halve"], rules_for("register"))

    def test_input_is_invariant_to_labels_goals_and_suffix_intervention(self):
        poisoned = copy.deepcopy(self.record)
        poisoned["labels"] = {"states": ["FUTURE_SECRET"]}
        poisoned["goal"] = "HIDDEN_GOAL"
        poisoned["solution"] = "SOLVER_SECRET"
        poisoned["actions"][2:] = ["double", "finish"]
        self.assertEqual(conditioning(self.record, 2), conditioning(poisoned, 2))
        self.assertEqual(causal_prompt(self.record, 2), causal_prompt(poisoned, 2))
        self.assertEqual(set(conditioning(self.record, 2)), {"domain", "rules", "s0", "actions"})
        self.assertEqual(len(conditioning(self.record, 2)["actions"]), 2)
        self.assertNotEqual(conditioning(self.record, 4), conditioning(poisoned, 4))

    def test_inference_projection_never_executes_semantics(self):
        with patch("dsv3wm.domains._semantics", side_effect=AssertionError("oracle in inference")):
            causal_prompt(self.record, 4)
            conditioning(self.record, 1)
            output = target(self.record, 2, "delta")
            self.assertEqual(apply_delta("register", self.record["s0"], output["delta"]), {"x": 6})

    def test_structural_and_causal_validation(self):
        validate_record(self.record, verify_labels=True)
        broken = copy.deepcopy(self.record)
        broken["labels"]["states"][1]["x"] = 555
        with self.assertRaisesRegex(ValueError, "Causal labels"):
            validate_record(broken, verify_labels=True)
        broken["labels"]["legal"][1] = 1
        with self.assertRaisesRegex(ValueError, "booleans"):
            validate_record(broken)
        with self.assertRaises(ValueError):
            conditioning(self.record, True)
        with self.assertRaises(ValueError):
            conditioning(self.record, 5)

    def test_training_horizons_and_confirmation_boundary(self):
        examples = training_examples([self.record])
        self.assertEqual([row["horizon"] for row in examples], [1, 2, 4])
        sealed = dict(self.record, split="confirmation")
        conditioning(sealed, 4)
        with self.assertRaises(PermissionError):
            target(sealed, 4)
        with self.assertRaises(PermissionError):
            training_examples([sealed])
        with self.assertRaises(PermissionError):
            validate_record(sealed, verify_labels=True)


class SplitIntegrityTest(unittest.TestCase):
    def test_block_renaming_fact_order_and_rule_prose_are_semantically_identical(self):
        state = {"clear": ["b"], "hand": "empty", "on": [["b", "a"]], "table": ["a"]}
        renamed = {"table": ["red"], "on": [["blue", "red"]], "hand": "empty", "clear": ["blue"]}
        rules = rules_for("blocks", objects=["a", "b"])
        alt = rules_for("blocks", objects=["blue", "red"])
        alt["text"] = "Irrelevant presentation changed"
        self.assertEqual(parent_world_key("blocks", state, rules), parent_world_key("blocks", renamed, alt))
        self.assertEqual(semantic_key("blocks", state, ["unstack b a", "stack b a"], rules),
                         semantic_key("blocks", renamed, ["unstack blue red", "stack blue red"], alt))
        with self.assertRaises(ValueError):
            parent_world_key("blocks", state, rules_for("blocks", objects=list("abcdef")))

    def test_ledger_renaming_and_keys_canonicalize(self):
        state = {"A": 1, "B": 2, "C": 3, "D": 4}
        renamed = {"z": 4, "y": 3, "x": 2, "w": 1}
        events = [{"kind": "transfer", "account": "A", "other": "B", "amount": 1, "status": "posted"}]
        ev2 = [dict(events[0], account="w", other="x")]
        self.assertEqual(semantic_key("ledger", state, events, rules_for("ledger")),
                         semantic_key("ledger", renamed, ev2, rules_for("ledger", objects=list("wxyz"))))

    def test_fresh_generation_is_reproducible_and_parent_disjoint(self):
        first, second = generate_splits(418, 18), generate_splits(418, 18)
        self.assertEqual(first, second)
        audit = integrity_audit(first)
        self.assertTrue(audit["parent_disjoint"] and audit["semantic_disjoint"] and audit["exact_disjoint"])
        for split in ("train", "dev"):
            for row in first[split]:
                validate_record(row, verify_labels=True)
                for horizon in (1, 2, 4):
                    full = target(row, horizon)["state"]
                    delta = target(row, horizon, "delta")["delta"]
                    self.assertEqual(apply_delta(row["domain"], row["s0"], delta), full)

    def test_renamed_parent_cross_split_is_rejected(self):
        train = make_record("register", {"x": 2}, ["add1"] * 4, rules_for("register"))
        dev = make_record("register", {"x": 2}, ["sub1"] * 4, rules_for("register"), split="dev")
        with self.assertRaisesRegex(ValueError, "Cross-split parent"):
            integrity_audit({"train": [train], "dev": [dev]})

    def test_sealing_manifest_public_inputs_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            output, vault = Path(directory) / "data", Path(directory) / "private"
            manifest = write_dataset(output, vault, seed=97, rows_per_split=9)
            self.assertTrue((vault / "confirmation.labels.jsonl").exists())
            self.assertEqual(len(manifest["sealed_confirmation"]["sha256"]), 64)
            # Do not read the sealed vault, even in tests.
            confirmation = load_records(output / "confirmation.jsonl", expected_split="confirmation")
            self.assertTrue(all("labels" not in row for row in confirmation))
            for name, metadata in manifest["files"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), metadata["sha256"])
            self.assertEqual(load_records(output / "train.jsonl", expected_split="train")[0]["split"], "train")
            with self.assertRaises(FileExistsError):
                write_dataset(output, vault, rows_per_split=9)
            labelled = make_record("register", {"x": 1}, ["add1"] * 4, rules_for("register"), split="confirmation")
            fake = Path(directory) / "labelled.jsonl"
            fake.write_text(json.dumps(labelled) + "\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                load_records(fake)


if __name__ == "__main__":
    unittest.main()
