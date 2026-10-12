"""Recovery collection semantics and split/information boundaries; no native model."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsv3wm.data import conditioning, make_record, validate_record
from dsv3wm.domains import parent_world_key, rules_for
from dsv3wm.recovery_data import (build_recovery, collection_distances, legacy_target, read_prior_public,
                                  write_recovery)


class RecoveryDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.excluded = {parent_world_key("register", {"x": 3}, rules_for("register", limit=32))}
        cls.data = build_recovery(291, exclusions=cls.excluded)

    def test_balanced_unique_training_and_changed_four_step_states(self):
        train = self.data["train"]
        self.assertEqual(len(train), 128)
        self.assertEqual(len({row["parent_world_id"] for row in train}), 128)
        self.assertEqual(sum(row["labels"]["legal"][0] for row in train), 64)
        self.assertEqual(sum(sum(row["labels"]["legal"]) for row in train), 256)
        for row in train:
            self.assertEqual(len(row["actions"]), 4)
            self.assertEqual(sum(row["labels"]["legal"]), 2)
            self.assertNotEqual(row["labels"]["states"][3], row["s0"])
            self.assertNotIn("finish", row["actions"][:3])
            self.assertIn(row["rules"]["limit"], (24, 32, 40))
            validate_record(row, verify_labels=True)
        self.assertEqual(sum(row["actions"][-1] == "finish" for row in train), 8)

    def test_dev_primary_cohorts_and_global_parent_registry(self):
        data = self.data
        self.assertEqual(len(data["dev"]), 24)
        self.assertEqual(len({row["parent_world_id"] for row in data["dev"]}), 24)
        for cohort, limits in (("recovery_id_teacher_known", (32, 48, 64)), ("recovery_ood_limit", (80, 96))):
            rows = [row for row in data["dev"] if row["provenance"]["cohort"] == cohort]
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(row["labels"]["legal"][0] for row in rows), 8)
            self.assertEqual(sum(not row["labels"]["legal"][0] for row in rows), 4)
            for row in rows:
                self.assertIn(row["rules"]["limit"], limits)
                self.assertNotEqual(row["labels"]["states"][3], row["s0"])
                self.assertGreaterEqual(collection_distances(row["s0"]["x"], row["rules"]["limit"])
                                        [row["labels"]["states"][3]["x"]], 2)
                if row["labels"]["legal"][0]:
                    self.assertNotEqual(row["labels"]["states"][0], row["s0"])
                self.assertEqual(row["provenance"]["seen_by_initial_F_one_step"], cohort.endswith("known"))
        self.assertEqual(sum(abs(row["s0"]["x"]) >= 65 for row in data["dev"]), 6)
        self.assertEqual(len(data["dev_invalid"]), 4)
        self.assertTrue(all(row["labels"]["invalid"][0] for row in data["dev_invalid"]))
        all_rows = [*data["train"], *data["dev"], *data["dev_invalid"]]
        self.assertEqual(len({row["parent_world_id"] for row in all_rows}), 156)
        self.assertFalse(self.excluded & {row["parent_world_id"] for row in all_rows})
        self.assertTrue(data["integrity"]["parent_disjoint"])
        self.assertNotIn("confirmation", data)

    def test_planning_goals_and_no_privileged_fields_in_task_inputs(self):
        scoring = {row["task_id"]: row for row in self.data["planning_scoring"]}
        parent_rows = {row["parent_world_id"]: row for row in [*self.data["train"], *self.data["dev"]]}
        self.assertEqual(len(self.data["planning_train"]), 16)
        self.assertEqual(len(self.data["planning_dev"]), 12)
        for split, low, high in (("train", 2, 4), ("dev", 3, 5)):
            tasks = self.data["planning_" + split]
            self.assertEqual(len(tasks), len({task["parent_world_id"] for task in tasks}))
            for task in tasks:
                self.assertEqual(task["split"], split)
                self.assertFalse({"actions", "plan", "solution", "labels", "optimal_moves", "states"} & set(task))
                row = parent_rows[task["parent_world_id"]]
                self.assertNotEqual(task["goal"]["state"], row["labels"]["states"][3])
                distance = collection_distances(task["s0"]["x"], task["rules"]["limit"])[task["goal"]["state"]["x"]]
                self.assertTrue(low <= distance <= high)
                self.assertEqual(scoring[task["id"]]["optimal_moves"], distance)

    def test_legacy_projection_is_causal_and_does_not_execute_oracles(self):
        row = self.data["train"][0]
        with patch("dsv3wm.domains._semantics", side_effect=AssertionError("oracle during inference")):
            inputs = conditioning(row, 1)
            self.assertEqual(set(inputs), {"domain", "rules", "s0", "actions"})
            for horizon in (1, 2, 4):
                target = legacy_target(row, horizon)
                self.assertEqual(set(target), {"x", "legal", "terminated"})
                self.assertEqual(target["x"], row["labels"]["states"][horizon - 1]["x"])
        with self.assertRaises(PermissionError):
            legacy_target(dict(row, split="confirmation"), 4)

    def test_public_exclusion_and_frozen_manifest_refuse_vault_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = []
            for split, x in (("train", 1), ("dev", 2), ("confirmation", 3)):
                row = make_record("register", {"x": x}, ["add1"] * 4, rules_for("register"), split=split)
                if split == "confirmation":
                    row.pop("labels")
                path = directory / (split + ".jsonl")
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                paths.append(path)
            excluded, sources = read_prior_public(paths)
            self.assertEqual(len(excluded), 3)
            self.assertEqual(len(sources), 3)
            with self.assertRaisesRegex(ValueError, "all three public"):
                read_prior_public(paths[:2])
            with self.assertRaises(ValueError):
                read_prior_public([directory / "confirmation.labels.jsonl"])
            output = directory / "recovery"
            manifest = write_recovery(output, seed=201, prior_public=paths)
            self.assertEqual(manifest["confirmation_rows"], 0)
            self.assertEqual(manifest["primary_parent_counts"], {"train": 128, "dev": 24})
            self.assertTrue(manifest["all_primary_h4_changed"])
            for filename, metadata in manifest["files"].items():
                self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), metadata["sha256"])
            self.assertFalse(any("confirmation" in path.name for path in output.iterdir()))
            with self.assertRaises(FileExistsError):
                write_recovery(output, seed=201, prior_public=paths)


if __name__ == "__main__":
    unittest.main()
