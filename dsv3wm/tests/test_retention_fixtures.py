"""Fresh fixture and comparison checks; never import torch or execute a model."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_retention.py"
spec = importlib.util.spec_from_file_location("_dsv3wm_retention_fixture_tests", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RetentionFixturesTest(unittest.TestCase):
    def test_six_fresh_fixtures_use_historical_prompt_contracts(self):
        first = audit.build_fixtures()
        self.assertEqual(first, audit.build_fixtures())
        self.assertEqual(first["fixture_count"], 6)
        rows = {row["id"]: row for row in first["fixtures"]}
        self.assertEqual(len(rows), 6)
        self.assertEqual(json.loads(rows["rw-odd-halving"]["target"]),
                         {"legal": False, "terminated": False, "x": 3})
        self.assertEqual(json.loads(rows["rw-limit-boundary"]["target"])["x"], 32)
        self.assertEqual(json.loads(rows["rw-negative-double"]["target"])["x"], -8)
        self.assertEqual(json.loads(rows["ledger-posted-transfer"]["target"]), {"A": 5, "B": 2, "C": 3, "D": 0})
        cancelled = rows["ledger-cancelled-scale"]
        self.assertEqual(json.loads(cancelled["target"]), cancelled["fresh_input"]["before"])
        self.assertEqual(json.loads(rows["ledger-posted-negative-scale"]["target"])["A"], 6)
        for row in rows.values():
            self.assertTrue(row["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
            self.assertEqual([message["role"] for message in row["messages"]], ["system", "user"])
            self.assertNotIn("target", row["messages"][1]["content"])

    def test_fixture_freeze_is_exclusive_hashed_and_model_free(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixtures.json"
            # Any accidental native load call before freezing would fail.
            with patch("dsv3wm.native.load", side_effect=AssertionError("native model executed")):
                document, digest = audit.freeze_fixtures(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), document)
            with self.assertRaises(FileExistsError):
                audit.freeze_fixtures(path)
            self.assertIn("no behavioral retention pass", document["claim_scope"])

    def test_token_weighted_summary_and_baseline_comparison(self):
        rows = [{"id": "one", "domain": "register", "token_mean_ce": 2.0, "supervised_tokens": 2,
                 "input_tokens": 10, "encoded_sha256": "a"},
                {"id": "two", "domain": "ledger", "token_mean_ce": 4.0, "supervised_tokens": 1,
                 "input_tokens": 20, "encoded_sha256": "b"}]
        baseline = {"status": "complete", "model_kind": "frozen_core", "rows": rows,
                    "fixture_sha256": "f", "parent_checkpoint_sha256": "p", "tokenizer_chat_template_sha256": "t",
                    "summary": audit.summarize(rows)}
        self.assertAlmostEqual(baseline["summary"]["overall"]["token_mean_ce"], 8 / 3)
        current = copy.deepcopy(baseline)
        current["model_kind"] = "unmerged_adapter"
        current["rows"][0]["token_mean_ce"] += 1
        current["summary"] = audit.summarize(current["rows"])
        comparison = audit.compare_baseline(current, baseline)
        self.assertAlmostEqual(comparison["overall_delta_ce"], 2 / 3)
        self.assertIsNone(comparison["retention_pass"])
        changed = copy.deepcopy(current)
        changed["fixture_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Baseline mismatch"):
            audit.compare_baseline(changed, baseline)
        changed = copy.deepcopy(current)
        changed["rows"][0]["encoded_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "tokenization mismatch"):
            audit.compare_baseline(changed, baseline)


if __name__ == "__main__":
    unittest.main()
