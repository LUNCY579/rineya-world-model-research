import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/analyze_recovery_selective_risk.py"
spec = importlib.util.spec_from_file_location("selective_risk", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SelectiveRiskTests(unittest.TestCase):
    def rows(self, flags):
        return [{"horizons": {"4": {"rollout_correct": correct, "rollout_malformed": malformed,
                                     "direct_correct": correct, "direct_malformed": malformed}}}
                for correct, malformed in flags]

    def test_valid_wrong_prediction_is_not_deferral(self):
        result = module.operating_point(self.rows([(False, True), (False, False), (True, False)]), mode="rollout")
        self.assertEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["accepted_error_risk"], .5)
        self.assertEqual(result["error_detection_recall"], .5)

    def test_unmeasured_zero_acceptance_and_zero_errors_remain_distinct(self):
        unknown = module.operating_point(self.rows([(None, None)]), mode="direct")
        self.assertEqual(unknown["status"], "not-measured")
        abstain = module.operating_point(self.rows([(False, True)]), mode="direct")
        self.assertIsNone(abstain["accepted_error_risk"])
        correct = module.operating_point(self.rows([(True, False)]), mode="direct")
        self.assertEqual(correct["accepted_error_risk"], 0)
        self.assertIsNone(correct["error_detection_recall"])

    def test_mixed_unknown_invalid_types_or_impossible_flags_refuse(self):
        for flags in ([(None, None), (True, False)], [(1, False)], [(True, True)]):
            with self.assertRaises(ValueError):
                module.operating_point(self.rows(flags), mode="direct")

    def test_mismatched_or_duplicate_parents_refuse(self):
        rows = self.rows([(True, False)] * 24)
        for index, row in enumerate(rows):
            row.update(parent_world_id=str(index), cohort="id" if index < 12 else "ood")
        record = {"status": "development-evaluated", "data_sha256": "same-data", "per_parent": rows}
        changed = copy.deepcopy(record)
        changed["per_parent"][0]["parent_world_id"] = "other"
        with self.assertRaises(ValueError):
            module.analyze({"A0": record, "A1": changed})
        changed["per_parent"][0] = changed["per_parent"][1]
        with self.assertRaises(ValueError):
            module.analyze({"A1": changed})
