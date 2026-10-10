"""Hand-written synthetic guard/custody fixtures, not future samples/targets."""
from pathlib import Path
import copy
import importlib.util
import unittest

SRC = Path(__file__).resolve().parents[1] / "reviews/adjudication/planning-gap-v2-preparation.py"
spec = importlib.util.spec_from_file_location("owned_planning_v2_preparation", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def records():
    rows = [{"sample_id": sid, "review_status": "PENDING", "producer_agent_id": "/synthetic/producer",
             "fixture_only": True, "typed_probe": True} for sid in ["fixture-a", "fixture-b"]]
    for row in rows:
        row["record_hash"] = m.record_digest(row)
    return rows


def receipts(rows):
    return [{"receipt_id": "receipt-" + row["sample_id"], "sample_id": row["sample_id"],
             "record_hash": row["record_hash"], "reviewer_role": "expert_reviewer",
             "reviewer_execution_id": "/synthetic/expert", "decision": "ACCEPT", "confidence": .9} for row in rows]


def source_task():
    return {"kind": "blocks", "on": {"cap": "base", "base": "TABLE", "free": "TABLE"},
            "goal_on": {"base": "free"}, "plan": [{"block": "cap", "to": "TABLE"}, {"block": "base", "to": "free"}]}


def destination_task():
    return {"kind": "blocks", "on": {"cap": "base", "base": "TABLE", "free": "TABLE"},
            "goal_on": {"free": "base"}, "plan": [{"block": "cap", "to": "TABLE"}, {"block": "free", "to": "base"}]}


def c1_evidence(task, reason):
    return {"swap_index": 0, "first_failure_index": 0, "first_failure_reason": reason,
            "guard_truth": m.guard_vector(task["on"], task["plan"][0])}


class CustodyTests(unittest.TestCase):
    def test_typed_equality_distinguishes_bool_int(self):
        self.assertFalse(m.typed_equal({"value": True}, {"value": 1}))
        self.assertTrue(m.typed_equal({"a": 1, "b": None}, {"b": None, "a": 1}))

    def test_exact_bytes_not_parsed_equivalence(self):
        raw = b'{"a":1}'; m.verify_bytes(raw, m.digest(raw))
        with self.assertRaises(m.EvidenceError): m.verify_bytes(b'{ "a": 1 }', m.digest(raw))

    def test_records_pending_full_hash(self):
        rows = records(); self.assertEqual(len(m.pending_record_index(rows, 2)), 2)
        rows[0]["typed_probe"] = 1
        with self.assertRaises(m.EvidenceError): m.pending_record_index(rows, 2)

    def test_status_change_rehashed_still_not_candidate(self):
        rows = records(); rows[0]["review_status"] = "ACCEPTED"; rows[0]["record_hash"] = m.record_digest(rows[0])
        with self.assertRaises(m.EvidenceError): m.pending_record_index(rows, 2)

    def test_record_duplicate_and_wrong_count(self):
        rows = records()
        with self.assertRaises(m.EvidenceError): m.pending_record_index(rows, True)
        with self.assertRaises(m.EvidenceError): m.pending_record_index(rows + [rows[0]], 3)

    def test_receipt_exact_join(self):
        rows = records(); self.assertEqual(len(m.exact_review_join(m.pending_record_index(rows, 2), receipts(rows), "expert_reviewer", "/synthetic/expert")), 2)

    def test_receipt_empty_partial_duplicate_rejected(self):
        rows = records(); index = m.pending_record_index(rows, 2); reviews = receipts(rows)
        for broken in [[], reviews[:1], [reviews[0], reviews[0]]]:
            with self.subTest(broken=broken), self.assertRaises(m.EvidenceError): m.exact_review_join(index, broken, "expert_reviewer", "/synthetic/expert")

    def test_receipt_role_execution_hash_drift(self):
        rows = records(); index = m.pending_record_index(rows, 2)
        for field, value in [("reviewer_role", "producer"), ("reviewer_execution_id", "/synthetic/producer"), ("record_hash", "0" * 64)]:
            broken = receipts(rows); broken[0][field] = value
            with self.subTest(field=field), self.assertRaises(m.EvidenceError): m.exact_review_join(index, broken, "expert_reviewer", "/synthetic/expert")

    def test_producer_identity_cannot_spoof_reviewer(self):
        rows = records(); review = receipts(rows)
        for row in review: row["reviewer_execution_id"] = "/synthetic/producer"
        with self.assertRaises(m.EvidenceError): m.exact_review_join(m.pending_record_index(rows, 2), review, "expert_reviewer", "/synthetic/producer")

    def test_decision_and_confidence_types(self):
        rows = records(); index = m.pending_record_index(rows, 2)
        for field, value in [("decision", "PENDING"), ("confidence", True), ("receipt_id", "")]:
            broken = receipts(rows); broken[0][field] = value
            with self.subTest(field=field), self.assertRaises(m.EvidenceError): m.exact_review_join(index, broken, "expert_reviewer", "/synthetic/expert")

    def test_mixed_final_decisions_preserved(self):
        rows = records(); review = receipts(rows); review[1]["decision"] = "REVISE"
        joined = m.exact_review_join(m.pending_record_index(rows, 2), review, "expert_reviewer", "/synthetic/expert")
        self.assertEqual(joined["fixture-b"]["decision"], "REVISE")

    def test_completed_attestation_typed_and_hash_bound(self):
        att = {"completed_execution": True, "actual_execution_id": "/synthetic/expert", "candidate_sha256": "a" * 64,
               "manifest_sha256": "b" * 64, "raw_receipt_sha256": "c" * 64, "first_predictions_frozen_before_target_comparison": True}
        m.completed_execution_attestation(att, "a" * 64, "b" * 64, "/synthetic/expert")
        for field, value in [("completed_execution", 1), ("candidate_sha256", "d" * 64), ("first_predictions_frozen_before_target_comparison", False)]:
            broken = dict(att, **{field: value})
            with self.subTest(field=field), self.assertRaises(m.EvidenceError): m.completed_execution_attestation(broken, "a" * 64, "b" * 64, "/synthetic/expert")

    def test_first_freeze_actual_time_order(self):
        m.first_freeze_before_comparison("2026-10-10T00:00:00Z", "2026-10-10T00:01:00+00:00")
        with self.assertRaises(m.EvidenceError): m.first_freeze_before_comparison("2026-10-10T00:02:00Z", "2026-10-10T00:01:00Z")
        with self.assertRaises(m.EvidenceError): m.first_freeze_before_comparison("2026-10-10T00:00:00", "2026-10-10T00:01:00")

    def test_public_version_and_seventh_binding(self):
        expected = {name: "a" * 64 for name in m.PUBLIC_DOCS}
        manifest = {"version": m.PUBLIC_VERSION, "public_specification_bindings": [{"path": k, "sha256": v} for k, v in expected.items()]}
        m.public_binding_check(manifest, expected)
        for version in [2, True, "rgm-planning-controls-1.0.0"]:
            with self.subTest(version=version), self.assertRaises(m.EvidenceError): m.public_binding_check(dict(manifest, version=version), expected)
        with self.assertRaises(m.EvidenceError): m.public_binding_check(dict(manifest, public_specification_bindings=manifest["public_specification_bindings"][:-1]), expected)


class PublicGuardTests(unittest.TestCase):
    def test_source_witness_four_exact_fields(self):
        e = {"action_index": 1, "precondition": "source_not_clear", "enabling_action_index": 0, "obstructing_block": "cap"}
        self.assertEqual(m.clearance_evidence_check(source_task(), e)["selected_object"], "base")

    def test_destination_witness_independent_source_clear(self):
        e = {"action_index": 1, "precondition": "destination_not_clear", "enabling_action_index": 0, "obstructing_block": "cap"}
        self.assertEqual(m.clearance_evidence_check(destination_task(), e)["selected_object"], "base")

    def test_wrong_witness_enum_indices_obstruction(self):
        e = {"action_index": 1, "precondition": "source_not_clear", "enabling_action_index": 0, "obstructing_block": "cap"}
        for field, value in [("precondition", "source"), ("precondition", "destination_not_clear"), ("action_index", True), ("enabling_action_index", False), ("obstructing_block", "free")]:
            with self.subTest(field=field, value=value), self.assertRaises(m.EvidenceError): m.clearance_evidence_check(source_task(), dict(e, **{field: value}))

    def test_v2_source_priority_is_not_v1_or_rule(self):
        task = {"on": {"a": "TABLE", "b": "a", "c": "TABLE", "d": "c"}, "goal_on": {"a": "c"},
                "plan": [{"block": "b", "to": "TABLE"}, {"block": "d", "to": "TABLE"}, {"block": "a", "to": "c"}]}
        m.clearance_evidence_check(task, {"action_index": 2, "precondition": "source_not_clear", "enabling_action_index": 0, "obstructing_block": "b"})
        with self.assertRaises(m.EvidenceError): m.clearance_evidence_check(task, {"action_index": 2, "precondition": "destination_not_clear", "enabling_action_index": 1, "obstructing_block": "d"})

    def test_selective_source_and_destination_actual_ablation(self):
        for original, reason in [(source_task(), "source_not_clear"), (destination_task(), "destination_not_clear")]:
            task = copy.deepcopy(original); task["plan"] = task["plan"][::-1]
            result = m.selective_failure_check(task, c1_evidence(task, reason))
            self.assertTrue(result["mutated_step_executed"])

    def test_source_noop_cofailure_not_selective(self):
        task = {"on": {"cap": "base", "base": "TABLE", "free": "TABLE"}, "goal_on": {"cap": "free"},
                "plan": [{"block": "base", "to": "TABLE"}, {"block": "cap", "to": "free"}]}
        with self.assertRaises(m.EvidenceError): m.selective_failure_check(task, c1_evidence(task, "source_not_clear"))

    def test_false_or_integer_guard_truth_rejected(self):
        task = source_task(); task["plan"] = task["plan"][::-1]
        for value in [0, True]:
            e = c1_evidence(task, "source_not_clear"); e["guard_truth"]["source_clear"] = value
            with self.subTest(value=value), self.assertRaises(m.EvidenceError): m.selective_failure_check(task, e)

    def test_isolated_table_noop_and_fixed_position(self):
        task = source_task(); task["plan"] = [task["plan"][0], {"block": "cap", "to": "TABLE"}]
        e = {"action_index": 1, "block": "cap", "current_support": "TABLE"}
        self.assertTrue(m.isolated_noop_check(task, e, 204)["no_op_omission_changes_validity"])
        with self.assertRaises(m.EvidenceError): m.isolated_noop_check(task, e, 205)

    def test_non_table_or_goal_prefix_noop_rejected(self):
        task = source_task(); task["plan"] = [task["plan"][0], {"block": "cap", "to": "TABLE"}]
        e = {"action_index": 1, "block": "cap", "current_support": "TABLE"}
        task["goal_on"] = {"cap": "TABLE"}
        with self.assertRaises(m.EvidenceError): m.isolated_noop_check(task, e, 204)
        task["goal_on"] = {"cap": "free"}; task["on"]["base"] = "free"; task["plan"][1] = {"block": "base", "to": "free"}
        with self.assertRaises(m.EvidenceError): m.isolated_noop_check(task, dict(e, block="base"), 204)

    def test_public_allocation_arithmetic_only(self):
        allocation = [m.expected_allocation(i) for i in range(180, 228)]
        self.assertEqual(sum(x["contexts"] for x in allocation), 204)
        self.assertEqual(sum(2 * x["contexts"] for x in allocation if x["split"] == "train"), 340)
        self.assertEqual(sum(2 * x["contexts"] for x in allocation if x["split"] == "dev_challenge"), 68)
        a_dev = [x["a_subtype"] for x in allocation if x["stratum"] == "A" and x["split"] == "dev_challenge"]
        self.assertEqual(set(a_dev), {"fresh_infeasible", "fuel_lengthened"})
        for reason in ["source_not_clear", "destination_not_clear"]:
            selected = [x for x in allocation if x["c1_reason"] == reason]
            self.assertEqual(len(selected), 6); self.assertEqual(sum(x["objects"] == 5 for x in selected), 3)
            self.assertEqual(sum(x["split"] == "dev_challenge" for x in selected), 1)
        with self.assertRaises(m.EvidenceError): m.expected_allocation(True)


class RendererControlTests(unittest.TestCase):
    def setUp(self):
        self.index, contexts = {}, []
        for context in ["A0", "A1", "A2", "A3"]:
            ids = []
            for style in ["style-one", "style-two"]:
                sid = context + "-" + style; ids.append(sid)
                self.index[sid] = {"renderer": style, "source_family_key": "synthetic-family", "split": "train"}
            contexts.append({"context_id": context, "sample_ids": ids})
        controls = [{"control_id": "control-" + style, "name": "fuel_only_counterfactual", "scope": "corpus_variant", "provenance": "supplied_tasks",
                     "sample_ids": ["A0-" + style, "A1-" + style]} for style in ["style-one", "style-two"]]
        self.family = {"primary_stratum": "A", "source_family_key": "synthetic-family", "split": "train", "contexts": contexts, "controls": controls}

    def test_repeated_control_name_per_style_is_allowed(self):
        report = m.control_renderer_bindings([self.family], self.index)
        self.assertEqual(report["verified_pair_binding_declarations"], 2)
        self.assertFalse(report["semantic_truth_checked"]); self.assertFalse(report["decision_emitted"])

    def test_missing_style_four_ids_duplicate_id_rejected(self):
        for mutation in ["missing", "four", "duplicate_id", "mixed", "unknown"]:
            f = copy.deepcopy(self.family)
            if mutation == "missing": f["controls"] = f["controls"][:1]
            elif mutation == "four": f["controls"] = [dict(f["controls"][0], sample_ids=f["controls"][0]["sample_ids"] + f["controls"][1]["sample_ids"])]
            elif mutation == "duplicate_id": f["controls"][1]["control_id"] = f["controls"][0]["control_id"]
            elif mutation == "mixed": f["controls"][0]["sample_ids"][1] = "A1-style-two"
            else: f["controls"][0]["sample_ids"][1] = "unknown"
            with self.subTest(mutation=mutation), self.assertRaises(m.EvidenceError): m.control_renderer_bindings([f], self.index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
