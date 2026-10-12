"""Synthetic future H3 readout ledgers; no native student outcomes or models."""
import copy
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_recovery_student_costs.py"
spec = importlib.util.spec_from_file_location("_h3_student_cost_tests", SCRIPT)
costs = importlib.util.module_from_spec(spec); spec.loader.exec_module(costs)


def outcome(index, *, hybrid=False):
    candidates = []
    for candidate_index, (g, eos) in enumerate(((5, True), (6, True), (7, True), (80, False))):
        malformed = candidate_index == 3
        raw = {"text": "malformed" if malformed else '{"actions":["add1","finish"]}',
               "model_calls": 1, "input_tokens": 100, "n_tokens": g, "hit_eos": eos,
               "decoder_forwards": g + int(not eos), "decoder_token_positions": 100 + g - 1 + int(not eos)}
        candidates.append({"index": candidate_index, "seed": None if candidate_index == 0 else candidate_index,
                           "raw": raw, "malformed": malformed, "plan": None if malformed else ["add1", "finish"]})
    success = index % 2 == 0
    scores = [{"index": candidate_index, "success": success if candidate_index == 0 else candidate_index == 1,
               "all_legal": candidate_index != 3, "verification": None} for candidate_index in range(4)]
    item = {"task_id": f"task{index}", "parent_world_id": f"parent{index}", "prompt": "synthetic registered prompt",
            "candidates": candidates, "selected_index": 0, "selected_plan": candidates[0]["plan"],
            "model_queries": 4, "input_tokens": 400, "output_tokens": 98, "processed_positions": 495,
            "decoder_forwards": 99, "oracle_feedback": False, "valid_syntax_count": 3,
            "top1_success": success, "pass_at_4": True, "scores": scores}
    if hybrid:
        traces = []
        for state, action, g, eos in ((0, "add1", 5, True), (1, "finish", 32, False)):
            traces.append({"conditioning": {"domain": "register", "rules": {"limit": 32}, "s0": {"x": state}, "actions": [action]},
                           "generated": {"n_tokens": g, "hit_eos": eos, "text": "synthetic forward state"}, "parsed": {}})
        selected = 0 if index % 3 == 0 else None
        item["learned_reranking"] = {"task_id": item["task_id"], "selected_index": selected,
                                     "selected_plan": candidates[0]["plan"] if selected is not None else None,
                                     "forward_queries": 2, "total_model_queries": 6, "processed_positions": 721,
                                     "oracle_feedback": False, "forward_traces": traces,
                                     "success": scores[0]["success"] if selected is not None else False}
    return item


def record(*, hybrid=False):
    rows = [outcome(index, hybrid=hybrid) for index in range(12)]
    result = {"status": "development-evaluated", "n": 12, "outcomes": rows,
              "config": {"student_k": 4, "student_tokens": 80, "max_length": 512, "generation_tokens": 32},
              "oracle_feedback": False, "confirmation_access": False,
              "all_generation_and_learned_reranking_fixed_before_exact_scoring": True,
              "proposal_queries": 48, "proposal_input_tokens": 4800, "proposal_output_tokens": 1176,
              "proposal_processed_positions": 5940, "top1_successes": 6, "pass_at_4_successes": 12,
              "reranking_forward_queries": 24 if hybrid else None,
              "with_search_successes": 2 if hybrid else None}
    if hybrid:
        result.update(forward_adapter_sha256=costs.INITIAL_F_SHA, adapter_switch={"student_adapter_restored": True})
    return result


def token_length(conditioning):
    return 90 if conditioning["s0"]["x"] == 0 else 100


class StudentCostsTest(unittest.TestCase):
    def test_pure_realized_k4_cost_charges_failures_and_keeps_physical_unknown(self):
        original = record()
        reduced = costs.reduce_record(original)
        summary = reduced["pure_top1_readout"]
        self.assertEqual(summary["solved"], 6)
        self.assertEqual(summary["totals_including_failed_tasks"]["model_queries"], 48)
        self.assertEqual(summary["model_queries_per_solved_including_failures"], 8)
        self.assertEqual(summary["totals_including_failed_tasks"]["processed_positions"], 5940)
        self.assertIsNone(summary["physical_decoder_forwards"])
        self.assertIsNone(summary["measured_flops"])
        self.assertIsNone(reduced["learned_reranked_readout"])
        self.assertEqual(reduced["offline_pass4_coverage"]["solved"], 12)
        self.assertEqual(original, record())

    def test_hybrid_combined_raw_costs_include_proposals_and_all_forward_work(self):
        reduced = costs.reduce_record(record(hybrid=True), token_length=token_length)
        first = reduced["per_parent"][0]
        self.assertEqual(first["forward"]["input_tokens"], 190)
        self.assertEqual(first["forward"]["output_tokens"], 37)
        self.assertEqual(first["forward"]["processed_positions"], 226)
        self.assertEqual(first["forward"]["decoder_forwards_estimate"], 38)
        self.assertEqual(first["combined"], {"model_queries": 6, "input_tokens": 590, "output_tokens": 135,
                                           "processed_positions": 721, "decoder_forwards_estimate": 137})
        self.assertTrue(first["combined_positions_reconstructed"])
        summary = reduced["learned_reranked_readout"]
        self.assertEqual(summary["solved"], 2)
        self.assertEqual(summary["totals_including_failed_tasks"]["model_queries"], 72)
        self.assertEqual(summary["model_queries_per_solved_including_failures"], 36)
        self.assertIsNone(summary["physical_processed_token_positions"])

    def test_missing_forward_input_lengths_are_null_not_silently_zero(self):
        reduced = costs.reduce_record(record(hybrid=True))
        first = reduced["per_parent"][0]
        self.assertIsNone(first["forward"]["input_tokens"])
        self.assertIsNone(first["combined"]["input_tokens"])
        self.assertFalse(first["combined_positions_reconstructed"])
        self.assertEqual(first["combined"]["processed_positions"], 721)  # recorded mixed formula estimate
        self.assertIsNone(reduced["learned_reranked_readout"]["totals_including_failed_tasks"]["input_tokens"])

    def test_query_cap_total_identity_and_reconstructed_positions_fail_closed(self):
        for field, value in (("total_model_queries", 7), ("forward_queries", 27), ("processed_positions", 120001)):
            with self.subTest(field=field):
                changed = record(hybrid=True)
                changed["outcomes"][0]["learned_reranking"][field] = value
                with self.assertRaises(ValueError):
                    costs.reduce_record(changed, token_length=token_length)
        changed = record(hybrid=True)
        changed["outcomes"][0]["learned_reranking"]["processed_positions"] += 1
        with self.assertRaisesRegex(ValueError, "position formula disagrees"):
            costs.reduce_record(changed, token_length=token_length)
        changed = record()
        changed["outcomes"][0]["candidates"][0]["raw"]["decoder_forwards"] += 1
        with self.assertRaisesRegex(ValueError, "formula counters disagree"):
            costs.reduce_record(changed)

    def test_pure_hybrid_selection_identity_and_bool_counters_are_strict(self):
        changed = record(hybrid=True)
        changed["outcomes"][0]["learned_reranking"]["selected_plan"] = ["sub1", "finish"]
        with self.assertRaises(ValueError):
            costs.reduce_record(changed)
        changed = record(hybrid=True)
        changed["outcomes"][0].pop("learned_reranking")
        with self.assertRaises(ValueError):
            costs.reduce_record(changed)
        changed = record()
        changed["outcomes"][0]["candidates"][0]["raw"]["model_calls"] = True
        with self.assertRaises(ValueError):
            costs.reduce_record(changed)
        changed = record(hybrid=True)
        changed["adapter_switch"]["student_adapter_restored"] = False
        with self.assertRaises(ValueError):
            costs.reduce_record(changed)

    def test_no_solved_case_and_refused_proposals_do_not_create_success_denominators(self):
        changed = record()
        for item in changed["outcomes"]:
            item["top1_success"] = False
            item["scores"][0]["success"] = False
        changed["top1_successes"] = 0
        reduced = costs.reduce_record(changed)
        self.assertIsNone(reduced["pure_top1_readout"]["model_queries_per_solved_including_failures"])
        item = copy.deepcopy(changed["outcomes"][0])
        for candidate in item["candidates"]:
            candidate["raw"] = {"text": None, "error": "overlength", "model_calls": 0, "input_tokens": 0, "n_tokens": 0}
            candidate.update(malformed=True, plan=None)
        item.update(selected_plan=None, model_queries=0, input_tokens=0, output_tokens=0,
                    processed_positions=0, decoder_forwards=0)
        self.assertEqual(costs.proposal_ledger(item)["model_queries"], 0)


if __name__ == "__main__":
    unittest.main()
