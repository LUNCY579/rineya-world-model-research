"""Explicit synthetic future outcomes; no native results, models or oracles."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from dsv3wm.domains import parent_world_key, rules_for, sha256 as object_hash
from dsv3wm.native import EXPECTED_CORE_SHA
from dsv3wm.recovery import RecoveryConfig, INITIAL_F_SHA
from dsv3wm.recovery_student import StudentConfig, registered_prompt, shared_orders

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_h3_offline_gate_tests", ROOT / "scripts/analyze_recovery_student_gates.py")
gates = importlib.util.module_from_spec(spec); spec.loader.exec_module(gates)


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def jsonl(rows):
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def aggregate_proposer(record):
    rows = record["outcomes"]
    record.update(n=12, top1_successes=sum(row["top1_success"] for row in rows), pass_at_4_successes=sum(row["pass_at_4"] for row in rows),
        with_search_successes=sum(row["learned_reranking"]["success"] for row in rows),
        valid_syntax_count=sum(row["valid_syntax_count"] for row in rows),
        proposal_queries=sum(row["model_queries"] for row in rows), proposal_input_tokens=sum(row["input_tokens"] for row in rows),
        proposal_output_tokens=sum(row["output_tokens"] for row in rows), proposal_processed_positions=sum(row["processed_positions"] for row in rows),
        reranking_forward_queries=sum(row["learned_reranking"]["forward_queries"] for row in rows))


def aggregate_guard(record):
    outcomes = record["outcomes"]
    outcomes["summary"] = {}
    for cohort in ("id", "ood", "all"):
        selected = [row for row in outcomes["per_parent"] if cohort == "all" or row["cohort"] == cohort]
        outcomes["summary"][cohort] = {h: {"n": len(selected), **{field: sum(row["horizons"][h][field] for row in selected)
            for field in selected[0]["horizons"][h]}} for h in ("1", "2", "4")}
    outcomes.update(behavior_n=6, behavior_correct=sum(row["correct"] for row in outcomes["behavior"]), retention_pass=None)


class StudentGateTests(unittest.TestCase):
    def fixture(self, *, arm="A3", control=False, successes=9, warm_successes=7):
        cfg = StudentConfig(arm=arm, plan_weight=0 if control else 1, control_extra_one_step=control)
        warm_cfg = StudentConfig(arm=arm)
        rules = rules_for("register", limit=32)
        dev_rows = [{"id": f"source-{i}", "parent_world_id": parent_world_key("register", {"x": i}, rules),
                     "domain": "register", "split": "dev", "rules": rules, "s0": {"x": i}, "actions": ["add1"] * 4,
                     "provenance": {"cohort": "recovery_id_teacher_known" if i < 12 else "recovery_ood_limit"}} for i in range(24)]
        tasks = [{"id": f"task-{i}", "parent_world_id": row["parent_world_id"], "domain": "register", "split": "dev",
                  "rules": rules, "s0": row["s0"], "goal": {"kind": "state", "state": {"x": row["s0"]["x"] + 1}},
                  "provenance": {"source_trajectory_id": row["id"]}} for i, row in enumerate(dev_rows[:6] + dev_rows[12:18])]
        training = []
        for i in range(128):
            limit = (24, 32, 40)[min(i // 43, 2)]
            source_rules, state = rules_for("register", limit=limit), {"x": -limit + i % 43}
            training.append({"id": f"explicit-synthetic-train-{i}", "split": "train", "domain": "register",
                "s0": state, "rules": source_rules, "actions": ["add1"] * 4, "labels": {"explicit_synthetic": True},
                "parent_world_id": parent_world_key("register", state, source_rules)})
        actions = ("add1", "sub1", "double", "halve")
        sequences = [[a, b, "finish"] for a in actions for b in actions] + [[a, "finish"] for a in actions]
        positives = []
        for i, source in enumerate(training[:9] + training[43:52] + training[86:95]):
            positives.append({key: copy.deepcopy(source[key]) for key in ("parent_world_id", "domain", "rules", "s0", "split")})
            positives[-1].update(verified=True, plan=sequences[i % 20],
                                provenance={"training_input_sha256": object_hash(source)})
        hashes = {"train": digest(jsonl(training)), "verified": digest(jsonl(positives)), "tasks": digest("explicit synthetic TRAIN tasks\n")}
        manifest = {"status": "corpus-qualified", "train_file_sha256": hashes["train"], "tasks_file_sha256": hashes["tasks"],
            "output_sha256": {"verified.jsonl": hashes["verified"]},
            "training_row_allowlist": {row["parent_world_id"]: object_hash(row) for row in training},
            "gate": {"passed": True, "accepted_train_parents": 27, "covered_limits": [24, 32, 40], "distinct_action_sequences": 20}}
        hashes["manifest"] = digest(json.dumps(manifest, sort_keys=True) + "\n")
        trajectory_order, plan_order = shared_orders(128, 27, cfg.steps, cfg.seed)
        warm_run = {"status": "pilot-trained", "config": RecoveryConfig(arm=cfg.warm_arm).dict(), "optimizer_steps": 32,
                    "core_sha256": EXPECTED_CORE_SHA, "adapter_sha256": digest("warm weights"), "initial_adapter_sha256": INITIAL_F_SHA,
                    "data_sha256": hashes["train"]}
        student_run = {"status": "student-pilot-trained", "config": cfg.dict(), "optimizer_steps": 32,
                       "core_sha256": EXPECTED_CORE_SHA, "adapter_sha256": digest("control weights" if control else "student weights"),
                       "adapter_config_sha256": digest("student PEFT"), "warm_arm": cfg.warm_arm,
                       "training_kind": gates.kind(cfg), "corpus_gate": {"passed": True, "parents": 27, "limits": [24, 32, 40], "sequences": 20},
                       "warm_run_sha256": digest("warm run"), "warm_adapter_sha256": warm_run["adapter_sha256"],
                       "warm_adapter_config_sha256": digest("warm PEFT"), "train_sha256": warm_run["data_sha256"],
                       "tasks_sha256": hashes["tasks"], "teacher_manifest_sha256": hashes["manifest"],
                       "verified_corpus_sha256": hashes["verified"], "trajectory_order": trajectory_order, "plan_order": plan_order}
        def proposer(*, warm):
            run, used_cfg = (warm_run, warm_cfg) if warm else (student_run, cfg)
            kind = "warm-parent-proposer-baseline" if warm else ("h3-continuation-control-" + gates.kind(cfg) if control else "h3-trained-student-proposer")
            rows = []
            for i, task in enumerate(tasks):
                top1 = i < (warm_successes if warm else successes)
                plans = [["add1", "finish"] if top1 else ["sub1", "finish"], ["add1", "finish"], ["add1", "add1", "sub1", "finish"], None]
                candidates, scores = [], []
                for index, plan in enumerate(plans):
                    g = 5
                    raw = {"text": json.dumps({"actions": plan}) if plan else "explicit malformed mock", "model_calls": 1,
                           "input_tokens": 100, "n_tokens": g, "hit_eos": True, "decoder_forwards": g, "decoder_token_positions": 104}
                    seed = None if index == 0 else int(hashlib.sha256(f"0|{task['parent_world_id']}|0|{index}".encode()).hexdigest()[:15], 16)
                    candidates.append({"index": index, "seed": seed, "raw": raw, "malformed": plan is None, "plan": plan})
                    success = top1 if index == 0 else index in {1, 2}
                    scores.append({"index": index, "success": success, "all_legal": plan is not None,
                                   "verification": {"synthetic": True, "success": success, "all_legal": True,
                                       "labels": {"terminated": [False] * (len(plan) - 1) + [True]}} if plan is not None else None})
                traces = [{"conditioning": {"domain": "register", "rules": rules, "s0": {"x": index}, "actions": ["add1"]},
                           "generated": {"text": "synthetic prediction", "n_tokens": 5, "hit_eos": True}, "parsed": {}} for index in range(8)]
                rows.append({"task_id": task["id"], "parent_world_id": task["parent_world_id"], "prompt": registered_prompt(task),
                    "candidates": candidates, "scores": scores, "selected_index": 0, "selected_plan": plans[0],
                    "model_queries": 4, "input_tokens": 400, "output_tokens": 20, "processed_positions": 416, "decoder_forwards": 20,
                    "oracle_feedback": False, "valid_syntax_count": 3, "top1_success": top1, "pass_at_4": True,
                    "learned_reranking": {"task_id": task["id"], "selected_index": 1, "selected_plan": plans[1],
                        "success": True, "forward_queries": 8, "total_model_queries": 12, "processed_positions": 1248,
                        "oracle_feedback": False, "forward_traces": traces}})
            result = {"status": "development-evaluated", "config": used_cfg.dict(), "readout_kind": kind,
                "run_sha256": digest("warm run" if warm else "student run"), "adapter_sha256": run["adapter_sha256"],
                "adapter_config_sha256": digest("warm PEFT") if warm else run["adapter_config_sha256"],
                "tasks_sha256": digest("DEV planning tasks"), "artifact_arm": run["config"]["arm"],
                "artifact_training_config": run["config"], "artifact_training_status": run["status"],
                "forward_adapter_sha256": INITIAL_F_SHA, "adapter_switch": {"student_adapter_restored": True},
                "oracle_feedback": False, "confirmation_access": False,
                "all_generation_and_learned_reranking_fixed_before_exact_scoring": True, "outcomes": rows}
            aggregate_proposer(result)
            return result
        def guard_record(proposer, *, warm):
            result = {key: copy.deepcopy(proposer[key]) for key in ("config", "run_sha256", "adapter_sha256", "adapter_config_sha256",
                       "artifact_arm", "artifact_training_status", "readout_kind")}
            result.update(schema=gates.GUARD_SCHEMA, status="development-guards-evaluated", warm_baseline=warm,
                artifact_training_kind="original-warm-dynamics" if warm else gates.kind(cfg), data_sha256=gates.DEV_SHA,
                fixtures_sha256=gates.FIXTURE_SHA, oracle_feedback=False, confirmation_access=False,
                all_generation_fixed_before_scoring=True, retention_pass=None)
            fields = ("rollout_correct", "rollout_malformed", "legality_correct", "termination_correct", "direct_correct",
                      "direct_malformed", "direct_legality_correct", "direct_termination_correct")
            parents = [{"parent_world_id": row["parent_world_id"], "cohort": "id" if i < 12 else "ood",
                        "horizons": {h: {field: False if "malformed" in field else True for field in fields} for h in ("1", "2", "4")}}
                       for i, row in enumerate(dev_rows)]
            behavior = [{"id": f"fixture-{i}", "domain": "register" if i < 3 else "ledger", "correct": i < 3} for i in range(6)]
            result["outcomes"] = {"per_parent": parents, "behavior": behavior}
            aggregate_guard(result)
            return result
        student, warm = proposer(warm=False), proposer(warm=True)
        return {"student_proposer": student, "warm_proposer": warm, "student_guards": guard_record(student, warm=False),
                "warm_guards": guard_record(warm, warm=True), "student_run": student_run, "warm_run": warm_run,
                "tasks": tasks, "dev_rows": dev_rows, "training_rows": training, "corpus_rows": positives,
                "corpus_manifest": manifest, "bound_hashes": hashes}

    def test_fixed_top1_gain_and_descriptive_uncertainty_without_model_or_oracle(self):
        inputs = self.fixture()
        original = copy.deepcopy(inputs)
        with patch("dsv3wm.native.load", side_effect=AssertionError("no native model")), \
             patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")), \
             patch("dsv3wm.domains.rollout_labels", side_effect=AssertionError("no oracle")):
            result = gates.analyze(**inputs)
        self.assertEqual(inputs, original)
        self.assertTrue(result["quantitative_conditions_met"])
        self.assertEqual(result["greedy_syntax"]["correct"], 12)
        paired = result["pure_top1_vs_own_warm"]
        self.assertEqual((paired["gains"], paired["losses"], paired["net_count"]), (2, 0, 2))
        self.assertEqual(paired["p_two_sided"], .5)
        self.assertEqual(len(paired["candidate_wilson95"]), 2)
        self.assertEqual(result["offline_pass4_coverage"]["student"], 12)
        self.assertFalse(result["confirmed"])
        self.assertIsNone(result["retention_pass"])

    def test_pass4_coverage_cannot_replace_failed_greedy_gain_or_syntax_gate(self):
        inputs = self.fixture(successes=7, warm_successes=7)
        result = gates.analyze(**inputs)
        self.assertEqual(result["offline_pass4_coverage"]["student"], 12)
        self.assertFalse(result["criteria"]["paired_top1_net_gain_at_least_2_of_12"])
        # Eight syntactically valid greedy candidates, while other sampled plans
        # cover every goal. This remains an operational syntax failure.
        inputs = self.fixture(successes=7)
        for row in inputs["student_proposer"]["outcomes"][8:]:
            row["candidates"][0].update(malformed=True, plan=None)
            row["candidates"][0]["raw"]["text"] = "explicit malformed mock"
            row["selected_plan"] = None; row["valid_syntax_count"] -= 1
            row["scores"][0].update(success=False, all_legal=False, verification=None)
        aggregate_proposer(inputs["student_proposer"])
        result = gates.analyze(**inputs)
        self.assertEqual(result["greedy_syntax"]["correct"], 8)
        self.assertFalse(result["criteria"]["greedy_syntax_at_least_9_of_12"])

    def test_any_hybrid_lost_success_fails_and_behavior_loss_requires_review(self):
        inputs = self.fixture()
        row = inputs["student_proposer"]["outcomes"][0]["learned_reranking"]
        row.update(selected_index=None, selected_plan=None, success=False)
        aggregate_proposer(inputs["student_proposer"])
        result = gates.analyze(**inputs)
        self.assertFalse(result["criteria"]["no_lost_matched_hybrid_success"])
        self.assertEqual(result["matched_fixed_B0_hybrid_vs_own_warm"]["losses"], 1)
        inputs = self.fixture()
        inputs["student_guards"]["outcomes"]["behavior"][0]["correct"] = False
        aggregate_guard(inputs["student_guards"])
        result = gates.analyze(**inputs)
        self.assertTrue(result["quantitative_conditions_met"])
        self.assertTrue(result["behavior_review_pending"])
        self.assertEqual(result["paired_dynamics_behavior"]["behavior"]["losses"], ["fixture-0"])
        self.assertIn("review required", result["classification"])
        self.assertEqual(len(result["paired_dynamics_behavior"]["behavior"]["unqualified_baseline_cases"]), 3)

    def test_id_h4_guard_uses_paired_net_loss_and_reports_cases(self):
        inputs = self.fixture()
        for row in inputs["student_guards"]["outcomes"]["per_parent"][:3]:
            row["horizons"]["4"]["rollout_correct"] = False
        aggregate_guard(inputs["student_guards"])
        result = gates.analyze(**inputs)
        self.assertFalse(result["criteria"]["id_h4_net_loss_at_most_2_of_12"])
        self.assertEqual(result["paired_dynamics_behavior"]["id_h4_net_loss"], 3)

    def test_full_typed_control_identity_allows_own_warm_default_config(self):
        for arm, expected in (("A3", "C3"), ("A4", "C4")):
            with self.subTest(arm=arm):
                inputs = self.fixture(arm=arm, control=True)
                result = gates.analyze(**inputs)
                self.assertEqual(result["artifact_kind"], expected)
                self.assertFalse(result["gate_applies_to_distilled_student"])
                self.assertTrue(result["quantitative_conditions_met"])
                inputs["student_guards"]["artifact_training_kind"] = arm
                with self.assertRaisesRegex(ValueError, "control identity"):
                    gates.analyze(**inputs)
        inputs = self.fixture(); inputs["student_proposer"]["config"]["plan_weight"] = True
        with self.assertRaises(ValueError): gates.analyze(**inputs)

    def test_optional_actual_control_binds_shared_sources_orders_and_warm_weights(self):
        inputs = self.fixture(); control = self.fixture(control=True, successes=7)
        options = {"control_proposer": control["student_proposer"], "control_guards": control["student_guards"], "control_run": control["student_run"]}
        result = gates.analyze(**inputs, **options)
        self.assertEqual(result["control_contrast"]["kind"], "C3")
        self.assertEqual(result["control_contrast"]["pure_top1"]["net_count"], 2)
        control["student_run"]["trajectory_order"] = list(reversed(range(32)))
        with self.assertRaisesRegex(ValueError, "deterministic shared_orders"):
            gates.analyze(**inputs, **options)

    def test_unrelated_control_guard_parent_and_out_of_range_sampling_orders_refuse(self):
        inputs = self.fixture(); control = self.fixture(control=True)
        control["student_guards"]["outcomes"]["per_parent"][-1]["parent_world_id"] = digest("unrelated unused control parent")
        with self.assertRaisesRegex(ValueError, "exact fixed DEV24 parents"):
            gates.analyze(**inputs, control_proposer=control["student_proposer"],
                          control_guards=control["student_guards"], control_run=control["student_run"])
        for key in ("trajectory_order", "plan_order"):
            inputs = self.fixture(); inputs["student_run"][key] = [1000000] * 32
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "deterministic shared_orders"):
                gates.analyze(**inputs)
        inputs = self.fixture(); inputs["corpus_rows"].pop()
        with self.assertRaisesRegex(ValueError, "qualified27"):
            gates.analyze(**inputs)
        inputs = self.fixture(); inputs["bound_hashes"]["train"] = digest("different actual source bytes")
        with self.assertRaisesRegex(ValueError, "actual hash-bound sampling input"):
            gates.analyze(**inputs)

    def test_control_behavior_requires_exact_warm_student_fixture_id_domain_map(self):
        for change in ("unrelated_id", "swapped_domains"):
            with self.subTest(change=change):
                inputs = self.fixture(); control = self.fixture(control=True)
                behavior = control["student_guards"]["outcomes"]["behavior"]
                if change == "unrelated_id":
                    behavior[0]["id"] = "unrelated control behavior fixture"
                else:
                    behavior[0]["domain"], behavior[3]["domain"] = behavior[3]["domain"], behavior[0]["domain"]
                with self.assertRaisesRegex(ValueError, "exact warm/student six-fixture IDs/domains"):
                    gates.analyze(**inputs, control_proposer=control["student_proposer"],
                                  control_guards=control["student_guards"], control_run=control["student_run"])

    def test_hash_parent_cohort_raw_syntax_and_scoring_boundary_corruption_refuse(self):
        alterations = [
            lambda d: d["student_run"].update(warm_adapter_sha256=digest("wrong warm weights")),
            lambda d: d["student_guards"].update(adapter_sha256=digest("wrong student weights")),
            lambda d: d["student_proposer"].update(tasks_sha256=digest("different task pool")),
            lambda d: d["student_proposer"].update(all_generation_and_learned_reranking_fixed_before_exact_scoring=False),
            lambda d: d["student_guards"]["outcomes"]["per_parent"][0].update(parent_world_id=digest("wrong parent")),
            lambda d: d["student_proposer"]["outcomes"][0]["candidates"][0].update(seed=1),
            lambda d: d["student_proposer"]["outcomes"][0]["scores"][0]["verification"].update(all_legal=False),
            lambda d: d["tasks"][0].update(split="confirmation"),
        ]
        for change in alterations:
            with self.subTest(change=change):
                inputs = self.fixture(); change(inputs)
                with self.assertRaises((ValueError, PermissionError)):
                    gates.analyze(**inputs)

    def test_missing_hybrid_evidence_and_overwrite_are_not_silent_passes(self):
        inputs = self.fixture()
        for row in inputs["student_proposer"]["outcomes"]: row.pop("learned_reranking")
        inputs["student_proposer"].update(with_search_successes=None, reranking_forward_queries=None)
        with self.assertRaisesRegex(ValueError, "hybrid readout required"):
            gates.analyze(**inputs)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "existing.json"; out.write_text("preserve")
            argv = ["analyze_recovery_student_gates.py", "--student-proposer", "missing", "--warm-proposer", "missing",
                    "--student-guards", "missing", "--warm-guards", "missing", "--out", str(out)]
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError): gates.main()
            self.assertEqual(out.read_text(), "preserve")

    def test_offline_cli_binds_real_synthetic_file_hashes_and_rejects_changed_run(self):
        inputs = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary); root = repo / "dsv3wm"
            dataset = root / "datasets/recovery-v1"; dataset.mkdir(parents=True)
            task_path, dev_path = dataset / "planning_dev.jsonl", dataset / "dev.jsonl"
            task_path.write_text("".join(json.dumps(row) + "\n" for row in inputs["tasks"]))
            dev_path.write_text("".join(json.dumps(row) + "\n" for row in inputs["dev_rows"]))
            fixtures = root / "results/retention_core_v1.fixtures.json"; fixtures.parent.mkdir()
            fixtures.write_text('{"explicit_synthetic_fixture_bytes":true}')
            train_path = dataset / "train.jsonl"; train_path.write_bytes(jsonl(inputs["training_rows"]).encode())
            training_tasks = root / "datasets/recovery-h3-v1/planning_train.jsonl"; training_tasks.parent.mkdir()
            training_tasks.write_bytes(b"explicit synthetic TRAIN tasks\n")
            teacher = root / "artifacts/recovery-h3-v1/explicit_synthetic_teacher"; teacher.mkdir(parents=True)
            (teacher / "verified.jsonl").write_bytes(jsonl(inputs["corpus_rows"]).encode())
            (teacher / "manifest.json").write_bytes((json.dumps(inputs["corpus_manifest"], sort_keys=True) + "\n").encode())
            inputs["student_run"]["command"] = ["explicit synthetic trainer", "--train", str(train_path),
                                               "--tasks", str(training_tasks), "--teacher-dir", str(teacher)]
            run_paths = {}
            for name in ("warm_run", "student_run"):
                path = repo / (name + ".json"); run_paths[name] = path
                if name == "student_run": inputs[name]["warm_run_sha256"] = gates.file_hash(run_paths["warm_run"])
                path.write_text(json.dumps(inputs[name]))
            record_paths = {}
            for name in ("student_proposer", "warm_proposer", "student_guards", "warm_guards"):
                record = inputs[name]; run_path = run_paths["warm_run" if name.startswith("warm_") else "student_run"]
                record["run_sha256"] = gates.file_hash(run_path)
                record["command"] = ["explicit-synthetic-input", "--run", str(run_path)]
                if name.endswith("proposer"):
                    record["tasks_sha256"] = gates.file_hash(task_path)
                    record["command"] += ["--tasks", str(task_path)]
                else:
                    record["data_sha256"], record["fixtures_sha256"] = gates.file_hash(dev_path), gates.file_hash(fixtures)
                    record["command"] += ["--data", str(dev_path), "--fixtures", str(fixtures)]
                path = repo / (name + ".json"); path.write_text(json.dumps(record)); record_paths[name] = path
            for name in ("scripts/analyze_recovery_student_gates.py", "src/dsv3wm/stats.py", "src/dsv3wm/recovery_student.py",
                         "scripts/analyze_recovery_student_costs.py", "scripts/evaluate_recovery_student_guards.py"):
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((ROOT / name).read_bytes())
            out = repo / "synthetic_gate_report.json"
            argv = ["analyze_recovery_student_gates.py"]
            for name, path in record_paths.items(): argv += ["--" + name.replace("_", "-"), str(path)]
            argv += ["--out", str(out)]
            with patch.object(gates, "ROOT", root), patch.object(gates, "REPO", repo), \
                 patch.object(gates, "DEV_SHA", gates.file_hash(dev_path)), patch.object(gates, "FIXTURE_SHA", gates.file_hash(fixtures)), \
                 patch.object(sys, "argv", argv), patch("dsv3wm.native.load", side_effect=AssertionError("no native")), \
                 patch("dsv3wm.domains.exact_transition", side_effect=AssertionError("no oracle")):
                gates.main()
                report = json.loads(out.read_text())
                self.assertTrue(report["quantitative_conditions_met"])
                self.assertFalse(report["confirmed"])
                self.assertEqual(report["input_sha256"][str(run_paths["student_run"].resolve())], gates.file_hash(run_paths["student_run"]))
                run_paths["student_run"].write_text('{"changed_run":true}')
                argv[-1] = str(repo / "must_not_be_created.json")
                with self.assertRaisesRegex(ValueError, "run metadata path/hash"):
                    gates.main()
                self.assertFalse((repo / "must_not_be_created.json").exists())


if __name__ == "__main__":
    unittest.main()
