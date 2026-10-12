"""TRAIN-only learned teacher collection; exact verification begins after all searches.

Inputs and teacher identity require a prospective freeze. Candidates come solely
from already-computed learned nodes; there is no exact repair or oracle proposer.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def validate_train_allowlist(tasks, train_rows):
    from dsv3wm.domains import canonical, parent_world_key
    from dsv3wm.recovery_planning import validate_tasks
    validate_tasks(tasks, expected_split="train")
    approved = {}
    for row in train_rows:
        if row.get("split") != "train" or row.get("domain") != "register":
            raise PermissionError("Teacher collection requires approved TRAIN RegisterWorld rows only")
        parent = parent_world_key("register", row["s0"], row["rules"])
        if row["parent_world_id"] != parent or parent in approved:
            raise ValueError("Approved training parents must be canonical and unique")
        approved[parent] = row
    selected = {}
    for task in tasks:
        row = approved.get(task["parent_world_id"])
        if row is None or task["provenance"].get("source_trajectory_id") != row["id"]:
            raise ValueError("Teacher task is outside the frozen training row allowlist")
        if any(canonical(task[key]) != canonical(row[key]) for key in ("domain", "rules", "s0")):
            raise ValueError("Teacher task input disagrees with its approved training parent")
        if task["goal"]["state"] == task["s0"]:
            raise ValueError("Teacher goals must be nontrivial")
        selected[task["id"]] = row
    return selected


def rank_computed_candidates(tasks, search):
    """Pure ranking of completed learned-node plans, fixed before any verification."""
    from dsv3wm.domains import canonical, sha256
    from dsv3wm.recovery_planning import ACTIONS
    if search.get("oracle_feedback") is not False or search.get("all_choices_fixed_before_scoring") is not True:
        raise ValueError("All learned searches must finish before teacher candidate ranking")
    lookup = {task["id"]: task for task in tasks}
    by_task = {task["id"]: {} for task in tasks}
    for trace in search["traces"]:
        task = lookup.get(trace["task_id"])
        if task is None or trace["parent_world_id"] != task["parent_world_id"]:
            raise ValueError("Learned trace/task provenance mismatch")
        source_plan = [*trace["parent_plan"], trace["action"]]
        parsed = trace["raw"]["parsed"]
        valid = (isinstance(parsed, dict) and not parsed.get("malformed")
                 and isinstance(parsed.get("state"), dict) and set(parsed["state"]) == {"x"}
                 and type(parsed["state"]["x"]) is int
                 and all(type(parsed.get(key)) is bool for key in ("legal", "terminated", "failed")))
        predicted_goal = bool(valid and parsed["state"] == task["goal"]["state"])
        eligible = bool(predicted_goal and parsed["legal"] and not parsed["failed"])
        distance = abs(parsed["state"]["x"] - task["goal"]["state"]["x"]) if valid else None
        plan = list(source_plan)
        origin, refusal = "already-final-finish", None
        if not plan or any(action not in ACTIONS for action in plan):
            refusal = "invalid_action_syntax"
        elif "finish" in plan[:-1]:
            refusal = "interior_finish"
        elif plan[-1] != "finish":
            if predicted_goal:
                plan.append("finish")
                origin = "deterministic-final-finish-completion-before-verification"
            else:
                origin = "no-completion-learned-goal-not-predicted"
        if len(plan) > 6:
            refusal = "overlength_completed_plan"
        candidate = {"task_id": task["id"], "parent_world_id": task["parent_world_id"],
                     "source_plan": source_plan, "plan": plan, "completion_origin": origin,
                     "grammar_refusal": refusal, "predicted_goal": predicted_goal,
                     "eligible_for_positive": eligible and refusal is None,
                     "predicted_distance": distance, "prediction_valid": valid,
                     "source_trace_sha256": sha256(trace), "source_prediction": copy.deepcopy(parsed)}
        # Match learned goal-distance/length/lexicographic ranking, with malformed
        # endpoints last. Exact labels never determine this ordering.
        key = (not predicted_goal, not valid, distance if distance is not None else 0,
               len(plan), canonical(plan), sha256(candidate))
        identity = canonical(plan)
        old = by_task[task["id"]].get(identity)
        if old is None or key < old[0]:
            by_task[task["id"]][identity] = (key, candidate)
    ranked = {}
    for identifier, values in by_task.items():
        rows = [candidate for _, candidate in sorted(values.values(), key=lambda item: item[0])]
        for rank, candidate in enumerate(rows):
            candidate["learned_rank"] = rank
            candidate["candidate_sha256"] = sha256(candidate)
        ranked[identifier] = rows
    return ranked


def corpus_gate(positives):
    from dsv3wm.domains import sha256
    if any(row.get("split") != "train" or row.get("verified") is not True for row in positives):
        raise ValueError("The student gate accepts verified TRAIN positives only")
    parents = {row["parent_world_id"] for row in positives}
    limits = {row["rules"]["limit"] for row in positives}
    sequences = {sha256(row["plan"]) for row in positives}
    criteria = {"accepted_train_parents": len(parents) >= 16, "covered_limits": len(limits) >= 2,
                "distinct_action_sequences": len(sequences) >= 4}
    return {"passed": all(criteria.values()), "criteria": criteria, "accepted_train_parents": len(parents),
            "covered_limits": sorted(limits), "distinct_action_sequences": len(sequences),
            "requirements": {"accepted_train_parents": 16, "covered_limits": 2, "distinct_action_sequences": 4},
            "on_failure": "preserve failed corpus; no student training or outcome-driven task-pool expansion"}


def collect_corpus(tasks, train_rows, predictor, *, teacher_provenance, verifier=None, on_rankings_fixed=None):
    """Finish every learned search/ranking, then verify fixed candidates once."""
    from dsv3wm.domains import _semantics, sha256
    from dsv3wm.recovery_planning import batched_search
    approved = validate_train_allowlist(tasks, train_rows)
    search = batched_search(tasks, predictor, expected_split="train")
    ranked = rank_computed_candidates(tasks, search)
    rankings_sha = sha256(ranked)
    if on_rankings_fixed is not None:
        on_rankings_fixed(copy.deepcopy(search), copy.deepcopy(ranked))
    # No learned model call can occur below this boundary.
    if verifier is None:
        from dsv3wm.planning import verify_plan
        verifier = verify_plan
    register = _semantics("register")
    positives, negatives, unretained = [], [], []
    attempts = []
    for task in tasks:
        source, retained = approved[task["id"]], False
        inputs = {"domain": "register", "rules": copy.deepcopy(task["rules"]),
                  "s0": copy.deepcopy(task["s0"]), "actions": []}
        for candidate in ranked[task["id"]]:
            verification = None
            if candidate["grammar_refusal"] is None:
                verification = verifier(inputs, copy.deepcopy(task["goal"]), copy.deepcopy(candidate["plan"]))
            exact_success = bool(verification and verification["success"] and verification["all_legal"]
                                 and verification["labels"]["terminated"][-1])
            provenance = {**copy.deepcopy(teacher_provenance), "training_input_sha256": sha256(source),
                          "task_sha256": sha256(task), "candidate_sha256": candidate["candidate_sha256"],
                          "all_rankings_sha256": rankings_sha, "source_trace_sha256": candidate["source_trace_sha256"],
                          "model_calls": next(row["model_queries"] for row in search["choices"] if row["task_id"] == task["id"]),
                          "budget": search["budget"], "selection": "learned-ranked-then-independent-exact",
                          "learned_rank": candidate["learned_rank"], "completion_origin": candidate["completion_origin"],
                          "learned_predicted_goal": candidate["predicted_goal"],
                          "all_searches_and_rankings_fixed_before_verification": True}
            common = {"split": "train", "domain": "register", "rules": copy.deepcopy(task["rules"]),
                      "s0": copy.deepcopy(task["s0"]), "goal": copy.deepcopy(task["goal"]),
                      "parent_world_id": task["parent_world_id"], "source_plan": copy.deepcopy(candidate["source_plan"]),
                      "plan": copy.deepcopy(candidate["plan"]), "verified": exact_success,
                      "verification": verification, "provenance": provenance}
            if exact_success and candidate["eligible_for_positive"] and not retained:
                common.update(prompt=register.proposer_prompt(task["s0"]["x"], task["goal"]["state"]["x"],
                                                              task["rules"]["limit"], 6),
                              target=register.target_actions(candidate["plan"]), target_contract="registered-rw2-actions-final-finish")
                positives.append(common)
                retained = True
                disposition = "first_exact_positive_in_fixed_eligible_learned_ranking"
            elif exact_success:
                common["reason"] = "keep_one_already_satisfied" if retained else "teacher_did_not_predict_legal_goal"
                unretained.append(common)
                disposition = "verified_but_unretained"
            else:
                common["reason"] = candidate["grammar_refusal"] or "exact_illegal_nonterminating_or_goal_failure"
                common["eligible_for_student_ce"] = False
                negatives.append(common)
                disposition = "hard_negative" if verification is not None else "grammar_rejection"
            attempts.append({"task_id": task["id"], "parent_world_id": task["parent_world_id"],
                             "candidate_sha256": candidate["candidate_sha256"], "learned_rank": candidate["learned_rank"],
                             "disposition": disposition, "exact_success": exact_success})
    if sha256(ranked) != rankings_sha:
        raise RuntimeError("Candidate ranking changed during exact verification")
    return {"verified": positives, "hard_negatives": negatives, "verified_unretained": unretained,
            "attempts": attempts, "ranked_candidates": ranked, "rankings_sha256": rankings_sha,
            "search": search, "gate": corpus_gate(positives), "goal_pool_sha256": sha256(tasks),
            "training_row_allowlist": {task["parent_world_id"]: sha256(approved[task["id"]]) for task in tasks}}


def main():
    from dsv3wm import REPO
    from dsv3wm.data import load_records
    from dsv3wm.native import EXPECTED_CORE_SHA, runtime, sha256, verify_freeze
    from dsv3wm.native_cpu11 import admission_profile, last_admission_check, load
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.recovery import LegacyPredictor, RecoveryConfig, load_warm
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "adapter", "tasks", "train", "freeze", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = RecoveryConfig.read(args.config)
    if cfg.device != "cpu" or cfg.max_length != 512 or cfg.generation_tokens != 32:
        raise ValueError("Frozen teacher collection binds CPU, prompt 512 and generation 32 caps")
    if args.out.exists():
        raise FileExistsError("Teacher output directory already exists")
    freeze_sha = verify_freeze(args.freeze)
    sources = [Path(__file__), REPO / "dsv3wm/src/dsv3wm/recovery_planning.py", REPO / "dsv3wm/src/dsv3wm/recovery.py",
               REPO / "dsv3wm/src/dsv3wm/native.py", REPO / "dsv3wm/src/dsv3wm/native_cpu11.py", REPO / "dsv3wm/src/dsv3wm/provenance.py",
               REPO / "dsv3wm/src/dsv3wm/planning.py", REPO / "dsv3wm/src/dsv3wm/domains.py",
               REPO / "dsv3wm/src/dsv3wm/data.py", REPO / "bwm/code/bwmn/rw2.py", REPO / "bwm/code/bwmn/lm.py",
               REPO / "bwm/code/bwmn/gpu_lease.py", REPO / "bwm/code/scripts/planbench_qwen.py",
               REPO / "dsv3wm/scripts/collect_recovery_teacher.py"]
    bound = [args.config, args.tasks, args.train, args.adapter / "adapter_model.safetensors",
             args.adapter / "adapter_config.json", *sources]
    verify_bound_inputs(args.freeze, bound)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    train = load_records(args.train, expected_split="train")
    validate_train_allowlist(tasks, train)
    if len(tasks) != 32:
        raise ValueError("Teacher collection binds the prospectively frozen 32-parent TRAIN task pool")
    provenance = {"config_sha256": sha256(args.config), "tasks_file_sha256": sha256(args.tasks),
                  "train_file_sha256": sha256(args.train), "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
                  "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"), "freeze_sha256": freeze_sha,
                  "parent_checkpoint_sha256": EXPECTED_CORE_SHA, "parent_checkpoint_verified_by_native_load": False,
                  "resource_admission_profile": admission_profile(),
                  "source_sha256": {str(path): sha256(path) for path in sources}}
    manifest = {"status": "started", "command": sys.argv, "config": cfg.dict(), **provenance,
                "scope": "TRAIN-only native learned teacher collection, not development/test reranking",
                "confirmation_access": False, "keep_per_parent": 1,
                "completion_policy": "append final finish only after a recorded learned goal prediction, before verification; reject interior finish and length greater than six"}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    started, corpus, predictor = time.perf_counter(), None, None
    def checkpoint_rankings(search, ranked):
        for name, value in (("search", search), ("ranked_candidates", ranked)):
            (args.out / (name + ".json")).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        with runtime(cfg, "recovery-teacher"):
            model, tokenizer, lm = load(cfg)
            manifest["admission_check"] = last_admission_check()
            model = load_warm(model, args.adapter)
            provenance["parent_checkpoint_verified_by_native_load"] = True
            provenance["library_versions"] = {name: importlib.metadata.version(name)
                                               for name in ("torch", "transformers", "peft", "tokenizers", "safetensors")}
            manifest.update(provenance)
            predictor = LegacyPredictor(model, tokenizer, cfg)
            corpus = collect_corpus(tasks, train, predictor, teacher_provenance=provenance, on_rankings_fixed=checkpoint_rankings)
            manifest.update(status="corpus-qualified" if corpus["gate"]["passed"] else "corpus-gate-failed",
                            gate=corpus["gate"], verified_count=len(corpus["verified"]),
                            hard_negative_count=len(corpus["hard_negatives"]), rankings_sha256=corpus["rankings_sha256"],
                            goal_pool_sha256=corpus["goal_pool_sha256"], training_row_allowlist=corpus["training_row_allowlist"],
                            model_queries=predictor.calls, input_tokens=predictor.input_tokens,
                            output_tokens=predictor.output_tokens, budget=corpus["search"]["budget"],
                            processed_positions_upper_bound=predictor.decoder_token_positions,
                            decoder_forwards_lower_bound=predictor.decoder_forwards,
                            decoder_forwards_scope="Lower bound/estimate; internal broad-batch splitting is not enumerated.")
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["admission_check"] = last_admission_check()
        manifest["duration_s"] = time.perf_counter() - started
        if predictor is not None:
            (args.out / "model_traces.json").write_text(json.dumps(predictor.traces, indent=2) + "\n", encoding="utf-8")
        if corpus is not None:
            for name in ("verified", "hard_negatives", "verified_unretained", "attempts"):
                (args.out / (name + ".jsonl")).write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                                                for row in corpus[name]), encoding="utf-8")
            for name, value in (("search", corpus["search"]), ("ranked_candidates", corpus["ranked_candidates"])):
                (args.out / (name + ".json")).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest["output_sha256"] = {path.name: sha256(path) for path in args.out.iterdir() if path.name != "manifest.json"}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "status": manifest["status"], "gate": manifest["gate"]}), flush=True)
    if not manifest["gate"]["passed"]:
        raise RuntimeError("Teacher corpus gate failed; artifacts preserved and student training forbidden")


if __name__ == "__main__":
    main()
