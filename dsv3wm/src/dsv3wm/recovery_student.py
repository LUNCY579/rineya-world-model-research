"""Warm H3 legacy plan contract. Validation/scoring never load a native model.

Generation and learned reranking are prediction-only; exact verification belongs
to pre-load TRAIN admission and post-choice DEV scoring, respectively.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from .recovery import RecoveryConfig, INITIAL_F, INITIAL_F_SHA

PEFT_METADATA = frozenset({"base_model_name_or_path", "revision", "peft_version", "inference_mode"})


def assert_registered_behavior(configuration, reference):
    """Unknown/missing behavior fields fail closed; no implicit default filling."""
    if not isinstance(configuration, dict) or not isinstance(reference, dict):
        raise ValueError("PEFT configuration objects required")
    def behavior(value):
        result = {key: item for key, item in value.items() if key not in PEFT_METADATA}
        # PEFT's target-module list serializes a set: order/repetition has no
        # operational meaning. Preserve the exact target set, not list order.
        targets = result.get("target_modules")
        if isinstance(targets, list):
            if any(type(target) is not str for target in targets):
                raise ValueError("target_modules must contain only strings")
            result["target_modules"] = sorted(set(targets))
        return result
    # Canonical JSON also distinguishes false from0 and true from1.
    render = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if render(behavior(configuration)) != render(behavior(reference)):
        raise ValueError("saved PEFT behavior differs from registered initial F")


def assert_adapter_configuration(path, *, recorded_sha256=None, reference_path=None):
    from .native import sha256
    path = Path(path)
    if recorded_sha256 is not None and sha256(path) != recorded_sha256:
        raise ValueError("saved adapter-config hash lineage mismatch")
    reference_path = INITIAL_F / "adapter_config.json" if reference_path is None else Path(reference_path)
    configuration = json.loads(path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    assert_registered_behavior(configuration, reference)
    return configuration


@contextmanager
def fixed_forward_adapter(model, path, name="h3_fixed_B0"):
    """Inference-only named switch; restore the original student even on failure."""
    original = getattr(model, "active_adapter", None)
    if not isinstance(original, str) or not original or name in model.peft_config:
        raise ValueError("one named active student and a fresh forward adapter name required")
    try:
        model.load_adapter(str(path), adapter_name=name, is_trainable=False)
        model.set_adapter(name)
        yield {"student_adapter": original, "forward_adapter": name}
    finally:
        model.set_adapter(original)
        if getattr(model, "active_adapter", None) != original:
            raise RuntimeError("student adapter restoration failed")


@dataclass(frozen=True)
class StudentConfig(RecoveryConfig):
    plan_weight: float = 1.
    control_extra_one_step: bool = False
    student_tokens: int = 80
    student_k: int = 4
    student_temperature: float = 1.
    student_top_p: float = .95

    def __post_init__(self):
        super().__post_init__()
        if self.arm not in {"A3", "A4"} or self.steps != 32 or self.batch_size != 1 or self.lr != .00002:
            raise ValueError("H3 binds A3/A4,32 updates,batch1,lr2e-5")
        if self.legality_weight != 0 or self.device != "cpu" or self.max_length != 512 or self.generation_tokens != 32:
            raise ValueError("H3 binds CPU/prompt512/dynamics32 and no extra legality head")
        if self.lambda2 != .5 or self.lambda4 != .5:
            raise ValueError("H3 binds state horizon weights1/.5/.5")
        if self.plan_weight not in {0., 1.} or isinstance(self.plan_weight, bool):
            raise ValueError("plan_weight=1 study or0 continuation control required")
        if (type(self.control_extra_one_step) is not bool or
                self.control_extra_one_step != (self.plan_weight == 0)):
            raise ValueError("control_extra_one_step must explicitly be true only for plan_weight0")
        if type(self.student_tokens) is not int or self.student_tokens != 80 or type(self.student_k) is not int or self.student_k != 4:
            raise ValueError("direct readout binds K4/80 tokens")
        if (isinstance(self.student_temperature, bool) or isinstance(self.student_top_p, bool)
                or self.student_temperature != 1. or self.student_top_p != .95):
            raise ValueError("direct readout binds temperature1/top-p.95")

    @property
    def warm_arm(self):
        return "A2" if self.arm == "A3" else "A0"


def registered_prompt(task):
    from .domains import _semantics
    return _semantics("register").proposer_prompt(task["s0"]["x"], task["goal"]["state"]["x"], task["rules"]["limit"], 6)


def registered_target(plan):
    from .domains import _semantics
    return _semantics("register").target_actions(plan)


def parse_registered_plan(text):
    """Strict complete legacy object syntax; no transition/legality execution."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("nonfinite value")
    try:
        value = json.loads(text.strip(), object_pairs_hook=unique, parse_constant=reject)
        if not isinstance(value, dict) or set(value) != {"actions"} or not isinstance(value["actions"], list):
            raise ValueError("registered actions object required")
        plan = value["actions"]
        if (not 1 <= len(plan) <= 6 or any(not isinstance(action, str) or action not in
              {"add1", "sub1", "double", "halve", "finish"} for action in plan)
                or plan[-1] != "finish" or "finish" in plan[:-1]):
            raise ValueError("bounded final-finish plan syntax required")
        # Reuse the established action parser after the explicit tightening.
        from .domains import _semantics
        parsed = _semantics("register").parse_plan(text, "actions")
        if parsed is None or parsed[0] != plan:
            raise ValueError("registered parser disagreement")
        return {"malformed": False, "plan": plan}
    except (ValueError, TypeError, AttributeError):
        return {"malformed": True, "plan": None, "text": text}


def verify_positive(row, training_rows, tasks):
    """Independently replay an allowlisted TRAIN positive before native load."""
    from .domains import canonical, parent_world_key, sha256
    from .planning import verify_plan
    try:
        if row.get("split") != "train" or row.get("domain") != "register" or row.get("verified") is not True:
            return False
        parent = parent_world_key("register", row["s0"], row["rules"])
        if row.get("parent_world_id") != parent or row.get("target_contract") != "registered-rw2-actions-final-finish":
            return False
        source = next((source for source in training_rows if source.get("split") == "train"
                       and sha256(source) == row["provenance"]["training_input_sha256"]), None)
        task = next((task for task in tasks if task.get("split") == "train"
                     and sha256(task) == row["provenance"]["task_sha256"]), None)
        if source is None or task is None or source["parent_world_id"] != parent or task["parent_world_id"] != parent:
            return False
        if task["provenance"].get("source_trajectory_id") != source["id"]:
            return False
        if any(canonical(row[key]) != canonical(source[key]) or canonical(row[key]) != canonical(task[key])
               for key in ("domain", "rules", "s0")) or canonical(row["goal"]) != canonical(task["goal"]):
            return False
        provenance = row["provenance"]
        if (provenance.get("selection") != "learned-ranked-then-independent-exact"
                or provenance.get("learned_predicted_goal") is not True
                or provenance.get("all_searches_and_rankings_fixed_before_verification") is not True):
            return False
        parsed = parse_registered_plan(row["target"])
        if parsed["malformed"] or parsed["plan"] != row["plan"]:
            return False
        original = row["source_plan"]
        if not isinstance(original, list) or not original:
            return False
        if original[-1] == "finish":
            if row["plan"] != original or provenance["completion_origin"] != "already-final-finish":
                return False
        elif (row["plan"] != original + ["finish"] or provenance["completion_origin"] !=
              "deterministic-final-finish-completion-before-verification"):
            return False
        if row["prompt"] != registered_prompt(task) or row["target"] != registered_target(row["plan"]):
            return False
        inputs = {"domain": "register", "rules": row["rules"], "s0": row["s0"], "actions": []}
        verification = verify_plan(inputs, row["goal"], row["plan"])
        return (verification == row["verification"] and verification["success"] and verification["all_legal"]
                and verification["labels"]["terminated"][-1] is True)
    except (KeyError, TypeError, ValueError, StopIteration):
        return False


def validate_corpus(positives, training_rows, tasks, *, require_gate=True):
    from .domains import sha256
    from .recovery_planning import validate_tasks
    validate_tasks(tasks, expected_split="train")
    if not positives or any(not verify_positive(row, training_rows, tasks) for row in positives):
        raise ValueError("TRAIN allowlist/prompt/terminal-plan independent re-verification failed")
    parents = {row["parent_world_id"] for row in positives}
    if len(parents) != len(positives):
        raise ValueError("one positive per parent required")
    limits, sequences = {row["rules"]["limit"] for row in positives}, {sha256(row["plan"]) for row in positives}
    gate = {"passed": len(parents) >= 16 and len(limits) >= 2 and len(sequences) >= 4,
            "parents": len(parents), "limits": sorted(limits), "sequences": len(sequences)}
    if require_gate and (len(tasks) != 32 or not gate["passed"]):
        raise ValueError("prospective32-task/16-parent/2-limit/4-sequence corpus gate failed")
    return gate


def load_teacher_corpus(directory, training_rows, tasks):
    from .native import sha256
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "corpus-qualified" or manifest.get("gate", {}).get("passed") is not True:
        raise ValueError("qualified frozen teacher corpus required")
    for name, digest in manifest["output_sha256"].items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError("teacher output hash/path mismatch")
    positives = [json.loads(line) for line in (directory / "verified.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if "verified.jsonl" not in manifest["output_sha256"]:
        raise ValueError("verified labels absent from teacher manifest")
    gate = validate_corpus(positives, training_rows, tasks)
    from .domains import sha256 as value_hash
    if manifest.get("goal_pool_sha256") != value_hash(tasks):
        raise ValueError("teacher task-pool hash mismatch")
    for name in ("ranked_candidates.json", "search.json"):
        if name not in manifest["output_sha256"]:
            raise ValueError("frozen learned candidate/search proof required")
    rankings = json.loads((directory / "ranked_candidates.json").read_text(encoding="utf-8"))
    search = json.loads((directory / "search.json").read_text(encoding="utf-8"))
    if (value_hash(rankings) != manifest["rankings_sha256"] or search.get("oracle_feedback") is not False
            or search.get("all_choices_fixed_before_scoring") is not True):
        raise ValueError("fixed learned rankings/search boundary proof mismatch")
    source_traces = {value_hash(trace) for trace in search["traces"]}
    task_lookup = {value_hash(task): task for task in tasks}
    for row in positives:
        provenance = row["provenance"]
        if (provenance["all_rankings_sha256"] != manifest["rankings_sha256"]
                or manifest["training_row_allowlist"].get(row["parent_world_id"]) != provenance["training_input_sha256"]
                or any(provenance.get(key) != manifest.get(key) for key in
                       ("config_sha256", "tasks_file_sha256", "train_file_sha256", "adapter_sha256",
                        "adapter_config_sha256", "freeze_sha256", "source_sha256"))):
            raise ValueError("positive/teacher-manifest provenance mismatch")
        task = task_lookup[provenance["task_sha256"]]
        candidate = next((candidate for candidate in rankings[task["id"]]
                          if candidate["candidate_sha256"] == provenance["candidate_sha256"]), None)
        if (candidate is None or candidate["source_trace_sha256"] not in source_traces
                or candidate["source_trace_sha256"] != provenance["source_trace_sha256"]
                or candidate["learned_rank"] != provenance["learned_rank"]
                or candidate["eligible_for_positive"] is not True or candidate["predicted_goal"] is not True
                or candidate["plan"] != row["plan"] or candidate["source_plan"] != row["source_plan"]):
            raise ValueError("positive row absent from frozen eligible learned candidate proof")
    return positives, manifest, gate


def warm_lineage(directory, cfg, train_path):
    from .native import sha256, EXPECTED_CORE_SHA
    directory = Path(directory)
    record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if (record.get("status") != "pilot-trained" or record["config"]["arm"] != cfg.warm_arm
            or record.get("optimizer_steps") != 32 or record.get("initial_adapter_sha256") != INITIAL_F_SHA
            or record.get("core_sha256") != EXPECTED_CORE_SHA or record.get("data_sha256") != sha256(train_path)
            or record.get("adapter_sha256") != sha256(directory / "adapter/adapter_model.safetensors")):
        raise ValueError("final32-step A2/A0 warm lineage mismatch")
    peft = assert_adapter_configuration(directory / "adapter/adapter_config.json",
                                        recorded_sha256=record.get("adapter_config_sha256"))
    if (peft["r"] != 16 or peft["lora_alpha"] != 32 or peft["lora_dropout"] != .05
            or set(peft["target_modules"]) != {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}):
        raise ValueError("registered adapter capacity/dropout changed")
    return record


def admit_readout_lineage(adapter, run_path, cfg, *, warm_baseline=False, train_path=None):
    """Return artifact identity and extra inputs a prospective freeze must bind."""
    from . import REPO
    from .native import sha256, EXPECTED_CORE_SHA
    adapter, run_path = Path(adapter).resolve(), Path(run_path).resolve()
    record = json.loads(run_path.read_text(encoding="utf-8"))
    if warm_baseline:
        if train_path is None or run_path != adapter.parent / "run.json":
            raise ValueError("warm-baseline requires original run directory and explicit source --train")
        record = warm_lineage(adapter.parent, cfg, train_path)
        command = record["command"]
        def source_arg(flag):
            if flag not in command:
                raise ValueError("warm run lacks source config/train/freeze/qualification provenance")
            path = Path(command[command.index(flag) + 1])
            return path.resolve() if path.is_absolute() else (REPO / path).resolve()
        config_path, recorded_train, training_freeze, qualification = [source_arg(flag) for flag in
            ("--config", "--train", "--freeze", "--qualification")]
        if (Path(train_path).resolve() != recorded_train
                or sha256(config_path) != record.get("config_sha256")
                or RecoveryConfig.read(config_path).dict() != record["config"]
                or sha256(training_freeze) != record.get("freeze_sha256")
                or sha256(qualification) != record.get("qualification_sha256")):
            raise ValueError("warm source config/train/training-freeze/qualification lineage mismatch")
        return record, [config_path, recorded_train, training_freeze, qualification], "warm-parent-proposer-baseline"
    if train_path is not None:
        raise ValueError("--train is only accepted in explicit --warm-baseline mode")
    if (record.get("status") != "student-pilot-trained" or record["config"]["arm"] != cfg.arm
            or record.get("optimizer_steps") != 32 or record.get("core_sha256") != EXPECTED_CORE_SHA
            or record.get("adapter_sha256") != sha256(adapter / "adapter_model.safetensors")
            or not isinstance(record.get("adapter_config_sha256"), str)):
        raise ValueError("H3-trained status/arm/weight/config-hash lineage required; use --warm-baseline for A0/A2")
    assert_adapter_configuration(adapter / "adapter_config.json", recorded_sha256=record["adapter_config_sha256"])
    kind = ("h3-continuation-control-C3" if cfg.arm == "A3" else "h3-continuation-control-C4") if record["config"].get("control_extra_one_step") is True else "h3-trained-student-proposer"
    return record, [], kind


def shared_orders(trajectory_count, plan_count, steps=32, seed=0):
    if min(trajectory_count, plan_count, steps) <= 0:
        raise ValueError("nonempty shared training pools required")
    rng = random.Random(177 + seed)
    trajectories = [rng.randrange(trajectory_count) for _ in range(steps)]
    plans = []
    plan_rng = random.Random(701 + seed)
    while len(plans) < steps:
        cycle = list(range(plan_count))
        plan_rng.shuffle(cycle)
        plans.extend(cycle)
    return trajectories, plans[:steps]


def continuation_slot(row, positive, cfg):
    """Real fourth/second CE slot: verified plan or SAME source-row h1 control."""
    if row.get("split") != "train" or row.get("domain") != "register":
        raise ValueError("continuation slot requires the current TRAIN trajectory")
    if cfg.plan_weight == 1:
        if positive.get("split") != "train" or positive.get("verified") is not True:
            raise ValueError("positive verified TRAIN plan required")
        return positive["prompt"], positive["target"], "plan"
    if not cfg.control_extra_one_step:
        raise ValueError("explicit real extra-h1 continuation control required")
    from .data import conditioning
    from .recovery import legacy_prompt
    from .recovery_data import legacy_target
    return (legacy_prompt(conditioning(row, 1)),
            json.dumps(legacy_target(row, 1), sort_keys=True, separators=(",", ":")), "extra_one_step")


def combine_student_losses(dynamics, slot_loss, cfg):
    """Both slot types contribute unit-weight gradients; no discarded forwards."""
    from .objectives import multi_horizon_loss
    return multi_horizon_loss(dynamics, cfg.lambda2, cfg.lambda4) + slot_loss


def generate_students(tasks, generator, *, base_seed=0):
    """ALL greedy/CRN generation precedes scoring; native sampler is injected."""
    from .recovery_planning import validate_tasks
    validate_tasks(tasks)
    results = []
    for task in tasks:
        prompt, candidates = registered_prompt(task), []
        for index in range(4):
            # Same SHA convention as BWM planner._seed without importing models.
            seed = None if index == 0 else int(hashlib.sha256(
                f"{base_seed}|{task['parent_world_id']}|0|{index}".encode()).hexdigest()[:15], 16)
            raw = generator(prompt, seed, 80)
            g = raw.get("n_tokens", 0)
            if type(g) is not int or not 0 <= g <= 80:
                raise ValueError("student generation cap exceeded")
            candidates.append({"index": index, "seed": seed, "raw": raw, **parse_registered_plan(raw.get("text"))})
        valid = [registered_target(row["plan"]) for row in candidates if not row["malformed"]]
        results.append({"task_id": task["id"], "parent_world_id": task["parent_world_id"], "prompt": prompt,
                        "candidates": candidates, "selected_index": 0, "selected_plan": candidates[0]["plan"],
                        "model_queries": sum(row["raw"].get("model_calls", 1) for row in candidates),
                        "input_tokens": sum(row["raw"].get("input_tokens", 0) for row in candidates),
                        "output_tokens": sum(row["raw"].get("n_tokens", 0) for row in candidates),
                        "processed_positions": sum(row["raw"].get("decoder_token_positions", 0) for row in candidates),
                        "decoder_forwards": sum(row["raw"].get("decoder_forwards", 0) for row in candidates),
                        "valid_syntax_count": len(valid), "unique_valid_plans": len(set(valid)),
                        "diversity_fraction": len(set(valid)) / 4, "oracle_feedback": False})
    return results


def rerank_students(tasks, generated, predictor):
    """Optional fixed-B0 learned reranking, <=26 F calls after four proposals."""
    from .recovery import legacy_prompt
    from .recovery_planning import validate_tasks, _valid_prediction
    validate_tasks(tasks)
    if len(tasks) != len(generated):
        raise ValueError("proposal/task cardinality mismatch")
    results = []
    for task, proposals in zip(tasks, generated):
        if proposals["task_id"] != task["id"] or proposals["oracle_feedback"] is not False:
            raise ValueError("proposal/task provenance mismatch")
        trace_start = len(predictor.traces)
        calls, positions, ranked = 0, proposals["processed_positions"], []
        if proposals["model_queries"] > 4 or positions > 120000:
            raise ValueError("proposal budget exceeded")
        for candidate in proposals["candidates"]:
            if candidate["malformed"]:
                continue
            state, valid, terminated = copy.deepcopy(task["s0"]), True, False
            for action in candidate["plan"]:
                inputs = {"domain": "register", "rules": task["rules"], "s0": state, "actions": [action]}
                if hasattr(predictor, "token_length"):
                    n = predictor.token_length(inputs)
                else:
                    n = len(predictor.tokenizer(legacy_prompt(inputs), add_special_tokens=False).input_ids)
                if terminated or calls >= 26 or n > 512 or positions + n + 32 > 120000:
                    valid = False
                    break
                before = len(predictor.traces)
                value = predictor(copy.deepcopy(inputs))
                calls += 1
                if len(predictor.traces) != before + 1:
                    raise ValueError("one-query raw forward trace required")
                raw = predictor.traces[-1]["generated"]
                g, extra = raw["n_tokens"], int(not raw["hit_eos"])
                if type(g) is not int or not 0 <= g <= 32:
                    raise ValueError("forward generation cap exceeded")
                positions += n + g - 1 + extra
                if not _valid_prediction(value) or not value["legal"] or value["failed"]:
                    valid = False
                    break
                state, terminated = value["state"], value["terminated"]
            if valid:
                distance = abs(state["x"] - task["goal"]["state"]["x"])
                ranked.append({"index": candidate["index"], "predicted_success": terminated and distance == 0,
                               "distance": distance, "plan": candidate["plan"]})
        ranked.sort(key=lambda row: (not row["predicted_success"], row["distance"], len(row["plan"]), row["index"]))
        choice = ranked[0] if ranked else None
        results.append({"task_id": task["id"], "selected_index": choice["index"] if choice else None,
                        "selected_plan": choice["plan"] if choice else None, "ranked": ranked,
                        "forward_queries": calls, "total_model_queries": calls + proposals["model_queries"],
                        "processed_positions": positions, "oracle_feedback": False,
                        "forward_traces": copy.deepcopy(predictor.traces[trace_start:]),
                        "instrument": "learned candidate reranking; fixed B0; no exact feedback"})
    return results


def score_students(tasks, generated, reranked=None, *, verifier=None):
    """Exact terminal-plan and first-action scoring after ALL native work."""
    from .planning import verify_plan
    from .domains import _semantics, exact_transition
    from .recovery_planning import validate_tasks
    validate_tasks(tasks)
    if len(tasks) != len(generated) or (reranked is not None and len(reranked) != len(tasks)):
        raise ValueError("task/candidate cardinality mismatch")
    verifier = verify_plan if verifier is None else verifier
    executor, results = _semantics("register"), []
    for index, (task, proposals) in enumerate(zip(tasks, generated)):
        if (proposals["task_id"] != task["id"] or proposals["parent_world_id"] != task["parent_world_id"]
                or proposals["selected_index"] != 0 or proposals["prompt"] != registered_prompt(task)
                or proposals["oracle_feedback"] is not False):
            raise ValueError("unaltered fixed-top1 proposal provenance required")
        inputs = {"domain": "register", "rules": task["rules"], "s0": task["s0"], "actions": []}
        scores = []
        for candidate in proposals["candidates"]:
            parsed = parse_registered_plan(candidate["raw"].get("text"))
            if parsed["malformed"] != candidate["malformed"] or parsed["plan"] != candidate["plan"]:
                raise ValueError("candidate/raw text mismatch")
            verification = verifier(inputs, task["goal"], candidate["plan"]) if not candidate["malformed"] else None
            scores.append({"index": candidate["index"], "success": bool(verification and verification["success"]
                           and verification["all_legal"] and verification["labels"]["terminated"][-1]),
                           "all_legal": bool(verification and verification["all_legal"]), "verification": verification})
        plan = proposals["selected_plan"]
        if plan != proposals["candidates"][0]["plan"]:
            raise ValueError("top1 was replaced")
        first_legal, first_optimal = False, False
        if plan:
            after, first_legal, _, invalid = exact_transition("register", task["s0"], plan[0], task["rules"])
            distance = executor.opt_moves(task["s0"]["x"], task["goal"]["state"]["x"], task["rules"]["limit"])
            remaining = executor.opt_moves(after["x"], task["goal"]["state"]["x"], task["rules"]["limit"]) if first_legal and not invalid else None
            first_optimal = bool(first_legal and distance is not None and remaining == distance - 1)
        result = {**proposals, "top1_success": scores[0]["success"], "pass_at_4": any(row["success"] for row in scores),
                  "unique_exact_valid_plans": len({registered_target(candidate["plan"]) for candidate, score in
                                                  zip(proposals["candidates"], scores) if score["success"]}),
                  "first_action_legal": first_legal, "first_action_optimal": first_optimal, "scores": scores,
                  "scope": "pure no-search top1; pass@4 offline coverage; exact scoring after generation"}
        if reranked is not None:
            chosen = reranked[index]
            if chosen["task_id"] != task["id"] or chosen["oracle_feedback"] is not False:
                raise ValueError("learned-rerank provenance mismatch")
            selected = chosen["selected_index"]
            if selected is not None and chosen["selected_plan"] != proposals["candidates"][selected]["plan"]:
                raise ValueError("rerank choice outside fixed proposals")
            result["learned_reranking"] = {**chosen, "success": False if selected is None else scores[selected]["success"]}
        results.append(result)
    return results
