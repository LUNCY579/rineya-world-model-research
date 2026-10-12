"""Fixed DEV dynamics/behavior guards for actual H3 students and warm parents.

This additive instrument does not change the prior P1 or retention results.
Every native generation finishes before labels or behavioral targets are scored.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
DEV_SHA = "7fe3c996a8ab4d00fe623edc8ad99f643ebc44f7b36f4a8935bd3c21ea012cc3"
FIXTURE_SHA = "0fd6dff4dc88e25c0588897cda93e18757116c6b4dfce603e0a02c9ecd32b9ba"
SCHEMA = "dsv3wm-h3-fixed-development-guards-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def reference_scorer():
    path = ROOT / "scripts/evaluate_recovery_retention.py"
    spec = importlib.util.spec_from_file_location("_h3_unchanged_retention_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_reply


def source_paths():
    modules = ("__init__", "recovery_student", "recovery", "recovery_planning", "recovery_data",
               "proposer", "native", "native_cpu11", "provenance", "data", "domains", "planning", "objectives", "config")
    return [Path(__file__), ROOT / "scripts/evaluate_recovery_student_guards.py", ROOT / "scripts/freeze_recovery_student_guards.py",
            ROOT / "scripts/freeze_recovery_student_readout.py", ROOT / "scripts/evaluate_recovery_retention.py",
            ROOT / "scripts/run_measured.py", *[ROOT / f"src/dsv3wm/{name}.py" for name in modules],
            REPO / "bwm/code/bwmn/rw2.py", REPO / "bwm/code/bwmn/lm.py",
            REPO / "bwm/code/bwmn/gpu_lease.py", REPO / "bwm/code/scripts/planbench_qwen.py",
            ROOT / "prereg/RECOVERY_H3_DRAFT_V1.md", ROOT / "prereg/RECOVERY_H3_CONTROL_CLARIFICATION_V1.md"]


def validate_rows(rows):
    """Input-only checks; the generation path never reads row labels."""
    if (len(rows) != 24 or len({row.get("parent_world_id") for row in rows}) != 24
            or any(not isinstance(row.get("parent_world_id"), str) or not row["parent_world_id"] for row in rows)):
        raise ValueError("fixed twenty-four distinct DEV parents required")
    cohorts = []
    for row in rows:
        if row.get("split") != "dev" or row.get("domain") != "register" or len(row.get("actions", [])) != 4:
            raise ValueError("only four-step RegisterWorld DEV rows permitted")
        if (not isinstance(row.get("s0"), dict) or set(row["s0"]) != {"x"} or type(row["s0"]["x"]) is not int
                or type(row.get("rules", {}).get("limit")) is not int or row["rules"]["limit"] <= 0):
            raise ValueError("strict initial state and limit required")
        cohorts.append(row.get("provenance", {}).get("cohort"))
    if cohorts.count("recovery_id_teacher_known") != 12 or cohorts.count("recovery_ood_limit") != 12:
        raise ValueError("unchanged twelve ID/twelve OOD cohort required")


def validate_fixtures(document):
    fixtures = document.get("fixtures")
    if (document.get("schema") != "dsv3wm-retention-fixtures-v1" or not isinstance(fixtures, list)
            or len(fixtures) != 6 or len({row.get("id") for row in fixtures}) != 6
            or sum(row.get("domain") == "register" for row in fixtures) != 3
            or sum(row.get("domain") == "ledger" for row in fixtures) != 3
            or any(not isinstance(row.get("prompt"), str) or not isinstance(row.get("messages"), list) for row in fixtures)):
        raise ValueError("original six historical-format behavior fixtures required")
    return fixtures


def load_fixed_inputs(data, fixtures):
    from dsv3wm.data import load_records
    from dsv3wm.native import sha256
    if sha256(data) != DEV_SHA or sha256(fixtures) != FIXTURE_SHA:
        raise ValueError("fixed recovery DEV/behavior bytes changed; no replacement cohort allowed")
    rows = load_records(data, expected_split="dev")
    validate_rows(rows)
    return rows, validate_fixtures(json.loads(Path(fixtures).read_text(encoding="utf-8")))


def admit_artifact(adapter, run, cfg, *, warm_baseline=False, train=None):
    """Use the existing full output/history admission, including control identity."""
    from dsv3wm.recovery_student import admit_readout_lineage
    from freeze_recovery_student_readout import validate_student, validate_warm_history
    adapter, run = Path(adapter).resolve(), Path(run).resolve()
    if run != adapter.parent / "run.json":
        raise ValueError("actual run.json must accompany the selected adapter")
    record, paths, kind = admit_readout_lineage(adapter, run, cfg, warm_baseline=warm_baseline, train_path=train)
    if warm_baseline:
        paths += validate_warm_history(adapter.parent, cfg, train)
    else:
        metadata, selected = validate_student(adapter.parent)
        if canonical(metadata["config"]) != canonical(cfg.dict()):
            raise ValueError("actual full student/control config differs from guard config")
        paths += selected
    return record, sorted(set(Path(path).resolve() for path in paths)), kind


def input_identity(rows):
    values = [{key: row[key] for key in ("parent_world_id", "domain", "rules", "s0", "actions", "split", "provenance")}
              for row in rows]
    return hashlib.sha256(canonical(values).encode()).hexdigest()


class TaggedPredictor:
    """Attach source ordinals BEFORE each inherited learned-only batch call."""
    def __init__(self, predictor, rows, trace_sink=None):
        from dsv3wm.recovery import legacy_parse
        self.parse = legacy_parse
        self.predictor, self.rows, self.trace_sink = predictor, rows, trace_sink
        self.phase, self.step = "rollout", 0
        self.active, self.states = list(range(len(rows))), [copy.deepcopy(row["s0"]) for row in rows]
        self.traces = []

    def direct(self, horizon):
        self.phase, self.step = "direct", horizon

    def predict_batch(self, inputs):
        from dsv3wm.data import conditioning
        if self.phase == "rollout":
            self.step += 1
            indices = self.active
            expected = [{"domain": "register", "rules": self.rows[i]["rules"], "s0": self.states[i],
                         "actions": [self.rows[i]["actions"][self.step - 1]]} for i in indices]
        else:
            indices = list(range(len(self.rows)))
            expected = [conditioning(row, self.step) for row in self.rows]
        if canonical(inputs) != canonical(expected):
            raise ValueError("generation departed from the source/own-predicted-state schedule")
        tags = [{"source_index": i, "parent_world_id": self.rows[i]["parent_world_id"], "phase": self.phase,
                 "step": self.step, "trace_id": f"{self.rows[i]['parent_world_id']}:{self.phase}:{self.step}"} for i in indices]
        start = len(self.predictor.traces)
        values = self.predictor.predict_batch(copy.deepcopy(inputs))
        raw = self.predictor.traces[start:]
        if len(raw) != len(inputs) or len(values) != len(inputs):
            raise ValueError("one preserved raw trace per requested parent required")
        remaining = []
        for index, item, value, trace, tag in zip(indices, inputs, values, raw, tags):
            parsed = self.parse(trace["generated"].get("text"))
            if (canonical(trace["conditioning"]) != canonical(item) or canonical(trace["parsed"]) != canonical(parsed)
                    or canonical(value) != canonical(parsed)):
                raise ValueError("raw reply/parsed output/own conditioning mismatch")
            generated = trace["generated"]
            if type(generated.get("n_tokens")) is not int or not 0 <= generated["n_tokens"] <= 32 or type(generated.get("hit_eos")) is not bool:
                raise ValueError("strict greedy32 raw counters required")
            self.traces.append({**tag, **copy.deepcopy(trace)})
            if self.phase == "rollout" and not value.get("malformed"):
                self.states[index] = copy.deepcopy(value["state"])
                if not value["terminated"]:
                    remaining.append(index)
        if self.phase == "rollout":
            self.active = remaining
        if self.trace_sink:
            self.trace_sink(copy.deepcopy(self.traces))
        return values


def generate_guards(rows, fixtures, predictor, behavior_generator, *, trace_sink=None):
    """No label/target reads, exact transition calls, or scoring in this function."""
    from dsv3wm.data import conditioning
    from dsv3wm.recovery import batched_rollout
    validate_rows(rows)
    if predictor.traces:
        raise ValueError("fresh guard predictor required")
    tagged = TaggedPredictor(predictor, rows, trace_sink)
    rollouts = batched_rollout(tagged, rows, 4)
    direct = {}
    for horizon in (1, 2, 4):
        tagged.direct(horizon)
        direct[str(horizon)] = tagged.predict_batch([conditioning(row, horizon) for row in rows])
    # Behavior generator receives only immutable historical messages/prompts.
    requests = [{key: fixture[key] for key in ("id", "domain", "prompt", "messages")} for fixture in fixtures]
    behavior = behavior_generator(copy.deepcopy(requests))
    if len(behavior["replies"]) != 6:
        raise ValueError("all six behavior replies must be preserved before scoring")
    for reply in behavior["replies"]:
        if type(reply.get("n_tokens")) is not int or not 0 <= reply["n_tokens"] <= 32 or type(reply.get("hit_eos")) is not bool:
            raise ValueError("strict behavior greedy32 raw counters required")
    return {"input_identity": input_identity(rows), "rollouts": rollouts, "direct": direct,
            "dynamics_traces": tagged.traces, "behavior": behavior, "behavior_ids": [row["id"] for row in fixtures],
            "oracle_feedback": False, "all_generation_fixed_before_scoring": True}


def replay_dynamics(rows, generated):
    """Reparse own-state traces without transition rules or gold labels."""
    from dsv3wm.data import conditioning
    from dsv3wm.recovery import batched_rollout, legacy_parse
    traces = generated["dynamics_traces"]
    class Replay:
        def __init__(self):
            self.cursor, self.traces = 0, []
        def predict_batch(self, inputs):
            batch = traces[self.cursor:self.cursor + len(inputs)]
            if len(batch) != len(inputs):
                raise ValueError("missing guard traces")
            outputs = []
            for item, trace in zip(inputs, batch):
                if canonical(item) != canonical(trace["conditioning"]):
                    raise ValueError("corrupt conditioning or uncertain trace order")
                value = legacy_parse(trace["generated"].get("text"))
                outputs.append(value)
                self.traces.append({key: copy.deepcopy(trace[key]) for key in ("conditioning", "generated", "parsed")})
            self.cursor += len(batch)
            return outputs
    replay = Replay()
    tagged = TaggedPredictor(replay, rows)
    rollouts = batched_rollout(tagged, rows, 4)
    direct = {}
    for horizon in (1, 2, 4):
        tagged.direct(horizon)
        direct[str(horizon)] = tagged.predict_batch([conditioning(row, horizon) for row in rows])
    if (replay.cursor != len(traces) or canonical(tagged.traces) != canonical(traces)
            or canonical(rollouts) != canonical(generated["rollouts"]) or canonical(direct) != canonical(generated["direct"])):
        raise ValueError("guard trace tags, schedule, or preserved outputs changed")
    return rollouts, direct


def score_guards(rows, fixtures, generated):
    validate_rows(rows)
    if (generated.get("input_identity") != input_identity(rows) or generated.get("oracle_feedback") is not False
            or generated.get("all_generation_fixed_before_scoring") is not True
            or generated.get("behavior_ids") != [row["id"] for row in fixtures]):
        raise ValueError("unchanged source order and completed generation required")
    rollouts, direct = replay_dynamics(rows, generated)
    parents = []
    for index, row in enumerate(rows):
        flags, cumulative_failed = [], False
        states_correct = []
        for step in range(4):
            value = rollouts[index][step] if len(rollouts[index]) > step else {"malformed": True, "unattempted": True}
            malformed = bool(value.get("malformed"))
            attempted = not value.get("skipped") and not value.get("unattempted")
            if malformed:
                cumulative_failed = True if cumulative_failed is True else None
            elif attempted:
                cumulative_failed = True if not value["legal"] else cumulative_failed
            gold = row["labels"]
            correct = not malformed and value.get("state") == gold["states"][step]
            states_correct.append(correct)
            flags.append({"step": step + 1, "state_correct": correct, "malformed": malformed,
                          "attempted": attempted, "legality_correct": not malformed and value.get("legal") == gold["legal"][step],
                          "termination_correct": not malformed and value.get("terminated") == gold["terminated"][step],
                          "predicted_cumulative_attempted_failure": cumulative_failed,
                          "cumulative_failure_correct": None if cumulative_failed is None else cumulative_failed == gold["failed"][step]})
        horizons = {}
        for horizon in (1, 2, 4):
            value = direct[str(horizon)][index]
            gold = row["labels"]
            horizons[str(horizon)] = {"rollout_correct": flags[horizon - 1]["state_correct"],
                "rollout_malformed": flags[horizon - 1]["malformed"],
                "legality_correct": flags[horizon - 1]["legality_correct"],
                "termination_correct": flags[horizon - 1]["termination_correct"],
                "direct_correct": not value.get("malformed") and value.get("state") == gold["states"][horizon - 1],
                "direct_malformed": bool(value.get("malformed")),
                "direct_legality_correct": not value.get("malformed") and value.get("legal") == gold["legal"][horizon - 1],
                "direct_termination_correct": not value.get("malformed") and value.get("terminated") == gold["terminated"][horizon - 1]}
        parents.append({"parent_world_id": row["parent_world_id"], "cohort": "id" if row["provenance"]["cohort"] == "recovery_id_teacher_known" else "ood",
                        "horizons": horizons, "steps": flags,
                        "correct_final_with_wrong_or_unknown_intermediate": states_correct[-1] and not all(states_correct[:-1])})
    summary = {}
    for cohort in ("id", "ood", "all"):
        selected = [row for row in parents if cohort == "all" or row["cohort"] == cohort]
        summary[cohort] = {str(h): {"n": len(selected), **{field: sum(row["horizons"][str(h)][field] for row in selected)
                            for field in selected[0]["horizons"][str(h)]}} for h in (1, 2, 4)}
    scorer = reference_scorer()
    behavior = [scorer(fixture, reply) for fixture, reply in zip(fixtures, generated["behavior"]["replies"])]
    return {"per_parent": parents, "summary": summary, "behavior": behavior,
            "behavior_correct": sum(row["correct"] for row in behavior), "behavior_n": 6,
            "retention_pass": None, "calibration": "unmeasured; no confidence probabilities captured",
            "scope": "fixed development state/flag and six-case behavior guards; no confirmation or broad retention pass"}


def paired_guards(current, baseline, *, warm_arm, warm_adapter_sha):
    if (baseline.get("schema") != SCHEMA or baseline.get("status") != "development-guards-evaluated"
            or baseline.get("warm_baseline") is not True or baseline.get("artifact_arm") != warm_arm
            or baseline.get("adapter_sha256") != warm_adapter_sha
            or baseline.get("oracle_feedback") is not False or baseline.get("all_generation_fixed_before_scoring") is not True
            or current["data_sha256"] != baseline.get("data_sha256")
            or current["fixtures_sha256"] != baseline.get("fixtures_sha256")):
        raise ValueError("paired guard reference must be the same fixed cohort and actual unchanged warm parent")
    before = {row["parent_world_id"]: row for row in baseline["outcomes"]["per_parent"]}
    after = {row["parent_world_id"]: row for row in current["outcomes"]["per_parent"]}
    if len(before) != 24 or set(before) != set(after) or any(before[p]["cohort"] != after[p]["cohort"] for p in before):
        raise ValueError("paired dynamics parents/cohorts differ")
    dynamics = {}
    for cohort in ("id", "ood", "all"):
        ids = [parent for parent in before if cohort == "all" or before[parent]["cohort"] == cohort]
        dynamics[cohort] = {}
        for horizon in ("1", "2", "4"):
            dynamics[cohort][horizon] = {}
            for metric in ("rollout_correct", "direct_correct", "legality_correct", "termination_correct"):
                gains = [p for p in ids if not before[p]["horizons"][horizon][metric] and after[p]["horizons"][horizon][metric]]
                losses = [p for p in ids if before[p]["horizons"][horizon][metric] and not after[p]["horizons"][horizon][metric]]
                dynamics[cohort][horizon][metric] = {"n": len(ids), "gains": gains, "losses": losses, "net": len(gains) - len(losses)}
    b = {row["id"]: row for row in baseline["outcomes"]["behavior"]}
    a = {row["id"]: row for row in current["outcomes"]["behavior"]}
    if len(b) != 6 or set(b) != set(a) or any(b[key]["domain"] != a[key]["domain"] for key in b):
        raise ValueError("paired behavior fixture identities differ")
    losses = [key for key in b if b[key]["correct"] and not a[key]["correct"]]
    gains = [key for key in b if not b[key]["correct"] and a[key]["correct"]]
    return {"dynamics": dynamics, "behavior": {"gains": gains, "losses": losses,
            "baseline_correct": sum(row["correct"] for row in b.values()), "unqualified_baseline_cases": [key for key in b if not b[key]["correct"]]},
            "id_h4_net_loss": -dynamics["id"]["4"]["rollout_correct"]["net"],
            "id_h4_net_loss_at_most_two": dynamics["id"]["4"]["rollout_correct"]["net"] >= -2,
            "behavior_losses_require_explanation": bool(losses), "combined_h3_gate": "not evaluated by this guard-only instrument"}


def main():
    from dsv3wm.native import runtime, sha256, verify_freeze
    from dsv3wm.native_cpu11 import admission_profile, last_admission_check, load
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.recovery import LegacyPredictor, load_warm
    from dsv3wm.recovery_student import StudentConfig
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "adapter", "run", "data", "fixtures", "freeze", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--warm-baseline", action="store_true")
    parser.add_argument("--train", type=Path, help="original source TRAIN; mandatory only in warm-baseline mode")
    parser.add_argument("--baseline", type=Path, help="actual own warm-parent guard result; required for trained student/control")
    args = parser.parse_args()
    cfg = StudentConfig.read(args.config)
    if cfg.device != "cpu" or cfg.max_length != 512 or cfg.generation_tokens != 32:
        raise ValueError("fixed guards bind CPU, prompt512, greedy32")
    if (args.warm_baseline and args.baseline is not None) or (not args.warm_baseline and args.baseline is None):
        raise ValueError("warm mode has no comparator; trained mode requires own warm-parent guard result")
    freeze_sha = verify_freeze(args.freeze)
    lineage, extra, kind = admit_artifact(args.adapter, args.run, cfg, warm_baseline=args.warm_baseline, train=args.train)
    bound = [args.config, args.adapter / "adapter_model.safetensors", args.adapter / "adapter_config.json", args.run,
             args.data, args.fixtures, *source_paths(), *extra]
    if args.baseline:
        bound.append(args.baseline)
    verify_bound_inputs(args.freeze, bound)
    rows, fixtures = load_fixed_inputs(args.data, args.fixtures)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    if baseline is not None:
        # Refuse wrong baseline before loading any model; complete paired metrics come later.
        paired_guards({"data_sha256": sha256(args.data), "fixtures_sha256": sha256(args.fixtures), "outcomes": baseline.get("outcomes", {})},
                      baseline, warm_arm=cfg.warm_arm, warm_adapter_sha=lineage["warm_adapter_sha256"])
    record = {"resource_admission_profile": admission_profile(), "schema": SCHEMA, "status": "started", "command": sys.argv, "config": cfg.dict(), "config_sha256": sha256(args.config),
              "run_sha256": sha256(args.run), "adapter_sha256": lineage["adapter_sha256"],
              "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"), "artifact_arm": lineage["config"]["arm"],
              "artifact_training_kind": lineage.get("training_kind", "original-warm-dynamics"), "artifact_training_status": lineage["status"],
              "readout_kind": kind, "warm_baseline": args.warm_baseline, "freeze_sha256": freeze_sha,
              "data_sha256": sha256(args.data), "fixtures_sha256": sha256(args.fixtures),
              "source_sha256": {str(path): sha256(path) for path in source_paths()},
              "artifact_source_inputs_sha256": {str(path): sha256(path) for path in extra},
              "baseline_sha256": sha256(args.baseline) if args.baseline else None,
              "confirmation_access": False, "oracle_feedback": False, "retention_pass": None}
    trace_path = args.out.with_suffix(".traces.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path, value in ((args.out, record), (trace_path, {"status": "reserved", "dynamics": [], "behavior": []})):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value) + "\n")
    started, generated, predictor = time.perf_counter(), None, None
    observed_behavior = [{}]
    counters = {phase: {"decoder_forwards": 0, "decoder_token_positions": 0} for phase in ("dynamics", "behavior")}
    phase = ["dynamics"]
    try:
        with runtime(cfg, "recovery-h3-development-guards"):
            model, tokenizer, lm = load(cfg)
            record["admission_check"] = last_admission_check()
            model = load_warm(model, args.adapter)
            model.eval()
            predictor = LegacyPredictor(model, tokenizer, cfg)
            def count_forward(module, values, kwargs):
                ids = kwargs.get("input_ids", values[0] if values else None)
                if ids is None:
                    raise ValueError("guard counter cannot identify decoder positions")
                counters[phase[0]]["decoder_forwards"] += 1
                counters[phase[0]]["decoder_token_positions"] += int(ids.numel())
            def behavior_generator(requests):
                phase[0] = "behavior"
                prompts = [row["prompt"] for row in requests]
                for row in requests:
                    rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    if rendered != row["prompt"]:
                        raise ValueError("historical behavior prompt/template drift")
                lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
                if any(length > 512 for length in lengths):
                    raise ValueError("overlength historical fixture; no truncation")
                lm.DUTY_RATIO = 0.0
                replies = lm.sample_crn(model, tokenizer, prompts, [None] * 6, max_new_tokens=32)
                observed_behavior[0] = {"replies": replies, "model_queries": 6, "input_tokens": sum(lengths),
                                        "output_tokens": sum(reply["n_tokens"] for reply in replies)}
                partial = json.loads(trace_path.read_text(encoding="utf-8"))
                partial["behavior"] = [{"fixture_id": row["id"], "generated": reply} for row, reply in zip(fixtures, replies)]
                trace_path.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
                return observed_behavior[0]
            def trace_sink(traces):
                trace_path.write_text(json.dumps({"status": "generating", "dynamics": traces, "behavior": []}, indent=2) + "\n", encoding="utf-8")
            hook = model.register_forward_pre_hook(count_forward, with_kwargs=True)
            generation_started = time.perf_counter()
            try:
                generated = generate_guards(rows, fixtures, predictor, behavior_generator, trace_sink=trace_sink)
            finally:
                hook.remove()
            record["generation_duration_s"] = time.perf_counter() - generation_started
            trace_path.write_text(json.dumps({"status": "generation-complete", "dynamics": generated["dynamics_traces"],
                                             "behavior": [{"fixture_id": row["id"], "generated": reply} for row, reply in
                                                           zip(fixtures, generated["behavior"]["replies"])]}, indent=2) + "\n", encoding="utf-8")
            # Scoring boundary: no model call follows this point.
            scoring_started = time.perf_counter()
            record["outcomes"] = score_guards(rows, fixtures, generated)
            if baseline is not None:
                record["paired_warm_guard"] = paired_guards(record, baseline, warm_arm=cfg.warm_arm,
                                                              warm_adapter_sha=lineage["warm_adapter_sha256"])
            record.update(status="development-guards-evaluated", all_generation_fixed_before_scoring=True,
                          scoring_duration_s=time.perf_counter() - scoring_started)
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["admission_check"] = last_admission_check()
        behavior = generated["behavior"] if generated else observed_behavior[0]
        if predictor is not None:
            record["costs"] = {"dynamics_queries": predictor.calls, "behavior_queries": behavior.get("model_queries"),
                "total_model_queries": predictor.calls + behavior["model_queries"] if generated else None,
                "input_tokens": predictor.input_tokens + behavior.get("input_tokens", 0),
                "output_tokens": predictor.output_tokens + behavior.get("output_tokens", 0),
                "actual_generation": counters, "decoder_token_positions": sum(row["decoder_token_positions"] for row in counters.values()),
                "decoder_forwards": sum(row["decoder_forwards"] for row in counters.values()),
                "legacy_dynamics_processed_positions_upper_bound": predictor.decoder_token_positions,
                "scope": "actual forward-pre-hook padded positions; token/query counters cover completed returned batches; total queries unknown if generation incomplete; no FLOP claim",
                "complete_generation": generated is not None}
        record["duration_s"] = time.perf_counter() - started
        record["duration_scope"] = "end-to-end load/generate/score; separate generation and scoring durations when complete"
        record["traces_sha256"] = sha256(trace_path)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "status": record["status"]}), flush=True)


if __name__ == "__main__":
    main()
