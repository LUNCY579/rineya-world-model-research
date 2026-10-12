"""Prospective conditional legal-field calibration; implementation only until frozen.

This probes true,/false, after a fixed legal-first JSON prefix. It is not
generated-state or planning confidence. No model runs without a new phase freeze.
"""
import argparse
import bisect
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
SCHEMA = "dsv3wm-conditional-legal-calibration-v1"
DEV_SHA = "7fe3c996a8ab4d00fe623edc8ad99f643ebc44f7b36f4a8935bd3c21ea012cc3"
PREFIX = '{"legal":'
LITERALS = ("true,", "false,")
BINS = tuple(index / 10 for index in range(11))
LOGLOSS_CLIP = 1e-12
OTHER_MASS_SCOPE = "All token paths outside the two canonical comma-terminated prefixes, including valid alternative whitespace/serialization; not invalid-JSON probability"
CONTRACT = {"endpoint": "conditional legal-first boolean literal plus comma", "horizon": 1,
            "prefix": PREFIX, "continuations": list(LITERALS), "eos": False, "max_length": 512,
            "bins": list(BINS), "logloss_clip": LOGLOSS_CLIP, "parents": 24, "id_parents": 12, "ood_parents": 12,
            "known_h1_label_counts": {"true": 16, "false": 8},
            "native_loader": "dsv3wm.native_cpu11.load", "cache": False,
            "native_forward_dtype": "float32", "probability_reduction_dtype": "float64",
            "label_access": "after every probability trace is fixed", "state_or_planning_confidence": False}
CONTRACT["other_mass_scope"] = OTHER_MASS_SCOPE


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def versions_record():
    versions, missing = {"python": sys.version}, []
    for name in ("torch", "transformers", "peft", "tokenizers", "safetensors"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None; missing.append(name)
    return {"versions": versions, "unavailable_distributions": missing,
            "scope": "actual calling environment metadata; no enforced library-version profile"}


def source_paths():
    modules = ("__init__", "config", "data", "domains", "native", "native_cpu11", "recovery", "recovery_data",
               "recovery_student", "planning", "provenance")
    return [Path(__file__), ROOT / "scripts/freeze_recovery_readout.py", ROOT / "scripts/train_recovery.py",
            ROOT / "scripts/evaluate_recovery.py", ROOT / "scripts/run_measured.py",
            ROOT / "prereg/RECOVERY_LEGAL_CALIBRATION_V1.md",
            *[ROOT / f"src/dsv3wm/{name}.py" for name in modules],
            *[REPO / f"bwm/code/bwmn/{name}.py" for name in ("__init__", "lm", "rw2", "blocks", "flops", "planbench_extract", "gpu_lease")],
            REPO / "bwm/code/scripts/planbench_qwen.py"]


def command_path(record, flag):
    command = record.get("command")
    if not isinstance(command, list) or any(type(item) is not str for item in command) or command.count(flag) != 1:
        raise ValueError("Unambiguous actual training/qualification command required")
    index = command.index(flag)
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError("Missing recorded command path")
    return (REPO / command[index + 1]).resolve()


def require_phase_freeze(path):
    from dsv3wm.native import verify_freeze
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if (record.get("instrument") != SCHEMA or record.get("confirmation_access") is not False
            or record.get("probability_outputs_accessed") is not False
            or canonical(record.get("calibration_contract")) != canonical(CONTRACT)):
        raise ValueError("New prospective conditional-legality calibration freeze/contract required")
    return verify_freeze(path)


def _base_run_admission(directory, train):
    spec = importlib.util.spec_from_file_location("_legal_calibration_existing_run_admission", ROOT / "scripts/freeze_recovery_readout.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.validate_run(directory, train)


def admit_lineage(config, adapter, run, train, data, phase_freeze, core):
    from dsv3wm.native import sha256, EXPECTED_CORE_SHA, EXPECTED_SIDECARS
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.recovery import RecoveryConfig, INITIAL_F, INITIAL_F_SHA
    from dsv3wm.recovery_student import assert_adapter_configuration
    config, adapter, run, train, data, core = [Path(value).resolve() for value in (config, adapter, run, train, data, core)]
    cfg = RecoveryConfig.read(config)
    if (cfg.arm not in {"A0", "A1", "A2", "A5"} or cfg.seed != 0 or cfg.device != "cpu" or cfg.threads != 4
            or cfg.steps != (96 if cfg.arm == "A5" else 32) or cfg.max_length != 512 or cfg.batch_size != 1
            or cfg.target_mode != "full" or cfg.legality_weight != 0):
        raise ValueError("Actual fixed seed0 warm A0/A1/A2/A5 recipe only")
    if run != adapter.parent / "run.json" or adapter != run.parent / "adapter" or train != data.with_name("train.jsonl"):
        raise ValueError("Actual selected adapter/run and source dataset paths required")
    record = json.loads(run.read_text(encoding="utf-8"))
    if (record.get("status") != "pilot-trained" or type(record.get("optimizer_steps")) is not int
            or record["optimizer_steps"] != cfg.steps or record.get("core_sha256") != EXPECTED_CORE_SHA
            or record.get("initial_adapter_sha256") != INITIAL_F_SHA or canonical(record.get("config")) != canonical(cfg.dict())
            or command_path(record, "--config") != config or command_path(record, "--train") != train
            or command_path(record, "--out") != run.parent or sha256(config) != record.get("config_sha256")):
        raise ValueError("Actual warm run identity/full configuration mismatch")
    arm, selected = _base_run_admission(run.parent, train)
    if arm != cfg.arm:
        raise ValueError("Selected warm artifact arm differs")
    assert_adapter_configuration(adapter / "adapter_config.json", recorded_sha256=record.get("adapter_config_sha256"))
    training_freeze, qualification = command_path(record, "--freeze"), command_path(record, "--qualification")
    frozen_training = json.loads(training_freeze.read_text(encoding="utf-8"))
    qualified = json.loads(qualification.read_text(encoding="utf-8"))
    if (qualified.get("status") != "development-evaluated" or qualified.get("baseline_qualified") is not True
            or qualified.get("adapter_sha256") != INITIAL_F_SHA or qualified.get("oracle_feedback") is not False
            or frozen_training.get("qualification_sha256") != sha256(qualification)
            or frozen_training.get("qualification_freeze_sha256") != qualified.get("freeze_sha256")):
        raise ValueError("Historical warm/B0 qualification links disagree")
    q_config, q_data, q_freeze = [command_path(qualified, flag) for flag in ("--config", "--data", "--freeze")]
    if (q_data != data or sha256(q_data) != qualified.get("data_sha256") or sha256(q_freeze) != qualified.get("freeze_sha256")
            or canonical(RecoveryConfig.read(q_config).dict()) != canonical(qualified.get("config"))):
        raise ValueError("Historical B0 qualification data/config/freeze drift")
    if "config_sha256" in qualified and sha256(q_config) != qualified["config_sha256"]:
        raise ValueError("Historical B0 recorded config drift")
    program = (REPO / qualified["command"][0]).resolve()
    if program != (ROOT / "scripts/evaluate_recovery.py").resolve() or "--baseline-only" not in qualified["command"]:
        raise ValueError("Registered original B0 qualification command required")
    initial = [INITIAL_F / "adapter_model.safetensors", INITIAL_F / "adapter_config.json"]
    if sha256(initial[0]) != INITIAL_F_SHA:
        raise ValueError("Registered initial F weight changed")
    verify_bound_inputs(training_freeze, [config, train, qualification, *initial])
    verify_bound_inputs(q_freeze, [q_config, q_data, *initial])
    # Preserve the registered NONexecuted v1-trainer snapshot edge; every other
    # historical qualification source must still match its original byte hash.
    snapshot_sha = "04f7f231238605ab607480e05aa4569f4f4e85e5a8fcd6e7b9a0071bc88c57dd"
    q_paths = []
    for name, digest in json.loads(q_freeze.read_text(encoding="utf-8"))["files"].items():
        original = (REPO / name).resolve()
        if original.is_file() and sha256(original) == digest:
            q_paths.append(original)
        elif original == (ROOT / "scripts/train_recovery.py").resolve() and digest == snapshot_sha:
            snapshot = ROOT / f"prereg/source_snapshots/{snapshot_sha}.py"
            if sha256(snapshot) != digest: raise ValueError("Historical nonexecuted snapshot changed")
            q_paths.append(snapshot)
        else:
            raise ValueError("Historical qualification pin changed outside the declared snapshot edge")
    core_paths = [core / name for name in ("model.safetensors", *EXPECTED_SIDECARS)]
    for path, expected in zip(core_paths, (EXPECTED_CORE_SHA, *EXPECTED_SIDECARS.values())):
        if sha256(path) != expected: raise ValueError("Frozen core or tokenizer sidecar changed")
    bound = [config, adapter / "adapter_model.safetensors", adapter / "adapter_config.json", run, train, data,
             *selected, *initial, q_config, q_data, q_freeze, *q_paths, *core_paths, *source_paths(),
             *[(REPO / name).resolve() for name in frozen_training["files"]]]
    verify_bound_inputs(phase_freeze, bound)
    return cfg, record, sorted(set(bound))


def input_projection(rows):
    from dsv3wm.domains import parent_world_key
    selected, seen, cohorts = [], set(), []
    for row in rows:
        if row.get("split") != "dev" or row.get("domain") != "register" or len(row.get("actions", [])) != 4:
            raise ValueError("Original four-action RegisterWorld DEV inputs required")
        parent = parent_world_key("register", row["s0"], row["rules"])
        if parent != row["parent_world_id"] or parent in seen:
            raise ValueError("Unique authentic original DEV parent required")
        cohort = row["provenance"]["cohort"]
        seen.add(parent); cohorts.append(cohort)
        selected.append({key: row[key] for key in ("id", "split", "domain", "parent_world_id", "rules", "s0", "actions", "provenance")})
    if len(rows) != 24 or cohorts.count("recovery_id_teacher_known") != 12 or cohorts.count("recovery_ood_limit") != 12:
        raise ValueError("Only original24 DEV parents,12 ID/12 OOD")
    return selected


def checked_encoding(tokenizer, lm, prompt, literal):
    p = list(tokenizer(prompt, add_special_tokens=False).input_ids)
    t = list(tokenizer(literal, add_special_tokens=False).input_ids)
    joint = list(tokenizer(prompt + literal, add_special_tokens=False).input_ids)
    special = set(getattr(tokenizer, "all_special_ids", [tokenizer.eos_token_id]))
    if not p or not t or p + t != joint or special.intersection(t):
        raise ValueError("Nonmatching prompt/continuation token boundary or EOS literal")
    encoded = lm.encode_pair(tokenizer, prompt, literal, 512, eos=False)
    if (encoded is None or list(encoded.input_ids) != joint or list(encoded.labels) != [-100] * len(p) + t
            or encoded.n_target != len(t)):
        raise ValueError("No-EOS exact boundary/response mask and <=512 tokens required")
    return encoded


def binary_mass(logp_true, logp_false):
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value > 0 for value in (logp_true, logp_false))):
        return {"status": "finite-failure", "reason": "nonfinite/nonprobability literal log likelihood",
                "raw_logp_repr": [repr(logp_true), repr(logp_false)]}
    largest = max(logp_true, logp_false)
    log_mass = largest + math.log(math.exp(logp_true - largest) + math.exp(logp_false - largest))
    mass = math.exp(log_mass)
    if not math.isfinite(log_mass) or mass > 1:
        return {"status": "finite-failure", "reason": "literal paths have invalid total mass",
                "raw_logp_true": logp_true, "raw_logp_false": logp_false}
    log_true, log_false = logp_true - log_mass, logp_false - log_mass
    return {"status": "captured", "raw_logp_true": logp_true, "raw_logp_false": logp_false,
            "literal_mass_true": math.exp(logp_true), "literal_mass_false": math.exp(logp_false),
            "log_valid_literal_mass": log_mass, "valid_literal_mass": mass, "other_mass": 1 - mass,
            "literal_mass_underflow": mass == 0, "conditional_p_true": math.exp(log_true),
            "conditional_logp_true": log_true, "conditional_logp_false": log_false}


def capture_probabilities(rows, tokenizer, lm, scorer, *, sink=None, work_snapshot=None):
    """Input-only phase; all probability traces finish before labels are indexed."""
    from dsv3wm.data import conditioning
    from dsv3wm.recovery import legacy_prompt
    rows = input_projection(rows)
    traces = []
    def flush():
        if sink: sink(traces)
    def path_work(before):
        if before is None: return None
        try:
            after = work_snapshot()
            if (set(before) != set(after) or any(type(before[key]) is not int or type(after[key]) is not int
                    or after[key] < before[key] for key in before)):
                raise ValueError("Monotone integer literal-work counters required")
            return {key: after[key] - before[key] for key in before}
        except Exception as error:
            return {"status": "unavailable", "error": repr(error)}
    for index, row in enumerate(rows):
        inputs = conditioning(row, 1)
        prompt = legacy_prompt(inputs) + PREFIX
        trace = {"source_index": index, "id": row["id"], "parent_world_id": row["parent_world_id"],
                 "cohort": row["provenance"]["cohort"], "conditioning": inputs, "prompt": prompt,
                 "continuations": list(LITERALS), "eos": False, "status": "capture-started", "literal_paths": []}
        traces.append(trace)
        flush()
        try:
            encodings = []
            for literal in LITERALS:
                path = {"literal": literal, "status": "encoding-started", "scoring_attempted": False,
                        "returned": False, "work": None}
                trace["literal_paths"].append(path)
                flush()
                encoded = checked_encoding(tokenizer, lm, prompt, literal)
                encodings.append(encoded)
                path.update(status="encoded", input_tokens=len(encoded.input_ids), supervised_literal_tokens=encoded.n_target,
                            literal_token_ids=list(encoded.labels[-encoded.n_target:]))
                flush()
            targets = [[token for token in encoded.labels if token != -100] for encoded in encodings]
            if targets[0][:len(targets[1])] == targets[1] or targets[1][:len(targets[0])] == targets[0]:
                raise ValueError("Literal token paths must be disjoint, neither a prefix of the other")
            trace.update(status="capturing", input_tokens=[len(encoded.input_ids) for encoded in encodings],
                         supervised_literal_tokens=[encoded.n_target for encoded in encodings], literal_token_paths=targets,
                         prompt_token_ids=list(encodings[0].input_ids[:-encodings[0].n_target]))
            flush()
            scores = []
            for path, literal, encoded in zip(trace["literal_paths"], LITERALS, encodings):
                before = dict(work_snapshot()) if work_snapshot else None
                path.update(status="scoring", scoring_attempted=True)
                flush()
                try:
                    value = scorer(prompt, literal, encoded)
                except BaseException as error:
                    path.update(status="score-failure" if isinstance(error, (ValueError, TypeError)) else "score-aborted",
                                error=repr(error), work=path_work(before))
                    raise
                scores.append(value)
                finite = (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value))
                path.update(status="returned" if finite else "returned-nonfinite", returned=True, finite=finite,
                            raw_logp=value if finite else None, work=path_work(before))
                if not finite: path["raw_logp_repr"] = repr(value)
                flush()
            trace.update(binary_mass(*scores))
        except (ValueError, TypeError) as error:
            trace.update(status="capture-failure", reason=str(error))
        except BaseException as error:
            trace.update(status="capture-aborted", reason=repr(error))
            flush()
            raise
        flush()
    return {"schema": SCHEMA, "all_probability_traces_fixed_before_labels": True, "traces": traces}


def capture_completeness(traces):
    successes = sum(trace.get("status") == "captured" for trace in traces)
    finished = sum(trace.get("status") in {"captured", "finite-failure", "capture-failure"} for trace in traces)
    return {"capture_parents_started": len(traces), "capture_rows_completed": finished, "successful_probability_captures": successes,
            "expected_parent_count": 24, "complete_capture_attempts": len(traces) == 24 and finished == 24,
            "complete_probability_capture": len(traces) == 24 and successes == 24}


def score_probabilities(rows, captured):
    from dsv3wm.data import conditioning
    from dsv3wm.recovery import legacy_prompt
    inputs = input_projection(rows)
    traces = captured.get("traces")
    if captured.get("all_probability_traces_fixed_before_labels") is not True or not isinstance(traces, list) or len(traces) != 24:
        raise ValueError("Every probability trace must be fixed before labels")
    for index, (source, trace) in enumerate(zip(inputs, traces)):
        if (trace.get("id") != source["id"] or trace.get("parent_world_id") != source["parent_world_id"]
                or trace.get("cohort") != source["provenance"]["cohort"] or trace.get("source_index") != index
                or canonical(trace.get("conditioning")) != canonical(conditioning(source, 1))
                or trace.get("prompt") != legacy_prompt(conditioning(source, 1)) + PREFIX
                or trace.get("continuations") != list(LITERALS) or trace.get("eos") is not False):
            raise ValueError("Calibration parent/cohort/order mismatch")
        if trace.get("status") == "captured":
            expected = binary_mass(trace.get("raw_logp_true"), trace.get("raw_logp_false"))
            if expected["status"] != "captured" or any(canonical(trace.get(key)) != canonical(value) for key, value in expected.items()):
                raise ValueError("Captured probability/mass provenance changed")
        elif trace.get("status") not in {"finite-failure", "capture-failure"}:
            raise ValueError("Explicit captured/failure status required for every parent")
    scored = []
    for row, trace in zip(rows, traces):
        gold = row["labels"]["legal"][0]
        if type(gold) is not bool: raise ValueError("Strict original h1 boolean legality label required")
        outcome = {**trace, "gold_legal": gold, "brier": None, "logloss": None, "unclipped_logloss": None, "logloss_clipped": None}
        if trace["status"] == "captured":
            p = trace["conditional_p_true"]
            reporting_p = min(max(p, LOGLOSS_CLIP), 1 - LOGLOSS_CLIP)
            outcome.update(brier=(p - int(gold)) ** 2,
                           logloss=-math.log(reporting_p) if gold else -math.log1p(-reporting_p),
                           unclipped_logloss=-trace["conditional_logp_true"] if gold else -trace["conditional_logp_false"],
                           logloss_clipped=reporting_p != p)
        scored.append(outcome)
    if sum(row["gold_legal"] for row in scored) != 16:
        raise ValueError("Original24 h1 labels must preserve16 true/8 false; no cohort expansion")
    summary = {}
    for cohort in ("recovery_id_teacher_known", "recovery_ood_limit", "all"):
        group = [row for row in scored if cohort == "all" or row["cohort"] == cohort]
        valid = [row for row in group if row["status"] == "captured"]
        bins = []
        for index, (low, high) in enumerate(zip(BINS, BINS[1:])):
            selected = [row for row in valid if min(bisect.bisect_right(BINS, row["conditional_p_true"]) - 1, 9) == index]
            mean = sum(row["conditional_p_true"] for row in selected) / len(selected) if selected else None
            frequency = sum(row["gold_legal"] for row in selected) / len(selected) if selected else None
            bins.append({"low": low, "high": high, "high_inclusive": index == 9, "n": len(selected),
                         "mean_p_true": sum(row["conditional_p_true"] for row in selected) / len(selected) if selected else None,
                         "legal_frequency": frequency, "absolute_reliability_gap": abs(mean - frequency) if selected else None})
        summary[cohort] = {"n": len(group), "captured": len(valid), "failures": len(group) - len(valid),
            "finite_failures": sum(row["status"] == "finite-failure" for row in group),
            "literal_mass_underflows": sum(row.get("literal_mass_underflow", False) for row in valid),
            "logloss_clipped_cases": sum(row["logloss_clipped"] for row in valid),
            **{name: sum(row[name] for row in valid) / len(valid) if valid else None
               for name in ("brier", "logloss", "valid_literal_mass", "other_mass")}, "fixed_bins": bins,
            "ece": sum(b["n"] * abs(b["mean_p_true"] - b["legal_frequency"]) for b in bins if b["n"]) / len(valid) if valid else None}
    return {"per_parent": scored, "summary": summary, "logloss_reporting_clip": LOGLOSS_CLIP,
            "other_mass_scope": OTHER_MASS_SCOPE,
            **capture_completeness(traces),
            "scope": "descriptive fixed24 DEV seed0 conditional legal-first likelihood; not generated-state/planning confidence or broad calibration"}


def main():
    from dsv3wm.native import sha256, runtime, boundary, EXPECTED_CORE_SHA
    from dsv3wm.native_cpu11 import load, admission_profile, last_admission_check
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "adapter", "run", "data", "train", "core", "freeze", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    traces_path = args.out.with_suffix(".probabilities.json")
    if args.out.exists() or traces_path.exists(): raise FileExistsError("Calibration output/trace exists")
    freeze_sha = require_phase_freeze(args.freeze)
    if sha256(args.data) != DEV_SHA: raise ValueError("Only original24 DEV input bytes permitted")
    cfg, lineage, bound = admit_lineage(args.config, args.adapter, args.run, args.train, args.data, args.freeze, args.core)
    # Input projection never indexes labels. Labels are reopened after capture.
    inputs = input_projection([json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": SCHEMA, "status": "started", "command": sys.argv, "contract": CONTRACT,
              "environment": versions_record(), "other_mass_scope": OTHER_MASS_SCOPE,
              "config": cfg.dict(), "freeze_sha256": freeze_sha, "core_sha256": EXPECTED_CORE_SHA,
              "resource_admission_profile": admission_profile(), "confirmation_access": False,
              "adapter_sha256": lineage["adapter_sha256"], "run_sha256": sha256(args.run), "data_sha256": sha256(args.data),
              "config_sha256": sha256(args.config), "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"),
              "source_sha256": {str(path): sha256(path) for path in source_paths()},
              "bound_input_sha256": {str(path): sha256(path) for path in bound}, "retention_pass": None}
    for path, value in ((args.out, record), (traces_path, {"status": "reserved", "traces": []})):
        with path.open("x", encoding="utf-8") as stream: stream.write(canonical(value) + "\n")
    counters = {"literal_queries": 0, "input_tokens": 0, "supervised_literal_tokens": 0,
                "literal_queries_returned": 0, "decoder_forwards": 0, "decoder_token_positions": 0}
    progress = []
    def save_progress(traces):
        progress[:] = traces
        traces_path.write_text(canonical({"status": "capturing", "traces": traces}) + "\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        with runtime(cfg, "recovery-conditional-legality-calibration"):
            import torch
            from bwmn import lm as runtime_lm
            from dsv3wm.recovery import load_warm
            if Path(runtime_lm.MODELS[cfg.core]).resolve() != args.core.resolve(): raise ValueError("Bound core directory differs from actual native loader")
            model, tokenizer, lm = load(cfg)
            record["admission_check"] = last_admission_check()
            model = load_warm(model, args.adapter)
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()): raise ValueError("FP32 unmerged parent/adapter required")
            model.eval(); model.config.use_cache = False
            decoder, _ = lm._decoder_and_head(model)
            def count_forward(module, values, kwargs):
                ids = kwargs.get("input_ids", values[0] if values else None)
                if ids is None: raise ValueError("Cannot identify actual decoder positions")
                counters["decoder_forwards"] += 1; counters["decoder_token_positions"] += int(ids.numel())
            def score_literal(prompt, literal, encoded):
                boundary(cfg); lm.thermal_guard()
                ids, labels, attention = (value.to(next(model.parameters()).device) for value in lm.collate([encoded], tokenizer.pad_token_id))
                counters["literal_queries"] += 1; counters["input_tokens"] += int(ids.numel())
                counters["supervised_literal_tokens"] += encoded.n_target
                with torch.no_grad():
                    logits, target, _ = lm.supervised_logits(model, ids, attention, labels)
                    if int(target.numel()) != encoded.n_target: raise ValueError("Literal response mask changed")
                    result = float(torch.log_softmax(logits.to(dtype=torch.float64), -1).gather(-1, target.unsqueeze(-1)).sum())
                counters["literal_queries_returned"] += 1
                return result
            hook = decoder.register_forward_pre_hook(count_forward, with_kwargs=True)
            try:
                captured = capture_probabilities(inputs, tokenizer, lm, score_literal, sink=save_progress,
                                                   work_snapshot=lambda: dict(counters))
            finally: hook.remove()
            traces_path.write_text(canonical({"status": "capture-complete", **captured}) + "\n", encoding="utf-8")
        # No native call follows this boundary; only now reopen/index gold labels.
        rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
        record.update(score_probabilities(rows, captured), status="development-calibration-evaluated",
                      all_probability_traces_fixed_before_labels=True)
    except BaseException as error:
        record.update(status="failed", error=repr(error)); raise
    finally:
        peak, rss = None, None
        try:
            import psutil
            memory = psutil.Process().memory_info()
            peak, rss = getattr(memory, "peak_wset", None), memory.rss
        except Exception as error:
            record["host_memory_error"] = repr(error)
        record.update(admission_check=last_admission_check(), costs=counters, duration_s=time.perf_counter() - started,
                      **capture_completeness(progress),
                      peak_host_process_bytes=peak, current_host_process_rss_bytes=rss,
                      host_memory_scope="current process only; external run_measured.py reports sampled process tree",
                      traces_sha256=sha256(traces_path), cost_scope="actual decoder hooks and literal-scoring forwards; no measured FLOPs")
        args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({"out": str(args.out), "status": record["status"], "scope": record["scope"]}), flush=True)


if __name__ == "__main__":
    main()
