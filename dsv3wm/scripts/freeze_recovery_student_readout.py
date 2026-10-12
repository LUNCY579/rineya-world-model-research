"""Post-training output admission for a prospective H3 DEV readout freeze.

Hashes/JSON and exact TRAIN-corpus re-verification only: never loads a model or
opens confirmation labels. A training freeze cannot bind its later output weights.
"""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))


def source_argument(record, flag):
    command = record["command"]
    if flag not in command or command.count(flag) != 1:
        raise ValueError(f"unambiguous recorded {flag} source required")
    path = Path(command[command.index(flag) + 1])
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def validate_warm_history(directory, cfg, train_path):
    """Return fully verified historical warm/B0 source paths for new binding.

    Whole warm V2 verification is strict. B0 qualification permits exactly the
    registered, NONexecuted v1 trainer snapshot edge; its executed code/data/
    configs must remain at unchanged original pins. No model or bank is opened.
    """
    from dsv3wm.native import sha256, verify_freeze
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.recovery import INITIAL_F, INITIAL_F_SHA, RecoveryConfig
    from dsv3wm.recovery_student import admit_readout_lineage
    directory = Path(directory).resolve()
    warm, selected, _ = admit_readout_lineage(directory / "adapter", directory / "run.json", cfg,
                                             warm_baseline=True, train_path=train_path)
    config, training, warm_freeze, qualification_path = selected
    if verify_freeze(warm_freeze) != warm["freeze_sha256"]:
        raise ValueError("historical warm freeze identity mismatch")
    initial = [INITIAL_F / "adapter_model.safetensors", INITIAL_F / "adapter_config.json"]
    if sha256(initial[0]) != INITIAL_F_SHA:
        raise ValueError("registered initial F weight identity drift")
    verify_bound_inputs(warm_freeze, [config, training, qualification_path, *initial])
    frozen_warm = json.loads(warm_freeze.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if (qualification.get("status") != "development-evaluated" or qualification.get("baseline_qualified") is not True
            or qualification.get("adapter_sha256") != INITIAL_F_SHA or qualification.get("oracle_feedback") is not False
            or frozen_warm.get("qualification_sha256") != sha256(qualification_path)
            or frozen_warm.get("qualification_freeze_sha256") != qualification.get("freeze_sha256")):
        raise ValueError("historical warm/B0 qualification scalar/identity links disagree")
    q_config, q_data, q_freeze = [source_argument(qualification, flag) for flag in ("--config", "--data", "--freeze")]
    if sha256(q_freeze) != qualification["freeze_sha256"] or sha256(q_data) != qualification.get("data_sha256"):
        raise ValueError("historical B0 qualification freeze/data identity drift")
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(RecoveryConfig.read(q_config).dict()) != canonical(qualification["config"]):
        raise ValueError("historical B0 expanded configuration drift")
    if "config_sha256" in qualification and sha256(q_config) != qualification["config_sha256"]:
        raise ValueError("historical B0 recorded configuration digest drift")
    # Historical B0 has no config hash field. Original qualification-freeze
    # membership supplies its actual config pin, without inventing metadata.
    verify_bound_inputs(q_freeze, [q_config, q_data, *initial])
    q_document = json.loads(q_freeze.read_text(encoding="utf-8"))
    program = Path(qualification["command"][0])
    program = program.resolve() if program.is_absolute() else (REPO / program).resolve()
    if program != (ROOT / "scripts/evaluate_recovery.py").resolve() or "--baseline-only" not in qualification["command"]:
        raise ValueError("registered B0 qualification command identity required")
    snapshot_digest = "04f7f231238605ab607480e05aa4569f4f4e85e5a8fcd6e7b9a0071bc88c57dd"
    nonexecuted_trainer = (ROOT / "scripts/train_recovery.py").resolve()
    snapshot = (ROOT / f"prereg/source_snapshots/{snapshot_digest}.py").resolve()
    q_paths = []
    for name, expected in q_document["files"].items():
        original = (REPO / name).resolve()
        if original.is_file() and sha256(original) == expected:
            q_paths.append(original)
        elif original == nonexecuted_trainer and expected == snapshot_digest and sha256(snapshot) == snapshot_digest:
            # This v1 trainer was never executed in the qualification command;
            # its archived bytes explain the prospective V2 optimizer revision.
            q_paths.append(snapshot)
        else:
            raise ValueError(f"historical qualification pin drift outside registered snapshot edge: {original}")
    warm_paths = [(REPO / name).resolve() for name in frozen_warm["files"]]
    return sorted({directory / "run.json", directory / "adapter/adapter_model.safetensors",
                   directory / "adapter/adapter_config.json", *selected, q_config, q_data, q_freeze,
                   *initial, *warm_paths, *q_paths})


def validate_student(directory):
    from dsv3wm.native import sha256
    from dsv3wm.recovery import INITIAL_F
    from dsv3wm.recovery_student import StudentConfig, admit_readout_lineage, load_teacher_corpus
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.data import load_records
    directory = Path(directory).resolve()
    run = directory / "run.json"
    record = json.loads(run.read_text(encoding="utf-8"))
    if record.get("status") != "student-pilot-trained":
        raise ValueError("completed native student/control run required before readout freeze")
    config_path = source_argument(record, "--config")
    cfg = StudentConfig.read(config_path)
    admit_readout_lineage(directory / "adapter", run, cfg)
    if record.get("config_sha256") != sha256(config_path):
        raise ValueError("student training config hash drift")
    kind = ("C3" if cfg.arm == "A3" else "C4") if cfg.control_extra_one_step else cfg.arm
    if record.get("training_kind") != kind:
        raise ValueError("student/control label differs from actual full configuration")
    paths = [run, directory / "adapter/adapter_model.safetensors", directory / "adapter/adapter_config.json", config_path]
    inputs = {}
    for flag, key in (("--train", "train_sha256"), ("--tasks", "tasks_sha256"),
                      ("--qualification", "qualification_sha256"), ("--freeze", "freeze_sha256")):
        path = source_argument(record, flag)
        if sha256(path) != record.get(key):
            raise ValueError(f"student source {flag} hash lineage mismatch")
        inputs[flag] = path
        paths.append(path)
    qualification = json.loads(inputs["--qualification"].read_text(encoding="utf-8"))
    from dsv3wm.recovery import INITIAL_F_SHA
    if (qualification.get("status") != "development-evaluated" or qualification.get("baseline_qualified") is not True
            or qualification.get("adapter_sha256") != INITIAL_F_SHA):
        raise ValueError("unchanged qualified B0 record required")
    teacher = source_argument(record, "--teacher-dir")
    manifest_path, verified = teacher / "manifest.json", teacher / "verified.jsonl"
    if (sha256(manifest_path) != record.get("teacher_manifest_sha256")
            or sha256(verified) != record.get("verified_corpus_sha256")):
        raise ValueError("student teacher-manifest/corpus lineage mismatch")
    training = load_records(inputs["--train"], expected_split="train")
    tasks = [json.loads(line) for line in inputs["--tasks"].read_text(encoding="utf-8").splitlines() if line.strip()]
    positives, manifest, gate = load_teacher_corpus(teacher, training, tasks)
    if (manifest["train_file_sha256"] != sha256(inputs["--train"])
            or manifest["tasks_file_sha256"] != sha256(inputs["--tasks"])):
        raise ValueError("teacher source pools differ from actual student source pools")
    paths += [manifest_path, *[teacher / name for name in manifest["output_sha256"]]]
    warm = source_argument(record, "--warm-dir")
    warm_paths = [warm / "run.json", warm / "adapter/adapter_model.safetensors", warm / "adapter/adapter_config.json"]
    for path, key in zip(warm_paths, ("warm_run_sha256", "warm_adapter_sha256", "warm_adapter_config_sha256")):
        if sha256(path) != record.get(key):
            raise ValueError("student warm artifact lineage mismatch")
    warm_record, warm_sources, _ = admit_readout_lineage(warm / "adapter", warm / "run.json", cfg,
                                                       warm_baseline=True, train_path=inputs["--train"])
    if record.get("warm_arm") != cfg.warm_arm:
        raise ValueError("A3/C3 must warm A2; A4/C4 must warm A0")
    paths += warm_paths + warm_sources + validate_warm_history(warm, cfg, inputs["--train"])
    reference = INITIAL_F / "adapter_config.json"
    if sha256(reference) != record.get("registered_adapter_config_sha256"):
        raise ValueError("registered PEFT behavior reference drift")
    paths.append(reference)
    # Verify every pre-training selected input against the ORIGINAL training
    # freeze, excluding generated outputs, run.json and the freeze file itself.
    pretraining = [path for path in paths if path not in {run, directory / "adapter/adapter_model.safetensors",
                  directory / "adapter/adapter_config.json", inputs["--freeze"]}]
    verify_bound_inputs(inputs["--freeze"], pretraining)
    frozen = json.loads(inputs["--freeze"].read_text(encoding="utf-8"))
    frozen_sources = {(REPO / name).resolve(): digest for name, digest in frozen["files"].items()}
    if not record.get("source_sha256") or any(frozen_sources.get(Path(name).resolve()) != digest
                                             for name, digest in record["source_sha256"].items()):
        raise ValueError("recorded training source hashes disagree with original training freeze")
    return {"kind": kind, "warm_arm": cfg.warm_arm, "directory": str(directory),
            "config": cfg.dict(), "config_path": str(config_path), "corpus_gate": gate,
            "verified_parent_count": len(positives)}, paths


def validate_core(core):
    from dsv3wm.native import sha256, EXPECTED_CORE_SHA, EXPECTED_SIDECARS
    core = Path(core).resolve()
    expected = {"model.safetensors": EXPECTED_CORE_SHA, **EXPECTED_SIDECARS}
    paths = []
    for name, digest in expected.items():
        path = core / name
        if sha256(path) != digest:
            raise ValueError(f"fixed core/sidecar drift: {name}")
        paths.append(path)
    return paths


def build_freeze(*, student_dirs, warm_dirs=(), warm_configs=(), train, tasks, core):
    from dsv3wm.native import sha256, EXPECTED_CORE_SHA
    from dsv3wm.recovery import INITIAL_F, INITIAL_F_SHA
    from dsv3wm.recovery_student import StudentConfig, admit_readout_lineage
    from dsv3wm.recovery_planning import validate_tasks
    if not student_dirs and not warm_dirs:
        raise ValueError("select at least one completed student or explicit warm baseline")
    if len(warm_dirs) != len(warm_configs):
        raise ValueError("each warm baseline requires one explicit matching StudentConfig")
    task_rows = [json.loads(line) for line in Path(tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_tasks(task_rows)
    if len(task_rows) != 12:
        raise ValueError("fixed twelve DEV planning parents required")
    dependencies = ("__init__", "recovery_student", "recovery", "recovery_planning", "proposer", "native",
                    "provenance", "domains", "planning", "config", "data", "recovery_data", "objectives")
    paths = [*[ROOT / f"src/dsv3wm/{name}.py" for name in dependencies],
             ROOT / "scripts/evaluate_recovery_student.py", Path(__file__), ROOT / "scripts/run_measured.py",
             ROOT / "prereg/RECOVERY_H3_DRAFT_V1.md", ROOT / "prereg/RECOVERY_H3_CONTROL_CLARIFICATION_V1.md",
             Path(tasks).resolve(), INITIAL_F / "adapter_model.safetensors", INITIAL_F / "adapter_config.json",
             REPO / "bwm/code/bwmn/rw2.py", REPO / "bwm/code/bwmn/lm.py", REPO / "bwm/code/bwmn/gpu_lease.py",
             REPO / "bwm/code/scripts/planbench_qwen.py"]
    artifacts, kinds = [], set()
    for directory in student_dirs:
        metadata, selected = validate_student(directory)
        if metadata["kind"] in kinds:
            raise ValueError("duplicate student/control kind")
        kinds.add(metadata["kind"])
        artifacts.append(metadata)
        paths += selected
    for directory, config in zip(warm_dirs, warm_configs):
        directory, config = Path(directory).resolve(), Path(config).resolve()
        cfg = StudentConfig.read(config)
        record, selected, kind = admit_readout_lineage(directory / "adapter", directory / "run.json", cfg,
                                                     warm_baseline=True, train_path=train)
        selected += validate_warm_history(directory, cfg, train)
        identity = "warm-" + cfg.warm_arm
        if identity in kinds:
            raise ValueError("duplicate warm parent baseline")
        kinds.add(identity)
        artifacts.append({"kind": kind, "artifact_arm": record["config"]["arm"], "directory": str(directory),
                          "config_path": str(config), "readout_config": cfg.dict()})
        paths += [config, directory / "run.json", directory / "adapter/adapter_model.safetensors",
                  directory / "adapter/adapter_config.json", *selected]
    if sha256(INITIAL_F / "adapter_model.safetensors") != INITIAL_F_SHA:
        raise ValueError("fixed-B0 dynamics identity drift")
    paths += validate_core(core)
    unique = sorted({path.resolve() for path in paths})
    return {"instrument": "recovery-H3-actual-output-readout-v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_core_sha256": EXPECTED_CORE_SHA, "core_directory": str(Path(core).resolve()),
            "artifacts": artifacts, "confirmation_access": False,
            "files": {str(path).replace("\\", "/"): sha256(path) for path in unique},
            "scope": "prospective fixed DEV direct/reranked readouts; actual completed weights/configs; no efficacy confirmation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-dir", type=Path, action="append", default=[])
    parser.add_argument("--warm-dir", type=Path, action="append", default=[])
    parser.add_argument("--warm-config", type=Path, action="append", default=[])
    parser.add_argument("--train", type=Path, default=ROOT / "datasets/recovery-v1/train.jsonl")
    parser.add_argument("--tasks", type=Path, default=ROOT / "datasets/recovery-v1/planning_dev.jsonl")
    parser.add_argument("--core", type=Path, default=Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("student readout freeze already exists")
    record = build_freeze(student_dirs=args.student_dir, warm_dirs=args.warm_dir, warm_configs=args.warm_config,
                          train=args.train, tasks=args.tasks, core=args.core)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
