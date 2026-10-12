"""Prospectively bind actual H3 outputs/warm history to fixed DEV guard inputs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def build_freeze(*, student_dirs, warm_dirs=(), warm_configs=(), train, data, fixtures, core, baselines=()):
    from dsv3wm.native import sha256, EXPECTED_CORE_SHA
    from dsv3wm.recovery_student import StudentConfig
    from evaluate_recovery_student_guards import admit_artifact, load_fixed_inputs, source_paths, paired_guards
    from freeze_recovery_student_readout import validate_student, validate_core
    load_fixed_inputs(data, fixtures)
    if not student_dirs and not warm_dirs:
        raise ValueError("select completed actual students/controls or explicit warm parents")
    if len(warm_dirs) != len(warm_configs):
        raise ValueError("one explicit StudentConfig per unchanged warm parent required")
    references = {}
    for path in baselines:
        path = Path(path).resolve()
        record = json.loads(path.read_text(encoding="utf-8"))
        arm = record.get("artifact_arm")
        if arm in references:
            raise ValueError("duplicate warm-parent guard reference")
        references[arm] = (path, record)
    paths = [Path(data), Path(fixtures), *source_paths(), ROOT / "prereg/RECOVERY_H3_GUARDS_V1.md"]
    artifacts, kinds, used_references = [], set(), set()
    for directory in student_dirs:
        metadata, selected = validate_student(directory)
        cfg = StudentConfig(**metadata["config"])
        record, admitted, kind = admit_artifact(Path(directory) / "adapter", Path(directory) / "run.json", cfg)
        if metadata["kind"] in kinds:
            raise ValueError("duplicate student/control guard kind")
        if cfg.warm_arm not in references:
            raise ValueError("each student/control needs its actual warm-parent guard result")
        baseline_path, baseline = references[cfg.warm_arm]
        paired_guards({"data_sha256": sha256(data), "fixtures_sha256": sha256(fixtures), "outcomes": baseline.get("outcomes", {})},
                      baseline, warm_arm=cfg.warm_arm, warm_adapter_sha=record["warm_adapter_sha256"])
        kinds.add(metadata["kind"]); used_references.add(cfg.warm_arm)
        artifacts.append({**metadata, "readout_kind": kind, "baseline": str(baseline_path)})
        paths += [*selected, *admitted, baseline_path]
    for directory, config in zip(warm_dirs, warm_configs):
        directory, config = Path(directory).resolve(), Path(config).resolve()
        cfg = StudentConfig.read(config)
        record, selected, kind = admit_artifact(directory / "adapter", directory / "run.json", cfg,
                                               warm_baseline=True, train=train)
        identity = "warm-" + cfg.warm_arm
        if identity in kinds:
            raise ValueError("duplicate warm-parent guard kind")
        kinds.add(identity)
        artifacts.append({"kind": identity, "directory": str(directory), "config": cfg.dict(),
                          "config_path": str(config), "artifact_arm": record["config"]["arm"], "readout_kind": kind})
        paths += [config, directory / "run.json", directory / "adapter/adapter_model.safetensors",
                  directory / "adapter/adapter_config.json", *selected]
    if set(references) != used_references:
        raise ValueError("unused or unrelated baseline references cannot enter guard freeze")
    paths += validate_core(core)
    unique = sorted({Path(path).resolve() for path in paths})
    return {"instrument": "recovery-H3-fixed-development-guards-v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_core_sha256": EXPECTED_CORE_SHA, "core_directory": str(Path(core).resolve()),
            "artifacts": artifacts, "files": {str(path).replace("\\", "/"): sha256(path) for path in unique},
            "confirmation_access": False, "scope": "actual output lineage; unchanged24 DEV +six behavioral fixtures; no efficacy or broad retention pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("student-dir", "warm-dir", "warm-config", "baseline"):
        parser.add_argument("--" + name, type=Path, action="append", default=[])
    parser.add_argument("--train", type=Path, default=ROOT / "datasets/recovery-v1/train.jsonl")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/recovery-v1/dev.jsonl")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "results/retention_core_v1.fixtures.json")
    parser.add_argument("--core", type=Path, default=Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("guard freeze exists; refusing overwrite")
    record = build_freeze(student_dirs=args.student_dir, warm_dirs=args.warm_dir, warm_configs=args.warm_config,
                          train=args.train, data=args.data, fixtures=args.fixtures, core=args.core, baselines=args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
