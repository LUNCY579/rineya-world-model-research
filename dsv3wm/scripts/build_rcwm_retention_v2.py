"""Prospective tick-complete R/P v2 fixtures, paired exactly to frozen v1 inputs."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
V1_FIXTURE_SHA = "0d14dd4302b9c39ec8eb8e30b4f0252ab24b35974e5b66135d0bfee3baf4357c"


def legacy_builder():
    path = ROOT / "scripts/build_rcwm_retention.py"
    spec = importlib.util.spec_from_file_location("_rcwm_retention_v1_builder_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths():
    return [Path(__file__), ROOT / "scripts/evaluate_rcwm_retention_v2.py",
            ROOT / "scripts/build_rcwm_retention.py", ROOT / "scripts/evaluate_rcwm_retention.py",
            ROOT / "src/dsv3wm/__init__.py", ROOT / "src/dsv3wm/config.py", ROOT / "src/dsv3wm/native.py",
            ROOT / "src/dsv3wm/provenance.py", ROOT / "src/dsv3wm/domains.py", ROOT / "src/dsv3wm/ledger_recovery.py",
            ROOT / "src/dsv3wm/data.py",
            REPO / "rcwm/experiments/RCWM-T1/rcwmt/env.py", REPO / "bwm/code/bwmn/__init__.py",
            REPO / "bwm/code/bwmn/lm.py", REPO / "bwm/code/bwmn/flops.py", REPO / "bwm/code/bwmn/blocks.py",
            REPO / "bwm/code/bwmn/planbench_extract.py", REPO / "bwm/code/bwmn/gpu_lease.py",
            REPO / "bwm/code/scripts/planbench_qwen.py", ROOT / "prereg/RCWM_RETENTION_V2.md",
            ROOT / "prereg/EXECUTION_SOURCE_CLOSURE_V1.md"]


def original_tick_support(path, *, expected_sha):
    path = Path(path)
    if path.name != "t2_train.jsonl" or file_hash(path) != expected_sha:
        raise ValueError("SHA-bound original TRAIN tick input support required")
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["family"] == "register_machine":
            value = row["event"].get("tick")
            if type(value) is not int:
                raise ValueError("Every original register event must have an integer tick")
            values.add(value)
    if not values:
        raise ValueError("Original register TRAIN tick support is empty")
    return sorted(values)


def original_frame_audit(path, *, expected_sha):
    """SHA-bound original prompt/input metadata only; never inspect targets."""
    from dsv3wm.domains import _semantics
    path = Path(path)
    if path.name != "t2_train.jsonl" or file_hash(path) != expected_sha:
        raise ValueError("SHA-bound original TRAIN frame audit required")
    env = _semantics("ledger")
    groups = {family: {"systems": Counter(), "recorded_frames": Counter(), "user_fields": Counter(), "chain_lengths": Counter()}
              for family in ("register_machine", "prerequisite_plan")}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        family = row["family"]
        if family not in groups:
            continue
        prompt = row["prompt"]
        system = prompt.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]
        user = json.loads(prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
        name = "SYSTEM_ONE" if system == env.SYSTEM_ONE[family] else "SYSTEM_MULTI" if system == env.SYSTEM_MULTI[family] else "other"
        group = groups[family]
        group["systems"][name] += 1
        group["recorded_frames"][str(row.get("frame"))] += 1
        group["user_fields"][",".join(sorted(user))] += 1
        if "events" in user:
            group["chain_lengths"][str(len(user["events"]))] += 1
    return {"source_sha256": expected_sha, "labels_used": False,
            "families": {family: {name: dict(counter) for name, counter in group.items()} for family, group in groups.items()}}


def upgrade(v1, *, audit, ticks, tokenizer=None):
    from dsv3wm.domains import _semantics, canonical, sha256
    old = legacy_builder()
    if (v1.get("schema") != "dsv3wm-rcwm-retention-v1" or len(v1["fixtures"]) != 24
            or not audit.get("available") or not ticks or any(type(value) is not int for value in ticks)):
        raise ValueError("Fixed v1 inputs and authenticated TRAIN input/tick audit required")
    env, fixtures, mapping = _semantics("ledger"), [], []
    register_index = 0
    for original in v1["fixtures"]:
        row = copy.deepcopy(original)
        family = row["family"]
        before, event = row["input"]["before"], row["input"]["event"]
        original_key = old.input_key(family, before, event)
        if original_key != row["semantic_input_id"] or original_key in audit["input_keys"][family]:
            raise ValueError("V1 operative input is exposed or has an incorrect semantic identity")
        if family == "register_machine":
            if set(event) != {"kind", "other", "register", "threshold", "value"}:
                raise ValueError("Expected the fixed five-field v1 R event surface")
            event["tick"] = ticks[register_index % len(ticks)]
            register_index += 1
            if set(event) != {"kind", "other", "register", "threshold", "tick", "value"}:
                raise ValueError("V2 register event must match all six original event fields")
        elif family == "prerequisite_plan":
            if set(event) != {"add", "cost", "delete", "kind", "name", "pre_off", "pre_on"}:
                raise ValueError("Original seven-field prerequisite event surface required")
        else:
            raise ValueError("R/P families only")
        new_key = old.input_key(family, before, event)
        if new_key != original_key or new_key in audit["input_keys"][family]:
            raise ValueError("Neutral tick must not disguise an exposed operative event")
        if (env.step(family, before, original["input"]["event"]) != env.step(family, before, event)
                or env.dumps(env.step(family, before, event)) != original["target"]):
            raise ValueError("Tick changes semantics or v1 target bytes")
        if old.parent_key(family, before) != row["parent_world_id"]:
            raise ValueError("V1/V2 initial parent identity mismatch")
        row["messages"] = env.one_step_messages(family, before, event)
        row["prompt"] = "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in row["messages"])
        row["prompt"] += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        row["original_train_input_exposure"] = "audited_semantically_novel"
        row["original_train_parent_exposure"] = "audited_seen" if row["parent_world_id"] in audit["parent_keys"][family] else "audited_novel"
        row["frame_relation_to_t2_train"] = ("original T2 SYSTEM_ONE event/message surface restored" if family == "register_machine"
                                            else "original Titan SYSTEM_ONE behavior; exact event schema, not T2 trained m1 frame parity")
        if tokenizer is not None:
            row["prompt_tokens"] = len(tokenizer.encode(row["prompt"], add_special_tokens=False).ids)
            row["target_tokens_including_eos"] = len(tokenizer.encode(row["target"], add_special_tokens=False).ids) + 1
            if row["prompt_tokens"] > 512 or row["target_tokens_including_eos"] > 64:
                raise ValueError("V2 exceeds unchanged caps; no truncation")
        else:
            row.pop("prompt_tokens", None)
            row.pop("target_tokens_including_eos", None)
        mapping.append({"v1_id": original["id"], "v2_id": row["id"], "family": family,
                        "parent_world_id": row["parent_world_id"], "graph_group_id": row["graph_group_id"],
                        "semantic_input_id": new_key, "targets_identical": row["target"] == original["target"],
                        "target_sha256": hashlib.sha256(row["target"].encode()).hexdigest(),
                        "v1_exact_input_sha256": sha256(original["input"]), "v2_exact_input_sha256": sha256(row["input"]),
                        "v1_prompt_sha256": hashlib.sha256(original["prompt"].encode()).hexdigest(),
                        "v2_prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                        "tick": event.get("tick"), "operative_key_unchanged": True})
        fixtures.append(row)
    if register_index != 12 or sum(row["family"] == "prerequisite_plan" for row in fixtures) != 12:
        raise ValueError("Fixed paired 12 R/12 P bank required")
    document = {"schema": "dsv3wm-rcwm-retention-v2", "seed": v1["seed"], "fixtures": fixtures,
                "family_counts": v1["family_counts"], "parent_counts": v1["parent_counts"],
                "graph_group_counts": v1["graph_group_counts"], "train_input_audit": audit["source"],
                "inference_contract": copy.deepcopy(v1["inference_contract"]),
                "tick_support": ticks, "tick_assignment": "cycle sorted audited original TRAIN tick values in fixed R case order",
                "claim_scope": "Supplementary R/P behavior during Ledger adaptation. R uses its original T2 one-step frame; P uses original Titan one-step framing, not T2 trained m1 parity. No broad retention pass; B0 failures do not establish skill loss.",
                "coverage_limitations": [
                    "V2 restores the original six-field R and seven-field P event surfaces; v1 remains an altered-surface record.",
                    "All 500 audited T2 P TRAIN prompts use SYSTEM_MULTI/events:[one event]; paired P fixtures intentionally use original Titan SYSTEM_ONE/event. This measures cross-frame P behavior, not trained-frame parity.",
                    "A P case failed by unchanged B0 cannot establish loss of a previously demonstrated capability under this frame.",
                    *v1["coverage_limitations"]],
                "source_sha256": {str(path): file_hash(path) for path in source_paths()}}
    return document, mapping


def main():
    from dsv3wm.native import EXPECTED_SIDECARS
    old = legacy_builder()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-fixtures", type=Path, default=ROOT / "datasets/rcwm-retention-v1/fixtures.json")
    parser.add_argument("--t2-train", type=Path, default=old.DEFAULT_TRAIN)
    parser.add_argument("--tokenizer-json", type=Path, default=old.DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, required=True, help="new v2 dataset directory")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V2 retention output exists")
    if file_hash(args.v1_fixtures) != V1_FIXTURE_SHA:
        raise ValueError("Original v1 fixture bytes drifted")
    if file_hash(args.tokenizer_json) != EXPECTED_SIDECARS["tokenizer.json"]:
        raise ValueError("Registered tokenizer bytes drifted")
    audit = old.training_input_audit(args.t2_train)
    ticks = original_tick_support(args.t2_train, expected_sha=old.TRAIN_SHA)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    document, mapping = upgrade(json.loads(args.v1_fixtures.read_text(encoding="utf-8")), audit=audit, ticks=ticks, tokenizer=tokenizer)
    frame_audit = original_frame_audit(args.t2_train, expected_sha=old.TRAIN_SHA)
    document["original_training_frame_audit"] = frame_audit
    document["tokenizer_json_sha256"] = file_hash(args.tokenizer_json)
    args.out.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for source in source_paths():
        relative = source.resolve().relative_to(REPO)
        destination = args.out / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        snapshots[relative.as_posix()] = {"sha256": file_hash(destination), "bytes": destination.stat().st_size,
                                         "snapshot": destination.relative_to(args.out).as_posix()}
    for name, value in (("fixtures.json", document), ("v1_to_v2.json", mapping)):
        with (args.out / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    manifest = {"schema": "dsv3wm-rcwm-retention-v2-manifest", "status": "prospective; no native v2 outcomes",
                "v1_fixture_path": str(args.v1_fixtures.resolve()), "v1_fixture_sha256": V1_FIXTURE_SHA,
                "v1_preserved": True, "source_snapshots": snapshots, "train_input_audit": audit["source"],
                "original_training_frame_audit": frame_audit, "frame_scope": document["claim_scope"],
                "files": {name: {"sha256": file_hash(args.out / name), "bytes": (args.out / name).stat().st_size}
                          for name in ("fixtures.json", "v1_to_v2.json")}, "native_model_loaded": False,
                "max_prompt_tokens": max(row["prompt_tokens"] for row in document["fixtures"]),
                "max_target_tokens_including_eos": max(row["target_tokens_including_eos"] for row in document["fixtures"])}
    with (args.out / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"out": str(args.out), "fixtures_sha256": manifest["files"]["fixtures.json"]["sha256"],
                      "max_prompt_tokens": manifest["max_prompt_tokens"],
                      "max_target_tokens_including_eos": manifest["max_target_tokens_including_eos"]}), flush=True)


if __name__ == "__main__":
    main()
