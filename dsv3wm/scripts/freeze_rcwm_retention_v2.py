"""Prospective T2-only B0 freeze for tick-complete R/P v2 and source closure."""
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))


def validate_pairing(v1_path, document, mapping, *, expected_sha):
    """Verify tick-only paired changes against actual immutable v1 bytes."""
    from dsv3wm.domains import canonical, sha256 as object_sha256
    from dsv3wm.native import sha256
    import copy
    import hashlib
    v1_path = Path(v1_path).resolve()
    if sha256(v1_path) != expected_sha:
        raise ValueError("Original v1 fixture bytes differ from the declared immutable source")
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    old, new = v1["fixtures"], document["fixtures"]
    if (v1.get("schema") != "dsv3wm-rcwm-retention-v1" or len(old) != 24 or len(new) != 24 or len(mapping) != 24
            or [row["id"] for row in old] != [row["id"] for row in new]):
        raise ValueError("V1/V2 fixed paired case/order mismatch")
    for original, upgraded, pair in zip(old, new, mapping):
        for key in ("id", "family", "cohort", "target", "parent_world_id", "graph_group_id", "semantic_input_id", "changed", "applicable"):
            if upgraded[key] != original[key]:
                raise ValueError(f"V1/V2 paired invariant changed: {key}")
        expected_input = copy.deepcopy(original["input"])
        if upgraded["family"] == "register_machine":
            tick = upgraded["input"]["event"].get("tick")
            if type(tick) is not int:
                raise ValueError("Paired R tick must be an integer")
            expected_input["event"]["tick"] = tick
        if canonical(upgraded["input"]) != canonical(expected_input):
            raise ValueError("Only neutral R tick may change the paired input")
        messages = [copy.deepcopy(original["messages"][0]), {"role": "user", "content": canonical(expected_input)}]
        prompt = "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages)
        prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if upgraded["messages"] != messages or upgraded["prompt"] != prompt:
            raise ValueError("Paired original rule/chat framing changed")
        expected_pair = {"v1_id": original["id"], "v2_id": upgraded["id"], "family": upgraded["family"],
                         "parent_world_id": upgraded["parent_world_id"], "graph_group_id": upgraded["graph_group_id"],
                         "semantic_input_id": upgraded["semantic_input_id"], "targets_identical": True,
                         "target_sha256": hashlib.sha256(upgraded["target"].encode()).hexdigest(),
                         "v1_exact_input_sha256": object_sha256(original["input"]), "v2_exact_input_sha256": object_sha256(upgraded["input"]),
                         "v1_prompt_sha256": hashlib.sha256(original["prompt"].encode()).hexdigest(),
                         "v2_prompt_sha256": hashlib.sha256(upgraded["prompt"].encode()).hexdigest(),
                         "tick": upgraded["input"]["event"].get("tick"), "operative_key_unchanged": True}
        if pair != expected_pair:
            raise ValueError("V1/V2 paired mapping evidence disagrees with actual cases")
    return v1_path


def main():
    from dsv3wm.native import sha256
    from dsv3wm.ledger_recovery import T2_ADAPTER, T2_SHA, T2_CONFIG_SHA
    spec = importlib.util.spec_from_file_location("_rp_v2_freeze_input_contract", ROOT / "scripts/evaluate_rcwm_retention_v2.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/rcwm-retention-v2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("R/P v2 freeze exists")
    if sha256(T2_ADAPTER / "adapter_model.safetensors") != T2_SHA or sha256(T2_ADAPTER / "adapter_config.json") != T2_CONFIG_SHA:
        raise ValueError("Registered T2 identity drift")
    fixture = args.dataset / "fixtures.json"
    module.validate_document(json.loads(fixture.read_text(encoding="utf-8")))
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["files"].items():
        if sha256(args.dataset / name) != metadata["sha256"]:
            raise ValueError("V2 fixture/mapping manifest digest drift")
    for relative, metadata in manifest["source_snapshots"].items():
        original, snapshot = REPO / relative, args.dataset / metadata["snapshot"]
        if sha256(original) != metadata["sha256"] or sha256(snapshot) != metadata["sha256"]:
            raise ValueError("V2 source snapshot/current source differs from construction manifest")
    document = json.loads(fixture.read_text(encoding="utf-8"))
    if any(sha256(path) != expected for path, expected in document["source_sha256"].items()):
        raise ValueError("V2 construction source drift")
    if manifest["v1_fixture_sha256"] != module.builder.V1_FIXTURE_SHA:
        raise ValueError("V2 manifest changed the immutable v1 linkage")
    original_v1 = validate_pairing(manifest["v1_fixture_path"], document,
                                   json.loads((args.dataset / "v1_to_v2.json").read_text(encoding="utf-8")),
                                   expected_sha=module.builder.V1_FIXTURE_SHA)
    paths = [ROOT / "configs/rcwm_retention_v1.json", *module.source_paths(), Path(__file__),
             ROOT / "scripts/run_measured.py", ROOT / "prereg/RECOVERY_RESOURCE_INSTRUMENT_V1.md",
             ROOT / "prereg/RECOVERY_RESOURCE_CLARIFICATION_2.md", T2_ADAPTER / "adapter_model.safetensors",
             T2_ADAPTER / "adapter_config.json", original_v1, *[path for path in args.dataset.rglob("*") if path.is_file()]]
    record = {"instrument": "rcwm-RP-retention-v2-B0", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "files": {str(path.resolve()).replace("\\", "/"): sha256(path) for path in sorted(set(paths))},
              "initial_T2_sha256": T2_SHA, "initial_T2_config_sha256": T2_CONFIG_SHA,
              "v1_fixture_sha256": manifest["v1_fixture_sha256"], "v2_fixture_sha256": sha256(fixture),
              "confirmation_access": False, "scope": "prospective tick-complete R/P B0; v1 is unchanged/unexecuted"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
