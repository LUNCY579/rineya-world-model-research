"""Causal trajectories, strict split boundaries and fresh confirmation sealing.

Training/inference projections never contain labels. Only collection or verifier
code can ask for label targets. No historical banks or frozen plans are read.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .domains import (DOMAINS, REPO, apply_delta, canonical, delta_state, parent_world_key,
                      rollout_labels, sample_actions, sample_world, semantic_key, sha256, source_hashes)

SCHEMA_VERSION = "dsv3wm-trajectory-v1"
HORIZONS = (1, 2, 4)
SPLITS = ("train", "dev", "confirmation")
STRESS = ("id", "invalid_action", "reordered_keys", "object_rename", "unseen_range", "long_actions",
          "unseen_object_count", "altered_rule_tokens")


def make_record(domain: str, s0: dict, actions: list, rules: dict, *, split: str = "train",
                seed: int = 0, stress: list[str] | None = None, hashes: dict | None = None) -> dict:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    record = {"schema_version": SCHEMA_VERSION, "split": split, "domain": domain,
              "parent_world_id": parent_world_key(domain, s0, rules), "rules": copy.deepcopy(rules),
              "s0": copy.deepcopy(s0), "actions": copy.deepcopy(actions),
              "labels": rollout_labels(domain, s0, actions, rules),
              "provenance": {"generator": "dsv3wm-fresh-v1", "seed": seed,
                             "source_sha256": hashes if hashes is not None else source_hashes()},
              "stress": list(stress or ["id"])}
    record["id"] = sha256({"domain": domain, "rules": rules, "s0": s0, "actions": actions})
    validate_record(record)
    return record


def validate_record(record: dict, *, labels_required: bool = True, verify_labels: bool = False) -> None:
    """Structural validation by default; optional exact replay for train/dev auditing.

    Exact replay is forbidden for confirmation rows. A sealed verifier must use its
    own explicit access path; ordinary data loading cannot expose/recompute labels.
    """
    required = {"schema_version", "id", "split", "domain", "parent_world_id", "rules", "s0", "actions",
                "provenance", "stress"}
    if not required <= set(record) or set(record) - required - {"labels"}:
        raise ValueError("Unexpected or missing trajectory fields")
    if record["schema_version"] != SCHEMA_VERSION or record["domain"] not in DOMAINS or record["split"] not in SPLITS:
        raise ValueError("Invalid trajectory schema/domain/split")
    if not isinstance(record["s0"], dict) or not isinstance(record["rules"], dict):
        raise ValueError("State and rules must be objects")
    if not isinstance(record["actions"], list) or not 4 <= len(record["actions"]) <= 32:
        raise ValueError("A trajectory must have 4..32 actions")
    if "labels" not in record:
        if labels_required:
            raise ValueError("Missing labels")
    else:
        labels = record["labels"]
        expected = {"states", "legal", "terminated", "failed", "invalid", "executed"}
        if not isinstance(labels, dict) or set(labels) != expected:
            raise ValueError("Invalid label fields")
        n = len(record["actions"])
        if any(not isinstance(v, list) or len(v) != n for v in labels.values()):
            raise ValueError("Labels and actions have different lengths")
        if any(type(v) is not bool for k in expected - {"states"} for v in labels[k]):
            raise ValueError("Flags must be booleans")
        if any(not isinstance(state, dict) for state in labels["states"]):
            raise ValueError("States must be objects")
    if verify_labels:
        if record["split"] == "confirmation":
            raise PermissionError("Confirmation label replay is forbidden in the ordinary loader")
        if rollout_labels(record["domain"], record["s0"], record["actions"], record["rules"]) != record["labels"]:
            raise ValueError("Causal labels disagree with exact semantics")


def conditioning(record: dict, horizon: int) -> dict:
    """Pure causal projection, safe for model inference and all splits."""
    if type(horizon) is not int or not 1 <= horizon <= len(record["actions"]):
        raise ValueError("Invalid horizon")
    return copy.deepcopy({"domain": record["domain"], "rules": record["rules"], "s0": record["s0"],
                          "actions": record["actions"][:horizon]})


def causal_prompt(record: dict, horizon: int) -> str:
    # Preserve presentation order so reordered representation probes survive I/O.
    inputs = json.dumps(conditioning(record, horizon), separators=(",", ":"), allow_nan=False)
    return "Predict the state after these ordered actions and the final action's legality. Input=" + inputs + "\nOutput="


def target(record: dict, horizon: int, mode: str = "full") -> dict:
    """Training/development target. Ordinary callers cannot read confirmation labels."""
    if record.get("split") == "confirmation":
        raise PermissionError("Confirmation targets are sealed")
    if mode not in ("full", "delta"):
        raise ValueError("Target mode must be full or delta")
    conditioning(record, horizon)  # horizon validation, without touching later labels
    labels, index = record["labels"], horizon - 1
    output = {"legal": labels["legal"][index], "terminated": labels["terminated"][index],
              "failed": labels["failed"][index]}
    state = labels["states"][index]
    if mode == "full":
        output["state"] = copy.deepcopy(state)
    else:
        output["delta"] = delta_state(record["domain"], record["s0"], state)
    return output


def training_examples(records: list[dict], *, horizons: tuple[int, ...] = HORIZONS,
                      target_mode: str = "full") -> list[dict]:
    examples = []
    for record in records:
        if record["split"] not in ("train", "dev"):
            raise PermissionError("Only training/development records may expose targets")
        for horizon in horizons:
            examples.append({"id": record["id"], "parent_world_id": record["parent_world_id"], "horizon": horizon,
                             "domain": record["domain"], "prompt": causal_prompt(record, horizon),
                             "target": canonical(target(record, horizon, target_mode))})
    return examples


def load_records(path: str | Path, *, expected_split: str | None = None) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if expected_split is not None and row["split"] != expected_split:
                raise ValueError("Unexpected data split")
            if row["split"] == "confirmation" and "labels" in row:
                raise PermissionError("Use the independent verifier for sealed confirmation labels")
            validate_record(row, labels_required=row["split"] != "confirmation")
            rows.append(row)
    return rows


def integrity_audit(splits: dict[str, list[dict]]) -> dict:
    """No label access: verify parent, exact and semantic identity across splits."""
    owners: dict[str, dict[str, str]] = {kind: {} for kind in ("parent", "exact", "semantic")}
    counts = {}
    for split, rows in splits.items():
        within: set[str] = set()
        for row in rows:
            if row["split"] != split:
                raise ValueError("Split identity disagrees with containing split")
            parent = parent_world_key(row["domain"], row["s0"], row["rules"])
            if parent != row["parent_world_id"]:
                raise ValueError("Incorrect parent identity")
            inputs = conditioning(row, len(row["actions"]))
            # Exact presentation fingerprint; semantic one ignores keys/renaming/prose.
            exact = hashlib.sha256(json.dumps(inputs, separators=(",", ":")).encode()).hexdigest()
            sem = semantic_key(row["domain"], row["s0"], row["actions"], row["rules"])
            if sem in within:
                raise ValueError("Duplicate semantic trajectory within split")
            within.add(sem)
            for kind, value in (("parent", parent), ("exact", exact), ("semantic", sem)):
                old = owners[kind].get(value)
                if old is not None and old != split:
                    raise ValueError(f"Cross-split {kind} contamination: {old}/{split}")
                owners[kind][value] = split
        counts[split] = {"rows": len(rows), "parent_worlds": len({r["parent_world_id"] for r in rows})}
    return {"counts": counts, "parent_disjoint": True, "exact_disjoint": True, "semantic_disjoint": True,
            "object_renaming_canonicalization": "exact permutation minimization, <=5 objects"}


def _stress_for(split: str, domain: str, index: int) -> str:
    if split == "train":
        return "invalid_action" if index % 7 == 1 else "id"
    options = ["id", "invalid_action", "reordered_keys", "long_actions", "altered_rule_tokens"]
    if domain in ("ledger", "blocks"):
        options.append("object_rename")
    if split == "confirmation":
        options.append("unseen_object_count" if domain == "blocks" else "unseen_range")
    return options[index % len(options)]


def _reorder(state: dict, rules: dict, actions: list, rng: random.Random) -> tuple:
    def ordered(obj):
        if isinstance(obj, dict):
            keys = list(obj)
            rng.shuffle(keys)
            return {key: ordered(obj[key]) for key in keys}
        if isinstance(obj, list):
            return [ordered(item) for item in obj]
        return obj
    state, rules, actions = ordered(state), ordered(rules), ordered(actions)
    # Fact list order carries no meaning; action list order always does.
    for key in ("clear", "table", "on"):
        if key in state:
            rng.shuffle(state[key])
    return state, rules, actions


def generate_splits(seed: int, rows_per_split: int = 120) -> dict[str, list[dict]]:
    if type(seed) is not int or type(rows_per_split) is not int or rows_per_split < 3:
        raise ValueError("Integer seed and >=3 rows per split required")
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {split: [] for split in SPLITS}
    hashes, parent_owners, semantic_seen = source_hashes(), {}, set()
    for split in SPLITS:
        tries = 0
        while len(splits[split]) < rows_per_split:
            tries += 1
            if tries > rows_per_split * 500:
                raise RuntimeError("Fresh parent/semantic sampling exhausted; reduce rows or broaden protocol")
            index = len(splits[split])
            domain = DOMAINS[index % len(DOMAINS)]
            stress = _stress_for(split, domain, index // len(DOMAINS))
            s0, rules = sample_world(domain, rng, stress=stress)
            length = 8 if stress == "long_actions" else 4
            actions = sample_actions(domain, s0, rules, rng, length=length, stress=stress)
            if stress == "reordered_keys":
                s0, rules, actions = _reorder(s0, rules, actions, rng)
            if stress == "altered_rule_tokens":
                rules["text"] = "Display label: easy-success-legal. Read the complete operative rules: " + rules["text"]
            parent = parent_world_key(domain, s0, rules)
            # Hash assigns parent ownership before draws, eliminating order-induced
            # exhaustion and keeping every rename/presentation variant in one split.
            bucket = int(parent[:8], 16) % 10
            owner = "train" if bucket < 6 else "dev" if bucket < 8 else "confirmation"
            if owner != split or (parent in parent_owners and parent_owners[parent] != split):
                continue
            sem = semantic_key(domain, s0, actions, rules)
            if sem in semantic_seen:
                continue
            semantic_seen.add(sem)
            parent_owners[parent] = split
            record = make_record(domain, s0, actions, rules, split=split, seed=seed, stress=[stress], hashes=hashes)
            splits[split].append(record)
    integrity_audit(splits)
    return splits


def _bytes(rows: list[dict]) -> bytes:
    return ("".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)).encode("utf-8")


def write_dataset(output: str | Path, vault: str | Path, *, seed: int = 20261011,
                  rows_per_split: int = 120) -> dict:
    """Create once. Full confirmation rows are written only to an external vault.

    Hashes are computed from generated bytes, with no post-sealing label reads or
    label logging. Manifests report inputs/provenance/integrity, never label values.
    """
    output, vault = Path(output).resolve(), Path(vault).resolve()
    if output.exists() or vault.exists():
        raise FileExistsError("Dataset/vault destinations must not already exist")
    try:
        vault.relative_to(REPO)
    except ValueError:
        pass
    else:
        raise ValueError("Confirmation vault must be outside the repository")
    if output == vault or output in vault.parents or vault in output.parents:
        raise ValueError("Dataset and vault directories must not overlap")
    splits = generate_splits(seed, rows_per_split)
    audit = integrity_audit(splits)
    files, private = {}, {}
    for split in SPLITS:
        rows = splits[split]
        if split == "confirmation":
            private["confirmation.labels.jsonl"] = _bytes(rows)
            rows = [{key: value for key, value in row.items() if key != "labels"} for row in rows]
        files[f"{split}.jsonl"] = _bytes(rows)
    manifest = {"schema_version": SCHEMA_VERSION, "generator": "dsv3wm-fresh-v1", "seed": seed,
                "rows_per_split": rows_per_split, "horizons": list(HORIZONS), "source_sha256": source_hashes(),
                "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                          for name, content in files.items()},
                "sealed_confirmation": {"location": str(vault / "confirmation.labels.jsonl"),
                                        "sha256": hashlib.sha256(private["confirmation.labels.jsonl"]).hexdigest(),
                                        "bytes": len(private["confirmation.labels.jsonl"]),
                                        "access": "independent verifier only; do not use for tuning"},
                "integrity": audit,
                "stress_counts": {split: {stress: sum(stress in row["stress"] for row in rows)
                                          for stress in STRESS} for split, rows in splits.items()},
                "coverage_limitations": [
                    "Dynamics records contain no goals; held-out goal families and hard planning require a separate benchmark.",
                    "Only fixed executor rule families are covered; unseen ranges/counts are fresh parameter stresses, not new transition laws.",
                    "Novel rule-composition withholding is not guaranteed; it needs a separately frozen composition registry.",
                    "Renaming deduplication is exact only for <=5 objects; this generator rejects larger domains.",
                    "Unknown-action legality follows documented wrapper no-op semantics, extending historical parser behavior.",
                    "Stress cohorts use disjoint parents; matched counterfactual probes should use permitted dev parents only.",
                    "Disjointness is checked within this new dataset; historical sealed banks are neither accessed nor cross-compared.",
                ]}
    # Both directories are created with exclusive behavior before any files are written.
    output.mkdir(parents=True, exist_ok=False)
    vault.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (output / name).write_bytes(content)
    for name, content in private.items():
        (vault / name).write_bytes(content)
    # Ignore everything in the external vault if it later becomes part of a Git root.
    (vault / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
