"""Fresh, information-rich RegisterWorld G3 recovery collection.

Exact rules and BFS are collection instruments only. No confirmation bank is
created, no native model is imported and no historical evaluation row is read.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
from collections import deque
from functools import lru_cache
from pathlib import Path

from .data import integrity_audit, make_record
from .domains import REPO, _semantics, canonical, parent_world_key, rules_for, sha256, source_hashes

VERSION = "dsv3wm-register-recovery-v1"
TRAIN_LIMITS = (24, 32, 40)
ID_LIMITS = (32, 48, 64)
OOD_LIMITS = (80, 96)
MOVES = ("add1", "sub1", "double", "halve")


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recovery_sources() -> dict:
    hashes = source_hashes()
    for path in (Path(__file__), REPO / "dsv3wm/src/dsv3wm/data.py", REPO / "dsv3wm/src/dsv3wm/domains.py",
                 REPO / "dsv3wm/scripts/build_recovery_data.py"):
        hashes[path.relative_to(REPO).as_posix()] = file_hash(path)
    return hashes


def read_prior_public(paths: list[str | Path]) -> tuple[set[str], dict]:
    """Read only known public trajectory files, extracting input parent identities.

    All old parents are excluded, even same-split ones. This is stricter than just
    cross-split avoidance and enables an explicit new registry rather than reusing
    the finite 60% hash-owner pool. Never accept a label-vault filename.
    """
    excluded, sources = set(), {}
    resolved = [Path(value).resolve() for value in paths]
    expected = {"train.jsonl", "dev.jsonl", "confirmation.jsonl"}
    groups: dict[Path, set[str]] = {}
    for path in resolved:
        if path.name not in {"train.jsonl", "dev.jsonl", "confirmation.jsonl"}:
            raise ValueError("Prior exclusions accept public train/dev/confirmation.jsonl files only")
        groups.setdefault(path.parent, set()).add(path.name)
    if not resolved or any(names != expected for names in groups.values()):
        raise ValueError("Each prior dataset requires all three public split files for complete exclusions")
    for path in resolved:
        raw = path.read_bytes()
        sources[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") not in {"train", "dev", "confirmation"}:
                raise ValueError("Invalid prior public split")
            if row["split"] != path.stem:
                raise ValueError("Prior public split/file identity mismatch")
            if row["split"] == "confirmation" and "labels" in row:
                raise PermissionError("Confirmation labels are forbidden; use public inputs only")
            parent = parent_world_key(row["domain"], row["s0"], row["rules"])
            if parent != row["parent_world_id"]:
                raise ValueError("Prior public parent identity drift")
            excluded.add(parent)
    return excluded, sources


@lru_cache(maxsize=2048)
def collection_distances(x: int, limit: int) -> dict[int, int]:
    """Exact legal-move distances only; does not construct or save solver plans."""
    if type(x) is not int or type(limit) is not int or limit <= 0 or abs(x) > limit:
        raise ValueError("Invalid RegisterWorld collection state/limit")
    executor = _semantics("register").exact_step
    distances, queue = {x: 0}, deque([x])
    while queue:
        state = queue.popleft()
        for action in MOVES:
            after, legal, _ = executor(state, action, limit)
            if legal and after not in distances:
                distances[after] = distances[state] + 1
                queue.append(after)
    return distances


def _options(x: int, limit: int, *, first_legal: bool, balanced: bool,
             finish_last: bool = False, min_distance: int = 1) -> list[list[str]]:
    executor, result = _semantics("register").exact_step, []
    suffix = ("finish",) if finish_last else MOVES
    distances = collection_distances(x, limit)
    for prefix in itertools.product(MOVES, repeat=3):
        for last in suffix:
            actions, state, flags = [*prefix, last], x, []
            for action in actions:
                state, legal, _ = executor(state, action, limit)
                flags.append(legal)
            first_state, _, _ = executor(x, actions[0], limit)
            if (flags[0] != first_legal or (first_legal and first_state == x) or state == x
                    or distances[state] < min_distance or (balanced and sum(flags) != 2)):
                continue
            result.append(actions)
    return result


def _pool(limits: tuple[int, ...], rng: random.Random, *, high_range: bool | None = None) -> list[tuple[int, int]]:
    candidates = [(x, limit) for limit in limits for x in range(-limit, limit + 1)
                  if high_range is None or (abs(x) >= 65) == high_range]
    rng.shuffle(candidates)
    return candidates


def _select(pool: list[tuple[int, int]], count: int, rng: random.Random, used: set[str], *, split: str,
            first_legal: bool, cohort: str, hashes: dict, seed: int, balanced: bool = False,
            min_distance: int = 1, terminal_rows: int = 0) -> list[dict]:
    rows = []
    for x, limit in pool:
        rules = rules_for("register", limit=limit)
        parent = parent_world_key("register", {"x": x}, rules)
        if parent in used:
            continue
        terminal = len(rows) < terminal_rows
        choices = _options(x, limit, first_legal=first_legal, balanced=balanced,
                           finish_last=terminal, min_distance=min_distance)
        if not choices:
            continue
        actions = rng.choice(choices)
        row = make_record("register", {"x": x}, actions, rules, split=split, seed=seed,
                          stress=[cohort, "first_legal_changed" if first_legal else "first_illegal"], hashes=hashes)
        row["provenance"].update(generator=VERSION, cohort=cohort,
                                 seen_by_initial_F_one_step=limit in ID_LIMITS,
                                 novelty="new parent trajectory; ID one-step teacher exposure is known" if limit in ID_LIMITS
                                 else "unseen limit for the original registered F")
        rows.append(row)
        used.add(parent)
        if len(rows) == count:
            return rows
    raise RuntimeError(f"Not enough fresh {cohort}/first_legal={first_legal} parents for {count} records")


def _task(row: dict, rng: random.Random, low: int, high: int) -> tuple[dict, dict]:
    x, limit = row["s0"]["x"], row["rules"]["limit"]
    distances = collection_distances(x, limit)
    # Goal differs from the trajectory's h4 label and is not selected from a solver plan.
    goals = sorted(goal for goal, distance in distances.items() if low <= distance <= high
                   and goal != row["labels"]["states"][3]["x"])
    if not goals:
        raise RuntimeError("No independent goal in the registered collection distance range")
    goal = rng.choice(goals)
    task = {"split": row["split"], "domain": "register", "parent_world_id": row["parent_world_id"],
            "rules": copy.deepcopy(row["rules"]), "s0": copy.deepcopy(row["s0"]),
            "goal": {"kind": "state", "state": {"x": goal}},
            "provenance": {"generator": VERSION, "source_trajectory_id": row["id"],
                           "source_sha256": row["provenance"]["source_sha256"],
                           "selection": "fresh desired state; exact distance used only for collection"}}
    task["id"] = sha256({key: task[key] for key in ("domain", "rules", "s0", "goal")})
    scoring = {"task_id": task["id"], "parent_world_id": row["parent_world_id"], "split": row["split"],
               "optimal_moves": distances[goal], "cohort": row["provenance"]["cohort"],
               "seen_by_initial_F_one_step": limit in ID_LIMITS,
               "score_policy": "independent exact scoring after learned selection; no feedback to search"}
    return task, scoring


def build_recovery(seed: int = 202610112, *, exclusions: set[str] | None = None) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    rng, used, hashes = random.Random(seed), set(exclusions or ()), recovery_sources()
    prior = set(used)
    dev = []
    # Reserve informative DEV parents before allocating the finite TRAIN pool.
    # Each ID/OOD cohort has exactly eight legal-changed and four illegal starts.
    for cohort, limits in (("recovery_id_teacher_known", ID_LIMITS), ("recovery_ood_limit", OOD_LIMITS)):
        groups = [(False, 4, None), (True, 8, None)] if limits == ID_LIMITS else [
            (False, 2, True), (False, 2, False), (True, 4, True), (True, 4, False)]
        for legal, count, high_range in groups:
            dev.extend(_select(_pool(limits, rng, high_range=high_range), count, rng, used, split="dev",
                               first_legal=legal, cohort=cohort, hashes=hashes, seed=seed, min_distance=2))
    train = []
    # Exactly two legal and two illegal transitions per trajectory: 256/256 labels.
    # Reserve scarce illegal-start parents first; four finish-at-h4 cases per group.
    for legal in (False, True):
        train.extend(_select(_pool(TRAIN_LIMITS, rng), 64, rng, used, split="train", first_legal=legal,
                             cohort="recovery_train", hashes=hashes, seed=seed, balanced=True, terminal_rows=4))
    rng.shuffle(train)
    rng.shuffle(dev)
    invalid = []
    for limits in (ID_LIMITS, OOD_LIMITS):
        for x, limit in _pool(limits, rng):
            rules = rules_for("register", limit=limit)
            parent = parent_world_key("register", {"x": x}, rules)
            if parent in used:
                continue
            action = "sub1" if x >= limit else "add1"
            row = make_record("register", {"x": x}, ["teleport", action, action, action], rules,
                              split="dev", seed=seed, stress=["recovery_invalid_action_extra"], hashes=hashes)
            if row["labels"]["states"][3] == row["s0"]:
                continue
            row["provenance"].update(generator=VERSION, cohort="recovery_invalid_action_extra",
                                     primary=False, seen_by_initial_F_one_step=limit in ID_LIMITS)
            invalid.append(row)
            used.add(parent)
            if len(invalid) in (2, 4):
                break
    if len(invalid) != 4:
        raise RuntimeError("Could not collect four separate invalid stress parents")
    selected_dev = []
    for cohort in ("recovery_id_teacher_known", "recovery_ood_limit"):
        choices = [row for row in dev if row["provenance"]["cohort"] == cohort]
        rng.shuffle(choices)
        selected_dev.extend(choices[:6])
    planning_train, planning_dev, scoring = [], [], []
    for rows, destination, low, high in ((train[:16], planning_train, 2, 4), (selected_dev, planning_dev, 3, 5)):
        for row in rows:
            task, label = _task(row, rng, low, high)
            destination.append(task)
            scoring.append(label)
    audit = integrity_audit({"train": train, "dev": [*dev, *invalid]})
    if prior & {row["parent_world_id"] for row in [*train, *dev, *invalid]}:
        raise AssertionError("Recovery reused an excluded prior parent")
    return {"train": train, "dev": dev, "dev_invalid": invalid, "planning_train": planning_train,
            "planning_dev": planning_dev, "planning_scoring": scoring, "integrity": audit,
            "parent_registry": {row["parent_world_id"]: {"split": row["split"], "cohort": row["provenance"]["cohort"]}
                                for row in [*train, *dev, *invalid]}, "source_sha256": hashes}


def legacy_target(row: dict, horizon: int = 1) -> dict:
    """Pure label projection for the registered x/legal/terminated response schema."""
    if row.get("split") not in {"train", "dev"} or row.get("domain") != "register":
        raise PermissionError("Legacy target projection accepts TRAIN/DEV RegisterWorld only")
    if type(horizon) is not int or not 1 <= horizon <= 4:
        raise ValueError("Recovery horizons are 1..4")
    index, labels = horizon - 1, row["labels"]
    return {"x": labels["states"][index]["x"], "legal": labels["legal"][index],
            "terminated": labels["terminated"][index]}


def write_recovery(output: str | Path, *, seed: int = 202610112, prior_public: list[str | Path]) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Recovery destination already exists")
    if not prior_public:
        raise ValueError("Prior public input exclusions are required; no vault labels are accepted")
    exclusions, prior_sources = read_prior_public(prior_public)
    data = build_recovery(seed, exclusions=exclusions)
    contents = {}
    for name in ("train", "dev", "dev_invalid", "planning_train", "planning_dev"):
        contents[name + ".jsonl"] = ("".join(canonical(row) + "\n" for row in data[name])).encode("utf-8")
    for name in ("planning_scoring", "parent_registry"):
        contents[name + ".json"] = (json.dumps(data[name], sort_keys=True, indent=2) + "\n").encode("utf-8")
    train, dev = data["train"], data["dev"]
    manifest = {"generator": VERSION, "seed": seed, "status": "frozen G3 TRAIN/DEV collection; no confirmation generated",
                "counts": {name: len(data[name]) for name in ("train", "dev", "dev_invalid", "planning_train", "planning_dev")},
                "primary_parent_counts": {"train": len({r["parent_world_id"] for r in train}),
                                          "dev": len({r["parent_world_id"] for r in dev})},
                "horizons": [1, 2, 4], "confirmation_rows": 0, "confirmation_labels_accessed": False,
                "split_ownership": "explicit recovery parent registry; excludes every prior public parent in every split",
                "prior_public_sources": prior_sources, "excluded_prior_parent_count": len(exclusions),
                "source_sha256": data["source_sha256"], "integrity": data["integrity"],
                "train_legality": {"first_legal": sum(r["labels"]["legal"][0] for r in train),
                                   "first_illegal": sum(not r["labels"]["legal"][0] for r in train),
                                   "all_legal": sum(sum(r["labels"]["legal"]) for r in train),
                                   "all_illegal": sum(sum(not v for v in r["labels"]["legal"]) for r in train)},
                "all_primary_h4_changed": all(r["labels"]["states"][3] != r["s0"] for r in [*train, *dev]),
                "dev_cohorts": {cohort: {"rows": len(rows), "first_legal_changed": sum(r["labels"]["legal"][0] for r in rows),
                                        "first_illegal": sum(not r["labels"]["legal"][0] for r in rows),
                                        "abs_x_65_plus": sum(abs(r["s0"]["x"]) >= 65 for r in rows),
                                        "limits": sorted({r["rules"]["limit"] for r in rows})}
                                for cohort in ("recovery_id_teacher_known", "recovery_ood_limit")
                                for rows in [[r for r in dev if r["provenance"]["cohort"] == cohort]]},
                "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                          for name, raw in contents.items()},
                "coverage_limitations": [
                    "ID32/48/64 one-step keys were exposed to the original F; this is novel trajectory composition, not teacher-unseen one-step generalization.",
                    "OOD80/96 are unseen old-F rule limits; six OOD initial states have abs(x)>=65. Recovery student training uses24/32/40.",
                    "Balanced legality deliberately changes the training distribution; transfer to natural action frequencies is unmeasured.",
                    "Only RegisterWorld is collected. Goal distances select development/teacher tasks, not a sealed hard-planning benchmark.",
                    "Task inputs contain desired goals but no labels, distances, candidate plans or solver traces; scoring metadata is separate.",
                    "No confirmation split or scientific efficacy inference is authorized by this recovery collection.",
                    "Old sealed banks and vault labels are never inspected; exclusion checks use new v1 public inputs only.",
                ]}
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in contents.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest
