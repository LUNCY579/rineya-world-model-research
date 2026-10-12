"""Pure Ledger collection and registered-T2-compatible causal interfaces.

No model/runtime imports. The six-field event contract includes neutral tick.
Exact rules run only during collection/replay, never inside prompt construction,
response parsing or own-state rollout. This deliberately bypasses the frozen
generic ledger wrapper, which rejects tick.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import REPO
from .data import SCHEMA_VERSION
from .domains import _semantics, canonical, parent_world_key, semantic_key

VERSION = "dsv3wm-ledger-recovery-v1"
ACCOUNTS = ("A", "B", "C", "D")
EVENT_FIELDS = frozenset(("account", "amount", "kind", "other", "status", "tick"))
KINDS = ("add", "scale", "transfer")
SURFACES = ("canonical", "reversed_keys")
T2_ADAPTER = REPO / "rcwm/experiments/RCWM-T1/runs/t2/t2_s0/adapter"
T2_SHA = "92bc7c7497c3cb2fc0c7c917b4f3b6bb931f0ed108144d969b26b18b42670b8a"
T2_CONFIG_SHA = "50d9a776e093a952ac440a77c9cd7cd8e51eb22a85eb93b383aee1eacc185d9d"
T2_TRAIN = REPO / "rcwm/experiments/RCWM-T1/data/t2_train.jsonl"
T2_TRAIN_SHA = "bf5a22765757856517496b64e54d4d7b307dbb50af8f4e30540ff69937a1d744"
COHORTS = {"canonical_id": 12, "held_status": 4, "reordered_keys": 4, "shifted_tick": 4}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = [Path(__file__), REPO / "rcwm/experiments/RCWM-T1/rcwmt/env.py",
             REPO / "bwm/code/bwmn/rw2.py", REPO / "dsv3wm/src/dsv3wm/domains.py",
             REPO / "dsv3wm/src/dsv3wm/data.py"]
    script = REPO / "dsv3wm/scripts/build_ledger_recovery.py"
    if script.exists():
        paths.append(script)
    return {p.relative_to(REPO).as_posix(): file_sha256(p) for p in paths}


def validate_balances(state):
    if (not isinstance(state, dict) or set(state) != set(ACCOUNTS)
            or any(type(state[k]) is not int for k in ACCOUNTS)):
        raise ValueError("Canonical four-account integer balances required; renamed accounts are unsupported")
    return {key: state[key] for key in ACCOUNTS}


def event_is_valid(event):
    """Input syntax only; status is never an illegality predicate.

    Same-account transfers are mathematically legal under env.step, although
    primary collection excludes them to avoid posted no-op/copy shortcuts.
    """
    return (isinstance(event, dict) and set(event) == EVENT_FIELDS
            and isinstance(event["kind"], str) and event["kind"] in KINDS
            and isinstance(event["account"], str) and event["account"] in ACCOUNTS
            and isinstance(event["other"], str) and event["other"] in ACCOUNTS
            and type(event["amount"]) is int and type(event["tick"]) is int
            and isinstance(event["status"], str))


def ledger_rules(surface="canonical"):
    if surface not in SURFACES:
        raise ValueError("Unsupported ledger presentation")
    env = _semantics("ledger")
    return {"semantics": "rcwm-ledger-replay", "objects": list(ACCOUNTS),
            "text": env.SYSTEM_MULTI["ledger_replay"], "presentation": surface,
            "input_contract": "six fields; tick ignored by dynamics; malformed events are wrapper illegal no-ops"}


def validate_conditioning(inputs):
    if not isinstance(inputs, dict) or set(inputs) != {"domain", "rules", "s0", "actions"}:
        raise ValueError("Only causal ledger conditioning is accepted; labels/goals forbidden")
    rules = inputs["rules"]
    if (inputs["domain"] != "ledger" or not isinstance(rules, dict)
            or rules.get("semantics") != "rcwm-ledger-replay"
            or rules.get("objects") != list(ACCOUNTS)
            or rules.get("presentation", "canonical") not in SURFACES):
        raise ValueError("Fixed canonical Ledger semantics/accounts required")
    validate_balances(inputs["s0"])
    if not isinstance(inputs["actions"], list) or not 1 <= len(inputs["actions"]) <= 32:
        raise ValueError("A nonempty ordered ledger event prefix is required")
    if any(isinstance(event, dict) and not set(event) <= EVENT_FIELDS for event in inputs["actions"]):
        raise ValueError("Extra event metadata/labels are forbidden in prediction conditioning")


def ledger_prompt(inputs):
    """H1 is byte-identical T2 for valid canonical events; no oracle calls.

    Reversed keys and invalid-event wrapper instructions are named extensions.
    H2/H4 use the existing explicit multi-step rule and initial balances only.
    """
    validate_conditioning(inputs)
    env = _semantics("ledger")
    before = validate_balances(inputs["s0"])
    events = copy.deepcopy(inputs["actions"])
    if len(events) == 1:
        messages = env.one_step_messages("ledger_replay", before, events[0])
    else:
        messages = env.multi_step_messages("ledger_replay", before, events)
    if any(not event_is_valid(e) for e in events):
        messages[0]["content"] += " Malformed events are illegal and leave all balances unchanged."
    if inputs["rules"].get("presentation", "canonical") == "reversed_keys":
        def reverse(obj):
            if isinstance(obj, dict):
                return {key: reverse(obj[key]) for key in reversed(list(obj))}
            if isinstance(obj, list):
                return [reverse(value) for value in obj]
            return obj
        messages[1]["content"] = json.dumps(reverse(json.loads(messages[1]["content"])), separators=(",", ":"))
    return _semantics("register").render_chat(messages[0]["content"], messages[1]["content"])


def parse_ledger_response(text, inputs, *, include_flags=True):
    """Strict bare balances; optional flags derive only from input syntax.

    The model does not predict legality/termination. No status rule, target,
    exact transition, clipping, repair or gold state is used in this parser.
    """
    validate_conditioning(inputs)
    if type(include_flags) is not bool:
        raise ValueError("include_flags must be boolean")
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant")
    try:
        value = json.loads(text.strip(), object_pairs_hook=unique, parse_constant=reject_constant)
        result = {"state": validate_balances(value)}
        if include_flags:
            valid = [event_is_valid(event) for event in inputs["actions"]]
            result.update(legal=valid[-1], terminated=False, failed=not all(valid),
                          flags_source="known input contract; not model predictions")
        return result
    except (ValueError, TypeError, AttributeError, OverflowError):
        return {"malformed": True, "text": text}


def exact_ledger_transition(before, event):
    """Collection only. Canonical tick-compatible validity plus audited env.step."""
    before = validate_balances(before)
    if not event_is_valid(event):
        return copy.deepcopy(before), False, False, True
    after = _semantics("ledger").step("ledger_replay", before, copy.deepcopy(event))
    return validate_balances(after), True, False, False


def ledger_rollout_labels(s0, events):
    state = validate_balances(s0)
    labels = {key: [] for key in ("states", "legal", "terminated", "failed", "invalid", "executed")}
    failed = False
    for event in events:
        state, legal, terminated, invalid = exact_ledger_transition(state, event)
        failed = failed or not legal
        for key, value in (("states", copy.deepcopy(state)), ("legal", legal), ("terminated", terminated),
                           ("failed", failed), ("invalid", invalid), ("executed", True)):
            labels[key].append(value)
    return labels


def ledger_parent_id(before):
    return parent_world_key("ledger", validate_balances(before), ledger_rules())


def operative_key(before, event):
    """Renaming-safe input key, ignoring neutral tick and unused other fields.

    Status spelling is retained: a held-out token cue is part of this test.
    This is operative-input identity, not a claim all no-op queries are distinct
    mathematical functions. Does not execute dynamics or inspect target labels.
    """
    if not event_is_valid(event):
        raise ValueError("Operative key requires a valid canonical event")
    fields = ("kind", "account", "amount", "status", "other") if event["kind"] == "transfer" else (
        "kind", "account", "amount", "status")
    ev = {key: event[key] for key in fields}
    return semantic_key("ledger", validate_balances(before), [ev], ledger_rules())


@dataclass(frozen=True)
class TrainingSupport:
    parents: frozenset = field(default_factory=frozenset)
    keys: frozenset = field(default_factory=frozenset)
    amounts: tuple = ()
    balances: tuple = ()
    metadata: dict = field(default_factory=lambda: {"available": False, "exact_prior_input_exposure": "unknown"})


def load_training_support(path=T2_TRAIN, *, expected_sha256=T2_TRAIN_SHA):
    """Only registered original TRAIN is accepted; no eval/bank/vault path.

    Unit fixtures may pass their own expected digest. Labels/prompt targets are
    never used or returned; support comes only from before/event/family fields.
    """
    path = Path(path)
    if not path.exists():
        return TrainingSupport(metadata={"available": False, "requested_path": str(path),
                                         "exact_prior_input_exposure": "unknown"})
    if path.name != "t2_train.jsonl":
        raise ValueError("Only original t2_train.jsonl is permitted for support extraction")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Registered TRAIN support hash mismatch")
    parents, keys, amounts, balances, strata = set(), set(), set(), set(), Counter()
    count = 0
    for line in raw.decode("utf-8").splitlines():
        row = json.loads(line)
        if row.get("family") != "ledger_replay":
            continue
        before, event = validate_balances(row["before"]), row["event"]
        parents.add(ledger_parent_id(before)); keys.add(operative_key(before, event))
        amounts.add(event["amount"]); balances.update(before.values())
        strata[f"{event['kind']}/{event['status']}"] += 1; count += 1
    if not count or (actual == T2_TRAIN_SHA and count != 501):
        raise ValueError("Original ledger TRAIN quota mismatch")
    digest = hashlib.sha256(canonical(sorted(keys)).encode()).hexdigest()
    return TrainingSupport(frozenset(parents), frozenset(keys), tuple(sorted(amounts)), tuple(sorted(balances)),
        {"available": True, "source_path": str(path.resolve()), "source_sha256": actual,
         "ledger_rows": count, "operative_key_count": len(keys), "operative_support_sha256": digest,
         "unique_canonical_parents": len(parents), "amounts": sorted(amounts),
         "max_abs_before": max(abs(v) for v in balances), "strata": dict(sorted(strata.items())),
         "key_policy": "account-renaming invariant; ignore tick and unused other; preserve status word",
         "limits": "known input exposure only; frozen-parent prior exposure and historical eval overlap are unknown"})


def prior_public_parents(paths):
    """Known new public inputs only; never accept a labelled confirmation file."""
    excluded, metadata = set(), {}
    for value in paths:
        path = Path(value)
        if path.name not in {"train.jsonl", "dev.jsonl", "confirmation.jsonl"}:
            raise ValueError("Prior parent exclusion accepts only public trajectory split filenames")
        raw = path.read_bytes()
        for line in raw.decode("utf-8").splitlines():
            row = json.loads(line)
            if row.get("split") != path.stem:
                raise ValueError("Prior public file/split mismatch")
            if row["split"] == "confirmation" and "labels" in row:
                raise PermissionError("Sealed confirmation labels must never be read")
            if row.get("domain") == "ledger":
                # Exclusion-only canonicalization accepts old public rename
                # probes. It does not broaden the native A-D prompt contract.
                before, rules = row["s0"], row["rules"]
                if (not isinstance(before, dict) or len(before) != 4
                        or any(not isinstance(k, str) or type(v) is not int for k, v in before.items())
                        or rules.get("semantics") != "rcwm-ledger-replay"
                        or set(rules.get("objects", [])) != set(before)):
                    raise ValueError("Invalid prior public ledger parent")
                parent = parent_world_key("ledger", before, rules)
                if parent != row["parent_world_id"]:
                    raise ValueError("Prior public parent identity mismatch")
                excluded.add(parent)
        metadata[str(path.resolve())] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return excluded, metadata


def make_ledger_record(s0, events, *, split, seed, cohort, surface="canonical", support=None, source_hashes=None):
    if split not in {"train", "dev"}:
        raise PermissionError("Ledger recovery creates TRAIN/DEV only")
    support = support or TrainingSupport()
    before = validate_balances(s0)
    if not isinstance(events, list) or not 4 <= len(events) <= 32:
        raise ValueError("Ledger recovery trajectories require 4..32 events")
    labels = ledger_rollout_labels(before, events)
    prior_matches, state = 0, before
    for event, after in zip(events, labels["states"]):
        if event_is_valid(event) and operative_key(state, event) in support.keys:
            prior_matches += 1
        state = after
    row = {"schema_version": SCHEMA_VERSION, "split": split, "domain": "ledger",
           "parent_world_id": ledger_parent_id(before), "rules": ledger_rules(surface), "s0": before,
           "actions": copy.deepcopy(events), "labels": labels,
           "provenance": {"generator": VERSION, "seed": seed, "cohort": cohort,
                          "source_sha256": source_hashes or sources(), "initial_adapter_sha256": T2_SHA,
                          "initial_adapter_config_sha256": T2_CONFIG_SHA,
                          "prior_training_support": copy.deepcopy(support.metadata),
                          "known_prior_transition_matches": prior_matches,
                          "novelty": "new generated trajectory; teacher-unseen operative inputs only when TRAIN support was verified",
                          "flags_source": "input-structure contract; no learned legality or termination claim"},
           "stress": [cohort]}
    row["id"] = hashlib.sha256(canonical({key: row[key] for key in ("domain", "rules", "s0", "actions")}).encode()).hexdigest()
    validate_ledger_record(row, replay=True)
    return row


def validate_ledger_record(row, *, replay=False):
    if row.get("schema_version") != SCHEMA_VERSION or row.get("split") not in {"train", "dev"} or row.get("domain") != "ledger":
        raise ValueError("TRAIN/DEV ledger schema required")
    validate_conditioning({key: row[key] for key in ("domain", "rules", "s0", "actions")})
    if row["parent_world_id"] != ledger_parent_id(row["s0"]):
        raise ValueError("Canonical balance parent mismatch")
    expected = {"states", "legal", "terminated", "failed", "invalid", "executed"}
    labels = row["labels"]
    if set(labels) != expected or any(not isinstance(v, list) or len(v) != len(row["actions"]) for v in labels.values()):
        raise ValueError("Invalid ledger label arrays")
    for state in labels["states"]:
        validate_balances(state)
    if any(type(v) is not bool for k in expected - {"states"} for v in labels[k]):
        raise ValueError("Contract flags must be booleans")
    if replay and labels != ledger_rollout_labels(row["s0"], row["actions"]):
        raise ValueError("Six-field ledger labels disagree with the audited executor")


def ledger_target(row, horizon):
    if row.get("split") not in {"train", "dev"}:
        raise PermissionError("Only TRAIN/DEV targets are accessible")
    if type(horizon) is not int or not 1 <= horizon <= len(row["actions"]):
        raise ValueError("Invalid horizon")
    return copy.deepcopy(validate_balances(row["labels"]["states"][horizon - 1]))


def batched_ledger_rollout(predictor, records, max_horizon=4):
    """Current predicted states only; malformed outputs stop without repair."""
    if type(max_horizon) is not int or max_horizon <= 0 or any(len(r["actions"]) < max_horizon for r in records):
        raise ValueError("Invalid rollout horizon")
    states = [copy.deepcopy(r["s0"]) for r in records]
    outputs, dead = [[] for _ in records], [False] * len(records)
    for step in range(max_horizon):
        active = [i for i in range(len(records)) if not dead[i]]
        inputs = [{"domain": "ledger", "rules": copy.deepcopy(records[i]["rules"]),
                   "s0": copy.deepcopy(states[i]), "actions": [copy.deepcopy(records[i]["actions"][step])]} for i in active]
        results = predictor.predict_batch(inputs) if active else []
        if len(results) != len(active):
            raise ValueError("Predictor batch length mismatch")
        for i, result in zip(active, results):
            outputs[i].append(result)
            try:
                if result.get("malformed"):
                    raise ValueError("Malformed output")
                states[i] = validate_balances(result["state"])
            except (ValueError, TypeError, KeyError, AttributeError):
                dead[i] = True
                outputs[i][-1] = {"malformed": True, "raw": result}
    return outputs


def build_ledger_recovery(seed=202610113, *, support=None, exclusions=()):
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative integer seed required")
    support = support or TrainingSupport()
    rng = random.Random(seed)
    used = set(exclusions) | set(support.parents)
    original_used = set(used)
    hashes = sources()
    pool = [v for v in support.balances if abs(v) <= 24] or list(range(-24, 25))
    amounts = [v for v in support.amounts if 0 < abs(v) <= 4] or [-2, -1, 1, 2]
    scale = [v for v in amounts if v != 1]
    if not scale:
        raise ValueError("A nonidentity scale amount is required")
    result = {"train": [], "dev": [], "dev_invalid": []}

    def collect(split, cohort, count, noop_words, surface="canonical", shifted=False):
        rows, attempts = [], 0
        while len(rows) < count:
            attempts += 1
            if attempts > count * 2000:
                raise RuntimeError("Fresh independent Ledger parent sampling exhausted")
            before = {key: rng.choice(pool) for key in ACCOUNTS}
            parent = ledger_parent_id(before)
            if parent in used:
                continue
            first_changed = len(rows) % 2 == 0
            statuses = ["posted" if first_changed else rng.choice(noop_words)]
            remaining = ["posted"] * (1 if first_changed else 2) + [rng.choice(noop_words)] * (2 if first_changed else 1)
            rng.shuffle(remaining); statuses += remaining
            ticks = rng.sample(range(1001, 10000) if shifted else range(1, 100), 4)
            kinds = [*KINDS, rng.choice(KINDS)]; rng.shuffle(kinds)
            events = []
            for kind, status, tick in zip(kinds, statuses, ticks):
                account = rng.choice(ACCOUNTS)
                events.append({"account": account, "amount": rng.choice(scale if kind == "scale" else amounts),
                               "kind": kind, "other": rng.choice([a for a in ACCOUNTS if a != account]),
                               "status": status, "tick": tick})
            row = make_ledger_record(before, events, split=split, seed=seed, cohort=cohort,
                                     surface=surface, support=support, source_hashes=hashes)
            states = [before, *row["labels"]["states"]]
            if (states[-1] == before or any(abs(v) > 98 for state in states for v in state.values())
                    or any(event["status"] == "posted" and states[i] == states[i + 1] for i, event in enumerate(events))
                    or row["provenance"]["known_prior_transition_matches"]):
                continue
            if any(ledger_parent_id(state) in support.parents for state in states):
                continue
            used.add(parent); rows.append(row)
        return rows, attempts

    result["train"], train_attempts = collect("train", "train", 128, ("void", "cancelled"))
    draws = {"train": train_attempts}
    for cohort, count in COHORTS.items():
        words = ("pending", "draft") if cohort == "held_status" else ("void",)
        rows, attempts = collect("dev", cohort, count, words,
                                 surface="reversed_keys" if cohort == "reordered_keys" else "canonical",
                                 shifted=cohort == "shifted_tick")
        result["dev"].extend(rows); draws[cohort] = attempts
    invalid_parents, attempts = collect("dev", "invalid_extension", 4, ("void",))
    for index, base in enumerate(invalid_parents):
        events = copy.deepcopy(base["actions"])
        if index == 0:
            events[0]["account"] = "missing"
        elif index == 1:
            events[0]["kind"] = "teleport"
        elif index == 2:
            events[0]["amount"] = 1.5
        else:
            events[0].pop("tick")
        row = make_ledger_record(base["s0"], events, split="dev", seed=seed, cohort="invalid_extension",
                                 support=support, source_hashes=hashes)
        row["provenance"]["primary"] = False
        result["dev_invalid"].append(row)
    draws["invalid_extension"] = attempts
    all_rows = [*result["train"], *result["dev"], *result["dev_invalid"]]
    parents = [row["parent_world_id"] for row in all_rows]
    if len(set(parents)) != len(parents) or original_used & set(parents):
        raise ValueError("Independent/excluded parent integrity failed")
    result.update(source_sha256=hashes, training_support=copy.deepcopy(support.metadata),
                  attempts=draws, parent_registry={row["parent_world_id"]: row["split"] for row in all_rows},
                  integrity={"one_row_per_canonical_parent": True, "cross_split_parent_disjoint": True,
                             "supplied_prior_parent_disjoint": True, "excluded_parent_count": len(original_used),
                             "known_prior_operative_keys_excluded": support.metadata.get("available", False)},
                  numeric_pool={"balances": pool, "amounts": amounts,
                                "original_marginal_membership_verified": support.metadata.get("available", False)})
    return result


def write_ledger_recovery(output, *, seed=202610113, prior_train=T2_TRAIN, prior_public=()):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Ledger recovery output already exists")
    support = load_training_support(prior_train)
    exclusions, prior_sources = prior_public_parents(prior_public)
    data = build_ledger_recovery(seed, support=support, exclusions=exclusions)
    contents = {f"{split}.jsonl": ("".join(canonical(row) + "\n" for row in data[split])).encode()
                for split in ("train", "dev", "dev_invalid")}
    contents["parent_registry.json"] = (canonical(data["parent_registry"]) + "\n").encode()
    manifest = {"generator": VERSION, "seed": seed, "status": "prospective TRAIN/DEV inputs; no native result",
                "counts": {split: len(data[split]) for split in ("train", "dev", "dev_invalid")},
                "confirmation_rows": 0, "confirmation_labels_accessed": False,
                "initializer": {"adapter": str(T2_ADAPTER), "weight_sha256": T2_SHA, "config_sha256": T2_CONFIG_SHA},
                "source_sha256": data["source_sha256"], "training_support": data["training_support"],
                "prior_public_sources": prior_sources, "integrity": data["integrity"], "attempts": data["attempts"],
                "prior_public_exclusion_scope": "supplied public files only" if prior_sources else "not checked; no prior-public files supplied",
                "numeric_pool": data["numeric_pool"], "dev_cohorts": COHORTS,
                "training_status_counts": dict(Counter(e["status"] for r in data["train"] for e in r["actions"])),
                "training_kind_counts": dict(Counter(e["kind"] for r in data["train"] for e in r["actions"])),
                "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} for name, raw in contents.items()},
                "limitations": ["Bare A-D balances only; renamed account extension is unsupported.",
                    "Structural flags are contract-derived, not learned legality/termination predictions.",
                    "Tick-compatible labels use the new collection adapter, never frozen domains.exact_transition.",
                    "TRAIN expands nonposted words beyond original void; every arm must share this data-design change.",
                    "Stress cohorts contain different parents; cross-cohort effects are descriptive.",
                    "Historic evaluation overlap and parent-model exposure are unknown; no sealed bank was inspected.",
                    "Independent confirmation, behavioral retention and native efficacy remain unmeasured."]}
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in contents.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest
