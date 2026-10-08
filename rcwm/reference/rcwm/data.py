"""Stable parent grouping and strict open-loop record validation."""
from __future__ import annotations
from hashlib import sha256
import math


def split_for_parent(parent_id: str, seed: int = 0) -> str:
    if not isinstance(parent_id, str) or not parent_id:
        raise ValueError("nonempty parent ID required")
    digest = sha256(f"{seed}:{parent_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if bucket < .7 else "dev" if bucket < .85 else "test"


def validate_open_loop(record: dict) -> None:
    required = ("parent_id", "split", "plan_origin", "decision_time",
                "observations", "actions")
    if any(k not in record for k in required):
        raise ValueError("missing required record field")
    if record["plan_origin"] != "open_loop_precommitted":
        raise ValueError("behavior-selected future suffix is not an open-loop plan")
    if record["split"] not in {"train", "dev", "test"}:
        raise ValueError("invalid split")
    if not record["parent_id"] or not record["actions"]:
        raise ValueError("parent and actions required")
    t = record["decision_time"]
    if not isinstance(t, (int, float)) or not math.isfinite(t):
        raise ValueError("finite decision time required")
    for o in record["observations"]:
        if o.get("kind") != "observed" or not math.isfinite(o.get("available_at", math.inf)):
            raise ValueError("only time-stamped observations belong in observed prefix")
        if o["available_at"] > t:
            raise ValueError("future observation leak")
    for a in record["actions"]:
        if not isinstance(a.get("operator"), str) or not a["operator"]:
            raise ValueError("operator required")
        if not isinstance(a.get("executor_version"), str) or not a["executor_version"]:
            raise ValueError("executor version required")
        if "observed_outcome" in a or "observed_reward" in a:
            raise ValueError("future labels may not be embedded in candidate actions")
