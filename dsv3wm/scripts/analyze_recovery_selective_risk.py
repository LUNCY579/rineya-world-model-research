"""Posthoc malformed-only deferral operating point; no probabilities or models.

Uses completed frozen DEV outcomes only. It neither changes generation nor
claims calibrated confidence. A valid reply may still be a wrong state.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def operating_point(rows, *, mode, horizon=4):
    from dsv3wm.stats import wilson_interval
    if mode not in {"direct", "rollout"} or not rows:
        raise ValueError("nonempty direct/rollout parent outcomes required")
    correct_key, malformed_key = mode + "_correct", mode + "_malformed"
    flags = [(row["horizons"][str(horizon)][correct_key], row["horizons"][str(horizon)][malformed_key]) for row in rows]
    if all(correct is None and malformed is None for correct, malformed in flags):
        return {"status": "not-measured", "n": len(rows)}
    if any(type(correct) is not bool or type(malformed) is not bool or (correct and malformed)
           for correct, malformed in flags):
        raise ValueError("complete typed binary outcomes required; malformed cannot be correct")
    rejected = sum(malformed for _, malformed in flags)
    accepted = len(flags) - rejected
    accepted_errors = sum(not correct and not malformed for correct, malformed in flags)
    total_errors = sum(not correct for correct, _ in flags)
    return {"status": "measured", "n": len(flags), "accepted": accepted, "rejected": rejected,
            "accepted_correct": accepted - accepted_errors, "accepted_errors": accepted_errors,
            "all_case_errors": total_errors, "coverage": accepted / len(flags),
            "accepted_error_risk": accepted_errors / accepted if accepted else None,
            "accepted_error_risk_wilson95": list(wilson_interval(accepted_errors, accepted)) if accepted else None,
            "detected_errors": rejected,
            "error_detection_recall": rejected / total_errors if total_errors else None,
            "rejection_rule": "reject only frozen malformed reply; no new confidence threshold",
            "probability_calibration": "unmeasured; no predictive probabilities/logits captured"}


def analyze(records):
    if not records:
        raise ValueError("completed records required")
    reference, output = None, {}
    for arm, record in records.items():
        rows = record.get("per_parent")
        if record.get("status") != "development-evaluated" or not isinstance(rows, list) or len(rows) != 24:
            raise ValueError("completed fixed 24-parent DEV record required")
        identities = {(row["parent_world_id"], row["cohort"]) for row in rows}
        if len(identities) != 24 or len({row["parent_world_id"] for row in rows}) != 24:
            raise ValueError("duplicate parent units cannot be pooled")
        identity = (record["data_sha256"], identities)
        if reference is not None and identity != reference:
            raise ValueError("matched fixed DEV data/parent/cohort identity required")
        reference = identity
        output[arm] = {mode: {cohort: operating_point(
            [row for row in rows if cohort == "all" or row["cohort"] == cohort], mode=mode)
            for cohort in ("id", "ood", "all")} for mode in ("rollout", "direct")}
    return output


def main():
    from dsv3wm.native import sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/recovery-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("selective-risk supplement exists; use a new path")
    paths = {arm: args.results / ("B0_qualification.json" if arm == "B0" else arm + "_dynamics.json")
             for arm in ("B0", "A0", "A1", "A2", "A5")}
    records = {arm: json.loads(path.read_text(encoding="utf-8")) for arm, path in paths.items()}
    result = {"analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "posthoc fixed DEV seed0; structural deferral, not calibrated uncertainty or confirmation",
              "native_model_loaded": False, "confirmation_access": False, "horizon": 4,
              "input_sha256": {str(path.resolve()): sha256(path) for path in paths.values()},
              "source_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                                 str(ROOT / "src/dsv3wm/stats.py"): sha256(ROOT / "src/dsv3wm/stats.py")},
              "arms": analyze(records),
              "uncertainty_scope": "Wilson intervals describe case outcomes only; one seed and shared rule families do not establish seed or broad-world uncertainty"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
