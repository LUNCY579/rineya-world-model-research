#!/opt/anaconda3/bin/python3
"""Small final readback of expert-owned input-stage outputs; no canonical IO."""
import datetime
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "reviews/expert"
ROLE = "/root/expert_reviewer"
NAME = "belief-gap-v4-input-attempt01"
FINALIZER = "belief-gap-v4-input-finalize-attempt01"
BOUND = {
    NAME + ".predictions.jsonl": "d9a55125b127edc6ab691f4981ea61edfa60b03b333e90ad4fc7c865b1ccc9ed",
    NAME + ".first-input-judgments.jsonl": "5a0ba9a30c7fbec74c8c69322ba592941046e5b98d375f36bda2118343fe7815",
    NAME + ".analysis.json": "202cdc93647b3783eae775bd6b28f88bc49254292787bda1441c384ae18aaede",
    NAME + ".execution.json": "7e6cba6b28a926e5ee8782c57783420e957868008817cd426bfeed35eba5aced",
    NAME + ".preexecution.json": "c0f935707c0cf0f3195e69c76af6ac0c3003f9606d6ff65c1245d5f3d14c6bb1",
    "belief-gap-v4-input-review-attempt01.py": "9f16f085aca910718e83dae721bb77ff42736177160b1b6da69b502e7b102db1",
}
EXTRA = [NAME + ".source.preexecution.py", NAME + ".freeze.json", NAME + ".handmade-calibration.json", NAME + ".semantic-inspection.md"]


def canon(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v):
    return hashlib.sha256(canon(v).encode()).hexdigest()


def hash_path(p):
    h = hashlib.sha256()
    size = 0
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "length_bytes": size}


def write(path, v):
    with path.open("x", encoding="utf-8") as f:
        f.write(canon(v) + "\n")


def binding():
    out = {n: hash_path(OUT / n) for n in list(BOUND) + EXTRA}
    assert all(out[n]["sha256"] == h for n, h in BOUND.items())
    return out


def main():
    source = pathlib.Path(__file__).resolve()
    pre = OUT / (FINALIZER + ".preexecution.json")
    if sys.argv[1] == "freeze":
        with (OUT / (FINALIZER + ".source.preexecution.py")).open("xb") as f:
            f.write(source.read_bytes())
        write(pre, {"actual_author": ROLE, "reviewer_execution_id": ROLE, "source": hash_path(source), "before": binding(),
                    "purpose": "read back actual own frozen all-record inputs outputs and freeze agent inspection; no new target or data judgment"})
        print(canon(hash_path(pre)))
        return
    assert sys.argv[1] == "execute"
    p = json.loads(pre.read_text())
    assert p["source"] == hash_path(source) == hash_path(OUT / (FINALIZER + ".source.preexecution.py"))
    before = binding()
    assert before == p["before"]
    ids = set()
    count = 0
    with (OUT / (NAME + ".predictions.jsonl")).open() as predictions, (OUT / (NAME + ".first-input-judgments.jsonl")).open() as judgments:
        for pline, jline in zip(predictions, judgments, strict=True):
            pred, jud = json.loads(pline), json.loads(jline)
            assert pred["sample_id"] == jud["sample_id"] and pred["record_hash"] == jud["record_hash"]
            assert pred["identity"] == jud["identity"]
            assert pred["sample_id"] not in ids
            ids.add(pred["sample_id"])
            assert pred["actual_author"] == pred["reviewer_execution_id"] == jud["reviewer_execution_id"] == ROLE
            assert pred["canonical_target_not_read"] is True and pred["dataset_approval"] is False
            assert jud["decision"] == "INPUT_CLEAR_PENDING_CANONICAL_COMPARISON" and jud["scores"]["target_correctness"] is None
            assert digest(pred["expected_target"]) == jud["evidence"]["fresh_complete_target_sha256"]
            assert digest(pred["prefix_survivors"]) == jud["evidence"]["fresh_all_prefixes_sha256"]
            assert pred["expected_target"]["belief_sizes"] == [len(x) for x in pred["prefix_survivors"][1:]]
            assert pred["expected_target"]["surviving_worlds"] == pred["prefix_survivors"][-1]
            assert all(type(q["entailed"]) is bool and type(q["can_reduce_uncertainty"]) is bool for q in pred["expected_target"]["queries"])
            count += 1
    assert count == 3456 and before == binding()
    analysis = json.loads((OUT / (NAME + ".analysis.json")).read_text())
    execution = json.loads((OUT / (NAME + ".execution.json")).read_text())
    final = {"report_type": "EXPERT_COMPLETED_INPUT_ONLY_FINAL", "actual_author": ROLE, "reviewer_execution_id": ROLE,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "records": count,
        "source": hash_path(source), "finalizer_preexecution": hash_path(pre), "all_artifacts_before_after": before,
        "original_execution_exit_code": execution["exit_code"], "readback_checks_passed": True,
        "complete_fresh_typed_target_and_prefix_recomputation": count, "full_old_new_task_raw_prompt_split_track_joins": count,
        "first_input_judgments_completed": count, "input_findings": "clear determinate input tasks, with same-case finite-template and majority/shortcut limitations",
        "metadata_gold_record_hash_control_review": "PENDING; canonical access has not occurred", "no_dataset_approval": True,
        "stage_stop": "No dependent canonical comparison until a separate root authorization",
        "exposure": execution["exposure"], "compact_aggregates": {k: v for k, v in analysis["aggregates"].items() if k != "query_id_position_fact_name_conditional_histograms"},
        "storage": "No raw new output above 25MiB; previous raw evidence preserved unchanged; no compressed pointer required"}
    final_path = OUT / (NAME + ".FINAL.json")
    write(final_path, final)
    print(canon({"final": hash_path(final_path), "inspection": before[NAME + ".semantic-inspection.md"], "rows": count}))


if __name__ == "__main__":
    main()
