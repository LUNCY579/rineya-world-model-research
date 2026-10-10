"""Producer regression checks for uniqueness and public finite edge cases.

Run with /opt/anaconda3/bin/python3 -m generators.other_tracks.regression.
Independent oracle validation and approval are outside this module.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from factory.common import canonical, digest, load_jsonl, record_hash
from generators.other_tracks import generate as producer
from generators.other_tracks.reference import solve


def semantic_content(value):
    if isinstance(value, dict):
        return {key: semantic_content(item) for key, item in value.items() if key not in ("annotation", "annotations")}
    if isinstance(value, list):
        return [semantic_content(item) for item in value]
    return value


def check_rows(rows):
    by_track = defaultdict(list)
    for row in rows:
        by_track[row["track_id"]].append(row)
        assert row["record_hash"] == record_hash(row)
        assert row["review_status"] == "PENDING"
        assert 1 <= row["structural_difficulty"] == row["difficulty"] <= 4
        prefix, payload = row["rendered_input"].split("Task JSON:\n")
        assert json.loads(payload) == row["semantic_task"]
        assert row["semantic_parent_id"] not in prefix and row["sample_id"] not in prefix
        assert "Scalar equality is type-sensitive" in prefix
    for track, records in by_track.items():
        groups = defaultdict(list)
        for row in records:
            groups[row["semantic_parent_id"]].append(row)
        hashes, prompts = [], []
        for pair in groups.values():
            assert len(pair) == 2
            a, b = pair
            assert a["verified_target"] == b["verified_target"]
            assert a["semantic_task"] == b["semantic_task"]
            assert a["split"] == b["split"]
            assert a["rendered_input"] != b["rendered_input"]
            hashes.append(digest(semantic_content(a["semantic_task"])))
            prompts.extend([a["rendered_input"], b["rendered_input"]])
        assert len(hashes) == len(set(hashes)), f"Duplicate semantic parents: {track}"
        assert len(prompts) == len(set(prompts)), f"Duplicate prompts: {track}"
        print(track, "parents", len(hashes), "unique semantic hashes", len(set(hashes)))


def edge_cases():
    task = dict(kind="skills", state=dict(x=1, y=2), procedures={
        "sum": [dict(op="add", register="x", value=3)],
        "cond": [{"op": "if", "test": dict(register="x", cmp="ge", value=4),
                  "then": [dict(op="call", name="sum")], "else": [dict(op="scale", register="y", value=3)]}]},
                main=[dict(op="call", name="sum"), dict(op="call", name="cond"), dict(op="swap")])
    assert solve(task) == dict(after=dict(x=2, y=7), executed_primitives=3, call_order=["sum", "cond", "sum"], conditions=[True])
    task = dict(kind="belief", worlds=[dict(world_id="integer", facts={"v": 1}), dict(world_id="boolean", facts={"v": True})],
                observations=[dict(fact="v", equals=1)], queries=[dict(query_id="q", fact="v")])
    assert solve(task)["surviving_worlds"] == ["integer"]
    task["observations"].append(dict(fact="v", equals=True))
    output = solve(task)
    assert output["surviving_worlds"] == [] and not output["queries"][0]["entailed"]
    task = dict(kind="episodic", events=[dict(key="k", version=2, at=2, op="put", value=None),
                                            dict(key="k", version=2, at=3, op="delete", value=None),
                                            dict(key="k", version=1, at=10, op="put", value="older")],
                queries=[dict(query_id="q0", key="k", as_of=2), dict(query_id="q1", key="k", as_of=20)])
    assert [r["status"] for r in solve(task)["results"]] == ["present", "deleted"]
    task = dict(kind="predictive_abstraction", states=["a", "b", "c"], actions=["step"],
                observable={"a": 0, "b": 0, "c": 1},
                transition={"a": {"step": "b"}, "b": {"step": "c"}, "c": {"step": "c"}})
    assert solve(task)["classes"] == [["a"], ["b"], ["c"]]
    task = dict(kind="constraint_repair", alphabet=["a", "b"], slots=2,
                constraints=[dict(kind="allowed", position=0, values=[])],
                candidates=[dict(candidate_id="c", assignment=["a", "b"])], min_distance=1)
    assert solve(task) == dict(feasible_count=0, repairs=[dict(candidate_id="c", original_valid=False, edit_distance=None, repaired=None)], diverse_selection=[])


def main():
    edge_cases()
    with tempfile.TemporaryDirectory() as directory, patch.object(producer, "ROOT", Path(directory)):
        rows = producer.generate(batch="regression-v2")
        check_rows(rows)
        assert rows == producer.generate(batch="regression-v2")
        try:
            producer.generate(count=13, batch="regression-v2")
        except ValueError:
            pass
        else:
            raise AssertionError("Existing batch was not protected")
    print("Producer finite edge cases, uniqueness, field-order equivalence and immutable-batch checks passed")


if __name__ == "__main__":
    main()
