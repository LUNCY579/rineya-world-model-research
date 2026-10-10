#!/opt/anaconda3/bin/python3
"""New prospective inspection closure; retain original machine FINAL unchanged."""
import datetime
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROLE = "/root/expert_reviewer"
BIND = {
    "solver-attempt01.py": "aefd06ae9de6d661fb7d7e61e0d9721fdac090d69064a6a79fec06d7ac081967",
    "tests-attempt01.py": "e034e30d73264f48dd3e40d9c8cbfeeafa3c9acf8e4347514f8227b1d433cf09",
    "hand-calibrations-attempt01.json": "c7638941ba2c1fae6c93e0cf0543f6f68f4c0fe0ddab5eacd65b098522a9ac9d",
    "config-attempt01.json": "c82ac516dc64e04d9b788c34fdd0d5825458941c488257bdaf256e640cb6d5cd",
    "run-preparation-attempt01.py": "86bc13798ebd36d18c4c5e6d4a4c85546b66325bbb9a809966b7e34f60d1ebe6",
    "source-freeze-attempt01.json": "8eecb0c8a15249f9a99169f43c07be587bf39a71d9699a6207d14d0e873c5a7e",
    "calibration-results-attempt01.json": "5a2ccd85045a15188b69d47a0d964a597f7e0fd610177c3b5a37cac334c2c454",
    "execution-attempt01.json": "abf7a90c2bf755dbb1c605d4d88188e78cca4ef071be519122af6ab205fa48ad",
    "FINAL-attempt01.json": "404fb616d3e46000ddc2477c9786716cba52716e9a592cb763126db88f74d99d",
}


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hp(path):
    content = path.read_bytes()
    return {"sha256": hashlib.sha256(content).hexdigest(), "length_bytes": len(content)}


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canon(value) + "\n")
    return hp(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bindings():
    result = {name: hp(HERE / name) for name in BIND}
    assert all(result[name]["sha256"] == digest for name, digest in BIND.items())
    result.update({name: hp(HERE / name) for name in ("SEMANTIC-INSPECTION.md", "PROSPECTIVE-PROCEDURE.md")})
    return result


def freeze():
    source = pathlib.Path(__file__).resolve()
    archive = HERE / "finalize-source-attempt01.archive.py"
    with archive.open("xb") as stream:
        stream.write(source.read_bytes())
    receipt = {"actual_author": ROLE, "reviewer_execution_id": ROLE, "created_utc": now(),
               "source": hp(source), "archive": hp(archive), "bindings_before": bindings(),
               "scope": "Finalize actual prospective public-only preparation/inspection; no solver or test import or additional execution"}
    print(canon({"source": hp(source), "freeze": save(HERE / "finalization-freeze-attempt01.json", receipt)}))


def run():
    prepath = HERE / "finalization-freeze-attempt01.json"
    pre = json.loads(prepath.read_text())
    source = pathlib.Path(__file__).resolve()
    before = bindings()
    assert before == pre["bindings_before"]
    assert hp(source) == pre["source"] == hp(HERE / "finalize-source-attempt01.archive.py")
    original = json.loads((HERE / "execution-attempt01.json").read_text())
    calibration = json.loads((HERE / "calibration-results-attempt01.json").read_text())
    assert original["exit_code"] == 0 and original["actual_passed_checks"] == calibration["passed"] == 290
    assert original["actual_failed_checks"] == calibration["failed"] == 0
    assert original["bindings_before"] == original["bindings_after"]
    assert not list(HERE.glob("failure-*.json"))
    after = bindings()
    assert before == after
    execution = {"actual_author": ROLE, "reviewer_execution_id": ROLE, "completed_utc": now(),
                 "source_before_after": hp(source), "freeze": hp(prepath), "bindings_before": before,
                 "bindings_after": after, "actual_calibration_rerun": False,
                 "previous_actual_execution_verified": True, "exit_code": 0, "data_approval": False}
    ep = HERE / "finalization-execution-attempt01.json"
    save(ep, execution)
    final = {"report_type": "ACTUAL_EXPERT_RSM_PUBLIC_PREPARATION_FINAL",
             "actual_author": ROLE, "reviewer_execution_id": ROLE, "completed_utc": now(),
             "status": "PUBLIC_ONLY_PROSPECTIVE_PREPARATION_COMPLETE",
             "source": before["solver-attempt01.py"], "tests": before["tests-attempt01.py"],
             "handmade_calibrations": before["hand-calibrations-attempt01.json"],
             "config": before["config-attempt01.json"], "pre_import_source_freeze": before["source-freeze-attempt01.json"],
             "actual_execution": before["execution-attempt01.json"], "actual_calibration_results": before["calibration-results-attempt01.json"],
             "actual_passed_checks": 290, "actual_failed_checks": 0, "actual_first_failures": [],
             "handmade_cases": 19, "manual_event_cuts": 10, "resume_cut_budget_checks": 120,
             "separate_recursive_restart_checks": 10, "typed_malformed_stage_checks": 29,
             "actual_inspection": before["SEMANTIC-INSPECTION.md"], "future_procedure": before["PROSPECTIVE-PROCEDURE.md"],
             "finalization_source": hp(source), "finalization_freeze": hp(prepath), "finalization_execution": hp(ep),
             "all_sources_and_outputs_frozen": True, "original_machine_FINAL_unchanged": before["FINAL-attempt01.json"],
             "exposure": original["exposure"], "legacy_target_free_view_read": False,
             "no_candidate_read_or_approved": True, "no_corpus_checker_or_novelty_approval": True,
             "future_216_input_predictions_first_judgments_require_root_authorization_before_gold": True,
             "data_generation_or_publication_authority": False,
             "scope": "Handmade finite supplied-program execution/parser feasibility only;12 prospective novel families/108tasks/216views are not measured production coverage"}
    finalpath = HERE / "PREPARATION-FINAL.json"
    print(canon({"FINAL": save(finalpath, final), "closure_execution": hp(ep), "all_files_frozen": True}))


if __name__ == "__main__":
    {"freeze": freeze, "run": run}[sys.argv[1]]()
