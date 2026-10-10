#!/opt/anaconda3/bin/python3
"""Freeze before first solver/test import; execute own public-only calibration.

Only stdlib is imported before source/fixture/config/public byte validation.
No producer/formal/private review file or production candidate is consumed.
"""
import datetime
import hashlib
import importlib.util
import json
import pathlib
import resource
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ROLE = "/root/expert_reviewer"
PUBLIC = {
    "schemas/RSM_OPTION_RESUMPTION_V1.md": "44652656ad18e60c98c96db14e0a96b4e46b0dad0bd57f257bca30384ab53b3f",
    "research/curricula/RSM-OPTION-RESUMPTION-PROSPECTIVE-20261010.md": "8bda187009cc91756248ddd86792c28aabe0793a3f6be304bb9433bfdf48be34",
    "research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json": "422f67c836639d0accbf411c3080ef4fc7695ad8630120a140488c90a19061fa",
    "schemas/RSM_OPTION_RESUMPTION_METADATA_V1.md": "1612b14d7549b7fb406cbef8e127440dcfddaf30548ef29d217bc49aefb6c854",
    "schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R1.md": "5f7f6a2c45dd6a0b51c7a912b28fd49badd9aa3aa2eac8632e9b9781c201b57b",
    "schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md": "b62663c3082c9e2ee2bf34aadc1fdd10f263a6497396b6768f1475b12f568fe4",
    "research/curricula/RSM_OPTION_METADATA_V1_AMENDMENT_R2_ROOT_FREEZE.json": "3d82da83535c6e8c0a0884d870945d064605e1f58ae2072153c5da8c8d8daeaa",
}
LOCAL = ["solver-attempt01.py", "tests-attempt01.py", "run-preparation-attempt01.py",
         "hand-calibrations-attempt01.json", "config-attempt01.json"]


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def hp(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "length_bytes": size}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canon(value) + "\n")
    return hp(path)


def bindings():
    result = {}
    for path, expected in PUBLIC.items():
        result[path] = hp(ROOT / path)
        assert result[path]["sha256"] == expected, ("public source changed", path)
    for name in LOCAL:
        path = HERE / name
        result[str(path.relative_to(ROOT))] = hp(path)
    return result


def freeze():
    frozen = bindings()
    archive_dir = HERE / "source-archive-attempt01"
    archive_dir.mkdir(exist_ok=False)
    archived = {}
    for path, record in frozen.items():
        original = ROOT / path
        archive = archive_dir / (record["sha256"] + ".blob")
        with original.open("rb") as source, archive.open("xb") as destination:
            for block in iter(lambda: source.read(1048576), b""):
                destination.write(block)
        assert hp(archive) == record
        archived[path] = {"path": str(archive.relative_to(ROOT)), **record}
    config = json.loads((HERE / "config-attempt01.json").read_text())
    receipt = {
        "report_type": "ACTUAL_EXPERT_RSM_PUBLIC_PREPARATION_PREEXECUTION_FREEZE",
        "actual_author": ROLE, "reviewer_execution_id": ROLE, "created_utc": now(),
        "bindings_before_first_solver_test_import_execution": frozen,
        "exact_archives": archived, "configuration": config,
        "python_binary": hp(pathlib.Path(sys.executable).resolve()),
        "python_executable": str(pathlib.Path(sys.executable).resolve()),
        "python_version": sys.version, "platform": sys.platform,
        "only_stdlib_imported_before_this_freeze": True,
        "no_candidate_or_other_private_implementation_access": True,
        "no_corpus_checker_import_or_novelty_check": True,
        "target_free_legacy_view_read": False,
        "prospective_only_no_acceptance_authority": True,
    }
    output = HERE / "source-freeze-attempt01.json"
    print(canon({"freeze": save(output, receipt), "own_sources": {name: hp(HERE / name) for name in LOCAL}}))


def own_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def execute():
    start = now()
    timer = time.monotonic()
    prepath = HERE / "source-freeze-attempt01.json"
    pre = json.loads(prepath.read_text())
    before = bindings()
    assert before == pre["bindings_before_first_solver_test_import_execution"]
    assert hp(pathlib.Path(sys.executable).resolve()) == pre["python_binary"]
    for path, record in pre["exact_archives"].items():
        assert hp(ROOT / record["path"]) == {key: record[key] for key in ("sha256", "length_bytes")}
    config = json.loads((HERE / "config-attempt01.json").read_text())
    fixtures = json.loads((HERE / "hand-calibrations-attempt01.json").read_text())
    # First import occurs here, after source/public/config/fixtures/archive checks.
    solver = own_module("expert_rsm_solver_public_preparation_attempt01", HERE / "solver-attempt01.py")
    tests = own_module("expert_rsm_tests_public_preparation_attempt01", HERE / "tests-attempt01.py")
    assert solver.VERSION == config["solver_version"]
    contract = (ROOT / "schemas/RSM_OPTION_RESUMPTION_V1.md").read_text()
    results = tests.run(solver, fixtures, contract)
    results.update({"actual_author": ROLE, "reviewer_execution_id": ROLE,
                    "stage": "PROSPECTIVE_HANDMADE_CALIBRATION_ONLY", "exposure": config["exposure"]})
    output = HERE / "calibration-results-attempt01.json"
    save(output, results)
    after = bindings()
    assert before == after
    for path, record in pre["exact_archives"].items():
        assert hp(ROOT / record["path"]) == {key: record[key] for key in ("sha256", "length_bytes")}
    report = {
        "report_type": "ACTUAL_EXPERT_RSM_PUBLIC_PREPARATION_EXECUTION",
        "actual_author": ROLE, "reviewer_execution_id": ROLE, "start_utc": start,
        "finish_utc": now(), "elapsed_seconds": round(time.monotonic() - timer, 6),
        "python_executable": sys.executable, "python_version": sys.version,
        "source_freeze": hp(prepath), "bindings_before": before, "bindings_after": after,
        "calibration_results": hp(output), "actual_passed_checks": results["passed"],
        "actual_failed_checks": results["failed"],
        "handmade_full_target_cases": results["handmade_full_target_cases"],
        "manual_event_cuts": results["manual_event_cuts"],
        "resume_all_cut_budget_checks": results["all_cut_budget_resume_checks"],
        "separate_recursive_restart_crosschecks": results["restart_recursive_crosschecks"],
        "typed_malformed_stage_checks": results["malformed_stage_checks"],
        "peak_native_ru_maxrss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "exit_code": 0, "exposure": config["exposure"],
        "implementation_scope": "Own new typed stack solver/parser plus own separate recursive complete-run simulator;same-author/common-model public-guided evidence",
        "no_candidate_or_legacy_gold_read": True, "no_other_review_or_formal_results_read": True,
        "pending_future_gates": config["future_scope"], "publication_or_data_approval": False,
    }
    execution = HERE / "execution-attempt01.json"
    save(execution, report)
    final = {
        "report_type": "ACTUAL_EXPERT_RSM_PUBLIC_PREPARATION_COMPLETED_FINAL",
        "actual_author": ROLE, "reviewer_execution_id": ROLE, "completed_utc": now(),
        "status": "PROSPECTIVE_PREPARATION_COMPLETE_NO_DATA_JUDGMENT",
        "files": {str(path.relative_to(ROOT)): hp(path) for path in
                  [HERE / name for name in LOCAL] + [prepath, output, execution]},
        "actual_passed_checks": results["passed"], "actual_failed_checks": results["failed"],
        "handmade_cases": results["handmade_full_target_cases"],
        "first_execution_failures": [], "exposure": config["exposure"],
        "future_predictions_require_separate_input_view_authorization": True,
        "no_new_candidate_inspected_or_approved": True, "publication_or_data_approval": False,
    }
    finalpath = HERE / "FINAL-attempt01.json"
    save(finalpath, final)
    print(canon({"FINAL": hp(finalpath), "execution": hp(execution), "calibration": hp(output),
                 "freeze": hp(prepath), "passed": results["passed"], "failed": results["failed"]}))


if __name__ == "__main__":
    stage = sys.argv[1]
    try:
        {"freeze": freeze, "execute": execute}[stage]()
    except BaseException as error:
        failure = {
            "actual_author": ROLE, "reviewer_execution_id": ROLE, "created_utc": now(),
            "stage": stage, "source": hp(pathlib.Path(__file__).resolve()),
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "actual_completed_checks": getattr(error, "completed_checks", []),
            "failure_evidence": getattr(error, "evidence", None),
            "publication_or_data_approval": False,
        }
        path = HERE / ("failure-" + stage + "-attempt01.json")
        if not path.exists():
            save(path, failure)
        raise
