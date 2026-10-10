"""Isolated join rejection tests; synthetic remote evidence stays in memory only.

These tests never execute main, call the network or certify a remote payload.
Actual root stage and ledger fixtures are immutable metadata, without candidate rows.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
LOGS = Path(__file__).resolve().parent


class R2JoinProofGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        freeze = json.loads((LOGS / "BELIEF_V4_R2_JOIN_SOURCE_FREEZE.json").read_bytes())
        path = LOGS / "join_belief_v4_r2_completed_backup_custody.py"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == freeze["source"]["sha256"]
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == freeze["tests"]["sha256"]
        for row in freeze["frozen_inputs"]:
            assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
        sp = importlib.util.spec_from_file_location("frozen_r2_join_tests", path)
        cls.module = importlib.util.module_from_spec(sp);sp.loader.exec_module(cls.module)
        cls.module.RESUMPTION = cls.module.import_frozen_resumption()
        cls.spec = json.loads((LOGS / "belief-v4-validated-code-candidate-r2_SPEC.json").read_bytes())
        cls.stage = json.loads((LOGS / "belief-v4-validated-code-candidate-r2_RESUMED_STAGE.json").read_bytes())
        cls.local = json.loads((LOGS / "belief-v4-validated-code-candidate-r2_RESUMED_LOCAL_COMMIT.json").read_bytes())
        cls.stage_lines = [json.loads(line) for line in (LOGS / "belief-v4-validated-code-candidate-r2_RESUMED_STAGE_LEDGER.jsonl").read_text().splitlines()]
        cls.blobs = [{k: row[k] for k in ("git_blob", "paths", "sha256", "size")}
                     for row in cls.stage_lines if row["phase"] == "actual_index_blob_reverified"]
        cls.tree = "b" * 40
        cls.full_sha = "a" * 64

    def assert_stage(self, stage):
        m = self.module
        m.validate_stage(self.spec, stage, m.COMMIT, m.PARENT, m.SPEC_SHA)

    def synthetic_remote_fixture_DO_NOT_PUBLISH(self):
        m = self.module
        blobs = [{**copy.deepcopy(row), "verification_source": "actual_remote_raw_GitHub_blob"}
                 for row in self.blobs]
        coverage = {name: row for row in blobs for name in row["paths"]}
        full = {
            "status": "BACKED_UP", "commit_sha": m.COMMIT, "remote_commit_sha": m.COMMIT,
            "spec_sha256": m.SPEC_SHA, "inventory": self.spec["files"], "files": 2685,
            "total_bytes": 1080978676, "verification_source_sha256": m.DEPENDENCIES["backup_logs/recover_remote_checkpoint.py"],
            "runner_execution": "/root", "repository": "LUNCY579/rineya-world-model-research",
            "branch": "research/rineya-high-quality-dataset-v1", "local_git_verified_unique_blobs": 2153,
            "current_source_matching_files": 2685,
            "prior_evidence": [{"path": str(LOGS / Path(m.STAGE).name), "sha256": m.STAGE_SHA}],
            "private_repository_evidence": {"isPrivate": True, "viewerPermission": "ADMIN", "nameWithOwner": "LUNCY579/rineya-world-model-research"},
            "current_source_evolution": [], "current_source_evolution_at_finish": [],
            "remote_verification": {
                "verified_files": 2685, "verified_unique_blobs": 2153, "blob_receipts": blobs,
                "remote_tree_sha": self.tree, "http_accept": "application/vnd.github.raw+json",
                "cache_policy": "no gh --cache; Cache-Control: no-cache; every blob fetched this run",
            },
            "oversized_raw_local_verification": [
                {"path": e["path"], "size": e["size"], "sha256": e["sha256"], "lossless_local_roundtrip": True}
                for e in self.spec["raw_size_exclusions"]
            ],
        }
        proofs = []
        for raw in self.spec["raw_size_exclusions"]:
            packed = raw["lossless_representations"][0]["gzip"]
            blob = coverage[packed["path"]]
            proofs.append({"raw_path": raw["path"], "raw_bytes": raw["size"], "raw_sha256": raw["sha256"],
                "remote_gzip_path": packed["path"], "remote_gzip_bytes": packed["size"], "remote_gzip_sha256": packed["sha256"],
                "remote_git_blob": blob["git_blob"], "actual_remote_fetch_this_run": True,
                "decoded_byte_for_byte_equal_to_preserved_raw": True})
        gzip = {
            "status": "PASS_ACTUAL_REMOTE_GZIP_DECODE", "commit_sha": m.COMMIT, "spec_sha256": m.SPEC_SHA,
            "stage_receipt_sha256": m.STAGE_SHA, "verified_inventory_receipt_sha256": self.full_sha,
            "source_sha256": m.DEPENDENCIES["backup_logs/verify_archival_remote_lossless.py"],
            "source_freeze_sha256": m.DEPENDENCIES["backup_logs/ARCHIVAL_REMOTE_LOSSLESS_SOURCE_FREEZE.json"],
            "dependency_sha256": m.DEPENDENCIES["backup_logs/recover_remote_checkpoint.py"],
            "actual_runner_execution": "/root/backup_manager", "grants_data_approval": False,
            "private_admin_confirmed": True, "actual_remote_tree_sha": self.tree,
            "stream_chunk_bytes": 65536, "bounded_spool_memory_bytes": 8388608,
            "actual_remote_gzip_proofs": proofs, "actual_remote_gzip_fetches": 26,
            "oversized_raw_artifacts": 26, "raw_bytes_exactly_recovered": 1353229114,
            "actual_decoded_bytes_including_multiple_representations": 1353229114,
        }
        return full, gzip

    def validate(self, full, gzip):
        m = self.module
        return m.validate_proofs(self.spec, self.stage, full, gzip, m.COMMIT, m.PARENT,
                                m.SPEC_SHA, m.STAGE_SHA, self.full_sha, self.tree)

    def test_actual_stage_and_real_source_target_index_ledger_pass(self):
        self.assert_stage(self.stage)
        coverage = self.module.validate_stage_ledger(self.spec, self.stage, self.stage_lines, self.local)
        self.assertEqual(len(coverage), 2685)
        self.assertEqual(len({v["git_blob"] for v in coverage.values()}), 2153)

    def test_unqualified_strict_pass_or_changed_exit_warning_rejected(self):
        for mutate in (lambda s: s.update(strict_diff_check="PASS"),
                       lambda s: s["actual_diff_check"].update(actual_exit_code=0),
                       lambda s: s["actual_diff_check"].update(warnings=6),
                       lambda s: s["precommit_actual_diff_check"].update(exception_line=213),
                       lambda s: s["actual_diff_check"].update(exception_source_sha256="0" * 64)):
            bad = copy.deepcopy(self.stage);mutate(bad)
            with self.assertRaises(ValueError):self.assert_stage(bad)

    def test_pending_stage_wrong_commit_parent_spec_or_role_rejected(self):
        for key, value in (("status", "PENDING"), ("commit_sha", "0" * 40),
                           ("expected_parent_sha", "0" * 40), ("spec_sha256", "0" * 64),
                           ("runner_execution", "/root/backup_manager")):
            bad = copy.deepcopy(self.stage);bad[key] = value
            with self.assertRaises(ValueError):self.assert_stage(bad)

    def test_failed_evidence_exception_source_bindings_and_guard_flag_rejected(self):
        for key, value in (("prior_failed_stage_sha256", "0" * 64), ("prior_failed_ledger_sha256", "0" * 64),
                           ("exception_configuration_sha256", "0" * 64), ("source_sha256", "0" * 64),
                           ("full_index_exact_scope_verified", False), ("protected_original_six_baselines_verified", False)):
            bad = copy.deepcopy(self.stage);bad[key] = value
            with self.assertRaises(ValueError):self.assert_stage(bad)

    def test_stage_missing_source_duplicate_blob_path_and_changed_hash_rejected(self):
        variants = []
        bad = copy.deepcopy(self.stage_lines);bad.remove(next(x for x in bad if x["phase"] == "source_target_reverified_no_copy"));variants.append(bad)
        bad = copy.deepcopy(self.stage_lines);x = next(x for x in bad if x["phase"] == "actual_index_blob_reverified");x["paths"].append(x["paths"][0]);variants.append(bad)
        bad = copy.deepcopy(self.stage_lines);next(x for x in bad if x["phase"] == "source_target_reverified_no_copy")["sha256"] = "0" * 64;variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):self.module.validate_stage_ledger(self.spec, self.stage, bad, self.local)

    def test_local_commit_or_warning_payload_changed_rejected(self):
        bad = copy.deepcopy(self.local);bad["status"] = "PENDING"
        with self.assertRaises(ValueError):self.module.validate_stage_ledger(self.spec, self.stage, self.stage_lines, bad)
        bad = copy.deepcopy(self.stage);bad["exception_actual_five_source_target_index_proofs"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):self.module.validate_stage_ledger(self.spec, bad, self.stage_lines, self.local)

    def test_isolated_synthetic_full_and_gzip_graph_passes_no_execution(self):
        full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH()
        self.assertEqual(len(self.validate(full, gzip)), 2685)

    def test_incomplete_full_or_wrong_frozen_inventory_rejected(self):
        for key, value in (("status", "PUSHED_PENDING"), ("files", 2684), ("total_bytes", 1080978675),
                           ("local_git_verified_unique_blobs", 2152), ("current_source_matching_files", 2684)):
            full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();full[key] = value
            with self.assertRaises(ValueError):self.validate(full, gzip)

    def test_full_source_evolution_or_wrong_actor_transport_tree_rejected(self):
        for key, value in (("runner_execution", "/other"), ("verification_source_sha256", "0" * 64),
                           ("current_source_evolution", [{"path": "changed"}]),
                           ("current_source_evolution_at_finish", [{"path": "changed"}])):
            full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();full[key] = value
            with self.assertRaises(ValueError):self.validate(full, gzip)
        for key, value in (("http_accept", "cached"), ("cache_policy", "inherited"), ("remote_tree_sha", "0" * 40)):
            full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();full["remote_verification"][key] = value
            with self.assertRaises(ValueError):self.validate(full, gzip)

    def test_remote_cached_provenance_empty_duplicate_missing_outside_or_hash_rejected(self):
        mutations = [lambda b: b[0].update(verification_source="local_git_cache"),
                     lambda b: b[0].update(paths=[]), lambda b: b[0]["paths"].append(b[0]["paths"][0]),
                     lambda b: b[0]["paths"].pop(), lambda b: b[0]["paths"].append("outside"),
                     lambda b: b[0].update(sha256="0" * 64)]
        for mutate in mutations:
            full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();mutate(full["remote_verification"]["blob_receipts"])
            with self.assertRaises(ValueError):self.validate(full, gzip)

    def test_full_actual_stage_binding_or_private_ADMIN_required(self):
        full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();full["prior_evidence"] = []
        with self.assertRaises(ValueError):self.validate(full, gzip)
        full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();full["private_repository_evidence"]["viewerPermission"] = "WRITE"
        with self.assertRaises(ValueError):self.validate(full, gzip)

    def test_wrong_local_remote_git_sha1_rejected(self):
        full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();coverage = self.validate(full, gzip)
        checked = {name: {k: v[k] for k in ("git_blob", "size", "sha256")} for name, v in coverage.items()}
        self.module.validate_local_remote_blob_coverage(coverage, checked)
        bad = copy.deepcopy(checked);bad[next(iter(bad))]["git_blob"] = "0" * 40
        with self.assertRaises(ValueError):self.module.validate_local_remote_blob_coverage(coverage, bad)

    def test_pending_gzip_missing_actual_fetch_or_wrong_raw_count_rejected(self):
        for key, value in (("status", "PENDING"), ("actual_remote_gzip_fetches", 25),
                           ("oversized_raw_artifacts", 25), ("raw_bytes_exactly_recovered", 1353229113)):
            full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();gzip[key] = value
            with self.assertRaises(ValueError):self.validate(full, gzip)
        for key in ("actual_remote_fetch_this_run", "decoded_byte_for_byte_equal_to_preserved_raw"):
            full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();gzip["actual_remote_gzip_proofs"][0][key] = False
            with self.assertRaises(ValueError):self.validate(full, gzip)

    def test_gzip_wrong_receipt_source_actor_or_bounded_settings_rejected(self):
        for key, value in (("stage_receipt_sha256", "0" * 64), ("verified_inventory_receipt_sha256", "0" * 64),
                           ("source_sha256", "0" * 64), ("source_freeze_sha256", "0" * 64),
                           ("actual_runner_execution", "/root"), ("stream_chunk_bytes", 131072),
                           ("bounded_spool_memory_bytes", 16777216)):
            full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();gzip[key] = value
            with self.assertRaises(ValueError):self.validate(full, gzip)

    def test_gzip_wrong_compressed_blob_or_duplicate_raw_rejected(self):
        full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();gzip["actual_remote_gzip_proofs"][0]["remote_gzip_sha256"] = "0" * 64
        with self.assertRaises(ValueError):self.validate(full, gzip)
        full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();gzip["actual_remote_gzip_proofs"][1] = gzip["actual_remote_gzip_proofs"][0]
        with self.assertRaises(ValueError):self.validate(full, gzip)

    def test_full_local_raw_missing_roundtrip_rejected(self):
        full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();full["oversized_raw_local_verification"][0]["lossless_local_roundtrip"] = False
        with self.assertRaises(ValueError):self.validate(full, gzip)

    def ledgers(self, full, gzip):
        full_lines = [{"phase": "actual_remote_blob_verified", **x} for x in full["remote_verification"]["blob_receipts"]]
        full_lines.append({"phase": "verification_complete", "status": "BACKED_UP"})
        gzip_lines = [{"phase": "ACTUAL_REMOTE_GZIP_BYTE_PROOF", **x} for x in gzip["actual_remote_gzip_proofs"]]
        gzip_lines.append({"phase": "COMPLETE_ACTUAL_REMOTE_GZIP_PROOF", "status": "PASS_ACTUAL_REMOTE_GZIP_DECODE", "raw_count": 26, "actual_fetches": 26})
        return full_lines, gzip_lines

    def test_full_ledger_complete_tuples_and_finish_required(self):
        full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();fl, gl = self.ledgers(full, gzip)
        self.module.validate_ledgers(full, gzip, fl, gl)
        bad = copy.deepcopy(fl);bad[0]["paths"] = ["outside"]
        with self.assertRaises(ValueError):self.module.validate_ledgers(full, gzip, bad, gl)
        with self.assertRaises(ValueError):self.module.validate_ledgers(full, gzip, fl[:-1], gl)

    def test_gzip_ledger_tuple_cross_wiring_and_missing_finish_rejected(self):
        full, gzip = self.synthetic_remote_fixture_DO_NOT_PUBLISH();fl, gl = self.ledgers(full, gzip)
        bad = copy.deepcopy(gl);bad[0]["remote_gzip_sha256"], bad[1]["remote_gzip_sha256"] = bad[1]["remote_gzip_sha256"], bad[0]["remote_gzip_sha256"]
        with self.assertRaises(ValueError):self.module.validate_ledgers(full, gzip, fl, bad)
        with self.assertRaises(ValueError):self.module.validate_ledgers(full, gzip, fl, gl[:-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
