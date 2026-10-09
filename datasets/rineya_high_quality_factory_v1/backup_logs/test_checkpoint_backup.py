"""Safety regression tests for the scoped checkpoint backup helper."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

location = Path(__file__).with_name("checkpoint_backup.py")
module_spec = importlib.util.spec_from_file_location("checkpoint_backup", location)
backup = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(backup)


class BackupSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.factory = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def spec(self, data=b"safe bytes\n"):
        path = self.factory / "sample.jsonl"
        path.write_bytes(data)
        return {"approval": "provisional_checkpoint", "approved_by": "unit-test",
                "validation_status": "passed", "files": [{"path": path.name,
                "size": len(data), "sha256": backup.digest(data)}]}

    def test_source_mutation_is_rejected(self):
        spec = self.spec()
        (self.factory / "sample.jsonl").write_bytes(b"changed bytes\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            backup.validated_inventory(self.factory, spec)

    def test_path_escape_and_noncanonical_paths_are_rejected(self):
        for path in ["../outside.txt", "/outside.txt", "a/../sample.jsonl", "./sample.jsonl", "a//b"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                backup.safe_path(self.factory, path)

    def test_symlink_is_rejected(self):
        self.spec()
        (self.factory / "link.jsonl").symlink_to(self.factory / "sample.jsonl")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            backup.safe_path(self.factory, "link.jsonl")

    def test_secrets_are_rejected_without_exposing_content(self):
        credential = b"gh" + b"p_" + b"A" * 30
        spec = self.spec(credential)
        with self.assertRaisesRegex(ValueError, "content withheld") as caught:
            backup.validated_inventory(self.factory, spec)
        self.assertNotIn(credential.decode(), str(caught.exception))

    def test_size_limit_and_missing_approval_are_rejected(self):
        spec = self.spec()
        with patch.object(backup, "MAX_BYTES", 1), self.assertRaisesRegex(ValueError, "cap"):
            backup.validated_inventory(self.factory, spec)
        spec["approved_by"] = ""
        with self.assertRaisesRegex(ValueError, "approval"):
            backup.validated_inventory(self.factory, spec)

    def test_duplicate_entries_are_rejected(self):
        spec = self.spec()
        spec["files"].append(dict(spec["files"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            backup.validated_inventory(self.factory, spec)

    def test_remote_blob_bytes_must_match_sha256(self):
        inventory = self.spec()["files"]
        remote_commit = {"tree": {"sha": "tree-sha"}}
        tree = {"tree": [{"path": backup.DESTINATION + "/sample.jsonl", "type": "blob",
                          "sha": "blob-sha", "size": inventory[0]["size"]}]}
        import base64
        blob = {"encoding": "base64", "content": base64.b64encode(b"wrong bytes").decode()}
        def mock_command(args, *unused, **kwargs):
            if args[0] == "git":
                return "tree-sha"
            resource = args[-1]
            if "/git/commits/" in resource:
                return json.dumps(remote_commit)
            if "/git/trees/" in resource:
                return json.dumps(tree)
            return json.dumps(blob)
        with patch.object(backup, "command", side_effect=mock_command):
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                backup.verify_remote_bytes(self.factory, "commit-sha", inventory)


if __name__ == "__main__":
    unittest.main()
