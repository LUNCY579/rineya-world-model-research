"""Safety tests for the separate frozen-checkpoint streaming recovery verifier."""
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("recovery", Path(__file__).with_name("recover_remote_checkpoint.py"))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def entry(path, payload):
    return {"path": path, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class RecoverySafetyTests(unittest.TestCase):
    def test_stream_actual_bytes_sha256_and_git_object_hash(self):
        payload = b"verified payload\n" * 10000
        blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        self.assertEqual(recovery.hash_stream(io.BytesIO(payload), len(payload), blob),
                         {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})

    def test_truncated_excess_and_wrong_object_are_rejected(self):
        for payload, size, blob in [(b"x", 2, None), (b"xx", 1, None), (b"x", 1, "0" * 40)]:
            with self.subTest(payload=payload, size=size), self.assertRaises(ValueError):
                recovery.hash_stream(io.BytesIO(payload), size, blob)

    def test_json_response_cannot_certify_raw_blob(self):
        payload = b"{}"
        actual_blob = hashlib.sha1(b"blob 2\0" + b"ab").hexdigest()
        with self.assertRaises(ValueError):
            recovery.hash_stream(io.BytesIO(payload), 2, actual_blob)

    def test_secret_across_stream_boundary_is_rejected(self):
        marker = b"gh" + b"p_" + b"Q" * 25
        payload = b" " * (recovery.CHUNK - 2) + marker
        with self.assertRaises(ValueError):
            recovery.hash_stream(io.BytesIO(payload), scan_secrets=True)

    def test_safe_paths_and_file_cap(self):
        for path in ["../outside", "/outside", "data/.git/config", "data//row", "data\nrow"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                recovery.relative_path(path)
        with self.assertRaises(ValueError):
            recovery.check_entry({"path": "data/a", "size": recovery.MAX_BYTES + 1, "sha256": "a" * 64})

    def test_evolving_python_requires_explicit_mode_and_exact_archive_if_present(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = Path(directory)
            (factory / "pipeline.py").write_bytes(b"new\n")
            frozen = entry("pipeline.py", b"old\n")
            with self.assertRaises(ValueError):
                recovery.check_source(factory, frozen, False)
            result = recovery.check_source(factory, frozen, True)
            self.assertEqual(result["status"], "EVOLVED_PYTHON_SOURCE_FROZEN_COMMIT_BYTES_REQUIRED")
            archive = factory / "reproducibility/source_blobs" / (frozen["sha256"] + ".source")
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"wrong\n")
            with self.assertRaises(ValueError):
                recovery.check_source(factory, frozen, True)
            archive.write_bytes(b"old\n")
            self.assertEqual(recovery.check_source(factory, frozen, True)["frozen_source_archive"],
                             str(archive.relative_to(factory)))

    def test_non_source_and_symlink_cannot_use_evolving_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = Path(directory)
            (factory / "review.json").write_bytes(b"new")
            with self.assertRaises(ValueError):
                recovery.check_source(factory, entry("review.json", b"old"), True)
            (factory / "linked.py").symlink_to(factory / "review.json")
            with self.assertRaises(ValueError):
                recovery.check_source(factory, entry("linked.py", b"old"), True)

    def test_subprocess_failure_and_stream_success(self):
        result = recovery.process_stream([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'abc')"],
                                         lambda handle: recovery.hash_stream(handle, 3))
        self.assertEqual(result["sha256"], hashlib.sha256(b"abc").hexdigest())
        with self.assertRaises(RuntimeError):
            recovery.process_stream([sys.executable, "-c", "raise SystemExit(3)"],
                                    lambda handle: recovery.hash_stream(handle))


if __name__ == "__main__":
    unittest.main()
