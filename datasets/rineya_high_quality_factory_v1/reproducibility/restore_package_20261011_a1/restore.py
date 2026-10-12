"""SOURCE ONLY, UNEXECUTED: restore one exact captured Rineya factory cut.

Uses Python's standard library and the caller's authenticated gh CLI. It does
not import dataset helpers, consult a local Git repository, extract tar paths,
write Git configuration, or resolve a moving branch. See RESTORE_README.txt.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unicodedata

REPOSITORY = "LUNCY579/rineya-world-model-research"
DESCRIPTOR_SHA256 = "238c0b9d992e4721f0776d57cc14f7d4974675c15681687decc68b658aed3034"
DESCRIPTOR_BYTES = 8857032
FACTORY_PREFIX = "datasets/rineya_high_quality_factory_v1/"
PART_PREFIX = FACTORY_PREFIX + "reproducibility/compact_factory_custody_20261011_a1_part"
BRANCH_PREFIX = "research/rineya-complete-factory-custody-20261011-a1-part"
CUSTODY_PARENT = "064c5baeceb46045c41b1cea6e170048995c0d0b"
EXPANDED_BYTES = 5083117507
CHUNK = 65536
LEAF_CAP = 25 * 1024 * 1024
CONTAINER_CAP = 96 * 1024 * 1024
DECODED_PART_CAP = 64 * 1024 * 1024
METADATA_CAP = 32 * 1024 * 1024
RESERVE = 512 * 1024 * 1024
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class RestoreError(Exception):
    """Only fixed diagnostic codes are emitted; never remote stderr or tokens."""


def require(value, code):
    if not value:
        raise RestoreError(code)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def stable_stat(value):
    # Reads may update atime. It is not a custody invariant.
    return tuple(getattr(value, key) for key in
                 ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode"))


def literal_path(name):
    require(type(name) is str and bool(name), "unsafe_path_type")
    require(not any(c in name for c in ("\\", "\0", "\n", "\r")), "unsafe_path_character")
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix() == name and bool(p.parts), "path_alias")
    require(not any(c in (".", "..", ".git") for c in p.parts), "path_traversal")
    return p


def unique_paths(names):
    """Reject file/directory collisions and portable case/Unicode aliases."""
    seen, aliases, directories = set(), {}, set()
    for name in names:
        p = literal_path(name)
        require(name not in seen, "duplicate_path")
        seen.add(name)
        directories.update(q.as_posix() for q in p.parents if q.as_posix() != ".")
    require(not (seen & directories), "file_directory_collision")
    for name in sorted(seen | directories):
        key = unicodedata.normalize("NFC", name).casefold()
        require(key not in aliases or aliases[key] == name, "portable_path_alias")
        aliases[key] = name
    return directories


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def strict_json(raw):
    try:
        return json.loads(raw, object_pairs_hook=strict_object,
                          parse_constant=lambda unused: (_ for _ in ()).throw(RestoreError("nonfinite_json")))
    except (ValueError, UnicodeError) as exc:
        raise RestoreError("invalid_json") from exc


def validate_row(row, cap=None):
    require(type(row) is dict and set(row) == {"path", "size", "sha256", "git_blob", "mode"},
            "source_tuple_shape")
    literal_path(row["path"])
    require(type(row["size"]) is int and row["size"] >= 0 and
            (cap is None or row["size"] <= cap), "source_size")
    require(type(row["sha256"]) is str and HEX64.fullmatch(row["sha256"]), "source_sha256")
    require(type(row["git_blob"]) is str and HEX40.fullmatch(row["git_blob"]), "source_git_blob")
    require(row["mode"] in ("100644", "100755"), "source_regular_mode")
    return row


def triple(row):
    return {key: row[key] for key in ("path", "size", "sha256")}


def canonical_absolute(value, fresh=False, directory=False):
    p = Path(value)
    require(p.is_absolute() and str(p) == value, "absolute_canonical_path_required")
    require(not any(c in value for c in ("\0", "\n", "\r")), "unsafe_local_path")
    require(p.name not in ("", ".", ".."), "unsafe_local_basename")
    for ancestor in reversed(p.parents):
        require(ancestor.exists() and not ancestor.is_symlink() and
                stat.S_ISDIR(ancestor.lstat().st_mode), "local_parent_not_regular_directory")
    require(p.parent.resolve(strict=True) == p.parent, "local_parent_alias")
    if fresh:
        require(not p.exists() and not p.is_symlink(), "destination_or_report_already_exists")
    else:
        require(p.exists() and not p.is_symlink(), "local_input_missing_or_link")
        require(stat.S_ISDIR(p.lstat().st_mode) if directory else stat.S_ISREG(p.lstat().st_mode),
                "local_input_wrong_type")
        require(p.resolve(strict=True) == p, "local_input_alias")
    return p


def descriptor_from(path, external_sha):
    require(external_sha == DESCRIPTOR_SHA256, "external_descriptor_hash_not_this_cut")
    p = canonical_absolute(path)
    require(p.stat().st_size == DESCRIPTOR_BYTES, "descriptor_size")
    before = p.stat()
    with p.open("rb") as source:
        raw = source.read(DESCRIPTOR_BYTES + 1)
    after = p.stat()
    require(stable_stat(before) == stable_stat(after) and len(raw) == DESCRIPTOR_BYTES and sha256(raw) == external_sha,
            "descriptor_changed_or_hash_differs")
    d = strict_json(raw)
    require(type(d) is dict and d.get("schema") == "compact-complete-factory-snapshot-v1" and
            d.get("git_factory_prefix") == FACTORY_PREFIX and d.get("custody_parent") == CUSTODY_PARENT and
            d.get("data_or_publication_approval") is False and d.get("SOURCE06_stage_success") is False,
            "descriptor_authority")
    current, raw_entries, stats = d.get("current_files"), d.get("raw_closure"), d.get("source_stats")
    require(type(current) is list and len(current) == 12679 and type(raw_entries) is list and
            len(raw_entries) == 210 and type(stats) is dict and len(stats) == 12889, "descriptor_counts")
    current_map = {validate_row(row, LEAF_CAP)["path"]: row for row in current}
    require(len(current_map) == len(current) and list(current_map) == sorted(current_map), "current_namespace_order")
    raw_map, associations = {}, set()
    for entry in raw_entries:
        require(type(entry) is dict and set(entry) == {"raw", "representations", "origins"}, "raw_entry_shape")
        row = validate_row(entry["raw"])
        require(row["path"] not in current_map and row["path"] not in raw_map, "raw_namespace_duplicate")
        raw_map[row["path"]] = row
        require(type(entry["representations"]) is list and bool(entry["representations"]) and
                type(entry["origins"]) is list, "raw_association_shape")
        for rep in entry["representations"]:
            require(type(rep) is dict and set(rep) == {"gzip", "manifest"}, "representation_shape")
            for role in ("gzip", "manifest"):
                bound = rep[role]
                require(type(bound) is dict and set(bound) == {"path", "size", "sha256"}, "representation_binding_shape")
                require(bound["path"] in current_map and triple(current_map[bound["path"]]) == bound,
                        "representation_not_current_capped")
            association = (row["path"], rep["gzip"]["path"], rep["manifest"]["path"])
            require(association not in associations, "duplicate_raw_association")
            associations.add(association)
    require(list(raw_map) == sorted(raw_map) and len(associations) == 211 and
            sum(row["size"] > LEAF_CAP for row in raw_map.values()) == 43, "raw_complete_closure")
    all_rows = dict(current_map)
    all_rows.update(raw_map)
    directories = unique_paths(all_rows)
    require(set(stats) == set(all_rows), "permission_namespace")
    permissions, mode_counts = {}, {}
    for name, row in all_rows.items():
        value = stats[name]
        require(type(value) is list and len(value) == 6 and all(type(v) is int for v in value) and
                value[2] == row["size"] and stat.S_ISREG(value[5]), "source_stat_shape")
        mode = value[5] & 0o777
        require(mode in (0o644, 0o444, 0o600) and row["mode"] == "100644", "original_permission_bits")
        permissions[name] = mode
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    require(mode_counts == {0o644: 11706, 0o444: 1068, 0o600: 115}, "permission_counts")
    require(sum(row["size"] for row in all_rows.values()) == EXPANDED_BYTES, "expanded_size")
    parts, delta = d.get("parts"), d.get("delta_paths")
    require(type(parts) is list and len(parts) == 18 and type(delta) is list and len(delta) == 2907 and
            len(set(delta)) == len(delta) and delta == sorted(delta) and set(delta) <= set(current_map), "delta_plan")
    flattened = []
    for ordinal, part in enumerate(parts, 1):
        require(type(part) is dict and set(part) == {"ordinal", "paths", "raw_bytes"} and
                type(part["ordinal"]) is int and part["ordinal"] == ordinal and
                type(part["paths"]) is list and bool(part["paths"]), "part_plan_shape")
        require(all(name in current_map for name in part["paths"]) and
                type(part["raw_bytes"]) is int and
                part["raw_bytes"] == sum(current_map[name]["size"] for name in part["paths"]) and
                part["raw_bytes"] <= DECODED_PART_CAP, "part_plan_bytes")
        flattened.extend(part["paths"])
    require(flattened == delta and sum(part["raw_bytes"] for part in parts) == 327794609, "part_exact_union")
    for entry in raw_entries:
        for origin in entry["origins"]:
            require(type(origin) is dict and set(origin) == {"path", "size", "sha256"} and
                    origin["path"] in all_rows and triple(all_rows[origin["path"]]) == origin and
                    (origin["size"], origin["sha256"]) == (entry["raw"]["size"], entry["raw"]["sha256"]),
                    "raw_origin_binding")
    return d, current_map, raw_map, all_rows, directories, permissions


def gh_json(suffix):
    endpoint = "repos/" + REPOSITORY + ("/" + suffix if suffix else "")
    process = subprocess.Popen(["gh", "api", endpoint], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        raw = process.stdout.read(METADATA_CAP + 1)
        require(len(raw) <= METADATA_CAP, "github_metadata_cap")
        require(process.stdout.read(1) == b"", "github_metadata_extra_bytes")
        process.stdout.close()
        require(process.wait() == 0, "github_metadata_native_failure")
        return strict_json(raw)
    except BaseException:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise


def private_repository():
    metadata = gh_json("")
    require(type(metadata) is dict and metadata.get("full_name") == REPOSITORY and
            metadata.get("private") is True and type(metadata.get("permissions")) is dict and
            metadata["permissions"].get("pull") is True, "private_repository_read_access_required")


def pinned_chain(final_commit):
    commits, sha = {}, final_commit
    for ordinal in range(18, 0, -1):
        value = gh_json("git/commits/" + sha)
        require(type(value) is dict and value.get("sha") == sha and type(value.get("parents")) is list and
                len(value["parents"]) == 1 and type(value.get("tree")) is dict and
                type(value["tree"].get("sha")) is str and HEX40.fullmatch(value["tree"]["sha"]),
                "pinned_single_parent_commit_required")
        parent = value["parents"][0].get("sha")
        require(type(parent) is str and HEX40.fullmatch(parent), "pinned_commit_parent")
        commits[ordinal] = {"commit": sha, "parent": parent, "tree": value["tree"]["sha"]}
        sha = parent
    require(sha == CUSTODY_PARENT, "final_commit_not_exact_eighteen_part_chain")
    return commits


def pinned_tree(tree_sha):
    metadata = gh_json("git/trees/" + tree_sha + "?recursive=1")
    require(type(metadata) is dict and metadata.get("sha") == tree_sha and
            metadata.get("truncated") is False and type(metadata.get("tree")) is list,
            "complete_untruncated_tree_required")
    leaves, all_paths = {}, set()
    for row in metadata["tree"]:
        require(type(row) is dict and type(row.get("path")) is str, "tree_tuple_shape")
        path = row["path"]
        literal_path(path)
        require(path not in all_paths, "duplicate_tree_path")
        all_paths.add(path)
        require(type(row.get("sha")) is str and HEX40.fullmatch(row["sha"]), "tree_hash")
        if row.get("type") == "tree":
            require(row.get("mode") == "040000", "tree_directory_mode")
        else:
            require(row.get("type") == "blob" and row.get("mode") in ("100644", "100755") and
                    type(row.get("size")) is int and row["size"] >= 0, "tree_links_or_unsupported_leaf")
            leaves[path] = {key: row[key] for key in ("mode", "type", "sha", "size")}
    directories = unique_paths(leaves)
    require(all_paths == set(leaves) | directories, "tree_directory_namespace")
    return leaves, directories


class HashReader:
    def __init__(self, source, size):
        require(source is not None, "archive_member_reader_missing")
        self.source, self.size, self.count = source, size, 0
        self.sha256 = hashlib.sha256()
        self.sha1 = hashlib.sha1(f"blob {size}\0".encode("ascii"))

    def read(self, size):
        require(type(size) is int and size >= 0, "unbounded_archive_read")
        value = self.source.read(size)
        self.count += len(value)
        require(self.count <= self.size, "archive_member_overrun")
        self.sha256.update(value)
        self.sha1.update(value)
        return value


def verified_archive(stream, leaves, directories, visitor):
    seen, seen_dirs, wrapper, ordinal, total = set(), set(), None, 0, 0
    ledger = hashlib.sha256()
    try:
        with gzip.GzipFile(fileobj=stream, mode="rb") as outer:
            with tarfile.open(fileobj=outer, mode="r|") as archive:
                for member in archive:
                    ordinal += 1
                    p = literal_path(member.name)
                    if wrapper is None:
                        wrapper = p.parts[0]
                    require(p.parts[0] == wrapper, "mixed_archive_wrapper")
                    if len(p.parts) == 1:
                        require(ordinal == 1 and member.isdir(),
                                "archive_wrapper_not_first_unique_directory")
                        continue
                    path = "/".join(p.parts[1:])
                    literal_path(path)
                    if member.isdir():
                        require(path in directories and path not in seen_dirs, "archive_directory_duplicate_or_extra")
                        seen_dirs.add(path)
                        continue
                    require(path in leaves and path not in seen and member.isfile() and
                            not member.issym() and not member.islnk(), "archive_leaf_link_duplicate_or_extra")
                    expected = leaves[path]
                    actual_mode = "100755" if member.mode & 0o111 else "100644"
                    require(member.size == expected["size"] and actual_mode == expected["mode"], "archive_size_or_mode")
                    reader = HashReader(archive.extractfile(member), member.size)
                    visitor(path, reader, member.size)
                    require(reader.count == member.size and reader.read(1) == b"" and
                            reader.sha1.hexdigest() == expected["sha"], "archive_full_git_blob")
                    seen.add(path)
                    total += member.size
                    measured = {"path": path, **expected, "sha256": reader.sha256.hexdigest()}
                    ledger.update((json.dumps(measured, sort_keys=True, separators=(",", ":")) + "\n").encode())
                require(not any(archive.fileobj.buf), "archive_hidden_nonzero_tail")
            while True:
                tail = outer.read(CHUNK)
                if not tail:
                    break
                require(not any(tail), "archive_nonzero_tail")
    except (OSError, EOFError, tarfile.TarError, UnicodeError) as exc:
        raise RestoreError("archive_framing_or_crc_eof") from exc
    require(seen == set(leaves), "archive_missing_git_leaves")
    return {"leaf_count": len(seen), "full_leaf_bytes": total,
            "ordered_full_leaf_sha256_ledger": ledger.hexdigest(), "outer_gzip_CRC_EOF": True,
            "complete_pinned_git_namespace_and_modes": True}


def local_target(destination, name, create_parents=False):
    p = destination
    parts = literal_path(name).parts
    for component in parts[:-1]:
        p = p / component
        if create_parents and not p.exists() and not p.is_symlink():
            p.mkdir(mode=0o700)
        require(p.exists() and not p.is_symlink() and stat.S_ISDIR(p.lstat().st_mode), "restore_parent_link_or_missing")
    leaf = p / parts[-1]
    require(not leaf.is_symlink(), "restore_leaf_link")
    return leaf


def exclusive_binary(path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.fdopen(os.open(str(path), flags, 0o600), "wb")


def measured_file(path, expected):
    require(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), "restored_nonregular_file")
    before = path.stat()
    sha, git, size = hashlib.sha256(), hashlib.sha1(f"blob {expected['size']}\0".encode("ascii")), 0
    with path.open("rb") as source:
        while True:
            value = source.read(CHUNK)
            if not value:
                break
            size += len(value)
            require(size <= expected["size"], "restored_file_overrun")
            sha.update(value)
            git.update(value)
    require(stable_stat(before) == stable_stat(path.stat()) and size == expected["size"] and sha.hexdigest() == expected["sha256"] and
            git.hexdigest() == expected["git_blob"], "restored_full_source_tuple")
    return before


class Restorer:
    def __init__(self, args, report):
        self.args, self.report = args, report
        self.stage = "descriptor"
        self.destination, self.workspace = None, None
        self.restored = set()
        self.part_states = {}

    def capacity(self, planned_bytes=0):
        parent = self.destination.parent
        free = shutil.disk_usage(parent).free
        require(free >= RESERVE + planned_bytes, "fresh_disk_reserve_unavailable")
        return free

    def write_source(self, name, handle, size):
        require(name in self.all_rows and name not in self.restored, "duplicate_or_unplanned_restore")
        row = self.all_rows[name]
        require(size == row["size"], "restored_source_size")
        self.capacity(size)
        target = local_target(self.destination, name, create_parents=True)
        sha, git, count = hashlib.sha256(), hashlib.sha1(f"blob {size}\0".encode("ascii")), 0
        with exclusive_binary(target) as output:
            while True:
                value = handle.read(min(CHUNK, size - count + 1))
                if not value:
                    break
                count += len(value)
                require(count <= size, "restored_source_expansion_overrun")
                output.write(value)
                sha.update(value)
                git.update(value)
            require(count == size and sha.hexdigest() == row["sha256"] and git.hexdigest() == row["git_blob"],
                    "restored_source_full_hash")
            output.flush()
            os.fsync(output.fileno())
        target.chmod(self.permissions[name])
        self.restored.add(name)

    def part_manifest(self, ordinal, raw):
        value = strict_json(raw)
        keys = {"schema", "scope", "at_utc", "branch", "parent", "config_sha256", "source_sha256",
                "data_or_publication_approval", "files", "payload", "container_bytes", "container_sha256"}
        require(type(value) is dict and set(value) == keys and
                value["schema"] == "root-urgent-review-custody-manifest-v1" and
                value["data_or_publication_approval"] is False and
                value["branch"] == BRANCH_PREFIX + f"{ordinal:02d}" and
                value["parent"] == self.commits[ordinal]["parent"] and
                type(value["scope"]) is str and type(value["at_utc"]) is str, "part_manifest_identity")
        for key in ("config_sha256", "source_sha256", "container_sha256"):
            require(type(value[key]) is str and HEX64.fullmatch(value[key]), "part_manifest_hash")
        require(type(value["container_bytes"]) is int and 0 <= value["container_bytes"] <= CONTAINER_CAP,
                "part_container_cap")
        expected = [self.current[name] for name in self.descriptor["parts"][ordinal - 1]["paths"]]
        require(type(value["files"]) is list, "part_manifest_file_list")
        for row in value["files"]:
            require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and
                    type(row["path"]) is str and type(row["bytes"]) is int and
                    type(row["sha256"]) is str, "part_manifest_file_closed_types")
        require(value["files"] == [{"path": r["path"], "bytes": r["size"], "sha256": r["sha256"]}
                                   for r in expected], "part_manifest_exact_ordered_union")
        payload = value["payload"]
        payload_name = PART_PREFIX + f"{ordinal:02d}/critical_review_bytes.json.gz"
        require(type(payload) is dict and set(payload) == {"path", "bytes", "sha256"} and
                payload["path"] == payload_name and type(payload["bytes"]) is int and
                0 <= payload["bytes"] <= LEAF_CAP and payload["bytes"] == self.leaves[payload_name]["size"] and
                type(payload["sha256"]) is str and HEX64.fullmatch(payload["sha256"]), "part_payload_binding")
        return value

    def consume_pair(self, ordinal):
        state = self.part_states[ordinal]
        if state["manifest"] is None or state["spool"] is None:
            return
        require(not state["done"], "part_pair_consumed_twice")
        manifest = state["manifest"]
        with state["spool"].open("rb") as source:
            packed = source.read(LEAF_CAP + 1)
        require(len(packed) == manifest["payload"]["bytes"] and
                sha256(packed) == manifest["payload"]["sha256"], "part_compressed_full_sha")
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(packed), mode="rb") as source:
                decoded = source.read(manifest["container_bytes"] + 1)
                require(len(decoded) == manifest["container_bytes"] and source.read(1) == b"", "part_container_full_eof")
        except (OSError, EOFError) as exc:
            raise RestoreError("part_container_crc_eof") from exc
        require(sha256(decoded) == manifest["container_sha256"], "part_container_full_sha")
        container = strict_json(decoded)
        require(type(container) is dict and set(container) == {"schema", "scope", "data_or_publication_approval", "entries"} and
                container["schema"] == "root-exact-byte-preservation-container-v1" and
                container["scope"] == manifest["scope"] and container["data_or_publication_approval"] is False and
                type(container["entries"]) is list, "part_container_shape")
        paths = self.descriptor["parts"][ordinal - 1]["paths"]
        require(len(container["entries"]) == len(paths), "part_entry_count")
        total = 0
        for name, entry in zip(paths, container["entries"]):
            row = self.current[name]
            require(type(entry) is dict and set(entry) == {"path", "bytes", "sha256", "raw_base64"} and
                    entry["path"] == name and type(entry["bytes"]) is int and entry["bytes"] == row["size"] and
                    entry["sha256"] == row["sha256"] and type(entry["raw_base64"]) is str and
                    len(entry["raw_base64"]) == 4 * ((row["size"] + 2) // 3), "part_entry_exact_tuple")
            try:
                raw = base64.b64decode(entry["raw_base64"], validate=True)
            except (ValueError, TypeError, binascii.Error) as exc:
                raise RestoreError("part_strict_base64") from exc
            require(base64.b64encode(raw).decode("ascii") == entry["raw_base64"] and len(raw) == row["size"] and
                    sha256(raw) == row["sha256"] and git_blob(raw) == row["git_blob"], "part_full_decoded_tuple")
            total += len(raw)
            require(total <= DECODED_PART_CAP, "part_decoded_raw_cap")
            self.write_source(name, io.BytesIO(raw), len(raw))
        require(total == self.descriptor["parts"][ordinal - 1]["raw_bytes"], "part_exact_decoded_bytes")
        state["spool"].unlink()
        state["spool"] = None
        state["done"] = True

    def visit(self, path, handle, size):
        if path in self.part_roles:
            ordinal, role = self.part_roles[path]
            state = self.part_states[ordinal]
            require(size <= LEAF_CAP and not state["done"], "part_leaf_cap_or_duplicate")
            if role == "manifest":
                require(state["manifest"] is None, "duplicate_part_manifest")
                raw = handle.read(size)
                require(len(raw) == size, "part_manifest_short_read")
                state["manifest"] = self.part_manifest(ordinal, raw)
            else:
                require(state["spool"] is None, "duplicate_part_payload")
                self.capacity(size)
                spool = self.workspace / f"part{ordinal:02d}.json.gz"
                with exclusive_binary(spool) as output:
                    count = 0
                    while True:
                        value = handle.read(CHUNK)
                        if not value:
                            break
                        output.write(value)
                        count += len(value)
                    require(count == size, "part_payload_short_read")
                state["spool"] = spool
            self.consume_pair(ordinal)
            return
        name = path[len(FACTORY_PREFIX):] if path.startswith(FACTORY_PREFIX) else None
        if name in self.current and name not in self.delta:
            self.write_source(name, handle, size)
            return
        while handle.read(CHUNK):
            pass

    def compare_files(self, first, second, size):
        before_first, before_second = stable_stat(first.stat()), stable_stat(second.stat())
        with first.open("rb") as a, second.open("rb") as b:
            count = 0
            while True:
                value = a.read(CHUNK)
                require(value == b.read(len(value)), "raw_origin_full_bytes_differ")
                if not value:
                    require(b.read(1) == b"", "raw_origin_extra_bytes")
                    break
                count += len(value)
            require(count == size, "raw_origin_full_size")
        require(stable_stat(first.stat()) == before_first and stable_stat(second.stat()) == before_second,
                "raw_origin_changed_during_comparison")

    def compare_gzip(self, compressed, row):
        gzip_path = local_target(self.destination, compressed["path"])
        measured_file(gzip_path, self.current[compressed["path"]])
        before = gzip_path.stat()
        try:
            with gzip_path.open("rb") as packed, gzip.GzipFile(fileobj=packed, mode="rb") as decoded:
                if row["path"] not in self.restored:
                    self.write_source(row["path"], decoded, row["size"])
                else:
                    original = local_target(self.destination, row["path"])
                    original_before = original.stat()
                    sha, git, count = hashlib.sha256(), hashlib.sha1(f"blob {row['size']}\0".encode("ascii")), 0
                    with original.open("rb") as target:
                        while True:
                            value = decoded.read(CHUNK)
                            require(value == target.read(len(value)), "second_raw_representation_full_bytes")
                            if not value:
                                require(target.read(1) == b"", "second_raw_representation_shorter")
                                break
                            count += len(value)
                            require(count <= row["size"], "second_raw_representation_overrun")
                            sha.update(value)
                            git.update(value)
                    require(count == row["size"] and sha.hexdigest() == row["sha256"] and
                            git.hexdigest() == row["git_blob"] and
                            stable_stat(original.stat()) == stable_stat(original_before),
                            "second_raw_representation_full_tuple")
                require(decoded.read(1) == b"", "raw_gzip_extra_decoded_bytes")
        except (OSError, EOFError) as exc:
            raise RestoreError("raw_gzip_crc_eof") from exc
        require(stable_stat(gzip_path.stat()) == stable_stat(before), "raw_gzip_changed_during_decode")

    def raw_closure(self):
        associations = 0
        for entry in self.descriptor["raw_closure"]:
            for rep in entry["representations"]:
                manifest = local_target(self.destination, rep["manifest"]["path"])
                measured_file(manifest, self.current[rep["manifest"]["path"]])
                self.compare_gzip(rep["gzip"], entry["raw"])
                associations += 1
        require(associations == 211 and set(self.raw) <= self.restored, "all_raw_associations_required")
        # Origins may themselves be raw paths later in the sorted closure.
        # Create every raw path first, then compare all origin associations.
        for entry in self.descriptor["raw_closure"]:
            for origin in entry["origins"]:
                self.compare_files(local_target(self.destination, entry["raw"]["path"]),
                                   local_target(self.destination, origin["path"]), entry["raw"]["size"])
        history = self.descriptor["historical_archive_binding"]
        old = history["old_mutable_raw"]
        current = history["bound_current_mutable_raw"]
        require(old["sha256"] == "279beb55eff0c5b6ea60c04d614a67d00606c25c4d1dbdaf99b8ae4ef825fe25" and
                current == self.raw["audits/diversity/belief-gap-v4-belief-controls.json"] and
                current["sha256"] == "14909842a90afdb3de9d78b7e7ff843f9c0ecdb0aa8e5e4bff927850513323ee" and
                len(history["archived_raws"]) == 2 and
                history["historical_representation_is_not_current_raw_mapping"] is True, "historical_current_separation")
        old_manifest_row = history["old_representation"]["manifest"]
        require(self.current[old_manifest_row["path"]] == old_manifest_row, "historical_manifest_tuple")
        old_manifest_path = local_target(self.destination, old_manifest_row["path"])
        measured_file(old_manifest_path, old_manifest_row)
        with old_manifest_path.open("rb") as source:
            old_manifest = strict_json(source.read(LEAF_CAP + 1))
        compressed = history["old_representation"]["gzip"]
        require(self.current[compressed["path"]] == compressed, "historical_gzip_tuple")
        require(old_manifest.get("codec") == "gzip" and old_manifest.get("data_approval") is False and
                old_manifest.get("raw_path") == old["path"] and old_manifest.get("raw_bytes") == old["size"] and
                old_manifest.get("raw_sha256") == old["sha256"] and
                old_manifest.get("gzip_path") == compressed["path"] and
                old_manifest.get("gzip_bytes") == compressed["size"] and
                old_manifest.get("gzip_sha256") == compressed["sha256"], "historical_manifest_describes_old279")
        for row in history["archived_raws"]:
            require(row["path"] != current["path"] and self.raw[row["path"]] == row and
                    (row["size"], row["sha256"]) == (old["size"], old["sha256"]), "historical_archive_raw_tuple")
            self.compare_gzip(compressed, row)
        return {"raw_paths": 210, "current_representation_full_gzip_CRC_EOF_checks": associations,
                "historical_old279_immutable_raw_comparisons": 2,
                "historical279_not_used_for_current149": True}

    def final_namespace(self):
        files, directories = set(), set()
        for parent, dirs, names in os.walk(self.destination, followlinks=False):
            base = Path(parent)
            require(not base.is_symlink() and stat.S_ISDIR(base.lstat().st_mode), "final_directory_link")
            for name in dirs:
                path = base / name
                require(not path.is_symlink() and stat.S_ISDIR(path.lstat().st_mode), "final_directory_nonregular")
                directories.add(path.relative_to(self.destination).as_posix())
            for name in names:
                path = base / name
                require(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), "final_file_nonregular")
                files.add(path.relative_to(self.destination).as_posix())
        require(files == set(self.all_rows) and directories == self.directories and self.restored == files,
                "final_exact_namespace")
        ledger = hashlib.sha256()
        for name in sorted(files):
            row = self.all_rows[name]
            actual = measured_file(local_target(self.destination, name), row)
            require(stat.S_IMODE(actual.st_mode) == self.permissions[name], "final_original_unix_permissions")
            ledger.update((json.dumps({**row, "unix_permission_bits": self.permissions[name]},
                                     sort_keys=True, separators=(",", ":")) + "\n").encode())
        return {"files": len(files), "directories": len(directories), "expanded_file_bytes": EXPANDED_BYTES,
                "sorted_full_source_tuple_permission_ledger_sha256": ledger.hexdigest(),
                "permission_counts": {"0644": 11706, "0444": 1068, "0600": 115},
                "historical_inode_device_timestamp_requirements_applied": False}

    def execute(self):
        self.descriptor, self.current, self.raw, self.all_rows, self.directories, self.permissions = descriptor_from(
            self.args.descriptor, self.args.descriptor_sha256)
        require(HEX40.fullmatch(self.args.final_commit), "exact_pinned_final_commit_required")
        self.destination = canonical_absolute(self.args.destination, fresh=True)
        require(self.report != self.destination and self.destination not in self.report.parents and
                self.report not in self.destination.parents, "report_must_be_outside_destination")
        descriptor_path = Path(self.args.descriptor)
        require(self.destination not in descriptor_path.parents, "descriptor_must_be_outside_destination")
        self.delta = set(self.descriptor["delta_paths"])
        self.stage = "private_repository_metadata"
        private_repository()
        self.stage = "pinned_commit_chain_and_tree"
        self.commits = pinned_chain(self.args.final_commit)
        self.leaves, archive_dirs = pinned_tree(self.commits[18]["tree"])
        self.part_roles = {}
        workspace_bytes = 0
        for ordinal in range(1, 19):
            prefix = PART_PREFIX + f"{ordinal:02d}/"
            self.part_states[ordinal] = {"manifest": None, "spool": None, "done": False}
            for role, basename in (("manifest", "MANIFEST.json"), ("payload", "critical_review_bytes.json.gz")):
                path = prefix + basename
                require(path in self.leaves and self.leaves[path]["mode"] == "100644" and
                        self.leaves[path]["size"] <= LEAF_CAP, "all_exact_part_leaves_required")
                self.part_roles[path] = (ordinal, role)
                if role == "payload":
                    workspace_bytes += self.leaves[path]["size"]
        require({path for path in self.leaves if path.startswith(PART_PREFIX)} == set(self.part_roles),
                "unexpected_part_namespace")
        for name, row in self.current.items():
            if name not in self.delta:
                path = FACTORY_PREFIX + name
                require(path in self.leaves and self.leaves[path] ==
                        {"mode": row["mode"], "type": "blob", "sha": row["git_blob"], "size": row["size"]},
                        "base_current_source_not_in_pinned_tree")
        self.stage = "fresh_capacity"
        free = self.capacity(EXPANDED_BYTES + workspace_bytes)
        self.capacity_report = {"initial_free_bytes": free, "expanded_source_bytes": EXPANDED_BYTES,
                                "maximum_individual_part_spool_union_bytes": workspace_bytes,
                                "fixed_reserve_bytes": RESERVE,
                                "container_memory_cap_bytes": CONTAINER_CAP,
                                "decoded_part_memory_cap_bytes": DECODED_PART_CAP}
        self.destination.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.workspace = Path(tempfile.mkdtemp(prefix=".rineya-restore-parts-", dir=str(self.destination.parent)))
        self.stage = "pinned_archive_full_bytes_and_capped_restore"
        process = subprocess.Popen(["gh", "api", "repos/" + REPOSITORY + "/tarball/" + self.args.final_commit],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            archive = verified_archive(process.stdout, self.leaves, archive_dirs, self.visit)
            require(process.stdout.read(1) == b"", "archive_http_stdout_not_eof")
            process.stdout.close()
            require(process.wait() == 0, "archive_native_gh_failure")
        except BaseException:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()
            raise
        require(all(state["done"] and state["spool"] is None for state in self.part_states.values()) and
                self.restored == set(self.current), "all_eighteen_parts_and_capped_files_required")
        self.workspace.rmdir()
        self.workspace = None
        self.stage = "all_raw_full_gzip_reconstruction"
        raw = self.raw_closure()
        self.stage = "final_full_namespace_hashes_permissions"
        final = self.final_namespace()
        self.stage = "private_repository_final_readback"
        private_repository()
        self.capacity()
        self.stage = "complete"
        return {"schema": "rineya-portable-restoration-report-v1", "status": "PASS_EXACT_CAPTURED_CUT_RESTORED",
                "repository": REPOSITORY, "pinned_data_commit": self.args.final_commit,
                "pinned_data_tree": self.commits[18]["tree"], "descriptor_sha256": DESCRIPTOR_SHA256,
                "destination": str(self.destination), "capacity": self.capacity_report,
                "fresh_archive_requests": 1, "fresh_per_blob_requests": 0, "native_gh_archive_exit": 0,
                "whole_pinned_archive": archive, "raw_closure": raw, "final_namespace": final,
                "later_born_portable_metadata_is_outside_captured_cut": True,
                "SOURCE06_stage_success": False, "CODE_DATA_science_publication_approval": False}


def main():
    parser = argparse.ArgumentParser(description="Restore the exact descriptor-pinned Rineya cut to a new directory.")
    parser.add_argument("--descriptor", required=True, help="Absolute path to the separately saved snapshot descriptor.")
    parser.add_argument("--descriptor-sha256", required=True, help="Exact externally checked descriptor SHA256.")
    parser.add_argument("--final-commit", required=True, help="Exact 40-character final data commit; no branch or latest alias.")
    parser.add_argument("--destination", required=True, help="Fresh absolute directory; existing paths are rejected.")
    parser.add_argument("--report", required=True, help="Fresh absolute report file outside the destination.")
    args = parser.parse_args()
    report_handle, restorer, result, exit_code = None, None, None, 1
    try:
        report = canonical_absolute(args.report, fresh=True)
        destination = canonical_absolute(args.destination, fresh=True)
        require(report != destination and destination not in report.parents and report not in destination.parents,
                "report_must_be_outside_destination")
        report_handle = exclusive_binary(report)
        restorer = Restorer(args, report)
        result = restorer.execute()
        exit_code = 0
    except BaseException as exc:
        result = {"schema": "rineya-portable-restoration-report-v1", "status": "FAILED_PARTIAL_RESTORE_PRESERVED",
                  "repository": REPOSITORY, "descriptor_sha256": DESCRIPTOR_SHA256,
                  "diagnostic_code": str(exc) if isinstance(exc, RestoreError) else "unexpected_local_or_runtime_error",
                  "exception_type": type(exc).__name__, "stage": restorer.stage if restorer else "fresh_local_paths",
                  "restored_file_count": len(restorer.restored) if restorer else 0,
                  "destination": str(restorer.destination) if restorer and restorer.destination else None,
                  "retained_part_workspace": str(restorer.workspace) if restorer and restorer.workspace else None,
                  "partial_destination_deleted": False, "existing_paths_overwritten": False,
                  "SOURCE06_stage_success": False, "CODE_DATA_science_publication_approval": False}
    finally:
        if report_handle is not None:
            result["at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            raw = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8")
            try:
                report_handle.write(raw)
                report_handle.flush()
                os.fsync(report_handle.fileno())
            except OSError:
                exit_code = 1
                result["status"] = "FAILED_REPORT_WRITE_RESTORE_PRESERVED"
            finally:
                report_handle.close()
    # No tracebacks, HTTP stderr, signed redirects, environment, or credentials.
    print(json.dumps({"status": result["status"], "exit_code": exit_code,
                      "report_created": report_handle is not None}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
