"""One authorized dedicated-worktree sparse expansion, with exact custody evidence."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
FACTORY = Path(__file__).resolve().parents[1]
ORIGINAL = FACTORY.parent / "rineya-world-model-research"
WORKTREE = FACTORY.parent / ".rineya-dataset-git-worktree"
PREFIX = "datasets/rineya_high_quality_factory_v1"
HEAD = "b46d96a5db0f1cc94e894f15d61942af49e61819"
ORIGINAL_HEAD = "a223884d55ff47baf3890c7d4a82054d07223215"
BRANCH = "research/rineya-high-quality-dataset-v1"
REPOSITORY = "LUNCY579/rineya-world-model-research"
ROLE = "/root/backup_manager"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(root, *args, raw=False):
    result = subprocess.run(list(args), cwd=root, check=True, capture_output=True)
    return result.stdout if raw else result.stdout.decode().strip()


def git(root, *args, raw=False):
    return command(root, "git", *args, raw=raw)


def custody(root):
    return {"path": str(root), "head": git(root, "rev-parse", "HEAD"),
            "tree": git(root, "rev-parse", "HEAD^{tree}"),
            "branch": git(root, "branch", "--show-current"),
            "status": git(root, "status", "--porcelain=v1", "--untracked-files=all"),
            "index_entries_sha256": sha(git(root, "ls-files", "-s", "-z", raw=True)),
            "index_flags_sha256": sha(git(root, "ls-files", "-t", "-z", raw=True))}


def check_guards():
    original, dedicated = custody(ORIGINAL), custody(WORKTREE)
    if (original["head"] != ORIGINAL_HEAD or original["branch"] != "main"
            or original["status"] or dedicated["head"] != HEAD
            or dedicated["branch"] != BRANCH or dedicated["status"]):
        raise ValueError("Exact clean original/dedicated custody guard failed")
    if git(WORKTREE, "remote", "get-url", "origin") != f"https://github.com/{REPOSITORY}.git":
        raise ValueError("Dedicated remote identity changed")
    if git(WORKTREE, "config", "--get", "extensions.worktreeConfig") != "true":
        raise ValueError("Worktree-specific configuration is not already enabled")
    if git(WORKTREE, "config", "--get", "index.sparse") != "false":
        raise ValueError("Dedicated index must remain full")
    for key in ["core.sparseCheckout", "core.sparseCheckoutCone"]:
        if git(WORKTREE, "config", "--get", key) != "true":
            raise ValueError("Expected existing cone sparse checkout unavailable")
    metadata = json.loads(command(FACTORY, "gh", "repo", "view", REPOSITORY,
                                  "--json", "isPrivate,viewerPermission,nameWithOwner"))
    if metadata != {"isPrivate": True, "nameWithOwner": REPOSITORY, "viewerPermission": "ADMIN"}:
        raise ValueError("Actual GitHub private ADMIN identity guard failed")
    if git(WORKTREE, "ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}").split() != [HEAD, f"refs/heads/{BRANCH}"]:
        raise ValueError("Actual remote branch no longer matches authorized baseline")
    return original, dedicated, metadata


def save_json(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def flags(root):
    return {record[2:].decode(): record[:1].decode()
            for record in git(root, "ls-files", "-t", "-z", raw=True).split(b"\0") if record}


def main():
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = FACTORY / "backup_logs" / ("SPARSE_DATASET_RESTORE_" + timestamp)
    evidence.mkdir()
    saved = evidence / "pre_bytes"
    saved.mkdir()
    receipt = {"author_execution": ROLE, "runner_execution": ROLE, "started_at_utc": now(),
               "execution_role": "Independent GitHub Integrity and Backup Manager",
               "model_disclosure": "Same-model agent execution: backup_manager inherited root model; independence refers to separate execution, not model diversity",
               "status": "PENDING", "execution_source_sha256": sha(Path(__file__).read_bytes()),
               "authorized_operation": ["git", "sparse-checkout", "set", "--cone", "--no-sparse-index", "tools", PREFIX],
               "scope": "Only reversible sparse expansion of root-owned dedicated clean backup worktree plus new own backup_logs evidence. No source/review inventory freeze, copy, commit, push, deletion, force, reset, original-clone mutation, or data approval"}
    mutation_started = False
    try:
        original, dedicated, metadata = check_guards()
        if git(WORKTREE, "sparse-checkout", "list") != "tools" or (WORKTREE / PREFIX).exists():
            raise ValueError("Expected tools-only baseline/dataset absence changed")
        original_git = Path(git(ORIGINAL, "rev-parse", "--absolute-git-dir"))
        dedicated_git = Path(git(WORKTREE, "rev-parse", "--absolute-git-dir"))
        common = Path(git(WORKTREE, "rev-parse", "--git-common-dir"))
        metadata_files = {
            "common_config": common / "config", "common_HEAD": common / "HEAD",
            "original_index": original_git / "index", "original_worktree_config": original_git / "config.worktree",
            "dedicated_git_pointer": WORKTREE / ".git", "dedicated_HEAD": dedicated_git / "HEAD",
            "dedicated_index": dedicated_git / "index", "dedicated_worktree_config": dedicated_git / "config.worktree",
            "dedicated_sparse_rules": dedicated_git / "info/sparse-checkout",
            "dedicated_commondir": dedicated_git / "commondir", "dedicated_gitdir": dedicated_git / "gitdir",
            "main_ref": common / "refs/heads/main", "dedicated_ref": common / "refs/heads" / BRANCH,
            "remote_ref": common / "refs/remotes/origin" / BRANCH, "packed_refs": common / "packed-refs",
        }
        loaded = importlib.util.spec_from_file_location("existing_checkpoint", FACTORY / "backup_logs/checkpoint_backup.py")
        helper = importlib.util.module_from_spec(loaded)
        loaded.loader.exec_module(helper)
        pre_files = {}
        for label, path in metadata_files.items():
            if not path.exists():
                pre_files[label] = {"path": str(path), "exists": False}
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("Unexpected symlink/non-file metadata")
            payload = path.read_bytes()
            if any(pattern.search(payload) for pattern in helper.SECRET_PATTERNS):
                raise ValueError("Metadata secret scan failed; content withheld")
            target = saved / (label + ".bin")
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            pre_files[label] = {"path": str(path), "exists": True, "size": len(payload),
                                "sha256": sha(payload), "preserved_bytes": str(target.relative_to(FACTORY))}
        raw_tree = git(WORKTREE, "ls-tree", "-r", "-l", "-z", HEAD, "--", PREFIX, raw=True)
        records = []
        for record in raw_tree.split(b"\0"):
            if record:
                meta, path = record.split(b"\t", 1)
                mode, kind, blob, size = meta.decode().split()
                if kind != "blob" or mode not in {"100644", "100755"}:
                    raise ValueError("Unexpected dataset tree object mode")
                records.append({"path": path.decode(), "mode": mode, "git_blob": blob, "size": int(size)})
        if len({item["path"] for item in records}) != len(records):
            raise ValueError("Duplicate committed dataset path")
        checkout_bytes = sum(item["size"] for item in records)
        growth_allowance = 256 * 1024 * 1024
        next_inventory_budget = checkout_bytes + growth_allowance
        required = checkout_bytes + 2 * next_inventory_budget + 32 * 1024 * 1024 + 512 * 1024 * 1024
        available = shutil.disk_usage(WORKTREE).free
        pre_flags = flags(WORKTREE)
        recovered = FACTORY / "backup_logs/belief-v3-final-review_RECOVERY_VERIFIED_RECEIPT.json"
        recovery = json.loads(recovered.read_bytes())
        if recovery.get("status") != "BACKED_UP" or recovery.get("commit_sha") != HEAD or recovery.get("remote_commit_sha") != HEAD:
            raise ValueError("Actual completed b46 recovery receipt unavailable")
        expected = {f"{PREFIX}/{entry['path']}": entry for entry in recovery["inventory"]}
        if set(expected) != {item["path"] for item in records}:
            raise ValueError("Frozen recovered inventory differs from exact dataset commit tree")
        receipt.update({"pre_original_custody": original, "pre_dedicated_custody": dedicated,
                        "private_repository_evidence": metadata, "pre_metadata": pre_files,
                        "frozen_dataset_tree_sha256": sha(raw_tree), "frozen_dataset_files": len(records),
                        "dataset_checkout_logical_bytes": checkout_bytes,
                        "recovery_receipt_sha256": sha(recovered.read_bytes()),
                        "helper_source_sha256": sha((FACTORY / "backup_logs/checkpoint_backup.py").read_bytes()),
                        "capacity_guard": {"available_bytes_before": available, "required_before_bytes": required,
                            "next_inventory_budget_bytes": next_inventory_budget, "growth_allowance_bytes": growth_allowance,
                            "future_helper_prerequisite_bytes": 2 * next_inventory_budget + 32 * 1024 * 1024,
                            "additional_reserve_bytes": 512 * 1024 * 1024,
                            "scope": "Conservative provisional capacity budget; actual future frozen inventory and free space must be checked again before the later helper runs"}})
        save_json(evidence / "PRE_EXECUTION.json", receipt)
        if available < required:
            receipt["status"] = "PENDING_INSUFFICIENT_SPACE_NO_MUTATION"
            receipt["finished_at_utc"] = now()
            save_json(evidence / "RECEIPT.json", receipt)
            print(json.dumps({"status": receipt["status"], "evidence": str(evidence)}))
            return
        # Recheck exact custody and free capacity directly before the single mutation.
        original_now, dedicated_now, _ = check_guards()
        if original_now != original or dedicated_now != dedicated:
            raise ValueError("Custody changed after immutable pre-execution capture")
        if shutil.disk_usage(WORKTREE).free < required:
            raise ValueError("Runtime capacity changed below expansion/future-copy reserve")
        mutation_started = True
        result = subprocess.run(receipt["authorized_operation"], cwd=WORKTREE, capture_output=True, timeout=180)
        receipt["git_sparse_execution"] = {"returncode": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}
        if result.returncode:
            raise RuntimeError("Authorized sparse expansion failed; exact failure preserved without rollback/deletion")
        original_after, dedicated_after, metadata_after = check_guards()
        if original_after != original:
            raise ValueError("Original custody/index entries or flags changed")
        for key in ["head", "tree", "branch", "status", "index_entries_sha256"]:
            if dedicated_after[key] != dedicated[key]:
                raise ValueError("Dedicated committed/full-index custody changed unexpectedly")
        if set(git(WORKTREE, "sparse-checkout", "list").splitlines()) != {"tools", PREFIX}:
            raise ValueError("Sparse expansion scope does not match authorized paths")
        post_flags = flags(WORKTREE)
        if set(pre_flags) != set(post_flags):
            raise ValueError("Full index path coverage changed")
        changes = {path: {"before": before, "after": post_flags[path]}
                   for path, before in pre_flags.items() if before != post_flags[path]}
        if set(changes) != set(expected) or any(item != {"before": "S", "after": "H"} for item in changes.values()):
            raise ValueError("Unexpected skip-worktree flag changes outside exact dataset expansion")
        post_files = {}
        for label, path in metadata_files.items():
            post = {"path": str(path), "exists": path.exists()}
            if path.exists():
                data = path.read_bytes()
                post.update({"size": len(data), "sha256": sha(data)})
            post_files[label] = post
            if label not in {"dedicated_index", "dedicated_sparse_rules"}:
                old = pre_files[label]
                if post["exists"] != old["exists"] or (post["exists"] and post["sha256"] != old["sha256"]):
                    raise ValueError("Metadata outside authorized sparse/index files changed")
        file_proofs = []
        for item in records:
            path = WORKTREE / item["path"]
            if path.is_symlink() or not path.is_file():
                raise ValueError("Restored committed path is absent or a symlink")
            digest, git_digest, length = hashlib.sha256(), hashlib.sha1(), 0
            git_digest.update(f"blob {item['size']}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(64 * 1024), b""):
                    length += len(chunk)
                    digest.update(chunk)
                    git_digest.update(chunk)
            if (length != item["size"] or git_digest.hexdigest() != item["git_blob"]
                    or digest.hexdigest() != expected[item["path"]]["sha256"]):
                raise ValueError("Restored bytes differ from committed verified checkpoint")
            file_proofs.append({**item, "sha256": digest.hexdigest(), "verified": True})
        receipt.update({"status": "SPARSE_DATASET_EXPANDED_AND_VERIFIED", "finished_at_utc": now(),
                        "post_original_custody": original_after, "post_dedicated_custody": dedicated_after,
                        "post_private_repository_evidence": metadata_after, "post_metadata": post_files,
                        "skip_worktree_changes": len(changes), "post_sparse_paths": ["tools", PREFIX],
                        "restored_actual_file_proofs": file_proofs,
                        "available_bytes_after": shutil.disk_usage(WORKTREE).free,
                        "unchanged_boundaries": "Original HEAD/branch/clean status/index entries/index flags/index bytes and common configuration/refs/HEAD remained exact; dedicated HEAD/branch/clean status/full-index entries remained exact. Only dedicated sparse rules and index metadata changed, plus restored tracked committed dataset checkout bytes. New recovery/future source files were not copied or inventoried."})
        save_json(evidence / "RECEIPT.json", receipt)
        print(json.dumps({"status": receipt["status"], "files": len(records), "bytes": checkout_bytes,
                          "free_bytes_after": receipt["available_bytes_after"], "evidence": str(evidence)}))
    except Exception as error:
        receipt.update({"status": "FAILED_EXECUTION_PRESERVED" if mutation_started else "PENDING_GUARD_FAILURE_NO_MUTATION",
                        "finished_at_utc": now(), "error_type": type(error).__name__, "error": str(error),
                        "mutation_started": mutation_started})
        save_json(evidence / "FAILURE.json", receipt)
        raise


if __name__ == "__main__":
    main()
