"""Own metadata/opaque-byte reader; never imports reviewed sources or decodes gzip."""
import datetime
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).absolute().parents[2]
HERE = pathlib.Path(__file__).absolute().parent
P = "backup_logs/NEXT_CHECKPOINT_653_EXECUTION_PREPARATION_ATTEMPT03/"
INPUTS = {
    P + "ALIAS_CREATOR_SOURCE_READY_R2.json": "c2b564b4bf303eeed98709314681995b39ee696f794c68f8e7f12f10ed176aff",
    P + "create_exact_oversized_literal_alias_manifests_r2.py": "6cf1d2fac779a0e2f90f0699589c3dac3f29aa0cc446ede3610c5504d54fe554",
    P + "EXACT_EIGHT_ALIAS_MANIFEST_PLAN_R2.json": "b235ad6d015803dcc1411b32d22f525ec3f6c0b5e9884df9bd164c3cd6b90501",
    P + "ALIAS_CREATOR_SOURCE_PREPARATION_FREEZE_R2.json": "abee90148f86c956035a5afbe11e2a318e9255867a5e63a86d8992c92ecc13bf",
    P + "ALIAS_CREATOR_R2_MINIMAL.diff": "24865c94fad3f23cdbc633a4488bcd579b92806ac9a82ebd98ebf900f361be0a",
    "quality_reports/RSM_OVERSIZED_LITERAL_ALIAS_ROOT_DIAGNOSIS_ATTEMPT01.json": "3977e24b0ee8c22e325bd4482e252d88617311f846f700f1fdfdca904dc95b29",
}
PREFIXES = (
    "quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT02/preexecution_raw_archives/",
    "quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT02/raw_source_archives/",
    "quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/preexecution_raw_archives/",
    "quality_reports/RSM_CORE_INSTALLATION_ATTEMPT01/raw_source_archives/",
)
TUPLES = {
    "38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c": 41453404,
    "7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c": 40541020,
}

def safe(rel):
    assert type(rel) is str and rel and "\\" not in rel
    p = pathlib.PurePosixPath(rel)
    assert not p.is_absolute() and p.as_posix() == rel and not any(x in (".", "..", ".git") for x in p.parts)
    for n in range(1, len(p.parts) + 1):
        assert not ROOT.joinpath(*p.parts[:n]).is_symlink(), rel
    return ROOT / rel

def binding(rel):
    p = safe(rel)
    before = p.stat()
    assert p.is_file()
    h = hashlib.sha256()
    n = 0
    with p.open("rb") as stream:
        while chunk := stream.read(65536):
            h.update(chunk)
            n += len(chunk)
    after = p.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    return {"path": rel, "size": n, "sha256": h.hexdigest()}

def main():
    expected_source = sys.argv[1]
    me = binding(pathlib.Path(__file__).absolute().relative_to(ROOT).as_posix())
    assert me["sha256"] == expected_source
    checked = []
    for rel, sha in INPUTS.items():
        got = binding(rel)
        assert got["sha256"] == sha, rel
        checked.append(got)
    freeze = json.loads(safe(P + "ALIAS_CREATOR_SOURCE_PREPARATION_FREEZE_R2.json").read_bytes())
    plan = json.loads(safe(P + "EXACT_EIGHT_ALIAS_MANIFEST_PLAN_R2.json").read_bytes())
    cache = {r["path"]: r for r in checked}
    frozen = freeze["frozen_inputs"]
    assert len(frozen) == len({r["path"] for r in frozen})
    for row in frozen:
        assert set(row) == {"path", "size", "sha256"} and type(row["size"]) is int
        got = cache.get(row["path"]) or binding(row["path"])
        assert got == row, row["path"]
        cache[row["path"]] = got
    expected = {p + sha + ".source": (size, sha) for p in PREFIXES for sha, size in TUPLES.items()}
    rows = plan["mappings"]
    assert len(rows) == 8 and {r["raw"]["path"] for r in rows} == set(expected)
    assert len({r["new_manifest_path"] for r in rows}) == 8
    seen_origins = set()
    for row in rows:
        raw = row["raw"]
        assert (raw["size"], raw["sha256"]) == expected[raw["path"]]
        origin = row["existing_origin"]
        assert origin["raw"]["path"] == "quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/raw_source_archives/" + raw["sha256"] + ".source"
        assert (origin["raw"]["size"], origin["raw"]["sha256"]) == (raw["size"], raw["sha256"])
        for part in (raw, *origin.values()):
            assert cache[part["path"]] == part
        manifest = json.loads(safe(origin["manifest"]["path"]).read_bytes())
        for role in ("raw", "gzip"):
            for suffix, field in (("path", "path"), ("bytes", "size"), ("sha256", "sha256")):
                assert type(manifest[role + "_" + suffix]) is type(origin[role][field]) and manifest[role + "_" + suffix] == origin[role][field]
        seen_origins.add(origin["gzip"]["path"])
        dest = row["new_manifest_path"]
        assert dest == "manifests/lossless_archives/rsm-installed-runtime-aliases-r1/" + hashlib.sha256(raw["path"].encode()).hexdigest() + "-" + raw["sha256"] + ".lossless.json"
        assert not safe(dest).exists()
    for row in plan["frozen_inputs"]:
        assert cache[row["path"]] == row
    for name in ("receipt", "failure"):
        assert not safe(plan[name]).exists()
    assert safe(P + "source_blobs/" + INPUTS[P + "create_exact_oversized_literal_alias_manifests_r2.py"] + ".source").read_bytes() == safe(P + "create_exact_oversized_literal_alias_manifests_r2.py").read_bytes()
    assert safe(P + "source_blobs/" + INPUTS[P + "EXACT_EIGHT_ALIAS_MANIFEST_PLAN_R2.json"] + ".source").read_bytes() == safe(P + "EXACT_EIGHT_ALIAS_MANIFEST_PLAN_R2.json").read_bytes()
    result = {"status": "PASS_METADATA_OPAQUE_BYTES_ONLY", "actual_reader_execution": "/root/formal_oracle", "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "own_source": me, "checked_inputs": checked, "frozen_inputs_checked": frozen, "counts": {"top_bindings": len(checked), "frozen_bindings": len(frozen), "plan_frozen_bindings": len(plan["frozen_inputs"]), "literal_aliases": len(rows), "distinct_reused_gzips": len(seen_origins), "literal_alias_raw_bytes": sum(r["raw"]["size"] for r in rows), "fresh_manifest_destinations": len(rows)}, "source_raw_archive_equal": True, "plan_raw_archive_equal": True, "reviewed_source_imports_or_executions": 0, "gzip_decodes": 0, "global_inventory_scan": False, "approval": False}
    out = HERE / "STATIC_BINDING_READBACK.json"
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, sort_keys=True))

if __name__ == "__main__":
    main()
