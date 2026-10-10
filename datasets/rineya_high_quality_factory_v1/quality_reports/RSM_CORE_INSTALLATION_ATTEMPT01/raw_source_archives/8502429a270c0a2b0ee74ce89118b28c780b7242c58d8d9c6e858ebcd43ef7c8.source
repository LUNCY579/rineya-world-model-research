"""Standard-library atomic storage and canonical hashing."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_json(path, value):
    atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    atomic(path, "".join(canonical(row) + "\n" for row in rows))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def record_hash(record):
    return digest({k: v for k, v in record.items() if k != "record_hash"})


def split_for(index):
    # Assigned at semantic-family creation, BEFORE twins/rendering.
    return "dev_challenge" if index % 10 >= 8 else "train"


def record(track, batch, index, variant, task, prompt, target, tags, difficulty,
           target_format="structured", renderer="forward"):
    family = f"{track}-{batch}-p{index:04d}"
    r = dict(sample_id=f"{family}-{variant}-{renderer}", semantic_parent_id=family,
             semantic_family_id=family, track_id=track, skill_tags=tags,
             difficulty=difficulty, generator_version=VERSION,
             oracle_version="independent-1.0.0", generation_seed=20261009,
             source_provenance={"origin": "original_synthetic", "license": "repository-private-research",
                                "sealed_evaluation_used": False},
             semantic_task=task, rendered_input=prompt, verified_target=target,
             target_format=target_format, split=split_for(index), review_status="PENDING",
             review_receipt_ids=[], renderer=renderer)
    r["record_hash"] = record_hash(r)
    return r


def candidate_paths(batch):
    return sorted(ROOT.glob(f"data/*/candidate_renderings/{batch}.jsonl"))
