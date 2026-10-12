"""Small deterministic persistence helpers."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as out:
        out.write(canonical(value) + '\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)

def immutable_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    try:
        os.link(temp, path)  # Atomic publish, without replacing accepted bytes.
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError('immutable artifact conflict: ' + str(path))
    finally:
        temp.unlink(missing_ok=True)

def source_hashes():
    paths = sorted(ROOT.rglob('*.py')) + sorted((ROOT / 'configs').glob('*.json'))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
