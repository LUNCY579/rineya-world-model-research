"""Generator reference routing; distinct from the formal option interpreter.

Prepared for factory/option_resume_reference.py.  The producer's original
implementation remains at its immutable original path and is preserved as raw
source by the release.  Routing supplies no review or data-approval decision.
"""
import hashlib
import importlib.util
import os
import stat

from factory.common import ROOT

AUTHOR_EXECUTION = '/root'
REFERENCE_AUTHOR_EXECUTION = '/root/relational_critic_producer'
REFERENCE_PATH = 'generators/skills/prospective_option_resumption_attempt01/reference_attempt01.py'
REFERENCE_SHA256 = 'e1a3141157b4cdac8fa00a510f2040996d589c575faf1fc03683477851491ae0'
_REFERENCE = None


def _read_reference():
    fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    leaf = None
    try:
        parts = REFERENCE_PATH.split('/')
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        before = os.fstat(leaf)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('RSM producer reference must be a regular source file')
        with os.fdopen(leaf, 'rb') as stream:
            leaf = None
            raw = stream.read(25 * 1024 * 1024 + 1)
            after = os.fstat(stream.fileno())
        if len(raw) > 25 * 1024 * 1024 or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('RSM producer reference changed during its bounded read')
        if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
            raise ValueError('RSM producer reference source differs from its frozen version')
        return raw
    finally:
        if leaf is not None:
            os.close(leaf)
        os.close(fd)


def reference():
    global _REFERENCE
    raw = _read_reference()
    if _REFERENCE is None:
        filename = str(ROOT / REFERENCE_PATH)
        module = importlib.util.module_from_spec(importlib.util.spec_from_file_location('rsm_frozen_producer_reference', filename))
        # Execute the exact verified bytes, avoiding a second path read at import.
        exec(compile(raw, filename, 'exec'), module.__dict__)
        _REFERENCE = module
    if _REFERENCE.__file__ != str(ROOT / REFERENCE_PATH):
        raise ValueError('RSM producer reference original location differs')
    _read_reference()
    return _REFERENCE


def solve(task):
    result = reference().solve(task)
    _read_reference()
    return result
