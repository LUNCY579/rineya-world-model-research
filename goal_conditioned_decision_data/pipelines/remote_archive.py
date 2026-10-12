"""GCDL-only cold custody: committed GitHub bytes before local evacuation.

The catalog is committed by the root backup workflow, which also creates the
immutable snapshot branch. This module never pushes, changes authentication,
materializes Git blobs, or deletes Git objects. Payload reads use RAM only.
GitHub's raw blob media type is documented at
https://docs.github.com/en/rest/git/blobs#get-a-blob .
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import stat
import subprocess
import time

REPOSITORY = 'LUNCY579/rineya-world-model-research'
OWNER = 'LUNCY579'
PROGRAMME = 'goal_conditioned_decision_data'
AUTHORIZATION = 'user-authorized-gcdl-only-2026-10-11'
INDEX_PATH = 'manifests/REMOTE_ARCHIVE.json'
MAX_READ_BYTES = 8 * 1024 ** 2
MAX_BLOB_BYTES = 96 * 1024 ** 2
MAX_BATCH_BYTES = 128 * 1024 ** 2
MAX_BATCH_FILES = 128
MAX_INDEX_BYTES = 8 * 1024 ** 2
RAM_CACHE_BYTES = 16 * 1024 ** 2
TREE_CACHE_BYTES = 4 * 1024 ** 2
CATALOG_CACHE_BYTES = 4 * 1024 ** 2
REQUEST_SECONDS = 20
OPERATION_SECONDS = 180
_CACHE = OrderedDict()
_CACHE_SIZE = 0
_READ_SESSION = ContextVar('gcdl_archive_read_session', default=None)
_HEX40 = re.compile(r'^[0-9a-f]{40}$')
_HEX64 = re.compile(r'^[0-9a-f]{64}$')
_PREFIXES = ('data/accepted/', 'data/sealed/', 'data/quarantined/', 'data/quarantine/',
             'data/demo/', 'data/demonstrations/', 'data/backups/', 'data/rendered/',
             'backups/', 'reports/review_history/', 'reports/history/', 'reports/evidence_history/')
_PROTECTED_REPORTS = {'reports/independent_review.md', 'reports/production_status.md', 'reports/quality_report.json'}
_CONTROL_NAMES = {'stop', 'paused.json', 'worker.lock', 'remote_archive.json', 'sealed_freeze.json', 'validated_code.json'}
_LEGACY = {'development_archive.tar.gz', 'development_archive.tar', 'development_archive.zip'}
_META = {'commit', 'git_blob_sha1', 'sha256', 'bytes'}


class ArchiveError(RuntimeError):
    pass


class ReadDeadlineExceeded(ArchiveError):
    """A bounded read/publication cycle exhausted its time allowance."""


class _ReadSession:
    def __init__(self, root, deadline):
        self.root = root; self.deadline = deadline; self._transport = None
        self.catalogs = OrderedDict(); self.catalog_bytes = 0
        self.local_control_facts = OrderedDict(); self.local_control_bytes = 0

    def check(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise ReadDeadlineExceeded('archive read/publication cycle deadline exceeded')

    @property
    def transport(self):
        self.check()
        if self._transport is None or time.monotonic() >= self._transport.deadline:
            self.catalogs.clear(); self.catalog_bytes = 0
            seconds = OPERATION_SECONDS if self.deadline is None else min(OPERATION_SECONDS, self.deadline-time.monotonic())
            self._transport = GitHubTransport(self.root, seconds=seconds)
        if self.deadline is not None:
            self._transport.deadline = min(self._transport.deadline, self.deadline)
        return self._transport


def check_read_deadline(root=None):
    session = _READ_SESSION.get()
    if session is not None:
        if root is not None and _root(root) != session.root:
            raise ArchiveError('archive read session root mismatch')
        session.check()


def read_transport(root, *, deadline=None):
    root = _root(root); check_read_deadline(root)
    if deadline is not None and time.monotonic() >= deadline:
        raise ReadDeadlineExceeded('archive read deadline exceeded')
    session = _READ_SESSION.get()
    if session is not None:
        transport=session.transport
    else:
        transport=GitHubTransport(root, seconds=OPERATION_SECONDS if deadline is None else min(OPERATION_SECONDS, deadline-time.monotonic()))
    if deadline is not None:transport.deadline=min(transport.deadline,deadline)
    return transport


@contextmanager
def read_session(root, *, deadline=None):
    """Share bounded RAM-only read metadata; evacuation proof stays separate.

    Mutable index/control files are still freshly validated for every read.
    Nested sessions must name the same programme and cannot extend the deadline.
    """
    root = _root(root)
    if deadline is not None and (type(deadline) not in (int,float) or not deadline < float('inf')):
        raise ArchiveError('invalid archive read session deadline')
    previous = _READ_SESSION.get()
    if previous is not None:
        if previous.root != root:
            raise ArchiveError('archive read session root mismatch')
        old_deadline = previous.deadline
        if deadline is not None:
            previous.deadline = deadline if old_deadline is None else min(old_deadline, deadline)
        try:
            previous.check(); yield previous
        finally:
            previous.deadline = old_deadline
        return
    session = _ReadSession(root, deadline); token = _READ_SESSION.set(session)
    try:
        session.check(); yield session
    finally:
        session.catalogs.clear(); session.catalog_bytes = 0
        session.local_control_facts.clear(); session.local_control_bytes = 0
        _READ_SESSION.reset(token)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def git_blob_sha1(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def _relative(path):
    value = str(path)
    parsed = PurePosixPath(value)
    if (not value or value.startswith('/') or '\\' in value or '\0' in value
            or any(char in value for char in '\r\n') or any(part in ('.', '..') for part in parsed.parts)
            or str(parsed) != value):
        raise ArchiveError('noncanonical archive path')
    return value


def is_payload_path(relative):
    relative = _relative(relative)
    if any(part.startswith('.') for part in PurePosixPath(relative).parts):
        return False
    lower = relative.casefold()
    if lower.endswith(('.py', '.pyc', '.pyo')) or PurePosixPath(lower).name in _CONTROL_NAMES:
        return False
    if lower in _PROTECTED_REPORTS:
        return False
    return relative.startswith('reports/') or relative in _LEGACY or any(relative.startswith(prefix) for prefix in _PREFIXES)


def _root(root):
    given = Path(root)
    if given.is_symlink():
        raise ArchiveError('programme root cannot be a symlink')
    root = given.resolve()
    if root.name != PROGRAMME or not root.is_dir():
        raise ArchiveError('archive scope is the GCDL programme directory only')
    return root


def _path(root, relative):
    relative = _relative(relative)
    current = root
    for index, part in enumerate(PurePosixPath(relative).parts):
        current = current / part
        try:
            details = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode):
            raise ArchiveError('archive path or ancestry is a symlink')
        leaf = index == len(PurePosixPath(relative).parts) - 1
        if leaf:
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise ArchiveError('archive file must be regular with one hardlink')
        elif not stat.S_ISDIR(details.st_mode):
            raise ArchiveError('archive ancestry is not a directory')
    if current.resolve() != current or root not in current.parents:
        raise ArchiveError('archive path escapes programme')
    return current


def _identity(details):
    return [details.st_dev, details.st_ino, details.st_mode, details.st_nlink,
            details.st_size, details.st_mtime_ns, details.st_ctime_ns]


def _ancestry(root, relative):
    path = _path(root, relative)
    parents = [root]
    for part in PurePosixPath(relative).parts[:-1]:
        parents.append(parents[-1] / part)
    return [[str(parent.relative_to(root)), parent.stat().st_dev, parent.stat().st_ino]
            for parent in parents]


def _local(root, relative, max_bytes=MAX_BLOB_BYTES, want_bytes=True):
    path = _path(root, relative)
    before_parents = _ancestry(root, relative)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > max_bytes:
            raise ArchiveError('local payload type/link/size bound failed')
        sha = hashlib.sha256(); oid = hashlib.sha1(b'blob ' + str(before.st_size).encode() + b'\0')
        chunks = [] if want_bytes else None
        total = 0
        while True:
            raw = os.read(fd, 65536)
            if not raw:
                break
            total += len(raw)
            if total > max_bytes:
                raise ArchiveError('local payload exceeded read bound')
            sha.update(raw); oid.update(raw)
            if want_bytes:
                chunks.append(raw)
        after = os.fstat(fd)
        now = _path(root, relative).lstat()
        if (_identity(before) != _identity(after) or _identity(before) != _identity(now)
                or total != before.st_size or before_parents != _ancestry(root, relative)):
            raise ArchiveError('local payload identity or ancestry changed during validation')
        meta = {'sha256': sha.hexdigest(), 'git_blob_sha1': oid.hexdigest(), 'bytes': total}
        return (b''.join(chunks) if want_bytes else None), meta, _identity(before), before_parents
    finally:
        os.close(fd)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError('duplicate archive JSON key')
        result[key] = value
    return result


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ArchiveError('nonfinite archive JSON')))
    except (ValueError, UnicodeError) as error:
        raise ArchiveError('invalid archive JSON') from error


def _metadata(meta):
    if (not isinstance(meta, dict) or set(meta) != _META
            or not _HEX40.fullmatch(str(meta.get('commit', '')))
            or not _HEX40.fullmatch(str(meta.get('git_blob_sha1', '')))
            or not _HEX64.fullmatch(str(meta.get('sha256', '')))
            or type(meta.get('bytes')) is not int or not 0 <= meta['bytes'] <= MAX_BLOB_BYTES):
        raise ArchiveError('invalid immutable payload metadata')


def _matches(raw, meta):
    if len(raw) != meta['bytes'] or hashlib.sha256(raw).hexdigest() != meta['sha256'] or git_blob_sha1(raw) != meta['git_blob_sha1']:
        raise ArchiveError('returned raw blob length/SHA256/Git object mismatch')


def _catalog(catalog):
    if (not isinstance(catalog, dict) or set(catalog) != {'version', 'repository', 'programme', 'files', 'catalog_id'}
            or catalog['version'] != 'gcdl-archive-catalog-1' or catalog['repository'] != REPOSITORY
            or catalog['programme'] != PROGRAMME or not isinstance(catalog['files'], dict)
            or not 1 <= len(catalog['files']) <= MAX_BATCH_FILES):
        raise ArchiveError('invalid archive catalog scope/shape')
    body = {key: value for key, value in catalog.items() if key != 'catalog_id'}
    if catalog['catalog_id'] != _digest(body):
        raise ArchiveError('archive catalog identity mismatch')
    for path, meta in catalog['files'].items():
        if not is_payload_path(path):
            raise ArchiveError('catalog contains protected code/control/live-state path')
        _metadata(meta)
    if sum(meta['bytes'] for meta in catalog['files'].values()) > MAX_BATCH_BYTES:
        raise ArchiveError('archive catalog exceeds bounded batch bytes')
    return catalog


def catalog_path(catalog_id):
    if not _HEX64.fullmatch(str(catalog_id)):
        raise ArchiveError('invalid catalog identity')
    return 'manifests/remote_catalogs/catalog-' + catalog_id + '.json'


def archive_ref(catalog_commit):
    if not _HEX40.fullmatch(str(catalog_commit)):
        raise ArchiveError('invalid snapshot commit')
    return 'refs/heads/archive/gcdl-001/' + catalog_commit


def _write(root, relative, raw, immutable=True):
    if len(raw) > MAX_INDEX_BYTES:
        raise ArchiveError('archive control exceeds bounded bytes')
    path = _path(root, relative)
    if path.exists() and immutable:
        if _local(root, relative, MAX_INDEX_BYTES)[0] != raw:
            raise ArchiveError('immutable archive control conflict')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _path(root, relative)
    temporary = relative + '.tmp'
    temp = _path(root, temporary)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        if immutable:
            try:
                os.link(temp, path)
            except FileExistsError:
                if _local(root, relative, MAX_INDEX_BYTES)[0] != raw:
                    raise ArchiveError('immutable archive control conflict')
        else:
            _path(root, relative)
            os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _run(root, command, deadline=None, env=None, max_output=MAX_READ_BYTES):
    remaining = REQUEST_SECONDS if deadline is None else min(REQUEST_SECONDS, deadline - time.monotonic())
    if remaining <= 0:
        raise ArchiveError('archive operation deadline exceeded')
    environment = os.environ.copy() if env is None else env.copy()
    environment['GIT_NO_LAZY_FETCH'] = '1'; environment['GIT_TERMINAL_PROMPT'] = '0'
    # Older Git (including the host's Apple Git 2.39) may not understand
    # GIT_NO_LAZY_FETCH. Empty protocol allowlist blocks every implicit fetch
    # at the transport boundary as well. All Git calls here are metadata-only.
    # https://git-scm.com/docs/git/2.36.0#Documentation/git.txt-GITALLOWPROTOCOL
    environment['GIT_ALLOW_PROTOCOL'] = ''; environment['GIT_PROTOCOL_FROM_USER'] = '0'
    environment.pop('GH_DEBUG', None)
    process = None; selector = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(command, cwd=root.parent, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        selector.register(process.stdout, selectors.EVENT_READ)
        limit = time.monotonic() + remaining; chunks = []; total = 0
        while selector.get_map():
            wait = limit - time.monotonic()
            if wait <= 0:
                raise ArchiveError('archive command deadline exceeded')
            for key, _ in selector.select(wait):
                raw = os.read(key.fileobj.fileno(), min(65536, max_output + 1 - total))
                if not raw:
                    selector.unregister(key.fileobj)
                    continue
                total += len(raw)
                if total > max_output:
                    raise ArchiveError('archive command response exceeded bounded bytes')
                chunks.append(raw)
        result = process.wait(timeout=max(0.001, limit - time.monotonic()))
        if result:
            raise ArchiveError('archive command failed; local payload retained')
        return b''.join(chunks)
    except (subprocess.TimeoutExpired, OSError):
        raise ArchiveError('archive command deadline or transport failure') from None
    finally:
        selector.close()
        if process is not None:
            if process.poll() is None:
                process.kill(); process.wait(timeout=2)
            process.stdout.close()


def _git_scope(root):
    top = Path(_run(root, ['git', 'rev-parse', '--show-toplevel']).decode().strip()).resolve()
    remote = _run(root, ['git', 'config', '--get', 'remote.origin.url']).decode().strip()
    if top != root.parent or remote not in ('https://github.com/' + REPOSITORY + '.git', 'git@github.com:' + REPOSITORY + '.git'):
        raise ArchiveError('archive Git repository is outside the owner programme scope')


def _git_file(root, relative, meta):
    _git_scope(root)
    path = PROGRAMME + '/' + relative
    tree = _run(root, ['git', 'ls-tree', '-z', meta['commit'], '--', path]).decode().split('\0')
    expected = [item for item in tree if item]
    if len(expected) != 1:
        raise ArchiveError('payload is not committed at its pinned source commit')
    head, tree_path = expected[0].split('\t', 1)
    mode, kind, oid = head.split()
    if tree_path != path or kind != 'blob' or mode not in ('100644', '100755') or oid != meta['git_blob_sha1']:
        raise ArchiveError('committed payload blob or mode mismatch')
    stage = _run(root, ['git', 'ls-files', '--stage', '-z', '--', path]).decode().split('\0')
    stage = [item for item in stage if item]
    if len(stage) != 1 or stage[0].split('\t', 1)[1] != path or stage[0].split()[1:3] != [oid, '0']:
        raise ArchiveError('untracked, staged-changed or conflicted payload refuses evacuation')
    # Local byte validation, not Git's stat cache/skip-worktree flag, proves
    # working bytes. Git metadata commands never fetch or cat-file a blob.
    diff = _run(root, ['git', 'diff', '--name-only', '--no-renames', meta['commit'], '--', path])
    if diff.strip():
        raise ArchiveError('changed committed payload refuses evacuation')


@contextmanager
def _lock(root, lock_fd=None):
    relative = 'state/worker.lock'
    path = _path(root, relative); path.parent.mkdir(parents=True, exist_ok=True)
    owned = lock_fd is None
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600) if owned else lock_fd
    try:
        details = os.fstat(fd)
        if _identity(details)[:4] != _identity(_path(root, relative).stat())[:4] or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ArchiveError('archive lock is not the live worker lock')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ArchiveError('worker/archive lock is held; local files retained') from None
        yield
    finally:
        if owned:
            os.close(fd)


class GitHubTransport:
    """Read-only, deadline-bounded gh API; owner credentials stay process-local."""
    def __init__(self, root, seconds=OPERATION_SECONDS):
        self.root = _root(root); self.deadline = time.monotonic() + min(seconds, OPERATION_SECONDS)
        self._env = None; self._trees = OrderedDict(); self._tree_bytes = 0

    def _environment(self):
        if self._env is None:
            env = os.environ.copy()
            for key in ('GH_TOKEN', 'GITHUB_TOKEN', 'GH_ENTERPRISE_TOKEN', 'GITHUB_ENTERPRISE_TOKEN', 'GH_DEBUG', 'GH_HOST'):
                env.pop(key, None)
            token = _run(self.root, ['gh', 'auth', 'token', '--user', OWNER, '--hostname', 'github.com'], self.deadline, env, max_output=4096).decode().strip()
            if not token or len(token) > 4096 or not re.fullmatch('[A-Za-z0-9_]+', token):
                raise ArchiveError('owner credential unavailable')
            env['GH_TOKEN'] = token; env['GH_PROMPT_DISABLED'] = '1'
            self._env = env
        return self._env

    def _api(self, suffix, max_bytes, raw=False):
        accept = 'application/vnd.github.raw+json' if raw else 'application/vnd.github+json'
        output = _run(self.root, ['gh', 'api', '--hostname', 'github.com', '--method', 'GET',
                                 'repos/' + REPOSITORY + '/' + suffix, '-H', 'Accept: ' + accept],
                      self.deadline, self._environment(), max_output=max_bytes)
        if len(output) > max_bytes:
            raise ArchiveError('GitHub response exceeded bounded bytes')
        return output if raw else _json(output)

    def fetch_ref(self, reference):
        if not re.fullmatch(r'refs/heads/archive/gcdl-001/[0-9a-f]{40}', reference):
            raise ArchiveError('unapproved archive snapshot ref')
        result = self._api('git/ref/' + reference[len('refs/'):], MAX_READ_BYTES)
        if result.get('ref') != reference or result.get('object', {}).get('type') != 'commit':
            raise ArchiveError('GitHub snapshot reference type/path mismatch')
        return result['object']['sha']

    def fetch_tree(self, commit):
        if not _HEX40.fullmatch(str(commit)):
            raise ArchiveError('invalid pinned commit')
        if commit in self._trees:
            mapping = self._trees.pop(commit); self._trees[commit] = mapping
            return mapping
        info = self._api('git/commits/' + commit, MAX_READ_BYTES)
        if info.get('sha') != commit or not _HEX40.fullmatch(str(info.get('tree', {}).get('sha', ''))):
            raise ArchiveError('GitHub commit identity mismatch')
        tree_sha = info['tree']['sha']
        top = self._api('git/trees/' + tree_sha, MAX_READ_BYTES)
        if top.get('sha') != tree_sha or top.get('truncated') is not False:
            raise ArchiveError('GitHub root tree identity/truncation mismatch')
        children = [entry for entry in top.get('tree', []) if entry.get('path') == PROGRAMME]
        if len(children) != 1 or children[0].get('type') != 'tree':
            raise ArchiveError('GitHub programme subtree missing')
        subtree_sha = children[0]['sha']
        result = self._api('git/trees/' + subtree_sha + '?recursive=1', MAX_READ_BYTES)
        if result.get('sha') != subtree_sha or result.get('truncated') is not False:
            raise ArchiveError('GitHub programme tree identity/truncation mismatch')
        mapping = {}
        for entry in result.get('tree', []):
            path = PROGRAMME + '/' + _relative(entry['path'])
            if path in mapping:
                raise ArchiveError('duplicate GitHub tree path')
            mapping[path] = entry
        size = len(_canonical(mapping).encode())
        if size <= TREE_CACHE_BYTES:
            while self._trees and self._tree_bytes + size > TREE_CACHE_BYTES:
                _, removed = self._trees.popitem(last=False)
                self._tree_bytes -= len(_canonical(removed).encode())
            self._trees[commit] = mapping; self._tree_bytes += size
        return mapping

    def fetch_blob(self, oid, max_bytes=MAX_BLOB_BYTES):
        if not _HEX40.fullmatch(str(oid)) or not 0 <= max_bytes <= MAX_BLOB_BYTES:
            raise ArchiveError('invalid bounded blob request')
        return self._api('git/blobs/' + oid, max_bytes, raw=True)


def _tree_entry(transport, commit, relative, meta):
    entry = transport.fetch_tree(commit).get(PROGRAMME + '/' + relative)
    if (not entry or entry.get('type') != 'blob' or entry.get('mode') not in ('100644', '100755')
            or entry.get('sha') != meta['git_blob_sha1'] or type(entry.get('size')) is not int or entry.get('size') != meta['bytes']):
        raise ArchiveError('GitHub committed tree path/OID/size mismatch')
    return entry


def _tree_blob(transport, commit, relative, meta):
    _tree_entry(transport, commit, relative, meta)
    raw = transport.fetch_blob(meta['git_blob_sha1'], max_bytes=meta['bytes'])
    _matches(raw, meta)
    return raw


def _empty_index():
    return {'version': 'gcdl-remote-archive-1', 'repository': REPOSITORY, 'programme': PROGRAMME,
            'files': {}, 'catalogs': {}, 'receipts': {}}


def _receipt(receipt):
    if (not isinstance(receipt, dict) or set(receipt) != {'version', 'status', 'repository', 'programme', 'catalog_id',
            'catalog_commit', 'archive_ref', 'catalog_path', 'catalog_sha256', 'catalog_git_blob_sha1', 'files', 'local_identities', 'local_ancestry', 'receipt_id'}
            or receipt['version'] != 'gcdl-archive-custody-1' or receipt['status'] != 'remote_verified'
            or receipt['repository'] != REPOSITORY or receipt['programme'] != PROGRAMME
            or receipt['archive_ref'] != archive_ref(receipt['catalog_commit'])
            or receipt['catalog_path'] != catalog_path(receipt['catalog_id'])
            or not _HEX64.fullmatch(str(receipt['catalog_sha256']))
            or not _HEX40.fullmatch(str(receipt['catalog_git_blob_sha1']))
            or not isinstance(receipt['files'], dict) or not 1 <= len(receipt['files']) <= MAX_BATCH_FILES
            or not isinstance(receipt['local_identities'], dict)
            or not isinstance(receipt['local_ancestry'], dict) or set(receipt['local_ancestry']) != set(receipt['local_identities'])
            or receipt['receipt_id'] != _digest({key: value for key, value in receipt.items() if key != 'receipt_id'})):
        raise ArchiveError('invalid verified archive custody receipt')
    for path, meta in receipt['files'].items():
        if not is_payload_path(path):
            raise ArchiveError('custody receipt contains protected path')
        _metadata(meta)
    for path, identity in receipt['local_identities'].items():
        if path not in receipt['files'] or not isinstance(identity, list) or len(identity) != 7 or any(type(value) is not int for value in identity) or identity[3] != 1 or identity[4] != receipt['files'][path]['bytes']:
            raise ArchiveError('invalid verified local identity')
        parents = receipt['local_ancestry'][path]
        expected_paths = ['.'] + ['/'.join(PurePosixPath(path).parts[:index]) for index in range(1, len(PurePosixPath(path).parts))]
        if (not isinstance(parents, list) or len(parents) != len(expected_paths)
                or any(not isinstance(parent, list) or len(parent) != 3 or parent[0] != expected_path
                       or any(type(value) is not int or value < 0 for value in parent[1:])
                       for parent, expected_path in zip(parents, expected_paths))):
            raise ArchiveError('invalid verified local parent identities')
    if sum(meta['bytes'] for meta in receipt['files'].values()) > MAX_BATCH_BYTES:
        raise ArchiveError('custody receipt exceeds bounded batch bytes')
    return receipt


def _local_control_value(root, body, validator):
    """Reuse only a successful structural fact for exact freshly read bytes.

    JSON objects are always newly decoded. All local file/hash and index binding
    checks stay with the caller; these bounded session facts supply no custody.
    """
    parser = _json
    value = parser(body)
    session = _READ_SESSION.get()
    if session is None:
        return validator(value)
    if session.root != root:
        raise ArchiveError('archive read session root mismatch')
    session.check()
    if type(body) is not bytes or len(body) > CATALOG_CACHE_BYTES or MAX_BATCH_FILES < 1:
        return validator(value)
    key = (validator, parser, body)
    facts = session.local_control_facts
    if key in facts:
        facts.move_to_end(key)
        return value
    value = validator(value)  # A failure is never retained.
    while facts and (len(facts) >= MAX_BATCH_FILES or session.local_control_bytes + len(body) > CATALOG_CACHE_BYTES):
        removed, _ = facts.popitem(last=False)
        session.local_control_bytes -= len(removed[2])
    facts[key] = True
    session.local_control_bytes += len(body)
    return value


def load_index(root):
    root = _root(root)
    check_read_deadline(root)
    if not _path(root, INDEX_PATH).exists():
        return _empty_index()
    raw = _local(root, INDEX_PATH, MAX_INDEX_BYTES)[0]; index = _json(raw)
    if (not isinstance(index, dict) or set(index) != {'version', 'repository', 'programme', 'files', 'catalogs', 'receipts', 'digest'}
            or index['version'] != 'gcdl-remote-archive-1' or index['repository'] != REPOSITORY or index['programme'] != PROGRAMME
            or index['digest'] != _digest({key: value for key, value in index.items() if key != 'digest'})
            or any(not isinstance(index[key], dict) for key in ('files', 'catalogs', 'receipts'))):
        raise ArchiveError('corrupt remote archive index')
    snapshot = 'manifests/remote_indexes/index-' + index['digest'] + '.json'
    if _local(root, snapshot, MAX_INDEX_BYTES)[0] != raw:
        raise ArchiveError('archive index missing its immutable snapshot')
    catalogs, receipts = {}, {}
    for identifier, descriptor in index['catalogs'].items():
        check_read_deadline(root)
        if (not isinstance(descriptor, dict) or set(descriptor) != {'catalog_commit', 'archive_ref', 'catalog_path', 'catalog_sha256', 'catalog_git_blob_sha1'}
                or descriptor['catalog_path'] != catalog_path(identifier) or descriptor['archive_ref'] != archive_ref(descriptor['catalog_commit'])
                or not _HEX64.fullmatch(str(descriptor['catalog_sha256'])) or not _HEX40.fullmatch(str(descriptor['catalog_git_blob_sha1']))):
            raise ArchiveError('corrupt archive catalog custody pointer')
        body = _local(root, descriptor['catalog_path'], MAX_INDEX_BYTES)[0]
        if hashlib.sha256(body).hexdigest() != descriptor['catalog_sha256'] or git_blob_sha1(body) != descriptor['catalog_git_blob_sha1']:
            raise ArchiveError('local committed catalog commitment mismatch')
        catalogs[identifier] = _local_control_value(root, body, _catalog)
        if catalogs[identifier]['catalog_id'] != identifier:
            raise ArchiveError('catalog pointer identity mismatch')
    for identifier, descriptor in index['receipts'].items():
        check_read_deadline(root)
        if (not isinstance(descriptor, dict) or set(descriptor) != {'path', 'sha256'} or not _HEX64.fullmatch(str(identifier))
                or not _HEX64.fullmatch(str(descriptor['sha256'])) or descriptor['path'] != 'manifests/remote_custody/custody-' + identifier + '.json'):
            raise ArchiveError('invalid custody receipt pointer')
        body = _local(root, descriptor['path'], MAX_INDEX_BYTES)[0]
        if hashlib.sha256(body).hexdigest() != descriptor['sha256']:
            raise ArchiveError('custody receipt hash mismatch')
        receipts[identifier] = _local_control_value(root, body, _receipt)
        if receipts[identifier]['receipt_id'] != identifier:
            raise ArchiveError('custody pointer identity mismatch')
        receipt = receipts[identifier]; cat = catalogs.get(receipt['catalog_id'])
        if (cat is None or cat['files'] != receipt['files']
                or any(receipt[key] != value for key, value in index['catalogs'][receipt['catalog_id']].items())):
            raise ArchiveError('orphan or unbound custody receipt')
    for path, entry in index['files'].items():
        if not is_payload_path(path) or set(entry) != _META | {'catalog_id', 'receipt_id'}:
            raise ArchiveError('archive index contains protected or malformed payload')
        meta = {key: entry[key] for key in _META}; _metadata(meta)
        cat, receipt = catalogs.get(entry['catalog_id']), receipts.get(entry['receipt_id'])
        if (cat is None or receipt is None or cat['files'].get(path) != meta or receipt['files'].get(path) != meta
                or receipt['catalog_id'] != entry['catalog_id']
                or any(receipt[key] != value for key, value in index['catalogs'][entry['catalog_id']].items())):
            raise ArchiveError('unbound archive index payload/custody entry')
    check_read_deadline(root); return index


def deletion_custody(root, relative):
    root = _root(root); relative = _relative(relative)
    if not is_payload_path(relative):
        raise ArchiveError('protected path has no deletion custody')
    index = load_index(root); entry = index['files'].get(relative)
    if entry is None:
        raise ArchiveError('payload has no verified committed archive custody')
    return {**entry, **index['catalogs'][entry['catalog_id']], 'archived': True}


def is_archived(root, relative):
    root = _root(root); return _relative(relative) in load_index(root)['files']


def payload_info(root, relative):
    root = _root(root); relative = _relative(relative); path = _path(root, relative)
    index = load_index(root); entry = index['files'].get(relative)
    if path.exists():
        _, meta, _, _ = _local(root, relative, MAX_BLOB_BYTES, False)
        if entry and any(meta[key] != entry[key] for key in ('sha256', 'bytes', 'git_blob_sha1')):
            raise ArchiveError('local payload differs from archived immutable bytes')
        if not entry:
            return {**meta, 'archived': False}
    elif not entry:
        raise FileNotFoundError(str(path))
    return {**entry, **index['catalogs'][entry['catalog_id']], 'archived': True}


def exists(root, relative):
    root = _root(root); relative = _relative(relative)
    check_read_deadline(root)
    if _path(root, relative).exists():
        return True
    return relative in load_index(root)['files']


def iter_paths(root, pattern):
    root = _root(root); pattern = str(pattern)
    if pattern.startswith('/') or '..' in PurePosixPath(pattern).parts or '\\' in pattern:
        raise ArchiveError('unsafe archive inventory pattern')
    found = set()
    for path in root.glob(pattern):
        relative = str(path.relative_to(root))
        checked = _path(root, relative)
        if checked.is_file():
            found.add(relative)
    found.update(path for path in load_index(root)['files'] if fnmatch.fnmatchcase(path, pattern))
    return sorted(found)


def _remote_entry(root, relative, index, transport):
    entry = index['files'][relative]; descriptor = index['catalogs'][entry['catalog_id']]
    if transport.fetch_ref(descriptor['archive_ref']) != descriptor['catalog_commit']:
        raise ArchiveError('immutable archive snapshot ref changed or missing')
    catalog_meta = {'commit': descriptor['catalog_commit'], 'git_blob_sha1': descriptor['catalog_git_blob_sha1'],
                    'sha256': descriptor['catalog_sha256'], 'bytes': len(_local(root, descriptor['catalog_path'], MAX_INDEX_BYTES)[0])}
    session = _READ_SESSION.get()
    if session is not None and transport is not session._transport:
        session = None  # Explicit providers keep their own verification boundary.
    # Catalog proofs are reused only inside this read session, for this exact
    # provider and immutable commit/blob/hash tuple. Ref checks remain fresh.
    key = (transport, REPOSITORY, descriptor['catalog_commit'], descriptor['catalog_path'],
           descriptor['catalog_git_blob_sha1'], descriptor['catalog_sha256'], catalog_meta['bytes'])
    body = session.catalogs.pop(key, None) if session is not None else None
    if body is None:
        body = _tree_blob(transport, descriptor['catalog_commit'], descriptor['catalog_path'], catalog_meta)
        if session is not None and len(body) <= CATALOG_CACHE_BYTES:
            while session.catalogs and session.catalog_bytes + len(body) > CATALOG_CACHE_BYTES:
                _, removed = session.catalogs.popitem(last=False); session.catalog_bytes -= len(removed)
            session.catalogs[key] = body; session.catalog_bytes += len(body)
    else:
        session.catalogs[key] = body; _matches(body, catalog_meta)
    catalog = _catalog(_json(body))
    meta = {key: entry[key] for key in _META}
    if catalog['files'].get(relative) != meta:
        raise ArchiveError('remote catalog does not bind requested payload')
    _tree_entry(transport, descriptor['catalog_commit'], relative, meta)
    raw = _tree_blob(transport, meta['commit'], relative, meta)
    if transport.fetch_ref(descriptor['archive_ref']) != descriptor['catalog_commit']:
        raise ArchiveError('immutable archive snapshot ref changed during read')
    return raw


def read_bytes(root, relative, *, max_bytes=MAX_READ_BYTES, allow_sealed=False, transport=None):
    global _CACHE_SIZE
    root = _root(root); relative = _relative(relative); check_read_deadline(root)
    if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_BLOB_BYTES:
        raise ArchiveError('invalid bounded read size')
    if relative.startswith('data/sealed/') and not allow_sealed:
        raise ArchiveError('sealed raw read requires explicit custody/release authorization')
    if _path(root, relative).exists():
        raw, meta, _, _ = _local(root, relative, max_bytes)
        entry = load_index(root)['files'].get(relative)
        if entry and any(meta[key] != entry[key] for key in ('sha256', 'bytes', 'git_blob_sha1')):
            raise ArchiveError('local archive payload corruption')
        check_read_deadline(root); return raw
    index = load_index(root); entry = index['files'].get(relative)
    if entry is None:
        raise FileNotFoundError(str(root / relative))
    if entry['bytes'] > max_bytes:
        raise ArchiveError('remote payload exceeds caller read bound')
    key = (str(root), relative, _digest(entry), _digest(index['catalogs'][entry['catalog_id']]))
    if key in _CACHE:
        raw = _CACHE.pop(key); _CACHE[key] = raw; _matches(raw, entry)
        check_read_deadline(root); return raw
    try:
        raw = _remote_entry(root, relative, index, transport if transport is not None else read_transport(root))
    except ArchiveError:
        check_read_deadline(root)
        raise
    check_read_deadline(root)
    if len(raw) <= RAM_CACHE_BYTES:
        while _CACHE and _CACHE_SIZE + len(raw) > RAM_CACHE_BYTES:
            _, removed = _CACHE.popitem(last=False); _CACHE_SIZE -= len(removed)
        _CACHE[key] = raw; _CACHE_SIZE += len(raw)
    return raw


def read_text(root, relative, *, max_bytes=MAX_READ_BYTES, allow_sealed=False, transport=None):
    return read_bytes(root, relative, max_bytes=max_bytes, allow_sealed=allow_sealed, transport=transport).decode('utf-8')


def plan_archive(root, paths=None, commit=None, *, max_files=MAX_BATCH_FILES, max_bytes=MAX_BATCH_BYTES, lock_fd=None):
    root = _root(root)
    if type(max_files) is not int or not 1 <= max_files <= MAX_BATCH_FILES or type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BATCH_BYTES:
        raise ArchiveError('invalid archive plan bounds')
    with _lock(root, lock_fd):
        _git_scope(root)
        commit = commit or _run(root, ['git', 'rev-parse', 'HEAD']).decode().strip()
        if not _HEX40.fullmatch(str(commit)):
            raise ArchiveError('archive plan requires immutable commit SHA')
        selected = sorted(_relative(path) for path in paths) if paths is not None else sorted(
            str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() and is_payload_path(str(path.relative_to(root))))
        if len(set(selected)) != len(selected):
            raise ArchiveError('duplicate selected archive payload')
        files, total = {}, 0
        for relative in selected:
            if not is_payload_path(relative):
                raise ArchiveError('archive plan selected protected code/control/live state')
            _, meta, _, _ = _local(root, relative, MAX_BLOB_BYTES, False)
            if len(files) >= max_files or total + meta['bytes'] > max_bytes:
                if paths is not None:
                    raise ArchiveError('explicit archive selection exceeds bounded batch')
                break
            meta['commit'] = commit; _git_file(root, relative, meta)
            files[relative] = meta; total += meta['bytes']
        if not files:
            raise ArchiveError('no committed local payload selected')
        body = {'version': 'gcdl-archive-catalog-1', 'repository': REPOSITORY, 'programme': PROGRAMME, 'files': files}
        catalog = {**body, 'catalog_id': _digest(body)}; _catalog(catalog)
        _write(root, catalog_path(catalog['catalog_id']), (_canonical(catalog) + '\n').encode())
        return catalog


def _remote_proof(root, catalog, catalog_commit, transport, *, required_blob_ids=None):
    """Prove custody; only owned Git compaction may select present raw blobs.

    The catalog and every snapshot/source tree entry remain fully checked.
    Public archive verification and eviction omit the selector and therefore
    verify complete returned bytes for every payload.
    """
    if required_blob_ids is not None:
        if (type(required_blob_ids) not in (set, frozenset)
                or any(type(oid) is not str or not _HEX40.fullmatch(oid) for oid in required_blob_ids)):
            raise ArchiveError('invalid required archive blob selector')
        required_blob_ids = frozenset(required_blob_ids)
        if not required_blob_ids.issubset({meta['git_blob_sha1'] for meta in catalog['files'].values()}):
            raise ArchiveError('required archive blob selector is outside catalog')
    reference = archive_ref(catalog_commit)
    if transport.fetch_ref(reference) != catalog_commit:
        raise ArchiveError('archive snapshot ref is missing or not immutable commit')
    raw = (_canonical(catalog) + '\n').encode()
    meta = {'commit': catalog_commit, 'git_blob_sha1': git_blob_sha1(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    _tree_blob(transport, catalog_commit, catalog_path(catalog['catalog_id']), meta)
    for relative, expected in catalog['files'].items():
        # The durable snapshot itself retains the exact payload tree entry.
        # An unreferenced source commit is insufficient remote custody.
        _tree_entry(transport, catalog_commit, relative, expected)
        if required_blob_ids is None or expected['git_blob_sha1'] in required_blob_ids:
            _tree_blob(transport, expected['commit'], relative, expected)
        else:
            _tree_entry(transport, expected['commit'], relative, expected)
    if transport.fetch_ref(reference) != catalog_commit:
        raise ArchiveError('archive snapshot ref changed during raw-byte proof')
    return {'catalog_commit': catalog_commit, 'archive_ref': reference, 'catalog_path': catalog_path(catalog['catalog_id']),
            'catalog_sha256': meta['sha256'], 'catalog_git_blob_sha1': meta['git_blob_sha1']}


def verify_archive(root, catalog, catalog_commit, *, transport=None, lock_fd=None):
    root = _root(root); catalog = _catalog(deepcopy_json(catalog))
    with _lock(root, lock_fd):
        old = load_index(root)
        identities, ancestry = {}, {}
        for relative, meta in catalog['files'].items():
            if _path(root, relative).exists():
                _git_file(root, relative, meta)
                _, local_meta, identity, parents = _local(root, relative, MAX_BLOB_BYTES, False)
                if any(local_meta[key] != meta[key] for key in ('sha256', 'bytes', 'git_blob_sha1')):
                    raise ArchiveError('local payload changed before archive verification')
                identities[relative] = identity
                ancestry[relative] = parents
            elif relative not in old['files'] or any(old['files'][relative][key] != meta[key] for key in _META):
                raise ArchiveError('orphan or missing unverified local payload')
        descriptor = _remote_proof(root, catalog, catalog_commit, transport or GitHubTransport(root))
        # Detect writes/inode replacement while remote proof was in flight.
        for relative, identity in identities.items():
            _, local_meta, current, parents = _local(root, relative, MAX_BLOB_BYTES, False)
            if current != identity or parents != ancestry[relative] or any(local_meta[key] != catalog['files'][relative][key] for key in ('sha256', 'bytes', 'git_blob_sha1')):
                raise ArchiveError('local payload changed after remote verification')
        body = {'version': 'gcdl-archive-custody-1', 'status': 'remote_verified', 'repository': REPOSITORY,
                'programme': PROGRAMME, 'catalog_id': catalog['catalog_id'], **descriptor,
                'files': catalog['files'], 'local_identities': identities, 'local_ancestry': ancestry}
        receipt = {**body, 'receipt_id': _digest(body)}; _receipt(receipt)
        receipt_relative = 'manifests/remote_custody/custody-' + receipt['receipt_id'] + '.json'
        receipt_raw = (_canonical(receipt) + '\n').encode()
        _write(root, descriptor['catalog_path'], (_canonical(catalog) + '\n').encode())
        _write(root, receipt_relative, receipt_raw)
        index = {key: deepcopy_json(value) for key, value in old.items() if key != 'digest'}
        prior = index['catalogs'].get(catalog['catalog_id'])
        if prior is not None and prior != descriptor:
            raise ArchiveError('immutable catalog custody cannot be rebound')
        index['catalogs'][catalog['catalog_id']] = descriptor
        index['receipts'][receipt['receipt_id']] = {'path': receipt_relative, 'sha256': hashlib.sha256(receipt_raw).hexdigest()}
        for relative, meta in catalog['files'].items():
            index['files'][relative] = {**meta, 'catalog_id': catalog['catalog_id'], 'receipt_id': receipt['receipt_id']}
        index['digest'] = _digest(index)
        index_raw = (_canonical(index) + '\n').encode()
        _write(root, 'manifests/remote_indexes/index-' + index['digest'] + '.json', index_raw)
        _write(root, INDEX_PATH, index_raw, immutable=False)
        load_index(root)
        return receipt


def deepcopy_json(value):
    return _json(_canonical(value).encode())


def _before_unlink(root, relative):
    """Test seam for adversarial races; production has no callback behavior."""


def _before_move(root, relative):
    """Test seam immediately before the anchored atomic move."""


def _check_stop(root):
    if _path(root, 'state/STOP').exists():
        raise ArchiveError('STOP prevents archive evacuation; remaining local bytes retained')


def _open_parent(root, relative):
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(root, flags)
    try:
        for part in PurePosixPath(relative).parts[:-1]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd); fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _held_meta(directory_fd, name):
    fd = os.open(name, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0), dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_BLOB_BYTES:
            raise ArchiveError('holding file type/link/size mismatch; bytes retained')
        sha = hashlib.sha256(); oid = hashlib.sha1(b'blob ' + str(before.st_size).encode() + b'\0')
        total = 0
        while True:
            raw = os.read(fd, 65536)
            if not raw:
                break
            total += len(raw)
            if total > MAX_BLOB_BYTES:
                raise ArchiveError('holding file read bound exceeded')
            sha.update(raw); oid.update(raw)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _identity(before) != _identity(os.fstat(fd)) or _identity(before) != _identity(current) or total != before.st_size:
            raise ArchiveError('holding file changed during validation; bytes retained')
        return {'sha256': sha.hexdigest(), 'git_blob_sha1': oid.hexdigest(), 'bytes': total}, _identity(before)
    finally:
        os.close(fd)


def _holding_path(relative, receipt_id):
    parent = str(PurePosixPath(relative).parent)
    name = '.gcdl-evacuate-' + receipt_id + '-' + hashlib.sha256(relative.encode()).hexdigest()[:16]
    return (parent + '/' if parent != '.' else '') + name + '/payload'


def _journal(root, relative, value):
    _write(root, relative, (_canonical(value) + '\n').encode(), immutable=False)


def _restore_held(parent_fd, holding_fd, basename):
    # link fails if another writer already created the source pathname. Never
    # overwrite either file; leave the holding pointer in the journal instead.
    os.link('payload', basename, src_dir_fd=holding_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
    os.unlink('payload', dir_fd=holding_fd)


def _recover_pending(root, receipt, journal):
    recovered = {}
    if not _path(root, journal).exists():
        return recovered
    prior = _json(_local(root, journal, MAX_INDEX_BYTES)[0])
    if prior.get('receipt_id') != receipt['receipt_id']:
        raise ArchiveError('evacuation journal custody mismatch')
    pending = prior.get('pending')
    if pending is None:
        return recovered
    if (not isinstance(pending, dict) or set(pending) != {'relative', 'holding', 'identity', 'parents'}
            or pending['relative'] not in receipt['files']
            or pending['holding'] != _holding_path(pending['relative'], receipt['receipt_id'])
            or pending['identity'] != receipt['local_identities'].get(pending['relative'])
            or pending['parents'] != receipt['local_ancestry'].get(pending['relative'])):
        raise ArchiveError('invalid interrupted evacuation pointer')
    relative = pending['relative']; holding = pending['holding']
    if _ancestry(root, relative) != pending['parents']:
        raise ArchiveError('interrupted evacuation ancestry changed; bytes retained')
    parent_fd = _open_parent(root, relative)
    holding_fd = None
    try:
        directory = str(PurePosixPath(holding).parent)
        directory_name = PurePosixPath(directory).name
        try:
            holding_fd = os.open(directory_name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_NOFOLLOW', 0), dir_fd=parent_fd)
        except FileNotFoundError:
            # A caught failure restored the original then removed its holding
            # directory; the durable pending marker may predate that cleanup.
            if not _path(root, relative).exists():
                prior.pop('pending'); _journal(root, journal, prior)
                return recovered
            _, meta, identity, _ = _local(root, relative, MAX_BLOB_BYTES, False)
            expected = receipt['files'][relative]
            if identity[:6] != pending['identity'][:6] or any(meta[key] != expected[key] for key in ('sha256', 'bytes', 'git_blob_sha1')):
                raise ArchiveError('restored pending payload differs from verified inode/bytes')
            recovered[relative] = identity
            # Keep the immutable-original-identity marker until successful
            # remote proof and evacuation. Offline retries must retain the
            # custody handoff explaining the rename-induced ctime change.
            return recovered
        try:
            meta, moved = _held_meta(holding_fd, 'payload')
        except FileNotFoundError:
            # Crash after marker but before move, or after the final unlink.
            pass
        else:
            expected = receipt['files'][relative]
            if moved[:6] != pending['identity'][:6] or any(meta[key] != expected[key] for key in ('sha256', 'bytes', 'git_blob_sha1')):
                raise ArchiveError('interrupted holding file changed; every byte retained')
            _restore_held(parent_fd, holding_fd, PurePosixPath(relative).name)
            recovered[relative] = _identity(os.stat(PurePosixPath(relative).name, dir_fd=parent_fd, follow_symlinks=False))
        os.rmdir(directory_name, dir_fd=parent_fd)
        if not _path(root, relative).exists():
            prior.pop('pending')
        _journal(root, journal, prior)
    except OSError as error:
        raise ArchiveError('interrupted evacuation cannot restore safely; bytes retained') from error
    finally:
        if holding_fd is not None:
            os.close(holding_fd)
        os.close(parent_fd)
    return recovered


def _move_then_unlink(root, relative, receipt, expected_identity, parents, journal, deleted, deletion_guard=None):
    """Unlink only an atomically isolated, reverified inode using directory FDs.

    The worker lock serializes programme writers. Owned mode-0700 holding dirs
    and anchored FDs avoid pathname replacement/ancestry races. A hostile process
    with the same UID can still alter open files; that OS trust boundary is not
    a claim of protection from an adversarial user account.
    """
    holding = _holding_path(relative, receipt['receipt_id'])
    directory = str(PurePosixPath(holding).parent); name = PurePosixPath(directory).name
    parent_fd = _open_parent(root, relative); holding_fd = None; moved = False
    basename = PurePosixPath(relative).name
    try:
        if _ancestry(root, relative) != parents:
            raise ArchiveError('payload ancestry changed before atomic move')
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        holding_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_NOFOLLOW', 0), dir_fd=parent_fd)
        marker = {'receipt_id': receipt['receipt_id'], 'verified_remote': True,
                  'deleted': deleted, 'status': 'evacuating', 'pending': {
                      'relative': relative, 'holding': holding, 'identity': receipt['local_identities'][relative],
                      'parents': receipt['local_ancestry'][relative]}}
        _journal(root, journal, marker)
        _before_move(root, relative)
        _check_stop(root)
        if deletion_guard is not None:deletion_guard()
        if _ancestry(root, relative) != parents:
            raise ArchiveError('payload ancestry changed before atomic move')
        os.rename(basename, 'payload', src_dir_fd=parent_fd, dst_dir_fd=holding_fd)
        moved = True
        local_meta, current = _held_meta(holding_fd, 'payload')
        expected = receipt['files'][relative]
        # rename changes ctime; inode, mode, links, size and mtime must match.
        if (current[:6] != expected_identity[:6] or _ancestry(root, relative) != parents
                or any(local_meta[key] != expected[key] for key in ('sha256', 'bytes', 'git_blob_sha1'))):
            raise ArchiveError('atomically moved file mismatch; bytes retained')
        _check_stop(root)
        if deletion_guard is not None:deletion_guard()
        # There is no pathname unlink of the original payload after a check.
        os.unlink('payload', dir_fd=holding_fd)
        moved = False
        os.rmdir(name, dir_fd=parent_fd)
    except BaseException:
        if moved:
            try:
                _restore_held(parent_fd, holding_fd, basename)
                moved = False
            except OSError:
                pass  # Original and holding files both survive a conflict.
        if not moved and holding_fd is not None:
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if holding_fd is not None:
            os.close(holding_fd)
        os.close(parent_fd)


def evict_verified(root, receipt, *, transport=None, lock_fd=None, deletion_guard=None):
    root = _root(root); receipt = _receipt(deepcopy_json(receipt))
    with _lock(root, lock_fd):
        _check_stop(root)
        index = load_index(root)
        if receipt['receipt_id'] not in index['receipts']:
            raise ArchiveError('evacuation receipt is not bound in immutable index')
        journal = 'state/archive_evacuations/' + receipt['receipt_id'] + '.json'
        recovered = _recover_pending(root, receipt, journal)
        catalog = _catalog(_json(_local(root, receipt['catalog_path'], MAX_INDEX_BYTES)[0]))
        descriptor = _remote_proof(root, catalog, receipt['catalog_commit'], transport or GitHubTransport(root))
        _check_stop(root)
        if deletion_guard is not None:deletion_guard()
        if any(receipt[key] != value for key, value in descriptor.items()):
            raise ArchiveError('evacuation custody pointer mismatch')
        present, ancestry = [], {}
        # Whole batch validation completes before the first unlink.
        for relative, meta in receipt['files'].items():
            custody = deletion_custody(root, relative)
            if custody['receipt_id'] != receipt['receipt_id'] or any(custody[key] != meta[key] for key in _META):
                raise ArchiveError('payload no longer belongs to verified evacuation')
            if not _path(root, relative).exists():
                continue  # Verified prior interruption/cold sparse absence.
            _git_file(root, relative, meta)
            _, local_meta, identity, parents = _local(root, relative, MAX_BLOB_BYTES, False)
            if (identity != recovered.get(relative, receipt['local_identities'].get(relative))
                    or parents != receipt['local_ancestry'].get(relative)
                    or any(local_meta[key] != meta[key] for key in ('sha256', 'bytes', 'git_blob_sha1'))):
                raise ArchiveError('payload hash/inode changed since verified custody')
            present.append(relative); ancestry[relative] = parents
        deleted = []
        for relative in present:
            _check_stop(root)
            _before_unlink(root, relative)
            _check_stop(root)
            _, local_meta, identity, parents = _local(root, relative, MAX_BLOB_BYTES, False)
            expected = receipt['files'][relative]
            if (identity != recovered.get(relative, receipt['local_identities'][relative]) or parents != ancestry[relative]
                    or any(local_meta[key] != expected[key] for key in ('sha256', 'bytes', 'git_blob_sha1'))):
                raise ArchiveError('immediate pre-unlink hash/identity/ancestry mismatch; remaining files retained')
            if deletion_guard is None:
                _move_then_unlink(root, relative, receipt, identity, parents, journal, deleted)
            else:
                deletion_guard()
                _move_then_unlink(root, relative, receipt, identity, parents, journal, deleted,deletion_guard=deletion_guard)
            deleted.append(relative)
            _write(root, journal, (_canonical({'receipt_id': receipt['receipt_id'], 'verified_remote': True,
                                              'deleted': deleted, 'status': 'evacuating'}) + '\n').encode(), immutable=False)
        result = {'status': 'evacuated', 'receipt_id': receipt['receipt_id'], 'archive_ref': receipt['archive_ref'],
                  'deleted': deleted, 'deleted_bytes': sum(receipt['files'][path]['bytes'] for path in deleted),
                  'catalog_commit': receipt['catalog_commit'], 'Git_objects_removed': False}
        _write(root, journal, (_canonical(result) + '\n').encode(), immutable=False)
        return result


def maybe_archive(root, config, *, lock_fd=None, transport=None):
    root = _root(root)
    if config.get('archive_enabled') is not True or config.get('archive_authorization') != AUTHORIZATION:
        return None
    trigger = config.get('archive_trigger_free_bytes', 10 * 1024 ** 3)
    if type(trigger) is not int or trigger != 10 * 1024 ** 3:
        raise ArchiveError('archive trigger must preserve the 10 GiB programme floor')
    if shutil.disk_usage(root).free >= trigger:
        return None
    index = load_index(root)
    receipt_ids = sorted({entry['receipt_id'] for path, entry in index['files'].items() if _path(root, path).exists()})
    if not receipt_ids:
        return {'status': 'backup_required', 'deleted': [], 'reason': 'no committed verified local payload custody'}
    identifier = receipt_ids[0]  # One bounded transaction per worker invocation.
    receipt = _receipt(_json(_local(root, index['receipts'][identifier]['path'], MAX_INDEX_BYTES)[0]))
    return evict_verified(root, receipt, transport=transport, lock_fd=lock_fd)
