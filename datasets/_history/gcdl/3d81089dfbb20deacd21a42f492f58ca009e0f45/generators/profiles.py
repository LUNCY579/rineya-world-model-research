"""Trusted generation profiles over immutable, versioned curriculum archives.

Record text selects an exact known identity, never a filename or executable.
The registry itself is pinned in reviewed code; both raw archive bytes and
canonical configuration values are authenticated on every resolution.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from ..pipelines.common import digest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_VERSION = 'generation-profiles-1.0'
# Updated only with the allowlisted registry in the same reviewed source change.
REGISTRY_SHA256 = 'f54978838729e85eb849724d2b2c512da65b9969b8bd4e9876aab7271cf9388a'
MAX_CONFIG_BYTES = 1024 * 1024
IDENTITY_FIELDS = ('config_version', 'config_hash', 'generator_version', 'environment_version')
_IDENTITIES = {
    'v1': ('curriculum-v1.json', 'curriculum-1.0', 'generator-1.0'),
    'v2': ('curriculum-v2.json', 'curriculum-2.0', 'generator-2.0'),
}
_ENTRY_FIELDS = {'config_file', 'config_version', 'config_hash', 'config_file_sha256',
                 'generator_version', 'environment_version', 'schema_version'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate generation profile/config key')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('nonfinite generation profile/config value: ' + value)


def _read_json(path, expected_hash, description):
    if path.is_symlink() or not path.is_file():
        raise ValueError(description + ' must be a regular, nonsymlink file')
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError(description + ' exceeds bounded configuration size')
    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(description + ' file hash mismatch')
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(description + ' is not valid UTF-8 JSON') from error
    if type(value) is not dict:
        raise ValueError(description + ' must be an object')
    return value


def _registry(root):
    base = ROOT if root is None else Path(root)
    base = base.resolve()
    directory = base / 'configs'
    if directory.resolve() != directory:
        raise ValueError('generation profile configuration directory cannot be a symlink')
    registry = _read_json(directory / 'generation_profiles.json', REGISTRY_SHA256,
                          'generation profile registry')
    if set(registry) != {'version', 'profiles'} or registry['version'] != REGISTRY_VERSION:
        raise ValueError('unsupported generation profile registry')
    entries = registry['profiles']
    if type(entries) is not dict or set(entries) != set(_IDENTITIES):
        raise ValueError('generation profile registry must contain exactly v1 and v2')
    identities = set()
    for name, entry in entries.items():
        if type(entry) is not dict or set(entry) != _ENTRY_FIELDS:
            raise ValueError('generation profile entry shape mismatch')
        filename, config_version, generator_version = _IDENTITIES[name]
        if (entry['config_file'] != filename or entry['config_version'] != config_version
                or entry['generator_version'] != generator_version
                or entry['environment_version'] != 'environments-1.0'
                or entry['schema_version'] != 'gcdl-1.0'):
            raise ValueError('unsupported generation profile identity or path')
        for key in ('config_hash', 'config_file_sha256'):
            if type(entry[key]) is not str or re.fullmatch('[a-f0-9]{64}', entry[key]) is None:
                raise ValueError('malformed generation profile digest')
        identity = tuple(entry[key] for key in IDENTITY_FIELDS)
        if identity in identities:
            raise ValueError('duplicate generation profile identity')
        identities.add(identity)
    return base, entries


def _resolved(base, name, entry):
    path = base / 'configs' / entry['config_file']
    config = _read_json(path, entry['config_file_sha256'], 'archived curriculum')
    if (digest(config) != entry['config_hash'] or config.get('version') != entry['config_version']
            or config.get('generator_version') != entry['generator_version']):
        raise ValueError('archived curriculum canonical hash/version mismatch')
    if type(config.get('split_templates')) is not dict or set(config['split_templates']) != {'train', 'dev', 'sealed', 'ood'}:
        raise ValueError('archived curriculum split reservations mismatch')
    return {'id': name, 'schema_version': entry['schema_version'],
            'generator_version': entry['generator_version'],
            'environment_version': entry['environment_version'],
            'config_version': entry['config_version'], 'config_hash': entry['config_hash'],
            'config': deepcopy(config), 'config_path': str(path),
            'config_file_sha256': entry['config_file_sha256']}


def resolve_profile(profile_id='v1', *, root=None):
    """Resolve an operator-selected known profile; legacy default stays v1.

    ``root`` is an operator/test override for a complete authenticated archive
    checkout. It is never selected from record provenance or model input.
    """
    if type(profile_id) is not str or profile_id not in _IDENTITIES:
        raise ValueError('unknown generation profile id')
    base, entries = _registry(root)
    return _resolved(base, profile_id, entries[profile_id])


def resolve_record_profile(provenance, *, root=None):
    """Resolve exactly the four recorded version/hash fields, without fallback."""
    if type(provenance) is not dict or any(type(provenance.get(key)) is not str for key in IDENTITY_FIELDS):
        raise ValueError('malformed generation profile provenance')
    identity = tuple(provenance[key] for key in IDENTITY_FIELDS)
    base, entries = _registry(root)
    matches = [name for name, entry in entries.items()
               if tuple(entry[key] for key in IDENTITY_FIELDS) == identity]
    if len(matches) != 1:
        raise ValueError('unknown generation profile provenance identity')
    name = matches[0]
    return _resolved(base, name, entries[name])
