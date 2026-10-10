"""Bind the frozen, independently authored planning controls to publication.

No producer targets or review judgments enter the input-derived family audit.
"""
import hashlib
import json
from pathlib import Path

from oracles import rgm_planning_controls_independent as checker

CHECKER_SHA256 = 'cd92cff9bb92eb38afd5a989109718cd600390a9609a13fc070f9cc1e1677bf3'
PRIOR_PATH = 'research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl'
PRIOR_SHA256 = '896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d'
PUBLIC_BINDINGS = {
    'schemas/SEMANTICS.md': checker.SEMANTICS_SHA256,
    'research/curricula/RGM-NEXT-GAP-PROSPECTIVE.md': '6100d43fe1cf83d798bb34799fac70569f54d33f66692150d188581d9c940761',
    'schemas/RGM_PLANNING_CONTROL_PROPERTIES.md': 'f591747cc1b6d0199b76311b5b8283666bf219589f534b42ef195c7b81a7be3a',
    'schemas/RGM_PLANNING_MANIFEST_ENCODING_SUPPLEMENT.md': checker.ENCODING_SUPPLEMENT_SHA256,
}
PLANNING_CONTEXTS = {stratum + str(index) for stratum in 'ABCD' for index in range(4)} | {'C4'}


def planning_control_rows(rows, batch=None):
    selected = []
    for row in rows:
        if row.get('track_id') != 'RGM-001' or row.get('semantic_task', {}).get('kind') not in {'graph', 'blocks'}:
            continue
        version = row.get('generator_version', '')
        tags = {part for tag in row.get('skill_tags', []) if isinstance(tag, str) for part in tag.split(':')}
        if ((isinstance(version, str) and version.startswith('rgm-planning-gap-'))
                or tags.intersection(checker.PROPERTIES)
                or row.get('controlled_context') in PLANNING_CONTEXTS
                or row.get('primary_stratum') in {'A', 'B', 'C', 'D'}
                or (isinstance(batch, str) and batch.startswith('planning-gap-v'))):
            selected.append(row)
    return selected


def planning_control_report(rows, batch, root):
    selected = planning_control_rows(rows, batch)
    if not selected:
        return None
    root = Path(root)
    sources = {str(Path(checker.__file__).resolve()): CHECKER_SHA256}
    sources.update({str(root / path): sha for path, sha in PUBLIC_BINDINGS.items()})
    manifest_path = root / 'data/RGM-001/manifests' / (batch + '.family_controls.json')
    prior_path = root / PRIOR_PATH
    evidence = {}
    errors = []
    for path, expected in sources.items():
        try:
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            evidence[path] = actual
            if actual != expected:
                errors.append('frozen planning source binding mismatch: ' + path)
        except OSError as exc:
            errors.append('missing planning source: ' + path + ': ' + str(exc))
    manifest = None
    prior = None
    try:
        raw = manifest_path.read_bytes()
        evidence[str(manifest_path)] = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
    except (OSError, ValueError) as exc:
        errors.append('missing or malformed planning family manifest: ' + str(exc))
    try:
        raw = prior_path.read_bytes()
        evidence[str(prior_path)] = hashlib.sha256(raw).hexdigest()
        if evidence[str(prior_path)] != PRIOR_SHA256:
            errors.append('frozen prior planning input inventory binding mismatch')
        else:
            prior = [json.loads(line) for line in raw.splitlines() if line]
    except (OSError, ValueError) as exc:
        errors.append('missing or malformed prior planning input inventory: ' + str(exc))
    if errors:
        report = {'errors': errors, 'counts': {'rows': len(selected), 'complete_families': 0},
                  'scope': 'Publication blocked before family checks: missing or changed required evidence.'}
    else:
        report = checker.check_batch(selected, manifest, prior_inputs=prior)
    for path, expected in evidence.items():
        try:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                report['errors'].append('planning evidence changed during audit: ' + path)
        except OSError:
            report['errors'].append('planning evidence disappeared during audit: ' + path)
    report['evidence_sha256'] = evidence
    report['checker_author_execution'] = '/root/formal_oracle'
    report['check_runner_execution'] = '/root'
    return report
