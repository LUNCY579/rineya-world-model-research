"""Exact-custody bridge for prospective public BELIEF V4 controls.

Legacy V1/V2/V3 routing remains in pipeline. V4 is selected first by its public
batch or revision cue; evidence bytes and the frozen V3 prior are mandatory.
No new canonical field, private manifest top shape or ID syntax is imposed.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

CHECKER_NAME = 'oracles.belief_controls_v4_independent'
CHECKER_SHA256 = '0ff69cc8499d5f1fa8cfd2515d118f3a3a1ad43359bdf5c22bd3d4a81cab8613'
PRIOR_BINDING = {'path':'data/RBM-001/candidate_renderings/belief-gap-v3.jsonl',
                 'sha256':'7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c',
                 'bytes':40541020}
SOURCE_BINDINGS = {
    'oracles/belief_controls_v4_independent.py': CHECKER_SHA256,
    'oracles/belief_controls_independent.py':'ec9f387485f2a2d58ab557db0a2a7d95c24e77cb3108afa503a80c575e04b03e',
    'oracles/belief_controls_v2_independent.py':'5b97b73df0b9248b1ad155307e75e2a73b995dff64e81232382a6c05a2fe4984',
    'oracles/belief_controls_v3_independent.py':'ff1a72153e985810d2771fecf299944b39cc8afa52effbd833feac056f143271',
    'factory/common.py':'8502429a270c0a2b0ee74ce89118b28c780b7242c58d8d9c6e858ebcd43ef7c8',
    'schemas/canonical/v1.schema.json':'f4d8276fe000ceafb0f70c69dbfd1608b823767487d7c0a79623f82cf0daf25f',
    'schemas/EXTENDED_SEMANTICS.md':'2a67e05e10f99d1fd242ee1691d5b93755093fa701c48040c2d90f47422e7e23',
    'schemas/BELIEF_CONTROL_PROPERTIES.md':'200631973256bbe5dcd7b99f3d52cb95f89c442acd29ea4de1fe82dd8b5f3034',
    'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md':'46e3120dfd1ee931eec9d1e5e0e75f0b2aedb1d85764fd755afe9f7d36cf121f',
    'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md':'3a223952d9664577d7b637ddfb3e18d329863a3d5371a6c47cad233a1f50ff11',
    'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md':'40b552ecf47c11bfffb2669283640dda991d6dc6e7094ed3051fb9cad2489fce',
    'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md':'925ccae4e6fdc79fec2d0778d871e5d316257c3026b59c6e166556da71eb346c',
    'research/curricula/RBM-EVIDENCE-GAP.md':'004fb0ec0c1a1104ba052fbda4882f48523d7d34dd45b2b88e8c890713b4a120',
    'research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md':'c2eb841ed546848088db4e0499e8e8f7087aa3d213af333ff39237ae6558bd33',
    'research/curricula/RBM-EVIDENCE-GAP-REVISION-V3.md':'7e9c9f9160425ab7e455f57d72385e54f399835dfa351dcf75f6fc2842b1dc33',
    'research/curricula/BELIEF-METADATA-V4-CLARIFICATION-EXECUTION.json':'2d100ce2ad27ded95275316991f3ec8ace9ef9daf2a40c50c3f06946c777ec0d',
    'research/curricula/BELIEF_METADATA_V4_ENCODING_ROOT_FREEZE.json':'2bbd9d577e0b7eb488143bd7f532f9ce3fd1c3c51be918d90470b29bc7b61625',
}
_LOADED_BRIDGE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def is_v4_cohort(rows, batch=None):
    """Routing cues are broader than acceptance, so tampering fails closed."""
    if type(batch) is str and batch.startswith('belief-gap-v4'):
        return True
    return any(type(row) is dict and type(row.get('generator_version')) is str
               and row['generator_version'].startswith('rbm-belief-gap-4') for row in rows)


def coherent_batch(batch):
    # Coordinator-authorized fresh aliases remain new immutable candidate files.
    return type(batch) is str and re.fullmatch(r'belief-gap-v4(?:-[A-Za-z0-9][A-Za-z0-9_-]*)?', batch) is not None


def checker():
    location=Path(__file__).resolve().parents[1]/'oracles/belief_controls_v4_independent.py'
    if hashlib.sha256(location.read_bytes()).hexdigest()!=CHECKER_SHA256:
        raise ValueError('frozen V4 checker source mismatch before import')
    return importlib.import_module(CHECKER_NAME)


def _components(relative):
    if (type(relative) is not str or not relative or '\x00' in relative
            or relative.startswith(('/', '\\')) or re.match(r'^[A-Za-z]:', relative)
            or '..' in relative.replace('\\','/').split('/')):
        raise ValueError('unsafe ROOT-relative evidence path')
    parts = PurePosixPath(relative).parts
    if not parts or parts == ('.',):
        raise ValueError('evidence path must name a file')
    return parts


def safe_read(root, relative):
    """Open each descendant through no-follow descriptors, rejecting symlinks."""
    parts = _components(relative); root = Path(root).resolve(strict=True)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    leaf = None
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = child
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(leaf).st_mode):
            raise ValueError('evidence must be a regular file')
        with os.fdopen(leaf, 'rb') as handle:
            leaf = None
            return handle.read()
    finally:
        if leaf is not None: os.close(leaf)
        os.close(directory)


def load_v4_evidence(rows, batch, root):
    """Return exact bytes and their observed ledger, or fail before proof calls."""
    errors=[]; observed={}; manifest_sources={}; candidate_bytes=None; previous_rows=None
    module=None
    def read(relative, expected=None, size=None):
        raw=safe_read(root,relative); sha=hashlib.sha256(raw).hexdigest()
        observed[relative]={'sha256':sha,'bytes':len(raw)}
        if expected is not None and sha != expected: raise ValueError('frozen evidence SHA256 mismatch: '+relative)
        if size is not None and len(raw) != size: raise ValueError('frozen evidence byte count mismatch: '+relative)
        return raw
    try:
        module=checker()
        if not coherent_batch(batch): raise ValueError('V4 requires coherent belief-gap-v4 batch or fresh alias')
        if type(rows) is not list or not rows: raise ValueError('V4 requires nonempty supplied rows')
        for row in rows:
            if (type(row) is not dict or row.get('track_id')!='RBM-001'
                    or type(row.get('semantic_task')) is not dict or row['semantic_task'].get('kind')!='belief'
                    or type(row.get('generator_version')) is not str
                    or re.fullmatch(r'rbm-belief-gap-4\.0\.[0-9]+',row['generator_version']) is None):
                raise ValueError('V4 requires RBM belief rows with coherent rbm-belief-gap-4.0.x revision')
        if len({row['generator_version'] for row in rows})!=1: raise ValueError('V4 cohort mixes generator revisions')
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=CHECKER_SHA256:
            raise ValueError('imported frozen V4 checker source mismatch')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=_LOADED_BRIDGE_SHA256:
            raise ValueError('bridge source changed after import')
        for path,sha in SOURCE_BINDINGS.items(): read(path,sha)
        read('factory/belief_controls.py',_LOADED_BRIDGE_SHA256)
        candidate_path='data/RBM-001/candidate_renderings/'+batch+'.jsonl'
        candidate_bytes=read(candidate_path)
        candidate_rows,by,sha=module._candidate(candidate_bytes)
        if any(row.get('review_status')!='PENDING' or row.get('review_receipt_ids')!=[] for row in candidate_rows):
            raise ValueError('preserved V4 source candidate must remain PENDING with empty receipt list')
        if any(row.get('generator_version')!=rows[0]['generator_version'] or row.get('track_id')!='RBM-001' or type(row.get('semantic_task')) is not dict or row['semantic_task'].get('kind')!='belief' for row in candidate_rows):
            raise ValueError('candidate revision/domain does not match V4 supplied rows')
        module._bind_rows(rows,by,sha)
        for row in rows:
            for claim in module._claims(row):
                if claim['property'] in module.INHERITED and claim['scope']=='corpus_variant':
                    path=claim['evidence_manifest']
                    if path not in manifest_sources: manifest_sources[path]=read(path)
        # No top-level version/batch/public-SHA manifest keys are assumed.
        prior_raw=read(PRIOR_BINDING['path'],PRIOR_BINDING['sha256'],PRIOR_BINDING['bytes'])
        previous_rows,_,_=module._candidate(prior_raw)
    except (OSError,ValueError,KeyError,TypeError,UnicodeError) as error:
        errors.append('UNKNOWN: V4 required evidence/custody: '+str(error))
    return {'errors':errors,'candidate_bytes':candidate_bytes,'manifest_sources':manifest_sources,'previous_rows':previous_rows,'evidence':observed,'module':module}


def belief_v4_control_report(rows, batch, root):
    loaded=load_v4_evidence(rows,batch,root)
    if loaded['errors']:
        report={'errors':list(loaded['errors']),'counts':{'records':len(rows) if type(rows) is list else 0},'v4_preservation':{'status':'NOT_RUN'},'scope':'Publication blocked before full proofs: required exact bytes/custody missing or changed.'}
    else:
        try:
            report=loaded['module'].check_batch(rows,manifest_sources=loaded['manifest_sources'],candidate_bytes=loaded['candidate_bytes'],previous_rows=loaded['previous_rows'])
        except Exception as error:
            report={'errors':['UNKNOWN: frozen V4 checker execution failed: '+type(error).__name__+': '+str(error)],'counts':{'records':len(rows)},'v4_preservation':{'status':'NOT_RUN'}}
        if report.get('v4_preservation',{}).get('status')!='PROVED':
            report['errors'].append('V4 zero-growth preservation requires actual frozen prior/current full triples proof')
    for path,binding in loaded['evidence'].items():
        try:
            if hashlib.sha256(safe_read(root,path)).hexdigest()!=binding['sha256']:
                report['errors'].append('V4 evidence changed during audit: '+path)
        except (OSError,ValueError) as error:
            report['errors'].append('V4 evidence disappeared or became unsafe during audit: '+path+': '+str(error))
    report.update(evidence_sha256={path:item['sha256'] for path,item in loaded['evidence'].items()}, evidence_bytes={path:item['bytes'] for path,item in loaded['evidence'].items()}, checker_sha256=CHECKER_SHA256, checker_revision='V4 public metadata/owner resolver', bridge_sha256=_LOADED_BRIDGE_SHA256, checker_author_execution='/root/formal_oracle', check_runner_execution='/root', approval_verdict=None)
    return report
