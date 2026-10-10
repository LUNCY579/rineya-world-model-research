"""RSM raw-evidence factory adapter; independent task/control routing.

Only own frozen R2 checker/official solver are loaded. No producer reference is
read or reused. Gates are machine-proof demonstrations on hand inputs, never
data approvals or substitute expert/adversarial/reference executions.
"""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat

ROOT=Path(__file__).resolve().parents[1]
ADAPTER_PATH='factory/rsm_controls.py'
BATCH='option-resumption-pilot-v1'
CHECKER_PATH='reviews/oracle/RSM-option-resumption-corpus-20261010T0956Z/corpus_checker_attempt04.py'
CHECKER_SHA='bdbc7287d5a9167013c21665f4793457481db55612fc9e60fef11fa7157fb9a9'
SOURCES={
 CHECKER_PATH:CHECKER_SHA,
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/option_resume_solver_v1.py':'42726260ec8a1d713c257f09efdd0910a9f30a836058ab0d7f662e95d1112684',
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/legacy_contrast_v1.py':'bb628fd918aeb82c3422a029b6ca671ff694090749f9d23d85a8a1d473cdf566',
 'schemas/RSM_OPTION_RESUMPTION_V1.md':'44652656ad18e60c98c96db14e0a96b4e46b0dad0bd57f257bca30384ab53b3f',
 'research/curricula/RSM-OPTION-RESUMPTION-PROSPECTIVE-20261010.md':'8bda187009cc91756248ddd86792c28aabe0793a3f6be304bb9433bfdf48be34',
 'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json':'422f67c836639d0accbf411c3080ef4fc7695ad8630120a140488c90a19061fa',
 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1.md':'1612b14d7549b7fb406cbef8e127440dcfddaf30548ef29d217bc49aefb6c854',
 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R1.md':'5f7f6a2c45dd6a0b51c7a912b28fd49badd9aa3aa2eac8632e9b9781c201b57b',
 'schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md':'b62663c3082c9e2ee2bf34aadc1fdd10f263a6497396b6768f1475b12f568fe4',
 'schemas/canonical/v1.schema.json':'f4d8276fe000ceafb0f70c69dbfd1608b823767487d7c0a79623f82cf0daf25f',
 'factory/common.py':'8502429a270c0a2b0ee74ce89118b28c780b7242c58d8d9c6e858ebcd43ef7c8',
}
LEGACY_PATH='reviews/input_only/RSM-legacy-v0001.jsonl'
LEGACY_SHA='ba62d973531521bfc57a5b52ee630a61f3384d9e8dfd41becea6a1eb63ca86c1'
CONTROLS={'uninterrupted_resume_suffix','resume_restart_replay','budget_pause_after_effect',
 'second_checkpoint_suffix','positive_bound_argument_counterfactual','initiation_argument_counterfactual',
 'success_contract_counterfactual','active_checkpoint_tamper_rejection'}
LOADED_ADAPTER_SHA=sha256(Path(__file__).read_bytes()).hexdigest()
_CHECKER=None
_TASK_SOLVER=None
TASK_SOLVER_PATH='reviews/oracle/RSM-option-resumption-official-20261010T0932Z/option_resume_solver_v1.py'
TASK_SOLVER_SHA=SOURCES[TASK_SOLVER_PATH]

def need(ok,message):
    if not ok: raise ValueError(message)
def components(relative):
    need(type(relative) is str and relative and '\x00' not in relative and '\\' not in relative,'safe ROOT-relative path required')
    p=PurePosixPath(relative)
    need(not p.is_absolute() and all(c not in ('','..','.') for c in relative.split('/')),'safe ROOT-relative path required')
    return p.parts
def safe_read(root,relative):
    """Every path component opened without following links; regular files only."""
    parts=components(relative); root=Path(root).resolve(strict=True)
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY); leaf=None
    try:
        for part in parts[:-1]:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd); fd=child
        leaf=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        need(stat.S_ISREG(os.fstat(leaf).st_mode),'regular evidence file required')
        with os.fdopen(leaf,'rb') as f: leaf=None; return f.read()
    finally:
        if leaf is not None: os.close(leaf)
        os.close(fd)

def checker():
    global _CHECKER
    # Keep original location: its ROOT=parents[3] must still resolve to factory.
    raw_sources={}
    for path,h in SOURCES.items():
        raw=safe_read(ROOT,path); need(sha256(raw).hexdigest()==h,'frozen implementation/public source changed: '+path)
        raw_sources[path]=raw
    need(sha256(Path(__file__).read_bytes()).hexdigest()==LOADED_ADAPTER_SHA,'adapter changed after import')
    if _CHECKER is None:
        spec=importlib.util.spec_from_file_location('rsm_frozen_original_bdbc',ROOT/CHECKER_PATH)
        module=importlib.util.module_from_spec(spec)
        exec(compile(raw_sources[CHECKER_PATH],str(ROOT/CHECKER_PATH),'exec'),module.__dict__)
        need(Path(module.__file__).resolve()==(ROOT/CHECKER_PATH).resolve() and module.ROOT==ROOT,'checker original location/ROOT binding')
        _CHECKER=module
    need(Path(_CHECKER.__file__).resolve()==(ROOT/CHECKER_PATH).resolve() and _CHECKER.ROOT==ROOT,'cached checker original location/ROOT binding')
    return _CHECKER

def is_rsm_cohort(rows,batch=None):
    if type(batch) is str and batch.startswith(BATCH): return True
    if type(rows) is not list: return False
    for row in rows:
        if type(row) is not dict: continue
        task=row.get('semantic_task')
        if type(task) is dict and task.get('kind')=='option_resume': return True
        revision=row.get('generator_version')
        if type(revision) is str and revision.startswith('rsm-') and ('option-resum' in revision or 'option_resume' in revision): return True
        claims=row.get('family_control_claims')
        if type(claims) is list and any(type(c) is dict and any(type(c.get(k)) is str and c[k] in CONTROLS for k in ('name','property')) for c in claims): return True
    return False
def rsm_control_rows(rows,batch=None):
    # Closed public pilot is216 RSM rows. Never filter by current corrupted kind,
    # track, version, claim, or approval flags once its batch/domain is selected.
    return list(rows) if type(rows) is list and is_rsm_cohort(rows,batch) else []

def load_evidence(rows,batch,root=None,candidate_relative=None):
    root=ROOT if root is None else Path(root)
    observed={}; errors=[]; module=None; candidate=None; manifest_sources={}; source_bytes={}; legacy=None
    def read(path,expected=None):
        raw=safe_read(root,path); h=sha256(raw).hexdigest(); observed[path]={'sha256':h,'bytes':len(raw)}
        need(expected is None or h==expected,'exact evidence/source hash mismatch: '+path); return raw
    try:
        need(batch==BATCH,'operative public batch must be '+BATCH)
        need(type(rows) is list and rows,'complete current rows required')
        module=checker()
        for path,h in SOURCES.items(): source_bytes[path]=read(path,h)
        source_bytes[ADAPTER_PATH]=read(ADAPTER_PATH,LOADED_ADAPTER_SHA)
        candidate=read(candidate_relative or ('data/RSM-001/candidate_renderings/'+BATCH+'.jsonl'))
        original=[module.strict_json(line) for line in candidate.splitlines() if line.strip()]
        need(len(original)==216,'whole original PENDING216 required')
        for row in original:
            need(type(row) is dict and row.get('track_id')=='RSM-001' and row.get('review_status')=='PENDING' and row.get('review_receipt_ids')==[],'original PENDING row domain/review binding')
            task=row.get('semantic_task'); need(type(task) is dict and task.get('kind')=='option_resume' and type(task.get('version')) is int and task['version']==1,'original task kind/strict version')
            for claim in module.claims(row).values():
                path=claim['evidence_manifest']
                if path not in manifest_sources:
                    raw=read(path); manifest_sources[path]=raw; manifest=module.strict_json(raw)
                    sb=manifest.get('source_bindings'); need(type(sb) is dict,'manifest source bindings object')
                    for p,h in sb.items():
                        need(module.is_sha(h),'source hash type'); source_bytes[p]=read(p,h)
        legacy=read(LEGACY_PATH,LEGACY_SHA)
    except (OSError,ValueError,TypeError,KeyError,UnicodeError,RecursionError) as e:
        errors.append('UNKNOWN: RSM whole-cohort evidence/custody: '+str(e))
    return {'errors':errors,'module':module,'candidate_bytes':candidate,'manifest_sources':manifest_sources,
            'source_bytes':source_bytes,'legacy_input_bytes':legacy,'evidence':observed,'root':Path(root)}

def recheck(loaded):
    errors=[]
    for path,b in loaded['evidence'].items():
        try: need(sha256(safe_read(loaded['root'],path)).hexdigest()==b['sha256'],'evidence changed during gate: '+path)
        except (OSError,ValueError) as e: errors.append('UNKNOWN: '+str(e))
    try: checker()
    except (OSError,ValueError,TypeError,KeyError) as e: errors.append('UNKNOWN: source changed during gate: '+str(e))
    return errors

def rsm_control_report(rows,batch=None,root=None,candidate_relative=None):
    if not is_rsm_cohort(rows,batch): return None
    # Default legacy calls omit the optional batch. Domain/control markers select the
    # exact public cohort; an explicitly supplied wrong batch still fails.
    batch=BATCH if batch is None else batch
    loaded=load_evidence(rsm_control_rows(rows,batch),batch,root,candidate_relative)
    if loaded['errors']:
        report={'errors':list(loaded['errors']),'counts':{'records':len(rows) if type(rows) is list else 0}}
    else:
        try:
            report=loaded['module'].check_batch(rows,manifest_sources=loaded['manifest_sources'],source_bytes=loaded['source_bytes'],candidate_bytes=loaded['candidate_bytes'],legacy_input_bytes=loaded['legacy_input_bytes'])
            for row in rows:
                truth=loaded['module'].solve(row['semantic_task'])
                if not loaded['module'].eq(truth,row.get('verified_target')): report['errors'].append('independent typed target mismatch: '+str(row.get('sample_id')))
        except Exception as e: report={'errors':['UNKNOWN: frozen RSM checker execution: '+type(e).__name__+': '+str(e)],'counts':{'records':len(rows)}}
    report['errors'].extend(recheck(loaded))
    report.update(checker_path=CHECKER_PATH,checker_sha256=CHECKER_SHA,adapter_sha256=LOADED_ADAPTER_SHA,
                  evidence=loaded['evidence'],reference_agreement='NOT_RUN_SEPARATE_AUTHOR_REQUIRED',production_data_approval=False,
                  actual_checker_author_execution='/root/formal_oracle',scope='Raw-input machine controls and independent targets; no data approval or separate-reference assertion.')
    return report


def task_solver():
    """Exact standalone official427 task route, distinct from producer reference."""
    global _TASK_SOLVER
    path=ROOT/TASK_SOLVER_PATH
    raw=safe_read(ROOT,TASK_SOLVER_PATH)
    need(sha256(raw).hexdigest()==TASK_SOLVER_SHA,'official task solver source changed')
    if _TASK_SOLVER is None:
        spec=importlib.util.spec_from_file_location('rsm_frozen_official_427',path)
        module=importlib.util.module_from_spec(spec)
        exec(compile(raw,str(path),'exec'),module.__dict__); _TASK_SOLVER=module
    need(Path(_TASK_SOLVER.__file__).resolve()==path.resolve(),'official task solver original location')
    return _TASK_SOLVER

def option_resume_solve(task):
    result=task_solver().solve(task)
    task_solver() # Recheck exact source after execution; no import fallback.
    return result

def check(row):
    try:
        result=checker().check(row); checker(); return result
    except Exception as e:
        return ['UNKNOWN: frozen RSM row checker: '+type(e).__name__+': '+str(e)]

class RSMGateError(ValueError):
    def __init__(self,report):
        self.report=report; super().__init__('RSM whole-cohort gate blocked: '+str(report['errors']))

def require_rsm_controls(rows,batch=None,root=None,candidate_relative=None):
    report=rsm_control_report(rows,batch,root,candidate_relative)
    if report is not None and report['errors']: raise RSMGateError(report)
    return report

def validate_rsm_specialized(path,rows,batch=None,root=None):
    """Whole cohort BEFORE directory reads, then both typed skill split files."""
    root=ROOT if root is None else Path(root)
    report=require_rsm_controls(rows,batch,root)
    if report is None: return None
    path=Path(path); path=path if path.is_absolute() else root/path
    relative=str(path.relative_to(root)); module=checker(); counts={}
    for split in ('train','dev_challenge'):
        name=split+'-skill_demonstrations.jsonl'
        raw=safe_read(root,relative+'/'+name)
        records=[module.strict_json(line) for line in raw.splitlines() if line.strip()]
        expected=[{'input':r['rendered_input'],'target':module.solve(r['semantic_task'])} for r in rows if r['split']==split]
        need(Counter(module.canonical(v) for v in records)==Counter(module.canonical(v) for v in expected),'RSM both-split exact typed specialized coverage/target mismatch: '+name)
        counts[name]=len(records)
    checker(); return {'counts':counts,'report':report}

def archive_raw(root,h,raw):
    """No-follow directories and leaf; exact existing blobs are immutable."""
    fd=os.open(Path(root).resolve(strict=True),os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in ('reproducibility','source_blobs'):
            try: os.mkdir(part,dir_fd=fd)
            except FileExistsError: pass
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd); fd=child
        name=h+'.source'
        try: leaf=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
        except FileExistsError:
            leaf=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
            with os.fdopen(leaf,'rb') as f:
                need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'regular archive blob required')
                need(f.read()==raw,'RSM source archive mismatch')
        else:
            with os.fdopen(leaf,'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    finally: os.close(fd)

def preserve_rsm_bytes(report,root=None):
    """Archive ALL actual runtime/raw evidence bytes at original restore paths.

    Includes bdbc/427/bb628, public6/config, canonical/common, adapter, legacy,
    original PENDING candidate, raw manifests and every asserted extra source.
    Unlike text-only generic archives, arbitrary asserted bytes remain exact.
    """
    if report is None: return {}
    if report['errors']: raise RSMGateError(report)
    root=ROOT if root is None else Path(root); ledger={}
    for relative,b in report['evidence'].items():
        raw=safe_read(root,relative); h=sha256(raw).hexdigest()
        need(h==b['sha256'] and len(raw)==b['bytes'],'RSM evidence changed before archive: '+relative)
        archive_raw(root,h,raw)
        ledger[relative]={'sha256':h,'bytes':len(raw)}
    for relative,b in ledger.items():
        raw=safe_read(root,relative); need(sha256(raw).hexdigest()==b['sha256'] and len(raw)==b['bytes'],'RSM evidence changed during archive')
    checker(); return ledger

def validate_rsm_release_bindings(report,manifest,root=None):
    if report is None: return
    root=ROOT if root is None else Path(root)
    need(manifest.get('source_archive')=='reproducibility/source_blobs/{sha256}.source','RSM exact source archive format required')
    need(type(manifest.get('rsm_source_and_evidence')) is dict,'RSM release source/evidence ledger required')
    need(checker().eq(manifest['rsm_source_and_evidence'],report['evidence']),'RSM release exact typed source/evidence custody ledger')
    recorded=manifest.get('reproducibility_code_sha256')
    need(type(recorded) is dict and all(recorded.get(p)==b['sha256'] for p,b in report['evidence'].items()),'RSM original-path bytes absent from release source archive')
    # Never depend on the generic optional source_archive flag for restoration.
    for relative,b in report['evidence'].items():
        raw=safe_read(root,relative)
        archived=safe_read(root,'reproducibility/source_blobs/'+b['sha256']+'.source')
        need(len(archived)==b['bytes'] and sha256(archived).hexdigest()==b['sha256'] and archived==raw,'RSM exact archived evidence bytes missing/corrupt: '+relative)
    # Also enforce no-follow archive bytes for generic runtime/reference sources.
    for relative,h in recorded.items():
        need(checker().is_sha(h),'RSM runtime archive hash type')
        archived=safe_read(root,'reproducibility/source_blobs/'+h+'.source')
        need(sha256(archived).hexdigest()==h,'RSM exact runtime source archive corrupt: '+relative)
