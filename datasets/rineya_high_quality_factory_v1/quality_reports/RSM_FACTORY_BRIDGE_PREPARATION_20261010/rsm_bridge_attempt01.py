"""ISOLATED RSM bridge preparation; not installed in the shared factory.

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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
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
LOADED_BRIDGE_SHA=sha256(Path(__file__).read_bytes()).hexdigest()
_CHECKER=None

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
    for path,h in SOURCES.items(): need(sha256(safe_read(ROOT,path)).hexdigest()==h,'frozen implementation/public source changed: '+path)
    need(sha256(Path(__file__).read_bytes()).hexdigest()==LOADED_BRIDGE_SHA,'bridge changed after import')
    if _CHECKER is None:
        spec=importlib.util.spec_from_file_location('rsm_frozen_original_bdbc',ROOT/CHECKER_PATH)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        need(Path(module.__file__).resolve()==(ROOT/CHECKER_PATH).resolve() and module.ROOT==ROOT,'checker original location/ROOT binding')
        _CHECKER=module
    need(Path(_CHECKER.__file__).resolve()==(ROOT/CHECKER_PATH).resolve() and _CHECKER.ROOT==ROOT,'cached checker original location/ROOT binding')
    return _CHECKER

def is_cohort(rows,batch=None):
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
def cohort_rows(rows,batch=None):
    # Closed public pilot is216 RSM rows. Never filter by current corrupted kind,
    # track, version, claim, or approval flags once its batch/domain is selected.
    return list(rows) if type(rows) is list and is_cohort(rows,batch) else []

def load_evidence(rows,batch,root=ROOT,candidate_relative=None):
    observed={}; errors=[]; module=None; candidate=None; manifest_sources={}; source_bytes={}; legacy=None
    def read(path,expected=None):
        raw=safe_read(root,path); h=sha256(raw).hexdigest(); observed[path]={'sha256':h,'bytes':len(raw)}
        need(expected is None or h==expected,'exact evidence/source hash mismatch: '+path); return raw
    try:
        need(batch==BATCH,'operative public batch must be '+BATCH)
        need(type(rows) is list and rows,'complete current rows required')
        module=checker()
        for path,h in SOURCES.items(): source_bytes[path]=read(path,h)
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

def control_report(rows,batch=None,root=ROOT,candidate_relative=None):
    if not is_cohort(rows,batch): return None
    loaded=load_evidence(cohort_rows(rows,batch),batch,root,candidate_relative)
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
    report.update(checker_path=CHECKER_PATH,checker_sha256=CHECKER_SHA,bridge_sha256=LOADED_BRIDGE_SHA,
                  evidence=loaded['evidence'],reference_agreement='NOT_RUN_SEPARATE_AUTHOR_REQUIRED',production_data_approval=False,
                  actual_checker_author_execution='/root/formal_oracle',scope='Isolated machine-gate preparation; no core installation or production approval.')
    return report

def gate(stage,rows,batch=None,root=ROOT,candidate_relative=None,exports=None):
    need(stage in ('audit','publish','validate_release','exports'),'stage')
    report=control_report(rows,batch,root,candidate_relative)
    if report is None: return {'stage':stage,'route':'LEGACY_DELEGATE','errors':[],'production_data_approval':False}
    if exports is not None and not report['errors']:
        report['errors'].extend(_validate_exports(rows,exports,checker()))
    report['stage']=stage; report['route']='RSM_WHOLE_COHORT'; report['machine_controls_pass']=not report['errors']
    # This function deliberately performs no directory creation/publication.
    return report

def _export_payloads(rows,module):
    out={}
    for split in ('train','dev_challenge'):
        selected=[r for r in rows if r['split']==split]
        structured=[{'input':r['rendered_input'],'target':module.solve(r['semantic_task'])} for r in selected]
        out[split+'-structured.jsonl']=structured
        out[split+'-skill_demonstrations.jsonl']=structured
        out[split+'-sft.jsonl']=[{'messages':[{'role':'user','content':v['input']},{'role':'assistant','content':module.canonical(v['target'])}]} for v in structured]
    return out
def _validate_exports(rows,exports,module):
    errors=[]
    try:
        need(type(exports) is dict,'derived exports mapping required')
        expected=_export_payloads(rows,module)
        need(set(expected)<=set(exports),'all two-split SFT/structured/skill files required')
        for name,items in expected.items():
            observed=exports[name]; need(type(observed) is list,'export list '+name)
            need(Counter(module.canonical(v) for v in observed)==Counter(module.canonical(v) for v in items),'exact typed one-per-row export coverage/labels: '+name)
    except (ValueError,TypeError,KeyError,RecursionError) as e: errors.append('RSM derived exports: '+str(e))
    return errors

class GateError(ValueError):
    def __init__(self,report):
        self.report=report; super().__init__('RSM whole-cohort gate blocked: '+str(report['errors']))
def _require_exports(rows,batch,root,candidate_relative=None):
    report=gate('exports',rows,batch,root,candidate_relative)
    need(report['route']=='RSM_WHOLE_COHORT','legacy export delegates to existing implementation, absent from isolated prototype')
    if report['errors']: raise GateError(report)
    return report
def specialized(rows,split,batch=BATCH,root=ROOT,candidate_relative=None):
    _require_exports(rows,batch,root,candidate_relative)
    need(split in ('train','dev_challenge'),'split')
    name=split+'-skill_demonstrations.jsonl'
    return {name:_export_payloads(rows,checker())[name]}
def write_specialized(destination,rows,batch=BATCH,root=ROOT,candidate_relative=None):
    report=_require_exports(rows,batch,root,candidate_relative)
    destination=Path(destination).resolve(); need(destination.is_relative_to(HERE),'isolated export destination only')
    # Full cohort proof precedes creating directories or opening output files.
    destination.mkdir(parents=True,exist_ok=True); payloads=_export_payloads(rows,checker())
    for split in ('train','dev_challenge'):
        name=split+'-skill_demonstrations.jsonl'
        with (destination/name).open('xb') as f:
            f.write(b''.join((checker().canonical(item)+'\n').encode() for item in payloads[name]))
    return report
def validate_specialized(exports,rows,batch=BATCH,root=ROOT,candidate_relative=None):
    report=_require_exports(rows,batch,root,candidate_relative)
    module=checker(); expected=_export_payloads(rows,module)
    for split in ('train','dev_challenge'):
        name=split+'-skill_demonstrations.jsonl'
        need(type(exports) is dict and name in exports and type(exports[name]) is list,'both specialized split files required')
        need(Counter(module.canonical(item) for item in exports[name])==Counter(module.canonical(item) for item in expected[name]),'typed specialized coverage/target mismatch '+name)
    return report

def package_sources(destination):
    """Write NEW isolated source blobs only; preserve original import paths."""
    destination=Path(destination).resolve(); need(destination.is_relative_to(HERE),'isolated preparation destination only')
    need(not destination.exists(),'immutable package destination exists')
    checker(); raw_sources={p:safe_read(ROOT,p) for p in SOURCES}
    raw_sources[LEGACY_PATH]=safe_read(ROOT,LEGACY_PATH)
    bridge_path=str(Path(__file__).relative_to(ROOT)); raw_sources[bridge_path]=Path(__file__).read_bytes()
    destination.mkdir(); blobs=destination/'source_blobs'; blobs.mkdir(); ledger={}
    for path,raw in raw_sources.items():
        h=sha256(raw).hexdigest(); (blobs/(h+'.source')).write_bytes(raw); ledger[path]={'sha256':h,'bytes':len(raw),'archive':'source_blobs/'+h+'.source'}
    plan={'source_bindings':ledger,'restore_paths':'Extract blobs to original ROOT-relative paths; checker stays at original reviews/oracle location.',
      'checker_original_path':CHECKER_PATH,'source_archive_kind':'Isolated packaging demonstration only','production_reference':'NOT_INCLUDED_NOT_AUTHORED_OR_READ',
      'production_data_approval':False}
    (destination/'SOURCE_PACKAGE.json').write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n')
    checker(); return plan

def verify_package(destination):
    destination=Path(destination); plan=json.loads((destination/'SOURCE_PACKAGE.json').read_bytes())
    for path,b in plan['source_bindings'].items():
        components(path); raw=safe_read(destination,b['archive']); need(len(raw)==b['bytes'] and sha256(raw).hexdigest()==b['sha256'],'packaged exact source bytes '+path)
    return {'source_files':len(plan['source_bindings']),'all_exact':True,'production_data_approval':False}
