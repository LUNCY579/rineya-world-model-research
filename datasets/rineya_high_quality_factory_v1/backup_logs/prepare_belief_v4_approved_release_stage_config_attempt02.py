#!/usr/bin/env python3
"""Exact future stage config preparation and read-only validation; no Git mutation."""
from pathlib import Path
import argparse,datetime as dt,hashlib,importlib.util,json,os,sys
from types import SimpleNamespace
sys.dont_write_bytecode=True
F=Path(__file__).resolve().parent.parent;L=F/'backup_logs';AUTHOR='/root/backup_manager'
SPEC_SHA='6c21c64d1cff3ce38d548ed024f6bb9da9f9c8869463d7c4cbd5038240cd97f9'
RUNNER_SHA='f0561eb5f0e2ebe16bc46f51ed1fcbb1ad74236e90fba4d9b33da166c6767715'
STAGE_FREEZE_SHA='3c312ce1c2295b78b565191571e2103f4070f92694aeaae6d6953f80e34648ee'
ADAPTER_SHA='7978cab638a00c185a9d391259c6a5c77b8a5c97ea5fefdc79fa67a31e58e023'
WARNING_SHA='bd6639e10289bcbd7c27e3bf68241e83dbf7f0d9985400b7abf9df9ae9b9fa4b'
META_SHA='2e18c2c156aaefcee75ad6ad92fe51c66abf864150924d5ac1cd26dbeed5aa05'
PARENT='4d0b5eff1af3184fedf7fa25ada729a6cc909268'
def bind(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  while b:=s.read(65536):h.update(b)
 return {'path':p.relative_to(F).as_posix(),'size':p.stat().st_size,'sha256':h.hexdigest()}
def write(p,d):
 with p.open('x') as s:json.dump(d,s,indent=2);s.write('\n');s.flush();os.fsync(s.fileno())
 return bind(p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--source-freeze',type=Path,required=True);a.add_argument('--expected-freeze-sha256',required=True);args=a.parse_args()
 if bind(args.source_freeze)['sha256']!=args.expected_freeze_sha256:raise ValueError('Config preparation freeze differs')
 freeze=json.loads(args.source_freeze.read_bytes())
 if freeze['source']!=bind(Path(__file__)):raise ValueError('Config source not frozen')
 for e in freeze['frozen_inputs']:
  if bind(F/e['path'])!=e:raise ValueError('Bound input changed')
 source=L/'run_provisional_archival_checkpoint_v4_attempt02.py'
 if bind(source)['sha256']!=RUNNER_SHA:raise ValueError('Reviewed stage changed')
 sp=importlib.util.spec_from_file_location('frozen_attempt02_stage',source);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 if bind(source)['sha256']!=RUNNER_SHA:raise ValueError('Stage changed during import')
 spec=json.loads((L/'belief-v4-approved-release-r1_SPEC.json').read_bytes());report=json.loads((L/'belief-v4-approved-release-r1_INVENTORY_REPORT.json').read_bytes());prior=json.loads((L/'belief-v4-validated-code-candidate-r2_SPEC.json').read_bytes())
 config={'schema':'prospective-archival-stage-v3-execution-v1','author_execution':AUTHOR,'runner_execution':'/root','expected_parent':PARENT,'prior_spec_sha256':m.PRIOR_SPEC_SHA,'prior_tracked_path_count':2685,'prior_release_file_count':210,'expected_release_file_count':222,'readonly_reference_count':65,'grants_data_approval':False,
 'spec_sha256':SPEC_SHA,'source_freeze_sha256':STAGE_FREEZE_SHA,'custody_adapter_sha256':ADAPTER_SHA,'exception_config_sha256':WARNING_SHA,'metadata_exception_config_sha256':META_SHA,'checkpoint_id':spec['checkpoint_id'],'release_files':m.release_inventory(spec),'raw_lossless_required_count':31,'readonly_reference_files':report['custody_preflight_after']['readonly_reference_files'],'historical_raw_relocations':spec['historical_raw_relocations'],'runner_source_sha256':RUNNER_SHA,'metadata_scanner_source_sha256':m.METADATA_HELPER_SHA,'schema_disclosure':'Published v3 execution schema inherited by NEW source-frozen v4 attempt02 stage with exact four-file public metadata exceptions; no existing helper/config edits.','secret_scan_truth':'PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS; exact4fullyverifiedknownpaths;unbound0','postcut_boundary':'This new config and its preparation/validation outputs are LOCAL postinventory; never claimed included by3158fileSPEC.'}
 cp=L/'BELIEF_V4_APPROVED_RELEASE_R1_STAGE_V4_ATTEMPT02_EXECUTION_CONFIG.json';cb=write(cp,config)
 pre=write(L/'BELIEF_V4_APPROVED_RELEASE_R1_STAGE_V4_ATTEMPT02_CONFIG_VALIDATION_PREEXEC.json',{'author_execution':AUTHOR,'actual_runner_execution':AUTHOR,'at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'frozen_config':cb,'source':bind(Path(__file__)),'source_freeze':bind(args.source_freeze),'validation_scope':'Read-only actual local source/index metadata and config guard validation; no production Git mutation.'})
 checked=m.inventory(F,spec)
 boundargs=SimpleNamespace(expected_spec_sha256=SPEC_SHA,expected_freeze_sha256=STAGE_FREEZE_SHA,expected_custody_sha256=ADAPTER_SHA,expected_exception_sha256=WARNING_SHA,expected_parent=PARENT,runner='/root')
 validation=m.validate_execution_config(config,boundargs,spec,prior,checked)
 m.verify_readonly_references(config);m.validate_exception(json.loads((L/'PROSPECTIVE_ARCHIVAL_V3_EXACT_EXCEPTION.json').read_bytes()))
 secret=m.metadata_helper().report(require_complete=True)
 W=F.parent/'.rineya-dataset-git-worktree';original=F.parent/'rineya-world-model-research';adapter=json.loads((L/'PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER.json').read_bytes())
 m.original_custody(original,adapter);private=m.dedicated_custody(W,PARENT)
 tracked=set(m.index(W));names=sorted(checked);paths=[m.PREFIX+'/'+n for n in names]
 ignored={p.decode()[len(m.PREFIX)+1:] for p in m.git(W,'check-ignore','--no-index','-z','--stdin',allowed=(0,1),data=('\0'.join(paths)+'\0').encode()).split(b'\0') if p}
 for e in freeze['frozen_inputs']:
  if bind(F/e['path'])!=e:raise ValueError('Input changed during validation')
 if bind(cp)!=cb:raise ValueError('Frozen config changed during validation')
 out=write(L/'BELIEF_V4_APPROVED_RELEASE_R1_STAGE_V4_ATTEMPT02_CONFIG_VALIDATION.json',{'schema':'actual-exact-stage-config-readonly-validation-v1','status':'PASS_CONFIG_ONLY_NO_GIT_MUTATION','author_execution':AUTHOR,'actual_runner_execution':AUTHOR,'eventual_stage_runner':'/root','at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'config':cb,'preexecution':pre,'scope_validation':validation,'all_actual_capped_source_size_sha256_gitsha1_revalidated':len(checked),'actual_unique_source_git_blob_groups':len({q['git_blob'] for q in checked.values()}),'actual_secret_scan':secret,'all65_reference_bytehashes_rechecked':True,'original_main_clean':True,'dedicated_parent_clean':True,'six_original_git_byte_guards_rechecked':True,'private_admin':private,'ignored_exact_inventory_paths':sorted(ignored),'ignored_already_tracked':sorted(ignored&tracked),'new_ignored_paths_requiring_cli_allow_ignored_path':sorted(ignored-tracked),'all_source_config_bindings_unchanged':True,'no_copy_stage_commit_push':True,'data_approval':False})
 print(json.dumps({'config':cb,'validation':out,'ignored_count':len(ignored),'new_ignored_paths':sorted(ignored-tracked),'source_paths':len(checked),'unique_groups':len({q['git_blob'] for q in checked.values()})}))
if __name__=='__main__':main()
