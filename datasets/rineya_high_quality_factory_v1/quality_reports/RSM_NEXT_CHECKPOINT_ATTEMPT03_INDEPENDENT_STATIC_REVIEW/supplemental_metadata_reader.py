"""Own stdlib-only static set comparison against prior explicit byte readback.

Parses source constants as quoted text, never evaluates/imports target code.
Freeze the source/raw archive before the first invocation.
"""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent
ROOT=HERE.parents[1]
NEW='backup_logs/NEXT_CHECKPOINT_653_IMPLEMENTATION_ATTEMPT03/'
def raw(path):return (ROOT/path).read_bytes()
readback_path=str((HERE/'STATIC_BYTE_READBACK.json').relative_to(ROOT))
readback_raw=raw(readback_path)
if sha256(readback_raw).hexdigest()!='f2d95e23ed646b1da81f2aca54d63ced75dbf8580c4b77bec05833193a23c335':raise ValueError('prior actual explicit byte readback changed')
prior=json.loads(readback_raw)
if prior['status']!='PASS_EXPLICIT_METADATA_BYTE_READBACK_ONLY':raise ValueError('prior byte reader not successful')
def checked(path):
    value=raw(path)
    observed=prior['observations'][path]
    if sha256(value).hexdigest()!=observed['sha256'] or len(value)!=observed['size']:raise ValueError('metadata evolved after byte reader')
    return value
bootstrap=checked(NEW+'bootstrap.py').decode()
own_block=bootstrap.split('OWN_NAMES = ',1)[1].split('\nFIXED = {',1)[0]
own_names=set(re.findall(r"'([^']+)'",own_block))
fixed_block=bootstrap.split('\nFIXED = {',1)[1].split('\n}',1)[0]
fixed=dict(re.findall(r"^\s*'([^']+)'\s*:\s*'([0-9a-f]{64})',?$",fixed_block,re.M))
if len(own_names)!=22 or len(fixed)!=13:raise ValueError('source literal memberships differ')
freeze=json.loads(checked(NEW+'SOURCE_PREPARATION_FREEZE.json'))
runtime=json.loads(checked(NEW+'RUNTIME_SOURCE_FREEZE.json'))
policy=json.loads(checked(NEW+'ADMISSION_POLICY.json'))
preexec=json.loads(checked(NEW+'RUNTIME_COLLECTOR_PREEXECUTION.json'))
dependencies=[r for label in ('core_fixed','core_result_fixed','producer_fixed','living_transition_fixed') for r in policy[label].values()]
dependencies.extend(policy[label] for label in ('preserved_failed_full_execution','preserved_focused_execution'))
deps={r['path']:r for r in dependencies}
required={NEW+n for n in own_names}|set(fixed)|set(deps)
required.update(r['raw_archive']['path'] for r in runtime['runtime_bindings'])
required.add(preexec['raw_collector_source']['path'])
required.update(r['archive_path'] for r in freeze['raw_source_archives'])
actual={r['path']:r for r in freeze['frozen_inputs']}
if len(actual)!=276 or set(actual)!=required:raise ValueError('bootstrap required closure differs from actual frozen membership')
if any(actual[p]!=row for p,row in deps.items()):raise ValueError('supplemental actual admission metadata binding substituted')
if any(actual[p]['sha256']!=digest for p,digest in fixed.items()):raise ValueError('fixed helper metadata identity substituted')
operative=sorted({NEW+n for n in own_names if n.endswith('.py')}|{p for p in fixed if p.endswith('.py')})
if operative!=freeze['operative_source_paths']:raise ValueError('complete source closure differs from bootstrap')
out={'status':'PASS_EXACT_BOOTSTRAP_SUPPLEMENTAL_METADATA_MEMBERSHIP_ONLY','author_execution':'/root/formal_oracle',
     'required_actual_members':len(required),'own_source_policy_names':len(own_names),'fixed_dependencies':len(fixed),
     'actual_admission_dependency_paths':len(deps),'operative_source_paths':len(operative),
     'bound_prior_byte_readback_sha256':sha256(readback_raw).hexdigest(),
     'own_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
     'reviewed_import_compile_tests_phase_inventory_git_network':0,'data_approval':False}
with (HERE/'SUPPLEMENTAL_METADATA_READBACK.json').open('x') as f:json.dump(out,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(out,sort_keys=True))
