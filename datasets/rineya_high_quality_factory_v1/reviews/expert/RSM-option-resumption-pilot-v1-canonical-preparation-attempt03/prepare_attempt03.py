"""Static own source preparation only. No semantic module import, compilation,
tests, candidate target parsing or comparison. Attempt02 failure is immutable.
"""
import datetime
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OLD=HERE.parent/'RSM-option-resumption-pilot-v1-canonical-preparation-attempt02'
def sha(b):return hashlib.sha256(b).hexdigest()
def put(p,b):
 with p.open('xb') as f:f.write(b)
def write_json(p,v):put(p,(json.dumps(v,sort_keys=True,indent=2)+'\n').encode())

expected={'comparison_review.py':'7065db0dc411fd131652c9e6ccf3ef6aaacfcb7854e8739af7fb337865f0682c',
 'own_normalization.py':'d422159fdc02255ba0dc30e6022e9c1897b4b4264d5151e43ace42d21f67151c',
 'new_calibrations.py':'78753a8d3c39e3ff2756baa54b9bb223d27deb2220c3196cf3baae579853c344',
 'custody_guard.source-fragment.txt':'ef1a9a0b5d1e6c7ecf81ab9e54f857d0fa0baedd07f4140bdc88c784ceaf5981'}
copies=[]
for n,h in expected.items():
 b=(OLD/n).read_bytes()
 if sha(b)!=h:raise ValueError('protected owned source differs: '+n)
 put(HERE/n,b);copies.append({'path':n,'bytes':len(b),'sha256':h,'whole_file_exact_unchanged':True})
write_json(HERE/'STATIC-SEMANTIC-PRESERVATION.json',{'actual_author_execution_id':'/root/expert_reviewer',
 'scope':'WHOLE_FILE_BYTE_IDENTITY_CHECK_ONLY_NO_IMPORT_COMPILE_CALIBRATION_OR_TARGET_COMPARISON',
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preserved':copies,
 'comparison_logic_custody_guard_normalizer_calibrations_all_exact_unchanged':True})

cfg=json.loads((OLD/'config.json').read_text())
cfg['scope']='STATIC_ATTEMPT03_STARTUP_RUNTIME_CLOSURE_REPAIR_ONLY_NO_SEMANTIC_IMPORT_CALIBRATION_TARGET_COMPARISON_OR_RECEIPTS'
cfg['preparation_attempt']='03'
cfg['runtime_preparation']={'real_comparator_and_emitter_argparse_shapes_materialized':True,
 'declared_additional_stdlib_startup_modules':['shutil','tempfile','bz2','lzma'],
 'collector_external_source_and_config_SHA_and_exact_preexecution_archives_required':True,
 'existing_exact_raw_archive_hardlink_reuse_allowed':True,
 'guard_weakened':False,'actual_comparison_or_emitter_invocation_in_preparation':False}
cfg['disclosures'].append('Attempt02 actual comparison failed at initial runtime membership before all own helper imports/calibrations/targets: _bz2 was absent from frozen collector closure. Preserve exact stdout/stderr/failure and original source. Attempt03 materializes actual argparse comparison/emitter startup and declared stdlib lazy dependencies in custody bootstrap only; entire main/guards/semantic/helpers/calibrations unchanged, no correctness finding inferred.')
write_json(HERE/'config.json',cfg)
procedure=(OLD/'PROCEDURE.md').read_text()+'''

Attempt03 repairs only runtime discovery/paths after the recorded attempt02
startup failure. The entire comparison main, including guards, and both helpers,
normalizer and calibration file are byte-identical to attempt02. Its failure
FIRST/execution and all source/config/raw/stdout/stderr remain protected. _bz2
was observed before helper imports; this establishes a startup-closure omission,
not a target, semantic contract or dataset defect. No silent retry is performed.

The new collector receives mandatory external source/config SHA values. Exact
source/config raw preexecution archives and SHA/length equality must be checked
by an external stdlib-only launcher before launching it and checked again by
the collector. This bootstrap is disclosed; it is not a pre-bootstrap OS freeze.
Before collecting closure, it constructs and parses the actual compare and
receipts argparse CLI shape using benign placeholder values. It executes help
formatting and preserves the parsed benign CLI settings as runtime materialization
evidence, without calling comparator/emitter functions. It explicitly imports
shutil, tempfile, bz2 and lzma plus known stdlib dependencies and the exception
formatting path. No own solver/calibration/normalizer/comparison source is imported.
Module source/cache/extensions, local linked libraries and resolved Python are
collected from actual loaded modules and verified against exact raw archives.
Strict membership guards in the comparator/emitter remain unchanged.

Raw archive reuse is confined to verified exact older own raw/runtime archives:
new content-addressed paths may be hardlinks to existing bytes after exact
SHA/size/equality checks. Source files themselves are copied as independent
files. No symlink import/archive shortcut, metadata-only pointer or new producer
payload alias is admitted. Existing unextracted Apple shared-cache dependencies
remain disclosed. This is preparation evidence only. Future actual comparison
and all receipt publication still require separate root readback and grant.
'''
put(HERE/'PROCEDURE.md',procedure.encode())

inventory=[]
for folder in [HERE.parent/'RSM-option-resumption-pilot-v1-canonical-preparation-attempt01',OLD,
 ROOT/'reviews/expert/RSM-option-resumption-pilot-v1-input-first-attempt01']:
 for p in sorted(folder.rglob('*')):
  if p.is_file():
   b=p.read_bytes();inventory.append({'path':str(p.relative_to(ROOT)),'absolute_path':str(p.resolve()),'sha256':sha(b),'bytes':len(b)})
write_json(HERE/'PROTECTED-ORIGINAL-INVENTORY.json',{'actual_author_execution_id':'/root/expert_reviewer',
 'scope':'HASH_ONLY_PROTECTED_INPUT_FIRST_ATTEMPT01_ATTEMPT02_AND_FAILED_EXECUTION_RAW_ALL_UNCHANGED',
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':inventory})

s=(OLD/'collect_source_runtime_freeze.py').read_text()
marker="source_bytes=Path(__file__).read_bytes();source_hash=sha(source_bytes)\ncfg=json.loads((HERE/'config.json').read_text())\n"
replacement="""source_bytes=Path(__file__).read_bytes();source_hash=sha(source_bytes)
custody_parser=argparse.ArgumentParser(description='Expert stdlib-only source/runtime custody preparation')
custody_parser.add_argument('--expected-collector-source-sha256',required=True)
custody_parser.add_argument('--expected-config-sha256',required=True)
custody_args=custody_parser.parse_args()
config_bytes=(HERE/'config.json').read_bytes()
for value in [custody_args.expected_collector_source_sha256,custody_args.expected_config_sha256]:
 if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:raise ValueError('external collector/config exact SHA256 required')
if source_hash!=custody_args.expected_collector_source_sha256 or sha(config_bytes)!=custody_args.expected_config_sha256:
 raise ValueError('external collector/config SHA mismatch')
if (HERE/'COLLECTOR-SOURCE-PREEXECUTION.raw').read_bytes()!=source_bytes or (HERE/'CONFIG-PREEXECUTION.raw').read_bytes()!=config_bytes:
 raise ValueError('exact external preexecution collector/config raw archive mismatch')
cfg=json.loads(config_bytes)

# Actual CLI shape copied from the exact own comparator, not imported from it.
# Benign placeholders exercise startup paths only; no compare/emitter invocation.
runtime_parser=argparse.ArgumentParser();runtime_parser.add_argument('phase',choices=['compare','receipts']);runtime_parser.add_argument('--quality-map')
runtime_parser.add_argument('--expected-config-sha256',required=True);runtime_parser.add_argument('--expected-source-freeze-sha256',required=True)
runtime_parser.add_argument('--expected-quality-map-sha256');runtime_parser.add_argument('--expected-quality-freeze-sha256')
benign_compare=runtime_parser.parse_args(['compare','--expected-config-sha256',custody_args.expected_config_sha256,'--expected-source-freeze-sha256','0'*64])
benign_receipts=runtime_parser.parse_args(['receipts','--quality-map','NEVER_OPENED_BENIGN_RUNTIME_PLACEHOLDER.json','--expected-config-sha256',custody_args.expected_config_sha256,'--expected-source-freeze-sha256','0'*64,'--expected-quality-map-sha256','0'*64,'--expected-quality-freeze-sha256','0'*64])
runtime_help=runtime_parser.format_help()
import shutil
import tempfile
import bz2
import lzma
try:raise RuntimeError('benign stdlib exception-format startup materialization; no semantic test')
except RuntimeError:startup_exception_text=traceback.format_exc()
startup={'scope':'BENIGN_STDLIB_STARTUP_PATH_MATERIALIZATION_NOT_SEMANTIC_CALIBRATION_OR_COMPARISON',
 'actual_comparator_cli_shape_constructed':True,'actual_compare_args_parsed':vars(benign_compare),'actual_receipts_args_parsed':vars(benign_receipts),
 'placeholder_hashes_are_not_target_expectations_or_execution_authority':True,'help_text_sha256':sha(runtime_help.encode()),
 'declared_lazy_stdlib_imports':['shutil','tempfile','bz2','lzma'],'exception_format_path_materialized':True,
 'exception_text_sha256':sha(startup_exception_text.encode()),'own_semantic_module_imported':False,
 'external_source_SHA':custody_args.expected_collector_source_sha256,'external_config_SHA':custody_args.expected_config_sha256}
"""
if s.count(marker)!=1:raise ValueError('collector entry marker not unique')
s=s.replace(marker,replacement,1)
list_old="['comparison_review.py','own_normalization.py','new_calibrations.py','config.json','PROCEDURE.md','collect_source_runtime_freeze.py','custody_guard.source-fragment.txt','author_attempt02.py','author_attempt02.py.before-source-text-execution.raw','STATIC-SEMANTIC-PRESERVATION.json','PROTECTED-ORIGINAL-INVENTORY.json','SOURCE-TEXT-AUTHORING-EXECUTION.json']"
list_new="['comparison_review.py','own_normalization.py','new_calibrations.py','config.json','PROCEDURE.md','collect_source_runtime_freeze.py','custody_guard.source-fragment.txt','prepare_attempt03.py','STATIC-SEMANTIC-PRESERVATION.json','PROTECTED-ORIGINAL-INVENTORY.json','COLLECTOR-SOURCE-PREEXECUTION.raw','CONFIG-PREEXECUTION.raw','CUSTODY-COLLECTOR-PREEXECUTION.json']"
if s.count(list_old)!=1:raise ValueError('collector own path marker not unique')
s=s.replace(list_old,list_new,1)
marker="for own in [ROOT/cfg['own_solver'],ROOT/cfg['own_calibration'],HERE/'comparison_review.py',HERE/'own_normalization.py',HERE/'new_calibrations.py']:"
s=s.replace(marker,"""failed='reviews/expert/RSM-option-resumption-pilot-v1-canonical-preparation-attempt02/'
for n in ['comparison_review.py','config.json','collect_source_runtime_freeze.py','SOURCE-PREPARATION-FREEZE.json','SOURCE-PREPARATION-FINAL.json','ACTUAL-COMPARE-STDOUT.txt','ACTUAL-COMPARE-STDERR.txt','FAILURE-compare.txt','ACTUAL-COMPARE-FAILED-EXECUTION.json','ACTUAL-COMPARE-FAILED-FIRST.json']:
 paths[failed+n]='OWN_PROTECTED_ATTEMPT02_AND_ACTUAL_STARTUP_FAILURE_NOT_IMPORTED'
"""+marker,1)
# Verified hardlink reuse of exact own archives; no new opaque alias destination
# points at a symlink or is admitted without byte equality.
marker="failures=[];bindings=[]\n"
reuse="""def store_exact_archive(destination,raw):
 if destination.exists():
  if destination.read_bytes()!=raw:raise ValueError('existing archive does not equal exact current bytes')
  return
 for earlier in ['RSM-option-resumption-pilot-v1-canonical-preparation-attempt02','RSM-option-resumption-pilot-v1-canonical-preparation-attempt01']:
  possible=HERE.parent/earlier/destination.parent.name/destination.name
  if possible.is_file() and not possible.is_symlink() and possible.read_bytes()==raw:
   os.link(possible,destination)
   if destination.read_bytes()!=raw:raise ValueError('hardlinked exact archive changed during reuse')
   return
 with destination.open('xb') as f:f.write(raw)
 if destination.read_bytes()!=raw:raise ValueError('new exact raw archive write differs')
failures=[];bindings=[]
"""
s=s.replace(marker,reuse,1)
s=s.replace(" if not ap.exists():ap.write_bytes(b)\n assert ap.read_bytes()==b", " store_exact_archive(ap,b)\n assert ap.read_bytes()==b",1)
s=s.replace(" if not a.exists():a.write_bytes(b)\n assert a.read_bytes()==b", " store_exact_archive(a,b)\n assert a.read_bytes()==b",1)
s=s.replace("'runtime':{'python_executable':", "'startup_materialization':startup,'runtime':{'python_executable':",1)
s=s.replace('RSM-option-resumption-pilot-v1-canonical-preparation-attempt02/collect_source_runtime_freeze.py','RSM-option-resumption-pilot-v1-canonical-preparation-attempt03/collect_source_runtime_freeze.py --expected-collector-source-sha256 <EXTERNALLY_SUPPLIED_COLLECTOR_SOURCE_SHA256> --expected-config-sha256 <EXTERNALLY_SUPPLIED_CONFIG_SHA256>')
s=s.replace("'actual_freeze_command':'/opt/anaconda3/bin/python3 -I -B -S reviews/expert/RSM-option-resumption-pilot-v1-canonical-preparation-attempt03/collect_source_runtime_freeze.py --expected-collector-source-sha256 <EXTERNALLY_SUPPLIED_COLLECTOR_SOURCE_SHA256> --expected-config-sha256 <EXTERNALLY_SUPPLIED_CONFIG_SHA256>',", "'actual_freeze_command':'/opt/anaconda3/bin/python3 -I -B -S '+str(Path(__file__).resolve().relative_to(ROOT))+' '+' '.join(sys.argv[1:]),")
s=s.replace('RSM-option-resumption-pilot-v1-canonical-preparation-attempt02/comparison_review.py compare','RSM-option-resumption-pilot-v1-canonical-preparation-attempt03/comparison_review.py compare')
s=s.replace("'scope':'ATTEMPT02_SOURCE_ONLY_CUSTODY_REPAIR_AWAITING_ROOT_READBACK_AND_EXECUTION_GRANT'", "'scope':'ATTEMPT03_SOURCE_ONLY_REAL_CLI_RUNTIME_CLOSURE_REPAIR_AWAITING_ROOT_READBACK_AND_EXECUTION_GRANT'")
s=s.replace("'semantic_preservation':json.loads((HERE/'STATIC-SEMANTIC-PRESERVATION.json').read_text())", "'semantic_preservation':json.loads((HERE/'STATIC-SEMANTIC-PRESERVATION.json').read_text()),'startup_materialization':startup,'exact_archive_hardlink_reuse_permitted':True,'attempt02_failure_recorded_preserved_no_retry':True")
# No collector source is run/imported/compiled by this static authoring program.
put(HERE/'collect_source_runtime_freeze.py',s.encode())
print(json.dumps({'actual_author_execution_id':'/root/expert_reviewer','scope':'STATIC_SOURCE_TEXT_ONLY',
 'main_byteidentical_SHA':expected['comparison_review.py'],'config_SHA':sha((HERE/'config.json').read_bytes()),
 'collector_SHA':sha(s.encode()),'protected_original_files':len(inventory),'no_semantic_import_test_or_target_comparison':True}))
