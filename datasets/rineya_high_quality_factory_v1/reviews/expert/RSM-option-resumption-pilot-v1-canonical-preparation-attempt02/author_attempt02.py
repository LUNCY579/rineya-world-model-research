"""Source-text custody repair only. Does not import, compile, test or execute
the reviewer/solver/normalizer/calibration, or parse candidate target rows.
"""
import datetime
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=HERE.parent/'RSM-option-resumption-pilot-v1-canonical-preparation-attempt01'
def sha(b):return hashlib.sha256(b).hexdigest()
def put(p,b):
    with p.open('xb') as f:f.write(b)
def data(p,v):put(p,(json.dumps(v,sort_keys=True,indent=2)+'\n').encode())

original=(OLD/'comparison_review.py').read_bytes()
if len(original)!=35354 or sha(original)!='b11b0c9c1f077c337a94cf80851c40ccca73e5747c0cfaa12475f4cb63311951':
    raise ValueError('immutable attempt01 main differs')
text=original.decode()
text=text.replace("VERSION='expert-rsm-canonical-1.0.1'","VERSION='expert-rsm-canonical-1.0.2'",1)
text=text.replace("spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m",
  "spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m",1)
guard=(HERE/'custody_guard.source-fragment.txt').read_text()
text=text.replace('\ndef identifier(entry):','\n'+guard+'\n\ndef identifier(entry):',1)
old_begin="""    cfg=json.loads((HERE/'config.json').read_text());freeze=json.loads((HERE/'SOURCE-PREPARATION-FREEZE.json').read_text())
    for item in freeze['bindings']:
        p=Path(item['absolute_path']);assert sha(p.read_bytes())==item['sha256'],item['path']
        assert (HERE/item['archive_path']).read_bytes()==p.read_bytes()
    for item in freeze['runtime']['file_bindings']:assert sha(Path(item['absolute_path']).read_bytes())==item['sha256']
"""
if text.count(old_begin)!=1:raise ValueError('comparison custody entry not uniquely located')
text=text.replace(old_begin,"    cfg,freeze,custody_before=frozen_guard('COMPARE_BEFORE_HELPER_IMPORTS')\n",1)
old_end="    assert all(sha(Path(item['absolute_path']).read_bytes())==item['sha256'] for item in freeze['bindings'])\n"
if text.count(old_end)!=1:raise ValueError('comparison custody exit not uniquely located')
text=text.replace(old_end,"    _,_,custody_after=frozen_guard('COMPARE_AFTER_CALIBRATION_AND_COMPARISON')\n",1)
text=text.replace("'source_freeze_sha256':sha((HERE/'SOURCE-PREPARATION-FREEZE.json').read_bytes()),'frozen_first_predictions_unchanged'",
  "'source_freeze_sha256':CUSTODY['expected_source_freeze_sha256'],'config_sha256':CUSTODY['expected_config_sha256'],'custody_before':custody_before,'custody_after':custody_after,'frozen_first_predictions_unchanged'",1)
old_quality="""    quality_path=Path(quality_path);q=json.loads(quality_path.read_text());auth=json.loads((HERE/'QUALITY-MAP-FREEZE.json').read_text())
    assert q['actual_author_execution_id']==ACTOR and auth['quality_map_sha256']==sha(quality_path.read_bytes())
"""
if text.count(old_quality)!=1:raise ValueError('receipt custody entry not uniquely located')
text=text.replace(old_quality,"""    start=datetime.datetime.now(datetime.timezone.utc).isoformat();timer=time.perf_counter()
    _,_,custody_before=frozen_guard('RECEIPTS_BEFORE_QUALITY_INPUTS')
    quality_path=Path(quality_path).resolve();q,quality_before=quality_guard(quality_path,'RECEIPTS_BEFORE_BUILD')
""",1)
old_publish="""    first=write_jsonl(HERE/'canonical-judgments.first.jsonl',rs)
    output=ROOT/'reviews/expert/option-resumption-pilot-v1.jsonl';final=write_jsonl(output,rs)
    print(canonical({'first':first,'final':final,'decisions':dict(collections.Counter(r['decision'] for r in rs)),'publication_approval':False}))
"""
new_publish="""    # Recheck all current FIRST, canonical, comparison, quality and runtime
    # bytes immediately before either immutable receipt is written.
    _,_,custody_prewrite=frozen_guard('RECEIPTS_IMMEDIATELY_BEFORE_FIRST_WRITE')
    _,quality_prewrite=quality_guard(quality_path,'RECEIPTS_IMMEDIATELY_BEFORE_FIRST_WRITE')
    first=write_jsonl(HERE/'canonical-judgments.first.jsonl',rs)
    _,_,custody_before_final=frozen_guard('RECEIPTS_BEFORE_DESIGNATED_FINAL_WRITE')
    _,quality_before_final=quality_guard(quality_path,'RECEIPTS_BEFORE_DESIGNATED_FINAL_WRITE')
    output=ROOT/'reviews/expert/option-resumption-pilot-v1.jsonl';final=write_jsonl(output,rs)
    _,_,custody_after=frozen_guard('RECEIPTS_AFTER_PUBLICATION')
    _,quality_after=quality_guard(quality_path,'RECEIPTS_AFTER_PUBLICATION')
    first_bytes=(HERE/'canonical-judgments.first.jsonl').read_bytes();final_bytes=output.read_bytes()
    if first_bytes!=final_bytes or sha(first_bytes)!=first['sha256'] or len(first_bytes)!=first['bytes'] or sha(final_bytes)!=final['sha256'] or len(final_bytes)!=final['bytes']:
        raise ValueError('published receipt bytes differ from the exact newly written first/final descriptors')
    execution=write_json('RECEIPTS-EXECUTION.json',{'actual_author_execution_id':ACTOR,
      'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-timer,
      'argv':sys.argv,'exit_code':0,'source_version':VERSION,'config_sha256':CUSTODY['expected_config_sha256'],
      'source_freeze_sha256':CUSTODY['expected_source_freeze_sha256'],'quality_map_sha256':CUSTODY['expected_quality_map_sha256'],
      'quality_map_freeze_sha256':CUSTODY['expected_quality_freeze_sha256'],
      'custody_before':custody_before,'custody_prewrite':custody_prewrite,'custody_before_final':custody_before_final,'custody_after':custody_after,
      'quality_before':quality_before,'quality_prewrite':quality_prewrite,'quality_before_final':quality_before_final,'quality_after':quality_after,
      'first':first,'final':final,'exact_first_final_byte_equality':True,'publication_approval':False})
    print(canonical({'first':first,'final':final,'execution':execution,'decisions':dict(collections.Counter(r['decision'] for r in rs)),'publication_approval':False}))
"""
if text.count(old_publish)!=1:raise ValueError('receipt publication block not uniquely located')
text=text.replace(old_publish,new_publish,1)
old_cli="""    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['compare','receipts']);parser.add_argument('--quality-map');args=parser.parse_args()
    try:
"""
new_cli="""    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['compare','receipts']);parser.add_argument('--quality-map')
    parser.add_argument('--expected-config-sha256',required=True);parser.add_argument('--expected-source-freeze-sha256',required=True)
    parser.add_argument('--expected-quality-map-sha256');parser.add_argument('--expected-quality-freeze-sha256');args=parser.parse_args()
    try:
        # No own helper imports occur until these externally supplied values
        # are checked against actual bytes by the guards in the chosen phase.
        CUSTODY['expected_config_sha256']=require_sha(args.expected_config_sha256,'external config SHA')
        CUSTODY['expected_source_freeze_sha256']=require_sha(args.expected_source_freeze_sha256,'external source freeze SHA')
        if args.phase=='receipts':
            CUSTODY['expected_quality_map_sha256']=require_sha(args.expected_quality_map_sha256,'external quality map SHA')
            CUSTODY['expected_quality_freeze_sha256']=require_sha(args.expected_quality_freeze_sha256,'external quality freeze SHA')
"""
if text.count(old_cli)!=1:raise ValueError('CLI authority block not uniquely located')
text=text.replace(old_cli,new_cli,1)
put(HERE/'comparison_review.py',text.encode())
for n in ('own_normalization.py','new_calibrations.py'):put(HERE/n,(OLD/n).read_bytes())
cfg=json.loads((OLD/'config.json').read_text());cfg['source_version']='expert-rsm-canonical-1.0.2'
cfg['scope']='STATIC_ATTEMPT02_CUSTODY_REPAIR_ONLY_NO_REVIEWER_IMPORT_TEST_COMPARISON_OR_RECEIPT_EXECUTION'
cfg['custody_policy']={'external_config_SHA_required':True,'external_source_freeze_SHA_required':True,
  'comparison_guard_before_helper_imports_and_after':True,'receipt_guard_before_and_after':True,
  'external_quality_map_and_quality_freeze_SHA_required_for_receipts':True,
  'typed_SHA_length_exact_raw_archive_runtime_checks':True,'loaded_module_membership_checked':True,
  'resolved_python_equality_checked':True,'semantic_comparisons_and_normalization_unchanged':True}
cfg['disclosures'].append('Attempt02 only repairs custody/runtime/CLI/publication boundaries after root source read; its checker/helpers/calibrations and receipt publication have not been imported or executed during source preparation. Attempt01 is immutable; no new target comparison or verdict is inferred.')
data(HERE/'config.json',cfg)

# Static exact textual preservation, not execution or AST/bytecode validation.
old_text=original.decode();new_text=text
checks=[]
for label,start,end in [('semantic_helpers','def identifier(entry):','def write_json(name,data):'),
 ('comparison_semantic_loop',"    solver=load_module('expert_own_rsm_solver'", "    assert all(sha(Path(item['absolute_path']).read_bytes())==item['sha256'] for item in freeze['bindings'])"),
 ('receipt_quality_decision_loop','    compared=[json.loads(x)',"    first=write_jsonl(HERE/'canonical-judgments.first.jsonl',rs)")]:
    a=old_text[old_text.index(start):old_text.index(end,old_text.index(start))]
    newer_end={'comparison_semantic_loop':"    _,_,custody_after=frozen_guard('COMPARE_AFTER_CALIBRATION_AND_COMPARISON')",
      'receipt_quality_decision_loop':'    # Recheck all current FIRST, canonical, comparison, quality and runtime'}.get(label,end)
    b=new_text[new_text.index(start):new_text.index(newer_end,new_text.index(start))]
    if a!=b:raise ValueError('static semantic text changed: '+label)
    checks.append({'region':label,'bytes':len(a.encode()),'sha256':sha(a.encode()),'exact_text_unchanged':True})
for n in ('own_normalization.py','new_calibrations.py'):
    a=(OLD/n).read_bytes();b=(HERE/n).read_bytes()
    if a!=b:raise ValueError('immutable copied helper changed: '+n)
    checks.append({'region':n,'bytes':len(a),'sha256':sha(a),'exact_bytes_unchanged':True})
data(HERE/'STATIC-SEMANTIC-PRESERVATION.json',{'actual_author_execution_id':'/root/expert_reviewer',
  'scope':'EXACT_SOURCE_TEXT_CHECK_ONLY_NOT_IMPORT_COMPILE_TEST_OR_TARGET_EXECUTION','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'original_main_sha256':sha(original),'new_main_sha256':sha(text.encode()),'regions':checks})
protected=[]
for folder in (OLD,ROOT/'reviews/expert/RSM-option-resumption-pilot-v1-input-first-attempt01'):
    for p in sorted(folder.rglob('*')):
        if p.is_file():
            b=p.read_bytes();protected.append({'path':str(p.relative_to(ROOT)),'absolute_path':str(p.resolve()),'bytes':len(b),'sha256':sha(b)})
data(HERE/'PROTECTED-ORIGINAL-INVENTORY.json',{'actual_author_execution_id':'/root/expert_reviewer',
  'scope':'READ_ONLY_HASH_INVENTORY_OF_PROTECTED_ATTEMPT01_AND_INPUT_FIRST_FILES_INCLUDING_RAW_ARCHIVES',
  'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':protected,'no_original_file_modified':True})
print(json.dumps({'source_sha256':sha(text.encode()),'config_sha256':sha((HERE/'config.json').read_bytes()),'preserved_regions':len(checks),'protected_files':len(protected),'no_reviewer_source_import_compile_test_or_comparison':True}))
