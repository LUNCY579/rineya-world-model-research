from pathlib import Path
import hashlib,json
L=Path('/Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset/backup_logs')
p=L/'recover_remote_checkpoint_exact_public_metadata_v2.py';r=p.read_text()
pos=r.index('def validate_metadata_report')
r=r[:pos]+'''def typed_equal(a,b):
    if type(a) is not type(b):return False
    if type(b) is dict:return set(a)==set(b) and all(typed_equal(a[k],b[k]) for k in b)
    if type(b) is list:return len(a)==len(b) and all(typed_equal(x,y) for x,y in zip(a,b))
    return a==b


'''+r[pos:]
r=r.replace("or value.get('matches')!=row['matches'] or value.get(flag) is not True", "or not typed_equal(value.get('matches'),row['matches']) or value.get(flag) is not True")
pos=r.index('def full_checkpoint_guards')
r=r[:pos]+'''def validate_stage_observation(stage,m):
    expected={'status':'PUSHED_REMOTE_BYTES_VERIFICATION_PENDING','commit_sha':EXPECTED_COMMIT,'remote_commit_sha':EXPECTED_COMMIT,
        'expected_parent_sha':EXPECTED_PARENT,'spec_sha256':EXPECTED_SPEC_SHA,'source_sha256':STAGE_SOURCE_SHA,'source_freeze_sha256':STAGE_FREEZE_SHA,
        'files':3158,'total_bytes':1263090770,'all_source_target_index_sha256_and_sha1_verified':True,
        'whole_index_and_outside_parent_guard':'PASS','protected_original_baselines_verified':True,'actual_release_coverage_verified':222,
        'readonly_reference_count_verified':65,'restricted_strict_diff_check':'PASS','restricted_changed_path_count':482,'excluded_exact_path_count':2}
    for key,value in expected.items():
        if not typed_equal(stage.get(key),value):raise ValueError('Actual stage binding/guard differs: '+key)
    validate_metadata_report(stage.get('secret_scan',{}))
    import base64
    result={'returncode':stage.get('actual_exit_code'),'stdout':base64.b64decode(stage['actual_stdout_base64'],validate=True),
            'stderr':base64.b64decode(stage['actual_stderr_base64'],validate=True)}
    config=json.loads((FACTORY/'backup_logs/PROSPECTIVE_ARCHIVAL_V3_EXACT_EXCEPTION.json').read_bytes())
    actual=m.validate_diagnostics(result,config)
    if any(not typed_equal(stage.get(k),v) for k,v in actual.items()):raise ValueError('Exact six-warning diagnostics differ')
    return {'unrestricted_exit':2,'exception':'EXACT_BOUND_SIX_WARNINGS_MATCH','restricted_strict_pass_paths':482}


'''+r[pos:]
r=r.replace("    m.validate_execution_config(config,args,spec,prior,checked)", "    stage=json.loads((FACTORY/'backup_logs/belief-v4-approved-release-r1_STAGE.json').read_bytes())\n    validate_stage_observation(stage,m)\n    m.validate_execution_config(config,args,spec,prior,checked)")
r=r.replace("        'sole_parent_sha':EXPECTED_PARENT", "        'unrestricted_diff_exit':2,'frozen_exact_trailing_whitespace_warnings':6,'restricted_strict_paths':482,\n        'sole_parent_sha':EXPECTED_PARENT")
p.write_text(r)
sha=hashlib.sha256(p.read_bytes()).hexdigest()
p=L/'verify_archival_remote_lossless_exact_public_metadata_v2.py';g=p.read_text().replace('f1faf86d1092720c2818a430240bd9113d439d424341b8856ce32bb878eb6ebf',sha)
p.write_text(g)
print(json.dumps({'recovery':sha,'gzip':hashlib.sha256(p.read_bytes()).hexdigest()}))
