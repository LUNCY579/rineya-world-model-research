from pathlib import Path
import hashlib,json
L=Path('/Users/chengli/Downloads/Rineya_World_Model/High-Quality_Dataset/backup_logs')
p=L/'recover_remote_checkpoint_exact_public_metadata_v2.py';s=p.read_text()
extra="""METADATA_GIT_BLOBS={'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/FINAL_RECEIPT.json':'7e9d12e54b4235d20653fdbccf74e689097ed84f',
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/attempt01-execution.json':'b40be3a0a775a8118f45fe51ba339b3161452ad8',
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/attempt01-source-blobs/c35c317807c522a5d695528c4937ee755f8bb0e938f0c6ba7c2ec60a1ed6dc5e.source':'62ccac42b959c7cead65ddf689f298de6c18b0bb',
 'reviews/oracle/RSM-option-resumption-official-20261010T0932Z/run_task_solver_attempt01.py':'62ccac42b959c7cead65ddf689f298de6c18b0bb'}
MANDATORY_FROZEN_INPUTS={'backup_logs/recover_remote_checkpoint.py':ORIGINAL_SOURCE_SHA,
 'backup_logs/exact_four_public_metadata_secret_scan_attempt02.py':METADATA_SOURCE_SHA,
 'backup_logs/EXACT_FOUR_PUBLIC_ID_SECRET_EXCEPTION_CONFIG.json':METADATA_CONFIG_SHA,
 'backup_logs/run_provisional_archival_checkpoint_v4_attempt02.py':STAGE_SOURCE_SHA,
 'backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_STAGE_V4_ATTEMPT02_EXECUTION_CONFIG.json':STAGE_CONFIG_SHA,
 'backup_logs/PROSPECTIVE_ARCHIVAL_V4_ATTEMPT02_SOURCE_FREEZE.json':STAGE_FREEZE_SHA,
 'backup_logs/PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER.json':CUSTODY_SHA,
 'backup_logs/PROSPECTIVE_ARCHIVAL_V3_EXACT_EXCEPTION.json':WARNING_CONFIG_SHA,
 'backup_logs/belief-v4-approved-release-r1_SPEC.json':EXPECTED_SPEC_SHA,
 'backup_logs/belief-v4-approved-release-r1_STAGE.json':'68e764dc40fad0d2b28c351cb0fd5bfea2dff4f628323b48e14b406f281b530e',
 'backup_logs/belief-v4-approved-release-r1_LOCAL_COMMIT.json':'ca8902559856f61552609ff8fd3d024a52b5f4fbd5150600e66aace7a8075976',
 'backup_logs/belief-v4-approved-release-r1_STAGE_LEDGER.jsonl':'b79917cb31f677a0da447d4f84a683728e4daa9a755cae797abb0bb8024e90f3',
 'backup_logs/belief-v4-validated-code-candidate-r2_SPEC.json':'6b310f0656a468dd65920c64f15efe82e0b592d3d76250366f9555cef46aaad5',
 'quality_reports/RGM_RENDERER_ARCHIVAL_BOUNDARY_PREFLIGHT.json':'03ec34f4f778698c9dc52b3bd4624f324abc525aaca1e33417d4c6042d8168a1'}
REQUIRED_NEW_PREPARATION_PATHS={'backup_logs/recover_remote_checkpoint_exact_public_metadata_v3.py',
 'backup_logs/verify_archival_remote_lossless_exact_public_metadata_v3.py','backup_logs/test_remote_checkpoint_exact_public_metadata_v3.py',
 'backup_logs/test_archival_remote_lossless_exact_public_metadata_v3.py','backup_logs/REMOTE_EXACT_PUBLIC_METADATA_V3_CONFIG.json'}
"""
s=s.replace("_METADATA=None",extra+"\n_METADATA=None")
s=s.replace("    helper=metadata_helper();row=canonical_stream_paths(original_paths)", "    if type(git_blob) is not str or not re.fullmatch(r'[0-9a-f]{40}',git_blob):raise ValueError('Exact expected Git object ID required before stream')\n    if context not in ('local_git','actual_remote'):raise ValueError('Only actual localGit/remote stream context accepted')\n    helper=metadata_helper();row=canonical_stream_paths(original_paths)")
s=s.replace('def validate_metadata_report(report,stream=False):','def validate_metadata_report(report,stream=False,expected_context=None):')
s=s.replace("    rows=report.get('four_exception_files')", "    if type(stream) is not bool:raise ValueError('Metadata report stream typing differs')\n    if stream:\n        if expected_context not in ('local_git','actual_remote') or report.get('context')!=expected_context:raise ValueError('Metadata report actual provenance context differs')\n    elif expected_context is not None or 'context' in report:raise ValueError('Source report must not substitute a stream context')\n    rows=report.get('four_exception_files')")
s=s.replace("not re.fullmatch(r'[0-9a-f]{40}',value.get('git_blob','')) or report.get('scanner_source_sha256')!=METADATA_SOURCE_SHA", "value.get('git_blob')!=METADATA_GIT_BLOBS[value['path']] or report.get('scanner_source_sha256')!=METADATA_SOURCE_SHA")
pos=s.index('def check_new_source_freeze')
s=s[:pos]+'''def validate_frozen_inputs(rows):
    if type(rows) is not list or not rows or any(type(row) is not dict or set(row)!={'path','size','sha256'} for row in rows):
        raise ValueError('Complete typed preparation dependency list required')
    names=[row['path'] for row in rows]
    if len(set(names))!=len(names) or not (set(MANDATORY_FROZEN_INPUTS)|REQUIRED_NEW_PREPARATION_PATHS)<=set(names):
        raise ValueError('Missing/duplicate frozen preparation dependency')
    for row in rows:
        relative_path(row['path'])
        if type(row['size']) is not int or row['size']<0 or type(row['sha256']) is not str or not re.fullmatch(r'[0-9a-f]{64}',row['sha256']):
            raise ValueError('Frozen dependency size/hash typing differs')
        if row['path'] in MANDATORY_FROZEN_INPUTS and row['sha256']!=MANDATORY_FROZEN_INPUTS[row['path']]:raise ValueError('Fixed dependency SHA differs')
        if hash_file(source_path(FACTORY,row['path']))!={k:row[k] for k in ('size','sha256')}:raise ValueError('Frozen preparation dependency changed')
    return True


'''+s[pos:]
s=s.replace("    for row in freeze['frozen_inputs']:\n        if hash_file(source_path(FACTORY,row['path']))!={'size':row['size'],'sha256':row['sha256']}:\n            raise ValueError('Frozen preparation input changed')", "    validate_frozen_inputs(freeze['frozen_inputs'])")
s=s.replace('validate_metadata_report(complete_metadata_stream_report("local_git"),True)','validate_metadata_report(complete_metadata_stream_report("local_git"),True,"local_git")')
s=s.replace('validate_metadata_report(complete_metadata_stream_report("actual_remote"),True)','validate_metadata_report(complete_metadata_stream_report("actual_remote"),True,"actual_remote")')
p=L/'recover_remote_checkpoint_exact_public_metadata_v3.py';p.open('x').write(s);sha=hashlib.sha256(p.read_bytes()).hexdigest()
g=(L/'verify_archival_remote_lossless_exact_public_metadata_v2.py').read_text().replace('39ae8005427f41feb30342d5a9b741649324656eb1e3d0d4463e23d4f6caef91',sha).replace('recover_remote_checkpoint_exact_public_metadata_v2.py','recover_remote_checkpoint_exact_public_metadata_v3.py')
g=g.replace("bounded.validate_metadata_report(verified.get(key,{}),key.startswith('actual_'))", "bounded.validate_metadata_report(verified.get(key,{}),key.startswith('actual_'),{'actual_local_git_metadata_exception_scan':'local_git','actual_remote_metadata_exception_scan':'actual_remote'}.get(key))")
g=g.replace('    prior_raws =', '''    for key in ('actual_local_git_metadata_exception_scan','actual_remote_metadata_exception_scan'):
        for row in verified[key]['four_exception_files']:
            if actual_by_path.get(row['path'],{}).get('git_blob')!=row['git_blob']:
                raise ValueError('Metadata actual stream Git ID differs from frozen remote path coverage')
    if verified.get('current_source_evolution')!=[] or verified.get('current_source_evolution_at_finish')!=[]:
        raise ValueError('Held checkpoint does not permit evolving sources')
    prior_raws =''')
g=g.replace("    for row in freeze['frozen_inputs']:\n        if bounded.hash_file(bounded.source_path(FACTORY,row['path']))!={k:row[k] for k in ('size','sha256')}:\n            raise ValueError('Frozen gzip preparation input changed')", "    bounded.validate_frozen_inputs(freeze['frozen_inputs'])")
p=L/'verify_archival_remote_lossless_exact_public_metadata_v3.py';p.open('x').write(g)
print(json.dumps({'recovery':sha,'gzip':hashlib.sha256(p.read_bytes()).hexdigest()}))
