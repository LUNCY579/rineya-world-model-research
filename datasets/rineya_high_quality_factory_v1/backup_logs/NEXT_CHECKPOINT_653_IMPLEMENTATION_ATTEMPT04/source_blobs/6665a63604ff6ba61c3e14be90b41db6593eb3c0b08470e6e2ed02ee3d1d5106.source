"""Every unique actual remote Git blob, two workers, bounded streams, no cache."""
from __future__ import annotations
import concurrent.futures
import next653_common as c
from next653_stage import validate_stage_report

def remote_tree(metadata):
    if type(metadata) is not dict or metadata.get('truncated') is not False or type(metadata.get('tree')) is not list:raise ValueError('Complete nontruncated remote tree required')
    rows={}
    for row in metadata['tree']:
        if row.get('type')=='tree':continue
        if row.get('type')!='blob' or row.get('mode') not in ('100644','100755','120000'):raise ValueError('Unsupported remote Git tree object')
        p=row.get('path')
        if type(p) is not str or p in rows:raise ValueError('Duplicate/invalid whole remote path')
        c.sha(row.get('sha'),40);rows[p]=(row['mode'],row['sha'])
    return rows

def validate_commit_metadata(value,commit,tree):
    if value.get('sha')!=commit or value.get('tree',{}).get('sha')!=tree or [r.get('sha') for r in value.get('parents',[])]!=[c.PARENT]:raise ValueError('Actual remote commit/tree/sole parent differs')
    return True

def verify_remote_metadata(spec,commit,tree):
    actual=c.api(f'repos/{c.REPOSITORY}/git/commits/{commit}');validate_commit_metadata(actual,commit,tree)
    wanted=c.verify_git_scope(spec,commit)
    actualtree=c.api(f'repos/{c.REPOSITORY}/git/trees/{tree}?recursive=1')
    if actualtree.get('sha')!=tree or remote_tree(actualtree)!=wanted:raise ValueError('Full actual remote tree/path coverage or outside-prefix differs')
    files,_=c.all_entries(spec);by_path={r['path']:r for r in actualtree['tree'] if r.get('type')=='blob'}
    for p,row in files.items():
        remote=by_path.get(c.PREFIX+'/'+p,{})
        if (remote.get('mode')!='100644' or remote.get('sha')!=row['git_blob']
                or type(remote.get('size')) is not int or remote['size']!=row['size']):raise ValueError('Actual remote path/blob metadata size or mode differs')
    return {'commit_sha':commit,'tree_sha':tree,'sole_parent_sha':c.PARENT,'complete_remote_path_coverage':True,'remote_tree_truncated':False}

def validate_stage_dependencies(cfg,spec):
    c.validate_spec_closure(cfg,spec)
    stage=c.bound_json(cfg['inputs']['stage']);commit=c.sha(cfg['fields'].get('commit_sha'),40)
    validate_stage_report(stage,spec,commit)
    if stage.get('spec')!=cfg['inputs']['spec'] or stage.get('inventory_report')!=cfg['inputs']['inventory_report'] or stage.get('local_commit')!=cfg['inputs']['local_commit'] or stage.get('stage_ledger')!=cfg['inputs']['stage_ledger']:raise ValueError('Stage exact dependency bindings differ')
    local=c.bound_json(cfg['inputs']['local_commit'])
    c.validate_completed_proof(cfg,'stage',stage,cfg['inputs']['stage'],spec)
    c.validate_completed_proof(cfg,'stage',local,cfg['inputs']['local_commit'],spec,local_commit=True)
    if local.get('status')!='LOCAL_COMMIT_PRESERVED_REMOTE_PENDING' or local.get('commit_sha')!=commit or local.get('sole_parent_sha')!=c.PARENT or local.get('spec')!=cfg['inputs']['spec'] or not c.typed_equal(local.get('exact_scope'),c.scope(spec)):raise ValueError('Local commit preserved evidence cross-wired')
    files,_=c.all_entries(spec);expected=c.groups(files)
    c.validate_actual_groups(stage['local_git_blob_groups'],expected,'actual_local_git_stream_this_run')
    c.validate_actual_groups(local['local_git_blob_groups'],expected,'actual_local_git_stream_this_run')
    copies=c.ledger_rows(cfg['inputs']['stage_ledger'],'source_target_verified')
    if len(copies)!=len(files) or len({r.get('path') for r in copies})!=len(files):raise ValueError('Stage source/target ledger coverage differs')
    for row in copies:
        if row.get('path') not in files or any(not c.typed_equal(row.get(k),files[row['path']][k]) for k in ('size','sha256','git_blob')):raise ValueError('Stage source/target bytes ledger cross-wired')
    c.validate_actual_groups(c.ledger_rows(cfg['inputs']['stage_ledger'],'local_git_blob_verified'),expected,'actual_local_git_stream_this_run')
    return stage,local,commit

def fetch_group(row,session,fetch=None):
    # Injection is a direct-function isolated-test seam, never a CLI/config field.
    fetch=c.raw_blob if fetch is None else fetch
    actual=fetch(row['git_blob'],lambda handle:session.stream(handle,row['size'],row['paths'],'actual_remote',row['git_blob']))
    c.equal(row,actual)
    return {**row,'actual_remote_fetch_this_run':True,'cache_or_inherited_payload_proof_used':False,
            'accept_header':'application/vnd.github.raw+json','cache_control':'no-cache'}

def validate_full_report(report,spec,commit,spec_binding):
    if report.get('status')!='BACKED_UP' or report.get('commit_sha')!=commit or report.get('sole_parent_sha')!=c.PARENT or report.get('spec')!=spec_binding or not c.typed_equal(report.get('exact_scope'),c.scope(spec)):raise ValueError('Actual full BACKED_UP proof scope/commit differs')
    if report.get('source_evolution')!=[] or report.get('allow_evolving_source') is not False or type(report.get('workers')) is not int or report['workers']!=2:raise ValueError('Frozen unchanged source/two-worker requirement differs')
    files,_=c.all_entries(spec);expected=c.groups(files)
    c.validate_actual_groups(report['actual_remote_blob_groups'],expected,'actual_remote_fetch_this_run')
    c.validate_actual_groups(report['actual_local_git_blob_groups'],expected,'actual_local_git_stream_this_run')
    if type(report.get('actual_unique_remote_payload_fetches_this_run')) is not int or report['actual_unique_remote_payload_fetches_this_run']!=len(expected):raise ValueError('Actual unique fresh fetch count differs')
    for row in report['actual_remote_blob_groups']:
        if row.get('cache_or_inherited_payload_proof_used') is not False or row.get('accept_header')!='application/vnd.github.raw+json' or row.get('cache_control')!='no-cache':raise ValueError('Actual remote uncached fetch provenance absent')
    for ctx in ('source_before','source_after','local_git','actual_remote'):c.validate_metadata_report(report['metadata_reports'][ctx],ctx)
    return report

def execute(cfg):
    if set(cfg['fields'])!={'commit_sha','workers','allow_evolving_source'} or cfg['fields']['workers']!=2 or type(cfg['fields']['workers']) is not int or cfg['fields']['allow_evolving_source'] is not False:raise ValueError('Only two bounded workers, no evolving source mode')
    spec=c.bound_json(cfg['inputs']['spec']);stage,local,commit=validate_stage_dependencies(cfg,spec)
    before=c.custody(commit);session=c.SecretSession();vector=c.scope(spec)
    if before['api_core_counts_only']['remaining']<vector['unique_git_blob_groups']+32:raise ValueError('Current API core budget insufficient for every unique fresh blob plus metadata reserve')
    source_before=c.verify_sources(spec,session,'source_before');roundtrips=c.verify_local_lossless(spec)
    c.preserve_prior(c.fixed_json('prior'),{**c.all_entries(spec)[0],**c.all_entries(spec)[1]})
    tree=c.text('rev-parse',commit+'^{tree}');metadata=verify_remote_metadata(spec,commit,tree)
    ledgerpath=c.new_ledger(cfg['outputs']['ledger']);local_groups,local_report=c.local_groups(spec,session,ledgerpath)
    files,_=c.all_entries(spec);plan=c.groups(files);remote=[]
    # Two streams at a time; no subprocess payload is buffered as a whole file.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        pending={executor.submit(fetch_group,row,session):blob for blob,row in plan.items()}
        for future in concurrent.futures.as_completed(pending):
            row=future.result();remote.append(row);c.ledger(ledgerpath,{'event':'actual_remote_blob_verified',**row})
    remote.sort(key=lambda r:r['git_blob']);c.validate_actual_groups(remote,plan,'actual_remote_fetch_this_run')
    remote_report=session.complete('actual_remote');source_after=c.verify_sources(spec,session,'source_after')
    c.verify_git_scope(spec,commit);verify_remote_metadata(spec,commit,tree);after=c.custody(commit);c.recheck_inputs(cfg)
    receipt={'status':'BACKED_UP','commit_sha':commit,'sole_parent_sha':c.PARENT,'tree_sha':tree,'spec':cfg['inputs']['spec'],
             'stage':cfg['inputs']['stage'],'local_commit':cfg['inputs']['local_commit'],'stage_ledger':cfg['inputs']['stage_ledger'],
             'exact_scope':vector,'actual_remote_blob_groups':remote,'actual_local_git_blob_groups':local_groups,
             'actual_unique_remote_payload_fetches_this_run':len(remote),'actual_http_attempt_count_not_asserted':True,
             'workers':2,'allow_evolving_source':False,'source_evolution':[],'metadata_reports':{'source_before':source_before,'source_after':source_after,'local_git':local_report,'actual_remote':remote_report},
             'local_roundtrips':roundtrips,'actual_remote_tree_coverage':metadata,'custody_before':before,'custody_after':after,
             'ledger':c.file_binding(ledgerpath),'actual_runner_execution':cfg['runner_execution'],'author_execution':'/root/backup_manager',
             'source_freeze':cfg['_source_freeze'],'execution_config':cfg['_config'],'finished_at_utc':c.now(),'grants_data_approval':False,
             'self_inventory_boundary':'This proof and ledger are postinventory local until a future exact checkpoint'}
    validate_full_report(receipt,spec,commit,cfg['inputs']['spec'])
    row=c.write_exclusive(c.safe_output(cfg['outputs']['receipt']),receipt)
    return {'status':'BACKED_UP','receipt':row,'commit_sha':commit,'exact_scope':vector}
