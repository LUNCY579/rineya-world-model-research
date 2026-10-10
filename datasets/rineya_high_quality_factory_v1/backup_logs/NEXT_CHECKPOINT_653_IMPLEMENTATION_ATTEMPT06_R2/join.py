"""Authoritative joined custody with actual ledgers and dynamic full path coverage."""
from __future__ import annotations
import json, os
import next653_common as c
from next653_stage import validate_stage_report
from next653_recovery import validate_full_report, validate_stage_dependencies
from next653_lossless import validate_gzip_report, representation_plan, validate_decode_rows

def validate_ledger_chain(spec,full,packed,fullledger,localledger,gzledger):
    files,_=c.all_entries(spec);expected=c.groups(files)
    c.validate_actual_groups(fullledger,expected,'actual_remote_fetch_this_run')
    byblob={r['git_blob']:r for r in full['actual_remote_blob_groups']}
    if len(byblob)!=len(expected):raise ValueError('Full proof group membership differs')
    for row in fullledger:
        if row!={'event':'actual_remote_blob_verified',**byblob[row['git_blob']]}:
            raise ValueError('Actual remote ledger and receipt differ')
    c.validate_actual_groups(localledger,expected,'actual_local_git_stream_this_run')
    pairs,_=representation_plan(spec);validate_decode_rows(gzledger,pairs)
    decoded={(r['raw_path'],r['remote_gzip_path']):r for r in packed['actual_remote_decodes']}
    if len(decoded)!=len(pairs):raise ValueError('Gzip proof pair membership differs')
    for row in gzledger:
        if row!={'event':'actual_remote_gzip_decoded',**decoded[(row['raw_path'],row['remote_gzip_path'])]}:
            raise ValueError('Actual gzip ledger and receipt differ')
    return True

def validate_proof_bindings(cfg,spec):
    stage,local,commit=validate_stage_dependencies(cfg,spec)
    full=c.bound_json(cfg['inputs']['full_proof']);packed=c.bound_json(cfg['inputs']['gzip_proof'])
    validate_full_report(full,spec,commit,cfg['inputs']['spec']);validate_gzip_report(packed,spec,commit,cfg['inputs'])
    if (full.get('stage')!=cfg['inputs']['stage'] or full.get('stage_ledger')!=cfg['inputs']['stage_ledger']
            or full.get('local_commit')!=cfg['inputs']['local_commit'] or full.get('ledger')!=cfg['inputs']['full_ledger']
            or packed.get('ledger')!=cfg['inputs']['gzip_ledger']):raise ValueError('Joined exact receipt/ledger bindings differ')
    tree=c.sha(full.get('tree_sha'),40)
    if stage.get('tree_sha')!=tree or local.get('tree_sha')!=tree or packed.get('tree_sha')!=tree:raise ValueError('Proof tree cross-wired')
    files,_=c.all_entries(spec);expected=c.groups(files)
    fullledger=c.ledger_rows(cfg['inputs']['full_ledger'],'actual_remote_blob_verified')
    c.validate_actual_groups(fullledger,expected,'actual_remote_fetch_this_run')
    byblob={r['git_blob']:r for r in full['actual_remote_blob_groups']}
    for row in fullledger:
        if row!= {'event':'actual_remote_blob_verified',**byblob[row['git_blob']]}:raise ValueError('Actual remote ledger and receipt differ')
    localledger=c.ledger_rows(cfg['inputs']['full_ledger'],'local_git_blob_verified')
    c.validate_actual_groups(localledger,expected,'actual_local_git_stream_this_run')
    pairs,refs=representation_plan(spec);gzledger=c.ledger_rows(cfg['inputs']['gzip_ledger'],'actual_remote_gzip_decoded')
    validate_decode_rows(gzledger,pairs)
    decoded={(r['raw_path'],r['remote_gzip_path']):r for r in packed['actual_remote_decodes']}
    for row in gzledger:
        if row!={'event':'actual_remote_gzip_decoded',**decoded[(row['raw_path'],row['remote_gzip_path'])]}:raise ValueError('Actual gzip ledger and receipt differ')
    validate_ledger_chain(spec,full,packed,fullledger,localledger,gzledger)
    if full['actual_runner_execution']!=stage['actual_runner_execution'] or packed['actual_runner_execution']!='/root/backup_manager':raise ValueError('Actual runner roles cross-wired')
    for proof in (stage,local,full,packed):
        if proof.get('author_execution')!='/root/backup_manager' or proof.get('grants_data_approval') is not False:raise ValueError('Backup proof role/approval boundary differs')
    c.validate_completed_proof(cfg,'recovery',full,cfg['inputs']['full_proof'],spec)
    c.validate_completed_proof(cfg,'lossless',packed,cfg['inputs']['gzip_proof'],spec)
    return stage,local,full,packed,commit,tree

def pending_boundary(spec):
    files,raw=c.all_entries(spec);historical=spec.get('deferred_observed_pending_paths',[])
    if type(historical) is not list:raise ValueError('Historical pending-at-cut evidence must be a list')
    oldfiles,oldraw=c.all_entries(c.fixed_json('prior'));known=set(oldfiles)|set(oldraw)
    names=set()
    for row in historical:
        c.entry(row)
        if row.get('status_at_cut')!='RUNNING' or row['path'] in names or row['path'] in known or row['path'] in files or row['path'] in raw:raise ValueError('Invented/duplicate/prior/included pending boundary')
        names.add(row['path'])
    # Name listing only; no private prospective source/candidate contents are read.
    current={p.relative_to(c.FACTORY).as_posix() for p in c.FACTORY.rglob('*') if p.is_file()}
    return historical,sorted(current-set(files)-set(raw))

def execute(cfg):
    if set(cfg['fields'])!={'commit_sha'}:raise ValueError('Exact actual commit field required')
    spec=c.bound_json(cfg['inputs']['spec']);stage,local,full,packed,commit,tree=validate_proof_bindings(cfg,spec)
    vector=c.scope(spec);before=c.custody(commit);session=c.SecretSession()
    source_before=c.verify_sources(spec,session,'source_before');files,raw=c.all_entries(spec)
    c.preserve_prior(c.fixed_json('prior'),{**files,**raw})
    whole=c.verify_git_scope(spec,commit)
    if c.text('rev-parse',commit+'^{tree}')!=tree:raise ValueError('Actual current Git tree differs')
    # Recheck every actual source, target and current index/blob. No new network payload fetch.
    for p,row in files.items():c.equal(row,session.file(c.safe(c.WORKTREE,c.PREFIX+'/'+p),p,'target'))
    session.complete('target');local_groups,local_metadata=c.local_groups(spec,session)
    roundtrips=c.verify_local_lossless(spec);strict=c.strict_diff_commit(commit)
    historical,postlocal=pending_boundary(spec)
    source_after=c.verify_sources(spec,session,'source_after');after=c.custody(commit);c.recheck_inputs(cfg)
    receipt={'status':'BACKED_UP_EXACT_FROZEN_INVENTORY_AND_ACTUAL_REMOTE_LOSSLESS','commit_sha':commit,'sole_parent_sha':c.PARENT,
             'tree_sha':tree,'spec':cfg['inputs']['spec'],'inventory_report':cfg['inputs']['inventory_report'],
             'stage':cfg['inputs']['stage'],'local_commit':cfg['inputs']['local_commit'],'stage_ledger':cfg['inputs']['stage_ledger'],
             'actual_completed_proofs':{'full_remote':cfg['inputs']['full_proof'],'full_remote_ledger':cfg['inputs']['full_ledger'],
                                      'separate_remote_gzip':cfg['inputs']['gzip_proof'],'separate_remote_gzip_ledger':cfg['inputs']['gzip_ledger']},
             'exact_scope':vector,'current_actual_custody':{'before':before,'after':after,'whole_tree_index_paths':len(whole),
                 'source_target_index_commit_equal':True,'actual_local_git_payload_groups_verified_this_join':len(local_groups),
                 'metadata_reports':{ctx:session.complete(ctx) for ctx in ('source_before','target','local_git','source_after')},
                 'local_actual_raw_byte_comparisons':roundtrips,'actual_commit_strict_diff':strict},
             'actual_unique_remote_blob_groups_proved_by_full_runner':vector['unique_git_blob_groups'],
             'actual_remote_gzip_fetches_in_separate_proof':vector['unique_raw_gzip_pairs'],
             'actual_remote_payload_fetches_this_join':0,'actual_root_http_attempt_count_not_asserted':True,
             'historical_observed_pending_atcut':historical,'current_postinventory_local_path_names_only':postlocal,
             'copy_stage_commit_push_full_verification_runner_execution':stage['actual_runner_execution'],
             'gzip_runner_execution':packed['actual_runner_execution'],'actual_runner_execution':cfg['runner_execution'],
             'author_execution':'/root/backup_manager','same_model_disclosure':'Independent custody execution; inherited same model, no model diversity claim',
             'source_freeze':cfg['_source_freeze'],'execution_config':cfg['_config'],'at_utc':c.now(),'grants_data_approval':False,
             'self_inventory_boundary':'This joined receipt/addendum and all postcut paths are local until a subsequent exact inventory; earlier failures are preserved',
             'current_byte_evolution_boundary':'Historical commit653 is immutable; future current code/living bytes are claimed only by this new exact SPEC'}
    row=c.write_exclusive(c.safe_output(cfg['outputs']['receipt']),receipt)
    addendum=c.safe_output(cfg['outputs']['addendum'])
    text=(f"Backup custody completed for `{commit}` (sole parent `{c.PARENT}`).\n\n"
          f"The exact frozen scope is {vector['files']} capped files, {vector['bytes']} bytes and "
          f"{vector['unique_git_blob_groups']} unique blobs actually fetched by the full verifier. "
          f"The separately executed gzip proof fetched {vector['unique_raw_gzip_pairs']} actual raw/gzip pairs and "
          f"compared {vector['remote_decoded_bytes']} decoded bytes. "
          f"All {vector['release_union_files']} release files, 65 readonly references, six Git baselines, "
          "both clean checkouts and private ADMIN custody passed. The new delta passed strict whitespace checks with no warning exception.\n\n"
          f"Joined receipt: `{row['path']}` SHA256 `{row['sha256']}`. "
          "This receipt and addendum are postinventory local. Pending-at-cut and later newborn paths are explicitly outside the frozen scope. "
          "Backup custody supplies no data approval. The helper author and gzip/join runner are `/root/backup_manager`; actual copy/push/full runner is recorded in each bound proof.\n")
    with addendum.open('x',encoding='utf-8') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
    return {'status':receipt['status'],'receipt':row,'addendum':c.file_binding(addendum),'exact_scope':vector,'actual_remote_payload_fetches_this_join':0}
