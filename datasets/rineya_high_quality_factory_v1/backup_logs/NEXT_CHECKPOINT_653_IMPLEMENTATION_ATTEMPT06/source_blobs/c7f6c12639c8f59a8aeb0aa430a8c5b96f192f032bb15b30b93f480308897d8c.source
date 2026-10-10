"""Bounded exact copy/stage/commit/nonforce push. Ends pending actual remote bytes."""
from __future__ import annotations
import os, tempfile
import next653_common as c

def copy_file(row,session,factory=None,worktree=None):
    # Root injection is for isolated tests, never exposed by a CLI/config.
    factory=c.FACTORY if factory is None else factory;worktree=c.WORKTREE if worktree is None else worktree
    src=c.safe(factory,row['path']);target=c.safe(worktree,c.PREFIX+'/'+row['path'],False)
    c.equal(row,session.file(src,row['path'],'source_before'))
    if target.is_file() and c.hash_file(target)=={k:row[k] for k in ('size','sha256','git_blob')}:
        c.equal(row,session.file(target,row['path'],'target'));return 'UNCHANGED_TARGET_REVERIFIED'
    target.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.next653-copy-',dir=target.parent)
    before=c.stat_key(src)
    try:
        with src.open('rb') as a,os.fdopen(fd,'wb') as b:
            class Sink:
                def read(self,n):
                    value=a.read(n);b.write(value);return value
            actual=session.stream(Sink(),row['size'],[row['path']],'source_before',row['git_blob'])
            b.flush();os.fsync(b.fileno())
        c.equal(row,actual)
        if c.stat_key(src)!=before:raise ValueError('Source changed during copy')
        os.chmod(tmp,0o644);os.replace(tmp,target)
        c.equal(row,session.file(target,row['path'],'target'));return 'COPIED_AND_REVERIFIED'
    finally:
        # Only this invocation's incomplete temporary file; never frozen evidence.
        if os.path.exists(tmp):os.unlink(tmp)

def validate_stage_report(stage,spec,expected_commit=None):
    if stage.get('status')!='PUSHED_REMOTE_BYTES_VERIFICATION_PENDING' or stage.get('sole_parent_sha')!=c.PARENT or not c.typed_equal(stage.get('exact_scope'),c.scope(spec)):raise ValueError('Stage is not an exact pushed pending claim')
    c.sha(stage.get('commit_sha'),40)
    if expected_commit is not None and stage['commit_sha']!=expected_commit:raise ValueError('Stage commit cross-wired')
    if stage.get('all_source_target_index_sha256_and_sha1_verified') is not True or stage.get('strict_diff_check')!={'strict_diff_check':'PASS','unrestricted_exit_code':0,'warning_count':0,'historical_old_six_warning_exception_reused':False}:raise ValueError('Unqualified strict zero-new-warning/index gates required')
    for context in ('source_before','target','local_git','source_after'):c.validate_metadata_report(stage['metadata_reports'][context],context)
    return stage

def exact_stage(files,allowed):
    m=c.library('manager');tracked=set(m.index(c.WORKTREE));names=sorted(files)
    paths=[c.PREFIX+'/'+p for p in names]
    response=m.git(c.WORKTREE,'check-ignore','--no-index','-z','--stdin',allowed=(0,1),data=('\0'.join(paths)+'\0').encode())
    ignored={p.decode()[len(c.PREFIX)+1:] for p in response.split(b'\0') if p}
    forced=ignored-tracked
    if forced!=set(allowed) or not forced.issubset(files):raise ValueError('Exact newly ignored path declaration differs')
    for batch,force in (([p for p in names if p not in forced],False),(sorted(forced),True)):
        for start in range(0,len(batch),100):
            argv=['--literal-pathspecs','add']+(['--force'] if force else [])
            m.git(c.WORKTREE,*argv,'--',*[c.PREFIX+'/'+p for p in batch[start:start+100]])
    staged=[p.decode() for p in m.git(c.WORKTREE,'diff','--cached','--name-only','-z').split(b'\0') if p]
    if not staged or any(not p.startswith(c.PREFIX+'/') or p[len(c.PREFIX)+1:] not in files for p in staged):raise ValueError('Unrelated/empty staged paths')
    if m.git(c.WORKTREE,'diff','--cached','--diff-filter=D','--name-only','-z'):raise ValueError('Deletion is not authorized')
    return {'ignored_inventory_paths':sorted(ignored),'new_ignored_exact_paths_force_staged':sorted(forced),'staged_paths':staged,
            **c.strict_diff_cached()}

def execute(cfg):
    if set(cfg['fields'])!={'allow_ignored_exact_paths','message'} or type(cfg['fields']['message']) is not str or not cfg['fields']['message'].strip():raise ValueError('Exact ignored paths/message fields required')
    spec=c.bound_json(cfg['inputs']['spec']);files,raw=c.all_entries(spec)
    c.validate_spec_closure(cfg,spec)
    prior=c.fixed_json('prior');c.preserve_prior(prior,{**files,**raw})
    before=c.custody(c.PARENT);c.capacity(spec,before['disk_free_bytes']);session=c.SecretSession()
    c.verify_sources(spec,session,'source_before');roundtrips=c.verify_local_lossless(spec)
    m=c.library('manager');parent=m.whole_tree(c.WORKTREE,c.PARENT)
    old={p[len(c.PREFIX)+1:]:blob for p,(mode,blob) in parent.items() if p.startswith(c.PREFIX+'/')}
    if not set(old).issubset(files):raise ValueError('Previously committed path absent')
    ledgerpath=c.new_ledger(cfg['outputs']['ledger'])
    for p,row in files.items():
        action=copy_file(row,session);c.ledger(ledgerpath,{'event':'source_target_verified','path':p,'size':row['size'],'sha256':row['sha256'],'git_blob':row['git_blob'],'action':action})
    session.complete('target')
    allowed=cfg['fields']['allow_ignored_exact_paths']
    if type(allowed) is not list or len(set(allowed))!=len(allowed) or any(p not in files for p in allowed):raise ValueError('Only exact inventory ignored-path declarations allowed')
    # Newly ignored bytes are exact audited names; no ignore/whitespace config change.
    staged=exact_stage(files,allowed)
    c.verify_git_scope(spec,c.PARENT,True);m.validate_worktree_changes(c.WORKTREE,files,old)
    local,local_metadata=c.local_groups(spec,session,ledgerpath,True)
    c.verify_sources(spec,session,'source_after');c.recheck_inputs(cfg)
    strict=c.strict_diff_cached();c.custody(c.PARENT,False)
    messagepath=c.safe_output(cfg['outputs']['message'])
    with messagepath.open('x',encoding='utf-8') as f:f.write(cfg['fields']['message'].rstrip()+'\n');f.flush();os.fsync(f.fileno())
    c.command(['git','commit','--file',str(messagepath)],c.WORKTREE)
    commit=c.text('rev-parse','HEAD');c.verify_git_scope(spec,commit);c.strict_diff_commit(commit)
    # Current checkout exact commit, remote still parent, original protected bytes remain exact.
    local_custody=c.custody(commit,True,c.PARENT)
    localreceipt={'status':'LOCAL_COMMIT_PRESERVED_REMOTE_PENDING','commit_sha':commit,'sole_parent_sha':c.PARENT,
                  'tree_sha':c.text('rev-parse',commit+'^{tree}'),'spec':cfg['inputs']['spec'],'exact_scope':c.scope(spec),
                  'source_target_index_sha256_sha1_verified':True,'local_git_blob_groups':local,'staging':staged,
                  'strict_diff_check':strict,'custody':local_custody,'actual_runner_execution':cfg['runner_execution'],'author_execution':'/root/backup_manager',
                  'execution_config':cfg['_config'],'source_freeze':cfg['_source_freeze'],'grants_data_approval':False}
    localbinding=c.write_exclusive(c.safe_output(cfg['outputs']['local_commit']),localreceipt)
    c.recheck_inputs(cfg)
    c.command(['git','push','origin','HEAD:refs/heads/'+c.BRANCH],c.WORKTREE)
    after=c.custody(commit);c.verify_git_scope(spec,commit);c.verify_sources(spec,session,'source_after');c.recheck_inputs(cfg)
    receipt={'status':'PUSHED_REMOTE_BYTES_VERIFICATION_PENDING','commit_sha':commit,'sole_parent_sha':c.PARENT,
             'tree_sha':c.text('rev-parse',commit+'^{tree}'),'spec':cfg['inputs']['spec'],'inventory_report':cfg['inputs']['inventory_report'],
             'exact_scope':c.scope(spec),'all_source_target_index_sha256_and_sha1_verified':True,'local_git_blob_groups':local,
             'metadata_reports':{ctx:session.complete(ctx) for ctx in ('source_before','target','local_git','source_after')},
             'strict_diff_check':strict,'staging':staged,'local_roundtrips':roundtrips,'local_commit':localbinding,
             'stage_ledger':c.file_binding(ledgerpath),'custody_before':before,'custody_after':after,
             'actual_runner_execution':cfg['runner_execution'],'author_execution':'/root/backup_manager','source_freeze':cfg['_source_freeze'],
             'execution_config':cfg['_config'],'grants_data_approval':False,'remote_payload_fetches_this_stage':0,
             'postinventory_boundary':'Stage/local commit/ledger/config and later receipts are local until a future exact cut'}
    row=c.write_exclusive(c.safe_output(cfg['outputs']['receipt']),receipt)
    return {'status':receipt['status'],'receipt':row,'commit_sha':commit,'exact_scope':receipt['exact_scope']}
