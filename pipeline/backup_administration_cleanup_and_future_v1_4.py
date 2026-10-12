#!/usr/bin/env python3
"""Owned administrative backup pipeline; no foreign program contents are read."""
from pathlib import Path
import argparse,base64,collections,datetime,gzip,hashlib,io,json,os,re,shutil,stat,subprocess,tarfile,time

W=Path('/Users/chengli/Downloads/Rineya_Autonomous_Knowledge_Learning_Bundle_v1.0')
CONTROL=W/'backup_control'
BACKUP_ID='20261012T024825Z-attempt02'
STAGE=W/'.rakdc_github_backup'/BACKUP_ID
REPO='LUNCY579/rineya-world-model-research'
OWNER='LUNCY579'
BRANCH='backups/rakdc/'+BACKUP_ID
TAG='rakdc-backup-'+BACKUP_ID
CHUNK=95*1024*1024
SOURCE_EXT={'.py','.pyc','.js','.mjs','.ts','.sh','.zsh','.bash','.c','.cpp','.h','.hpp','.rs','.so','.dylib','.exe','.ipynb'}
SKIP_NAMES={'.git','node_modules','__pycache__','.rakdc_github_backup','backup_control'}
STRICT_REL=[
'output/revision26_research/root_metadata_writer_external_caller_journal',
'output/revision27_research/root_finite_policy_qualification_caller_journal',
'output/revision28_research/root_mathematical_qualification_caller_journal',
'output/revision29_research/mathematical_qualification_execution_runs',
'output/revision30_research/mathematical_qualification_execution_runs',
'output/revision30_research/root_mathematical_qualification_caller_journal',
'output/revision30_research/root_mathematical_qualification_caller_journal/attempt_v1_0_001']
STRICT=[W/p for p in STRICT_REL]
SECRET=re.compile(rb'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{32,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def enc(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def emit(v):print(json.dumps(v,sort_keys=True),flush=True)
def native(p):
 s=os.lstat(p)
 return {k:getattr(s,k)for k in dir(s)if k.startswith('st_')and type(getattr(s,k))in(int,float)}
def ordinary(n):return {k:v for k,v in n.items()if k not in('st_atime','st_atime_ns')}
def atomic(p,b):
 need(not os.path.lexists(p),'fresh output required: '+str(p));p.parent.mkdir(parents=True,exist_ok=True)
 f=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:
  v=memoryview(b)
  while v:n=os.write(f,v);need(n>0,'write progress');v=v[n:]
  os.fsync(f);os.fchmod(f,0o444)
 finally:os.close(f)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def gh_env():
 env=os.environ.copy()
 p=subprocess.run(['gh','auth','token','--hostname','github.com','--user',OWNER],capture_output=True,text=True)
 need(p.returncode==0 and p.stdout.strip(),'authenticated account credential unavailable')
 env['GH_TOKEN']=p.stdout.strip();env['GIT_TERMINAL_PROMPT']='0'
 return env
def run(args,env=None,cwd=None):
 p=subprocess.run(args,env=env,cwd=cwd,capture_output=True,text=True)
 need(p.returncode==0,'command failed '+repr(args)+': '+p.stderr[:3000])
 return p.stdout
def api(path,env):
 return json.loads(run(['gh','api',path],env))
def check_repo(env):
 u=api('user',env);r=api('repos/'+REPO,env)
 need(u['login']==OWNER and r['full_name']==REPO and r['private']is True and r['permissions']['push']is True,'account/private repository/push scope')
 return {'login':u['login'],'repo':r['full_name'],'private':r['private'],'permissions':r['permissions']}
def find_release(env):
 for page in range(1,101):
  rows=api('repos/'+REPO+'/releases?per_page=100&page='+str(page)+'&backup_inspection='+BACKUP_ID+'-'+str(time.time_ns()),env)
  found=[r for r in rows if r['tag_name']==TAG]
  need(len(found)<=1,'unique exact archival release tag')
  if found:return found[0]
  if len(rows)<100:break
 raise RuntimeError('exact archival release not found in owner authenticated listing')
def is_protected(p):return any(p==q or q in p.parents for q in STRICT)
def tree_inventory():
 files=[];sources={};dirs={};other=[]
 for root,ds,fs in os.walk(W,followlinks=False):
  p=Path(root);dirs[str(p.relative_to(W))]=native(p)
  ds[:]=sorted(d for d in ds if d not in SKIP_NAMES and not is_protected(p/d)and not os.path.islink(p/d))
  for n in sorted(fs):
   q=p/n
   if q==W/'AGENTS.md':other.append({'path':str(q),'reason':'standing local operational instructions retained; archived in metadata Git'});continue
   if q.name=='.DS_Store':other.append({'path':str(q),'reason':'nonresearch OS metadata retained'});continue
   s=os.lstat(q)
   if not stat.S_ISREG(s.st_mode):other.append({'path':str(q),'reason':'nonregular/link retained'});continue
   if q.suffix.lower()in SOURCE_EXT:sources[str(q)]=native(q);continue
   need(s.st_size<CHUNK,'individual DATA exceeds bounded raw shard size: '+str(q))
   files.append(q)
 failed=W/'.rakdc_github_backup/20261012T024825Z'
 for q in sorted(failed.glob('*.tar.gz')):
  need(stat.S_ISREG(os.lstat(q).st_mode),'owned failed archive regular DATA');files.append(q)
 # Explicitly archive closed Root research receipts as DATA; retain recovery/operational files.
 dirs[str(CONTROL.relative_to(W))]=native(CONTROL)
 for q in sorted(CONTROL.iterdir()):
  s=os.lstat(q)
  if not stat.S_ISREG(s.st_mode):
   continue
  if q.suffix.lower()in SOURCE_EXT:
   sources[str(q)]=native(q)
   continue
  selected=q.name.startswith(('ROOT_R51','ROOT_R52','ROOT_R53','ROOT_R54','ROOT_R55','R55_','R51_R52_MINIMUM_RESTORE'))
  if selected:
   need(q.suffix in {'.json','.gz','.txt'},'explicit closed Root research DATA type')
   need(s.st_size<CHUNK,'bounded Root research DATA')
   files.append(q)
 need(len(files)==len(set(files)),'no duplicate inventory paths')
 return sorted(files,key=lambda p:str(p.relative_to(W))),sources,dirs,other
class ArchiveInput:
 def __init__(self,f,path):self.f=f;self.path=path;self.h=hashlib.sha256();self.n=0;self.tail=b''
 def read(self,n=-1):
  b=self.f.read(n);self.h.update(b);self.n+=len(b)
  need(SECRET.search(self.tail+b)is None,'potential credential in selected DATA (value withheld): '+self.path)
  self.tail=(self.tail+b)[-256:]
  return b
def prepare():
 need(not STAGE.exists(),'fresh snapshot stage required');STAGE.mkdir(parents=True,mode=0o700)
 files,sources,dirs,other=tree_inventory();strict_before={str(p):native(p)for p in STRICT if os.path.lexists(p)}
 need(files,'nonempty DATA snapshot');emit({'phase':'prepare','selected_files':len(files),'selected_bytes':sum(os.lstat(p).st_size for p in files),'Source_only_retained':len(sources)})
 rows=[];shards=[];pos=0;total=0
 while pos<len(files):
  group=[];raw=0
  while pos<len(files)and(raw+os.lstat(files[pos]).st_size<=CHUNK or not group):
   p=files[pos];group.append(p);raw+=os.lstat(p).st_size;pos+=1
  name='rakdc-'+BACKUP_ID+'-data-'+str(len(shards)+1).zfill(3)+'.tar.gz';out=STAGE/name
  f=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  try:
   with os.fdopen(f,'wb',closefd=False)as rawout:
    with gzip.GzipFile(filename='',mode='wb',fileobj=rawout,compresslevel=6,mtime=0)as gz:
     with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT)as tar:
      for p in group:
       rel=str(p.relative_to(W));before=native(p);need(stat.S_ISREG(before['st_mode']),'regular DATA only')
       fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
       try:
        s=os.fstat(fd);need(ordinary(before)==ordinary({k:getattr(s,k)for k in before}),'pinned open DATA')
        with os.fdopen(fd,'rb',closefd=False)as inp:
         reader=ArchiveInput(inp,rel);info=tarfile.TarInfo(rel);info.size=before['st_size'];info.mode=stat.S_IMODE(before['st_mode'])
         info.uid=before['st_uid'];info.gid=before['st_gid'];info.mtime=before['st_mtime_ns']//1000000000
         info.pax_headers={'mtime':str(before['st_mtime_ns']//1000000000)+'.'+str(before['st_mtime_ns']%1000000000).zfill(9)}
         tar.addfile(info,reader)
        after=native(p);need(ordinary(before)==ordinary(after)and reader.n==before['st_size'],'whole stable DATA archive input')
        row={'path':rel,'shard':name,'bytes':reader.n,'sha256':reader.h.hexdigest(),'mode':format(info.mode,'04o'),'mtime_ns':before['st_mtime_ns'],'native_before':before,'native_after':after,'archival_role':'OPAQUE_BACKUP_ONLY_NO_TRAINING_ADMISSION_OR_SUBSTANTIVE_REVIEW'}
        rows.append(row);total+=reader.n
       finally:os.close(fd)
    rawout.flush();os.fsync(f)
   os.fchmod(f,0o444)
  finally:os.close(f)
  size=os.lstat(out).st_size;need(size<96*1024*1024,'bounded compressed shard')
  shards.append({'name':name,'bytes':size,'sha256':sha(out),'member_count':len(group),'uncompressed_DATA_bytes':raw})
  emit({'phase':'prepare','shard':len(shards),'files_done':len(rows),'files_total':len(files),'compressed_bytes':size})
 source_after={p:native(p)for p in sources};need(sources==source_after,'retained foreign Source ALL19 changed during administrative DATA copy')
 need(strict_before=={p:native(p)for p in strict_before},'strict7 stat-only ALL19 changed')
 atomic(STAGE/'MANIFEST_FILES.jsonl.gz',gzip.compress(b''.join(enc(r)for r in rows),mtime=0))
 atomic(STAGE/'DIRECTORY_MODES.json.gz',gzip.compress(enc(dirs),mtime=0))
 atomic(STAGE/'EXCLUSIONS.json.gz',gzip.compress(enc({'Source_programs_stat_only_not_read_hashed_uploaded_or_deleted':sources,'strict7_paths_not_inventoried_copied_or_deleted':STRICT_REL,'other_retained':other,'control_and_staging_excluded':list(SKIP_NAMES),'all_relevant_selected_DATA_exactbytes':True}),mtime=0))
 summary={'schema':'RAKDC-private-administrative-backup-prepared-v1','backup_id':BACKUP_ID,'workspace':str(W),'repository':REPO,'branch':BRANCH,'release_tag':TAG,'selected_files':len(rows),'selected_DATA_bytes':total,'compressed_bytes':sum(s['bytes']for s in shards),'shards':shards,'Source_programs_retained':len(sources),'Source_programs_bytes_retained':sum(x['st_size']for x in sources.values()),'foreign_Source_body_hash_AST_import_execute':0,'strict7_stat_only_ALL19_unchanged':True,'Source_stat_only_ALL19_unchanged':True,'credential_pattern_scan_selected_raw_DATA':'PASS_NO_VALUES_EMITTED','opaque_administrative_DATA_copy_not_semantic_review':True,'candidate_rights_intake_native_qualifications':'UNCHANGED_PENDING','new_admissions':0,'primary_mission_goal':'ACTIVE_NOT_COMPLETE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 atomic(STAGE/'PREPARED.json',enc(summary));emit({'phase':'PREPARED','selected_files':len(rows),'DATA_bytes':total,'compressed_bytes':summary['compressed_bytes'],'shards':len(shards)})
def metadata_repo(prepared):
 p=STAGE/'metadata_git';need(not p.exists(),'fresh metadata git stage');p.mkdir(mode=0o700)
 for n in ['MANIFEST_FILES.jsonl.gz','DIRECTORY_MODES.json.gz','EXCLUSIONS.json.gz','PREPARED.json']:shutil.copyfile(STAGE/n,p/n)
 checkpoint=CONTROL/('PREBACKUP_CHECKPOINT.'+BACKUP_ID+'.json')
 if not checkpoint.exists():checkpoint=CONTROL/'PREBACKUP_CHECKPOINT.v1.0.json'
 shutil.copyfile(checkpoint,p/'CHECKPOINT.json')
 if (W/'AGENTS.md').exists():shutil.copyfile(W/'AGENTS.md',p/'STANDING_STORAGE_POLICY.md')
 shutil.copyfile('/Users/chengli/.codex/attachments/781a1980-86ad-458a-8f62-c297c9b29e5a/pasted-text-1.txt',p/'MISSION.txt')
 (p/'pipeline').mkdir();shutil.copyfile(Path(__file__).absolute(),p/'pipeline/backup_administration.py')
 shutil.copyfile(CONTROL/'OWN_ROOT_VERIFIED_GITHUB_BACKUP.v1.0.py',p/'pipeline/backup_administration_failed_v1_0.py')
 for pattern in ['BACKUP_PREPARE_FAILURE*.json','INDEPENDENT_ARCHIVE_VERIFICATION*.json','BACKUP_SAFETY_CHECKS*.json','BACKUP_PUBLISH_FAILURE*.json']:
  for q in CONTROL.glob(pattern):shutil.copyfile(q,p/q.name)
 shutil.copyfile(CONTROL/'OWN_ROOT_VERIFIED_GITHUB_BACKUP.v1.1.py',p/'pipeline/backup_administration_prepared_v1_1.py')
 shutil.copyfile(CONTROL/'OWN_ROOT_VERIFIED_GITHUB_BACKUP.v1.2.py',p/'pipeline/backup_administration_failed_draft_lookup_v1_2.py')
 shutil.copyfile(CONTROL/'OWN_ROOT_VERIFIED_GITHUB_BACKUP.v1.4.py',p/'pipeline/backup_administration_prior_control_retention_v1_4.py')
 readme="""# RAKDC private archival snapshot
This backup preserves original dataset and evidence/record bytes. It is an administrative archive, not a newly admitted dataset release. The research goal remains active. See CHECKPOINT.json for exact reviewed, provisional, pending and historical qualification status.

Archive release: https://github.com/"""+REPO+"""/releases/tag/"""+TAG+"""
Dedicated branch: """+BRANCH+"""
Selected DATA files: """+str(prepared['selected_files'])+"""
Selected original DATA bytes: """+str(prepared['selected_DATA_bytes'])+"""
The manifest records each original relative path, SHA256, size, permission mode, mtime_ns and archive shard. Compression changes storage only; restored member bytes must match the manifest exactly.

## Restore
Authenticate as the repository owner or an authorized collaborator. Retrieve MANIFEST_FILES.jsonl.gz at the recorded commit, select the required original path, download only its named release asset, verify the asset SHA256 from PREPARED.json, read that regular tar member, verify its original bytes/size/SHA256, and restore permission mode and mtime_ns from the manifest. Do not execute archived foreign programs or use protected evaluator material. Restored inode/device/ctime history is a new administrative event, not continuity with an earlier native-stat gate. A complete manifest is necessary to restore a file; a release URL alone is insufficient.

## Standing storage policy
Below 10 GiB free counts as tight storage. During active work, check before substantial generation/staging. Close or checkpoint writers; create a bounded original-byte inventory; back up selected datasets and relevant records to the verified private repository; freshly fetch and verify complete remote bytes, member coverage, sizes, hashes and archived modes; only then delete exact unchanged verified local copies. Never delete unverified, changed, extra, active-writer, protected-program/journal or unrelated files. Keep a small recovery index. Preserve Source access rules, sealed-evaluation boundaries, historical failures/HOLDs and scientific admission gates. No paid API, native weight update, main mutation or force push is authorized by this archival policy.
"""
 (p/'README.md').write_text(readme)
 return p
def gitpush(p,env):
 run(['git','-C',str(p),'add','.'],env)
 run(['git','-C',str(p),'commit','-m','Preserve byte-verified RAKDC archival records '+BACKUP_ID],env)
 commit=run(['git','-C',str(p),'rev-parse','HEAD'],env).strip()
 run(['git','-C',str(p),'-c','credential.helper=!gh auth git-credential','push','https://github.com/'+REPO+'.git','HEAD:refs/heads/'+BRANCH],env)
 ref=api('repos/'+REPO+'/git/ref/heads/'+BRANCH,env)
 need(ref['object']['sha']==commit,'remote branch must equal actual commit')
 return commit
def create_draft_release(commit,notes,env):
 # One creation attempt only. A failed/ambiguous outcome requires exact state inspection.
 payload={'tag_name':TAG,'target_commitish':commit,'name':'Private RAKDC archival backup '+BACKUP_ID,'body':notes,'draft':True,'prerelease':True,'make_latest':'false'}
 result=subprocess.run(['gh','api','--method','POST','repos/'+REPO+'/releases','--input','-'],input=enc(payload),env=env,capture_output=True)
 response={'schema':'rakdc.actual-single-release-create-response.v1.8','backup_id':BACKUP_ID,'branch':BRANCH,'tag':TAG,'target_commit':commit,'POST_attempts':1,'returncode':result.returncode,'stdout':result.stdout.decode(errors='replace'),'stderr':result.stderr.decode(errors='replace'),'retry_policy':'DO_NOT_REPEAT_CREATE_AFTER_UNKNOWN_OUTCOME; inspect exact existing remote state'}
 atomic(STAGE/'CREATE_RELEASE_RESPONSE.DATA.json',enc(response))
 need(result.returncode==0,'release creation outcome failed or UNKNOWN; inspect exact remote state before any retry')
 release=json.loads(result.stdout)
 need(type(release.get('id'))is int and release['id']>0 and release.get('tag_name')==TAG and release.get('target_commitish')==commit and release.get('draft')is True and release.get('prerelease')is True,'returned exact release identity/draft/target; unknown outcome never auto-retried')
 actual=api('repos/'+REPO+'/releases/'+str(release['id']),env)
 need(actual.get('id')==release['id']and actual.get('tag_name')==TAG and actual.get('target_commitish')==commit and actual.get('draft')is True and actual.get('prerelease')is True and actual.get('assets')==[],'fresh exact returned release ID must verify before upload')
 return actual
def publish():
 prep=load(STAGE/'PREPARED.json');env=gh_env();identity=check_repo(env);p=metadata_repo(prep)
 run(['git','-C',str(p),'init','-b',BRANCH],env)
 run(['git','-C',str(p),'config','user.name','Rineya archival backup automation'],env)
 run(['git','-C',str(p),'config','user.email','115753424+LUNCY579@users.noreply.github.com'],env)
 run(['git','-C',str(p),'config','core.autocrlf','false'],env)
 commit=gitpush(p,env)
 initialcommit=commit
 release=create_draft_release(commit,(p/'README.md').read_text(),env)
 shutil.copyfile(STAGE/'CREATE_RELEASE_RESPONSE.DATA.json',p/'CREATE_RELEASE_RESPONSE.DATA.json')
 commit=gitpush(p,env) # Persist actual creation response before uploads and final verification.
 for i,s in enumerate(prep['shards'],1):
  run(['gh','release','upload',TAG,str(STAGE/s['name']),'--repo',REPO],env)
  emit({'phase':'upload','shard':i,'total':len(prep['shards']),'name':s['name'],'bytes':s['bytes']})
 atomic(STAGE/'PUBLISHED.json',enc({'schema':'RAKDC-private-archive-published-draft-v1','identity':identity,'manifest_commit':commit,'initial_manifest_commit':initialcommit,'branch':BRANCH,'tag':TAG,'release_id':release['id'],'release_url':'https://github.com/'+REPO+'/releases/tag/'+TAG,'original_main_not_modified_by_this_workflow':True,'no_force_push':True,'local_files_deleted':0}))
 emit({'phase':'PUBLISHED_DRAFT','commit':commit,'release_id':release['id'],'assets':len(prep['shards'])})
class RemoteInput:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.f.read(n);self.h.update(b);self.n+=len(b);return b
def verify_git(p,commit,env):
 commit_object=api('repos/'+REPO+'/git/commits/'+commit,env);tree=api('repos/'+REPO+'/git/trees/'+commit_object['tree']['sha']+'?recursive=1',env);need(not tree.get('truncated',False),'complete metadata git tree')
 expected={str(q.relative_to(p)):q for q in p.rglob('*')if q.is_file()and '.git'not in q.relative_to(p).parts};blobs={r['path']:r for r in tree['tree']if r['type']=='blob'}
 need(set(expected)==set(blobs),'complete remote metadata tree coverage')
 checked=[]
 for n,q in expected.items():
  b=api('repos/'+REPO+'/git/blobs/'+blobs[n]['sha'],env);need(b['encoding']=='base64','full metadata blob')
  raw=base64.b64decode(b['content']);need(raw==q.read_bytes(),'complete remote metadata bytes: '+n)
  checked.append({'path':n,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'git_blob':blobs[n]['sha']})
 return checked
def verify():
 prep=load(STAGE/'PREPARED.json');pub=load(STAGE/'PUBLISHED.json');env=gh_env();identity=check_repo(env)
 manifest=[json.loads(b)for b in gzip.decompress((STAGE/'MANIFEST_FILES.jsonl.gz').read_bytes()).splitlines()];expected={r['path']:r for r in manifest};need(len(expected)==len(manifest)==prep['selected_files'],'full local inventory unique')
 gitchecked=verify_git(STAGE/'metadata_git',pub['manifest_commit'],env)
 release=api('repos/'+REPO+'/releases/'+str(pub['release_id']),env);assets={a['name']:a for a in release['assets']}
 need(set(assets)=={s['name']for s in prep['shards']},'complete uploaded asset coverage')
 seen=set();checked=[];total=0
 for i,s in enumerate(prep['shards'],1):
  a=assets[s['name']];need(a['state']=='uploaded'and a['size']==s['bytes'],'uploaded full asset size')
  need(a.get('digest')in [None,'sha256:'+s['sha256']],'GitHub declared digest')
  proc=subprocess.Popen(['gh','api','repos/'+REPO+'/releases/assets/'+str(a['id']),'-H','Accept: application/octet-stream'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  reader=RemoteInput(proc.stdout);count=0;shard_data=0
  try:
   with tarfile.open(fileobj=reader,mode='r|gz')as tar:
    for member in tar:
     need(member.isfile()and member.name in expected and member.name not in seen,'regular expected unique remote member')
     row=expected[member.name];need(row['shard']==s['name']and member.size==row['bytes']and format(member.mode,'04o')==row['mode'],'remote member shard/bytes/mode')
     need(member.pax_headers.get('mtime')==str(row['mtime_ns']//1000000000)+'.'+str(row['mtime_ns']%1000000000).zfill(9),'archived exact nsmtime')
     h=hashlib.sha256();n=0
     f=tar.extractfile(member)
     for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
     need(n==row['bytes']and h.hexdigest()==row['sha256'],'every complete remote original DATA member SHA256')
     seen.add(member.name);count+=1;shard_data+=n
   for b in iter(lambda:reader.read(1024*1024),b''):pass
   error=proc.stderr.read(65536);rc=proc.wait()
   need(rc==0,'remote asset GET completed: '+error.decode(errors='replace'))
   need(reader.n==s['bytes']and reader.h.hexdigest()==s['sha256'],'fresh complete remote compressed asset bytes SHA256')
   need(count==s['member_count']and shard_data==s['uncompressed_DATA_bytes'],'per-shard inventory coverage')
  finally:
   if proc.poll()is None:proc.terminate();proc.wait()
  total+=shard_data;checked.append({'name':s['name'],'asset_id':a['id'],'bytes':reader.n,'sha256':reader.h.hexdigest(),'verified_original_member_count':count,'verified_original_DATA_bytes':shard_data,'complete_actual_remote_GET':True})
  emit({'phase':'verify','shard':i,'total':len(prep['shards']),'members_done':len(seen),'DATA_bytes_done':total})
 need(seen==set(expected)and total==prep['selected_DATA_bytes'],'every selected original byte and member confirmed remote')
 result={'schema':'RAKDC-full-remote-original-byte-verification-v1','identity':identity,'manifest_commit':pub['manifest_commit'],'branch':BRANCH,'tag':TAG,'release_id':pub['release_id'],'release_url':pub['release_url'],'selected_files':len(seen),'selected_DATA_bytes':total,'complete_actual_fresh_remote_assets':checked,'complete_actual_fresh_remote_metadata_git_blobs':gitchecked,'SHA256_size_permission_nsmtime_and_member_coverage':'PASS_EVERY_SELECTED_FILE','not_merely_push_receipt_or_declared_digest':True,'local_files_deleted':0}
 atomic(STAGE/'VERIFIED.json',enc(result));shutil.copyfile(STAGE/'VERIFIED.json',STAGE/'metadata_git/VERIFIED.json')
 verifycommit=gitpush(STAGE/'metadata_git',env);verify_git(STAGE/'metadata_git',verifycommit,env)
 run(['gh','release','edit',TAG,'--repo',REPO,'--draft=false','--latest=false'],env)
 latest=api('repos/'+REPO+'/releases/'+str(pub['release_id']),env);need(latest['draft']is False and latest['tag_name']==TAG,'published private archival release')
 persisted={'verification_commit':verifycommit,'manifest_commit':pub['manifest_commit'],'branch':BRANCH,'tag':TAG,'release_id':pub['release_id'],'release_url':pub['release_url'],'private':True,'selected_files':len(seen),'selected_DATA_bytes':total,'every_selected_byte_verified':True,'local_files_deleted':0}
 atomic(STAGE/'VERIFIED_AND_PERSISTED.json',enc(persisted))
 atomic(CONTROL/('REMOTE_VERIFIED_BEFORE_CLEANUP.'+BACKUP_ID+'.json'),enc({**persisted,'remote_manifest':'https://github.com/'+REPO+'/blob/'+verifycommit+'/MANIFEST_FILES.jsonl.gz','archive_shards':prep['shards'],'ordinary_DATA_deletion_still_pending':True}))
 emit({'phase':'VERIFIED_AND_PERSISTED','files':len(seen),'DATA_bytes':total,'verification_commit':verifycommit,'url':pub['release_url']})
def safe_parent(p):
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for x in p.parent.parts[1:]:
   n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  return fd
 except BaseException:os.close(fd);raise
def delete_verified():
 verified=load(STAGE/'VERIFIED_AND_PERSISTED.json');prep=load(STAGE/'PREPARED.json');env=gh_env();check_repo(env)
 remote=api('repos/'+REPO+'/git/ref/heads/'+BRANCH,env);need(remote['object']['sha']==verified['verification_commit'],'verified remote branch persistence before deletion')
 rows=[json.loads(b)for b in gzip.decompress((STAGE/'MANIFEST_FILES.jsonl.gz').read_bytes()).splitlines()]
 exclusion=json.loads(gzip.decompress((STAGE/'EXCLUSIONS.json.gz').read_bytes()))
 sources=exclusion['Source_programs_stat_only_not_read_hashed_uploaded_or_deleted'];source_before={p:native(p)for p in sources};strict_before={str(p):native(p)for p in STRICT if os.path.lexists(p)}
 need(verified['every_selected_byte_verified']is True and len(rows)==verified['selected_files'],'completed full remote verification prerequisite')
 # Global preflight precedes any unlink. Changed or extra files are never included.
 for i,r in enumerate(rows):
  rel=Path(r['path']);need(not rel.is_absolute()and '..'not in rel.parts and str(rel)==r['path'],'canonical relative DATA path')
  p=W/rel;need(W in p.parents and not is_protected(p)and p.suffix.lower()not in SOURCE_EXT,'only exact authorized DATA deletion')
  n=native(p);need(ordinary(n)==ordinary(r['native_after']),'unchanged original file native identity before cleanup: '+r['path'])
  need(sha(p)==r['sha256'],'unchanged complete original bytes before cleanup: '+r['path'])
  if (i+1)%5000==0:emit({'phase':'cleanup-preflight','files':i+1,'total':len(rows)})
 need(source_before=={p:native(p)for p in sources},'Source-only ALL19 unchanged before unlink');need(strict_before=={p:native(p)for p in strict_before},'strict7 ALL19 unchanged before unlink')
 free_before=shutil.disk_usage(W).free;deleted=[];parents={}
 for i,r in enumerate(rows):
  p=W/r['path'];fd=safe_parent(p)
  try:
   s=os.stat(p.name,dir_fd=fd,follow_symlinks=False);need(ordinary({k:getattr(s,k)for k in r['native_after']})==ordinary(r['native_after']),'final pinned unchanged DATA identity')
   pm=stat.S_IMODE(os.fstat(fd).st_mode);parents.setdefault(str(p.parent),pm)
   if not pm&0o200:os.fchmod(fd,pm|0o200)
   os.unlink(p.name,dir_fd=fd);deleted.append(r['path'])
  finally:os.close(fd)
  if (i+1)%5000==0:emit({'phase':'cleanup','files_deleted':i+1,'total':len(rows)})
 for p,mode in sorted(parents.items(),key=lambda x:len(Path(x[0]).parts),reverse=True):
  if os.path.lexists(p):
   try:os.chmod(p,mode,follow_symlinks=False)
   except OSError:pass
 need(all(not os.path.lexists(W/p)for p in deleted),'all exact verified DATA local copies removed')
 need(source_before=={p:native(p)for p in sources},'retained Source programs ALL19 unchanged after DATA deletion');need(strict_before=={p:native(p)for p in strict_before},'protected strict7 ALL19 unchanged after DATA deletion')
 receipt={'schema':'RAKDC-verified-local-DATA-cleanup-v1','repository':REPO,'branch':BRANCH,'tag':TAG,'release_url':verified['release_url'],'manifest_commit':verified['manifest_commit'],'verification_commit':verified['verification_commit'],'deleted_verified_original_files':len(deleted),'deleted_original_DATA_bytes':prep['selected_DATA_bytes'],'exact_selected_local_paths_absent':True,'no_unverified_changed_extra_Source_or_protected_file_deleted':True,'retained_Source_programs':len(sources),'retained_Source_program_bytes':prep['Source_programs_bytes_retained'],'protected_strict7_unchanged':True,'retained_foreign_Source_ALL19_unchanged':True,'free_bytes_before_original_cleanup':free_before,'free_bytes_after_original_cleanup':shutil.disk_usage(W).free,'ongoing_research_goal':'ACTIVE','pending_intake_and_historical_nativeHOLDs':'PRESERVED_IN_REMOTE_RECORDS','administrative_restore_is_new_native_epoch_not_history_repair':True,'remote_SHA256_verification_receipt':'VERIFIED.json at verification_commit'}
 atomic(STAGE/'DELETED.json',enc(receipt));shutil.copyfile(STAGE/'DELETED.json',STAGE/'metadata_git/CLEANUP_RECEIPT.json')
 shutil.copyfile(Path(__file__).absolute(),STAGE/'metadata_git/pipeline/backup_administration_cleanup_and_future_v1_4.py')
 if (W/'AGENTS.md').exists():shutil.copyfile(W/'AGENTS.md',STAGE/'metadata_git/STANDING_STORAGE_POLICY.md')
 finalcommit=gitpush(STAGE/'metadata_git',env);verify_git(STAGE/'metadata_git',finalcommit,env)
 recovery={**receipt,'final_cleanup_commit':finalcommit,'remote_guide':'https://github.com/'+REPO+'/blob/'+finalcommit+'/README.md','manifest_locator':'https://github.com/'+REPO+'/blob/'+finalcommit+'/MANIFEST_FILES.jsonl.gz','archive_shards':prep['shards'],'stage_local_copies_removed_after_final_remote_verification':True}
 recovery_path=CONTROL/('GITHUB_BACKUP_RECOVERY.'+BACKUP_ID+'.json')
 atomic(recovery_path,enc(recovery))
 latest_path=CONTROL/'GITHUB_BACKUP_RECOVERY.LATEST.json'
 latest_bytes=enc({'backup_id':BACKUP_ID,'recovery_index':str(recovery_path),'repository':REPO,'branch':BRANCH,'tag':TAG,'final_cleanup_commit':finalcommit,'release_url':verified['release_url']})
 latest_tmp=CONTROL/('LATEST_INDEX.'+BACKUP_ID+'.tmp')
 atomic(latest_tmp,latest_bytes);os.replace(latest_tmp,latest_path)
 # These are OWN administrative staging duplicates, all original data and metadata proven remotely above.
 shutil.rmtree(STAGE)
 emit({'phase':'COMPLETE_VERIFIED_BACKUP_AND_LOCAL_DATA_CLEANUP','files_deleted':len(deleted),'DATA_bytes_deleted':prep['selected_DATA_bytes'],'retained_Source_programs':len(sources),'final_commit':finalcommit,'release_url':verified['release_url'],'free_bytes_now':shutil.disk_usage(W).free,'recovery_index':str(recovery_path)})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','publish','verify','delete']);p.add_argument('--backup-id',default=BACKUP_ID);a=p.parse_args();need(re.fullmatch(r'[A-Za-z0-9_-]{5,80}',a.backup_id)is not None,'safe fresh backup id');BACKUP_ID=a.backup_id;STAGE=W/'.rakdc_github_backup'/BACKUP_ID;BRANCH='backups/rakdc/'+BACKUP_ID;TAG='rakdc-backup-'+BACKUP_ID
 {'prepare':prepare,'publish':publish,'verify':verify,'delete':delete_verified}[a.phase]()

