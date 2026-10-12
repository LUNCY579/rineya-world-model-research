#!/usr/bin/env python3
"""Root-owned recovery of the exact existing draft backup; no duplicate publication."""
from pathlib import Path
import importlib.util,json,shutil,os,hashlib,urllib.parse
C=Path('/Users/chengli/Downloads/Rineya_Autonomous_Knowledge_Learning_Bundle_v1.0/backup_control')
sp=importlib.util.spec_from_file_location('owned_v13',C/'OWN_ROOT_VERIFIED_GITHUB_BACKUP.v1.3.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
env=m.gh_env();identity=m.check_repo(env);prep=m.load(m.STAGE/'PREPARED.json')
release=m.find_release(env)
m.need(release['id']==410111227 and release['draft'] is True,'resume exact existing draft')
p=m.STAGE/'metadata_git'
previous=m.run(['git','-C',str(p),'rev-parse','HEAD'],env).strip()
m.need(previous=='8681618ba8233da05c2bb8f1103cab8c5080f7aa','known existing initial commit')
shutil.copyfile(C/'OWN_ROOT_VERIFIED_GITHUB_BACKUP.v1.3.py',p/'pipeline/backup_administration_v1_3.py')
shutil.copyfile(Path(__file__),p/'pipeline/resume_existing_draft_v1_0.py')
shutil.copyfile(C/'BACKUP_PUBLISH_FAILURE_DRAFT_TAG_LOOKUP.v1.0.json',p/'BACKUP_PUBLISH_FAILURE_DRAFT_TAG_LOOKUP.v1.0.json')
commit=m.gitpush(p,env)
m.verify_git(p,commit,env)
for i,s in enumerate(prep['shards'],1):
 current=m.api('repos/'+m.REPO+'/releases/'+str(release['id']),env)
 matches=[a for a in current['assets'] if a['name']==s['name']]
 m.need(len(matches)<=1,'unique existing asset')
 if matches:
  a=matches[0];m.need(a['state']=='uploaded' and a['size']==s['bytes'] and a.get('digest')=='sha256:'+s['sha256'],'existing asset full declared digest; actual retrieval still required')
 else:
  url='https://uploads.github.com/repos/'+m.REPO+'/releases/'+str(release['id'])+'/assets?name='+urllib.parse.quote(s['name'],safe='')
  uploaded=json.loads(m.run(['gh','api',url,'--method','POST','-H','Content-Type: application/gzip','--input',str(m.STAGE/s['name'])],env))
  m.need(uploaded['name']==s['name'] and uploaded['size']==s['bytes'] and uploaded['state']=='uploaded','actual exact-id upload response')
 m.emit({'phase':'upload-existing-draft','shard':i,'total':len(prep['shards']),'name':s['name'],'bytes':s['bytes']})
m.atomic(m.STAGE/'PUBLISHED.json',m.enc({'schema':'RAKDC-private-archive-published-draft-v1','identity':identity,'manifest_commit':commit,'original_initial_manifest_commit':previous,'branch':m.BRANCH,'tag':m.TAG,'release_id':release['id'],'release_url':'https://github.com/'+m.REPO+'/releases/tag/'+m.TAG,'original_main_not_modified_by_this_workflow':True,'no_force_push':True,'resumed_exact_prior_draft_after_preserved_404_lookup_failure':True,'local_files_deleted':0}))
m.emit({'phase':'PUBLISHED_DRAFT','commit':commit,'release_id':release['id'],'assets':len(prep['shards'])})
