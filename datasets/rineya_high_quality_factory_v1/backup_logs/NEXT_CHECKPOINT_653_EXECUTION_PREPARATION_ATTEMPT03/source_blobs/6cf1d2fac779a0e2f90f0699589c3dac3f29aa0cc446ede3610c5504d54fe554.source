#!/usr/bin/env python3
"""Exact eight literal raw aliases, existing frozen gzip, fresh manifests only."""
import argparse,datetime,gzip,hashlib,json,os,tempfile
from pathlib import Path
HERE=Path(__file__).absolute().parent
FACTORY=HERE.parent.parent
EXPECTED_ROWS=[{'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT02/preexecution_raw_archives/38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.source', 'sha256': '38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c', 'size': 41453404}, 'existing_origin': {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/raw_source_archives/38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.source', 'sha256': '38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c', 'size': 41453404}, 'gzip': {'path': 'reproducibility/lossless_archives/rsm-installed-runtime-r1/da98e29be0c797e60da317ce176913dae4ca8ba24ddffab420342f2b5d89eb2f-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.gz', 'sha256': '6fb8f6fe7b6d6767d797fc075b5eeda504f81e976950fa5386906bb23dce03d9', 'size': 1161465}, 'manifest': {'path': 'manifests/lossless_archives/rsm-installed-runtime-r1/da98e29be0c797e60da317ce176913dae4ca8ba24ddffab420342f2b5d89eb2f-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.lossless.json', 'sha256': '9e4294873c597d2a8898cd3f476c9e282eea86cef7d43da5f41288e2fe3933a0', 'size': 1177}}, 'new_manifest_path': 'manifests/lossless_archives/rsm-installed-runtime-aliases-r1/9f8fc010343df40249558a4a4663c4592c00a09cd2495645a8e38a7111196316-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.lossless.json'}, {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT02/preexecution_raw_archives/7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.source', 'sha256': '7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c', 'size': 40541020}, 'existing_origin': {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/raw_source_archives/7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.source', 'sha256': '7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c', 'size': 40541020}, 'gzip': {'path': 'reproducibility/lossless_archives/rsm-installed-runtime-r1/2210fe169d8802df280eab75e98dce63a4965a3becf3478b56082c6767fbc6a9-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.gz', 'sha256': 'ed2e85260fc065fb10a23397379e9ce7ce39c8f783fc204ebb0d326adc719e4a', 'size': 1101065}, 'manifest': {'path': 'manifests/lossless_archives/rsm-installed-runtime-r1/2210fe169d8802df280eab75e98dce63a4965a3becf3478b56082c6767fbc6a9-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.lossless.json', 'sha256': '6bef819d9bed82b036811777abc24dd1a1fe9b95c4b443814aad9e196f0480df', 'size': 1177}}, 'new_manifest_path': 'manifests/lossless_archives/rsm-installed-runtime-aliases-r1/7648d590ea4845e409a640031fb37a5c41e9352e822e5f2dfc19e039a9009c63-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.lossless.json'}, {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT02/raw_source_archives/38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.source', 'sha256': '38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c', 'size': 41453404}, 'existing_origin': {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/raw_source_archives/38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.source', 'sha256': '38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c', 'size': 41453404}, 'gzip': {'path': 'reproducibility/lossless_archives/rsm-installed-runtime-r1/da98e29be0c797e60da317ce176913dae4ca8ba24ddffab420342f2b5d89eb2f-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.gz', 'sha256': '6fb8f6fe7b6d6767d797fc075b5eeda504f81e976950fa5386906bb23dce03d9', 'size': 1161465}, 'manifest': {'path': 'manifests/lossless_archives/rsm-installed-runtime-r1/da98e29be0c797e60da317ce176913dae4ca8ba24ddffab420342f2b5d89eb2f-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.lossless.json', 'sha256': '9e4294873c597d2a8898cd3f476c9e282eea86cef7d43da5f41288e2fe3933a0', 'size': 1177}}, 'new_manifest_path': 'manifests/lossless_archives/rsm-installed-runtime-aliases-r1/28a8842262b4b8b584d93987a8399cd072eca40893050a50e5ff7529da07df7d-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.lossless.json'}, {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT02/raw_source_archives/7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.source', 'sha256': '7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c', 'size': 40541020}, 'existing_origin': {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/raw_source_archives/7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.source', 'sha256': '7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c', 'size': 40541020}, 'gzip': {'path': 'reproducibility/lossless_archives/rsm-installed-runtime-r1/2210fe169d8802df280eab75e98dce63a4965a3becf3478b56082c6767fbc6a9-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.gz', 'sha256': 'ed2e85260fc065fb10a23397379e9ce7ce39c8f783fc204ebb0d326adc719e4a', 'size': 1101065}, 'manifest': {'path': 'manifests/lossless_archives/rsm-installed-runtime-r1/2210fe169d8802df280eab75e98dce63a4965a3becf3478b56082c6767fbc6a9-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.lossless.json', 'sha256': '6bef819d9bed82b036811777abc24dd1a1fe9b95c4b443814aad9e196f0480df', 'size': 1177}}, 'new_manifest_path': 'manifests/lossless_archives/rsm-installed-runtime-aliases-r1/09fb8a6882b19caa0ee54f49cb1aef594a243fdd2e266c80fa95690f13f15350-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.lossless.json'}, {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/preexecution_raw_archives/38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.source', 'sha256': '38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c', 'size': 41453404}, 'existing_origin': {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/raw_source_archives/38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.source', 'sha256': '38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c', 'size': 41453404}, 'gzip': {'path': 'reproducibility/lossless_archives/rsm-installed-runtime-r1/da98e29be0c797e60da317ce176913dae4ca8ba24ddffab420342f2b5d89eb2f-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.gz', 'sha256': '6fb8f6fe7b6d6767d797fc075b5eeda504f81e976950fa5386906bb23dce03d9', 'size': 1161465}, 'manifest': {'path': 'manifests/lossless_archives/rsm-installed-runtime-r1/da98e29be0c797e60da317ce176913dae4ca8ba24ddffab420342f2b5d89eb2f-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.lossless.json', 'sha256': '9e4294873c597d2a8898cd3f476c9e282eea86cef7d43da5f41288e2fe3933a0', 'size': 1177}}, 'new_manifest_path': 'manifests/lossless_archives/rsm-installed-runtime-aliases-r1/5d7df1e218fc72a779c1a227697b72e55c041e644b4487893a016a5de0ff3011-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.lossless.json'}, {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/preexecution_raw_archives/7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.source', 'sha256': '7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c', 'size': 40541020}, 'existing_origin': {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/raw_source_archives/7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.source', 'sha256': '7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c', 'size': 40541020}, 'gzip': {'path': 'reproducibility/lossless_archives/rsm-installed-runtime-r1/2210fe169d8802df280eab75e98dce63a4965a3becf3478b56082c6767fbc6a9-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.gz', 'sha256': 'ed2e85260fc065fb10a23397379e9ce7ce39c8f783fc204ebb0d326adc719e4a', 'size': 1101065}, 'manifest': {'path': 'manifests/lossless_archives/rsm-installed-runtime-r1/2210fe169d8802df280eab75e98dce63a4965a3becf3478b56082c6767fbc6a9-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.lossless.json', 'sha256': '6bef819d9bed82b036811777abc24dd1a1fe9b95c4b443814aad9e196f0480df', 'size': 1177}}, 'new_manifest_path': 'manifests/lossless_archives/rsm-installed-runtime-aliases-r1/565ae6a073cba38c706eb460cb3e2a0e280926f07cec15e552a282cb5c525f4c-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.lossless.json'}, {'raw': {'path': 'quality_reports/RSM_CORE_INSTALLATION_ATTEMPT01/raw_source_archives/38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.source', 'sha256': '38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c', 'size': 41453404}, 'existing_origin': {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/raw_source_archives/38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.source', 'sha256': '38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c', 'size': 41453404}, 'gzip': {'path': 'reproducibility/lossless_archives/rsm-installed-runtime-r1/da98e29be0c797e60da317ce176913dae4ca8ba24ddffab420342f2b5d89eb2f-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.gz', 'sha256': '6fb8f6fe7b6d6767d797fc075b5eeda504f81e976950fa5386906bb23dce03d9', 'size': 1161465}, 'manifest': {'path': 'manifests/lossless_archives/rsm-installed-runtime-r1/da98e29be0c797e60da317ce176913dae4ca8ba24ddffab420342f2b5d89eb2f-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.lossless.json', 'sha256': '9e4294873c597d2a8898cd3f476c9e282eea86cef7d43da5f41288e2fe3933a0', 'size': 1177}}, 'new_manifest_path': 'manifests/lossless_archives/rsm-installed-runtime-aliases-r1/c21c0c90504504e47e5cecac82b1dee53bbf24ec03fd73ed1c95c810274d2bfc-38016627c8d43dbabaac5d9dfb279d7cdd55bc5ca49fdd65bcdb276d550a0a6c.lossless.json'}, {'raw': {'path': 'quality_reports/RSM_CORE_INSTALLATION_ATTEMPT01/raw_source_archives/7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.source', 'sha256': '7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c', 'size': 40541020}, 'existing_origin': {'raw': {'path': 'quality_reports/RSM_CORE_HAND_FIXTURE_INSTALLATION_ATTEMPT03/raw_source_archives/7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.source', 'sha256': '7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c', 'size': 40541020}, 'gzip': {'path': 'reproducibility/lossless_archives/rsm-installed-runtime-r1/2210fe169d8802df280eab75e98dce63a4965a3becf3478b56082c6767fbc6a9-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.gz', 'sha256': 'ed2e85260fc065fb10a23397379e9ce7ce39c8f783fc204ebb0d326adc719e4a', 'size': 1101065}, 'manifest': {'path': 'manifests/lossless_archives/rsm-installed-runtime-r1/2210fe169d8802df280eab75e98dce63a4965a3becf3478b56082c6767fbc6a9-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.lossless.json', 'sha256': '6bef819d9bed82b036811777abc24dd1a1fe9b95c4b443814aad9e196f0480df', 'size': 1177}}, 'new_manifest_path': 'manifests/lossless_archives/rsm-installed-runtime-aliases-r1/a262c2e26dec0e756f50c72154151206abf88d9a87a3bf4fa458d17e49a90208-7284cb67509120efdc0f234f4bff026588f4926e5a72e0675eff7421cdf6a90c.lossless.json'}]
CHECKPOINT='rsm-canonical-review-milestone-r1'
PARENT='653ba70f1f7035a5aed137102c14a1ff29d1cf99'
AUTHOR='/root/backup_manager'
RUNNER='/root'
OUTPUT_PREFIX='manifests/lossless_archives/rsm-installed-runtime-aliases-r1/'

def same(a,b):
 if type(a) is not type(b):return False
 if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
def safe(value):
 if type(value) is not str or not value or any(c in value for c in ('\\','\0','\n','\r')):raise ValueError('Literal relative path required')
 p=Path(value)
 if p.is_absolute() or p.as_posix()!=value or any(x in ('.','..','.git') for x in p.parts):raise ValueError('Exact path alias rejected')
 if any(FACTORY.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)):raise ValueError('Symlink rejected')
 return FACTORY/p
def stat(p):
 s=p.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
def proof(p):
 before=stat(p);h=hashlib.sha256();n=0
 with p.open('rb') as s:
  while chunk:=s.read(65536):h.update(chunk);n+=len(chunk)
 if stat(p)!=before:raise ValueError('Frozen bytes changed while hashing')
 return {'size':n,'sha256':h.hexdigest()}
def binding(p):return {'path':p.relative_to(FACTORY).as_posix(),**proof(p)}
def check(row):
 if type(row) is not dict or set(row)!={'path','size','sha256'} or type(row['size']) is not int or row['size']<0 or type(row['sha256']) is not str or len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):raise ValueError('Closed typed exact binding required')
 p=safe(row['path'])
 if not p.is_file() or not same(binding(p),row):raise ValueError('Frozen actual bytes differ: '+row['path'])
 return p
def exclusive(p,value):
 # Atomic publish with no overwrite. Only this invocation's own temporary inode is removed.
 p.parent.mkdir(parents=True,exist_ok=True)
 fd,temp=tempfile.mkstemp(prefix='.exact-alias-',suffix='.tmp',dir=p.parent)
 try:
  with os.fdopen(fd,'w') as s:json.dump(value,s,indent=2,sort_keys=True);s.write('\n');s.flush();os.fsync(s.fileno())
  os.link(temp,p)  # atomic exclusive publication: fails if final path already exists
 finally:
  if os.path.exists(temp):os.unlink(temp)
 return binding(p)
def compare(raw,gz):
 a,b=check(raw),check(gz);before=(stat(a),stat(b));h=hashlib.sha256();n=0
 with a.open('rb') as actual,gzip.open(b,'rb') as packed:
  while chunk:=packed.read(65536):
   if chunk!=actual.read(len(chunk)):raise ValueError('Actual gzip decode/raw byte comparison differs')
   h.update(chunk);n+=len(chunk)
  if actual.read(1):raise ValueError('Decoded gzip omitted raw suffix')
 if (stat(a),stat(b))!=before or n!=raw['size'] or h.hexdigest()!=raw['sha256']:raise ValueError('CRC/full decoded raw binding differs')
 check(raw);check(gz)
 return {'raw':raw,'gzip':gz,'raw_bytes_actually_compared':n,'raw_sha256':h.hexdigest(),'full_crc_and_byte_for_byte':'PASS'}
def validate_origin(row):
 origin=row['existing_origin'];check(origin['raw']);check(origin['gzip']);p=check(origin['manifest'])
 value=json.loads(p.read_bytes());records=value.get('artifacts',value.get('items',[value]))
 if type(records) is not list or len(records)!=1:raise ValueError('Exact original single manifest required')
 rec=records[0]
 expected={role+'_path':origin[role]['path'] for role in ('raw','gzip')}
 expected.update({role+'_bytes':origin[role]['size'] for role in ('raw','gzip')})
 expected.update({role+'_sha256':origin[role]['sha256'] for role in ('raw','gzip')})
 if any(not same(rec.get(k),v) for k,v in expected.items()):raise ValueError('Frozen03 manifest declared source/gzip differs')
 if (origin['raw']['size'],origin['raw']['sha256'])!=(row['raw']['size'],row['raw']['sha256']):raise ValueError('Alias logical bytes differ from frozen03 source')
 return origin

def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--expected-plan-sha256',required=True);a.add_argument('--preexecution',type=Path,required=True);a.add_argument('--expected-preexecution-sha256',required=True);a.add_argument('--expected-source-sha256',required=True);args=a.parse_args()
 if proof(Path(__file__))['sha256']!=args.expected_source_sha256 or proof(args.plan)['sha256']!=args.expected_plan_sha256 or proof(args.preexecution)['sha256']!=args.expected_preexecution_sha256:raise ValueError('Exact external source/plan/preexecution freeze differs')
 plan=json.loads(args.plan.read_bytes());pre=json.loads(args.preexecution.read_bytes())
 keys={'schema','checkpoint_id','expected_parent','author_execution','prospective_runner_execution','mappings','frozen_inputs','receipt','failure','grants_data_approval'}
 if type(plan) is not dict or set(plan)!=keys or plan['schema']!='next653-exact-eight-oversized-literal-alias-plan-v1' or plan['checkpoint_id']!=CHECKPOINT or plan['expected_parent']!=PARENT or plan['author_execution']!=AUTHOR or plan['prospective_runner_execution']!=RUNNER or plan['grants_data_approval'] is not False or not same(plan['mappings'],EXPECTED_ROWS):raise ValueError('Only fixed exact eight alias mappings are authorized')
 if pre.get('schema')!='next653-exact-eight-alias-actual-preexecution-v1' or pre.get('authorizer_execution')!='/root' or pre.get('phase_granted')!='CREATE_EXACT_EIGHT_ALIAS_MANIFESTS' or pre.get('actual_runner_execution')!=RUNNER or not same(pre.get('source'),binding(Path(__file__))) or not same(pre.get('plan'),binding(args.plan)) or pre.get('grants_data_approval') is not False:raise ValueError('Separate actual ROOT closed creator grant required')
 raw_source=check(pre['raw_source'])
 if proof(raw_source)!=proof(Path(__file__)):raise ValueError('Creator raw source barrier differs')
 if type(plan['frozen_inputs']) is not list or not plan['frozen_inputs'] or len({r['path'] for r in plan['frozen_inputs']})!=len(plan['frozen_inputs']):raise ValueError('Nonempty unique original frozen inputs required')
 for r in plan['frozen_inputs']:check(r)
 receipt,failure=safe(plan['receipt']),safe(plan['failure'])
 if receipt.parent!=HERE or failure.parent!=HERE or receipt.exists() or failure.exists():raise ValueError('Fresh own exclusive receipt/failure required')
 for row in EXPECTED_ROWS:
  target=safe(row['new_manifest_path']);expected=OUTPUT_PREFIX+hashlib.sha256(row['raw']['path'].encode()).hexdigest()+'-'+row['raw']['sha256']+'.lossless.json'
  if row['new_manifest_path']!=expected or target.exists():raise ValueError('Exact fresh content/path-addressed alias manifest required')
 completed=[]
 try:
  # Validate every original and full existing03 gzip before the first new write.
  for row in EXPECTED_ROWS:
   origin=validate_origin(row);check(row['raw']);compare(origin['raw'],origin['gzip']);compare(row['raw'],origin['gzip'])
  for row in EXPECTED_ROWS:
   origin=validate_origin(row);observation=compare(row['raw'],origin['gzip'])
   value={'schema':'exact-preserved-literal-raw-alias-lossless-v1','raw_path':row['raw']['path'],'raw_bytes':row['raw']['size'],'raw_sha256':row['raw']['sha256'],'gzip_path':origin['gzip']['path'],'gzip_bytes':origin['gzip']['size'],'gzip_sha256':origin['gzip']['sha256'],'existing_frozen_source_origin':origin,'actual_author_execution':AUTHOR,'actual_runner_execution':RUNNER,'full_crc_and_actual_raw_byte_comparison':'PASS','new_gzip_bytes_created':0,'grants_data_approval':False}
   result=exclusive(safe(row['new_manifest_path']),value);completed.append({'manifest':result,'comparison':observation})
   validate_origin(row);check(row['raw'])
  for r in plan['frozen_inputs']:check(r)
  check(pre['raw_source'])
  if proof(Path(__file__))['sha256']!=args.expected_source_sha256 or proof(args.plan)['sha256']!=args.expected_plan_sha256 or proof(args.preexecution)['sha256']!=args.expected_preexecution_sha256:raise ValueError('Source/config/preexecution changed')
  value={'status':'COMPLETE_EXACT_EIGHT_LITERAL_RAW_ALIAS_MANIFESTS','author_execution':AUTHOR,'actual_runner_execution':RUNNER,'completed':completed,'raw_paths_actually_compared':len(completed),'raw_bytes_actually_compared':sum(x['comparison']['raw_bytes_actually_compared'] for x in completed),'new_gzip_bytes_created':0,'raw_source_gzip_old_manifest_bytes_preserved':True,'source':binding(Path(__file__)),'plan':binding(args.plan),'preexecution':binding(args.preexecution),'grants_data_approval':False}
  out=exclusive(receipt,value);print(json.dumps({'status':value['status'],'receipt':out,'manifests':len(completed),'raw_bytes_compared':value['raw_bytes_actually_compared'],'new_gzip_bytes':0}))
 except BaseException as error:
  exclusive(failure,{'status':'FAILED_EXACT_ALIAS_ATTEMPT_BYTES_PRESERVED','error_type':type(error).__name__,'error':str(error),'completed':completed,'source':binding(Path(__file__)),'plan':binding(args.plan),'preexecution':binding(args.preexecution),'actual_runner_execution':RUNNER,'grants_data_approval':False});raise
if __name__=='__main__':main()
