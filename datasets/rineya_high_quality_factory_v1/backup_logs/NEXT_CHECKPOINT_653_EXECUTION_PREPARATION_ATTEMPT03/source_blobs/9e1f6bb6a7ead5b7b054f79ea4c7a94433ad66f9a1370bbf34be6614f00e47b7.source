#!/usr/bin/env python3
"""Metadata-only oversized literal-path diagnosis; no helper imports or raw parsing."""
import argparse,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).absolute().parent
FACTORY=HERE.parent.parent
CAP=25*1024*1024

def proof(p):
 before=p.stat();h=hashlib.sha256();n=0
 with p.open('rb') as s:
  while chunk:=s.read(65536):h.update(chunk);n+=len(chunk)
 after=p.stat()
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Metadata bytes changed while hashing')
 return {'size':n,'sha256':h.hexdigest()}
def safe(value):
 p=Path(value)
 if type(value) is not str or p.is_absolute() or p.as_posix()!=value or any(x in ('.','..','.git') for x in p.parts) or any(c in value for c in ('\\','\0','\n','\r')):raise ValueError('Exact literal relative metadata path required')
 target=FACTORY/p
 if any(FACTORY.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)):raise ValueError('Metadata symlink rejected')
 return target
def binding(p):return {'path':p.relative_to(FACTORY).as_posix(),**proof(p)}
def check(row):
 if type(row) is not dict or set(row)!={'path','size','sha256'} or type(row['size']) is not int or row['size']<0:raise ValueError('Closed metadata byte binding required')
 if binding(safe(row['path']))!=row:raise ValueError('Metadata binding differs')
 return safe(row['path'])
def normalize(rec):
 r=dict(rec)
 for dest,alias in [('raw_bytes','raw_length'),('gzip_path','compressed_path'),('gzip_bytes','compressed_length'),('gzip_sha256','compressed_sha256')]:
  if dest in r and alias in r and (type(r[dest]) is not type(r[alias]) or r[dest]!=r[alias]):raise ValueError('Lossless alias conflict')
  if dest not in r and alias in r:r[dest]=r[alias]
 for role in ('raw','gzip'):
  p=Path(r[role+'_path'])
  if p.is_absolute():
   if str(p)!=r[role+'_path'] or not p.is_relative_to(FACTORY):raise ValueError('Absolute manifest path differs')
   r[role+'_path']=p.relative_to(FACTORY).as_posix()
  safe(r[role+'_path'])
  if type(r[role+'_bytes']) is not int or r[role+'_bytes']<0:raise ValueError('Manifest actual byte type differs')
 return r

def main():
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True);a.add_argument('--expected-config-sha256',required=True);a.add_argument('--expected-source-sha256',required=True);args=a.parse_args()
 if proof(Path(__file__))['sha256']!=args.expected_source_sha256 or proof(args.config)['sha256']!=args.expected_config_sha256:raise ValueError('External diagnosis freeze differs')
 cfg=json.loads(args.config.read_bytes())
 if cfg['schema']!='next653-oversized-metadata-diagnosis-config-v1' or cfg['actor_execution']!='/root/backup_manager' or cfg['raw_semantic_body_parsing'] is not False:raise ValueError('Metadata-only diagnosis scope required')
 prior=json.loads(check(cfg['prior_spec']).read_bytes());preferred=json.loads(check(cfg['preferred_pack_plan']).read_bytes())
 inherited={rep['manifest']['path'] for row in prior['raw_size_exclusions'] for rep in row['lossless_representations']}
 preferred_paths={r['manifest_path'] for r in preferred['artifacts']}
 paths=sorted(p for p in FACTORY.rglob('*') if p.is_file())
 oversized={p.relative_to(FACTORY).as_posix():binding(safe(p.relative_to(FACTORY).as_posix())) for p in paths if p.stat().st_size>CAP}
 manifest_paths=[p for p in paths if p.name.endswith('compression_manifest.json') or p.name.endswith('.lossless.json') or p.relative_to(FACTORY).as_posix() in inherited]
 cache={};valid=[];historical=[];invalid=[]
 for p in manifest_paths:
  mb=binding(p);value=json.loads(p.read_bytes());records=value.get('artifacts',value.get('items',[value]))
  if type(records) is not list or not records:raise ValueError('Nonempty actual manifest records required')
  for source in records:
   r=normalize(source);rawname=r['raw_path'];gzname=r['gzip_path']
   if rawname not in cache:cache[rawname]=binding(safe(rawname))
   if gzname not in cache:cache[gzname]=binding(safe(gzname))
   expected_raw={'path':rawname,'size':r['raw_bytes'],'sha256':r['raw_sha256']};expected_gzip={'path':gzname,'size':r['gzip_bytes'],'sha256':r['gzip_sha256']}
   observation={'manifest':mb,'declared_raw':expected_raw,'current_raw':cache[rawname],'gzip':cache[gzname]}
   if cache[gzname]!=expected_gzip:invalid.append(observation);continue
   if cache[rawname]!=expected_raw:historical.append(observation);continue
   valid.append({'raw':expected_raw,'gzip':expected_gzip,'manifest':mb,'preferred_frozen03_package':mb['path'] in preferred_paths})
 covered={r['raw']['path'] for r in valid};missing=[];distinct=[]
 for name,row in sorted(oversized.items()):
  if name in covered:continue
  matches=[r for r in valid if (r['raw']['size'],r['raw']['sha256'])==(row['size'],row['sha256'])]
  matches.sort(key=lambda r:(not r['preferred_frozen03_package'],r['manifest']['path'],r['raw']['path']))
  if not matches:distinct.append(row)
  else:missing.append({'raw':row,'existing_same_byte_package':matches[0]})
 for p,row in oversized.items():check(row)
 for item in valid:check(item['raw']);check(item['gzip']);check(item['manifest'])
 result={'schema':'next653-oversized-literal-path-diagnosis-v1','status':'MISSING_LITERAL_ALIAS_MAPPINGS_DISCOVERED' if missing else 'COMPLETE_CURRENT_LITERAL_PATH_COVERAGE','actual_author_execution':'/root/backup_manager','actual_runner_execution':'/root/backup_manager','raw_semantic_body_parsing':False,'helper_imports_git_network_packing':0,'source':binding(Path(__file__)),'config':binding(args.config),'oversized_count':len(oversized),'oversized_raw_bytes':sum(r['size'] for r in oversized.values()),'oversized_bindings':list(oversized.values()),'recognized_manifest_count':len(manifest_paths),'valid_current_records':valid,'historical_noncurrent_records':historical,'invalid_gzip_bindings':invalid,'missing_aliases':missing,'new_distinct_content_needing_pack':distinct,'grants_data_approval':False}
 out=safe(cfg['output'])
 with out.open('x') as s:json.dump(result,s,indent=2,sort_keys=True);s.write('\n');s.flush();os.fsync(s.fileno())
 print(json.dumps({'output':binding(out),'oversized_count':len(oversized),'missing_aliases':len(missing),'new_distinct_content':len(distinct),'invalid_gzip_bindings':len(invalid),'historical_records':len(historical)}))
if __name__=='__main__':main()
