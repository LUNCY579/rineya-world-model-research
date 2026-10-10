"""Actual benign stdlib startup metadata, no review/helper/checker imports.
SOURCE/RAW-PREEXEC freeze must precede both isolated command invocations.
The 'old_comparator' and 'corrected_collector' shapes expose metadata differences;
these are not semantic comparisons, calibrations or corpus quality votes.
"""
import copy,datetime,hashlib,itertools,json,re,sys,time
from collections import Counter,defaultdict
from pathlib import Path,PurePosixPath
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--shape',choices=['old_comparator','corrected_collector'],required=True)
args=parser.parse_args()
if args.shape=='corrected_collector':
    import importlib.util,math,os
modules=[]
for name,module in sorted(sys.modules.items()):
    spec=getattr(module,'__spec__',None);origin=getattr(spec,'origin',None)
    source=getattr(module,'__file__',None);cache=getattr(module,'__cached__',None)
    modules.append({'module':name,'origin':origin,'source_path':str(Path(source).resolve()) if source else None,
        'cache_path':str(Path(cache).resolve()) if cache else None,
        'namespace_paths':[str(Path(p).resolve()) for p in getattr(module,'__path__',[])] if origin is None and hasattr(module,'__path__') else []})
print(json.dumps({'actor':'/root/adversarial_auditor','stage':'BENIGN_STDLIB_METADATA_ONLY','shape':args.shape,
    'resolved_python':str(Path(sys.executable).resolve()),'python_version':sys.version,'flags':str(sys.flags),
    'modules':modules,'owned_semantic_helper_imports':0,'targets_parsed_or_compared':0},indent=2))
