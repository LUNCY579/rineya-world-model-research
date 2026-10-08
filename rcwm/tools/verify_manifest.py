#!/usr/bin/env python3
"""Verify listed bundle files. The manifest intentionally excludes itself."""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'MANIFEST.sha256'
if not manifest.exists():raise SystemExit('MANIFEST.sha256 is missing')
failed=[];count=0
for line in manifest.read_text().splitlines():
    expected,relative=line.split('  ',1)
    path=ROOT/relative
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        failed.append((relative,'unsafe path'));continue
    if not path.is_file():failed.append((relative,'missing'));continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=expected:failed.append((relative,'hash mismatch'))
    count+=1
if failed:
    for item in failed:print(*item,sep=': ')
    raise SystemExit(1)
print(f'PASS: {count} listed files verified. Manifest itself is not self-hashed.')
