import importlib.util, sys, subprocess
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('exp', r'C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/gh_export/tools/export_bwm.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
keep, exc = m.plan(m.DEFAULT_SRC)  # plan() only reads; nothing is written
remote = set(subprocess.run(['git', '-C', r'C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/gh_export',
                             'ls-tree', '-r', '--name-only', 'origin/main', 'bwm'], capture_output=True, text=True).stdout.split())
def key(d): return 'bwm/' + d.as_posix()
new = [(s, d) for s, d in keep if key(d) not in remote]
print('keep', len(keep), 'new vs remote', len(new))
for s, d in new:
    print('  NEW', d.as_posix(), s.stat().st_size)
# changed files
import hashlib
chg = []
for s, d in keep:
    if key(d) in remote:
        blob = subprocess.run(['git', 'hash-object', str(s)], capture_output=True, text=True).stdout.strip()
        rb = subprocess.run(['git', '-C', r'C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/gh_export', 'rev-parse', 'origin/main:' + key(d)], capture_output=True, text=True).stdout.strip()
        if blob != rb:
            chg.append(d.as_posix())
print('changed', len(chg))
for c in chg: print('  CHG', c)
print('excluded total', len(exc))
for p, why in exc:
    r = p.relative_to(m.DEFAULT_SRC).as_posix()
    if 'DISTILL' in r or '_ko_' in r:
        print('  EXC', r, why)
