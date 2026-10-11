import marshal, struct, os, importlib.util
src = 'bwmn/blocks_planner.py'; pyc = 'bwmn/__pycache__/blocks_planner.cpython-311.pyc'
b = open(pyc, 'rb').read()
flags = struct.unpack('<I', b[4:8])[0]; mt, sz = struct.unpack('<II', b[8:16])
st = os.stat(src)
print('magic ok', b[:4] == importlib.util.MAGIC_NUMBER, 'flags', flags, 'mtime match', mt == int(st.st_mtime) & 0xFFFFFFFF,
      'size match', sz == st.st_size & 0xFFFFFFFF)
c1 = marshal.loads(b[16:]); c2 = compile(open(src, 'rb').read(), 'x', 'exec')
def flat(c):
    out = [c.co_code, c.co_names, c.co_varnames]
    for k in c.co_consts:
        out += flat(k) if hasattr(k, 'co_code') else [repr(k)]
    return out
print('code equal', flat(c1) == flat(c2))
