import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
NAME = '2D to 3D 분할기 v1'
VERSION = '1.0.0'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text())

def atomic_npy(path, array):
    import numpy as np
    path = Path(path)
    temp = path.with_suffix('.partial.npy')
    np.save(temp, array)
    temp.replace(path)

def record_files(paths):
    return {str(Path(p).resolve()): {'sha256': digest(p), 'bytes': Path(p).stat().st_size} for p in paths}

def valid_stage(path, identity):
    path = Path(path)
    if not path.exists():
        return False
    meta = read_json(path)
    if meta.get('identity') != identity:
        raise ValueError(f'Cache identity mismatch: {path}')
    for file, expected in meta['files'].items():
        p = Path(file)
        if not p.is_file() or p.stat().st_size != expected['bytes'] or digest(p) != expected['sha256']:
            return False
    return True

def complete_stage(path, identity, paths, **extra):
    save_json(path, dict(identity=identity, status='complete', files=record_files(paths), **extra))

def progress(out, stage, **values):
    row = dict(name=NAME, version=VERSION, status='running', stage=stage, pid=os.getpid(),
               updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
    save_json(Path(out) / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
