"""Small dependency-free contracts shared by isolated worker environments."""
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def record(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

def verify(rec):
    p = Path(rec['path'])
    if not p.is_file() or p.stat().st_size != rec['bytes'] or sha(p) != rec['sha256']:
        raise ValueError(f'Artifact changed or missing: {p}. Use a new output directory.')
    return p

def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def code_identity():
    return {str(p.relative_to(ROOT)): sha(p) for base in ['spacewatch_pipeline', 'vendor','apps']
            for p in sorted((ROOT/base).rglob('*')) if p.is_file() and p.suffix in ['.py','.c','.js','.css','.html']}

def result(output, **values):
    save(Path(output)/'result.json', values)
    return values
