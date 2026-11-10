"""Small dependency-free contracts shared by isolated worker environments."""
import hashlib
import os
import fcntl
from contextvars import ContextVar
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE_CACHE = ContextVar('spacewatch_file_cache',default=None)

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def sha(path):
    cache=FILE_CACHE.get() or os.environ.get('SPACEWATCH_VERIFY_CACHE')
    if cache:
        p=Path(path).resolve();before=p.stat()
        # Rapid same-size rewrites can share filesystem timestamp ticks, even
        # when mtime is restored. Cache only settled files, not fresh outputs.
        if time.time_ns()-max(before.st_mtime_ns,before.st_ctime_ns)<2_000_000_000:
            return _sha_file(p)
        fingerprint=(str(p),before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
        folder=Path(cache);folder.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256(repr(fingerprint).encode()).hexdigest()
        item=folder/(key+'.json')
        with (folder/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if item.exists():return read(item)['sha256']
            value=_sha_file(p)
            after=p.stat()
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('File changed while hashing: '+str(p))
            save(item,{'sha256':value,'fingerprint':fingerprint})
            return value
    return _sha_file(path)

def _sha_file(path):
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
    identity = {str(p.relative_to(ROOT)): sha(p) for base in ['spacewatch_pipeline', 'vendor','apps','tools']
            for p in sorted((ROOT/base).rglob('*')) if p.is_file() and p.suffix in ['.py','.c','.so','.js','.css','.html']}
    compatibility = 'configs/gaussian_reuse_compatibility.json'
    identity[compatibility] = sha(ROOT/compatibility)
    for p in sorted((ROOT/'profiles').glob('*/checksums.json')):
        identity[str(p.relative_to(ROOT))]=sha(p)
    for p in [ROOT/'spacewatch_pipeline/importance_prompt.txt',ROOT/'resources/reconstruction_template.json']:
        identity[str(p.relative_to(ROOT))]=sha(p)
    for p in sorted((ROOT/'resources/gaussian_reference').glob('*')):
        if p.is_file() and p.suffix in ('.py','.yml','.json'):identity[str(p.relative_to(ROOT))]=sha(p)
    return identity

def result(output, **values):
    save(Path(output)/'result.json', values)
    return values
