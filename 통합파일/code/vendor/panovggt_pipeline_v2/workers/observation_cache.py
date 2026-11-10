"""Run-local exact pair results shared by sequential recovery phases.

Changed match sources, geometry, settings, or implementation get different
keys. Cache files are optional and atomic; missing/corrupt entries recompute.
JSON's round-trip float representation preserves the existing Python rows.
"""
import hashlib
import json
from pathlib import Path


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def signature(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


class PairObservationCache:
    def __init__(self, directory, context):
        self.directory = Path(directory) / signature(context)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get(self, key):
        token = signature(key)
        try:
            envelope = json.loads((self.directory / (token + '.json')).read_text())
            result = envelope['result']
            if envelope['key'] != key or envelope['sha256'] != signature(result):
                raise ValueError('Cache content mismatch')
        except (OSError, ValueError, KeyError, TypeError):
            self.misses += 1
            return None
        self.hits += 1
        return result

    def put(self, key, result):
        path = self.directory / (signature(key) + '.json')
        temporary = path.with_suffix('.partial')
        temporary.write_text(encode(dict(key=key, result=result, sha256=signature(result))))
        temporary.replace(path)

    def statistics(self):
        return dict(hits=self.hits, misses=self.misses, context=self.directory.name)
