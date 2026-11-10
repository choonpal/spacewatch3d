"""Verified export correspondence, shared by filtering and semantic projection."""
from pathlib import Path
import numpy as np
from .io import read, sha


class ExportCorrespondence:
    def __init__(self, source):
        self.source = Path(source)
        self.directory = self.source/'unfiltered_result/correspondence'
        self.manifest = read(self.directory/'manifest.json')
        m = self.manifest
        if m.get('schema_version') != 1:
            raise ValueError('Unsupported correspondence schema')
        cloud = self.source/'unfiltered_result'/read(self.source/'config.json')['output']['ply_name']
        for path, expected in [(cloud, m['cloud_sha256']),
                *((self.source/p, h) for p,h in m['source_sha256'].items()),
                *((self.directory/p, h) for p,h in m['files'].items())]:
            if sha(path) != expected:
                raise ValueError(f'Export correspondence provenance mismatch: {path}')
        self.mapping = np.load(self.directory/'pixel_to_point.npy', mmap_mode='r')
        if self.mapping.dtype != np.uint32 or len(self.mapping) != m['pixel_count']:
            raise ValueError('Invalid export correspondence row map')
        self.frames = {r['frame']: r for r in m['frames']}

    @classmethod
    def available(cls, source):
        if not (Path(source)/'unfiltered_result/correspondence/manifest.json').exists():
            return None  # Legacy export: caller uses exact reconstruction fallback.
        return cls(source)

    def frame(self, frame):
        row = self.frames[frame]
        with np.load(self.directory/row['mask']) as z:
            shape = tuple(z['shape']); mask = np.unpackbits(z['packed'], count=int(np.prod(shape))).astype(bool).reshape(shape)
        ids = self.mapping[row['offset']:row['offset']+row['points']]
        if int(mask.sum()) != len(ids) or (len(ids) and ids.max() >= self.manifest['points']):
            raise ValueError('Pixel mask/point rows differ from export')
        return ids, mask

    def index(self):
        return dict(buckets=[{**r, 'path': str(self.directory/r['path'])} for r in self.manifest['buckets']],
                    points=self.manifest['points'], voxel=self.manifest['voxel'], exact_xyz_rgb_match=True)
