"""Exact export-time pixel -> PLY row correspondence (no nearest neighbours)."""
from pathlib import Path
import hashlib
import json
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


class CorrespondenceWriter:
    def __init__(self, directory, work, buckets):
        self.directory = Path(directory); self.directory.mkdir()
        self.work = Path(work)
        self.handles = [(self.work/f'pixels_{k:02d}.bin').open('wb') for k in range(buckets)]
        self.frames = []; self.count = 0; self.rows = []

    def collect(self, frame, mask, bucket):
        ids = np.arange(self.count, self.count+len(bucket), dtype=np.uint64)
        for k, handle in enumerate(self.handles):
            ids[bucket == k].tofile(handle)
        path = self.directory/f'mask_{frame:05d}.npz'
        np.savez_compressed(path, packed=np.packbits(mask.ravel()), shape=np.array(mask.shape))
        self.frames.append(dict(frame=int(frame), offset=self.count, points=len(bucket), mask=path.name))
        self.count += len(bucket)

    def prepare(self):
        for handle in self.handles:
            handle.close()
        self.mapping = np.lib.format.open_memmap(self.directory/'pixel_to_point.npy',
            mode='w+', dtype=np.uint32, shape=(self.count,))

    def bucket(self, k, keys, inverse, offset):
        if offset+len(keys) >= 2**32-1:
            raise ValueError('Correspondence exceeds uint32 PLY row capacity')
        pixels = np.fromfile(self.work/f'pixels_{k:02d}.bin', dtype=np.uint64)
        if len(pixels) != len(inverse):
            raise ValueError('Export pixel order differs from voxel shard')
        self.mapping[pixels] = inverse.astype(np.uint32) + offset
        path = self.directory/f'keys_{k:02d}.npy'; np.save(path, keys)
        self.rows.append(dict(bucket=k, offset=int(offset), points=len(keys), path=path.name))

    def finish(self, source, cloud, voxel, points):
        self.mapping.flush(); del self.mapping
        files = {p.name: sha(p) for p in self.directory.iterdir() if p.is_file()}
        inputs = [source/'prepared.npz', source/'optimized.npz', source/'config.json',
                  *sorted((source/'windows').glob('*.npz'))]
        manifest = dict(schema_version=1, points=points, pixel_count=self.count, voxel=voxel,
            frames=self.frames, buckets=self.rows, cloud_sha256=sha(cloud),
            source_sha256={str(p.relative_to(source)): sha(p) for p in inputs}, files=files)
        temp = self.directory/'manifest.partial'
        temp.write_text(json.dumps(manifest, indent=2)); temp.replace(self.directory/'manifest.json')
