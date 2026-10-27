"""Apply frozen structural region transforms while preserving original photo rays.

Triangles use one transform each; shared boundary vertices are duplicated. Camera
projection uses the transform of the observed region, never the camera's room.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .core import camera_transform, read, sha, similarity, transform


class RegionalAlignment:
    def __init__(self, context, partition):
        self.context = context
        self.zones = partition
        cells = np.argwhere(partition['labels'] > 0)
        self.centers = partition['origin'] + (cells + .5) * partition['step']
        self.tree = cKDTree(self.centers)
        self.values = partition['labels'][tuple(cells.T)]
        corrections = np.asarray(context['methods']['structure']['region_matrices_after_Geo'], float)
        for matrix in corrections:
            _, scale = similarity(matrix)
            if not np.isclose(scale, 1.):
                raise ValueError('Regional corrections must be rigid')
        self.corrections = np.concatenate([np.eye(4)[None], corrections])
        self.base = np.asarray(context['base_matrix'], float)

    def assign(self, world, epoch):
        points = np.asarray(world)
        if epoch == 1:
            points = transform(points, np.asarray(self.context['membership_base_matrix']))
            parent = self.context['membership_parent_transform']
            t = np.clip((points @ np.asarray(parent['axis']) - parent['transition_start_u']) /
                        parent['transition_width_u'], 0, 1)
            weights = t*t*(3-2*t)
            points = points + weights[:, None] * (transform(points, np.asarray(parent['correction_matrix'])) - points)
        q = points[:, [0, 2]] @ self.zones['axes'].T
        cells = np.floor((q-self.zones['origin']) / self.zones['step']).astype(int)
        valid = np.all((cells >= 0) & (cells < self.zones['labels'].shape), axis=1)
        labels = np.zeros(len(points), 'i4')
        labels[valid] = self.zones['labels'][tuple(cells[valid].T)]
        missing = labels == 0
        labels[missing] = self.values[self.tree.query(q[missing], workers=4)[1]]
        return labels

    def matrices(self, epoch):
        return self.corrections if epoch else np.repeat(np.eye(4)[None], len(self.corrections), axis=0)

    def move(self, global_points, regions, epoch):
        matrices = self.matrices(epoch)[np.asarray(regions)]
        return np.einsum('...ij,...j->...i', matrices[..., :3, :3], global_points) + matrices[..., :3, 3]

    def mesh(self, world, faces, ids, common, epoch):
        regions = self.assign(world[faces].mean(1), epoch)
        # One vertex per (source vertex, region); face order/instance IDs stay fixed.
        pairs = np.column_stack([faces.ravel(), np.repeat(regions, 3)])
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        vertices = self.move(transform(world[unique[:, 0]], common), unique[:, 1], epoch)
        return dict(vertices=vertices.astype('f4'), faces=inverse.reshape(-1, 3).astype('i4'),
                    face_instance_id=ids, face_region=regions, vertex_region=unique[:, 1].astype('i4'),
                    source_vertex=unique[:, 0].astype('i4'), original_face_ids=np.arange(len(faces), dtype='i4'))

    def view(self, geometry, face_regions, epoch):
        region = np.zeros(geometry['face'].shape, 'i4')
        hit = geometry['mesh_hit']
        region[hit] = face_regions[geometry['face'][hit]]
        geometry['points'] = self.move(geometry['points'], region, epoch).astype('f4')
        geometry['point_region'] = region
        geometry['region_poses'] = np.stack([camera_transform(geometry['pose'], m) for m in self.matrices(epoch)])
        # Depth remains measured along the original ray. Rigid region transforms
        # preserve it when the region-specific camera is used for projection.
        return geometry


class FreshRegionalAlignment(RegionalAlignment):
    """Use main's newly fitted structural regions with the original RGB rays."""
    def __init__(self, report):
        self.report = report
        self.base = np.asarray(report['global_after_to_before'])
        self.centers = np.asarray(report['centers'])
        self.tree = cKDTree(self.centers[:, [0, 2]])
        self.corrections = np.stack([np.eye(4)] + [np.asarray(r['matrix_after_to_common']) @ np.linalg.inv(self.base)
                                                   for r in report['regions']])

    def assign(self, world, epoch):
        p = transform(world, self.base) if epoch else np.asarray(world)
        return (self.tree.query(p[:, [0, 2]])[1]+1).astype('i4')


def load_regional(manifest, base, common):
    spec = manifest.get('regional_alignment')
    if not spec:
        return None, {}
    paths = {k: Path(base) / spec[k] for k in ['context', 'partition']}
    context = read(paths['context'])
    if not np.allclose(common[0], np.eye(4)) or not np.allclose(common[1], context['base_matrix']):
        raise ValueError('Frozen regional transforms require their original global alignment')
    with np.load(paths['partition'], allow_pickle=False) as data:
        partition = dict(data)
    return RegionalAlignment(context, partition), {str(p.resolve()): sha(p) for p in paths.values()}


def aligned_mesh(root, prepared, spec, epoch, manifest_base):
    cache = prepared['registration'].get('aligned_meshes')
    if cache:
        path = Path(root) / cache[epoch]['file']
        if sha(path) != cache[epoch]['sha256']:
            raise ValueError('Aligned mesh cache changed')
        with np.load(path, allow_pickle=False) as data:
            mesh = dict(data)
        return mesh['vertices'], mesh
    path = Path(manifest_base) / spec['mesh']
    with np.load(path, allow_pickle=False) as data:
        mesh = dict(data)
    matrix = np.asarray(prepared['registration']['common_from_world'][epoch]) @ np.asarray(spec.get('world_from_mesh', np.eye(4)))
    return transform(mesh['vertices'], matrix), mesh
