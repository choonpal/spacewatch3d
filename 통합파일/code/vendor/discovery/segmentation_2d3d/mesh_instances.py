"""Assign face-level IDs on an existing semantic triangle mesh without remeshing.

The default vertex connectivity matches SpaceWatch v2's planning adapter.
An edge-only mode is provided for auditing corner-contact sensitivity.
"""
import numpy as np


def face_components(faces, connectivity='vertex'):
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('faces must be an integer Nx3 array')
    if connectivity not in ('vertex', 'edge'):
        raise ValueError('connectivity must be vertex or edge')
    n = len(faces)
    if n == 0:
        return []
    parents = np.arange(n)

    def find(i):
        i = int(i)
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = int(parents[i])
        return i

    if connectivity == 'vertex':
        keys = faces.ravel()
        face_ids = np.repeat(np.arange(n), 3)
        order = np.argsort(keys, kind='stable')
        same = keys[order[1:]] == keys[order[:-1]]
    else:
        keys = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        face_ids = np.repeat(np.arange(n), 3)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        same = (keys[order[1:]] == keys[order[:-1]]).all(1)
    for a, b in zip(face_ids[order[1:]][same], face_ids[order[:-1]][same]):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    while not np.array_equal(parents, parents[parents]):
        parents = parents[parents]
    # Minimum face index ordering matches scipy's undirected component labels.
    return [np.flatnonzero(parents == r) for r in np.unique(parents)]


def assign_mesh_ids(vertices, faces, class_ids, catalog, min_area=.01, connectivity='vertex'):
    """Return full-order face IDs and exact source-face lists for object candidates.

    Area is in the input mesh's squared coordinate units. ID zero retains all
    structure, unknown/dynamic classes, and components below the area threshold.
    Shared vertices can belong to multiple semantic/instance faces, so no single
    vertex ID is assigned. Input geometry and all face ordering are unchanged.
    """
    v, f, labels = map(np.asarray, (vertices, faces, class_ids))
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError('vertices must be finite Nx3 coordinates')
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in 'iu':
        raise ValueError('faces must be an integer Nx3 array')
    if len(f) and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError('face vertex index is outside vertices')
    if labels.shape != (len(f),) or labels.dtype.kind not in 'iu':
        raise ValueError('one integer class ID is required per face')
    if not np.isfinite(min_area) or min_area < 0:
        raise ValueError('min_area must be finite and nonnegative')
    if connectivity not in ('vertex', 'edge'):
        raise ValueError('connectivity must be vertex or edge')
    # Keep the source floating dtype to reproduce the existing area threshold.
    t = v[f]
    area = np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)/2
    classes = {int(c['id']): c for c in catalog}
    ids = np.zeros(len(f), np.uint32)
    regions, objects = [], []
    structural_names = {'wall', 'floor', 'ceiling', 'tactile paving', 'unknown'}
    for cid in np.unique(labels):
        cls = classes.get(int(cid), {'name': 'unknown'})
        selected = np.flatnonzero(labels == cid)
        excluded = cls['name'] in structural_names or not cls.get('static_mesh', True)
        if excluded:
            continue
        for index, group in enumerate(face_components(f[selected], connectivity)):
            fi = selected[group]
            total = float(area[fi].sum())
            keep = total >= min_area and total > 0
            row = dict(region_key=f'class_{int(cid)}_region_{index:04d}',
                       class_id=int(cid), class_name=cls['name'], class_name_ko=cls.get('name_ko', cls['name']),
                       faces=len(fi), area=total, kept=keep, source_face_indices=fi)
            if keep:
                iid = len(objects)+1
                ids[fi] = iid
                p = v[np.unique(f[fi])]
                centers = t[fi].mean(axis=1, dtype=np.float64)
                row.update(instance_id=iid, vertices=len(p),
                           bounds=[p.min(0).tolist(), p.max(0).tolist()],
                           centroid=np.average(centers, axis=0, weights=area[fi]).tolist())
                objects.append(row)
            else:
                row['instance_id'] = 0
            regions.append(row)
    return ids, objects, regions, area
