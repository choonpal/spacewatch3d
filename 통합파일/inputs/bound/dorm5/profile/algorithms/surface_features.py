"""Read-only geometry measurements. No prediction or reference mask inputs.

Exact-coordinate canonicalization is for adjacency analysis only. Source
vertices, faces, colors and labels are never changed or exported differently.
Angles are unoriented normal-axis differences, in radians. Boundary terms
refer to the surrounding ORIGINAL mesh, not a cropped semantic-only mesh.
"""
import numpy as np


def weighted_quantiles(values, weights, quantiles=(0., .25, .5, .75, 1.)):
    x, w = np.asarray(values, float), np.asarray(weights, float)
    if len(x) == 0 or not w.sum() > 0:
        return None
    order = np.argsort(x, kind='stable')
    cumulative = (np.cumsum(w[order]) - .5 * w[order]) / w.sum()
    return np.interp(quantiles, cumulative, x[order]).tolist()


def mesh_adjacency(vertices, faces):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    unique_vertices, canonical = np.unique(vertices, axis=0, return_inverse=True)
    cf = canonical[faces]
    all_edges = np.sort(np.vstack((cf[:, [0, 1]], cf[:, [1, 2]], cf[:, [2, 0]])), axis=1)
    owners = np.tile(np.arange(len(faces)), 3)
    edges, inverse, counts = np.unique(all_edges, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    start = np.cumsum(np.r_[0, counts[:-1]])
    manifold_edge_ids = np.flatnonzero(counts == 2)
    manifold_owners = np.column_stack((owners[order[start[manifold_edge_ids]]],
                                      owners[order[start[manifold_edge_ids]+1]]))
    lengths = np.linalg.norm(unique_vertices[edges[:, 1]] - unique_vertices[edges[:, 0]], axis=1)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, norms[:, None], out=np.zeros_like(cross), where=norms[:, None] > 0)
    dot = np.einsum('ij,ij->i', normals[manifold_owners[:, 0]], normals[manifold_owners[:, 1]])
    absolute_dot = np.clip(np.abs(dot), 0., 1.)
    normal_angles = np.arccos(absolute_dot)
    normal_angles[1 - absolute_dot <= 64*np.finfo(float).eps] = 0.
    degenerate_edge_face = (norms[manifold_owners] == 0).any(axis=1)
    uncertain_owners = owners[order[np.repeat(counts != 2, counts)]]
    return dict(vertices=unique_vertices, canonical_faces=cf, edges=edges,
                uncertain_edge_faces=np.unique(uncertain_owners),
                edge_counts=counts, edge_lengths=lengths,
                edge_owner_sort=order, edge_owner_starts=start, all_edge_owners=owners,
                manifold_edge_ids=manifold_edge_ids, manifold_owners=manifold_owners,
                normal_angles=normal_angles, normal_dots=dot,
                valid_pair=~degenerate_edge_face, areas=norms/2, normals=normals)


def component_features(adjacency, component_face_indices, face_class_ids):
    """One label-blind shape/context record for each supplied connected region."""
    a = adjacency
    component_index = np.full(len(a['areas']), -1, dtype=int)
    for i, ff in enumerate(component_face_indices):
        ff = np.asarray(ff, int)
        if (component_index[ff] >= 0).any():
            raise ValueError('Overlapping input components')
        component_index[ff] = i
    owners = a['manifold_owners']
    components_at_edge = component_index[owners]
    boundary = components_at_edge[:, 0] != components_at_edge[:, 1]
    internal = ~boundary
    pair_length = a['edge_lengths'][a['manifold_edge_ids']]
    reports = []
    for i, ff in enumerate(component_face_indices):
        ff = np.asarray(ff, int)
        area = a['areas'][ff]
        total = float(area.sum())
        if total <= 0:
            raise ValueError('Component area must be positive')
        normals = a['normals'][ff]
        tensor = np.einsum('i,ij,ik->jk', area, normals, normals) / total
        eigen = np.linalg.eigvalsh(tensor)
        boundary_pair = boundary & (components_at_edge == i).any(axis=1)
        internal_pair = internal & (components_at_edge[:, 0] == i)
        valid_boundary = boundary_pair & a['valid_pair']
        valid_internal = internal_pair & a['valid_pair']
        perimeter = float(pair_length[boundary_pair].sum())
        b_weights, b_angles = pair_length[valid_boundary], a['normal_angles'][valid_boundary]
        in_weights, in_angles = pair_length[valid_internal], a['normal_angles'][valid_internal]
        if boundary_pair.any():
            bo = owners[boundary_pair]
            other = np.where(component_index[bo[:, 0]] == i, bo[:, 1], bo[:, 0])
            neighbor_classes = {str(c): float(pair_length[boundary_pair][np.asarray(face_class_ids)[other] == c].sum() / perimeter)
                                for c in np.unique(np.asarray(face_class_ids)[other])} if perimeter else {}
        else:
            neighbor_classes = {}
        mean_b = float(np.average(b_angles, weights=b_weights)) if len(b_angles) else None
        mean_i = float(np.average(in_angles, weights=in_weights)) if len(in_angles) else None
        # These integrated quantities do not change if an existing planar
        # triangle is split into coplanar subtriangles: new edges turn by zero.
        # A plain average over internal edges does NOT have that invariance.
        border_turning = float(np.dot(b_angles, b_weights))
        internal_turning = float(np.dot(in_angles, in_weights))
        reports.append(dict(
            area=total, face_count=len(ff), normal_tensor_eigenvalues=eigen.tolist(),
            uncertain_topology_faces=int(np.isin(ff, a['uncertain_edge_faces']).sum()),
            normal_axis_dispersion=float(1 - eigen[-1]),
            boundary_length=perimeter, boundary_complexity=float(perimeter / np.sqrt(total)),
            boundary_pairs=int(boundary_pair.sum()), internal_pairs=int(internal_pair.sum()),
            boundary_angle_quantiles=weighted_quantiles(b_angles, b_weights),
            internal_angle_quantiles=weighted_quantiles(in_angles, in_weights),
            # Ignore exactly coplanar subdivisions when measuring the bend
            # distribution. A no-bend patch is UNKNOWN, not noise evidence.
            internal_bend_angle_quantiles=weighted_quantiles(in_angles[in_angles > 0], in_weights[in_angles > 0]),
            boundary_angle_mean=mean_b, internal_angle_mean=mean_i,
            boundary_curvature_integral=border_turning,
            internal_curvature_integral=internal_turning,
            scaled_internal_turning=internal_turning / np.sqrt(total),
            scaled_boundary_turning=border_turning / np.sqrt(total),
            internal_turning_fraction=internal_turning/(internal_turning+border_turning) if internal_turning+border_turning > 0 else None,
            boundary_minus_internal_angle=mean_b - mean_i if mean_b is not None and mean_i is not None else None,
            neighboring_class_boundary_fractions=neighbor_classes,
            boundary_signed_normal_flip_fraction=float(np.average(a['normal_dots'][valid_boundary] < 0, weights=b_weights)) if len(b_weights) else None))
    return reports
