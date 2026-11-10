"""Semantic ground roles for a single-level navigation proxy, not observation.

No fabricated faces, convex hull closing, per-object exceptions, or target edits.
Wall/ceiling and non-ground object faces remain collision inputs. Unknown
support is accepted only as existing horizontal near-ground geometry joined by
actual shared mesh edges to semantic ground support. Unknown collision faces
remain conservatively present. Parameters match the producer's floor filters.
"""
from dataclasses import dataclass, asdict
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class GroundPolicy:
    minimum_abs_normal_z: float = .8
    height_tolerance_m: float = .25
    minimum_area: float = 1e-10


def polygon_parts(geometry):
    if geometry.geom_type == 'Polygon':
        return [geometry]
    return [p for g in getattr(geometry, 'geoms', []) for p in polygon_parts(g)]


def shared_edge_groups(vertices, faces):
    # Canonicalization is connectivity analysis only; export original arrays.
    _, canonical = np.unique(vertices, axis=0, return_inverse=True)
    f = canonical[faces]
    edges = np.sort(f[:, [[0,1],[1,2],[2,0]]].reshape(-1,2), axis=1)
    owner = np.repeat(np.arange(len(f)),3)
    order=np.lexsort((edges[:,1],edges[:,0]));e=edges[order];o=owner[order]
    same=(e[1:]==e[:-1]).all(1)
    a,b=o[1:][same],o[:-1][same]
    graph=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(f),len(f)))
    return connected_components(graph,directed=False)


def prepare_navigation(vertices, faces, labels, catalog, original_floor, floor_height, cfg=GroundPolicy()):
    v=np.asarray(vertices);f=np.asarray(faces);labels=np.asarray(labels)
    if len(f)!=len(labels):raise ValueError('One class per source triangle required')
    catalog={int(k):value for k,value in catalog.items()}
    floor_ids={k for k,n in catalog.items() if n in {'floor','tactile paving'}}
    wall_ids={k for k,n in catalog.items() if n in {'wall','walls'}}
    ceiling_ids={k for k,n in catalog.items() if n in {'ceiling','ceilings'}}
    unknown_ids={int(k) for k in np.unique(labels) if k not in catalog or catalog[k] in {'unknown','unclassified'}}
    if not floor_ids:raise ValueError('No semantic ground class declared')
    tri=v[f].astype(float);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);twice=np.linalg.norm(cross,axis=1)
    nz=np.divide(abs(cross[:,2]),twice,out=np.zeros(len(twice)),where=twice>0)
    horizontal=nz>cfg.minimum_abs_normal_z
    near=abs(tri.mean(1)[:,2]-floor_height)<cfg.height_tolerance_m
    positive=twice/2>cfg.minimum_area
    semantic=np.isin(labels,list(floor_ids))
    known_support=semantic&horizontal&near&positive
    unknown_candidate=np.isin(labels,list(unknown_ids))&horizontal&near&positive
    eligible=np.flatnonzero(known_support|unknown_candidate)
    if not known_support.any():raise ValueError('No semantic ground seed satisfies existing height/slope gates')
    count,groups=shared_edge_groups(v,f[eligible])
    seeded=np.unique(groups[known_support[eligible]])
    accepted=np.zeros(len(f),bool)
    accepted[eligible]=unknown_candidate[eligible]&np.isin(groups,seeded)
    patches=shapely.polygons(tri[accepted,:,:2])
    patches=patches[shapely.area(patches)>cfg.minimum_area]
    support=shapely.union_all(patches)
    floor=shapely.union_all([original_floor,support])
    # Keep steep/near-vertical floor-labeled faces: possible risers or mislabels.
    # Only semantic, surface-oriented ground faces lose obstacle status.
    ground_collision_exempt=semantic&horizontal&positive
    collision=~ground_collision_exempt
    assert collision[np.isin(labels,list(wall_ids|ceiling_ids))].all()
    report=dict(method='semantic_ground_role_and_observed_unknown_edge_support_v1',settings=asdict(cfg),
       original_source_faces=len(f),floor_class_ids=sorted(floor_ids),wall_class_ids=sorted(wall_ids),ceiling_class_ids=sorted(ceiling_ids),
       unknown_class_ids=sorted(unknown_ids),semantic_ground_faces=int(semantic.sum()),
       semantic_support_seed_faces=int(known_support.sum()),unknown_candidate_faces=int(unknown_candidate.sum()),
       unknown_edge_connected_support_faces=int(accepted.sum()),
       unknown_rejected_disconnected_faces=int((unknown_candidate&~accepted).sum()),
       source_ground_faces_exempt_from_collision=int(ground_collision_exempt.sum()),
       steep_or_degenerate_floor_faces_retained_as_obstacles=int((semantic&collision).sum()),
       wall_faces_preserved_as_collision=int(np.isin(labels,list(wall_ids)).sum()),
       ceiling_faces_preserved_as_collision=int(np.isin(labels,list(ceiling_ids)).sum()),
       original_floor_area=float(original_floor.area),new_floor_area=float(floor.area),
       added_observed_floor_support_area=float(floor.difference(original_floor).area),
       original_floor_bounds=list(original_floor.bounds),new_floor_bounds=list(floor.bounds),
       fabricated_triangles=0,original_labels_changed=0,observation_faces_removed=0,
       ground_role_assumption='Semantic horizontal floor is support, not a body obstacle; ground labeling and physical safety remain unverified',
       real_steps_not_generally_verified=True,unknown_collision_faces_preserved=True,
       observation_and_denoising_policy_unchanged=True)
    return floor,collision,accepted,report
