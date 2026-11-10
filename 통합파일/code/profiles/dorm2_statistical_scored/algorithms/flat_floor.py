"""Single-level semantic support in XY, using the user's flat-indoor prior.

This changes the navigation support domain, never source mesh or observations.
It deliberately does not interpret reconstructed floor height variation as steps.
"""
from pathlib import Path
import importlib.util
import numpy as np
import shapely
from shapely.geometry import Polygon

SERVER_PRIOR=Path('/workspace/task5b_dorm3_ground_roles_20261026_rU9tZ1/ground_roles.py')
LOCAL_PRIOR=Path(__file__).resolve().parent.parent/'dorm3_ground_roles_20261026_s0i5Qh/ground_roles.py'


def prior_module():
    spec=importlib.util.spec_from_file_location('previous_ground_roles',SERVER_PRIOR if SERVER_PRIOR.exists() else LOCAL_PRIOR)
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def polygon_parts(g):
    if g.geom_type=='Polygon':return [g]
    return [p for q in getattr(g,'geoms',[]) for p in polygon_parts(q)]


def prepare_navigation(vertices,faces,labels,catalog,original_floor,floor_height):
    v=np.asarray(vertices);f=np.asarray(faces);labels=np.asarray(labels)
    catalog={int(k):value for k,value in catalog.items()}
    semantic=np.isin(labels,[k for k,n in catalog.items() if n in {'floor','tactile paving'}])
    if not semantic.any():raise ValueError('Explicit semantic floor input required')
    tri=v[f].astype(float)
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    projected_area=abs(cross[:,2])/2
    projected=shapely.union_all(shapely.polygons(tri[semantic&(projected_area>1e-10),:,:2]))
    if projected.is_empty:raise ValueError('No nondegenerate semantic floor XY footprint')
    # Keep previously accepted actual unknown support, not a new semantic inference.
    twice=np.linalg.norm(cross,axis=1)
    nz=np.divide(abs(cross[:,2]),twice,out=np.zeros(len(twice)),where=twice>0)
    legacy_seed=semantic&(nz>.8)&(abs(tri.mean(1)[:,2]-floor_height)<.25)&(twice/2>1e-10)
    if legacy_seed.any():
        previous,_,accepted,_=prior_module().prepare_navigation(v,f,labels,catalog,original_floor,floor_height)
    else:
        # The legacy supplement is optional. Its former gates must never reject
        # the semantic floor or prevent this flat-ground policy from running.
        previous=original_floor
        accepted=np.zeros(len(f),bool)
    ceiling=np.isin(labels,[k for k,n in catalog.items() if n in {'ceiling','ceilings'}])
    ceiling_projection=shapely.union_all(shapely.polygons(tri[ceiling&(projected_area>1e-10),:,:2]))
    floor_evidence=shapely.union_all([projected,previous])
    # Single-level indoor prior: where ceiling exists above an XY position, its
    # missing floor is inferred for navigation only. The wall collision domain is
    # still independent and preserved; ceiling is never counted as observed floor.
    evidence=shapely.union_all([floor_evidence,ceiling_projection])
    # Interior holes in a source floor ring are navigation support under the
    # explicitly provided no-stairs/no-drop single-level prior. Outside concavities
    # and gaps between separate components are not filled (no convex hull).
    floor=shapely.union_all([Polygon(p.exterior) for p in polygon_parts(evidence)])
    collision=~semantic
    structural_ids=[k for k,n in catalog.items() if n in {'wall','walls','ceiling','ceilings'}]
    assert collision[np.isin(labels,structural_ids)].all()
    report=dict(method='semantic_floor_full_xy_flat_indoor_prior_v2',
        user_provided_prior='All indoor areas on this map are flat, same-level floor; no actual steps or floor holes.',
        single_level_prior_required=True,multi_level_or_real_steps_supported=False,
        original_source_faces=len(f),floor_class_ids=sorted(k for k,n in catalog.items() if n in {'floor','tactile paving'}),
        wall_class_ids=sorted(k for k,n in catalog.items() if n in {'wall','walls'}),ceiling_class_ids=sorted(k for k,n in catalog.items() if n in {'ceiling','ceilings'}),
        semantic_ground_faces=int(semantic.sum()),semantic_floor_height_or_slope_gates=False,
        semantic_floor_projected_area=float(projected.area),previous_floor_area=float(previous.area),
        original_floor_area=float(original_floor.area),new_floor_area=float(floor.area),
        recovered_semantic_floor_area_vs_previous=float(projected.difference(previous).area),
        ceiling_backed_inferred_floor_area=float(ceiling_projection.difference(floor_evidence).area),
        ceiling_projection_used_only_as_navigation_prior=True,
        inferred_bounded_hole_area=float(floor.difference(evidence).area),
        inferred_bounded_hole_count=sum(len(p.interiors) for p in polygon_parts(evidence)),
        source_ground_faces_exempt_from_collision=int(semantic.sum()),
        wall_faces_preserved_as_collision=int(np.isin(labels,[k for k,n in catalog.items() if n in {'wall','walls'}]).sum()),
        ceiling_faces_preserved_as_collision=int(np.isin(labels,[k for k,n in catalog.items() if n in {'ceiling','ceilings'}]).sum()),
        unknown_edge_connected_support_faces=int(accepted.sum()),unknown_collision_faces_preserved=True,
        original_floor_bounds=list(original_floor.bounds),new_floor_bounds=list(floor.bounds),
        convex_hull_used=False,between_component_gap_filling=False,
        source_observation_triangles_added_or_removed=0,original_labels_changed=0,
        original_observation_geometry_unchanged=True,physical_safety_verified=False,
        navigation_floor_is_projected_support_not_repaired_source_mesh=True)
    return floor,collision,accepted,report
