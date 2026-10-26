"""Opt-in, bounded planar hole completion for navigation, never observation.

An enclosed gap is not proof of missing material. Generated geometry is an
explicit, unverified hypothesis, not a glass detector or a safety certificate.
"""
from copy import copy, deepcopy
from pathlib import Path
import json

import numpy as np
import shapely
from shapely.geometry import Polygon
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

from .models import Config, Geometry, InputError, read_json


def completion_options(values, config):
    if not isinstance(values, dict):
        raise InputError('navigation_completion must be an options object')
    defaults = {'max_hole_span_m': config.grid_m,
                'plane_tolerance_m': config.grid_m / 20,
                'max_normal_angle_deg': 10.}
    if set(values) - set(defaults):
        raise InputError('Unknown navigation_completion options: ' + repr(sorted(set(values)-set(defaults))))
    result = {**defaults, **values}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise InputError('navigation_completion.' + key + ' must be finite and positive')
    if result['max_normal_angle_deg'] >= 90:
        raise InputError('navigation_completion.max_normal_angle_deg must be below 90')
    return result


def _span(hole):
    """Longest side of the minimum-area oriented rectangle, not world XY."""
    points = np.asarray(hole.minimum_rotated_rectangle.exterior.coords)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).max())


def _triangulate(poly, origin, basis):
    triangles = []
    for part in shapely.get_parts(shapely.constrained_delaunay_triangles(poly)):
        xy = np.asarray(part.exterior.coords)[:3,:2]
        triangles.append(origin + xy @ basis)
    return np.asarray(triangles).reshape((-1,3,3))


def _path_crosses(hole, origin, basis, normal, positions, tolerance):
    if not len(positions):
        return False
    distances = (positions-origin) @ normal
    near = positions[np.abs(distances) <= tolerance]
    if len(near) and np.any(shapely.covers(hole,shapely.points((near-origin) @ basis.T))):
        return True
    for a,b,da,db in zip(positions[:-1],positions[1:],distances[:-1],distances[1:]):
        if da*db >= 0 or da == db:
            continue
        point = a + (b-a) * (da/(da-db))
        if hole.covers(shapely.Point((point-origin) @ basis.T)):
            return True
    return False


def _patches(scene, options, report, structure_ids):
    # Semantic objects, shared semantic faces and navigation-only proxies are
    # not structural evidence. Original structures remain available even when
    # an explicit proxy already replaced their collision interpretation.
    structures = [g for g in scene.geometries if g.object_id is None and not g.object_ids
                  and g.geometry_id != 'floor' and len(g.triangles)
                  and (structure_ids is None or g.geometry_id in structure_ids)]
    if not structures:
        return
    triangles = np.concatenate([g.triangles for g in structures])
    owners = np.concatenate([np.full(len(g.triangles),i) for i,g in enumerate(structures)])
    vertices, indices = np.unique(triangles.reshape((-1,3)),axis=0,return_inverse=True)
    mesh = trimesh.Trimesh(vertices=vertices,faces=indices.reshape((-1,3)),process=False)
    normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    areas2 = np.linalg.norm(normals,axis=1)
    normals /= areas2[:,None]
    adjacency = mesh.face_adjacency
    if len(adjacency):
        angle_ok = np.abs(np.einsum('ij,ij->i',normals[adjacency[:,0]],normals[adjacency[:,1]])) >= np.cos(np.deg2rad(options['max_normal_angle_deg']))
        adjacency = adjacency[angle_ok]
    graph = coo_matrix((np.ones(len(adjacency)), (adjacency[:,0],adjacency[:,1])),shape=(len(triangles),len(triangles)))
    count, labels = connected_components(graph,directed=False)
    # Group once, rather than scanning every triangle for each small island.
    order = np.argsort(labels,kind='stable')
    offsets = np.r_[0,np.cumsum(np.bincount(labels,minlength=count))]
    for index in range(count):
        face_ids = order[offsets[index]:offsets[index+1]]
        if len(face_ids) < 3:  # A triangle/quad cannot enclose an interior gap.
            continue
        faces, weight = triangles[face_ids], areas2[face_ids]
        center = np.average(faces.mean(axis=1),axis=0,weights=weight)
        delta = faces-center
        covariance = np.einsum('i,ijk,ijl->kl',weight,delta,delta)
        _,vectors = np.linalg.eigh(covariance)
        normal = vectors[:,0]
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal
        residual = float(np.abs(delta @ normal).max())
        if residual > options['plane_tolerance_m']:
            report['nonplanar_patch_count'] += 1
            continue
        # Floor support is handled from the explicit navigation floor, not
        # extrapolated from nearby furniture or a noisy near-floor mesh.
        if np.abs(normal[2]) > np.cos(np.deg2rad(options['max_normal_angle_deg'])) and np.max(np.abs(faces[:,:,2]-scene.floor_height)) <= options['plane_tolerance_m']:
            continue
        tangent = np.cross(normal,np.eye(3)[np.argmin(np.abs(normal))])
        tangent /= np.linalg.norm(tangent)
        basis = np.asarray([tangent,np.cross(normal,tangent)])
        projected = delta @ basis.T
        surface = shapely.union_all(shapely.polygons(projected))
        support = sorted({structures[i].geometry_id for i in owners[face_ids]})
        for part in shapely.get_parts(surface):
            if not isinstance(part,Polygon):
                continue
            for ring in part.interiors:
                yield Polygon(ring),center,basis,normal,residual,support


def complete_navigation(scene, config, values, capture_positions=(), structure_ids=None):
    """Return a shallow geometry-preserving copy with only navigation additions."""
    from .loader import geometry_digest
    options = completion_options(values,config)
    result = copy(scene)
    result.geometries = list(scene.geometries)
    result.provenance = deepcopy(scene.provenance)
    positions = np.asarray(capture_positions,dtype=float).reshape((-1,3))
    report = {'method':'bounded_enclosed_planar_holes_navigation_only_v1',
              'frame_id':scene.reference_frame_id,'options':options,
              'default_option_fields':sorted(set(options)-set(values)),
              'span_definition':'longest_side_of_minimum_area_oriented_rectangle',
              'accepted_patches':[], 'rejected_holes':[], 'nonplanar_patch_count':0,
              'added_floor_area_m2':0.,'capture_position_count':len(positions),
              'structural_source_geometry_ids':sorted(structure_ids) if structure_ids is not None else None,
              'observation_surfaces_unchanged':True,'original_collisions_preserved':True,
              'physical_safety_verified':False,'glass_or_opening_classification':False,
              'exterior_gaps_and_disconnected_floor_islands_bridged':False}
    floor_additions = []
    for part in shapely.get_parts(scene.floor):
        for ring in part.interiors:
            hole = Polygon(ring)
            record = {'kind':'floor','area_m2':float(hole.area),'span_m':_span(hole)}
            if record['span_m'] > options['max_hole_span_m'] + 1e-10:
                report['rejected_holes'].append({**record,'reason':'span_limit'})
                continue
            triangles = _triangulate(hole,np.array([0,0,scene.floor_height]),np.array([[1,0,0],[0,1,0]]))
            floor_additions.append(hole)
            # Empty observation faces; collision faces participate in identity
            # and reference transforms, but this floor record is not an obstacle.
            result.geometries.append(Geometry('navigation_completion:floor:'+str(len(floor_additions)),
                np.empty((0,3,3)),None,obstacle=False,collision_triangles=triangles))
            report['accepted_patches'].append({**record,'triangles_m':triangles.tolist()})
    if floor_additions:
        result.floor = shapely.union_all([scene.floor,*floor_additions])
        report['added_floor_area_m2'] = float(result.floor.area-scene.floor.area)
    for hole,origin,basis,normal,residual,support in _patches(scene,options,report,structure_ids):
        record = {'kind':'structure','area_m2':float(hole.area),'span_m':_span(hole),
                  'max_plane_residual_m':residual,'support_geometry_ids':support}
        reason = ('span_limit' if record['span_m'] > options['max_hole_span_m']+1e-10 else
                  'capture_path_crossing' if _path_crosses(hole,origin,basis,normal,positions,options['plane_tolerance_m']) else None)
        if reason:
            report['rejected_holes'].append({**record,'reason':reason})
            continue
        triangles = _triangulate(hole,origin,basis)
        result.geometries.append(Geometry('navigation_completion:structure:'+str(len(report['accepted_patches'])),
            np.empty((0,3,3)),None,solid=False,collision_triangles=triangles))
        report['accepted_patches'].append({**record,'triangles_m':triangles.tolist()})
    report['accepted_patch_count'] = len(report['accepted_patches'])
    report['status'] = 'hypotheses_added' if report['accepted_patch_count'] else 'no_changes'
    result.provenance['navigation_completion'] = report
    result.provenance['geometry_sha256'] = geometry_digest(result.geometries)
    result.provenance.setdefault('input_warnings',[]).append({
        'code':'navigation_completion_unverified',
        'message':'Navigation-only planar completion is a hypothesis: may close real openings or infer unsafe floor; original observations and targets are unchanged',
        'accepted_patch_count':report['accepted_patch_count']})
    return result


def generate_completed_scene(scene_path, output, config=None, **options):
    """Create a reusable derived scene; original files and relative assets stay intact."""
    from .loader import load_scene
    config = Config.from_dict(config)
    scene_path, output = Path(scene_path).resolve(),Path(output).resolve()
    data = read_json(scene_path)
    options = {key:value for key,value in options.items() if value is not None}
    data['navigation_completion'] = completion_options(options,config)
    # References must keep referring to the original producer files, not the
    # new output directory. Upstream manifest-internal paths remain untouched.
    for key in ('objects_manifest','preprocessing_protocol'):
        if isinstance(data.get(key),str): data[key] = str((scene_path.parent/data[key]).resolve())
    for item in data.get('objects',[])+data.get('structures',[])+data.get('navigation_geometry',[]):
        if isinstance(item.get('mesh_uri'),str): item['mesh_uri'] = str((scene_path.parent/item['mesh_uri']).resolve())
    camera = data.get('camera',{})
    if isinstance(camera.get('trajectory_uri'),str):
        camera['trajectory_uri'] = str((scene_path.parent/camera['trajectory_uri']).resolve())
    output.mkdir(parents=True,exist_ok=False)
    target = output/'scene.json'
    with target.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
    completed = load_scene(target,config)
    with (output/'navigation_completion.json').open('x',encoding='utf-8') as stream:
        json.dump(completed.provenance['navigation_completion'],stream,ensure_ascii=False,indent=2,allow_nan=False)
    return target
