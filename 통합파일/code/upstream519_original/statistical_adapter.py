"""Apply the frozen v5.1.8 statistical policy to Dorm2's own semantic mesh."""
import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import shapely
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon

sys.dont_write_bytecode = True
ACTIVE = Path('/workspace/spacewatch_pipeline_v5_1/profiles/dorm3_statistical_scored')
SURFACE = Path('/workspace/task5b_dorm3_statistical_surface_20261026_EY66id')
RESIDUAL = Path('/workspace/task5b_dorm3_shape_residual_20261026_kAm2a3')
FLAT = Path('/workspace/task5b_dorm3_flat_floor_20261026_4d5KI3')
GROUND = Path('/workspace/task5b_dorm3_ground_roles_20261026_rU9tZ1')
BASE = Path('/workspace/spacewatch_runs/dorm2_dorm3_pipeline_v5_1_7_20261026')
DORM3_RUN = Path('/workspace/spacewatch_runs/dorm3_pipeline_v5_1_8_20261026')


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def components(faces):
    owners = np.repeat(np.arange(len(faces)), 3)
    corners = faces.ravel()
    order = np.argsort(corners, kind='stable')
    same = np.diff(corners[order]) == 0
    a, b = owners[order[1:]][same], owners[order[:-1]][same]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(faces), len(faces)))
    count, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == i) for i in range(count)]


def export_obj(p, vertices, faces):
    used, inv = np.unique(faces, return_inverse=True)
    with Path(p).open('w') as stream:
        np.savetxt(stream, vertices[used], fmt='v %.17g %.17g %.17g')
        np.savetxt(stream, inv.reshape(-1, 3) + 1, fmt='f %d %d %d')


def classify(data, catalog):
    from dual_size import pca_extents
    from surface_features import mesh_adjacency, component_features
    from surface_supplement import lower_components
    from residual_shape import extra_exclusions
    v, f, labels = (data[k] for k in ['vertices', 'faces', 'face_class_id'])
    tri = v[f].astype(float)
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    _, canonical = np.unique(v, axis=0, return_inverse=True)
    context = mesh_adjacency(v, f)
    rows, groups, reports = [], [], []
    for cid in np.unique(labels):
        entry = catalog.get(int(cid), {})
        name = entry.get('name', 'unknown')
        if name in ['wall', 'walls', 'floor', 'ceiling', 'ceilings', 'tactile paving', 'unknown'] or not entry.get('instance_eligible', True):
            continue
        selected = np.flatnonzero(labels == cid)
        indices = []
        for i, ids in enumerate(components(canonical[f[selected]])):
            ids = selected[ids]
            area = float(areas[ids].sum())
            indices.append(len(rows))
            groups.append(ids)
            rows.append(dict(instance_id=f'stat_class_{cid}_cc_{i:04d}', class_id=int(cid), class_name=name,
                             faces=len(ids), area=area, pca_spans=pca_extents(v[np.unique(f[ids])]).tolist(),
                             excluded=area <= 0, zero_area=area <= 0))
        positive = [i for i in indices if rows[i]['area'] > 0]
        features = component_features(context, [groups[i] for i in positive], labels)
        flags, report = lower_components([rows[i]['area'] for i in positive],
                                         [rows[i]['pca_spans'] for i in positive], features)
        for i, flag, feature in zip(positive, flags, features):
            rows[i].update(excluded=bool(flag), geometry_features=feature, surface_excluded=bool(flag))
        extra, residual = extra_exclusions([rows[i]['area'] for i in positive],
                                           [rows[i]['pca_spans'] for i in positive], flags)
        for i, flag in zip(positive, extra):
            rows[i]['excluded'] = rows[i]['excluded'] or bool(flag)
            rows[i]['residual_excluded'] = bool(flag)
        reports.append(dict(class_id=int(cid), class_name=name, surface=report, residual=residual,
                            source_components=len(indices), excluded=sum(rows[i]['excluded'] for i in indices),
                            retained=sum(not rows[i]['excluded'] for i in indices)))
    return rows, groups, reports


def main(root):
    profile = root / 'profile'
    profile.mkdir(parents=True, exist_ok=False)
    protected = {}
    for directory in ['code', 'native']:
        shutil.copytree(ACTIVE / directory, profile / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['profile_runner.py', 'integrated_runner.py', 'cache_optimization.py', 'fast_execution.py', 'manual.json', 'prompt.txt']:
        shutil.copy2(ACTIVE / name, profile / name)
    algorithms = profile / 'algorithms'
    algorithms.mkdir()
    files = {name: SURFACE for name in ['statistical_modes.py', 'dual_size.py', 'middle_size.py', 'fusion_guard.py',
                                       'tail_guard.py', 'surface_features.py', 'surface_supplement.py']}
    files.update({'residual_shape.py': RESIDUAL, 'flat_floor.py': FLAT, 'ground_roles.py': GROUND})
    for name, folder in files.items():
        shutil.copy2(folder / name, algorithms / name)
        protected[str(folder / name)] = sha(folder / name)
        assert sha(folder / name) == sha(algorithms / name)
    sys.path.insert(0, str(algorithms))
    # Read-only policy replay against the existing Dorm3 result ensures this
    # adapter composes all statistical stages exactly before processing Dorm2.
    d3 = read(DORM3_RUN / 'results.json')
    source3 = d3['dorm3.semantic']
    with np.load(source3['arrays'], allow_pickle=False) as z:
        arrays3 = {k: z[k] for k in ['vertices', 'faces', 'rgb', 'face_class_id']}
    binding = read(ACTIVE / 'input_binding.json')
    assert all(array_digest(arrays3[k]) == h for k, h in binding['arrays'].items())
    catalog3 = {r['id']: r for r in read(source3['vocabulary'])}
    rows3, _, _ = classify(arrays3, catalog3)
    original3 = {r['instance_id']: r['excluded'] for r in read(ACTIVE / 'filter.json')['objects']}
    actual3 = {r['instance_id']: r['excluded'] for r in rows3}
    assert actual3 == original3, 'Statistical adapter does not reproduce the frozen v5.1.8 policy'
    save(root / 'policy-replay.json', dict(passed=True, capture='dorm3', components=len(rows3),
                                         retained=sum(not r['excluded'] for r in rows3),
                                         exact_existing_exclusion_mask=True, planning_rerun=False))
    protected[str(ACTIVE / 'filter.json')] = sha(ACTIVE / 'filter.json')
    for name in ['plan', 'requests']:
        p = d3['dorm3.plan'][name]
        protected[p] = sha(p)
    del arrays3, rows3
    results = read(BASE / 'results.json')
    source = results['dorm2.semantic']
    base_plan = read(results['dorm2.plan']['plan'])
    base_scene = read(results['dorm2.scene']['scene'])
    with np.load(source['arrays'], allow_pickle=False) as z:
        data = {k: z[k] for k in ['vertices', 'faces', 'rgb', 'face_class_id']}
    v, f, labels = (data[k] for k in ['vertices', 'faces', 'face_class_id'])
    catalog = {r['id']: r for r in read(source['vocabulary'])}
    rows, groups, reports = classify(data, catalog)
    assets = profile / 'assets'
    assets.mkdir()
    targets = np.zeros(len(f), 'u4')
    assigned = np.zeros(len(f), 'u1')
    objects, mapping = [], []
    for i, (row, ids) in enumerate(zip(rows, groups), 1):
        if row['excluded']:
            continue
        path = assets / (row['instance_id'] + '.obj')
        export_obj(path, v, f[ids])
        targets[ids] = i
        assigned[ids] += 1
        objects.append(dict(instance_id=row['instance_id'], mesh_uri=str(path), class_name=row['class_name'],
                            class_id=row['class_id'], source_triangle_count=row['faces'], solid=False,
                            description='Connected semantic region; physical object identity unverified'))
        mapping.append(dict(region_id=i, instance_id=row['instance_id'], class_id=row['class_id'],
                            class_name=row['class_name'], faces=row['faces']))
    remainder = targets == 0
    export_obj(assets / 'remainder.obj', v, f[remainder])
    assigned[remainder] += 1
    assert np.all(assigned == 1)
    import flat_floor
    flat_floor.SERVER_PRIOR = algorithms / 'ground_roles.py'
    original_floor = shapely.union_all([Polygon(p['polygon'], p.get('holes', [])) for p in base_scene['floor']['components']])
    floor, collision, accepted, ground_report = flat_floor.prepare_navigation(
        v, f, labels, {k: r['name'] for k, r in catalog.items()}, original_floor, base_scene['floor']['height_m'])
    export_obj(assets / 'navigation_collision.obj', v, f[collision])
    scene = copy.deepcopy(base_scene)
    scene.update(objects=objects, structures=[dict(mesh_uri=str(assets / 'remainder.obj'), solid=False)],
                 navigation_geometry=[dict(mesh_uri=str(assets / 'navigation_collision.obj'), solid=False)])
    scene['floor']['components'] = [dict(polygon=list(p.exterior.coords), holes=[list(r.coords) for r in p.interiors])
                                    for p in flat_floor.polygon_parts(floor)]
    scene['floor']['estimation_method'] = 'semantic_floor_full_xy_flat_indoor_prior_v2'
    scene['navigation_profile'] = dict(adapter_version='flat-ground-role-v2',
        route_grid_anchor_xy_m=base_plan['route_graph']['grid_origin_xy_m'],
        fallback_tile_origin_xy_m=base_plan['coverage_report']['surface_sampling']['fallback_tiles']['origin_xy_m'],
        ground_role_report=str(profile / 'ground-role-report.json'), original_observation_geometry_unchanged=True,
        physical_safety_verified=False)
    scene['input_assumptions'].update(single_level_flat_indoor_assumption_from_v518=True,
        navigation_floor_projection_not_source_mesh_reconstruction=True, physical_safety_verified=False)
    ground_report['user_provided_prior'] = 'Same single-level flat-indoor policy as v5.1.8; transferred to Dorm2 at user request, physical ground truth unverified.'
    scene['source_context'].update(original_mesh_faces=len(f), required_objects=len(objects), residual_faces=int(remainder.sum()),
        class_size_templates_used=False, geometry_deleted_faces=0, source_arrays=source['arrays'],
        target_policy='v5.1.8 statistical semantic components with residual-shape screen',
        physical_object_identity_verified=False)
    cfg = read(ACTIVE / 'config.json')
    cfg['camera_height_m'] = base_plan['provenance']['config']['camera_height_m']
    assert cfg['require_all_objects'] is False
    assert all(cfg[k] is None for k in ['max_grid_cells', 'max_route_nodes', 'max_candidates', 'max_visibility_probes', 'max_triangles'])
    class_scores = read(ACTIVE / 'score_summary.json')['class_scores']
    scores, fallback = {}, []
    for row in objects:
        name = row['class_name']
        if name in class_scores:
            score = class_scores[name]
            scores[row['instance_id']] = dict(score=score['score'], reason=f"Existing v5.1.8 class-level score: L{score['level']} / {score['line']}")
        else:
            fallback.append(name)
    # Missing classes intentionally use the planner's documented default=0.5.
    # Do not label these defaults as a cached LLM classification.
    save(profile / 'importance_cache.json', dict(scene_id=scene['scene_id'], capture_id='dorm2', scores=scores,
        default_classes=sorted(set(fallback)), default_importance=cfg['default_importance'], new_llm_calls=0))
    save(profile / 'scene.json', scene)
    save(profile / 'config.json', cfg)
    save(profile / 'ground-role-report.json', ground_report)
    save(profile / 'target_mapping.json', mapping)
    save(profile / 'filter.json', dict(status='algorithm_applied_physical_labels_unverified',
         source_semantic_components=len(rows), excluded_components=sum(r['excluded'] for r in rows),
         retained_components=len(objects), geometry_deleted_faces=0, source_faces=len(f),
         class_size_templates_used=False, classes=reports, objects=rows))
    np.savez_compressed(profile / 'face_roles.npz', source_face_id=np.arange(len(f)), target_region_index=targets, source_class_id=labels)
    np.savez_compressed(profile / 'navigation_roles.npz', source_face_id=np.arange(len(f)), collision_mask=collision, accepted_unknown_support=accepted)
    save(profile / 'input_binding.json', dict(capture_id='dorm2', frame_id='dorm2_mesh',
         coordinates_sha256=sha(source['coordinates']), source_arrays=source['arrays'],
         arrays={k: array_digest(a) for k, a in data.items()}))
    # Independently reopen the exported geometry, including excluded residuals.
    triangles = np.concatenate([trimesh.load_scene(o['mesh_uri'], process=False).to_mesh().triangles
                                for o in scene['objects'] + scene['structures']])
    def signature(t):
        t = np.ascontiguousarray(t, dtype='f8').reshape(-1, 9)
        return np.sort(t.view(np.dtype((np.void, 72))).ravel())
    assert len(triangles) == len(f) and np.array_equal(signature(triangles), signature(v[f]))
    assert scene['camera'] == base_scene['camera'] and scene['coordinates'] == base_scene['coordinates']
    for p in [source['arrays'], source['vocabulary'], source['coordinates'], results['dorm2.plan']['plan'], results['dorm2.scene']['scene']]:
        protected[p] = sha(p)
    for p in ACTIVE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            protected[str(p)] = sha(p)
    save(root / 'protocol.json', dict(capture='dorm2', planning_policy_version='5.1.8', source_run=str(BASE),
         v518_profile=str(ACTIVE), source_hashes=protected, original_faces=len(f),
         all_original_triangles_preserved_exactly_once=True, dormant_dorm3_outputs_unchanged=True,
         camera_and_coordinate_metadata_exact=True, camera_height_m=cfg['camera_height_m'],
         only_config_difference_from_dorm3=['camera_height_m'], object_count=len(objects),
         expected_keyframe_count=None, no_keyframe_count_cap=True, no_class_size_templates=True,
         default_importance_classes=sorted(set(fallback)), default_importance=cfg['default_importance'],
         new_llm_calls=0, generated_views_used_as_training=False))
    save(profile / 'checksums.json', {p.relative_to(profile).as_posix(): sha(p) for p in profile.rglob('*')
                                    if p.is_file() and '__pycache__' not in p.parts})
    print(json.dumps(dict(stage='prepared', capture='dorm2', components=len(rows), targets=len(objects),
         excluded=sum(r['excluded'] for r in rows), faces=len(f), fallback_classes=sorted(set(fallback)),
         source_geometry_preserved=True, dorm3_policy_replay=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
