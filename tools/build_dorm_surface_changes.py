"""Publish geometry-first Dorm change suspicions and pose-linked original photos.

Reads only whole triangle maps, region transforms/error budgets, camera poses,
and original JPEGs. Never reads instance masks, classes, tracks or event labels.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import open3d as o3d

from dorm_surface_change import (POLICY, classify_patch, coherent_patches,
                                 compare_surfaces, error_baseline, face_geometry,
                                 triangle_scene)
from dorm_material_surfaces import common_wall_exclusions, pane_plane_from_border, reflective_pane_exclusion


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(blob):
    nv, nf = np.frombuffer(blob, '<u4', 2)
    if len(blob) != 8+int(nv)*24+int(nf)*12: raise ValueError('Invalid whole mesh')
    return (np.frombuffer(blob, '<f4', int(nv)*3, 8).reshape(-1, 3).astype(float),
            np.frombuffer(blob, '<u4', int(nf)*3, 8+int(nv)*24).reshape(-1, 3).copy())


def export(path, points, faces):
    ids, inv = np.unique(faces, return_inverse=True)
    p = points[ids]; f = inv.reshape(-1, 3).astype('<u4'); n = np.zeros_like(p)
    _, fn, area = face_geometry(p, f)
    for k in range(3): np.add.at(n, f[:, k], fn*area[:, None])
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    path.write_bytes(np.array([len(p), len(f)], '<u4').tobytes()+p.astype('<f4').tobytes()+n.astype('<f4').tobytes()+f.tobytes())
    return {'file': path.name, 'vertices': len(p), 'faces': len(f), 'bounds': [p.min(0).tolist(), p.max(0).tolist()]}


def panorama_box(pixels, width=1036, height=518):
    x = np.sort(pixels[:, 0] % width)
    gaps = np.diff(np.r_[x, x[0]+width]); start = x[(np.argmax(gaps)+1) % len(x)]
    span = float(np.max((x-start) % width)); pad = 5
    if span > .65*width: return None
    return [float(start-pad), max(0., float(pixels[:, 1].min())-pad),
            float(start+span+pad), min(float(height), float(pixels[:, 1].max())+pad)]


def main(source, output, materials=None, previous=None):
    started = time.time(); source = source.resolve(); output = output.resolve()
    if output.exists(): raise ValueError('Use a new output directory')
    output.mkdir(parents=True)
    ctx = read(source/'alignment-context.json'); meta = read(source/'maps/mesh-metadata.json')
    errors = ctx['methods']['structure']['errors']['regions']; height = ctx['height_u']
    input_names = ['alignment-context.json', 'maps/mesh-metadata.json', 'maps/reference.bin', 'maps/after.bin',
                   'new-inputs/poses-0.json', 'new-inputs/poses-1.json']
    hashes = {n: sha(source/n) for n in input_names}
    maps = []; points = []; faces = []; matrices = [{}, {}]
    keys = ['reference', 'after']
    for e, key in enumerate(keys):
        p, f = decode((source/meta[key]['url']).read_bytes()); parts = {}
        for z, part in meta[key]['parts'].items():
            start, count = part['face_start'], part['faces']; vi = np.unique(f[start:start+count])
            m = np.eye(4)
            if e:
                m = np.array(ctx['methods']['structure']['region_matrices_after_Geo'][int(z[1:])-1]) @ np.array(ctx['base_matrix'])
                bias = np.array(errors[z]['translation_bias_u']); p[vi] += bias; m[:3, 3] += bias
            matrices[e][z] = m; parts[z] = {'face_start': start, 'faces': count}
        # Geometry is unchanged; only the same frozen region bias used by the
        # preceding viewer is applied to the existing aligned display vertices.
        info = export(output/f'map-{e}.bin', p, f)
        info['parts'] = parts
        # export sorts all used vertex IDs, preserving the existing face order.
        maps.append(info); points.append(p); faces.append(f)
    poses = [read(source/f'new-inputs/poses-{e}.json')['poses'] for e in [0, 1]]
    photo_root = source/'object-comparison-v1/photos'
    available = [[r for r in rows if (photo_root/f'Dorm{e+2}-{r["frame"]:05d}.jpg').exists()] for e, rows in enumerate(poses)]
    scenes = [triangle_scene(points[e], faces[e]) for e in [0, 1]]
    whole_normals = [face_geometry(points[e], faces[e])[1] for e in [0, 1]]
    photo_hashes = {}; registry = {}; all_patches = []; audits = {}; zones = []
    material_config = read(materials) if materials else {'panes': []}
    material_audit = {'manual_material_annotations': bool(materials), 'panes': [], 'regions': {}}
    previous_data = read(previous/'tour.json') if previous else None
    previous_audit = read(previous/'surface-audit.json') if previous else None
    used_ids = set()
    next_id = max((int(i['id'][1:]) for i in previous_data['events']+previous_data['records']), default=0)+1 if previous_data else 1

    def views_for(q, epoch, zone, threshold):
        if len(q) > 128: q = q[np.linspace(0, len(q)-1, 128).astype(int)]
        m = matrices[epoch][zone]; inv = np.linalg.inv(m)
        raw = q @ inv[:3, :3].T + inv[:3, 3]
        rows = available[epoch]
        cameras = np.array([np.array(r['camera_to_world'])[:3, 3] for r in rows])
        order = np.argsort(np.linalg.norm(cameras-np.median(raw, 0), axis=1))[:32]
        result = []
        for k in order:
            r = rows[k]; pose = np.array(r['camera_to_world']); origin = cameras[k] @ m[:3, :3].T + m[:3, 3]
            delta = q-origin; depth = np.linalg.norm(delta, axis=1)
            direction = delta/np.maximum(depth[:, None], 1e-12)
            hit = scenes[epoch].cast_rays(o3d.core.Tensor(np.c_[np.broadcast_to(origin, q.shape), direction].astype('f4')), nthreads=2)['t_hit'].numpy()
            finite = np.isfinite(hit); margin = hit-depth
            visible = finite & (np.abs(margin) <= max(.025*height, threshold*.5)) & (depth > .10*height)
            free = finite & (margin > threshold)
            occluded = finite & (margin < -threshold)
            camera = (raw-pose[:3, 3]) @ pose[:3, :3]
            radius = np.linalg.norm(camera, axis=1)
            pix = np.c_[(np.arctan2(camera[:, 0], camera[:, 2])/(2*np.pi)+.5)*1036-.5,
                        (np.arcsin(np.clip(camera[:, 1]/np.maximum(radius, 1e-12), -1, 1))/np.pi+.5)*518-.5]
            box = panorama_box(pix)
            if box is None: continue
            area = (box[2]-box[0])*(box[3]-box[1])/(1036*518)
            if area < .0001: continue
            score = float(np.sqrt(area)*(.15+visible.mean()+free.mean()*.5)*(1-occluded.mean())/(1+np.median(depth)/height))
            result.append({'epoch': epoch, 'frame': int(r['frame']), 'time_s': float(r['timestamp_s']),
                           'file': f'Dorm{epoch+2}-{r["frame"]:05d}.jpg', 'width': 1036, 'height': 518,
                           'bbox_px': box, 'kind': 'surface_projection', 'observed': None, 'vacancy_supported': False,
                           'camera_in_map': origin.tolist(), 'visible_fraction': float(visible.mean()),
                           'free_fraction': float(free.mean()), 'occluded_fraction': float(occluded.mean()),
                           'finite_fraction': float(finite.mean()), 'score': score,
                           'projection': 'same frozen region transform inverted into original camera',
                           'rgb_identity_verified': False})
        return sorted(result, key=lambda v: -v['score'])

    def publish_view(view):
        if not view: return None
        name = view['file']
        if name not in photo_hashes:
            shutil.copy2(photo_root/name, output/name); photo_hashes[name] = sha(output/name)
        return view

    for z in sorted(errors):
        zparts = [maps[e]['parts'][z] for e in [0, 1]]
        zfaces = [faces[e][p['face_start']:p['face_start']+p['faces']] for e, p in enumerate(zparts)]
        zone_bounds = np.array([[points[e][np.unique(zfaces[e])].min(0), points[e][np.unique(zfaces[e])].max(0)] for e in [0, 1]])
        bounds = np.array([zone_bounds[:, 0].min(0), zone_bounds[:, 1].max(0)])
        baseline = []; directions = []
        wall_masks, wall_pairs = common_wall_exclusions(points, zfaces, height, errors[z]['tolerance_u']) if materials else ([np.zeros(len(f), bool) for f in zfaces], [])
        excluded = [m.copy() for m in wall_masks]
        for pane in material_config['panes']:
            if pane['zone'] != z: continue
            e = pane['epoch']; pose = np.array(next(r['camera_to_world'] for r in poses[e] if r['frame'] == pane['frame']))
            n, offset, support = pane_plane_from_border(scenes[e], pane['polygon_px'], pose, matrices[e][z], height)
            mask = reflective_pane_exclusion(points[e], zfaces[e], pane['polygon_px'], pose, matrices[e][z], (n, offset), height, pane.get('material', 'reflective'))
            excluded[e] |= mask
            material_audit['panes'].append({**pane, 'normal': n.tolist(), 'offset': offset, 'frame_support': support, 'excluded_faces': int(mask.sum())})
        material_audit['regions'][z] = {'common_wall_pairs': len(wall_pairs), 'wall_planes': wall_pairs,
                                        'wall_faces': [int(m.sum()) for m in wall_masks],
                                        'total_excluded_faces': [int(m.sum()) for m in excluded]}
        for e in [0, 1]:
            d, tn, nearest = compare_surfaces(points[e], zfaces[e], points[1-e], faces[1-e], scenes[1-e], whole_normals[1-e])
            centers, normals, area = face_geometry(points[e], zfaces[e])
            b = error_baseline(d, normals, tn, area, errors[z]['tolerance_u'], height)
            baseline.append(b); directions.append((d, centers, nearest))
        threshold = max(b['threshold_u'] for b in baseline)
        audits[z] = {'alignment': errors[z], 'baselines': baseline, 'threshold_u': threshold, 'directions': []}
        patch_count = 0
        for e in [0, 1]:
            d, centers, nearest = directions[e]
            patches, counts = coherent_patches(points[e], zfaces[e], d, threshold, height, excluded=excluded[e])
            audits[z]['directions'].append({'epoch': e, 'triangles_compared': len(zfaces[e]), **counts})
            for patch in patches:
                fi = patch.pop('face_ids')+zparts[e]['face_start']; patch['face_ids'] = fi
                patch.update(epoch=e, zone=z, threshold_u=threshold,
                             own_views=views_for(centers[fi-zparts[e]['face_start']], e, z, threshold),
                             opposite_views=views_for(centers[fi-zparts[e]['face_start']], 1-e, z, threshold))
                all_patches.append(patch); patch_count += 1
        middle = (bounds[0]+bounds[1])/2
        overview = []
        for e in [0, 1]:
            # A nearby original camera and the spatial region center are enough
            # for overview navigation; they do not influence patch detection.
            vv = views_for(np.array([middle, middle+[.1, .1, .1]]), e, z, threshold)
            if vv:
                view = dict(vv[0], bbox_px=None, kind='context'); overview.append(publish_view(view))
            else: overview.append(None)
        zones.append({'id': z, 'name': f'구역 {int(z[1:]):02d}', 'bounds': bounds.tolist(),
                      'map_error_u': errors[z]['tolerance_u'], 'threshold_u': threshold,
                      'alignment_supported': errors[z]['supported'], 'views': overview})
        print('REGION', z, 'tau', round(threshold, 4), 'patches', patch_count, flush=True)

    # Opposite-side nearby patches describe one spatial review area, not one
    # chair identity. Every exported face still belongs to a discrepant patch.
    groups = []
    for patch in sorted(all_patches, key=lambda p: (p['zone'], -p['area_u2'], p['epoch'])):
        partner = None
        for group in groups:
            if group[0]['zone'] != patch['zone'] or any(p['epoch'] == patch['epoch'] for p in group): continue
            other = group[0]; gap = np.maximum(0, np.maximum(patch['bounds'][0]-other['bounds'][1], other['bounds'][0]-patch['bounds'][1]))
            if np.linalg.norm(gap) <= .035*height: partner = group; break
        if partner is None: groups.append([patch])
        else: partner.append(patch)

    events = []; records = []; candidate_faces = [[], []]; disposition = {}; index = 0
    reason_text = {
        'coherent_surface_difference': '구역별 정합 오차와 메시 표면 잡음 기준을 넘는 차이가 공간적으로 이어지고, 원래 표면 관측과 반대 맵의 비점유를 서로 떨어진 카메라 시점에서 확인했습니다. 실제 객체 이동·추가·소실은 확정하지 않습니다.',
        'alignment_underconstrained': '표면 차이는 있지만 이 구역은 구조면 방향이 부족하여 정합 오차를 모든 방향에서 제한하지 못했습니다.',
        'insufficient_surface_observation': '표면 차이는 있지만 원본 카메라들에서 해당 메시 표면을 확인하는 독립 관측이 부족합니다.',
        'occlusion_or_incomplete_coverage': '표면 차이는 있지만 반대 촬영의 가림 또는 불완전한 재구성 때문에 실제 공간 차이와 관측 누락을 구분할 근거가 부족합니다.',
    }
    for group in groups:
        index += 1; z = group[0]['zone']
        # Previous geometry areas are used only to keep user-visible IDs stable.
        # Their decisions, thresholds and photos never select new candidates.
        id = None
        if previous_audit:
            best = 0.
            for old_id, old in previous_audit['patch_faces'].items():
                if old_id in used_ids: continue
                overlap = sum(len(np.intersect1d(p['face_ids'], old['epoch_face_ids'][p['epoch']])) for p in group)
                score = overlap/max(1, sum(len(p['face_ids']) for p in group))
                if score > best: best, id = score, old_id
            if best < .25: id = None
        if id is None: id = f'P{next_id:04d}'; next_id += 1
        used_ids.add(id)
        decision, reason, own_counts, free_counts = classify_patch(group, errors[z]['supported'], height)
        mesh_ids = [None, None]; per_epoch = [[], []]
        for p in group: per_epoch[p['epoch']].extend(p['face_ids'].tolist())
        for e, fi in enumerate(per_epoch):
            if not fi: continue
            name = f'object-{id}-{e}.bin'; info = export(output/name, points[e], faces[e][fi]); mid = name[:-4]
            registry[mid] = dict(info, id=mid, epoch=e, zone=z); mesh_ids[e] = mid
            if decision == 'candidate': candidate_faces[e].extend(fi)
        q = np.concatenate([points[p['epoch']][faces[p['epoch']][p['face_ids']]].mean(1) for p in group])
        view_lists = [views_for(q, e, z, group[0]['threshold_u']) for e in [0, 1]]
        chosen = []
        for vv in view_lists:
            distinct = []
            for v in vv:
                if all(abs(v['frame']-w['frame']) >= POLICY.minimum_frame_gap for w in distinct): distinct.append(v)
                if len(distinct) == 3: break
            chosen.append(distinct)
        pairs = [[publish_view(chosen[e][min(i, len(chosen[e])-1)]) if chosen[e] else None for e in [0, 1]]
                 for i in range(max(map(len, chosen), default=0))]
        ratio = float(np.median([p['threshold_ratio'] for p in group]))
        item = {'id': id, 'name': f'{z} · 표면 차이 {id}', 'class_name': 'surface_patch', 'zone': z,
                'decision': decision, 'alternatives': [], 'confirmed': False, 'is_group': False,
                'reason_code': reason, 'reason': reason_text[reason], 'map_error_u': errors[z]['tolerance_u'],
                'threshold_u': group[0]['threshold_u'], 'meshes': mesh_ids, 'views': pairs,
                'geometry_first': True, 'area_u2': sum(p['area_u2'] for p in group),
                'triangles': sum(len(fi) for fi in per_epoch), 'threshold_ratio': ratio,
                'distance_p50_u': float(np.median([p['distance_p50_u'] for p in group])),
                'own_view_counts': own_counts, 'opposite_free_view_counts': free_counts,
                'evidence_score': round(100*(.5*min(1., max(0., ratio-1)/2)+.25*np.mean([p['coherence'] for p in group])+.25*min(1., max(free_counts, default=0)/2))),
                'score_is_probability': False}
        (events if decision == 'candidate' else records).append(item)
        disposition[id] = {'reason': reason, 'epoch_face_ids': per_epoch}
    if previous_data:
        for old in previous_data['events']+previous_data['records']:
            if old['id'] in used_ids: continue
            # Keep the old area inspectable in the exclusion history, without
            # mixing its original faces back into the change overlay.
            item = dict(old, decision='suppressed', reason_code='material_excluded',
                        reason='공통 벽면 또는 명시적으로 지정한 유리·반사성 면을 제외한 뒤 유효한 변화 패치가 남지 않았습니다.')
            for mid in item['meshes']:
                if mid:
                    info = previous_data['object_meshes'][mid]; shutil.copy2(previous/info['file'], output/info['file']); registry[mid] = info
            for pair in item['views']:
                for v in pair: publish_view(v)
            records.append(item)
    highlights = []
    for e in [0, 1]:
        fi = np.unique(candidate_faces[e]).astype(int)
        if len(fi):
            parts = {}
            for z, p in maps[e]['parts'].items():
                local = np.flatnonzero((fi >= p['face_start']) & (fi < p['face_start']+p['faces']))
                if len(local): parts[z] = {'face_start': int(local[0]), 'faces': len(local)}
            highlights.append(dict(export(output/f'changes-{e}.bin', points[e], faces[e][fi]), epoch=e, parts=parts))
        else: highlights.append(None)
    summary = {'automatic_groups': len(events), 'all_automatic_records': len(events)+len(records),
               'record_dispositions': {'deferred': sum(r['decision']=='deferred' for r in records), 'suppressed': sum(r['decision']=='suppressed' for r in records)},
               'triangles_compared': sum(len(f) for f in faces),
               'suspected_triangles': [len(np.unique(fi)) for fi in candidate_faces],
               'input_triangles': [len(f) for f in faces]}
    data = {'kind': 'dorm_change_tour', 'version': 2, 'dataset_id': 'dorm-geometry-first-20261023',
            'analysis_mode': 'geometry_first', 'units': 'reference_units', 'labels': ['Dorm2', 'Dorm3'],
            'height_u': height, 'events': events, 'records': records, 'maps': maps, 'zones': zones,
            'object_meshes': registry, 'highlights': highlights, 'summary': summary,
            'physical_event_ground_truth': False, 'source_version': 'geometry-first-v1',
            'alignment': {'name': ctx['alignment_name'], 'object_specific_alignment': False, 'refitted': False},
            'scope_note': '구역별 정합 오차를 고려한 메시 표면의 변화 의심 구역입니다. 관측 누락과 실제 변화가 완전히 구분됐다는 뜻은 아닙니다.',
            'protocol': {'instance_segmentation_used': False, 'object_classes_used': False,
                         'object_tracks_used': False, 'change_labels_used': False, 'image_detection_gate': False,
                         'all_surface_directions': True, 'alignment_refitted': False,
                         'visibility_is_independent_rgbd_validation': False,
                         'manual_material_masks_used': bool(materials), 'previous_geometry_for_ids_only': bool(previous)}, 'policy': asdict(POLICY),
            'material_summary': {'enabled': bool(materials), 'excluded_areas': summary['record_dispositions']['suppressed'],
                                 'human_panes': sum(p.get('material') != 'static_wall' for p in material_config['panes']),
                                 'human_static_wall_views': sum(p.get('material') == 'static_wall' for p in material_config['panes']),
                                 'excluded_faces': [sum(r['total_excluded_faces'][e] for r in material_audit['regions'].values()) for e in [0,1]]}}
    write(output/'tour.json', data)
    write(output/'surface-audit.json', {'inputs': hashes, 'policy': asdict(POLICY), 'regions': audits,
                                      'patch_faces': disposition, 'photos': photo_hashes,
                                      'summary': summary, 'materials': material_audit, 'elapsed_s': time.time()-started})
    for name, digest in hashes.items():
        if sha(source/name) != digest: raise ValueError('Read-only source changed')
    print('COMPLETE', json.dumps(summary), 'seconds', round(time.time()-started, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--materials', type=Path); parser.add_argument('--previous', type=Path)
    args = parser.parse_args(); main(args.source, args.output, args.materials, args.previous)
