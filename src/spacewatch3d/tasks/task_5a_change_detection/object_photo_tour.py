"""Photo-first presentation: instance comparisons and remaining object patches.

The existing geometry detections are unchanged. Semantic foreground and weak
image-track support restrict presentation to object surfaces; they are not
treated as physical-object identity or ground truth.
"""
import argparse
from collections import Counter
import copy
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree

from .dorm_surface import decode, export, panorama_box, read, write
from .surface_geometry import compare_surfaces, face_geometry, triangle_scene

FOREGROUND = {'chair', 'swivel chair', 'table', 'dining table', 'coffee table',
              'cabinet', 'box', 'bag', 'refrigerator', 'ashcan', 'bench', 'sofa',
              'armchair', 'stool', 'computer', 'screen', 'television receiver',
              'tv', 'monitor', 'bottle', 'plant', 'vase', 'book', 'basket',
              'suitcase', 'laptop'}


def object_faces(classes, image_support, centers, height, foreground_ids):
    known = np.isin(classes, list(foreground_ids))
    # A missing semantic label needs separate foreground support and must not
    # reintroduce the low floor through an imprecise 2D detection box.
    weak = (classes == 255) & image_support & (centers[:, 1] > .10*height)
    return known | weak


def region_matrix(context, epoch, zone):
    m = np.eye(4)
    if epoch:
        m = (np.asarray(context['methods']['structure']['region_matrices_after_Geo'][int(zone[1:])-1])
             @ np.asarray(context['base_matrix']))
        m[:3, 3] += context['methods']['structure']['errors']['regions'][zone]['translation_bias_u']
    return m


def camera_matrix(context, pose, epoch, zone):
    result = region_matrix(context, epoch, zone) @ np.asarray(pose['camera_to_world'])
    result[:3, :3] /= np.cbrt(np.linalg.det(result[:3, :3]))
    return result


def build(source, surfaces, instances, output):
    if output.exists(): raise ValueError('Choose a new output directory')
    context = read(source/'alignment-context.json'); data = copy.deepcopy(read(surfaces/'tour.json'))
    inst = read(instances/'tour.json'); audit = read(surfaces/'surface-audit.json')
    if context['alignment_name'] != data['alignment']['name'] or inst['alignment']['name'] != data['alignment']['name']:
        raise ValueError('Instance and surface results must use the same frozen alignment')
    meshes = [decode((surfaces/m['file']).read_bytes()) for m in data['maps']]
    points, faces = [m[0] for m in meshes], [m[1] for m in meshes]
    height = data['height_u']; source_faces = []; gates = []; native_classes = []
    scenes = [triangle_scene(points[e], faces[e]) for e in [0, 1]]
    semantic = read(source/'semantic-lifecycle-newmesh-v1/instances.json')['instances']
    names = {o['class_id']:o['class_name'] for o in semantic}
    foreground_ids = {k for k,v in names.items() if v in FOREGROUND}
    face_audit = read(source/'all-object-review-v1/face-audit.json')
    review = read(source/'all-object-review-v1/review.json')
    image_ids = {i for r in review['records'] if r['mode']=='image' and r['class_name'] in FOREGROUND
                 for i in [r.get('reference'), r.get('source')] if i}
    image_classes = {i:r['class_name'] for r in review['records'] if r['mode']=='image'
                     for i in [r.get('reference'),r.get('source')] if i}
    face_image_classes = [{fi:image_classes[key] for key in image_ids if key in face_audit and face_audit[key]['epoch']==e
                           for fi in face_audit[key]['original_face_ids']} for e in [0,1]]
    for e in [0, 1]:
        native = np.load(source/f'new-inputs/labeled-mesh-{e}.npz')
        tree = cKDTree(native['p'][native['faces']].mean(1)); centers = points[e][faces[e]].mean(1)
        ids = np.zeros(len(faces[e]), int)
        for zone, part in data['maps'][e]['parts'].items():
            start, count = part['face_start'], part['faces']; q = centers[start:start+count]
            if e:
                matrix = region_matrix(context, e, zone)
                inv = np.linalg.inv(matrix); q = q@inv[:3, :3].T+inv[:3, 3]
            error, matched = tree.query(q)
            if error.max() > 1e-5: raise ValueError('Native semantic faces do not match the displayed map')
            ids[start:start+count] = matched
        source_faces.append(ids); classes = native['face_semantic'][ids]; native_classes.append(classes)
        support = set(fi for key in image_ids if key in face_audit and face_audit[key]['epoch']==e
                      for fi in face_audit[key]['original_face_ids'])
        gates.append(object_faces(classes, np.isin(ids, list(support)), centers, height, foreground_ids))
    groups = read(source/'automatic-lifecycle-v2/review.json')['event_groups']
    covered = [set() for _ in [0, 1]]
    for group in groups:
        for key in ['own_mesh', 'opposite_local_mesh']:
            info = group.get(key)
            if info: covered[info['epoch']].update(info['original_face_ids'])
    output.mkdir(parents=True)
    for path in surfaces.iterdir():
        if path.is_file(): shutil.copy2(path, output/path.name)
    registry = data['object_meshes']; events = []; records = []; decisions = {}; retained = {}
    for item in data['events']+data['records']:
        if item['decision']=='suppressed': records.append(item); continue
        kept = []
        for e, old in enumerate(audit['patch_faces'][item['id']]['epoch_face_ids']):
            ids = np.array(old, int)
            keep = gates[e][ids] & ~np.isin(source_faces[e][ids], list(covered[e]))
            kept.append(ids[keep])
        decisions[item['id']] = {'original_faces':[len(x) for x in audit['patch_faces'][item['id']]['epoch_face_ids']],
                                 'remaining_faces':[len(x) for x in kept]}
        if sum(map(len, kept)) < 6: continue
        item['category'] = 'suspect'; item['object_only'] = True; item['triangles'] = sum(map(len, kept))
        retained[item['id']] = [x.tolist() for x in kept]
        item['meshes'] = [None, None]
        counts = Counter(); distances = []; total_area = 0.
        for e, ids in enumerate(kept):
            if not len(ids): continue
            key = f'object-{item["id"]}-{e}'; info = export(output/f'{key}.bin', points[e], faces[e][ids])
            registry[key] = dict(info, id=key, epoch=e, zone=item['zone']); item['meshes'][e] = key
            for fi,cls in zip(ids,native_classes[e][ids]):
                counts[names.get(int(cls),face_image_classes[e].get(int(source_faces[e][fi]),'unclassified'))] += 1
            measured,_,_ = compare_surfaces(points[e],faces[e][ids],points[1-e],faces[1-e],scenes[1-e])
            distances.append(float(np.median(measured))); total_area += float(face_geometry(points[e],faces[e][ids])[2].sum())
        item['object_class'] = counts.most_common(1)[0][0]
        item['distance_p50_u'] = float(np.median(distances)); item['threshold_ratio'] = item['distance_p50_u']/item['threshold_u']; item['area_u2'] = total_area
        # Narrow the displayed region to the retained object surfaces, including
        # their counterpart location; removed background faces cannot widen it.
        q = np.concatenate([points[e][np.unique(faces[e][ids])] for e,ids in enumerate(kept) if len(ids)])
        pairs = ([item['photo_review']['views']] if item.get('photo_review') else [])+item['views']
        for pair in pairs:
            for e,view in enumerate(pair):
                if not view: continue
                pose = read(source/f'new-inputs/poses-{e}.json')['poses']
                camera = camera_matrix(context,next(p for p in pose if p['frame']==view['frame']),e,item['zone'])
                v = (q-camera[:3,3])@camera[:3,:3]; radius = np.linalg.norm(v,axis=1)
                uv = np.c_[(np.arctan2(v[:,0],v[:,2])/(2*np.pi)+.5)*1036-.5,
                           (np.arcsin(np.clip(v[:,1]/np.maximum(radius,1e-9),-1,1))/np.pi+.5)*518-.5]
                view['bbox_px'] = panorama_box(uv)
        (events if item['decision']=='candidate' else records).append(item)
    for item in inst['events']+inst['records']:
        if item['class_name'] not in FOREGROUND: continue
        item = copy.deepcopy(item); item['category'] = 'instance'; item['object_only'] = True
        for mid in item['meshes']:
            if mid:
                info = inst['object_meshes'][mid]; registry[mid] = info; shutil.copy2(instances/info['file'], output/info['file'])
        for pair in item['views']:
            for view in pair:
                if view and not (output/view['file']).exists(): shutil.copy2(instances/view['file'], output/view['file'])
        (events if item['decision']=='candidate' else records).append(item)
    events.sort(key=lambda i:(i['category']!='instance', i['zone'], i['id']))
    data.update(events=events, records=records, analysis_mode='object_photo_tour', photo_first=True)
    data.pop('highlights', None)
    data['summary'].update(automatic_groups=len(events), all_automatic_records=len(events)+len(records), record_dispositions={
        'deferred':sum(r['decision']=='deferred' for r in records),
        'suppressed':sum(r['decision']=='suppressed' for r in records)},
        instance_comparisons=sum(i['category']=='instance' for i in events),
        object_suspicions=sum(i['category']=='suspect' for i in events))
    data['summary']['suspected_triangles'] = [sum(registry[i['meshes'][e]]['faces'] for i in events if i['category']=='suspect' and i['meshes'][e]) for e in [0,1]]
    data['scope_note'] = '인스턴스 비교와 나머지 객체 표면의 변화 의심을 구분합니다. 분할·복원 오차 가능성이 있어 실제 이동·추가·소실 확정은 아닙니다.'
    data['protocol']['object_only_presentation'] = True
    data['protocol']['foreground_labels_used_for_presentation'] = True
    from .whole_object_photo import publish_whole_cutouts
    whole_photo_audit = publish_whole_cutouts(source, data, output)
    write(output/'whole-object-photo-audit.json', whole_photo_audit)
    write(output/'tour.json', data)
    write(output/'object-presentation-audit.json', {'filtering':decisions,'retained_face_ids':retained,'foreground_ids':sorted(foreground_ids),
          'geometry_detector_recomputed':False,'physical_instance_match_invented':False,
          'ghost_source':'complete original RGB instance masks; no mesh clipping; approximate photo-plane placement'})
    print('COMPLETE',data['summary'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','surfaces','instances','output']: p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();build(args.source,args.surfaces,args.instances,args.output)
