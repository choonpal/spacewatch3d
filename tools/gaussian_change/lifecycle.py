"""Automatic mesh-component identity + rendered occupancy evidence for event types.

Semantic components are object hypotheses, never physical identity ground truth.
Reuses the original mesh SimSaC matching/vacancy gates without capture-specific IDs.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tools.mesh_simsac.core import read, write, transform, sha, similarity
from tools.mesh_simsac.event_matching import POLICY, coverage, rigid_fit, decide_unmatched, movement_rejections, unique_assignments
from .inputs import load_manifest
from .registration import assign
from .instance_observations import build_instances, verification_observations, common_points
from tools.mesh_simsac import mask_instance_matching

LABELS = {8:'창문',10:'수납장',14:'문',15:'테이블',19:'의자',27:'거울',41:'상자',43:'표지판',50:'냉장고',63:'조명',64:'컴퓨터',74:'TV',82:'휴지통',144:'게시판'}


def groups(faces):
    """Face connectivity through shared original mesh vertices."""
    vertex = faces.ravel(); rows = np.repeat(np.arange(len(faces)), 3)
    order = np.argsort(vertex, kind='stable'); same = vertex[order[1:]] == vertex[order[:-1]]
    a, b = rows[order[:-1]][same], rows[order[1:]][same]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(faces), len(faces)))
    return connected_components(graph, directed=False)[1]


def objects_from_mesh(side, epoch, registration):
    path = Path(side['mesh'])
    if path.suffix.lower() != '.npz':
        return [], 'mesh_has_no_semantic_or_instance_arrays'
    with np.load(path, allow_pickle=False) as data:
        field = next((k for k in ['face_semantic','face_class_id'] if k in data), None)
        if field is None:
            return [], 'mesh_has_no_semantic_labels'
        p = (data['p'] if 'p' in data else data['vertices']).astype(float)
        faces, semantic = data['faces'], data[field]
        instance_field = next((k for k in ['face_instance_id','face_instance','face_instances'] if k in data), None)
        instance = data[instance_field].copy() if instance_field else None
    p = transform(p, side['world_from_mesh']); h = registration['height_u']
    common = p.copy()
    if epoch:
        labels = assign(transform(p, registration['global_after_to_before']), registration['centers'])
        for i, r in enumerate(registration['regions']):
            common[labels == i] = transform(p[labels == i], r['matrix_after_to_common'])
    result = []
    rng = np.random.default_rng(100+epoch)
    for cls in np.unique(semantic):
        if int(cls) in [0,3,5,12,255]:
            continue
        fi = np.flatnonzero(semantic == cls)
        ids = instance[fi] if instance is not None else groups(faces[fi])
        for component in np.unique(ids):
            if instance is not None and component < 0:
                continue
            own = fi[ids == component]
            if len(own) < 12:
                continue
            tri = common[faces[own]]
            area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
            if area.sum() < .002*h*h:
                continue
            choices = rng.choice(len(tri), 512, p=area/area.sum())
            uv = rng.random((512,2)); a = np.sqrt(uv[:,0]); weights = np.c_[1-a,a*(1-uv[:,1]),a*uv[:,1]]
            points = np.einsum('ni,nij->nj',weights,tri[choices])
            bounds = np.stack([tri.min((0,1)),tri.max((0,1))])
            region = int(assign(points.mean(0)[None],registration['centers'])[0])
            result.append({'id': f'S{epoch}-{len(result)+1:04d}', 'epoch': epoch, 'class_id': int(cls),
                           'label': LABELS.get(int(cls), f'분할 객체 {int(cls)}'), 'faces': len(own),
                           'segmentation': 'provided_instance' if instance is not None else 'semantic_connected_component',
                           'bounds': bounds.tolist(), 'center': bounds.mean(0).tolist(), 'region_index': region,
                           'regional_supported': registration['regions'][region]['supported'],
                           '_points': points, 'candidate_ids': [], 'link_scores': {}})
    return result, 'provided_instances' if instance is not None else 'semantic_components_not_verified_instances'


def project(points, pose, shape):
    height,width = shape
    q = (points-pose[:3,3]) @ pose[:3,:3]
    distance = np.linalg.norm(q,axis=1)
    longitude = np.arctan2(q[:,0],q[:,2]); latitude = np.arctan2(q[:,1],np.hypot(q[:,0],q[:,2]))
    u = np.floor((longitude/(2*np.pi)+.5)*width).astype(int)%width
    v = np.clip(np.floor((latitude/np.pi+.5)*height).astype(int),0,height-1)
    return u,v,distance


def independent(records, cameras, predicate):
    selected = []
    for r in sorted((r for r in records if predicate(r)),key=lambda r:r['observed_pixels'],reverse=True):
        pos = np.array(cameras[r['view']]['camera_to_common_cv'])[:3,3]
        if all(np.linalg.norm(pos-np.array(cameras[s['view']]['camera_to_common_cv'])[:3,3]) > r['separation_u'] for s in selected):
            selected.append(r)
    return selected


def evidence(objects, folder, cameras, observations, registration):
    h = registration['height_u']
    for obj in objects:
        obj['own_observations'] = []; obj['opposite_observations'] = []; obj['_colors'] = []
    for index,(camera,obs) in enumerate(zip(cameras,observations)):
        directory = folder/obs['directory']; pose = np.array(camera['camera_to_common_cv'])
        geom = []
        for phase in ['before','after']:
            with np.load(directory/(phase+'-geometry.npz')) as d:
                geom.append({k:d[k].copy() for k in ['depth','alpha']})
        images = [np.asarray(Image.open(directory/(p+'.png')).convert('RGB')) for p in ['before','after']]
        with np.load(directory/'inference.npz') as d:
            mask = d['person_union'].copy(); probabilities = [d['before_probability'].copy(),d['after_probability'].copy()]
        for obj in objects:
            points = obj['_points']; u,v,expected = project(points,pose,mask.shape)
            field = registration['regions'][obj['region_index']]['validation']
            err = field.get('plane_scatter_p90_u') or .025*h
            tolerance = max(.05*h,2*err)
            unique = len(np.unique(v*mask.shape[1]+u))
            for phase in [0,1]:
                dep,alpha = geom[phase]['depth'][v,u],geom[phase]['alpha'][v,u]
                valid = (alpha>=.85)&np.isfinite(dep)&(dep>0)&~mask[v,u]&(expected>.01*h)
                valid_fraction = float(valid.mean()); denom = max(int(valid.sum()),1)
                seen = valid & (abs(dep-expected)<=tolerance)
                record = {'view':index,'frame_id':camera['id'],'observed_pixels':unique,
                          'usable': bool(unique>=12 and valid_fraction>=.2), 'projected_fraction':valid_fraction,
                          'occupied_fraction':float(seen.sum()/denom),
                          'free_fraction':float((valid&(dep>expected+tolerance)).sum()/denom),
                          'occluded_fraction':float((valid&(dep<expected-tolerance)).sum()/denom),
                          'change_fraction':float((seen&(probabilities[phase][v,u]>.5)).sum()/max(int(seen.sum()),1)),
                          'separation_u':.1*h}
                if phase == obj['epoch']:
                    obj['own_observations'].append(record)
                    if seen.sum() >= 12:
                        obj['_colors'].append(images[phase][v[seen],u[seen]])
                else:
                    obj['opposite_observations'].append(record)
    for obj in objects:
        own = independent(obj['own_observations'], cameras, lambda r:r['usable'] and r['occupied_fraction']>=.25)
        free = independent(obj['opposite_observations'], cameras, lambda r:r['usable'] and r['free_fraction']>=POLICY['min_free_fraction'] and r['occluded_fraction']<=POLICY['max_occluded_fraction'])
        obj['own_views'] = [r['view'] for r in own]; obj['free_views'] = [r['view'] for r in free]
        if obj['_colors']:
            colors = np.concatenate(obj['_colors'])
            hist = np.concatenate([np.histogram(colors[:,k],bins=16,range=(0,256))[0] for k in [0,1,2]]).astype(float)
            hist /= max(np.linalg.norm(hist),1e-12); obj['_appearance'] = hist
        else:
            obj['_appearance'] = obj.get('_mesh_appearance')
        obj['appearance_source'] = 'gaussian_render_samples' if obj['_colors'] else 'segmented_mesh_vertex_rgb' if obj.get('_mesh_appearance') is not None else 'unavailable'
        obj['own_frame_count'] = obj.get('source_observation_count',len(obj['own_views']))


def retain_motion_origin_hypotheses(pairs, accepted, by_id):
    """Keep motion hypotheses distinct from a uniquely identified source object.

    All physical/appearance gates must already pass; only assignment ambiguity
    may remain. The UI receives every feasible origin, never a certain arrow.
    """
    accepted=list(accepted)
    used_before={p['before'] for p in accepted};used_after={p['after'] for p in accepted}
    ambiguous=[p for p in pairs if p['rejection_reasons']==['ambiguous_assignment']]
    for destination in sorted({p['after'] for p in ambiguous}):
        if destination in used_after:continue
        own=[p for p in ambiguous if p['after']==destination and p['before'] not in used_before]
        if not own:continue
        own.sort(key=lambda p:(-p['appearance'],-p['shape_f1'],p['before']))
        selected=dict(own[0]);selected['origin_identity_ambiguous']=True
        selected['origin_candidates']=[{'object_id':p['before'],'appearance':p['appearance'],'shape_f1':p['shape_f1'],
            'bounds':by_id[p['before']]['bounds'],'center':by_id[p['before']]['center'],'points':by_id[p['before']]['_points'][::2].tolist()} for p in own]
        accepted.append(selected);used_before.add(selected['before']);used_after.add(destination)
    return accepted


def classify(folder, sides, appearance_context=None, reuse_appearance=False):
    folder = Path(folder); result = read(folder/'changes.json'); registration = read(folder/'registration.json')
    cameras = read(folder/'cameras.json')['views']; h = registration['height_u']
    all_objects = []; segmentation = []; instance_audits = []
    for epoch,side in enumerate(sides):
        own,audit = build_instances(side,epoch,registration)
        if own is None:
            own,status = objects_from_mesh(side,epoch,registration)
        else:
            status = 'unary_instance_masks_bound_to_input_mesh'
        all_objects.extend(own); segmentation.append(status);instance_audits.append(audit)
    by_id = {o['id']:o for o in all_objects}
    with np.load(folder/'changes-3d.npz') as d:
        raw_points,raw_labels = d['points'],d['candidate_index']
    candidates = result['candidates']
    for obj in all_objects:
        if not obj['regional_supported']:
            continue
        tree = cKDTree(obj['_points'])
        for index,c in enumerate(candidates):
            if not c['phase_support_pixels'][obj['epoch']]:
                continue
            p = raw_points[raw_labels==index][::max(1,int((raw_labels==index).sum())//300)]
            if not len(p):
                continue
            score = float(np.mean(tree.query(p)[0]<.06*h))
            if score>=.15:
                obj['candidate_ids'].append(c['id']); obj['link_scores'][c['id']] = score
    pixel_links=None
    if all(instance_audits):
        from .instance_pixel_links import link
        pixel_links=link(all_objects,sides,registration,folder,result,cameras)
    # Inspect all objects, so an unchanged counterpart cannot masquerade as a move.
    evidence(all_objects,folder,cameras,result['observations'],registration)
    if all(instance_audits):
        verification_observations(all_objects,sides,registration)
    appearance_audit=None
    if appearance_context and all(instance_audits):
        from .instance_appearance import extract
        appearance_audit=extract(all_objects,sides,h,folder,appearance_context['model'],appearance_context['gaussian_models'])
    elif reuse_appearance and all(instance_audits):
        previous=read(folder/'object-events.json');appearance_audit=read(folder/'instance-appearance.json')
        if previous['source_hashes']['registration.json']!=sha(folder/'registration.json') or appearance_audit['descriptor_sha256']!=sha(folder/'instance-appearance.npz'):
            raise ValueError('Appearance replay input hashes differ')
        old={o['id']:o for o in previous['objects']}
        with np.load(folder/'instance-appearance.npz',allow_pickle=False) as features:
            for obj in all_objects:
                known=old.get(obj['id'],{})
                if known.get('original_face_ids')!=obj.get('original_face_ids') or known.get('source_observation_ids')!=obj.get('source_observation_ids'):
                    raise ValueError('Appearance replay instance binding differs')
                if obj['id'] in features:
                    obj['_embeddings']=features[obj['id']].copy();obj['appearance_source']='same_job_native_gaussian_simsac_vgg_replay'
    def opposite_records(obj):
        actual=[r for r in obj['opposite_observations'] if r.get('source')=='actual_source_camera_mesh_first_hit']
        return actual or obj['opposite_observations']
    def vacancy_records(obj):
        return [r for r in opposite_records(obj) if r['usable'] and r['free_fraction']>=POLICY['min_free_fraction'] and r['occluded_fraction']<=POLICY['max_occluded_fraction']]
    before = [o for o in all_objects if o['epoch']==0]; after = [o for o in all_objects if o['epoch']==1]
    mask_mode=bool(all(instance_audits))
    mesh_trees=[]
    if mask_mode:
        for epoch,side in enumerate(sides):
            with np.load(side['mesh'],allow_pickle=False) as d:
                points=d['p'] if 'p' in d else d['vertices']
            mesh_trees.append(cKDTree(common_points(points,side,epoch,registration)))
    stationary = []
    for a in before:
        for b in after:
            if a['class_id']!=b['class_id'] or np.linalg.norm(np.array(a['center'])-b['center'])>.4*h:
                continue
            f1,_ = coverage(a['_points'],b['_points'],.045*h)
            if f1>=POLICY['min_stationary_f1']:
                stationary.append({'before':a['id'],'after':b['id'],'score':f1})
    fixed = {key for p in unique_assignments(stationary) for key in [p['before'],p['after']]}
    pairs = []
    for a in before:
        for b in after:
            if a['class_id']!=b['class_id'] or a['id'] in fixed or b['id'] in fixed:
                continue
            if not a['candidate_ids'] and not b['candidate_ids']:
                continue
            extents = [np.linalg.norm(np.ptp(o['_points'],axis=0)) for o in [a,b]]
            if min(extents)<.03*h or max(extents)/min(extents)>2.5:
                continue
            if mask_mode:
                fit=mask_instance_matching.rigid_shape_fit(a['_points'],b['_points'],h)
                fit['before_to_after_rigid']=fit['reference_to_source_rigid']
            else:
                fit = rigid_fit(a['_points'],b['_points'],.045*h)
            if a.get('_embeddings') is not None and b.get('_embeddings') is not None:
                from .instance_appearance import similarity as feature_similarity
                appearance=feature_similarity(a['_embeddings'],b['_embeddings'])
            else:
                appearance = float(a['_appearance'] @ b['_appearance']) if a['_appearance'] is not None and b['_appearance'] is not None else 0.
            row = {**fit,'before':a['id'],'after':b['id'],'appearance':appearance,
                   'displacement_u':float(np.linalg.norm(np.array(b['center'])-a['center'])),
                   'frame_support':min(a['own_frame_count'],b['own_frame_count']),
                   'change_supported':bool(a['candidate_ids'] or b['candidate_ids']),
                   'score':float(.75*fit['shape_f1']+.25*appearance)}
            if mask_mode:
                origin=float(np.mean(mesh_trees[1].query(a['_points'])[0]<=.045*h));destination=float(np.mean(mesh_trees[0].query(b['_points'])[0]<=.045*h))
                row.update(reference=a['id'],source=b['id'],origin_stationary_surface_fraction=origin,destination_stationary_surface_fraction=destination,
                           origin_shape_gain=fit['directional_coverage'][0]-origin,destination_shape_gain=fit['directional_coverage'][1]-destination,
                           destination_vacancy_views=len(vacancy_records(b)),origin_free_views=len(vacancy_records(a)),same_position_partner=None)
            pairs.append(row)
    for pair in pairs:
        pair['margin']=1.
        a,b = by_id[pair['before']],by_id[pair['after']]
        error = max(registration['regions'][o['region_index']]['validation'].get('plane_scatter_p90_u') or .025*h for o in [a,b])
        if mask_mode:
            error=max(registration['regions'][o['region_index']]['validation'].get('tolerance_u') or .05*h for o in [a,b])
            pair['rejection_reasons']=mask_instance_matching.assignment_gate(pair,h,error)
            if appearance_audit and pair['appearance']<POLICY['min_appearance']:pair['rejection_reasons'].append('rgb_identity_weak')
        else:
            pair['rejection_reasons'] = movement_rejections(pair,vacancy_records(a),vacancy_records(b),error,.02*h)
    eligible=[p for p in pairs if not p['rejection_reasons']]
    if mask_mode:
        accepted=mask_instance_matching.select_assignments(pairs)
        for p in pairs:p['margin']=p.get('assignment_margin',0.)
        # Event type and exact origin identity are different questions. A newly
        # occupied destination with several viable disappeared counterparts is
        # a motion hypothesis, with ALL origins retained rather than a fabricated
        # certain identity or a blanket "unresolved" surface label.
        accepted=retain_motion_origin_hypotheses(pairs,accepted,by_id)
    else:
        for pair in eligible:
            competitors=[q['score'] for q in eligible if q is not pair and (q['before']==pair['before'] or q['after']==pair['after'])]
            pair['margin']=pair['score']-max(competitors,default=0.)
            if pair['margin']<POLICY['min_match_margin']:pair['rejection_reasons'].append('repeated_object_identity_ambiguous')
        accepted = unique_assignments([p for p in eligible if not p['rejection_reasons']])
    events = []; used = set(); assigned_candidates = set(); excluded=[]
    def event(kind, own, reason, pair=None):
        ids = sorted(set(c for o in own for c in o['candidate_ids']))
        if not ids:
            return
        crows = [c for c in candidates if c['id'] in ids]
        observations = sorted(set(v for c in crows for v in c['observations'])|set(v for o in own for v in o['own_views']))
        bounds = np.array([o['bounds'] for o in own])
        item = {'id':f'E{len(events)+1:04d}','type':kind,'label':own[0]['label'],
                'source_candidate_ids':ids,'object_ids':[o['id'] for o in own], 'reason':reason,
                'observations':observations,'bounds':np.stack([bounds[:,0].min(0),bounds[:,1].max(0)]).tolist(),
                'physical_identity_confirmed':False,'automatic':True}
        item['center'] = np.mean(item['bounds'],axis=0).tolist()
        for epoch,phase in enumerate(['before','after']):
            obj = next((o for o in own if o['epoch']==epoch),None)
            if obj:
                item[phase+'_bounds'] = obj['bounds']; item[phase+'_center'] = obj['center']
                item[phase+'_points'] = obj['_points'][::2].tolist()
                item[phase+'_own_views'] = obj['own_views']; item[phase+'_opposite_free_views'] = obj['free_views']
                item[phase+'_source_frames'] = obj.get('source_frames',[])
                item[phase+'_source_free_frames'] = obj.get('source_free_frames',[])
        if pair:
            item['correspondence'] = pair
            item['exact_origin_confirmed']=not pair.get('origin_identity_ambiguous',False)
            if pair.get('origin_candidates'):item['origin_candidates']=pair['origin_candidates']
        events.append(item); used.update(o['id'] for o in own); assigned_candidates.update(ids)
    for pair in accepted:
        event('moved',[by_id[pair['before']],by_id[pair['after']]],'motion_type_supported_but_origin_has_multiple_candidates' if pair.get('origin_identity_ambiguous') else 'mask_instance_correspondence_shape_gain_and_destination_vacancy_supported' if mask_mode else 'rigid_identity_and_both_rendered_vacancies_supported',pair)
    for obj in all_objects:
        if obj['id'] in used or obj['id'] in fixed or not obj['candidate_ids']:
            continue
        alternatives = [p for p in pairs if (p['before']==obj['id'] or p['after']==obj['id']) and p['shape_f1']>=.5 and p['appearance']>=.7]
        kind,reason = decide_unmatched(obj['epoch'],True,obj['own_frame_count'],opposite_records(obj),bool(alternatives))
        if kind in ['added','removed'] and len(vacancy_records(obj)) < 2:
            kind,reason = 'unresolved','opposite_location_occluded_or_unobserved'
        if kind in ['added','removed']:
            event(kind,[obj],reason)
        elif reason=='surface_still_occupies_expected_location' and any(c not in assigned_candidates for c in obj['candidate_ids']):
            event('modified',[obj],'location_remains_occupied_with_rgb_or_surface_difference')
    for stationary_pair in unique_assignments(stationary):
        own=[by_id[stationary_pair['before']],by_id[stationary_pair['after']]]
        if any(c not in assigned_candidates for o in own for c in o['candidate_ids']):
            event('modified',own,'same_position_object_with_surface_difference')
    for c in candidates:
        if c['id'] in assigned_candidates:
            continue
        linked = [o for o in all_objects if c['id'] in o['candidate_ids']]
        surface=(pixel_links or {}).get('candidate_surface_counts',{}).get(c['id'],{})
        total=sum(surface.values());structural=sum(v for k,v in surface.items() if int(k) in [0,3,5,8,27])
        if not linked and total and structural/total>=.8:
            excluded.append({'candidate_id':c['id'],'reason':'structural_surface_without_supported_object_instance',
                             'surface_class_pixels':surface,'structural_fraction':structural/total,'raw_detection_preserved':True})
            continue
        reasons = sorted(set(reason for p in pairs if any(o['id'] in [p['before'],p['after']] for o in linked) for reason in p['rejection_reasons']))
        item = {**c,'id':f'E{len(events)+1:04d}','type':'unresolved','source_candidate_ids':[c['id']],
                'object_ids':[o['id'] for o in linked], 'reason':'counterpart_or_observation_evidence_insufficient' if linked else 'no_segmented_object_support',
                'rejection_reasons':reasons,'automatic':True,'physical_identity_confirmed':False}
        for epoch,phase in enumerate(['before','after']):
            own = [o for o in linked if o['epoch']==epoch]
            if own:
                obj = max(own,key=lambda o:o['link_scores'][c['id']])
                item[phase+'_bounds'] = obj['bounds']; item[phase+'_center'] = obj['center']; item[phase+'_points'] = obj['_points'][::2].tolist()
        events.append(item)
    public_objects = [{k:v for k,v in o.items() if not k.startswith('_')} for o in all_objects]
    output = {'schema':'gaussian-mesh-object-events-v1','events':events,
              'counts':{kind:sum(e['type']==kind for e in events) for kind in ['added','moved','removed','modified','unresolved']},
              'excluded_surface_candidates':excluded,
              'raw_candidate_count':len(candidates),'segmentation':segmentation,'objects':public_objects,'pair_hypotheses':pairs,
              'instance_segmentation':{'used':any(a is not None for a in instance_audits),'captures':instance_audits},
              'appearance_evidence':'instance-appearance.json' if appearance_audit else None,
              'segmentation_pixel_links':pixel_links,
              'mask_instance_matching_parameters':mask_instance_matching.PARAMETERS if mask_mode else None,
              'stationary_pairs':stationary,'matching_method':'existing_mask_instance_rigid_assignment' if mask_mode else 'component_identity_and_two_vacancies',
              'policy':{**POLICY,'appearance':'Frozen SimSaC VGG features from native Gaussian RGB renders at instance source cameras' if appearance_audit else 'L2 RGB histograms from visible Gaussian render samples',
                        'depth':'Gaussian expected depth for render evidence; actual source camera mesh first-hit for instance vacancy when available',
                        'appearance_fallback':'input segmented mesh vertex RGB','independent_view_separation_h':.1},
              'registration_modified':False,'manual_overrides_used':False,'ground_truth_accuracy_evaluated':False,
              'source_hashes':{'changes-3d.npz':sha(folder/'changes-3d.npz'),'registration.json':sha(folder/'registration.json')}}
    write(folder/'object-events.json',output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest',type=Path,required=True); parser.add_argument('--output',type=Path,required=True);parser.add_argument('--runtime',type=Path);parser.add_argument('--reuse-appearance',action='store_true',help='Diagnostic same-job replay; never enabled by the fresh-input pipeline')
    args = parser.parse_args(); _,sides,_,_ = load_manifest(args.manifest)
    context=None
    if args.runtime:
        from tools.mesh_simsac.model import SimSaC
        from .plyio import load as load_gaussian
        runtime=read(args.runtime);context={'model':SimSaC(runtime['simsac_source'],runtime['simsac_checkpoint']),'gaussian_models':[load_gaussian(s['gaussian']) for s in sides]}
    print(classify(args.output,sides,context,args.reuse_appearance)['counts'])
