"""Attach existing segmented-mesh evidence to frozen SimSaC regions.

Type hints assist review. Semantic components and shape matches are not verified
physical identities. No SimSaC detections, registration or meshes are changed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from prepare import read,save,sha,transform,epoch_to_common
from automatic_object_matching import infer,PARAMETERS,track_points,rigid_shape_fit,assignment_gate

LABELS={'added':'추가 후보','removed':'소실 추정','moved':'이동 추정','unresolved':'유형 불확실'}
CLASS_NAMES={'chair':'의자','door':'문','windowpane':'창문','cabinet':'수납장','table':'테이블','signboard':'표지판'}
POLICY={'minimum_intersecting_faces':3,'minimum_object_face_fraction':.02,
        'minimum_multi_observation_fraction':.5,'minimum_shape_f1':.65,
        'movement_error_multiplier':3.,'minimum_movement_h':.15,
        'minimum_vacancy_views':2,'uses_simsac_pixel_direction_for_type':False,
        'physical_identity_confirmed':False,'labels_are_review_hypotheses':True,
        'searches_same_zone_outside_change_box':True}


def classify_with_context(linked,proposals,pairs,height,error,context,automatic_matches=()):
    """Classify automatically from guarded instance correspondences and presence."""
    automatic=review_type(linked,[],pairs,height,error)
    ids={o['id'] for o in linked};registry={o['id']:o for o in [*linked,*context]}
    eligible=[p for p in automatic_matches if not p['rejection_reasons']
              and (p['reference'] in ids or p['source'] in ids)
              and p['reference'] in registry and p['source'] in registry
              and all(registry[id]['multi_observation_fraction']>=POLICY['minimum_multi_observation_fraction'] for id in [p['reference'],p['source']])
              and registry[p['reference']]['epoch']==0 and registry[p['source']]['epoch']==1
              and registry[p['reference']]['class_name']==registry[p['source']]['class_name']
              and registry[p['reference']]['zone']==registry[p['source']]['zone']]
    if eligible:
        chosen=max(eligible,key=lambda p:p['shape_f1']);a,b=chosen['reference'],chosen['source']
        origin,target=registry[a],registry[b]
        automatic={**automatic,'type':'moved','label':LABELS['moved'],'confidence':'automatic_instance_correspondence',
            'class_name':target['class_name'],'class_label':target['class_label'],
            'reason':f"영상 인스턴스 트랙으로 메쉬를 자동 분리하고, 같은 위치의 기존 대응을 제외한 객체를 1:1 대응했습니다. 형상 F1 {chosen['shape_f1']:.2f}, 출발·도착 위치의 잔존 표면 대비 형상 개선 {chosen['origin_shape_gain']:.2f} / {chosen['destination_shape_gain']:.2f}, 도착 위치의 과거 빈 공간 관측 {chosen['destination_vacancy_views']}회입니다.",
            'selected_pair':[a,b],'alternatives':[],'physical_event_confirmed':False,
            'movement':{'origin_instances':[a],'destination_instance':b,
                'origin_scope':'automatic_mask_partition' if origin.get('automatic_partition') else 'automatic_instance_correspondence',
                'exact_origin_confirmed':False,'origin_label':'출발 · 자동 대응','destination_label':'도착 · 자동 대응',
                'arrow_is_correspondence_not_trajectory':True},
            'automatic_match':chosen}
    elif automatic['type'] in ('added','removed'):
        epoch=0 if automatic['type']=='added' else 1
        others=[o for o in context if o['epoch']==epoch and o['class_name']==automatic.get('class_name')
                and o['faces']>=20 and o['multi_observation_fraction']>=.5]
        if others:
            automatic={**automatic,'type':'unresolved','label':LABELS['unresolved'],'confidence':'ambiguous_correspondence',
                'reason':'같은 구역의 박스 밖에도 같은 종류의 객체가 있으나 자동 대응 조건을 충족하지 못했습니다. 추가·소실·이동을 구분할 근거가 부족합니다.',
                'alternatives':['moved','added','removed']}
    return {**automatic,'automatic':True,'manual_overrides_used':False}


def classify(objects,proposals,static_pairs,height,error):
    """Suggest a review type from object presence/correspondence, never pixels."""
    before=[o for o in objects if o['epoch']==0];after=[o for o in objects if o['epoch']==1]
    ids={o['id'] for o in objects};strong={o['id'] for o in objects if o['multi_observation_fraction']>=POLICY['minimum_multi_observation_fraction']}
    local=[p for p in proposals if p['reference'] in ids and p['source'] in ids and p['reference'] in strong and p['source'] in strong
           and not p['already_paired_with_other'] and p['shape_f1']>=POLICY['minimum_shape_f1']
           and p['displacement_u']>max(POLICY['movement_error_multiplier']*error,POLICY['minimum_movement_h']*height)]
    if local:
        chosen=max(local,key=lambda p:p['shape_f1'])
        return {'type':'moved','label':LABELS['moved'],'confidence':'hypothesis',
                'reason':'전후 분할 객체의 종류·형상 대응 제안과 정합 오차보다 큰 중심 차이가 있습니다. 같은 물리 객체인지는 미확정입니다.',
                'selected_pair':[chosen['reference'],chosen['source']],
                'alternatives':['added','removed'],'physical_event_confirmed':False}
    # A segmentation/appearance difference on an object present in both epochs
    # must not turn into a removal merely because only the before mask fired.
    if before and after:
        return {'type':'unresolved','label':LABELS['unresolved'],'confidence':'deferred',
                'reason':'영역에 전후 객체 메쉬가 모두 존재합니다. 픽셀 변화만으로 추가·소실·이동을 정할 수 없습니다.',
                'selected_pair':next(([p['reference'],p['source']] for p in static_pairs if p['reference'] in ids and p['source'] in ids),None),
                'alternatives':[],'physical_event_confirmed':False}
    if not objects:
        return {'type':'unresolved','label':LABELS['unresolved'],'confidence':'deferred','reason':'기존 분할 메쉬와 직접 겹치는 객체 조각이 없습니다. 재구성 차이·분할 누락 등을 확인해야 합니다.',
                'selected_pair':None,'alternatives':[],'physical_event_confirmed':False}
    present=after or before;epoch=1 if after else 0
    # Ignore other classes (e.g. a cabinet behind an appearing chair), while
    # retaining their mesh overlays as context in the result.
    for obj in sorted(present,key=lambda o:o['intersection_fraction'],reverse=True):
        if obj['id'] not in strong:continue
        if any(p['reference']==obj['id'] or p['source']==obj['id'] for p in static_pairs):continue
        votes=max((t['vacancy_views'] for t in obj['tracks'] if t['supported']),default=0)
        kind='added' if epoch else 'removed'
        return {'type':kind,'label':LABELS[kind],'confidence':'occupancy_supported' if votes>=POLICY['minimum_vacancy_views'] else 'segmentation_only',
                'reason':('후' if epoch else '전')+' 객체 메쉬만 연결되었습니다. '+(f'반대 회차의 독립 빈 공간 관측 {votes}회가 있습니다.' if votes else '반대쪽 부재 관측은 부족하므로 가림·분할 누락을 배제하지 못합니다.'),
                'selected_pair':[None,obj['id']] if epoch else [obj['id'],None],
                'alternatives':['moved'],'physical_event_confirmed':False}
    return {'type':'unresolved','label':LABELS['unresolved'],'confidence':'deferred','reason':'분할 조각의 관측 근거가 약하거나 같은 위치의 반대 객체 대응이 존재합니다.',
            'selected_pair':None,'alternatives':[],'physical_event_confirmed':False}


def review_type(linked,proposals,pairs,height,error):
    """Evaluate classes separately so background furniture cannot hide a chair."""
    classes=sorted({o['class_name'] for o in linked})
    options=[]
    for name in classes:
        objects=[o for o in linked if o['class_name']==name]
        result=classify(objects,proposals,pairs,height,error);result['class_name']=name
        result['class_label']=CLASS_NAMES.get(name,name)
        options.append(result)
    order={'moved':0,'added':1,'removed':2,'unresolved':3}
    options.sort(key=lambda e:(order[e['type']],e['class_name']))
    primary=options[0] if options else classify([],[],[],height,error)
    return {**primary,'per_class':options}


def build(folder,base):
    protocol=read(folder/'protocol.json');prediction=read(folder/'simsac-candidates.json')
    path=base/'semantic-lifecycle-newmesh-v1/instances.json';instances=read(path)
    review_path=base/'automatic-lifecycle-v2/review.json';review=read(review_path)
    (folder/'event-review-overrides.json').unlink(missing_ok=True)
    meshes=[dict(np.load(base/f'new-inputs/labeled-mesh-{e}.npz')) for e in [0,1]]
    errors=protocol['alignment_context']['methods']['structure']['errors']['regions'];height=protocol['height_u']
    prepared={};all_objects={};sources={str(path):sha(path),str(review_path):sha(review_path)}
    for epoch in [0,1]:
        p=base/f'new-inputs/labeled-mesh-{epoch}.npz';sources[str(p)]=sha(p)
    for obj in instances['instances']:
        epoch=obj['epoch'];mesh=meshes[epoch];indices=mesh['faces'][obj['faces']];unique,inverse=np.unique(indices,return_inverse=True)
        m=epoch_to_common(protocol['alignment_context'],epoch,obj['zone'])
        vertices=transform(mesh['p'][unique],m);faces=inverse.reshape(-1,3);centroids=vertices[faces].mean(1)
        np.testing.assert_allclose(transform(obj['raw_sample'],m),obj['common_sample'],atol=1e-7)
        prepared[obj['id']]={'vertices':vertices,'faces':faces,'centroids':centroids,'original_vertex_ids':unique.tolist(),'original_face_ids':obj['faces']}
        tracks=[];tree=cKDTree(vertices)
        for t in review['tracks']:
            family='table' if t['class_name']=='dining table' else t['class_name']
            if t['epoch']!=epoch or t['zone']!=obj['zone'] or family!=obj['class_name']:continue
            distance=float(tree.query(t['center'])[0]);limit=errors[obj['zone']]['tolerance_u']+t['spread_u']
            if distance>limit:continue
            decision=review['track_decisions'][t['id']]
            tracks.append({'id':t['id'],'supported':t['supported'],'frames':len(t['observations']),
                           'surface_distance_u':distance,'vacancy_views':len(decision['vacancy_frames']),
                           'vacancy_frames':decision['vacancy_frames'],'decision':decision['decision']})
        all_objects[obj['id']]={'id':obj['id'],'epoch':epoch,'zone':obj['zone'],'class_name':obj['class_name'],
            'class_label':CLASS_NAMES.get(obj['class_name'],obj['class_name']),'center':obj['center'],
            'bounds':[vertices.min(0).tolist(),vertices.max(0).tolist()],
            'faces':len(faces),'area_u2':obj['area_u2'],'multi_observation_fraction':obj['multi_observation_area_fraction'],
            'physical_instance_id':False,'tracks':tracks,'mesh_file':'object-meshes/'+obj['id']+'.json'}
    pairs=[];proposals={}
    for record in review['records']:
        if record.get('reference') and record.get('source'):
            pairs.append({'reference':record['reference'],'source':record['source'],'status':record['status'],'record_id':record['id']})
        for proposal in record.get('proposals',[]):
            a,b=proposal['reference'],proposal['source'];shape=proposal.get('shape_only')
            if not shape or a not in all_objects or b not in all_objects:continue
            quality=shape['quality'];key=(a,b)
            # The same pair may have been evaluated with a different regional
            # tolerance in each source record. Preserve the conservative score.
            item={'reference':a,'source':b,'reference_zone':all_objects[a]['zone'],'source_zone':all_objects[b]['zone'],
                  'shape_f1':quality['F1'],'shape_rmse_u':quality['inlier_rmse_u'],
                  'displacement_u':float(np.linalg.norm(np.asarray(all_objects[b]['center'])-all_objects[a]['center'])),
                  'origin_free_views':proposal.get('origin_free_views',0),'origin_vacancy_confirmed':proposal.get('origin_vacancy_confirmed',False),
                  'already_paired_with_other':proposal.get('already_paired_with_other'),'physical_id_confirmed':False}
            if key not in proposals or item['shape_f1']<proposals[key]['shape_f1']:proposals[key]=item
    derived,geometry,matching=infer(all_objects,prepared,review['tracks'],review['same_position_pairs'],meshes,protocol['alignment_context'],height,{b['zone'] for b in prediction['boxes']['raw']})
    all_objects.update(derived);prepared.update(geometry)
    save(folder/'automatic-matching.json',matching)
    result={'schema':'segmented-mesh-event-review-v1','policy':POLICY,'sources_sha256':sources,
            'automatic_matching_sha256':sha(folder/'automatic-matching.json'),
            'automatic':True,'manual_overrides_used':False,'new_object_correspondence_inference':True,
            'segmentation_method':instances['method'],'segmentation_limits':instances['limits'],
            'source_prediction_sha256':sha(folder/'simsac-candidates.json'),'regions':{},'objects':{},
            'source_candidates_changed':False,'new_detection_inference':False}
    export=set()
    for mode in ['raw','filtered']:
        result['regions'][mode]={}
        for box in prediction['boxes'][mode]:
            linked=[]
            for id,obj in all_objects.items():
                if obj['zone']!=box['zone']:continue
                p=prepared[id]['centroids'];inside=np.all((p>=box['bounds'][0])&(p<=box['bounds'][1]),axis=1)
                if inside.sum()<POLICY['minimum_intersecting_faces'] or inside.mean()<POLICY['minimum_object_face_fraction']:continue
                linked.append({**obj,'intersecting_faces':int(inside.sum()),'intersection_fraction':float(inside.mean())})
            ids={o['id'] for o in linked}
            matches=[p for p in proposals.values() if p['reference'] in ids or p['source'] in ids]
            matches.sort(key=lambda p:p['shape_f1'],reverse=True)
            static=[p for p in pairs if p['status']=='same_position' and (p['reference'] in ids or p['source'] in ids)]
            classes={o['class_name'] for o in linked if o['class_name'] in ('chair','table','cabinet')}
            context=[o for id,o in all_objects.items() if id not in ids and o['zone']==box['zone'] and o['class_name'] in classes]
            event=classify_with_context(linked,matches,static,height,errors[box['zone']]['tolerance_u'],context,matching['accepted_matches'])
            result['regions'][mode][box['id']]={'event':event,'linked_objects':linked,
                'same_zone_context':context,
                'correspondence_proposals':matches,'automatic_matches':[p for p in matching['accepted_matches'] if p['reference'] in ids or p['source'] in ids],'same_position_pairs':static,
                'original_bounds':box['bounds'],'pixel_counts_used_for_classification':False}
            export|=ids
            export|={o['id'] for o in context}
            if event.get('selected_pair'):export|={id for id in event['selected_pair'] if id}
    mesh_dir=folder/'object-meshes';mesh_dir.mkdir(exist_ok=True)
    for id in sorted(export):
        geo=prepared[id];save(folder/all_objects[id]['mesh_file'],{'positions':geo['vertices'].astype('f4').reshape(-1).tolist(),
            'indices':geo['faces'].astype('u4').reshape(-1).tolist(),'epoch':all_objects[id]['epoch'],'source_instance':id,
            'original_vertex_ids':geo['original_vertex_ids'],'original_face_ids':geo['original_face_ids']})
        all_objects[id]['mesh_sha256']=sha(folder/all_objects[id]['mesh_file'])
        result['objects'][id]=all_objects[id]
    result['counts']={t:sum(r['event']['type']==t for r in result['regions']['filtered'].values()) for t in LABELS}
    save(folder/'object-evidence.json',result)
    print('Object evidence:',result['counts'],len(export),'original segmented meshes')
    return result


def validate(folder,metadata_only=False):
    evidence=read(folder/'object-evidence.json');pred=read(folder/'simsac-candidates.json');protocol=read(folder/'protocol.json')
    assert sha(folder/'simsac-candidates.json')==evidence['source_prediction_sha256']
    assert not evidence['source_candidates_changed'] and not evidence['new_detection_inference']
    assert evidence['policy']==POLICY
    matching=read(folder/'automatic-matching.json')
    assert sha(folder/'automatic-matching.json')==evidence['automatic_matching_sha256']
    assert matching['parameters']==PARAMETERS and matching['automatic'] and not matching['manual_overrides_used']
    assert evidence['automatic'] and not evidence['manual_overrides_used']
    assert not (folder/'event-review-overrides.json').exists()
    errors=protocol['alignment_context']['methods']['structure']['errors']['regions']
    mesh_inputs={};source_instances=None;source_review=None
    if not metadata_only:
        for name,digest in evidence['sources_sha256'].items():
            assert sha(name)==digest,name
            if Path(name).name.startswith('labeled-mesh-'):
                epoch=int(Path(name).stem.rsplit('-',1)[1]);mesh_inputs[epoch]=dict(np.load(name))
            if Path(name).name=='instances.json':source_instances={o['id']:o for o in read(name)['instances']}
            if Path(name).name=='review.json':source_review=read(name)
    geometries={}
    for id,obj in evidence['objects'].items():
        path=folder/obj['mesh_file'];assert sha(path)==obj['mesh_sha256']
        geo=read(path);assert geo['source_instance']==id and geo['epoch']==obj['epoch']
        vertices=np.asarray(geo['positions']).reshape(-1,3);faces=np.asarray(geo['indices']).reshape(-1,3)
        assert np.isfinite(vertices).all() and len(faces)==obj['faces']
        assert faces.min()>=0 and faces.max()<len(vertices)
        if not metadata_only:
            epoch=obj['epoch'];raw=mesh_inputs[epoch];original=source_instances[obj.get('parent_instance',id)]
            if obj.get('automatic_partition'):assert set(geo['original_face_ids'])<=set(original['faces'])
            else:assert geo['original_face_ids']==original['faces']
            indices=np.asarray(geo['original_vertex_ids']);np.testing.assert_array_equal(indices,np.unique(raw['faces'][geo['original_face_ids']]))
            expected=transform(raw['p'][indices],epoch_to_common(protocol['alignment_context'],epoch,obj['zone']))
            np.testing.assert_allclose(vertices,expected,atol=2e-6)
            np.testing.assert_array_equal(indices[faces],raw['faces'][geo['original_face_ids']])
        geometries[id]=(vertices,faces)
    assert all(o.get('physical_instance_id') is False for o in evidence['objects'].values())
    for pair in matching['accepted_matches']:
        assert pair['method']=='automatic_instance_track_rigid_assignment' and not pair['rejection_reasons']
        error=errors[pair['reference_zone']]['tolerance_u'];assert not assignment_gate(pair,protocol['height_u'],error)
        assert pair['assignment_margin']>=PARAMETERS['minimum_assignment_margin']
        if not metadata_only:
            lookup={t['id']:t for t in source_review['tracks']};a=lookup[pair['reference_track']];b=lookup[pair['source_track']]
            assert a['supported'] and b['supported'] and a['zone']==b['zone'] and a['class_name']==b['class_name']
            protected={p[k] for p in source_review['same_position_pairs'] for k in ['reference','source']}
            assert a['id'] not in protected and b['id'] not in protected
            before=track_points(a,protocol['alignment_context'],protocol['height_u']);after=track_points(b,protocol['alignment_context'],protocol['height_u'])
            fit=rigid_shape_fit(before,after,protocol['height_u'])
            for k in ['shape_f1','directional_coverage','reference_to_source_rigid']:np.testing.assert_allclose(fit[k],pair[k],atol=1e-10)
            tolerance=protocol['height_u']*PARAMETERS['shape_tolerance_h']
            opposite=transform(mesh_inputs[1]['p'],epoch_to_common(protocol['alignment_context'],1,a['zone']))
            origin_fraction=float(np.mean(cKDTree(opposite).query(before)[0]<=tolerance))
            dest_fraction=float(np.mean(cKDTree(mesh_inputs[0]['p']).query(after)[0]<=tolerance))
            np.testing.assert_allclose([origin_fraction,dest_fraction],[pair['origin_stationary_surface_fraction'],pair['destination_stationary_surface_fraction']],atol=1e-10)
            assert pair['destination_vacancy_views']==len(source_review['track_decisions'][b['id']]['vacancy_frames'])
    for mode,boxes in pred['boxes'].items():
        assert set(evidence['regions'][mode])=={b['id'] for b in boxes}
        for box in boxes:
            record=evidence['regions'][mode][box['id']];assert record['original_bounds']==box['bounds']
            assert not record['pixel_counts_used_for_classification']
            for obj in record['linked_objects']:
                vertices,faces=geometries[obj['id']];points=vertices[faces].mean(1)
                inside=np.all((points>=box['bounds'][0])&(points<=box['bounds'][1]),axis=1)
                assert int(inside.sum())==obj['intersecting_faces']
                assert np.isclose(float(inside.mean()),obj['intersection_fraction'])
            expected=classify_with_context(record['linked_objects'],record['correspondence_proposals'],record['same_position_pairs'],protocol['height_u'],errors[box['zone']]['tolerance_u'],record['same_zone_context'],matching['accepted_matches'])
            assert expected==record['event']
            assert record['event']['automatic'] and not record['event']['manual_overrides_used']
            assert 'review' not in record['event'] and 'automatic_before_review' not in record['event']
            for obj in record['same_zone_context']:
                assert obj['zone']==box['zone'] and obj['id'] in geometries
            assert not record['event']['physical_event_confirmed']
    return {'passed':True,'metadata_only':metadata_only,'segmented_meshes':len(geometries),
            'review_hints':evidence['counts'],'original_faces_preserved':not metadata_only,
            'original_simsac_regions_unchanged':True,'simsac_pixel_direction_used_for_type':False,
            'physical_event_types_confirmed':False,'automatic':True,'manual_overrides_used':False,
            'automatic_matches':len(matching['accepted_matches'])}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('outputs/scenediff-mesh-rgb-20261024/gaussian-uncapped-tour-20261026'))
    p.add_argument('--base',type=Path,default=Path('outputs/registration-benchmark-20261019/new-meshfirst-20261022'))
    a=p.parse_args();build(a.directory,a.base);print(validate(a.directory))
