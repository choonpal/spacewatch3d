"""Adapt frozen Dorm alignment + automatic events for the Auto Tour viewer.

Inputs are read-only. No inference, registration fitting or decision overrides.
Background and object geometry use the same frozen region transforms/biases.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def import_tour(source,output):
    source=source.resolve();output=output.resolve()
    if output.exists():raise ValueError('Choose a new output directory')
    output.mkdir(parents=True)
    automatic_path=source/'automatic-lifecycle-v2/review.json';auto=read(automatic_path)
    review=read(source/'all-object-review-v1/review.json');context=read(source/'alignment-context.json');meta=read(source/'maps/mesh-metadata.json')
    hashes={name:sha(source/name) for name in ['automatic-lifecycle-v2/review.json','all-object-review-v1/review.json','alignment-context.json','input-provenance.json','maps/mesh-metadata.json']}
    poses=[{r['frame']:r for r in read(source/f'new-inputs/poses-{e}.json')['poses']} for e in [0,1]]
    files={};copied={}
    def copy(path,name):
        path=path.resolve()
        if not path.is_relative_to(source):raise ValueError('Asset leaves source bundle')
        if not re.fullmatch(r'[A-Za-z0-9_-]+\.(bin|jpg)',name):raise ValueError('Unsafe output asset name')
        if name in copied:
            if copied[name]!=path:raise ValueError('Asset name collision')
            return name
        shutil.copy2(path,output/name);copied[name]=path
        files[name]={'bytes':(output/name).stat().st_size,'sha256':sha(output/name),'source':path.relative_to(source).as_posix()}
        return name
    mesh_registry={}
    def object_mesh(info,base):
        if not info or not info.get('url') or not info.get('faces'):return None
        path=(base/info['url']).resolve();filename='object-'+path.name
        copy(path,filename)
        key=filename[:-4]
        mesh_registry[key]={'id':key,'file':filename,'epoch':info['epoch'],'zone':info['zone'],
            'vertices':info['vertices'],'faces':info['faces'],'bounds':info['bounds']}
        return key
    def frame(view,base,zone):
        if not view:return None
        e=int(view['epoch']);ordinal=int(view['frame']);name=f'Dorm{e+2}-{ordinal:05d}.jpg'
        copy((base/view['image_url']).resolve(),name)
        pose=np.array(poses[e][ordinal]['camera_to_world'],float)
        m=np.eye(4)
        if e:
            m=np.array(context['methods']['structure']['region_matrices_after_Geo'][int(zone[1:])-1])@np.array(context['base_matrix'])
            m[:3,3]+=np.array(context['methods']['structure']['errors']['regions'][zone]['translation_bias_u'])
        camera=pose[:3,3]@m[:3,:3].T+m[:3,3]
        return {'epoch':e,'frame':ordinal,'time_s':float(view.get('timestamp_s',poses[e][ordinal]['timestamp_s'])),
            'file':name,'width':1036,'height':518,'bbox_px':view.get('bbox_unwrapped_px'),
            'kind':view.get('kind','context'),'observed':view.get('detected'),
            'vacancy_supported':bool(view.get('vacancy_vote_accepted',False)),
            'camera_in_map':camera.tolist()}
    def pairs(views,base,zone):
        return [[frame(v.get('reference'),base,zone),frame(v.get('source'),base,zone)] for v in views]
    events=[]
    for group in auto['event_groups']:
        ids=[object_mesh(group.get(k),source/'automatic-lifecycle-v2') for k in ['own_mesh','opposite_local_mesh']]
        if any(ids[e] and mesh_registry[ids[e]]['epoch']!=e for e in [0,1]):raise ValueError('Event mesh epoch mismatch')
        events.append({'id':group['id'],'name':group['title'],'class_name':group['class_name'],'zone':group['zone'],
            'decision':'candidate','alternatives':group['event_alternatives'],'confirmed':False,
            'reason_code':'repeated_presence_and_opposite_vacancy','reason':group['review']['reason'],
            'map_error_u':group['map_error_u'],'meshes':ids,'views':pairs(group['views'],source/'automatic-lifecycle-v2',group['zone']),
            'trigger_tracks':group['entry_ids'],'is_group':True})
    records=[]
    for record in auto['records']:
        decision=record['automatic_decision'];ids=[]
        for key in ['reference','source']:
            info=review['meshes'].get(record.get(key));ids.append(object_mesh(info,source/'all-object-review-v1'))
        records.append({'id':record['id'],'name':record['class_name'],'class_name':record['class_name'],'zone':record['zone'],
            'decision':decision['decision'],'alternatives':[],'confirmed':False,'is_group':False,
            'reason_code':decision['reason_code'],'reason':record['review']['reason'],
            'map_error_u':record['map_error_u'],'meshes':ids,
            'views':pairs(record['views'],source/'automatic-lifecycle-v2',record['zone'])})
    zones=[]
    errors=context['methods']['structure']['errors']['regions']
    for region in review['regions']:
        zone=region['zone'];points=[]
        for epoch,key in enumerate(['reference','after']):
            bounds=np.array(meta[key]['parts'][zone]['bounds'])
            if epoch:bounds+=np.array(errors[zone]['translation_bias_u'])
            points.extend(bounds.tolist())
        points=np.array(points);views=[]
        for v in region['overview_views']:
            views.append(frame(v,source/'all-object-review-v1',zone))
        zones.append({'id':zone,'name':f'구역 {int(zone[1:]):02d}',
            'bounds':[points.min(0).tolist(),points.max(0).tolist()],
            'map_error_u':errors[zone]['tolerance_u'],'views':views})
    maps=[];map_audit=[]
    for e,key in enumerate(['reference','after']):
        original=(source/meta[key]['url']).read_bytes();blob=bytearray(original)
        nv,nf=map(int,np.frombuffer(blob,'<u4',2));points=np.ndarray((nv,3),'<f4',buffer=blob,offset=8)
        faces=np.ndarray((nf,3),'<u4',buffer=blob,offset=8+nv*24)
        if len(blob)!=8+nv*24+nf*12:raise ValueError('Whole mesh format mismatch')
        used=set();parts={}
        for zone,part in meta[key]['parts'].items():
            start,count=part['face_start'],part['faces'];vertices=np.unique(faces[start:start+count])
            if used.intersection(vertices.tolist()):raise ValueError('Expected duplicated region boundary vertices')
            used.update(vertices.tolist())
            if e:points[vertices]+=np.array(errors[zone]['translation_bias_u'])
            parts[zone]={'face_start':start,'faces':count}
        filename=f'map-{e}.bin';(output/filename).write_bytes(blob)
        files[filename]={'bytes':len(blob),'sha256':sha(output/filename),'source':meta[key]['url'],
            'fixed_region_bias_applied':bool(e),'triangle_indices_preserved':True}
        maps.append({'file':filename,'vertices':nv,'faces':nf,'parts':parts})
        map_audit.append({'epoch':e,'vertices':nv,'triangles':nf,'indices_unchanged':bytes(blob[8+nv*24:])==original[8+nv*24:],
            'unassigned_vertices':nv-len(used),'per_region_residual_bias_only':bool(e)})
    data={'kind':'dorm_change_tour','version':1,'dataset_id':'dorm-meshfirst-20261022','labels':['Dorm2','Dorm3'],
        'units':'reference_units','height_u':context['height_u'],'different_floors':auto['protocol']['different_floors'],
        'physical_event_ground_truth':False,'events':events,'records':records,'zones':zones,'maps':maps,'object_meshes':mesh_registry,
        'summary':auto['summary'],'source_thread':'Find Existing Map Alignment','source_version':'automatic-lifecycle-v2',
        'source_hashes':hashes,'policy_sha256':auto['policy_sha256'],
        'alignment':{'name':context['alignment_name'],'object_specific_alignment':False,
            'fixed_region_matrices_preserved':True,'map_uses_object_region_residual_bias':True},
        'scope_note':'구조가 같은 서로 다른 층의 배치 비교입니다. 이동·추가·소실 가능성과 판단 보류를 원본 자동 결과 그대로 표시합니다.'}
    write(output/'tour.json',data)
    for name,digest in hashes.items():
        if sha(source/name)!=digest:raise ValueError('Source changed during import')
    expected={r['id']:r['automatic_decision']['decision'] for r in auto['records']}
    assert {r['id']:r['decision'] for r in records}==expected
    assert all(e['alternatives']==g['event_alternatives'] for e,g in zip(events,auto['event_groups']))
    assert len([r for r in records if r['decision']=='deferred'])==auto['summary']['record_dispositions']['deferred']
    audit={'source_read_only':True,'source_hashes':hashes,'record_decisions_preserved':len(expected),
        'event_groups_preserved':len(events),'deferred_preserved':sum(r['decision']=='deferred' for r in records),
        'maps':map_audit,'copied_assets':len(copied),'files':files,'new_detection_or_alignment_fitting':False}
    write(output/'import-audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['files','source_hashes']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();import_tour(a.source,a.output)
