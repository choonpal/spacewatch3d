"""Publish an alignment-only overlay, preserving old change-view artifacts."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
from dorm_mesh_geometry import transform,erp_project
from run_dorm_mesh_comparison import panorama_boxes


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def statistics(before,after,nb,na,matrix,height):
    rotation=matrix[:3,:3]/np.cbrt(np.linalg.det(matrix[:3,:3]));moved=transform(after,matrix)
    d,j=cKDTree(before).query(moved);dr,jr=cKDTree(moved).query(before)
    compatible=np.abs(np.sum((na@rotation.T)*nb[j],axis=1))>.85
    rc=np.abs(np.sum((na[jr]@rotation.T)*nb,axis=1))>.85
    distances=np.r_[d,dr];normal=np.r_[np.abs(np.sum((moved-before[j])*nb[j],axis=1)),np.abs(np.sum((before-moved[jr])*(na[jr]@rotation.T),axis=1))]
    shared=np.r_[(d<.2*height)&compatible,(dr<.2*height)&rc]
    return {'overlap':float(((d<.05*height)&compatible).mean()/2+((dr<.05*height)&rc).mean()/2),
        'distance_median':float(np.median(distances)),'distance_p95':float(np.quantile(distances,.95)),
        'shared_normal_residual_median':float(np.median(normal[shared])) if shared.any() else None,
        'shared_normal_residual_p95':float(np.quantile(normal[shared],.95)) if shared.any() else None,
        'samples':[len(before),len(after)],'corresponding_fraction':float(shared.mean())}


def run(root,alignment_folder):
    output=alignment_folder/'published'
    if output.exists():raise ValueError('Use a new publication directory')
    previous=root/'run-v6/published';shutil.copytree(previous,output)
    legacy=json.loads((previous/'comparison.json').read_text());alignment=json.loads((alignment_folder/'alignment.json').read_text())
    matrix=np.array(alignment['after_to_before']);old=np.array(legacy['registration']['after_to_before'])
    height=alignment['reference_height'];cloud=np.load(alignment_folder/'structure.npz');original=[];meshes=[]
    origin_old=np.array(legacy['display_origin']);rotation_old=old[:3,:3]/np.cbrt(np.linalg.det(old[:3,:3]))
    for e in [0,1]:
        blob=bytearray((previous/f'{e}-comparison.bin').read_bytes());count=len(blob)//60
        floats=np.ndarray((count,15),dtype='<f4',buffer=blob);raw=floats[:,:3].astype(float)+origin_old
        raw=np.c_[raw[:,0],raw[:,2],-raw[:,1]];normal=floats[:,3:6].astype(float);normal=np.c_[normal[:,0],normal[:,2],-normal[:,1]]
        if e:raw=transform(raw,np.linalg.inv(old));normal=normal@rotation_old
        floats[:,:3]=raw;floats[:,3:6]=normal
        structure=cloud['before' if e==0 else 'after'];kind=cloud['kb' if e==0 else 'ka']
        d,near=cKDTree(structure).query(raw,workers=4);flags=np.where(d<.08*height,kind[near]+1,0).astype(np.uint8)
        labels=np.ndarray((count,),np.uint8,buffer=blob,offset=27,strides=(60,));labels[:]=flags
        (output/f'{e}-alignment.bin').write_bytes(blob);original.append(raw);meshes.append({'file':f'{e}-alignment.bin','count':count})
    def bounds(t):
        a=original[0];b=transform(original[1],t);points=np.vstack([a,b]);p=np.c_[points[:,0],-points[:,2],points[:,1]]
        return {'min':np.quantile(p,.003,axis=0).tolist(),'max':np.quantile(p,.997,axis=0).tolist()}
    cases=[]
    for key,label,t in [('raw','좌표 보정 전',np.eye(4)),('previous','이전 임시 정합',old),('aligned','다시 맞춘 구조 정합',matrix)]:
        groups={}
        for name,k in [('all',None),('floor',0),('wall',1)]:
            a=np.ones(len(cloud['before']),bool) if k is None else cloud['kb']==k;b=np.ones(len(cloud['after']),bool) if k is None else cloud['ka']==k
            groups[name]=statistics(cloud['before'][a],cloud['after'][b],cloud['nb'][a],cloud['na'][b],t,height)
        cases.append({'id':key,'label':label,'matrix':t.tolist(),'scale':float(np.cbrt(np.linalg.det(t[:3,:3]))),'statistics':groups,'bounds':bounds(t)})
    display_bounds=cases[-1]['bounds'];origin=(np.array(display_bounds['min'])+display_bounds['max'])/2
    inputs=root/'alignment-v7/input-copy';poses=[json.loads((inputs/str(e)/'cameras/poses.json').read_text())['poses'] for e in [0,1]]
    zones=[{'id':'all','title':'전체 배치','bounds':display_bounds,'photos':[]}]
    for e in [0,1]:
        file=f'frame-{e}-000012.jpg';shutil.copy2(inputs/str(e)/'photos/frame_00012.jpg',output/file)
        zones[0]['photos'].append({'file':file,'caption':f'Dorm{e+2} 원본 · 6초 · 안내판 '+('5F' if e==0 else '3F'),'boxes':[]})
    points=cloud['before'];floor=points[cloud['kb']==0]
    for i,(q,title) in enumerate([(.85,'입구 쪽 바닥·벽'),(.50,'복도 중앙 바닥·벽'),(.15,'안쪽 바닥·벽')]):
        z=float(np.quantile(points[:,2],q));local=floor[np.abs(floor[:,2]-z)<height*.6];center=np.median(local,axis=0)
        if not np.all(np.isfinite(center)):continue
        low=center-np.array([height*.65,height*.2,height*.7]);high=center+np.array([height*.65,height*.85,height*.7])
        box={'min':[float(low[0]),float(-high[2]),float(low[1])],'max':[float(high[0]),float(-low[2]),float(high[1])]}
        zone={'id':f'zone-{i}','title':title,'bounds':box,'photos':[]}
        corners=np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
        for e in [0,1]:
            local_center=center if e==0 else transform(center[None],np.linalg.inv(matrix))[0]
            camera_positions=np.array([p['camera_to_world'] for p in poses[e]])[:,:3,3];distance=np.linalg.norm(camera_positions-local_center,axis=1)
            index=int(np.argmin(np.abs(distance-height*1.1)))
            pose=np.array(poses[e][index]['camera_to_world']);local_corners=corners if e==0 else transform(corners,np.linalg.inv(matrix));pixels,_=erp_project(local_corners,pose,1036,518)
            file=f'frame-{e}-{index:06d}.jpg';shutil.copy2(inputs/str(e)/f'photos/frame_{index:05d}.jpg',output/file)
            zone['photos'].append({'file':file,'caption':f'Dorm{e+2} 원본 · 약 {index*.5:.1f}초 · 추정 카메라로 선택 구역 투영','boxes':panorama_boxes(pixels,1036,518)})
        zones.append(zone)
    audit=json.loads((root/'alignment-v7/source-link-audit.json').read_text())
    public={'kind':'dorm_structural_alignment_view','version':1,'units':'reference_units','labels_used':False,'physical_scale_known':False,
        'physical_identity_verified':False,'registration_status':'기하 배치 보정 · 동일 장소 확인 미완료',
        'identity_note':'원본 연결 기록은 일치하지만, 원본 영상 6초 장면의 안내판은 Dorm2 5F / Dorm3 3F입니다. 같은 공간이라는 설명과 영상 표기가 달라 동일한 물리적 바닥·벽인지는 아직 확인되지 않았습니다. 이 화면의 수치는 기하 형상의 맞음 정도이며, 실제 전후 변화의 증거로 사용하지 않습니다.',
        'method_note':'전체 복도 평면 배치로 회전·축척·이동 후보를 찾은 뒤, 국소 평면의 기하 법선과 높이 방향으로 이어지는 벽 표본을 사용해 Sim(3)를 보정했습니다. 정합 목적함수가 악화되는 보정은 거절합니다. 객체·클래스·변화 라벨, 수동 전후 대응은 사용하지 않았으며 원본 메시를 휘거나 평탄화하지 않았습니다.',
        'metric_note':'u는 Dorm2 복원 좌표 단위입니다. 겹침률은 같은 방식으로 고른 바닥·벽 표본의 양방향 최근접 3D 거리가 기준 구조 높이의 5% 이내이고 법선 내적 절댓값이 0.85를 넘는 비율입니다. 중앙값과 P95는 관측 범위가 다른 부분도 포함합니다. 이 수치는 동일 장소 증명이나 별도 정답에 대한 정확도가 아닙니다. 표본 간 거리에는 표본 간격의 영향도 있습니다.',
        'tolerance':height*.05,'reference_height':height,'identity':np.eye(4).tolist(),'origin':origin.tolist(),
        'upper_cut':height*.85,'mesh_hashes':legacy['source_hashes'],'meshes':meshes,'cases':cases,'zones':zones,'source_audit':audit}
    save(output/'alignment-view.json',public)
    print(json.dumps([{'case':c['id'],'scale':c['scale'],'statistics':c['statistics']} for c in cases],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('alignment_folder',type=Path);a=p.parse_args();run(a.root,a.alignment_folder)
