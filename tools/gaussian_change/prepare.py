"""Bridge frozen mesh decisions to the two independently trained Gaussian maps.

Uses recorded coordinate transforms; it never fits a new registration or changes
mesh decisions. All source-camera transforms are checked before rendering.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree

BASE=Path('outputs/registration-benchmark-20261019/new-meshfirst-20261022')
SOURCE=Path('outputs/gaussian-change-20261025/source')
OUT=Path('outputs/gaussian-change-20261025')
REMOTE='/workspace/gaussian_change_20261025'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def transform(points,m):return np.asarray(points)@np.asarray(m)[:3,:3].T+np.asarray(m)[:3,3]

class FrozenPartition:
    def __init__(self,arrays):
        self.arrays=arrays
        pixels=np.argwhere(arrays['labels']>0)
        self.tree=cKDTree(arrays['origin']+(pixels+.5)*arrays['step'])
        self.values=arrays['labels'][tuple(pixels.T)]
    def assign(self,points):
        a=self.arrays;q=np.asarray(points)[:,[0,2]]@a['axes'].T
        ids=np.floor((q-a['origin'])/a['step']).astype(int)
        valid=np.all((ids>=0)&(ids<np.array(a['labels'].shape)),axis=1)
        result=np.zeros(len(points),int);result[valid]=a['labels'][tuple(ids[valid].T)]
        missing=result==0
        result[missing]=self.values[self.tree.query(q[missing])[1]]
        return result

def epoch_to_common(ctx,epoch,zone):
    m=np.eye(4)
    if epoch:
        m=np.array(ctx['methods']['structure']['region_matrices_after_Geo'])[int(zone[1:])-1]@np.array(ctx['base_matrix'])
        m[:3,3]+=ctx['methods']['structure']['errors']['regions'][zone]['translation_bias_u']
    return m

def camera_frame(position,target,fallback):
    f=np.asarray(target)-position;f[1]=0
    if np.linalg.norm(f)<1e-6:f=np.asarray(fallback).copy();f[1]=0
    f/=np.linalg.norm(f);right=np.cross(f,[0,1,0]);right/=np.linalg.norm(right)
    m=np.eye(4);m[:3,:3]=np.column_stack([right,[0,-1,0],f]);m[:3,3]=position
    assert np.isclose(np.linalg.det(m[:3,:3]),1)
    return m

def build():
    OUT.mkdir(exist_ok=True)
    ctx=read(BASE/'alignment-context.json');review=read(BASE/'automatic-lifecycle-v2/review.json')
    provenance=read(BASE/'input-provenance.json');audit=read(BASE/'all-object-review-v1/face-audit.json')
    partition_path=BASE.parent/'regional-analysis-v1/partition.npz'
    partition={k:v for k,v in np.load(partition_path).items()};classifier=FrozenPartition(partition)
    source_hashes={str(p):sha(p) for p in [BASE/'alignment-context.json',BASE/'automatic-lifecycle-v2/review.json',BASE/'all-object-review-v1/face-audit.json',BASE/'input-provenance.json',partition_path]}
    scenes=[];camera_checks={};native_meshes=[]
    output=read(Path('.tools/gaussian-ssh-session/outbox/integration-identity-20261025-04.json'))['output']
    identities={c:{} for c in ['dorm2','dorm3']}
    for line in output.splitlines():
        if line.startswith(('CHECKPOINT_SHA256 ','VIDEO_SHA256 ')):
            key,c,digest=line.split();identities[c][key.lower()]=digest
    assert identities['dorm2']['video_sha256']!=identities['dorm3']['video_sha256']
    assert identities['dorm2']['checkpoint_sha256']!=identities['dorm3']['checkpoint_sha256']
    for epoch,scene in enumerate(['dorm2','dorm3']):
        mesh=np.load(BASE/f'new-inputs/labeled-mesh-{epoch}.npz');native_meshes.append(mesh)
        poses=read(BASE/f'new-inputs/poses-{epoch}.json')['poses']
        old={r['frame']:np.asarray(r['camera_to_world']) for r in poses}
        original=np.load(SOURCE/'data'/scene/'original_poses_glb.npy')
        mesh_to_glb=np.load(SOURCE/'data'/scene/'mesh_to_glb.npy')
        # This is an existing normalization from input provenance, not a new fit.
        native_to_mesh=np.array(provenance['inputs'][epoch]['native_mesh_to_canonical'])@np.linalg.inv(mesh_to_glb)
        scale=float(np.cbrt(np.linalg.det(native_to_mesh[:3,:3])));R=native_to_mesh[:3,:3]/scale
        assert np.allclose(R.T@R,np.eye(3),atol=1e-8)
        position_errors=[];orientation_errors=[]
        for i,cam in enumerate(original):
            if i not in old:continue
            position_errors.append(np.linalg.norm(transform(cam[:3,3],native_to_mesh)-old[i][:3,3]))
            orientation_errors.append(np.max(np.abs(R@cam[:3,:3]-old[i][:3,:3])))
        assert len(position_errors)==len(original)
        assert max(position_errors)<1e-6 and max(orientation_errors)<1e-6
        camera_checks[scene]=dict(cameras=len(position_errors),max_position_residual_u=float(max(position_errors)),max_rotation_matrix_residual=float(max(orientation_errors)),recorded_transform_verified=True)
        latest=read(SOURCE/f'models/{scene}/splatfacto/linked_recipe/latest.json')
        identity=read(SOURCE/f'models/{scene}/splatfacto/linked_recipe/identity.json')
        scenes.append(dict(epoch=epoch,capture=scene,checkpoint=latest['checkpoint'],gaussians=latest['gaussians'],step=latest['completed_steps'],**identities[scene],dataset_receipt=identity['dataset_receipt'],native_to_mesh=native_to_mesh.tolist(),dataparser=read(SOURCE/f'models/{scene}/splatfacto/linked_recipe/dataparser_transforms.json')))
        for p in [BASE/f'new-inputs/labeled-mesh-{epoch}.npz',BASE/f'new-inputs/poses-{epoch}.json',SOURCE/'data'/scene/'original_poses_glb.npy',SOURCE/'data'/scene/'mesh_to_glb.npy']:
            source_hashes[str(p)]=sha(p)
    boxes=[]
    for group in review['event_groups']:
        for entry in group['entry_ids']:
            a=audit[entry];epoch=a['epoch'];mesh=native_meshes[epoch]
            points=mesh['p'][mesh['faces'][a['original_face_ids']]].reshape(-1,3)
            points=transform(points,epoch_to_common(ctx,epoch,group['zone']))
            bounds=np.stack([points.min(0),points.max(0)])
            boxes.append(dict(id=group['id']+'-'+entry,method='baseline',zone=group['zone'],title=group['title'],source_epoch=epoch,source_entry=entry,original_face_ids=a['original_face_ids'],bounds=bounds.tolist(),center=bounds.mean(0).tolist(),frozen_decision=True))
    p=native_meshes[0]['p'];labels=classifier.assign(p)
    cameras=read(BASE/'new-inputs/poses-0.json')['poses'];positions=np.array([np.array(r['camera_to_world'])[:3,3] for r in cameras]);camera_labels=classifier.assign(positions)
    zones=[]
    for number in range(1,14):
        zone=f'R{number:02d}';points=p[labels==number];center=points.mean(0)
        existing=[b for b in boxes if b['zone']==zone]
        target=np.array(existing[0]['center']) if existing else center
        options=np.flatnonzero(camera_labels==number)
        if not len(options):options=np.arange(len(cameras))
        choice=int(options[np.argmin(np.linalg.norm(positions[options][:,[0,2]]-target[[0,2]],axis=1))])
        row=cameras[choice];old=np.array(row['camera_to_world']);pose=camera_frame(old[:3,3],target,old[:3,2])
        matrices=[epoch_to_common(ctx,e,zone)@np.array(scenes[e]['native_to_mesh']) for e in [0,1]]
        zones.append(dict(id=zone,camera_frame=row['frame'],camera_timestamp_s=row['timestamp_s'],camera_to_common_cv=pose.tolist(),target=target.tolist(),bounds=np.stack([points.min(0),points.max(0)]).tolist(),native_to_common=[m.tolist() for m in matrices]))
    protocol=dict(schema='mesh-baseline-gaussian-simsac-v1',remote_root=REMOTE,model_root='/workspace/v1_splatfacto_comparison_20261025',height_u=ctx['height_u'],scenes=scenes,zones=zones,baseline_boxes=boxes,camera_checks=camera_checks,partition='partition.npz',membership_base_matrix=ctx['membership_base_matrix'],membership_parent_transform=ctx['membership_parent_transform'],base_matrix=ctx['base_matrix'],alignment_context=ctx,source_hashes=source_hashes,
        decisions=dict(original_registration_and_mesh_decisions_unchanged=True,new_registration_fitted=False,temporal_ground_truth=False,different_floors=True,gaussian_input='Two independently trained original-video reconstructions, not two checkpoints of one map',simsac_input='New RGB renders from both Gaussian maps at identical registered common camera poses',region_transform='Frozen region membership and existing per-region Sim3/rigid correction including saved translation bias',render_size=[1024,512],cube_face_size=512,cube_fov_degrees=100,alpha_threshold=.85,change_threshold=.5))
    save(OUT/'protocol.json',protocol);shutil.copy2(partition_path,OUT/'partition.npz')
    save(OUT/'identity-audit.json',dict(distinct_checkpoint_hashes=True,distinct_original_video_hashes=True,scenes=scenes,camera_checks=camera_checks,integration_package='/workspace/v1_splatfacto_comparison_20261025/vendor/dorm1_3dgs_integration'))
    for path,digest in source_hashes.items():assert sha(path)==digest,path
    print('Gaussian protocol ready:',len(zones),'zones,',len(boxes),'frozen mesh candidates; camera bridges verified',camera_checks)

if __name__=='__main__':build()
