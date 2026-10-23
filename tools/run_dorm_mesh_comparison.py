"""Compare the actual Dorm meshes without reading class/instance annotations."""
import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from dorm_mesh_geometry import sample_mesh,extract_structure,align_structure,transform,erp_project

ROOTS=[Path('/workspace/dorm2_2d3d_v1_object_mesh_20261017'),Path('/workspace/dorm3_mesh_generator_v1_20261017')]
SOURCES=[Path('/workspace/dorm2_pointcloud_v1_20261017'),Path('/workspace/dorm3_repro_validation_20261016')]


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def matrix_public(plane):return {k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in plane.items()}


def alignment(output):
    import open3d as o3d
    if output.exists() and any(output.iterdir()):raise ValueError('Alignment requires a new empty output directory')
    output.mkdir(parents=True)
    save(output/'protocol.json',{'labels_used':False,'inputs':'mesh XYZ, triangles, RGB for display only; geometric surface normals',
        'reference':'broad geometric floor and tall walls; no semantic fields',
        'units':'Dorm2 reconstruction units; physical metric scale unknown','seed':20261019,
        'detector':{'sample_spacing_height_fraction':.025,'radii_height_fraction':[.035,.07,.12],
            'minimum_difference_height_fraction':.025,'minimum_connected_cores':12,'minimum_area_height_squared_fraction':.012,
            'noise_quantile':.95,'photo_projection':'equirectangular'}})
    samples=[];normals=[];planes=[];heights=[];hashes=[]
    for e,root in enumerate(ROOTS):
        path=root/'meshes/scene.ply';hashes.append(sha(path));mesh=o3d.io.read_triangle_mesh(str(path))
        p,n=sample_mesh(mesh,spacing=.065,seed=20261019+e)
        pl,h=extract_structure(p,n,20261019+e)
        samples.append(p);normals.append(n);planes.append(pl);heights.append(h)
        print('STRUCTURE',e,len(p),'height',h,[(x['kind'],x['count'],np.round(x['normal'],3).tolist(),np.round(x['center'],2).tolist()) for x in pl],flush=True)
    matrix,quality=align_structure(*samples,*normals,*planes,*heights)
    print('ALIGNMENT',quality,'matrix',matrix.tolist(),flush=True)
    if quality['structural_overlap']<.25:raise ValueError('Insufficient common structural alignment; refusing change publication')
    save(output/'alignment.json',{'after_to_before':matrix.tolist(),'quality':quality,'source_hashes':hashes,
        'physical_scale_known':False,'labels_used':False,'source_roots':list(map(str,ROOTS)),
        'planes':[[matrix_public(p) for p in group] for group in planes]})
    np.savez_compressed(output/'alignment-samples.npz',before=samples[0],after=samples[1],normal_before=normals[0],normal_after=normals[1])


def feature(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return {'count':len(values),'median':float(np.median(values)) if len(values) else None,
        'p95':float(np.quantile(values,.95)) if len(values) else None}


def boxes(points):return {'min':points.min(0).tolist(),'max':points.max(0).tolist()}


def panorama_boxes(pixels,width,height):
    u=np.sort(pixels[:,0]%width);gaps=np.diff(np.r_[u,u[0]+width]);start=u[(np.argmax(gaps)+1)%len(u)]
    unwrapped=(pixels[:,0]-start)%width+start
    x0,x1=np.quantile(unwrapped,[.02,.98]);y0,y1=np.quantile(pixels[:,1],[.02,.98])
    x0=max(start,x0-5);x1=min(start+width,x1+5);y0=max(0,y0-5);y1=min(height-1,y1+5)
    if x1-x0>width*.8:return []
    if x1<width:return [[float(x0),float(y0),float(x1),float(y1)]]
    return [[float(x0),float(y0),float(width-1),float(y1)],[0.,float(y0),float(x1-width),float(y1)]]


def load_frames(matrix):
    scale=float(np.cbrt(np.linalg.det(matrix[:3,:3])));rotation=matrix[:3,:3]/scale
    frames=[[],[]]
    for e,root in enumerate(ROOTS):
        acquisition=load(SOURCES[e]/'input/manifest.json');pose_data=load(root/'cameras/poses.json')
        for record in pose_data['poses']:
            ordinal=record['frame'];pose=np.array(record['camera_to_world'],float)
            if e:
                pose[:3,3]=transform(pose[None,:3,3],matrix)[0];pose[:3,:3]=rotation@pose[:3,:3]
            source=acquisition['frames'][ordinal];filename=f'frame-{e}-{ordinal:06d}.jpg'
            frames[e].append({'id':filename[:-4],'epoch':e,'url':'/residual-comparison/dorm2-dorm3/'+filename,
                'source_frame_index':source['source_frame_index'],'sample_index':ordinal,'time_s':source['timestamp_s'],
                'camera_to_world':pose.tolist(),'width':acquisition['width'],'height':acquisition['height'],'projection':'equirectangular',
                'source_sha256':source['sha256'],'source_path':str(SOURCES[e]/'input'/source['file'])})
    return frames


def visibility_gate(points,owners,status,scenes,frames,height,noise):
    import open3d as o3d
    ids=np.flatnonzero(status==1);occupied=np.zeros((len(ids),2),int);free=occupied.copy()
    for e in (0,1):
        # Two-second separated source frames; neighbouring pixels do not count as views.
        cameras=np.array([frame['camera_to_world'] for frame in frames[e][::4]])[:,:3,3]
        _,near=cKDTree(cameras).query(points[ids],k=min(8,len(cameras)))
        for camera_id in np.unique(near):
            selected=np.flatnonzero((near==camera_id).any(1));p=points[ids[selected]];center=cameras[camera_id]
            distance=np.linalg.norm(p-center,axis=1);direction=(p-center)/np.maximum(distance[:,None],1e-8)
            hit=scenes[e].cast_rays(o3d.core.Tensor(np.c_[np.tile(center,(len(p),1)),direction].astype('f4')))['t_hit'].numpy()
            seen=np.isfinite(hit)&(distance>height*.12)&(distance<height*8)
            tolerance=max(noise*1.5,height*.025)
            occupied[selected,e]+=seen&(np.abs(hit-distance)<=tolerance)
            free[selected,e]+=seen&(hit>distance+tolerance)
    own=occupied[np.arange(len(ids)),owners[ids]];opposite=free[np.arange(len(ids)),1-owners[ids]]
    supported=(own>=2)&(opposite>=2);status[ids[~supported]]=2
    return {'tested_cores':len(ids),'supported_cores':int(supported.sum()),'unobserved_or_occluded_cores':int((~supported).sum()),
        'rule':'At least two own occupied and opposite farther reconstructed surface rays from source frames at least 2s apart. No-hit is unknown.',
        'independent_depth_validation':False}


def publish(output):
    import open3d as o3d
    from PIL import Image,ImageDraw
    metadata=load(output/'alignment.json');matrix=np.array(metadata['after_to_before']);h=metadata['quality']['reference_height_units']
    published=output/'published'
    if published.exists():raise ValueError('Use an unpublished alignment output')
    published.mkdir()
    scale=float(np.cbrt(np.linalg.det(matrix[:3,:3])));rotation=matrix[:3,:3]/scale
    meshes=[];scenes=[];samples=[];normals=[]
    for e,root in enumerate(ROOTS):
        if sha(root/'meshes/scene.ply')!=metadata['source_hashes'][e]:raise ValueError('Input mesh changed')
        mesh=o3d.io.read_triangle_mesh(str(root/'meshes/scene.ply'))
        if e:mesh.transform(matrix)
        p,n=sample_mesh(mesh,count=500000,spacing=h*.025,seed=20261021+e)
        mesh.compute_vertex_normals();samples.append(p);normals.append(n);meshes.append(mesh)
        scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));scenes.append(scene)
    # Structural correspondences are used for empirical normal-direction noise.
    old=np.load(output/'alignment-samples.npz')
    structural=[];structural_n=[]
    for e in (0,1):
        p=old['before' if e==0 else 'after'];groups=metadata['planes'][e]
        ids=np.concatenate([np.array(group['ids'],int) for group in groups])
        n=np.concatenate([np.tile(group['normal'],(len(group['ids']),1)) for group in groups])
        p=p[ids]
        if e:p=transform(p,matrix);n=n@rotation.T
        structural.append(p);structural_n.append(n)
    distance,near=cKDTree(structural[0]).query(structural[1],workers=4)
    compatible=np.abs(np.einsum('ij,ij->i',structural_n[1],structural_n[0][near]))>.94
    stable=(distance<h*.065)&compatible
    reference_a=structural[0][near[stable]];reference_b=structural[1][stable];reference_n=structural_n[0][near[stable]]
    residual=np.abs(np.einsum('ij,ij->i',reference_n,reference_b-reference_a))
    cells=np.floor((reference_a+reference_b)/2/(h*.12)).astype(np.int64)
    _,membership=np.unique(cells,axis=0,return_inverse=True)
    cell_error=np.array([np.quantile(residual[membership==i],.95) for i in range(membership.max()+1)])
    noise=max(float(np.quantile(cell_error,.95)),h*.008)
    save(output/'stable-reference.json',{'pairs':int(stable.sum()),'spatial_cells':len(cell_error),'normal_error_p95_units':noise,
        'distribution':feature(residual),'labels_used':False,'method':'P95 of spatial-cell P95 normal residuals on common geometric floor/walls'})
    print('NOISE',noise,'height',h,'stable pairs',stable.sum(),flush=True)
    clearances=[scenes[1-e].compute_distance(o3d.core.Tensor(p.astype('f4'))).numpy() for e,p in enumerate(samples)]
    core_points=[];core_normals=[];owners=[]
    for e,p in enumerate(samples):
        ids=np.unique(np.floor(p/(h*.04)).astype(np.int64),axis=0,return_index=True)[1]
        core_points.append(p[ids]);core_normals.append(normals[e][ids]);owners.extend([e]*len(ids))
    p=np.vstack(core_points);n=np.vstack(core_normals);owner=np.array(owners,int);radii=h*np.array([.035,.07,.12])
    d=np.full((len(p),3),np.nan);rough=np.full(len(p),np.nan);counts=np.zeros((len(p),3),int)
    for e in (0,1):
        selected=np.flatnonzero(owner==e);tree=cKDTree(samples[e])
        for begin in range(0,len(selected),256):
            subset=selected[begin:begin+256]
            for i,neighbors in zip(subset,tree.query_ball_point(p[subset],radii[-1],workers=4)):
                ids=np.array(neighbors,int);ids=ids[np.abs(normals[e][ids]@n[i])>.65]
                delta=samples[e][ids]-p[i];length=np.linalg.norm(delta,axis=1)
                fine=delta[length<radii[1]]
                if len(fine)<6:continue
                _,_,vt=np.linalg.svd(fine-fine.mean(0),full_matrices=False)
                value=(fine-fine.mean(0))@vt[-1];rough[i]=1.4826*np.median(np.abs(value-np.median(value)))
                for j,radius in enumerate(radii):
                    values=clearances[e][ids[length<radius]];counts[i,j]=len(values)
                    if len(values)>=6:d[i,j]=np.median(values)
        print('DISTANCE',e,len(selected),flush=True)
    threshold=np.maximum(noise,2.77*rough)[:,None]*np.ones((1,3))
    valid=np.isfinite(d)&(counts>=6);over=valid&(d>threshold)&(d>h*.025)
    status=np.full(len(p),2,np.uint8);eligible=valid.sum(1)>=2;status[eligible]=0;status[eligible&(over.sum(1)>=2)]=1
    background=np.zeros(len(p),bool)
    for e in (0,1):
        selected=np.flatnonzero(owner==e)
        a,ia=cKDTree(structural[e]).query(p[selected],workers=4)
        b,ib=cKDTree(structural[1-e]).query(p[selected],workers=4)
        common=(a<h*.035)&(b<max(noise*2,h*.04))
        common &= np.abs(np.einsum('ij,ij->i',n[selected],structural_n[e][ia]))>.94
        common &= np.abs(np.einsum('ij,ij->i',n[selected],structural_n[1-e][ib]))>.94
        background[selected]=common
    status[background]=3
    frames=load_frames(matrix)
    visibility=visibility_gate(p,owner,status,scenes,frames,h,noise)
    print('VISIBILITY',visibility,flush=True)
    ratio=np.nanmedian(np.divide(d,threshold,out=np.full_like(d,np.nan),where=threshold>0),axis=1)
    score=np.clip(100*(.6*np.nan_to_num(ratio)/3+.4*over.sum(1)/3),0,100)
    core_tree=cKDTree(p);areas=[];displays=[];geometry=[]
    display_rotation=np.array([[1,0,0],[0,0,-1],[0,1,0.]])
    all_vertices=np.vstack([np.asarray(mesh.vertices)@display_rotation.T for mesh in meshes]);origin=(all_vertices.min(0)+all_vertices.max(0))/2
    packed_type=np.dtype([('xyz','<f4',(3,)),('normal','<f4',(3,)),('rgb','u1',(3,)),('pad','u1'),('unused','<f4',(8,))])
    for e,mesh in enumerate(meshes):
        reduced=mesh.simplify_vertex_clustering(h*.018)
        if len(reduced.triangles)>220000:reduced=reduced.simplify_quadric_decimation(220000)
        reduced.compute_vertex_normals();v=np.asarray(reduced.vertices);f=np.asarray(reduced.triangles);ids=f.ravel()
        nearest_distance,index=core_tree.query(v,workers=4)
        triangle=v[f];area=np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)/2
        areas.append(np.bincount(index[f].ravel(),weights=np.repeat(area/3,3),minlength=len(p)))
        packed=np.zeros(len(ids),packed_type);packed['xyz']=v[ids]@display_rotation.T-origin;packed['normal']=np.asarray(reduced.vertex_normals)[ids]@display_rotation.T
        packed['rgb']=np.clip(np.asarray(reduced.vertex_colors)[ids]*255,0,255).astype('u1')
        packed.tofile(published/f'{e}-comparison.bin')
        ids_core=index[ids];s=status[ids_core].astype(float);s[nearest_distance[ids]>h*.09]=2
        np.c_[np.nan_to_num(np.nanmedian(d,axis=1))[ids_core],threshold[ids_core,0],s,np.nan_to_num(ratio)[ids_core]].astype('<f4').tofile(published/f'{e}-empirical.bin')
        displays.append({'epoch':e,'url':f'/residual-comparison/dorm2-dorm3/{e}-comparison.bin','count':len(ids),'triangles':len(f),'source_triangles':len(mesh.triangles)})
        print('DISPLAY',e,len(f),flush=True)
    areas=np.array(areas)
    # A cluster is a surface patch, not an inferred instance or movement ID.
    candidates=np.flatnonzero(status==1);groups=[]
    if len(candidates):
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[candidates]))
        labels=np.array(cloud.cluster_dbscan(eps=h*.075,min_points=5))
        for label in np.unique(labels[labels>=0]):
            ids=candidates[labels==label]
            if len(ids)<12 or areas[:,ids].sum()<h*h*.012:continue
            if np.ptp(p[ids],axis=0).max()>h*1.5:
                tiles=np.floor(p[ids][:,[0,2]]/(h*.75)).astype(int);_,tile=np.unique(tiles,axis=0,return_inverse=True)
                chunks=[ids[tile==key] for key in np.unique(tile)]
            else:chunks=[ids]
            groups.extend(chunk for chunk in chunks if len(chunk)>=12 and areas[:,chunk].sum()>=h*h*.012)
    accepted=np.zeros(len(p),bool)
    for group in groups:accepted[group]=True
    status[(status==1)&~accepted]=0
    groups.sort(key=lambda ids:-float(np.median(score[ids])*np.sqrt(areas[:,ids].sum())))
    # Camera orientation keeps unit axes; only camera centres receive Sim(3) scale.
    def link_photos(ids,e):
        subset=p[ids[np.linspace(0,len(ids)-1,min(100,len(ids))).astype(int)]];center=np.median(subset,axis=0)
        ranked=sorted(frames[e],key=lambda f:np.linalg.norm(np.array(f['camera_to_world'])[:3,3]-center))[:35];links=[]
        for frame in ranked:
            pose=np.array(frame['camera_to_world']);pixels,depth=erp_project(subset,pose,frame['width'],frame['height'])
            direction=(subset-pose[:3,3])/np.maximum(depth[:,None],1e-8)
            rays=np.c_[np.tile(pose[:3,3],(len(subset),1)),direction].astype('f4')
            hit=scenes[e].cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
            visible=np.isfinite(hit)&(hit>=depth-max(noise*2,h*.04))&(depth>h*.12)&(depth<h*8)
            if visible.sum()<max(8,len(subset)*.15):continue
            boxes_px=panorama_boxes(pixels[visible],frame['width'],frame['height'])
            value=float(visible.mean()/(1+np.median(depth[visible])/h))
            links.append({'frame_id':frame['id'],'score':value,'bboxes_px':boxes_px,'camera_check':'estimated_erp_surface_visibility',
                'visible_sample_fraction':float(visible.mean())})
        links.sort(key=lambda link:-link['score']);chosen=[]
        frame_lookup={f['id']:f for f in frames[e]}
        for link in links:
            if all(abs(frame_lookup[link['frame_id']]['sample_index']-frame_lookup[prior['frame_id']]['sample_index'])>=8 for prior in chosen):chosen.append(link)
            if len(chosen)>=4:break
        return chosen
    def statistics(ids):
        valid_ids=ids[eligible[ids]&~background[ids]];changed=ids[accepted[ids]]
        result_status='background' if np.all(background[ids]) else 'change_candidate' if len(changed) else 'insufficient_change_evidence' if len(valid_ids) else 'unassessable'
        return {'status':result_status,'core_count':len(ids),'valid_cores':len(valid_ids),'valid_fraction':len(valid_ids)/len(ids),
            'exceeding_cores':len(changed),'largest_connected_exceedance_cores':len(changed),
            'exceeding_area_by_epoch_m2':areas[:,changed].sum(1).tolist(),
            'threshold_by_radius_m':[feature(threshold[ids,j]) for j in range(3)]}
    display_points=p@display_rotation.T;regions=[];used=set()
    for k,ids in enumerate(groups):
        links=[link_photos(ids,e) for e in (0,1)]
        for group in links:used.update(item['frame_id'] for item in group)
        stat=statistics(ids);regions.append({'id':f'candidate-{k+1}','title':f'변화 후보 {k+1:02d}','kind':'local_patch','bounds':boxes(display_points[ids]),
            'statistics':[stat.copy(),stat],'keyframes':links,'paired_originals':bool(all(links)),
            'distance_by_radius_m':[feature(d[ids,j]) for j in range(3)],'score':feature(score[ids]),
            'conditions':{'roughness_m':feature(rough[ids])},'reasons':{'all_direction_surface_difference':len(ids)}})
        print('CANDIDATE',k+1,len(ids),'photos',list(map(len,links)),flush=True)
    for kind,ids in [('background',np.flatnonzero(background)),('overview',np.arange(len(p)))]:
        if not len(ids):continue
        links=[link_photos(ids,e) for e in (0,1)]
        for group in links:used.update(item['frame_id'] for item in group)
        stat=statistics(ids);regions.append({'id':kind,'title':'공통 바닥·벽 기준면' if kind=='background' else '전체 촬영 범위','kind':kind,'bounds':boxes(display_points[ids]),
            'statistics':[stat.copy(),stat],'keyframes':links,'paired_originals':bool(all(links)),
            'distance_by_radius_m':[feature(d[ids,j]) for j in range(3)],'conditions':{'roughness_m':feature(rough[ids])},
            'reasons':{'common_structural_surface_excluded' if kind=='background' else 'all_direction_surface_difference':len(ids)}})
    keyframes=[]
    for group in frames:
        for frame in group:
            if frame['id'] not in used:continue
            source=Path(frame['source_path']);destination=published/(frame['id']+'.jpg')
            if sha(source)!=frame['source_sha256']:raise ValueError('Original photo checksum mismatch')
            shutil.copy2(source,destination);keyframes.append({k:v for k,v in frame.items() if k!='source_path'})
    # Apply connected-patch decisions to display fields after the clustering gate.
    for e,mesh in enumerate(meshes):
        packed=np.fromfile(published/f'{e}-comparison.bin',dtype=packed_type)
        world=(packed['xyz'].astype(float)+origin)@display_rotation
        distance,index=core_tree.query(world,workers=4);fields=np.fromfile(published/f'{e}-empirical.bin',dtype='<f4').reshape(-1,4)
        fields[:,2]=status[index];fields[distance>h*.09,2]=2;fields.tofile(published/f'{e}-empirical.bin')
    summary=statistics(np.arange(len(p)))
    report={'kind':'aligned_mesh_pair_comparison','version':1,'dataset_id':'dorm2-dorm3','scene':'Dorm2 → Dorm3 · 실제 전후 메시',
        'new_registration':True,'same_input_meshes_and_poses':False,'units':'reference_units','physical_scale_known':False,
        'labels_used_for_detection':False,'source_hashes':metadata['source_hashes'],'source_epochs':[2,3],
        'registration':metadata,'display_origin':origin.tolist(),'display_rotation':display_rotation.tolist(),
        'bounds':boxes(all_vertices-origin),'displays':displays,'regions':regions,'keyframes':keyframes,'models':[],
        'config':{'radii_m':radii.tolist(),'noise_budget_m':[noise]*3,'minimum_displacement_units':h*.025},
        'variants':[{'summary':summary},{'summary':summary}],
        'dorm':{'candidate_patches':len(groups),'candidate_cores':int(accepted.sum()),'paired_patches':sum(bool(r['paired_originals']) for r in regions if r['kind']=='local_patch'),
            'sample_counts':list(map(len,samples)),'reference_height_units':h,'stable_pairs':int(stable.sum()),'stable_spatial_cells':len(cell_error),
            'normal_error_p95_units':noise,'labels_used':False,'visibility':visibility,'registration_reference':'Geometrically extracted broad floor and tall walls only',
            'photo_projection':'PanoVGGT equirectangular, OpenCV camera axes; mesh ray visibility is estimated, not independent depth validation'},
        'limitations':['Unknown physical metric scale; all distances are Dorm2 reconstruction units.',
            'Geometrically stable structure is assumed unchanged; large nonrigid reconstruction distortion is not corrected.',
            'No object IDs or semantic labels are read. Input meshes were previously generated using semantic partitions.',
            'Candidate surface patches are not object counts or verified addition/removal/movement labels.',
            'Camera poses are estimated; visibility is tested against reconstructed meshes, not independent depth observations.',
            'Display meshes are simplified; distances are calculated against full original triangle surfaces.']}
    save(published/'comparison.json',report);save(output/'comparison.json',report)
    np.savez_compressed(output/'detection.npz',xyz=p,normals=n,owner=owner,distance=d,threshold=threshold,status=status,score=score,background=background,area=areas)
    # Lightweight floor-plan evidence for inspection of alignment before opening WebGL.
    image=Image.new('RGB',(1100,550),(14,22,32));draw=ImageDraw.Draw(image)
    low,high=np.quantile(np.vstack(samples)[:,[0,2]],[.01,.99],axis=0)
    for e,points in enumerate(samples):
        xy=(points[:,[0,2]]-low)/(high-low)*np.array([1000,450])+[50,50]
        for x,y in xy[::max(1,len(xy)//20000)]:draw.point((int(x),int(y)),fill=(255,160,90) if e==0 else (90,200,255))
    image.save(output/'alignment-top.png')
    delivery=output/'delivery';delivery.mkdir()
    with zipfile.ZipFile(delivery/'viewer.zip','w',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for path in published.iterdir():archive.write(path,'published/'+path.name)
        for filename in ['alignment.json','stable-reference.json','protocol.json','alignment-top.png']:archive.write(output/filename,filename)
    print('COMPLETE',json.dumps(report['dorm']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['alignment','publish'],default='alignment')
    parser.add_argument('--alignment-from',type=Path)
    args=parser.parse_args()
    if args.alignment_from:
        if args.output.exists():raise ValueError('Use a new output directory for a publication revision')
        args.output.mkdir(parents=True)
        for filename in ['alignment.json','alignment-samples.npz','protocol.json']:shutil.copy2(args.alignment_from/filename,args.output/filename)
        protocol=load(args.output/'protocol.json');protocol['visibility_gate']='2 own occupied / 2 opposite free reconstructed-mesh rays, source frames >=2s apart; no-hit unknown'
        protocol['alignment_reused_from']=str(args.alignment_from);save(args.output/'protocol.json',protocol)
    (alignment if args.stage=='alignment' else publish)(args.output)
