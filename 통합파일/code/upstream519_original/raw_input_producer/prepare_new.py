import sys,json,hashlib,subprocess,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
BASE=Path('/workspace/dorm4_dorm5_v5_20261025')
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def extract(cid):
 import cv2
 src=BASE/'source'/f'{cid}.MP4';out=BASE/'inputs'/cid;frames=out/'frames';frames.mkdir(parents=True,exist_ok=True)
 cap=cv2.VideoCapture(str(src));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4));rows=[]
 assert fps>0 and w==2*h
 for i in range(n):
  ok=cap.grab()
  if not ok:break
  if i%15:continue
  ok,im=cap.retrieve();assert ok
  p=frames/f'frame_{len(rows):05d}.jpg';im=cv2.resize(im,(1036,518),interpolation=cv2.INTER_AREA);cv2.imwrite(str(p),im,[cv2.IMWRITE_JPEG_QUALITY,95])
  rows.append(dict(file='frames/'+p.name,source_frame_index=i,timestamp_s=i/fps,sha256=sha(p)))
 cap.release();assert len(rows)>10
 manifest=dict(source_video=str(src),source_sha256=sha(src),source_width=w,source_height=h,source_frame_count=n,source_fps=fps,sample_step=15,sample_fps_approx=fps/15,projection='equirectangular',height=518,width=1036,frames=rows)
 save(out/'manifest.json',manifest)
 print(cid,'extracted',len(rows),flush=True)
def masks(cid,predictions):
 from scipy.ndimage import maximum_filter
 out=BASE/'inputs'/cid;m=read(out/'manifest.json');maskdir=out/'base_masks';maskdir.mkdir(exist_ok=True);stats=[];paths={};frames={};chosen=set(np.linspace(0,len(m['frames'])-1,16,dtype=int));sheet=Image.new('RGB',(4*518,4*283),'#18202a');draw=ImageDraw.Draw(sheet);k=0
 static=np.zeros((512,1024),np.uint16)
 for j in range(len(m['frames'])):
  with np.load(Path(predictions)/f'{j:05d}.npz') as z:static+=(z['labels']==16).astype(np.uint16)
 static=static>=max(3,int(len(m['frames'])*.10))
 static[:282]=False;static[:,307:717]=False
 static=np.maximum.accumulate(static,axis=0)
 for i,row in enumerate(m['frames']):
  mask=static.copy()
  for neighbor in range(max(0,i-2),min(len(m['frames']),i+3)):
   with np.load(Path(predictions)/f'{neighbor:05d}.npz') as z:
    person=z['labels']==16
    mask=person.copy() if mask is None else mask|person
  mask=np.array(Image.fromarray(mask).resize((1036,518),Image.Resampling.NEAREST));mask=maximum_filter(mask,size=(21,31),mode=('nearest','wrap'));mask[492:]=True
  p=maskdir/(Path(row['file']).stem+'.png');Image.fromarray(mask.astype('u1')*255).save(p)
  paths[str(i)]=record(p);frames[str(i)]=record(out/row['file']);stats.append(dict(frame=i,excluded_fraction=float(mask.mean())))
  if i in chosen:
   rgb=np.array(Image.open(out/row['file']).convert('RGB'));rgb[mask]=(rgb[mask]*.45+np.array([255,65,75])*.55).astype('u1');x=(k%4)*518;y=(k//4)*283;sheet.paste(Image.fromarray(rgb).resize((518,259)),(x,y));draw.text((x+8,y+262),f'{cid} {i} / {i*.5:.1f}s excluded {mask.mean():.1%}',fill='white');k+=1
 assert max(s['excluded_fraction'] for s in stats)<.45
 sheet.save(out/'mask_contact.jpg',quality=90);save(out/'mask_report.json',dict(policy='OOOPS person argmax, temporal union +/-2 sampled frames; capture-wide person envelope seen in >=10 percent of frames, limited to left/right 30 percent and lower 45 percent; 21x31 periodic dilation; bottom 5 percent camera/body exclusion',runtime_visual_review=False,frames=stats))
 save(out/'assets.json',dict(schema_version=1,source=record(BASE/'source'/f'{cid}.MP4'),frame_manifest=record(out/'manifest.json'),frames=frames,base_masks=paths,base_mask_provenance='New dorm4/dorm5 OOOPS person masks and explicit nadir exclusion; no masks borrowed from other captures'))
 cfg=read(BASE/'code/v5/vendor/panovggt_pipeline_v2/configs/dorm2.json');cfg['name']=cid+'_pointcloud_v2_for_pipeline_v5';cfg['input'].update(bundle=str(out/'assets.json'),source_sha256=m['source_sha256'],expected_frames=len(m['frames']));cfg['mask'].update(periodic_dilation_kernel=[1,1],polygons={},provenance='New OOOPS person mask, temporal +/-2 frame union and periodic 21x31 dilation already applied, fixed bottom 5% exclusion. See inputs mask_report.json; not manually reviewed per frame.')
 save(BASE/'configs'/f'{cid}_reconstruction.json',cfg)
def coordinates(cid,source,predictions):
 import open3d as o3d
 sys.path.insert(0,str(BASE/'code/v2/vendor/segmentation'))
 from geometry import source_frames
 from core import resize_labels_to_source
 poses=np.array([r['camera_to_world'] for r in read(Path(source)/'result/camera_poses.json')]);up=np.median(-poses[:,:3,1],axis=0);up/=np.linalg.norm(up);selected=np.linspace(0,len(poses)-1,min(28,len(poses)),dtype=int);samples={1:[],2:[]}
 for i,xyz,mask,_ in source_frames(dict(source_run=source),selected=selected):
  with np.load(Path(predictions)/f'{i:05d}.npz') as z:lab,support=resize_labels_to_source(z['labels'],z['support_fraction'],*mask.shape)
  for cls in samples:
   p=xyz[(lab[mask]==cls)&(support[mask]>=.35)];samples[cls].append(p[::max(1,len(p)//6000)])
 floor=np.concatenate(samples[1]);ceil=np.concatenate(samples[2]);assert len(floor)>2000 and len(ceil)>2000,'Insufficient floor/ceiling support for assumed metric scaling'
 pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(floor));o3d.utility.random.seed(20261025);plane,inliers=pc.segment_plane(distance_threshold=.07,ransac_n=3,num_iterations=1500);normal=np.asarray(plane[:3]);normal*=1 if normal@up>0 else -1
 assert normal@up>.85,'Floor normal disagrees with camera up'
 assert len(inliers)/len(floor)>.25,'Floor plane insufficient support'
 floor_z=float(np.median(floor[inliers]@normal));ceil_z=float(np.median(ceil@normal));gap=ceil_z-floor_z;assert .5<gap<10,gap;scale=2.7/gap
 x=np.array([1.,0,0]);x-=normal*(x@normal)
 if np.linalg.norm(x)<.1:x=np.array([0.,0,1]);x-=normal*(x@normal)
 x/=np.linalg.norm(x);y=np.cross(normal,x);r=np.stack([x,y,normal]);origin=poses[0,:3,3];t=-r@origin;t[2]=-floor_z
 tf=np.eye(4);tf[:3,:3]=r*scale;tf[:3,3]=t*scale
 assert np.allclose(r@r.T,np.eye(3),atol=1e-8) and np.linalg.det(r)>.999
 data=dict(source_to_mesh=tf.tolist(),mesh_to_source=np.linalg.inv(tf).tolist(),coordinate_frame=cid+'_mesh',up=[0,0,1],handedness='right',metric_scale_verified=False,assumed_ceiling_height_m=2.7,assumed_meters_per_source_unit=scale,scale_method='2.7 m assumed ceiling height / observed OOOPS ceiling-to-floor gap; shared assumption for both captures',floor_plane_inlier_fraction=len(inliers)/len(floor),floor_points=len(floor),ceiling_points=len(ceil),sample_frames=selected.tolist(),floor_ceiling_source_gap=gap,geometry_uses_semantic_labels=False,coordinate_estimation_uses_semantic_labels=True,source_run=source)
 save(BASE/'coordinates'/f'{cid}.json',data);print(json.dumps(data),flush=True)
if __name__=='__main__':
 mode,cid,*args=sys.argv[1:];globals()[mode](cid,*args)
