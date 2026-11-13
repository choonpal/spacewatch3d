"""CPU-only parent observations and conservative provisional mesh grouping."""
import numpy as np

def parent_pixels(score,labels,variant):
 """Keep raw parent evidence before child-margin filtering; never predict subtype."""
 score=np.asarray(score,np.float32);den=score.sum(0).clip(1e-12);other=np.max(np.concatenate([score[:3],score[5:]],axis=0),axis=0)
 if variant=='hard_union':
  raw=np.isin(labels,[3,4]);p=np.maximum(score[3],score[4]);cls=labels.copy();cls[raw]=3;support=score.max(0)/den
  top=np.partition(score,-2,axis=0)[-2:];margin=(top[-1]-top[-2])/den;support=np.where(raw,p/den,support);margin=np.where(raw,(p-other)/den,margin)
 elif variant=='score_sum':
  groups=np.concatenate([score[:3],(score[3]+score[4])[None],score[5:]],axis=0);group=groups.argmax(0);cls=np.where(group>3,group+1,group).astype('u1');raw=cls==3;top=np.partition(groups,-2,axis=0)[-2:];support=top[-1]/den;margin=(top[-1]-top[-2])/den
 else:raise ValueError(variant)
 # Original precision/thresholds, but the parent competes against non-door classes.
 support=support.astype('f2').astype('f4');margin=margin.astype('f2').astype('f4');eligible=(support>=.35)&(margin>=.1)&(cls!=16)
 return dict(raw=raw,labels=cls,support=support,margin=margin,eligible=eligible,parent_eligible=raw&eligible)

def periodic_components(mask):
 from scipy.ndimage import label
 cc,n=label(mask,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8));parent=np.arange(n+1)
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
  return i
 for a,b in zip(cc[:,0],cc[:,-1]):
  if a and b:parent[find(int(a))]=find(int(b))
 lut=np.array([find(i) for i in range(n+1)]);cc=lut[cc];_,inv=np.unique(cc,return_inverse=True);out=inv.reshape(mask.shape).astype('u4')
 # If an all-positive image has no zero, inverse would make its only component 0.
 if mask.all():out[:]=1
 else:out[~mask]=0
 return out

def dominant_face_component(face,pixel_cc,nfaces,minimum=2,fraction=.6):
 valid=pixel_cc>0;out=np.zeros(nfaces,np.uint32)
 if not valid.any():return out
 codes=face[valid].astype('u8')*(int(pixel_cc.max())+1)+pixel_cc[valid];keys,counts=np.unique(codes,return_counts=True);base=int(pixel_cc.max())+1;fid=(keys//base).astype(int);cid=(keys%base).astype('u4');total=np.bincount(face[valid],minlength=nfaces);order=np.lexsort((cid,-counts,fid));fid=fid[order];cid=cid[order];counts=counts[order];first=np.r_[True,np.diff(fid)!=0];fid=fid[first];cid=cid[first];counts=counts[first];ok=(counts>=minimum)&(counts>=fraction*total[fid]);out[fid[ok]]=cid[ok];return out

def face_geometry(v,f):
 tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1);return tri.mean(1),cross/np.maximum(length[:,None],1e-15),length/2

def min_segment_distance(a,b):
 """Exact boundary segment distance, not centroid/k-nearest-face distance."""
 best=(float('inf'),None,None)
 if not len(a) or not len(b):return best
 for start in range(0,len(a),64):
  aa=a[start:start+64];u=(aa[:,1]-aa[:,0])[:,None,:];v=(b[:,1]-b[:,0])[None,:,:];w=aa[:,0,None,:]-b[None,:,0,:];A=(u*u).sum(2);B=(u*v).sum(2);C=(v*v).sum(2);D=(u*w).sum(2);E=(v*w).sum(2);den=A*C-B*B
  ss=np.divide(B*E-C*D,den,out=np.zeros_like(den),where=den>1e-12*np.maximum(A*C,1e-30));ss=np.clip(ss,0,1);tt=(B*ss+E)/np.maximum(C,1e-30);low=tt<0;high=tt>1;ss=np.where(low,np.clip(-D/np.maximum(A,1e-30),0,1),ss);ss=np.where(high,np.clip((B-D)/np.maximum(A,1e-30),0,1),ss);tt=np.clip(tt,0,1);pa=aa[:,0,None,:]+ss[:,:,None]*u;pb=b[None,:,0,:]+tt[:,:,None]*v;dist=np.linalg.norm(pa-pb,axis=2);k=np.unravel_index(np.argmin(dist),dist.shape)
  if dist[k]<best[0]:best=(float(dist[k]),pa[k],pb[k])
 return best

def signatures(ids,cc,area,fraction=.6):
 out=[]
 for frame in cc:
  vals=frame[ids];valid=vals>0
  if not valid.any() or area[ids][valid].sum()<.25*area[ids].sum():out.append(0);continue
  labels=np.unique(vals[valid]);weights=np.array([area[ids][vals==x].sum() for x in labels]);j=int(weights.argmax());out.append(int(labels[j]) if weights[j]>=fraction*area[ids][valid].sum() else 0)
 return np.array(out,np.uint32)

def evidence(a,b):
 joint=(a>0)&(b>0);return int((joint&(a==b)).sum()),int((joint&(a!=b)).sum())

def proposals(v,f,accepted,cc,positive_frames,gap,max_normal_degrees=30,min_shared_frames=2,nonplanar_policy="hold"):
 if nonplanar_policy not in ["hold","retain_provisional"]:raise ValueError(nonplanar_policy)
 """No ROI/old class/old Instance inputs; IDs are provisional only."""
 center,norm,area=face_geometry(v,f);nodes=np.flatnonzero(accepted);parents=np.arange(len(f));edge_owner={};cosine=float(np.cos(np.deg2rad(max_normal_degrees)));cut_edges=[]
 def find(x):
  while parents[x]!=x:parents[x]=parents[parents[x]];x=int(parents[x])
  return x
 for q in nodes:
  for a,b in [(f[q,0],f[q,1]),(f[q,1],f[q,2]),(f[q,2],f[q,0])]:
   key=tuple(sorted((int(a),int(b))))
   if key in edge_owner:
    old=edge_owner[key];same,diff=evidence(cc[:,q],cc[:,old]);normal_ok=abs(float(norm[q]@norm[old]))>=cosine
    if normal_ok and diff<min_shared_frames:parents[find(int(q))]=find(old)
    else:cut_edges.append(dict(a=int(q),b=old,reason='multiview_separation' if diff>=min_shared_frames else 'normal_boundary',same_frames=same,different_frames=diff))
   else:edge_owner[key]=int(q)
 groups={}
 for q in nodes:groups.setdefault(find(int(q)),[]).append(int(q))
 parts=[np.array(ids,int) for ids in groups.values()];parts.sort(key=lambda x:int(x.min()));initial=len(parts)
 def describe(ids):
  points=v[np.unique(f[ids])];cov=np.einsum('ni,nj,n->ij',norm[ids],norm[ids],area[ids]);eig,vec=np.linalg.eigh(cov);normal=vec[:,-1];counts={}
  for q in ids:
   for a,b in [(f[q,0],f[q,1]),(f[q,1],f[q,2]),(f[q,2],f[q,0])]:
    key=tuple(sorted((int(a),int(b))));counts[key]=counts.get(key,0)+1
  boundary=np.array([[v[a],v[b]] for (a,b),n in counts.items() if n==1 and np.linalg.norm(v[a]-v[b])>1e-12],float).reshape(-1,2,3)
  return dict(ids=ids,lo=points.min(0),hi=points.max(0),normal=normal,planarity=float(eig[-1]/max(eig.sum(),1e-30)),signature=signatures(ids,cc,area),boundary=boundary,area=float(area[ids].sum()))
 desc=[describe(x) for x in parts];p=np.arange(len(parts))
 def root(x):
  while p[x]!=x:p[x]=p[p[x]];x=int(p[x])
  return x
 candidates=[];relations=[]
 for a in range(len(parts)):
  for b in range(a+1,len(parts)):
   da,db=desc[a],desc[b];lower=np.maximum(np.maximum(da['lo']-db['hi'],db['lo']-da['hi']),0)
   if np.linalg.norm(lower)>gap:continue
   distance,pa,pb=min_segment_distance(da['boundary'],db['boundary'])
   if distance>gap:continue
   same,diff=evidence(da['signature'],db['signature']);normal_ok=abs(float(da['normal']@db['normal']))>=cosine;reason='eligible'
   if not normal_ok:reason='normal_mismatch'
   elif diff>=min_shared_frames:reason='distinct_2d_components_in_multiple_frames'
   elif same<min_shared_frames:reason='insufficient_same_component_observations'
   row=dict(a=a,b=b,distance=distance,same_frames=same,different_frames=diff,normal_abs_dot=float(abs(da['normal']@db['normal'])),reason=reason,merged=False);relations.append(row)
   if reason=='eligible':candidates.append((distance,a,b,row))
 for distance,a,b,row in sorted(candidates,key=lambda x:(x[0],x[1],x[2])):
  ra,rb=root(a),root(b)
  if ra==rb:row['reason']='already_connected';continue
  da,db=desc[ra],desc[rb];same,diff=evidence(da['signature'],db['signature'])
  if diff>=min_shared_frames or same<min_shared_frames or abs(float(da['normal']@db['normal']))<cosine:row['reason']='group_evidence_conflict';continue
  merged=describe(np.r_[da['ids'],db['ids']])
  if merged['planarity']<.8:row['reason']='nonplanar_group';continue
  p[rb]=ra;desc[ra]=merged;row['merged']=True
 out=np.zeros(len(f),np.uint32);uncertain=np.zeros(len(f),np.uint8);records=[];nextid=1
 for a in range(len(parts)):
  if root(a)!=a:continue
  d=desc[a];ids=d['ids'];reason=[]
  if d['area']<gap*gap:reason.append('surface_smaller_than_depth_uncertainty_squared')
  if d['planarity']<.8 and (nonplanar_policy=='hold' or np.count_nonzero(d['signature'])<min_shared_frames):reason.append('nonplanar_region')
  if np.max(positive_frames[ids],initial=0)<min_shared_frames:reason.append('insufficient_parent_frames')
  pid=0
  if not reason:pid=nextid;nextid+=1;out[ids]=pid
  else:uncertain[ids]=1
  records.append(dict(proposal_id=pid,status='provisional' if pid else 'uncertain_fragment',identity_status='provisional' if pid else 'uncertain',physical_identity_verified=False,faces=int(len(ids)),area=d['area'],bounds=[d['lo'].tolist(),d['hi'].tolist()],planarity=d['planarity'],frame_component_signature=d['signature'].tolist(),source_face_ids=ids.tolist(),hold_reasons=reason))
 part_index={int(face):j for j,ids in enumerate(parts) for face in ids}
 for record in records:
  record['shape_uncertainty']=record['planarity']<.8
  record['identity_evidence']='multiview_parent_geometry_only'
  record['not_verified_reasons']=['no_full_object_GT','no_subclass_confirmation']+(['multi_surface_nonplanar_geometry'] if record['shape_uncertainty'] else [])
  record['initial_components']=sorted({part_index[q] for q in record['source_face_ids']})
  record['merge_link_indices']=[j for j,x in enumerate(relations) if x['merged'] and x['a'] in record['initial_components'] and x['b'] in record['initial_components']]
 return dict(ids=out,uncertain=uncertain,records=records,initial_component_faces=[x.tolist() for x in parts],relations=relations,cut_edges=cut_edges,initial_components=initial,merged_component_links=sum(x['merged'] for x in relations),settings=dict(nonplanar_policy=nonplanar_policy,max_gap_mesh_units=gap,max_normal_degrees=max_normal_degrees,min_shared_component_frames=min_shared_frames,min_area=gap*gap,min_planarity=.8,min_observed_component_area_fraction=.25,thresholds_are_engineering_guards_not_GT_calibrated=True))
