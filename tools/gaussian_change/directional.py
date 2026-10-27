"""Same-pose perspective tiles; no full-panorama resize is used for inference."""
import hashlib
import numpy as np


def tile_boxes(shape, size=1040, overlap=32):
    height,width=shape[:2]
    if size<32 or not 0<=overlap<size:
        raise ValueError('Invalid perspective tile size/overlap')
    def positions(length):
        if length<=size:return [0]
        return sorted(set(list(range(0,length-size+1,size-overlap))+[length-size]))
    return [(x,y,min(x+size,width),min(y+size,height)) for y in positions(height) for x in positions(width)]


def predict_face(model, before, after, tile_size=1040, overlap=32, self_control=False):
    if before.shape!=after.shape:raise ValueError('Paired perspective images must have the same shape')
    sums=[np.zeros(before.shape[:2],'f4') for _ in range(2)]
    count=np.zeros(before.shape[:2],'f4');audit=[];control=None
    for rect in tile_boxes(before.shape,tile_size,overlap):
        x0,y0,x1,y1=rect;images=[np.ascontiguousarray(p[y0:y1,x0:x1]) for p in [before,after]]
        for epoch in range(2):sums[epoch][y0:y1,x0:x1]+=model.predict(images[epoch],images[1-epoch])
        count[y0:y1,x0:x1]+=1
        row={'rect_xyxy':list(rect),'rgb_array_sha256':[hashlib.sha256(p.tobytes()).hexdigest() for p in images],
             'input_shape':list(images[0].shape),'bidirectional':True}
        audit.append(row)
        if self_control and control is None:
            cp=model.predict(images[0],images[0]);control={'max_probability':float(cp.max()),'probabilities':cp,'rgb_array_sha256':row['rgb_array_sha256'][0],'rect_xyxy':list(rect),'input_kind':'native perspective tile'}
    if not np.all(count>0):raise ValueError('Perspective tiling left uncovered pixels')
    return [p/count for p in sums],audit,control


def bases():
    result=[]
    for forward,up in [([0,0,1],[0,-1,0]),([1,0,0],[0,-1,0]),([0,0,-1],[0,-1,0]),([-1,0,0],[0,-1,0]),([0,-1,0],[0,0,-1]),([0,1,0],[0,0,1])]:
        down=-np.asarray(up);forward=np.asarray(forward)
        result.append(np.column_stack([np.cross(down,forward),down,forward]).astype('f4'))
    return result


def panorama_rays(width,height):
    lo=((np.arange(width,dtype='f4')+.5)/width-.5)*2*np.pi
    la=((np.arange(height,dtype='f4')+.5)/height-.5)*np.pi
    cos=np.cos(la)[:,None]
    return np.stack([np.sin(lo)[None,:]*cos,np.broadcast_to(np.sin(la)[:,None],(height,width)),np.cos(lo)[None,:]*cos],-1)


def stitch(faces,width,height,size):
    import cv2
    rays=panorama_rays(width,height);rotations=bases()
    # Axis-aligned cube faces make this identical to maximum forward cosine.
    forward=np.stack([rays[...,2],rays[...,0],-rays[...,2],-rays[...,0],-rays[...,1],rays[...,1]],-1)
    dominant=np.argmax(forward,axis=-1);del forward
    result={'rgb':np.zeros((height,width,3),'u1'),'depth':np.zeros((height,width),'f4'),
            'alpha':np.zeros((height,width),'f4'),'person':np.zeros((height,width),bool),
            'probability':np.zeros((height,width),'f4'),'rays':rays}
    focal=.5*size/np.tan(np.deg2rad(100)/2)
    for index,basis in enumerate(rotations):
        face=faces[index]
        # Sparse remapping avoids repeated full 4K direction/mapping arrays.
        y,x=np.nonzero(dominant==index);directions=rays[y,x]@basis;z=np.maximum(directions[:,2],1e-8)
        if not len(x):continue
        mx=(focal*directions[:,0]/z+size/2-.5).astype('f4')
        my=(focal*directions[:,1]/z+size/2-.5).astype('f4')
        # Reshape sparse samples so neither OpenCV remap dimension exceeds SHRT_MAX.
        map_width=min(2048,len(x));pad=(-len(x))%map_width
        mx=np.pad(mx,(0,pad),mode='edge').reshape(-1,map_width);my=np.pad(my,(0,pad),mode='edge').reshape(-1,map_width)
        for field in ['rgb','depth','alpha','person','probability']:
            image=face[field].astype('u1') if field=='person' else face[field]
            sample=cv2.remap(image,mx,my,cv2.INTER_NEAREST if field=='person' else cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            sample=sample.reshape(-1,3)[:len(x)] if field=='rgb' else sample.ravel()[:len(x)]
            if field=='depth':sample=sample/z
            result[field][y,x]=sample
    return result


def render_pair(renderers,pose,detector,model,width,height,size,directory,tile_size=1040,overlap=32,self_control=False,on_face=None):
    import cv2
    from .render import person_mask
    from tools.mesh_simsac.core import sha,write
    destination=directory/'perspective';destination.mkdir(parents=True,exist_ok=True)
    faces=[[],[]];audits=[];control=None
    for i,basis in enumerate(bases()):
        if on_face:on_face(i)
        camera=pose.copy();camera[:3,:3]=pose[:3,:3]@basis;pair=[]
        for epoch,renderer in enumerate(renderers):
            rgb,depth,alpha=renderer.face(camera,size);people=person_mask(detector,rgb)
            path=destination/f'F{i+1:02d}-{["before","after"][epoch]}.png';cv2.imwrite(str(path),rgb[...,::-1])
            pair.append({'rgb':rgb,'depth':depth,'alpha':alpha,'person':people,'file':path,'rgb_sha256':sha(path)})
        probabilities,tiles,cp=predict_face(model,pair[0]['rgb'],pair[1]['rgb'],tile_size,overlap,self_control and i==0)
        if cp is not None:control=cp
        for epoch in range(2):pair[epoch]['probability']=probabilities[epoch];faces[epoch].append(pair[epoch])
        audits.append({'face':i,'camera_to_common_cv':camera.tolist(),'files':[p['file'].relative_to(directory).as_posix() for p in pair],
                       'rgb_sha256':[p['rgb_sha256'] for p in pair],'native_face_size':size,'tiles':tiles})
    pair=[stitch(face,width,height,size) for face in faces]
    audit={'mode':'perspective_tiles','full_panorama_model_calls':0,'faces':audits,'tiles_per_phase':sum(len(f['tiles']) for f in audits),
           'tile_size':tile_size,'tile_overlap':overlap,'probability_projection':'maximum-forward-cosine cube face; average overlapping perspective tiles'}
    write(directory/'directional-inference.json',audit)
    return pair,[p['probability'] for p in pair],audit,control
