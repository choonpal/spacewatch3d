"""Gaussian renderer: transform means, covariance frames and SH directions together."""
class Renderer:
    def __init__(self, model, assignments, matrices):
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        from tools.mesh_simsac.core import transform
        self.model=model;self.degree=model['degree']
        means=np.empty_like(model['xyz']);quats=np.empty_like(model['quat']);scales=np.empty_like(model['scale'])
        rotations=np.empty((len(means),3,3),'f4')
        for i,m in enumerate(matrices):
            mask=assignments==i
            if not mask.any():continue
            m=np.asarray(m);s=np.cbrt(np.linalg.det(m[:3,:3]));r=m[:3,:3]/s
            means[mask]=transform(model['xyz'][mask],m);scales[mask]=model['scale'][mask]+np.log(s);rotations[mask]=r
            original=Rotation.from_quat(model['quat'][mask][:,[1,2,3,0]])
            q=(Rotation.from_matrix(r)*original).as_quat();quats[mask]=q[:,[3,0,1,2]]
        self.xyz=torch.from_numpy(means).cuda();self.quats=torch.from_numpy(quats).cuda()
        self.scales=torch.from_numpy(np.exp(scales)).cuda()
        self.opacity=torch.sigmoid(torch.from_numpy(model['opacity']).cuda())
        self.sh=torch.from_numpy(model['sh']).cuda();self.rotations=torch.from_numpy(rotations).cuda()

    def face(self, camera, size=384):
        import numpy as np
        import torch
        from gsplat import rasterization, spherical_harmonics
        focal=.5*size/np.tan(np.deg2rad(100)/2)
        position=torch.from_numpy(camera[:3,3].astype('f4')).cuda()
        directions=self.xyz-position;directions/=torch.linalg.vector_norm(directions,dim=-1,keepdim=True).clamp_min(1e-8)
        directions=torch.bmm(directions[:,None,:],self.rotations).squeeze(1)
        with torch.inference_mode():
            colors=(spherical_harmonics(self.degree,directions,self.sh)+.5).clamp_min(0)
            result,alpha,_=rasterization(self.xyz,self.quats,self.scales,self.opacity,colors,
                torch.from_numpy(np.linalg.inv(camera).astype('f4'))[None].cuda(),
                torch.tensor([[[focal,0,size/2],[0,focal,size/2],[0,0,1]]],dtype=torch.float32,device='cuda'),
                size,size,near_plane=.005,far_plane=10000.,packed=True,render_mode='RGB+ED',backgrounds=torch.zeros((1,3),device='cuda'))
        rgb=(result[0,...,:3].clamp(0,1).cpu().numpy()*255+.5).astype('u1')
        return rgb,result[0,...,3].cpu().numpy(),alpha[0,...,0].cpu().numpy()

def person_mask(detector,image):
    import numpy as np
    import cv2
    size=image.shape[0]
    prediction=detector.predict(image[...,::-1].copy(),classes=[0],conf=.25,imgsz=640,device=0,verbose=False)[0]
    mask=np.zeros(image.shape[:2],'u1')
    if prediction.masks is not None:
        for raw in prediction.masks.data.cpu().numpy():mask|=(cv2.resize(raw,(image.shape[1],size),interpolation=cv2.INTER_NEAREST)>.5).astype('u1')
    if prediction.boxes is not None:
        for box in prediction.boxes.xyxy.cpu().numpy():
            x0,y0,x1,y1=box;mx=(x1-x0)*.1;my=(y1-y0)*.1
            mask[max(0,int(y0-my)):min(size,int(y1+my)+1),max(0,int(x0-mx)):min(image.shape[1],int(x1+mx)+1)]=1
    return mask>0


def pano(renderer, pose, detector, width=1024, height=512, size=384):
    import numpy as np
    import cv2
    yy,xx=np.mgrid[:height,:width];lo=((xx+.5)/width-.5)*2*np.pi;la=((yy+.5)/height-.5)*np.pi
    rays=np.stack([np.sin(lo)*np.cos(la),np.sin(la),np.cos(lo)*np.cos(la)],-1)
    specs=[([0,0,1],[0,-1,0]),([1,0,0],[0,-1,0]),([0,0,-1],[0,-1,0]),([-1,0,0],[0,-1,0]),([0,-1,0],[0,0,-1]),([0,1,0],[0,0,1])]
    bases=[]
    for f,up in specs:
        down=-np.asarray(up);forward=np.asarray(f);right=np.cross(down,forward);bases.append(np.column_stack([right,down,forward]))
    dominant=np.argmax(np.stack([(rays@b)[...,2] for b in bases],-1),axis=-1)
    rgb=np.zeros((height,width,3),'u1');depth=np.zeros((height,width),'f4');alpha=depth.copy();person=np.zeros_like(depth,dtype=bool)
    focal=.5*size/np.tan(np.deg2rad(100)/2)
    for i,basis in enumerate(bases):
        camera=pose.copy();camera[:3,:3]=pose[:3,:3]@basis
        image,dep,a=renderer.face(camera,size)
        mask=person_mask(detector,image).astype('u1')
        directions=rays@basis;z=np.maximum(directions[...,2],1e-8)
        mx=(focal*directions[...,0]/z+size/2-.5).astype('f4');my=(focal*directions[...,1]/z+size/2-.5).astype('f4')
        choose=dominant==i
        sample=lambda image,interpolation=cv2.INTER_LINEAR:cv2.remap(image,mx,my,interpolation,borderMode=cv2.BORDER_REPLICATE)
        rgb[choose]=sample(image)[choose];depth[choose]=(sample(dep)/z)[choose];alpha[choose]=sample(a)[choose]
        person[choose]=sample(mask,cv2.INTER_NEAREST)[choose]>0
    return {'rgb':rgb,'depth':depth,'alpha':alpha,'person':person,'rays':rays}
