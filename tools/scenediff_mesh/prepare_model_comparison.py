"""Freeze common SceneDiff view pairing and corresponding complete ERP frames."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch,yaml
from PIL import Image

ROOT=Path('/workspace/pano_change_models_20261025')
OLD=Path('/workspace/scenediff_mesh_rgb_20261024')
sys.path.insert(0,'/workspace/scenediff_input_ab_20261024/deps')
sys.path.insert(0,'/workspace/scenediff_input_ab_20261024')
from compare_inputs import modules

def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
    utils,Geo,Mask,Scene=modules(OLD/'upstream')
    config=yaml.safe_load((OLD/'upstream/configs/scenediff_config.yml').read_text())
    pipe=Scene.__new__(Scene);pipe.config=config;pipe.device='cuda'
    pipe.geometry_model=Geo(config,'cuda')
    (ROOT/'inputs').mkdir(exist_ok=True)
    views=[];arrays={};weights={};erps={}
    for zone in ['R01','R07','R08']:
        data=np.load(OLD/'prepared'/zone/'inputs.npz')
        meta=json.loads((OLD/'prepared'/zone/'manifest.json').read_text())
        n=len(meta['views']);n1,n2=meta['side_counts'];H=W=meta['size']
        depth=torch.from_numpy(np.stack([data[f'depth_{i}'] for i in range(n)]))[None,...,None].cuda()
        points=np.stack([data[f'points_{i}'] for i in range(n)])
        poses=torch.from_numpy(np.stack([data[f'pose_{i}'] for i in range(n)]))[None].cuda()
        K=torch.from_numpy(np.stack([data[f'K_{i}'] for i in range(n)]))[None].cuda()
        depth,points,poses,voxel=pipe.geometry_model.normalize_scene_scale(points,depth,poses)
        xy=utils.get_img_coor(H,W).cuda();grid=torch.tensor([[2/W,0,-1],[0,2/H,-1]],device='cuda')
        similarity=pipe._compute_similarity_matrix(poses,depth,K,grid,xy,H,W,n1,n2)
        pair_weights=pipe._compute_similarity_weights(similarity).cpu().numpy()
        weights[zone]=pair_weights.tolist()
        for i,v in enumerate(meta['views']):
            key=f'{zone}-{i:02d}'
            arrays[key]=data[f'rgb_{i}']
            frame=v['map_frame'];epoch=v['epoch'];erpkey=f'D{epoch+2}-f{frame}'
            original=Path('/workspace/sam31_panorama_test_20261023/inputs')/f'dorm{epoch+2}'/'erp'/f'{frame-112:05d}.jpg'
            assert original.exists(),original
            if erpkey not in erps:
                image=np.array(Image.open(original).convert('RGB').resize((1024,512),Image.Resampling.LANCZOS))
                arrays[erpkey]=image
                erps[erpkey]=dict(key=erpkey,path=str(original),rgb_sha256=sha(image),source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),epoch=epoch,frame=frame)
                assert erps[erpkey]['source_sha256']==v['source_rgb_sha256']
            refs=[f'{zone}-{j:02d}' for j in range(n) if pair_weights[i,j]>0]
            assert all(meta['views'][int(j.rsplit('-',1)[1])]['epoch']!=epoch for j in refs)
            views.append(dict(key=key,zone=zone,index=i,epoch=epoch,frame=frame,yaw=v['yaw_degrees'],
                rgb_sha256=sha(arrays[key]),erp=erpkey,refs=refs,region_to_input_instance=v['region_to_input_instance']))
    np.savez_compressed(ROOT/'inputs/rgb.npz',**arrays)
    result=dict(views=views,erps=list(erps.values()),weights=weights,
        pairing='Exact upstream SceneDiff view-overlap weights from mesh-input branch; all nonzero counterparts used. No change scores used to choose frame pairs.',
        modes=dict(view='Same byte-identical 224x224 RGB views as SceneDiff',erp='Complete ERP frame resized to 1024x512, prediction projected to the common views; perspective-only networks retain their original input preprocessing'),
        temporal_ground_truth=False)
    (ROOT/'inputs/manifest.json').write_text(json.dumps(result,indent=2))
    print('views',len(views),'ERPframes',len(erps),'directed view pairs',sum(len(v['refs']) for v in views),flush=True)

if __name__=='__main__':main()
