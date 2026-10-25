"""Paired SceneDiff candidate discovery: RGB/Pi3/SAM vs external mesh/masks.

The same RGB tensors, DINO weights, SceneDiff costs, thresholding, object
discovery, merging and association are used on both sides. This does NOT score
provided object IDs instead of discovering changed objects.
"""
from __future__ import annotations
import argparse, ast, gc, hashlib, json, os, pickle, sys, time, types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from run import DinoFeatures, read, write, sha, scatter_mean, region_match_cost


def modules(upstream):
    sys.path.insert(0, str(upstream))
    import utils
    from modules.geometry_model import GeometryModel
    from modules.mask_model import MaskModel
    import matplotlib.pyplot as plt
    from pycocotools import mask as mask_utils
    # Load the complete upstream class without optional point-cloud viewers or
    # torch_scatter's binary dependency. All invoked downstream methods remain.
    tree = ast.parse((upstream/'modules/scenediff.py').read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SceneDiff')
    ns = dict(vars(utils), __file__=str(upstream/'modules/scenediff.py'),
              GeometryModel=GeometryModel, MaskModel=MaskModel, Path=Path,
              time=time, pickle=pickle, plt=plt, mask_utils=mask_utils,
              scatter_mean=scatter_mean, get_dino_matched_region_cost=region_match_cost)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[klass], type_ignores=[])),
                 str(upstream/'modules/scenediff.py'), 'exec'), ns)
    return utils, GeometryModel, MaskModel, ns['SceneDiff']


def stream_features(self, images, voxel_indices, H, W):
    """Same per-voxel mean as upstream, accumulated one image at a time."""
    count = int(voxel_indices.max()) + 1
    sums = torch.zeros(count, self.feature_dim, device='cpu')
    counts = torch.bincount(voxel_indices.cpu(), minlength=count).clamp_min(1)
    for i in range(len(images)):
        feat = self.semantic_model.extract_features(images[i].permute(1,2,0).cpu().numpy())
        ind = voxel_indices[i*H*W:(i+1)*H*W].cpu()
        sums.index_add_(0, ind, feat.permute(1,2,0).reshape(-1,self.feature_dim).cpu())
    return (sums / counts[:,None]).to(self.device)


def fast_pc(self, points, costs, visible, voxel_size, H, W):
    """Vectorized equivalent of the upstream per-visible-voxel assignment."""
    p=torch.as_tensor(points.reshape(-1,3),device=self.device)
    ids=self.utils.voxelization(p[None],voxel_size)[0]
    vis=visible.flatten()
    avg=scatter_mean(costs.flatten()[vis],ids[vis])
    full=torch.zeros(int(ids.max())+1,device=self.device)
    full[:len(avg)]=avg
    return dict(cost_maps=full[ids],voxel_indices=ids,voxel_locations=scatter_mean(p,ids))


def prepare_native(root, source, config, upstream):
    utils,GeometryModel,MaskModel,_=modules(upstream)
    sys.path.insert(0,str(root/'Pi3'))
    sys.path.insert(0,str(root/'sam'))
    from pi3.models.pi3 import Pi3
    from pi3.utils.geometry import depth_edge
    setup=read(root/'setup-status.json')
    geo=GeometryModel(config,'cuda')
    geo.model=Pi3.from_pretrained(setup['pi3_checkpoint']).to('cuda').eval()
    geo.depth_edge_fn=depth_edge
    audit={'pi3_checkpoint':setup['pi3_checkpoint'],'pi3_source':'0a505af9c261a7c8e44946fba5b1942cdc25914b',
           'sam_source':'100f1cac9d57f74b5f8abcacb6daa2bf10e5cae4','gpu':torch.cuda.get_device_name(0)}
    for zone in ['R01','R07','R08']:
        out=root/'prepared_native'/zone
        out.mkdir(parents=True,exist_ok=True)
        if (out/'geometry.npz').exists(): continue
        data=np.load(source/zone/'inputs.npz')
        n=len(read(source/zone/'manifest.json')['views'])
        images=torch.from_numpy(np.stack([data[f'rgb_{i}'] for i in range(n)])).permute(0,3,1,2)[None].float().cuda()/255
        H,W=images.shape[-2:]
        print('Pi3',zone,images.shape,flush=True)
        result=geo.estimate_depth_and_poses(images)
        K=geo.compute_intrinsics(result,range(n),H,W)
        valid=geo.get_valid_depth_masks(result)
        conf=torch.stack([geo.compute_per_image_confidence(images[0,i:i+1].clone()) for i in range(n)])
        np.savez_compressed(out/'geometry.npz',depth=result['depth_map'].float().cpu().numpy(),
            points=result['point_map'][0].float().cpu().numpy(),poses=result['poses'].float().cpu().numpy(),
            K=K.float().cpu().numpy(),valid=valid.cpu().numpy(),conf=conf.float().cpu().numpy())
        del result,images,K,valid,conf
        gc.collect();torch.cuda.empty_cache()
    geo.cleanup();del geo;gc.collect();torch.cuda.empty_cache()
    config['models']['sam']['checkpoint']=setup['sam_checkpoint']
    masks=MaskModel(config,'cuda')
    masks._initialize_sam()
    # Upstream masks_update assumes both SAM levels contain at least one mask.
    # Guard only genuinely empty output; retain its original NMS otherwise.
    import modules.mask_model as mask_module
    original_update=mask_module.masks_update
    def safe_update(*levels,**kwargs):
        return tuple(original_update(level,**kwargs)[0] if len(level) else [] for level in levels)
    mask_module.masks_update=safe_update
    for zone in ['R01','R07','R08']:
        out=root/'prepared_native'/zone
        if (out/'masks.npz').exists():continue
        data=np.load(source/zone/'inputs.npz')
        n=len(read(source/zone/'manifest.json')['views'])
        result={}
        for i in range(n):
            print('SAM',zone,i,flush=True)
            m=masks.predict_mask(data[f'rgb_{i}'],masks.mask_generator)
            result[f'mask_{i}']=utils.reorder_mask(utils.remove_small_masks(m))
        np.savez_compressed(out/'masks.npz',**result)
    masks.cleanup()
    audit['sam_checkpoint_sha256']=sha(setup['sam_checkpoint'])
    audit['pi3_weights']=[dict(file=str(p.relative_to(setup['pi3_checkpoint'])),bytes=p.stat().st_size,sha256=sha(p))
                            for p in Path(setup['pi3_checkpoint']).glob('*.safetensors')]
    write(root/'native-model-audit.json',audit)


def discover(root,source,zone,branch,semantic,config,upstream):
    utils,GeometryModel,MaskModel,SceneDiff=modules(upstream)
    pipe=SceneDiff.__new__(SceneDiff)
    pipe.config=config;pipe.device='cuda';pipe.feature_dim=1280
    pipe.geometry_model=GeometryModel(config,'cuda');pipe.mask_model=MaskModel(config,'cuda')
    pipe.semantic_model=semantic;pipe.utils=utils
    pipe.aggregate_features_to_voxels=types.MethodType(stream_features,pipe)
    pipe._process_point_cloud_costs=types.MethodType(fast_pc,pipe)
    raw_threshold=pipe._compute_changed_region_threshold
    def safe_threshold(self,cost,visible):
        values=cost[visible]
        if len(values)==0:return float('inf')
        q=torch.quantile(values,config['detection']['filter_percentage_before_threshold'])
        upper=values[values>q]
        if len(upper)<2 or float(upper.max()-upper.min())<1e-7:
            return float(values.max())+1e-6
        return raw_threshold(cost,visible)
    pipe._compute_changed_region_threshold=types.MethodType(safe_threshold,pipe)
    data=np.load(source/zone/'inputs.npz');meta=read(source/zone/'manifest.json')
    n=len(meta['views']);n1,n2=meta['side_counts'];H=W=meta['size']
    rgb=np.stack([data[f'rgb_{i}'] for i in range(n)])
    images=torch.from_numpy(rgb).permute(0,3,1,2)[None].float().cuda()/255
    if branch=='original':
        g=np.load(root/'prepared_native'/zone/'geometry.npz')
        masks_file=np.load(root/'prepared_native'/zone/'masks.npz')
        points=g['points'];depth=torch.from_numpy(g['depth']).cuda();poses=torch.from_numpy(g['poses']).cuda()
        K=torch.from_numpy(g['K']).cuda();valid=torch.from_numpy(g['valid']).cuda();confidence=torch.from_numpy(g['conf']).cuda()
        masks=[masks_file[f'mask_{i}'] for i in range(n)]
    else:
        points=np.stack([data[f'points_{i}'] for i in range(n)])
        depth=torch.from_numpy(np.stack([data[f'depth_{i}'] for i in range(n)]))[None,...,None].cuda()
        poses=torch.from_numpy(np.stack([data[f'pose_{i}'] for i in range(n)]))[None].cuda()
        K=torch.from_numpy(np.stack([data[f'K_{i}'] for i in range(n)]))[None].cuda()
        valid=torch.from_numpy(np.stack([data[f'valid_{i}'] for i in range(n)])).cuda()
        confidence=valid.float() # Mesh ray hits have no learned Pi3 confidence.
        masks=[data[f'mask_{i}'] for i in range(n)]
    # Deliberately retain the upstream normalization and returned voxel size on
    # BOTH branches, including its raw-unit voxel-size convention.
    depth,points,poses,voxel_size=pipe.geometry_model.normalize_scene_scale(points,depth,poses)
    xy=utils.get_img_coor(H,W).cuda()
    grid=torch.tensor([[2/W,0,-1],[0,2/H,-1]],device='cuda')
    out=root/'results'/zone/branch;out.mkdir(parents=True,exist_ok=True)
    print('downstream',zone,branch,'voxel',voxel_size,flush=True)
    similarity=pipe._compute_similarity_matrix(poses,depth,K,grid,xy,H,W,n1,n2)
    weights=pipe._compute_similarity_weights(similarity)
    costs,visible=pipe._compute_cost_maps(images,masks,poses,depth,K,[[c] for c in confidence],weights,valid,xy,grid,H,W,out,zone,root/'results')
    pc1,pc2,feat1,feat2=pipe._voxelize_and_extract_features(points,costs,visible,images,n1,n2,voxel_size,H,W)
    cluster1,cluster2=pipe._compute_cluster_costs(pc1,pc2,masks,visible,n1,n2,H,W)
    objects=pipe._detect_and_merge_objects(cluster1,cluster2,visible,masks,pc1,pc2,feat1,feat2,n1,n2,H,W,voxel_size)
    object_masks,unit_ids=pipe._associate_and_save_objects(objects,images,cluster1,cluster2,n1,n2,H,W,out)
    serial=json.loads(json.dumps(object_masks))
    write(out/'object_masks.json',serial)
    from pycocotools import mask as mask_utils
    labels=np.full((n,H,W),-1,dtype=np.int32)
    for objid,item in object_masks.items():
        if not isinstance(objid,int):continue
        for video,frames in item.items():
            offset=0 if video=='video_1' else n1
            for idx,frame in frames.items():
                labels[offset+idx][mask_utils.decode(frame['mask']).astype(bool)]=objid
    np.savez_compressed(out/'maps.npz',labels=labels,cost=costs.cpu().numpy(),visible=visible.cpu().numpy(),
                        cluster=torch.cat([cluster1,cluster2]).cpu().numpy(),masks=np.stack(masks),valid=valid.cpu().numpy())
    summary=dict(branch=branch,zone=zone,input_rgb_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in rgb],
        size=[H,W],frames=n,voxel_size=float(voxel_size),threshold=float(objects['threshold_1']),
        input_regions=[int((np.unique(x)>=0).sum()) for x in masks],
        per_side_candidates=[len(objects['merged_objects_1']),len(objects['merged_objects_2'])],
        unified_candidates=len([k for k in object_masks if isinstance(k,int)]),
        candidate_pixels=int((labels>=0).sum()),valid_fraction=float(valid.float().mean()),visible_fraction=float(visible.float().mean()))
    write(out/'summary.json',summary)
    print('RESULT',json.dumps(summary),flush=True)
    semantic.cache.clear();del pc1,pc2,feat1,feat2,objects,images
    gc.collect();torch.cuda.empty_cache()
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['native','compare','all'],default='all')
    args=parser.parse_args()
    root=Path('/workspace/scenediff_input_ab_20261024')
    old=Path('/workspace/scenediff_mesh_rgb_20261024')
    source=old/'prepared';upstream=old/'upstream'
    config=yaml.safe_load((upstream/'configs/scenediff_config.yml').read_text())
    config['visualization'].update(save_detections=False,save_gt_visualization=False,vis_pc=False)
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    start=time.time()
    write(root/'run-status.json',dict(stage=args.stage,state='running'))
    if args.stage in ['native','all']:prepare_native(root,source,config,upstream)
    if args.stage in ['compare','all']:
        semantic=DinoFeatures(read(old/'setup-status.json'),'cuda')
        semantic.compute_similarity=lambda a,b:torch.cosine_similarity(a,b,dim=0)
        results=[]
        for zone in ['R01','R07','R08']:
            for branch in ['original','mesh']:
                results.append(discover(root,source,zone,branch,semantic,config,upstream))
        write(root/'results.json',dict(results=results,model=semantic.audit,config=config,
            upstream_commit='62dcbd285fcfe984bb3db55dc01bdcd1a0055714',
            resolution_note='Controlled paired comparison at 224x224; upstream default RGB loader uses max_size=518.',
            elapsed_seconds=time.time()-start))
    write(root/'run-status.json',dict(stage=args.stage,state='complete',elapsed_seconds=time.time()-start))

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        Path('/workspace/scenediff_input_ab_20261024/error.txt').write_text(traceback.format_exc())
        raise
