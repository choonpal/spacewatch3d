"""OOOPS before geometric filtering; retain exact source rows and pixel provenance."""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from .io import ROOT,read,save,sha,digest,verify,result
from .geometry import context

def filtered_source(deps,prefix):
    return deps[prefix+'.filter'] if prefix+'.filter' in deps else deps[prefix+'.reconstruct']

def inputs(job):
    c,out,deps,prefix=context(job);external=[]
    if c.get('source_run'):
        root=Path(c['source_run']);manifest=read(root/'input/manifest.json')
        paths=[root/'input'/r['file'] for r in manifest['frames']]
        external.append(str(root/'input/manifest.json'))
    else:
        cfg=read(c['reconstruction_config']);bundle=read(cfg['input']['bundle'])
        if bundle['source']['sha256']!=cfg['input']['source_sha256']:raise ValueError('Source bundle mismatch')
        manifest=read(verify(bundle['frame_manifest']));paths=[verify(bundle['frames'][str(i)]) for i in range(len(manifest['frames']))]
        external.extend([c['reconstruction_config'],cfg['input']['bundle'],str(verify(bundle['frame_manifest']))])
    for path,row in zip(paths,manifest['frames']):
        if sha(path)!=row['sha256']:raise ValueError('Frame SHA mismatch: '+str(path))
        external.append(str(path))
    save(out/'manifest.json',manifest)
    result(out,manifest=str(out/'manifest.json'),images=[str(p) for p in paths],external_artifacts=external)

def dynamic_pixels(labels,support,margin,catalog,options):
    ids=[r['id'] for r in catalog if r['name'] in options.get('dynamic_classes',['person'])]
    return np.isin(labels,ids)&(support>=options.get('dynamic_support',.45))&(margin>=options.get('dynamic_margin',.15))

def masks(job):
    from scipy.ndimage import maximum_filter
    c,out,deps,prefix=context(job);catalog=read(job['config']['vocabulary']);opt=job['config'].get('early_semantics',{})
    data=deps[prefix+'.inputs'];manifest=read(data['manifest']);folder=out/'masks';folder.mkdir()
    stats=[]
    for i,(image,row) in enumerate(zip(data['images'],manifest['frames'])):
        with np.load(Path(deps[prefix+'.infer']['predictions'])/f'{i:05d}.npz') as pred:
            pixels=dynamic_pixels(pred['labels'],pred['support_fraction'],pred['margin_fraction'],catalog,opt)
        with Image.open(image) as source_image:width,height=source_image.size
        pixels=np.asarray(Image.fromarray(pixels).resize((width,height),Image.Resampling.NEAREST))
        dilation=opt.get('dynamic_dilation_pixels',2)
        if dilation:pixels=maximum_filter(pixels,size=2*dilation+1,mode=('nearest','wrap'))
        target=folder/(Path(row['file']).stem+'.png');Image.fromarray(pixels.astype('u1')*255).save(target)
        stats.append(dict(frame=i,excluded_pixels=int(pixels.sum()),fraction=float(pixels.mean()),sha256=sha(target)))
    save(out/'mask_report.json',dict(policy='OOOPS dynamic-class pixels, periodic horizontal dilation',classes=opt.get('dynamic_classes',['person']),frames=stats))
    result(out,masks=str(folder),report=str(out/'mask_report.json'),frames=len(stats),excluded_pixels=sum(s['excluded_pixels'] for s in stats))

def decide_points(counts,dynamic,base_keep,catalog,options):
    """At most one class vote per frame/voxel; unknown evidence never rescues noise."""
    total=counts.sum(1,dtype=np.uint32);winner=counts.argmax(1);maximum=counts[np.arange(len(counts)),winner]
    names=options.get('protect_classes',[r['name'] for r in catalog if r.get('static_mesh',True) and r['name'] not in ['wall','floor','ceiling','tactile paving']])
    ids=[r['id'] for r in catalog if r['name'] in names and r.get('static_mesh',True)]
    protected=np.isin(winner,ids)&(maximum>=options.get('minimum_static_frames',2))&(maximum>=total*options.get('static_agreement',.7))&~dynamic
    keep=(base_keep|protected)&~dynamic
    return keep,protected,winner

def filter(job):
    from plyfile import PlyData
    sys.path.insert(0,str(ROOT/'vendor/segmentation'))
    from geometry import source_frames,build_index
    from core import voxel_keys,buckets,exact_lookup,resize_labels_to_source,frame_votes,ply_header,RGB_DTYPE
    from filtered_source import build_row_map
    from common import complete_stage
    c,out,deps,prefix=context(job);src=deps[prefix+'.reconstruct'];root=Path(src['source_run'])
    catalog=read(job['config']['vocabulary']);opts=job['config'].get('early_semantics',{})
    projection=dict(source_run=str(root),output=str(out));external=[]
    if c.get('source_index'):
        indexdir=Path(c['source_index'])
        complete=read(indexdir/'complete.json')
        for path,record in complete['files'].items():
            if sha(path)!=record['sha256']:raise ValueError('Source index changed: '+path)
            external.append(path)
        index=read(indexdir/'index.json')
        original=PlyData.read(root/'unfiltered_result/map.ply',mmap='r')['vertex'].data
        target=PlyData.read(root/'result/map.ply',mmap='r')['vertex'].data
        build_row_map(original,target,np.load(root/'component_filter/keep_mask.npy',mmap_mode='r'),out/'base_rowmap.npy')
        if not np.array_equal(np.load(out/'base_rowmap.npy',mmap_mode='r'),np.load(index['unfiltered_to_target'],mmap_mode='r')):raise ValueError('Source index reconstruction mismatch')
    else:
        build_index(projection,dict(identity=digest(src),source_frames=src['frames']))
        indexdir=out/'source_index';index=read(indexdir/'index.json')
    cloud=PlyData.read(root/'unfiltered_result/map.ply',mmap='r')['vertex'].data;n=len(cloud)
    if cloud.dtype!=RGB_DTYPE:raise ValueError('Unexpected raw XYZ/RGB layout')
    manifest=read(root/'input/manifest.json')
    if src['frames']>65535:raise ValueError('Maximum 65535 frames for uint16 observation counts')
    keys=[np.load(r['path'],mmap_mode='r') for r in index['buckets']]
    counts=np.lib.format.open_memmap(out/'class_frame_counts.npy',mode='w+',dtype='u2',shape=(n,len(catalog)))
    dynamic=np.lib.format.open_memmap(out/'dynamic.npy',mode='w+',dtype='bool',shape=(n,))
    for frame,xyz,mask,_ in source_frames(projection):
        cells=voxel_keys(xyz,index['voxel']);bucket=buckets(cells,len(keys));ids=np.empty(len(xyz),np.uint32)
        for k,base in enumerate(keys):
            selected=bucket==k
            if selected.any():ids[selected]=exact_lookup(base,cells[selected])+index['buckets'][k]['offset']
        with np.load(Path(deps[prefix+'.infer']['predictions'])/f'{frame:05d}.npz') as pred:
            labels,support=resize_labels_to_source(pred['labels'],pred['support_fraction'],*mask.shape)
            _,margin=resize_labels_to_source(pred['labels'],pred['margin_fraction'],*mask.shape)
        name=Path(manifest['frames'][frame]['file']).stem+'.png'
        dynamic_mask=np.asarray(Image.open(Path(deps[prefix+'.masks']['masks'])/name))>0
        dynamic[ids[dynamic_mask[mask]]]=True
        support[(margin<opts.get('static_margin',.1))|(labels>=len(catalog))]=0
        vote=frame_votes(ids,labels[mask],support[mask],opts.get('static_support',.35))
        good=(vote['weight']>=opts.get('static_support',.35))&(vote['label']<len(catalog))
        counts[vote['index'][good],vote['label'][good]]+=1
        if frame%10==0:print('EARLY_SEMANTIC_VOTES',prefix,frame,src['frames'],flush=True)
    counts.flush();dynamic.flush()
    overlay=out/'source';overlay.mkdir();(overlay/'result').mkdir();(overlay/'component_filter').mkdir();(overlay/'masks').mkdir()
    for name in ['input','source','windows','prepared.npz','optimized.npz','unfiltered_result']:
        p=root/name
        if p.exists():(overlay/name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
    settings=read(root/'config.json');settings['integration_postprocess']='SpaceWatch integrated pipeline v2 early semantic filtering';save(overlay/'config.json',settings)
    (overlay/'result/camera_poses.json').symlink_to((root/'result/camera_poses.json').resolve())
    manifest=read(root/'input/manifest.json')
    for row in manifest['frames']:
        name=Path(row['file']).stem+'.png';base=np.asarray(Image.open(root/'masks'/name))>0;extra=np.asarray(Image.open(Path(deps[prefix+'.masks']['masks'])/name))>0
        if base.shape!=extra.shape:raise ValueError('Dynamic mask coordinate mismatch')
        Image.fromarray((base|extra).astype('u1')*255).save(overlay/'masks'/name)
    base_keep=np.load(root/'component_filter/keep_mask.npy',mmap_mode='r')
    keep=np.lib.format.open_memmap(overlay/'component_filter/keep_mask.npy',mode='w+',dtype='bool',shape=(n,))
    protect=np.lib.format.open_memmap(out/'protected_raw.npy',mode='w+',dtype='bool',shape=(n,))
    stats=dict(input_points=n,base_kept_points=int(base_keep.sum()),kept_points=0,rescued_points=0,dynamic_removed_from_base=0,dynamic_points=int(dynamic.sum()),protected_points=0)
    perclass=np.zeros(len(catalog),np.int64);chunk=1000000
    for a in range(0,n,chunk):
        b=min(n,a+chunk);k,p,l=decide_points(counts[a:b],dynamic[a:b],base_keep[a:b],catalog,opts);keep[a:b]=k;protect[a:b]=p
        stats['kept_points']+=int(k.sum());stats['rescued_points']+=int((k&~base_keep[a:b]).sum());stats['dynamic_removed_from_base']+=int((base_keep[a:b]&dynamic[a:b]).sum());stats['protected_points']+=int(p.sum())
        perclass+=np.bincount(l[k&~base_keep[a:b]],minlength=len(catalog))
    keep.flush();protect.flush()
    if not stats['kept_points']:raise ValueError('Semantic filtering removed every point')
    ply=overlay/'result/map.ply';protected=np.lib.format.open_memmap(out/'protected_points.npy',mode='w+',dtype='bool',shape=(stats['kept_points'],));offset=0
    with ply.open('wb') as stream:
        stream.write(ply_header(stats['kept_points'],RGB_DTYPE))
        for a in range(0,n,chunk):
            k=keep[a:a+chunk];rows=cloud[a:a+chunk][k];rows.tofile(stream);protected[offset:offset+len(rows)]=protect[a:a+chunk][k];offset+=len(rows)
    protected.flush()
    newindex=out/'filtered_index';newindex.mkdir()
    proof=build_row_map(cloud,PlyData.read(ply,mmap='r')['vertex'].data,keep,newindex/'unfiltered_to_target.npy')
    filtered=dict(index,unfiltered_to_target=str(newindex/'unfiltered_to_target.npy'),target_points=offset,**{k:v for k,v in proof.items() if k!='target_points'})
    save(newindex/'index.json',filtered)
    complete_stage(newindex/'complete.json',digest([src,opts]),[newindex/'index.json',newindex/'unfiltered_to_target.npy']+[Path(r['path']) for r in index['buckets']],**proof)
    report=dict(source=str((root/'unfiltered_result/map.ply').resolve()),output=str(ply),source_sha256=sha(root/'unfiltered_result/map.ply'),sha256=sha(ply),**stats,
        rescued_by_class={r['name']:int(perclass[r['id']]) for r in catalog},options=opts,validation=proof,
        base_source_run=str(root),ordering='OOOPS -> dynamic mask -> base reconstruction -> semantic component policy -> mesh',
        limitation='Semantic evidence is not ground truth; rescue cannot recover pixels already excluded by base depth/shooter masks')
    save(overlay/'component_filter/report.json',report);save(overlay/'status.json',dict(status='complete'));save(out/'filter_report.json',report)
    # Large frame counts are intermediate; keep compact masks and verifiable mapping.
    del counts
    (out/'class_frame_counts.npy').unlink()
    result(out,**{k:v for k,v in src.items() if k not in ['source_run','point_cloud','external_artifacts']},source_run=str(overlay),base_source_run=str(root),point_cloud=str(ply),
        protected_points=str(out/'protected_points.npy'),source_index=str(newindex),filter_report=str(out/'filter_report.json'),report=report,external_artifacts=external)
