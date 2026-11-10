"""Receipted bridge to pinned upstream mesh/original-RGB SimSaC stages.

No Gaussian dependency, old change predictions, cross-capture ID equality or
image synthesis. CPU preparation/classification and GPU inference use separate
configured runtimes. Each stage writes a fresh directory, never its predecessor.
"""
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from .io import read,save,result,sha

UPSTREAM_COMMIT='4fbecc358369644720d4afd4d77f3915ccb8572f'


def validate_settings(config):
    options=config.get('changes',{})
    allowed={'backend','enabled','frame_stride','point_limit','native_rgb','settings','keypoint_nearby'}
    if set(options)-allowed or options.get('backend')!='mesh_rgb_simsac':
        raise ValueError('Unknown mesh_rgb_simsac setting')
    provided=(config.get('segmentation',{}).get('source')=='provided' and
              all(c.get('captured_prediction') for c in config.get('captures',[])))
    if len(config.get('captures',[]))!=2 or (config.get('segmentation',{}).get('source')!='ooops' and not provided):
        raise ValueError('mesh_rgb_simsac needs two captures with their own whole-ERP OOOPS predictions')
    if config.get('gaussian_changes',{}).get('enabled'):
        raise ValueError('Do not run Gaussian and original RGB change pipelines together')
    for key,default in [('frame_stride',6),('point_limit',2000000)]:
        value=options.get(key,default)
        if type(value) is not int or value<1:raise ValueError('Invalid changes.'+key)
    if type(options.get('native_rgb',True)) is not bool:raise ValueError('Invalid changes.native_rgb')
    if options.get('keypoint_nearby') is True:
        from .paper_changes import validate_options
        validate_options(config)
    elif options.get('keypoint_nearby'):
        from .keypoint_rgb import validate_options
        validate_options(config)
    settings=options.get('settings',{})
    if settings.get('registration','auto')!='auto' or not settings.get('regional_correction',True):
        raise ValueError('New capture pair requires fresh global/regional registration')
    from tools.mesh_simsac.observation_quality import quality_settings
    quality_settings(settings)
    runtime=config['runtime']
    for key in ('change_python','change_root','change_torch_home','mesh_rgb_checkpoint','mesh_rgb_three_package'):
        if not runtime.get(key):raise ValueError('Missing runtime.'+key)


def clone(source,destination):
    """Read-only large arrays can share storage; no existing output is edited."""
    def link(a,b):
        if Path(a).suffix.lower() not in ('.npz','.npy','.png','.jpg','.jpeg','.bin'):
            return shutil.copy2(a,b)
        try:return os.link(a,b)
        except OSError:return shutil.copy2(a,b)
    shutil.copytree(source,destination,copy_function=link)


def native_frames(manifest,folder,sources):
    import cv2
    from .video_frames import FrameReader
    for phase,source in zip(('before','after'),sources):
        cap=cv2.VideoCapture(source['video'])
        if not cap.isOpened():raise ValueError('Cannot decode original RGB capture')
        reader=FrameReader(cap)
        try:
            for frame in manifest[phase]['frames']:
                bgr=reader.read(frame['source_frame_index'])
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                if rgb.shape[1]!=2*rgb.shape[0]:raise ValueError('Original frame is not ERP')
                image=folder/phase/'rgb'/(f"native_{frame['source_frame_index']:08d}.png")
                Image.fromarray(rgb).save(image,compress_level=1)
                frame['source_semantic_rgb_sha256']=frame['sha256']
                frame['image']=image.relative_to(folder).as_posix();frame['sha256']=sha(image)
                frame['segmentation_rgb_sha256']=frame['sha256']
                frame['source_rgb_pixels_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
                frame['semantic_resampling']='Same source frame; normalized whole-ERP pixel centers, no new inference'
                mask=folder/frame['person_mask']
                with Image.open(mask) as im:
                    resized=im.resize((rgb.shape[1],rgb.shape[0]),Image.Resampling.NEAREST)
                    destination=mask.with_name(mask.stem+'-native.png');resized.save(destination)
                frame['person_mask']=destination.relative_to(folder).as_posix()
        finally:cap.release()


def materialize_native_frames(folder,adapter,runtime):
    """Open3D and video decoding live in different installed environments."""
    python=runtime.get('video_python',runtime['reconstruction_python'])
    subprocess.run([python,'-m','spacewatch_pipeline.mesh_rgb','--native-frames',
        str(folder/'manifest.json'),'--adapter',str(adapter)],check=True,
        timeout=min(1800,runtime.get('stage_timeout_seconds',1800)))
    return read(folder/'manifest.json')


def rgb_inputs(job):
    validate_settings(job['config'])
    from tools.mesh_simsac.import_semantic_scene import export
    from tools.mesh_simsac.bundle import inspect_bundle
    out=Path(job['output']);cfg=job['config'];deps=job['dependencies'];options=cfg['changes']
    captures=[]
    for c in cfg['captures']:
        prefix=c['id'];src=deps[prefix+'.reconstruct'];prediction=deps[prefix+'.ooops']
        inference=read(prediction['inference']);run=Path(src['source_run'])
        manifest=read(run/'input/manifest.json')
        if (inference['source_manifest_sha256']!=sha(run/'input/manifest.json') or
                inference['vocabulary_sha256']!=sha(prediction['vocabulary']) or
                len(inference['frames'])!=len(manifest['frames'])):
            raise ValueError('RGB change input segmentation lineage mismatch')
        for i,row in enumerate(manifest['frames']):
            item=inference['frames'][i];p=Path(prediction['predictions'])/f'{i:05d}.npz'
            if item['source_sha256']!=row['sha256'] or item['prediction_sha256']!=sha(p):
                raise ValueError('RGB change input prediction/hash changed')
        captures.append(dict(id=prefix,mesh=deps[prefix+'.semantic']['arrays'],
            scene=deps[prefix+'.scene']['scene'],source_run=src['source_run'],coordinates=src['coordinates'],
            video=c['video'],classes=prediction['vocabulary'],predictions=prediction['predictions'],
            prediction_manifest=str(run/'input/manifest.json'),segmentation_model='OOOPS whole ERP; semantic surface candidates, not physical instances'))
    settings=dict(frame_stride=1,size=224,max_refs=2,min_frames=2,registration='auto',regional_correction=True)
    settings.update(options.get('settings',{}))
    adapter=dict(captures=captures,frame_stride=options.get('frame_stride',6),point_limit=options.get('point_limit',2000000),
        comparison_settings=settings,scope='Same-space before/after original RGB; automatically inferred surface change candidates; UNVERIFIED')
    save(out/'adapter.json',adapter)
    folder=out/'inputs';export(out/'adapter.json',folder,archive=False)
    manifest=read(folder/'manifest.json');manifest['scene_id']=cfg['scene_id']
    save(folder/'manifest.json',manifest)
    if options.get('native_rgb',True):
        manifest=materialize_native_frames(folder,out/'adapter.json',cfg['runtime'])
    _,files=inspect_bundle(folder)
    save(out/'input_lineage.json',dict(upstream_commit=UPSTREAM_COMMIT,files=files,
        gaussian_inputs_used=False,old_change_predictions_used=False,source_rgb_modified=False,
        native_original_rgb=options.get('native_rgb',True),physical_instance_ids_verified=False,
        reconstructed_all_person_masks_used_for_rgb=False,
        rgb_mask_policy='Unpadded same-frame nadir-connected photographer only; upstream semantics retains bystanders'))
    result(out,manifest=str(folder/'manifest.json'),lineage=str(out/'input_lineage.json'),
           captures=[c['id'] for c in cfg['captures']],source_mode='original_rgb',upstream_commit=UPSTREAM_COMMIT)


def rgb_prepare(job):
    from tools.mesh_simsac.prepare import prepare
    from tools.mesh_simsac.bundle import inspect_bundle
    out=Path(job['output']);manifest=Path(job['dependencies']['pair.rgb_inputs']['manifest'])
    inspect_bundle(manifest.parent)
    root=out/'detection';prepare(manifest,root,capturer_checkpoint=None)
    prepared=read(root/'manifest.json')
    if prepared['gaussian_inputs_used'] or prepared['existing_change_results_used']:
        raise ValueError('Unexpected Gaussian/old-result dependency')
    result(out,root=str(root),registration=str(root/'registration.json'),manifest=str(manifest),
           views=len(prepared['views']),status=read(root/'registration.json')['state'])


def rgb_infer(job):
    out=Path(job['output']);runtime=job['config']['runtime'];root=out/'detection'
    timeout=runtime.get('stage_timeout_seconds',86400)
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not np.isfinite(timeout) or timeout<=0:
        raise ValueError('runtime.stage_timeout_seconds must be positive and finite')
    deadline=time.monotonic()+timeout
    clone(job['dependencies']['pair.rgb_prepare']['root'],root)
    source=Path(runtime['change_root'])/'simsac';checkpoint=Path(runtime['mesh_rgb_checkpoint'])
    # The NN/ERP caches and appearance backbone must not coexist in this worker.
    # Both unchanged native CLIs inherit the SAME leased GPU/CUDA environment.
    # Do not create a session/process group: engine.run_worker owns this worker's
    # group and cancellation kills the worker plus both native descendants.
    environment=dict(os.environ)
    python=runtime.get('change_python',sys.executable)
    for module in ('tools.mesh_simsac.run','tools.mesh_simsac.appearance'):
        command=[python,'-m',module,'--root',str(root),'--simsac-source',str(source),
                 '--checkpoint',str(checkpoint),'--device','cuda']
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise subprocess.TimeoutExpired(command,timeout)
        subprocess.run(command,check=True,env=environment,cwd=Path(__file__).resolve().parents[1],
                       timeout=remaining,start_new_session=False)
    state=read(root/'inference-status.json')
    if state['state']!='complete':raise ValueError('SimSaC inference did not finish')
    result(out,root=str(root),pairs=state['pairs'],model=str(root/'results/model.json'),
           source_rgb=True,gaussian_used=False,fresh_predictions=True)


def rgb_finalize(job):
    from tools.mesh_simsac.classify import classify
    from tools.mesh_simsac.validate import validate
    from tools.mesh_simsac.preview import export
    from tools.mesh_simsac.build_event_viewer import build
    out=Path(job['output']);root=out/'detection';deps=job['dependencies']
    clone(deps['pair.rgb_infer']['root'],root)
    manifest=deps['pair.rgb_inputs']['manifest']
    classify(root,manifest);validate(root)
    export(root,manifest,job['config']['runtime']['mesh_rgb_three_package']);build(root,manifest)
    data=read(root/'viewer-data.json');registration=read(root/'registration.json')
    data['job_provenance']=dict(fresh_inference=True,input_source='pipeline_original_rgb',
        gaussian_inputs_used=False,upstream_commit=UPSTREAM_COMMIT,semantic_accuracy='UNVERIFIED')
    save(root/'viewer-data.json',data)
    result(out,changes=str(root/'results/events.json'),viewer_dir=str(root),viewer_manifest=str(root/'viewer-data.json'),
        backend='mesh_rgb_simsac',candidates=len(data['events']),deferred=data['counts'].get('unresolved',0),
        counts=data['counts'],alignment_status=registration['state'],comparison_performed=True,
        gaussian_used=False,quality_status='UNVERIFIED',upstream_commit=UPSTREAM_COMMIT,
        validation=str(root/'validation.json'),rgb_inference_pairs=deps['pair.rgb_infer']['pairs'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-frames',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    args=parser.parse_args()
    manifest=read(args.native_frames)
    native_frames(manifest,args.native_frames.parent,read(args.adapter)['captures'])
    save(args.native_frames,manifest)
