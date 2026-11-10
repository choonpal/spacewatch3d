"""Bind runnable Dorm4/5 cold requests to actual accepted files, not guesses."""
from pathlib import Path
import copy
import hashlib
import json

root=Path('/workspace/spacewatch_integrated_20261110_8QJn4J')
code=root/'code'
target=code/'upstream519_original/requests'
target.mkdir(exist_ok=True)
raw=Path('/workspace/dorm4_dorm5_v5_20261025')
pipeline=code/'configs/dorm45-paper.json'
main=json.loads(pipeline.read_text())
original=json.loads((raw/'code/v2/configs/dorm2_dorm3.json').read_text())


def record(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())


for cid in ['dorm4','dorm5']:
 cap=next(c for c in main['captures'] if c['id']==cid)
 vocabulary=Path(cap['captured_prediction']['vocabulary']['path'])
 profile=Path(cap['planning_profile']['path'])
 cfg=copy.deepcopy(original)
 cfg['runtime']=copy.deepcopy(main['runtime']);cfg['vocabulary']=str(vocabulary)
 cfg['scene_id']=main['scene_id'];cfg['mesh'].pop('protected_voxel_ratio',None)
 cfg['captures']=[dict(id=cid,reconstruction_config=str(raw/'configs'/f'{cid}_reconstruction.json'),
                       coordinates=str(raw/'coordinates'/f'{cid}.json'))]
 cfgpath=target/f'{cid}-original-producer-config.json'
 cfgpath.write_text(json.dumps(cfg,indent=2)+'\n')
 baseline=Path('/workspace/spacewatch_runs/dorm4_dorm5_pipeline_v5_20261025')
 baseline_results=json.loads((baseline/'results.json').read_text())
 plan=Path(baseline_results[cid+'.plan']['plan'])
 request=dict(mode='fresh',input_mode='original_raw_video',capture_id=cid,
              producer_config=record(cfgpath),pipeline_config_template=record(pipeline),
              video=record(cap['video']),vocabulary=record(vocabulary),profile_context_plan=record(plan),
              coordinate_mode='original_assumed_2p7m',minimum_available_memory_gib=16,
              stage_timeout_seconds=main['runtime']['stage_timeout_seconds'],
              gpu_wait_seconds=main['runtime']['gpu_wait_seconds'],
              manual_inputs={'manual':record(profile/'manual.json'),'prompt':record(profile/'prompt.txt')},
              scope='Original Dorm4/5 policy with current19-class vocabulary bound by class name; no cold execution claimed by this request.')
 (target/f'{cid}-fresh-request.json').write_text(json.dumps(request,indent=2)+'\n')
 binding=json.loads((profile/'input_binding.json').read_text())
 source=Path(binding['source_run'])
 reuse=dict(semantic_arrays=record(binding['source_arrays']),vocabulary=record(vocabulary),coordinates=record(cap['coordinates']),
            scene=record(profile/'scene.json'),profile_root=str(profile),profile_checksums=record(profile/'checksums.json'),
            profile_input_binding=record(profile/'input_binding.json'),source_run=str(source),captured_prediction=cap['captured_prediction'],
            reconstruction_files={key:record(source/rel) for key,rel in [('status','status.json'),('config','config.json'),('poses','result/camera_poses.json'),('manifest','input/manifest.json')]})
 reuse_request=dict(mode='verified_reuse',capture_id=cid,reuse=reuse,profile_context_plan=record(plan),
                    semantic_producer_provenance='Accepted inputs19_02 artifacts; exact artifact/profile/input binding verification, not claimed originalv2 stage replay or fresh model execution.')
 (target/f'{cid}-reuse-request.json').write_text(json.dumps(reuse_request,indent=2)+'\n')
 print(json.dumps(dict(capture_id=cid,request=str(target/f'{cid}-fresh-request.json'),
                       producer_config=str(cfgpath),video_bytes=request['video']['bytes'],vocabulary_sha256=request['vocabulary']['sha256'])))
